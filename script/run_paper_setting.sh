#!/usr/bin/env bash
#
# Download Grasp-Anything++ (cached) + train + eval following the LGD paper's protocol.
#
# OUR CONFIGURATION: **one GPU, 100,000 scenes (~445k samples, 10% of GA++), 50 epochs.**
# Those three numbers are resource choices, not the paper's -- stated here and in each run's
# config.txt so the report stays unambiguous.
#
# What the paper (Vuong et al., CVPR 2024) states:
#   * train/eval on the full GA++ (4.41M samples) -- we run on 10%
#   * Base/New split = 70/30 of categories by frequency (Sec. 5.1) -- kept
#   * success when IoU >= 0.25 AND angular error <= 30 degrees, reporting
#     Seen / Unseen / harmonic mean H -- kept
#   * 100 epochs (Fig. 6) -- we run 50
# What the paper does NOT state in the main text: batch size, optimizer, learning rate -- those
# are in a Supplementary we do not have. The values below are our own choices.
#
# A note on "epoch": this repo defines an epoch as BATCHES_PER_EPOCH batches, not one pass over
# the data. 50 x 2000 x 64 = 6.4M sample draws = ~14 passes over the subset's 445k samples --
# i.e. the 100k-scene subset is seen *several times*, quite unlike 100 epochs on the full GA++
# (2.9 passes). The number to report is the 6.4M-sample-draw budget, and it must be identical
# across every ablation arm.
#
# One GPU: train_network.py has no DataParallel/DDP, so each run uses exactly one GPU. Pick it
# with the GPU variable (the script exports CUDA_VISIBLE_DEVICES itself). The ablation table
# runs sequentially on the same GPU: script/run_ablation.sh.
#
# Every parameter can be overridden through the environment:
#   SCENES=200000 NUM_WORKERS=24 bash script/run_paper_setting.sh
#   GPU=1 bash script/run_paper_setting.sh
#
# Every step is cached: re-running skips whatever is already done.

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python3}

# One GPU for the whole run. train_network.py/evaluate.py take `cuda` (i.e. cuda:0) from
# hardware/device.py, so the physical GPU is selected with CUDA_VISIBLE_DEVICES, not a flag.
if [ -n "${GPU:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
fi

DATA_DIR=${DATA_DIR:-data/grasp-anything-pp-full}
SPLIT_DIR=${SPLIT_DIR:-split/grasp-anything-pp}
ZIPS_DIR=${ZIPS_DIR:-$DATA_DIR/_archives}

# GA++'s total scene count, used only to print "x / total". Do not write 994_860: bash printf
# reports "invalid number" and awk reads "100_000" as 100 -- the disk estimate comes out 0.
TOTAL_SCENES=994860
SCENES=${SCENES:-100000}              # default: a 10% subset (~445k samples)
PACK_MASKS=${PACK_MASKS:-1}           # 21 KB/mask instead of 173 KB, lossless
KEEP_ZIPS=${KEEP_ZIPS:-1}             # keep the zips so changing SCENES needs no re-download

NPROC=$(nproc)
NUM_WORKERS=${NUM_WORKERS:-$(( NPROC / 2 > 32 ? 32 : (NPROC / 2 < 2 ? 2 : NPROC / 2) ))}

NETWORK=${NETWORK:-stag}
EPOCHS=${EPOCHS:-50}                  # 50 x 2000 x 64 = 6.4M sample draws
BATCHES_PER_EPOCH=${BATCHES_PER_EPOCH:-2000}
BATCH_SIZE=${BATCH_SIZE:-64}
INPUT_SIZE=${INPUT_SIZE:-224}
VAL_SPLIT=${VAL_SPLIT:-0.998}         # fraction of the dev set used for training; rest is val
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-64}  # validation is batched (upstream hard-codes 1)
AMP=${AMP:-auto}                      # auto = bf16 if the GPU supports it. off = pure fp32
CHANNELS_LAST=${CHANNELS_LAST:-auto}
TOKENIZE_IN_LOADER=${TOKENIZE_IN_LOADER:-1}
LR=${LR:-1e-3}
LR_SCHEDULE=${LR_SCHEDULE:-cosine}
USE_TEXT=${USE_TEXT:-1}
W_ALIGN=${W_ALIGN:-0.3}
ALIGN_MODE=${ALIGN_MODE:-soft}        # soft = softmax over tokens; hard = argmax
REGION_TEXT=${REGION_TEXT:-1}         # feed R_T to the decoder
FUSION=${FUSION:-residual}            # residual = F + phi(...); gate = F * (1 + lambda*A_T)
ALIGN_STAGE=${ALIGN_STAGE:-bottleneck}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-3}
DIAG_INTERVAL=${DIAG_INTERVAL:-500}   # steps between fusion/token/gradient logs
PROBE_SAMPLES=${PROBE_SAMPLES:-8}     # fixed probe set redrawn across epochs
COUNTERFACTUAL_EVERY=${COUNTERFACTUAL_EVERY:-5}
DESCRIPTION=${DESCRIPTION:-ga-pp-paper}
LOGDIR=${LOGDIR:-logs/}

RESULTS_DIR=${RESULTS_DIR:-results}
N_FIGURES=${N_FIGURES:-4}             # samples per split to draw alignment figures for
COPY_CHECKPOINT=${COPY_CHECKPOINT:-0} # also copy the checkpoint (~9 MB) into results/
AUDIT_SAMPLES=${AUDIT_SAMPLES:-500}   # samples for the wrong-prompt counterfactual

SKIP_BUILD=${SKIP_BUILD:-0}
SKIP_SPLIT=${SKIP_SPLIT:-0}
SKIP_CHECK=${SKIP_CHECK:-0}
SKIP_TRAIN=${SKIP_TRAIN:-0}
SKIP_EVAL=${SKIP_EVAL:-0}
SKIP_AUDIT=${SKIP_AUDIT:-0}
SKIP_EXPORT=${SKIP_EXPORT:-0}

RUN_STAMP=$(date +%y%m%d_%H%M)
RUN_RESULTS="$RESULTS_DIR/${RUN_STAMP}_${DESCRIPTION}"

# ------------------------------------------------------------------- plan ---
SAMPLES=$(awk "BEGIN{printf \"%d\", $SCENES * 4.436}")
if [ "$PACK_MASKS" = "1" ]; then MASK_KB=24; else MASK_KB=176; fi
# Each .pt and .pkl file takes at least one 4 KB block; added in for a realistic estimate.
NEED_GB=$(awk "BEGIN{printf \"%d\", ($SAMPLES*($MASK_KB+8) + $SCENES*69)/1048576 + 1}")
AVAIL_GB=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')

cat <<EOF
== plan ==
  scenes           : $(printf "%'d" "$SCENES") / $(printf "%'d" $TOTAL_SCENES)
  samples (est.)   : $(printf "%'d" "$SAMPLES")
  pack-masks       : $PACK_MASKS
  disk needed      : ~${NEED_GB} GB (excluding ~76 GB of archives if KEEP_ZIPS=1)
  disk free        : ${AVAIL_GB} GB
  workers          : $NUM_WORKERS (machine has $NPROC cores)
  training budget  : $EPOCHS x $BATCHES_PER_EPOCH x $BATCH_SIZE = $(awk "BEGIN{printf \"%'d\", $EPOCHS*$BATCHES_PER_EPOCH*$BATCH_SIZE}") sample draws
EOF

if [ "$AVAIL_GB" -lt "$NEED_GB" ]; then
    FIT=$(awk "BEGIN{printf \"%d\", ($AVAIL_GB-20)*1048576/(4.436*($MASK_KB+8)+69)}")
    echo
    echo "Not enough disk. With ${AVAIL_GB} GB free the maximum is about SCENES=${FIT}." >&2
    echo "Re-run:  SCENES=${FIT} bash $0" >&2
    exit 1
fi
echo

# ------------------------------------------------------------------ build ---
if [ "$SKIP_BUILD" = "1" ]; then
    echo "== 1/7  skipping build (SKIP_BUILD=1) =="
else
    echo "== 1/7  building dataset =="
    BUILD_ARGS=(--out "$DATA_DIR" --zips-dir "$ZIPS_DIR" --scenes "$SCENES" --images-from-zip)
    [ "$PACK_MASKS" = "1" ] && BUILD_ARGS+=(--pack-masks)
    [ "$KEEP_ZIPS" = "1" ] && BUILD_ARGS+=(--keep-zips)
    # The script skips already-downloaded zips and already-extracted files, so re-running is cheap.
    "$PYTHON" script/build_ga_pp_subset.py "${BUILD_ARGS[@]}"
fi

# ------------------------------------------------------------------ split ---
if [ "$SKIP_SPLIT" = "1" ]; then
    echo
    echo "== 2/7  skipping split (SKIP_SPLIT=1) =="
elif [ -f "$SPLIT_DIR/seen.obj" ] && [ -f "$SPLIT_DIR/unseen.obj" ]; then
    echo
    echo "== 2/7  split already present in $SPLIT_DIR, skipping =="
else
    echo
    echo "== 2/7  building the 70/30 Base/New split =="
    "$PYTHON" split/build_grasp_anything_pp.py --data-dir "$DATA_DIR" --out-dir "$SPLIT_DIR"
fi

# ------------------------------------------------------------- data check ---
# Run once before spending GPU time: is part_mask really at part level, how much foreground is
# there, how many tokens survive filtering in each prompt. If the masks turn out to be at object
# level, L_align cannot teach part-level grounding and the whole ablation table will show small
# deltas -- better to know beforehand.
if [ "$SKIP_CHECK" = "1" ]; then
    echo
    echo "== 3/7  skipping data check (SKIP_CHECK=1) =="
else
    echo
    echo "== 3/7  data check =="
    mkdir -p "$RUN_RESULTS"
    "$PYTHON" script/check_dataset.py \
        --data-dir "$DATA_DIR" --split-path "$SPLIT_DIR" \
        --input-size "$INPUT_SIZE" --out "$RUN_RESULTS/dataset-check" 2>&1 \
    | tee "$RUN_RESULTS/dataset_check.log"
fi

# ------------------------------------------------------------------ train ---
if [ "$SKIP_TRAIN" = "1" ]; then
    echo
    echo "== 4/7  skipping training (SKIP_TRAIN=1) =="
else
    echo
    echo "== 4/7  train  (log: $RUN_RESULTS/train.log) =="
    mkdir -p "$RUN_RESULTS"
    # Record the exact configuration used -- the report needs real numbers, not recollection.
    printf "%s\n" \
        "data_dir=$DATA_DIR" "split_dir=$SPLIT_DIR" "scenes=$SCENES" \
        "pack_masks=$PACK_MASKS" "network=$NETWORK" "epochs=$EPOCHS" \
        "batches_per_epoch=$BATCHES_PER_EPOCH" "batch_size=$BATCH_SIZE" \
        "input_size=$INPUT_SIZE" "val_split=$VAL_SPLIT" "lr=$LR" \
        "lr_schedule=$LR_SCHEDULE" "use_text=$USE_TEXT" "w_align=$W_ALIGN" \
        "align_mode=$ALIGN_MODE" "region_text=$REGION_TEXT" "fusion=$FUSION" \
        "align_stage=$ALIGN_STAGE" "warmup_epochs=$WARMUP_EPOCHS" \
        "diag_interval=$DIAG_INTERVAL" "probe_samples=$PROBE_SAMPLES" \
        "counterfactual_every=$COUNTERFACTUAL_EVERY" \
        "num_workers=$NUM_WORKERS" "val_batch_size=$VAL_BATCH_SIZE" \
        "amp=$AMP" "channels_last=$CHANNELS_LAST" \
        "budget_samples=$(( EPOCHS * BATCHES_PER_EPOCH * BATCH_SIZE ))" \
        > "$RUN_RESULTS/config.txt"
    "$PYTHON" train_network.py \
        --dataset grasp-anything-pp \
        --dataset-path "$DATA_DIR" \
        --split-path "$SPLIT_DIR" \
        --network "$NETWORK" \
        --use-depth 0 --use-rgb 1 --seen 1 \
        --input-size "$INPUT_SIZE" \
        --split "$VAL_SPLIT" \
        --epochs "$EPOCHS" \
        --batches-per-epoch "$BATCHES_PER_EPOCH" \
        --batch-size "$BATCH_SIZE" \
        --val-batch-size "$VAL_BATCH_SIZE" \
        --num-workers "$NUM_WORKERS" \
        --tokenize-in-loader "$TOKENIZE_IN_LOADER" \
        --amp "$AMP" --channels-last "$CHANNELS_LAST" \
        --lr "$LR" --lr-schedule "$LR_SCHEDULE" \
        --use-text "$USE_TEXT" \
        --w-align "$W_ALIGN" \
        --align-mode "$ALIGN_MODE" --region-text "$REGION_TEXT" \
        --fusion "$FUSION" --align-stage "$ALIGN_STAGE" \
        --warmup-epochs "$WARMUP_EPOCHS" \
        --diag-interval "$DIAG_INTERVAL" \
        --probe-samples "$PROBE_SAMPLES" \
        --counterfactual-every "$COUNTERFACTUAL_EVERY" \
        --logdir "$LOGDIR" --description "$DESCRIPTION" 2>&1 | tee "$RUN_RESULTS/train.log"
fi

# ------------------------------------------------------------------- eval ---
if [ "$SKIP_EVAL" = "1" ]; then
    echo
    echo "== 5/7  skipping eval (SKIP_EVAL=1) =="
    exit 0
fi
mkdir -p "$RUN_RESULTS"

# -type d: $LOGDIR also holds run_ablation.sh .log files, which `ls -d` would match wrongly.
RUN_DIR=$(find "$LOGDIR" -maxdepth 1 -type d -name "*${DESCRIPTION}*" 2>/dev/null | sort | tail -1)
if [ -z "$RUN_DIR" ]; then
    echo "No log directory matching *$DESCRIPTION* found in $LOGDIR" >&2
    exit 1
fi
# Sort on the filename only -- the full path contains underscores, so `sort -t_` on it shifts columns.
BEST=$(cd "$RUN_DIR" && ls | grep '^epoch_' | sort -t_ -k4 -g | tail -1 || true)
CHECKPOINT=${BEST:+$RUN_DIR/$BEST}
if [ -z "$CHECKPOINT" ]; then
    echo "No checkpoint found in $RUN_DIR" >&2
    exit 1
fi

echo
echo "== 5/7  eval: $CHECKPOINT =="

run_eval() {  # run_eval <seen 1|0> <split> <log filename>
    "$PYTHON" evaluate.py \
        --dataset grasp-anything-pp \
        --dataset-path "$DATA_DIR" \
        --split-path "$SPLIT_DIR" \
        --network "$CHECKPOINT" \
        --iou-eval --use-depth 0 --use-rgb 1 \
        --seen "$1" --split "$2" \
        --num-workers "$NUM_WORKERS" 2>&1 \
    | tee "$RUN_RESULTS/$3" /dev/stderr | awk '/IOU Results/ { print $NF }' | tail -1
}

echo "-- seen (Base), exactly the part held out during training --"
SEEN=$(run_eval 1 "$VAL_SPLIT" eval_seen.log)
echo "-- unseen (New), all of it --"
UNSEEN=$(run_eval 0 0.0 eval_unseen.log)

"$PYTHON" - "$SEEN" "$UNSEEN" "$CHECKPOINT" <<'PY'
import sys

seen, unseen, ckpt = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
h = 0.0 if seen + unseen == 0 else 2 * seen * unseen / (seen + unseen)
print()
# A machine-readable line, so the ablation table script can pick up the numbers (see run_ablation.sh).
print(f"RESULT\t{seen:.4f}\t{unseen:.4f}\t{h:.4f}")
print(f"checkpoint: {ckpt}")
print(f"{'':24} {'Seen':>7} {'Unseen':>7} {'H':>7}")
print(f"{'ours':24} {seen:7.3f} {unseen:7.3f} {h:7.3f}")
for name, s, u, hh in [("GR-ConvNet + CLIP †", 0.37, 0.18, 0.24),
                       ("CLIP-Fusion †", 0.40, 0.29, 0.33),
                       ("LGD †", 0.48, 0.42, 0.45)]:
    print(f"{name:24} {s:7.2f} {u:7.2f} {hh:7.2f}")
print("\n† Table 2 of the paper (full GA++, original split over LVIS labels). The split here is")
print("  a reproduction of the protocol, so these are reference points only.")
PY

# ------------------------------------------------------------------ audit ---
# The mandatory counterfactual for a language-driven method: pair images with another object's
# prompt. If accuracy does not move, the Seen/Unseen table above proves nothing about the
# language mattering.
if [ "$SKIP_AUDIT" = "1" ] || [ "$USE_TEXT" = "0" ]; then
    echo
    echo "== 6/7  skipping audit =="
else
    echo
    echo "== 6/7  wrong-prompt counterfactual ($AUDIT_SAMPLES samples/split) =="
    "$PYTHON" script/audit_text_reliance.py \
        --checkpoint "$CHECKPOINT" \
        --dataset-path "$DATA_DIR" \
        --split-path "$SPLIT_DIR" \
        --input-size "$INPUT_SIZE" \
        --n-samples "$AUDIT_SAMPLES" \
        --figures "$RUN_RESULTS/figures" \
        --out "$RUN_RESULTS/audit_text_reliance.json" 2>&1 \
    | tee "$RUN_RESULTS/audit_text_reliance.log"
fi

if [ "$SKIP_EXPORT" = "1" ]; then
    echo
    echo "== 7/7  skipping export (SKIP_EXPORT=1) =="
    exit 0
fi

echo
echo "== 7/7  collecting results into $RUN_RESULTS =="
EXPORT_ARGS=(
    --checkpoint "$CHECKPOINT"
    --dataset-path "$DATA_DIR"
    --split-path "$SPLIT_DIR"
    --out "$RUN_RESULTS"
    --n-samples "$N_FIGURES"
    --input-size "$INPUT_SIZE"
    --seen-acc "$SEEN" --unseen-acc "$UNSEEN"
    --tensorboard-dir "$RUN_DIR"
)
[ "$COPY_CHECKPOINT" = "1" ] && EXPORT_ARGS+=(--copy-checkpoint)
"$PYTHON" script/export_results.py "${EXPORT_ARGS[@]}"

echo
echo "Done. Results in $RUN_RESULTS"
