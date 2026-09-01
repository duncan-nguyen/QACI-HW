#!/usr/bin/env bash
#
# The ablation table -- every arm runs **sequentially on one GPU**.
#
# Sequential, not parallel: train_network.py has no DataParallel/DDP, so each run uses exactly
# one GPU, and the real bottleneck is the CPU dataloader. Cramming four processes onto one GPU
# only makes them fight over the CPU and slows each arm down almost fourfold -- the same total
# time, but with no result finishing early to look at.
#
# Each line enables exactly ONE more component than the line above, so the difference between
# two adjacent lines is attributable to that component:
#
#   notext  : plain GR-ConvNet, no language               (use_text=0)
#   hardmax : argmax over tokens + gate F*(1+A_T)         (align_mode=hard, fusion=gate)
#   soft    : softmax over tokens + residual fusion       (align_mode=soft, fusion=residual)
#   full    : STAG, adding the region text R_T            (+ region_text=1)
#
# Two extra arms, for isolating a single variable:
#   noalign : A_T present but NO L_align -> isolates the mask supervision
#   conv4   : like full, but A_T computed at 113x113 instead of 56x56
#
# The training budget is identical for every arm (EPOCHS x BATCHES_PER_EPOCH x BATCH_SIZE) --
# a hard requirement for the table to mean anything. The script prints that number and writes
# it into each arm's config.txt.
#
#   bash script/run_ablation.sh                      # all arms, sequential, GPU 0
#   GPU=1 bash script/run_ablation.sh                # pick another GPU
#   ARMS="soft full" bash script/run_ablation.sh     # re-run a couple of arms
#   RESUME=1 bash script/run_ablation.sh             # skip arms that already have results
#
# If interrupted, re-run with RESUME=1: any arm with a RESULT line in its log is skipped.

set -euo pipefail
cd "$(dirname "$0")/.."

ARMS=${ARMS:-"notext soft full noalign conv4"}
GPU=${GPU:-0}
LOGDIR=${LOGDIR:-logs/}
DESCRIPTION=${DESCRIPTION:-ga-pp-abl}
RESUME=${RESUME:-0}

# The training budget, identical across arms. Changing it here changes the whole table.
EPOCHS=${EPOCHS:-50}
BATCHES_PER_EPOCH=${BATCHES_PER_EPOCH:-2000}
BATCH_SIZE=${BATCH_SIZE:-64}
SCENES=${SCENES:-100000}
export EPOCHS BATCHES_PER_EPOCH BATCH_SIZE SCENES

# One training process at a time -> the whole CPU is available to the dataloader.
NPROC=$(nproc)
NUM_WORKERS=${NUM_WORKERS:-$(( NPROC / 2 > 32 ? 32 : (NPROC / 2 < 2 ? 2 : NPROC / 2) ))}
export NUM_WORKERS

W=${W_ALIGN:-0.3}
arm_flags() {
    case "$1" in
        notext)  echo "USE_TEXT=0 W_ALIGN=0" ;;
        hardmax) echo "USE_TEXT=1 W_ALIGN=$W ALIGN_MODE=hard FUSION=gate REGION_TEXT=0" ;;
        soft)    echo "USE_TEXT=1 W_ALIGN=$W ALIGN_MODE=soft FUSION=residual REGION_TEXT=0" ;;
        full)    echo "USE_TEXT=1 W_ALIGN=$W ALIGN_MODE=soft FUSION=residual REGION_TEXT=1" ;;
        noalign) echo "USE_TEXT=1 W_ALIGN=0 ALIGN_MODE=soft FUSION=residual REGION_TEXT=1" ;;
        conv4)   echo "USE_TEXT=1 W_ALIGN=$W ALIGN_MODE=soft FUSION=residual REGION_TEXT=1 ALIGN_STAGE=conv4" ;;
        *) echo "invalid arm: $1" >&2; exit 1 ;;
    esac
}

N_ARMS=$(echo "$ARMS" | wc -w)
BUDGET=$(( EPOCHS * BATCHES_PER_EPOCH * BATCH_SIZE ))
ARM_LOGS="$LOGDIR/ablation"
mkdir -p "$ARM_LOGS"

cat <<EOF
== plan ==
  arms             : $ARMS  ($N_ARMS arms, run sequentially)
  GPU              : $GPU
  scenes           : $SCENES
  budget per arm   : $EPOCHS x $BATCHES_PER_EPOCH x $BATCH_SIZE = $BUDGET sample draws
  workers          : $NUM_WORKERS (machine has $NPROC cores)
  logs             : $ARM_LOGS/<arm>.log

Total time ~ $N_ARMS times a single run. Follow along with: tail -f $ARM_LOGS/<arm>.log
EOF
echo

echo "== 1/3  prepare and check the data (once, before the arm loop) =="
# Done fully here so no arm has to download/extract again. The data check
# (script/check_dataset.py) also only needs to run once -- the arms then use SKIP_CHECK=1.
GPU="$GPU" SKIP_TRAIN=1 SKIP_EVAL=1 bash script/run_paper_setting.sh

echo
echo "== 2/3  running $N_ARMS arms =="
fail=0
i=0
for arm in $ARMS; do
    i=$((i + 1))
    log="$ARM_LOGS/${arm}.log"
    if [ "$RESUME" = "1" ] && grep -q '^RESULT' "$log" 2>/dev/null; then
        echo "  [$i/$N_ARMS] skipping $arm -- results already in $log (RESUME=1)"
        continue
    fi
    echo "  [$i/$N_ARMS] $arm  ($(date +%H:%M:%S))  -> $log"
    if env $(arm_flags "$arm") \
        GPU="$GPU" \
        DESCRIPTION="${DESCRIPTION}-${arm}" \
        SKIP_BUILD=1 SKIP_SPLIT=1 SKIP_CHECK=1 \
        bash script/run_paper_setting.sh > "$log" 2>&1
    then
        echo "        done  ($(date +%H:%M:%S))  $(grep -h '^RESULT' "$log" | tail -1 || true)"
    else
        # One broken arm should not cost the others -- record it and carry on.
        echo "        FAILED -- see $log" >&2
        tail -5 "$log" >&2 || true
        fail=1
    fi
done

echo
echo "== 3/3  ablation table =="
echo "budget per arm: $EPOCHS x $BATCHES_PER_EPOCH x $BATCH_SIZE = $BUDGET sample draws · $SCENES scenes"
echo
printf "%-10s %-62s %7s %7s %7s\n" "arm" "configuration" "Seen" "Unseen" "H"
for arm in $ARMS; do
    log="$ARM_LOGS/${arm}.log"
    line=$(grep -h '^RESULT' "$log" 2>/dev/null | tail -1 || true)
    if [ -z "$line" ]; then
        printf "%-10s %-62s %7s %7s %7s\n" "$arm" "$(arm_flags "$arm")" "-" "-" "-"
    else
        printf "%-10s %-62s %7.3f %7.3f %7.3f\n" "$arm" "$(arm_flags "$arm")" \
            "$(echo "$line" | cut -f2)" "$(echo "$line" | cut -f3)" "$(echo "$line" | cut -f4)"
    fi
done
cat <<'EOF'

Reference points (Table 2 of the paper: full GA++, 4.41M samples, 100 epochs, the original
split over LVIS labels). We run 10% of the data for 50 epochs, so these are for orientation,
not for claiming a win:
  GR-ConvNet + CLIP   0.37 / 0.18 / 0.24
  CLIP-Fusion         0.40 / 0.29 / 0.33
  LGD                 0.48 / 0.42 / 0.45

After this table, the next number to look at is in results/<run>/audit_text_reliance.log:
Delta_shuffled ~ 0 means that arm is not really using the prompt, however high Seen/Unseen is.
EOF
exit "$fail"
