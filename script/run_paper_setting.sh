#!/usr/bin/env bash
#
# Tải dữ liệu Grasp-Anything++ (có cache) + train + eval theo setting của paper LGD.
#
# Paper (Vuong et al., CVPR 2024) nêu rõ những gì:
#   * train/eval trên GA++ đầy đủ
#   * split Base/New = 70/30 category theo tần suất  (§5.1)
#   * success khi IoU >= 0.25 VÀ lệch góc <= 30 độ, báo cáo Seen / Unseen / harmonic mean H
#   * 100 epoch (Fig 6)
# Paper KHÔNG nêu trong bản chính: batch size, optimizer, learning rate -- nằm ở Supplementary
# mà ta không có. Những giá trị đó ở dưới là lựa chọn của ta, ghi rõ để báo cáo trung thực.
#
# Lưu ý về "epoch": repo này định nghĩa epoch = BATCHES_PER_EPOCH batch, không phải một lượt
# qua hết dữ liệu. 100 x 2000 x 64 = 12,8 triệu lượt sample = ~2,9 lượt qua 4,41 triệu sample.
# Một lượt thật sự x100 là ~280 ngày, không ai làm thế.
#
# Mọi tham số override được bằng biến môi trường:
#   SCENES=200000 NUM_WORKERS=24 bash script/run_paper_setting.sh
#
# Các bước đều có cache: chạy lại sẽ bỏ qua phần đã xong.

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python3}

DATA_DIR=${DATA_DIR:-data/grasp-anything-pp-full}
SPLIT_DIR=${SPLIT_DIR:-split/grasp-anything-pp}
ZIPS_DIR=${ZIPS_DIR:-$DATA_DIR/_archives}

TOTAL_SCENES=994860
SCENES=${SCENES:-$TOTAL_SCENES}       # mặc định: toàn bộ
PACK_MASKS=${PACK_MASKS:-1}           # 21 KB/mask thay vì 173 KB, không mất mát
KEEP_ZIPS=${KEEP_ZIPS:-1}             # giữ zip để lần sau đổi SCENES không phải tải lại

NPROC=$(nproc)
NUM_WORKERS=${NUM_WORKERS:-$(( NPROC / 2 > 32 ? 32 : (NPROC / 2 < 2 ? 2 : NPROC / 2) ))}

NETWORK=${NETWORK:-grconvnet3_align}
EPOCHS=${EPOCHS:-100}
BATCHES_PER_EPOCH=${BATCHES_PER_EPOCH:-2000}
BATCH_SIZE=${BATCH_SIZE:-64}
INPUT_SIZE=${INPUT_SIZE:-224}
VAL_SPLIT=${VAL_SPLIT:-0.998}         # validate chạy batch_size=1 nên giữ tập val nhỏ
LR=${LR:-1e-3}
LR_SCHEDULE=${LR_SCHEDULE:-cosine}
USE_TEXT=${USE_TEXT:-1}
W_ALIGN=${W_ALIGN:-0.3}
W_AGNOSTIC=${W_AGNOSTIC:-0.3}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-3}
DESCRIPTION=${DESCRIPTION:-ga-pp-paper}
LOGDIR=${LOGDIR:-logs/}

RESULTS_DIR=${RESULTS_DIR:-results}
N_FIGURES=${N_FIGURES:-4}             # số sample vẽ alignment cho mỗi split
COPY_CHECKPOINT=${COPY_CHECKPOINT:-0} # chép cả checkpoint ~265 MB vào results/

SKIP_BUILD=${SKIP_BUILD:-0}
SKIP_SPLIT=${SKIP_SPLIT:-0}
SKIP_TRAIN=${SKIP_TRAIN:-0}
SKIP_EVAL=${SKIP_EVAL:-0}
SKIP_EXPORT=${SKIP_EXPORT:-0}

RUN_STAMP=$(date +%y%m%d_%H%M)
RUN_RESULTS="$RESULTS_DIR/${RUN_STAMP}_${DESCRIPTION}"

# --------------------------------------------------------------- kế hoạch ---
SAMPLES=$(awk "BEGIN{printf \"%d\", $SCENES * 4.436}")
if [ "$PACK_MASKS" = "1" ]; then MASK_KB=24; else MASK_KB=176; fi
# .pt và .pkl mỗi file chiếm tối thiểu một block 4 KB, cộng vào cho sát thực tế.
NEED_GB=$(awk "BEGIN{printf \"%d\", ($SAMPLES*($MASK_KB+8) + $SCENES*69)/1048576 + 1}")
AVAIL_GB=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')

cat <<EOF
== kế hoạch ==
  scene            : $(printf "%'d" "$SCENES") / $(printf "%'d" $TOTAL_SCENES)
  sample (ước tính): $(printf "%'d" "$SAMPLES")
  pack-masks       : $PACK_MASKS
  đĩa cần          : ~${NEED_GB} GB (chưa kể ~76 GB archive nếu KEEP_ZIPS=1)
  đĩa trống        : ${AVAIL_GB} GB
  workers          : $NUM_WORKERS (máy có $NPROC core)
  ngân sách train  : $EPOCHS x $BATCHES_PER_EPOCH x $BATCH_SIZE = $(awk "BEGIN{printf \"%'d\", $EPOCHS*$BATCHES_PER_EPOCH*$BATCH_SIZE}") lượt sample
EOF

if [ "$AVAIL_GB" -lt "$NEED_GB" ]; then
    FIT=$(awk "BEGIN{printf \"%d\", ($AVAIL_GB-20)*1048576/(4.436*($MASK_KB+8)+69)}")
    echo
    echo "Không đủ đĩa. Với ${AVAIL_GB} GB trống thì tối đa khoảng SCENES=${FIT}." >&2
    echo "Chạy lại:  SCENES=${FIT} bash $0" >&2
    exit 1
fi
echo

# ------------------------------------------------------------------ build ---
if [ "$SKIP_BUILD" = "1" ]; then
    echo "== 1/4  bỏ qua build (SKIP_BUILD=1) =="
else
    echo "== 1/4  dựng dataset =="
    BUILD_ARGS=(--out "$DATA_DIR" --zips-dir "$ZIPS_DIR" --scenes "$SCENES" --images-from-zip)
    [ "$PACK_MASKS" = "1" ] && BUILD_ARGS+=(--pack-masks)
    [ "$KEEP_ZIPS" = "1" ] && BUILD_ARGS+=(--keep-zips)
    # Script tự bỏ qua zip đã tải và file đã trích, nên chạy lại là rẻ.
    "$PYTHON" script/build_ga_pp_subset.py "${BUILD_ARGS[@]}"
fi

# ------------------------------------------------------------------ split ---
if [ "$SKIP_SPLIT" = "1" ]; then
    echo
    echo "== 2/4  bỏ qua split (SKIP_SPLIT=1) =="
elif [ -f "$SPLIT_DIR/seen.obj" ] && [ -f "$SPLIT_DIR/unseen.obj" ]; then
    echo
    echo "== 2/4  split đã có ở $SPLIT_DIR, bỏ qua =="
else
    echo
    echo "== 2/4  dựng split Base/New 70/30 =="
    "$PYTHON" split/build_grasp_anything_pp.py --data-dir "$DATA_DIR" --out-dir "$SPLIT_DIR"
fi

# ------------------------------------------------------------------ train ---
if [ "$SKIP_TRAIN" = "1" ]; then
    echo
    echo "== 3/4  bỏ qua train (SKIP_TRAIN=1) =="
else
    echo
    echo "== 3/5  train  (log: $RUN_RESULTS/train.log) =="
    mkdir -p "$RUN_RESULTS"
    # Ghi lại đúng cấu hình đã dùng -- báo cáo cần con số thật chứ không phải trí nhớ.
    printf "%s\n" \
        "data_dir=$DATA_DIR" "split_dir=$SPLIT_DIR" "scenes=$SCENES" \
        "pack_masks=$PACK_MASKS" "network=$NETWORK" "epochs=$EPOCHS" \
        "batches_per_epoch=$BATCHES_PER_EPOCH" "batch_size=$BATCH_SIZE" \
        "input_size=$INPUT_SIZE" "val_split=$VAL_SPLIT" "lr=$LR" \
        "lr_schedule=$LR_SCHEDULE" "use_text=$USE_TEXT" "w_align=$W_ALIGN" \
        "w_agnostic=$W_AGNOSTIC" "warmup_epochs=$WARMUP_EPOCHS" \
        "num_workers=$NUM_WORKERS" \
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
        --num-workers "$NUM_WORKERS" \
        --lr "$LR" --lr-schedule "$LR_SCHEDULE" \
        --use-text "$USE_TEXT" \
        --w-align "$W_ALIGN" --w-agnostic "$W_AGNOSTIC" \
        --warmup-epochs "$WARMUP_EPOCHS" \
        --logdir "$LOGDIR" --description "$DESCRIPTION" 2>&1 | tee "$RUN_RESULTS/train.log"
fi

# ------------------------------------------------------------------- eval ---
if [ "$SKIP_EVAL" = "1" ]; then
    echo
    echo "== 4/5  bỏ qua eval (SKIP_EVAL=1) =="
    exit 0
fi
mkdir -p "$RUN_RESULTS"

# -type d: $LOGDIR còn chứa cả file .log của run_ablation_4gpu.sh, `ls -d` sẽ khớp nhầm.
RUN_DIR=$(find "$LOGDIR" -maxdepth 1 -type d -name "*${DESCRIPTION}*" 2>/dev/null | sort | tail -1)
if [ -z "$RUN_DIR" ]; then
    echo "Không thấy thư mục log nào khớp *$DESCRIPTION* trong $LOGDIR" >&2
    exit 1
fi
# Sắp theo tên file thôi -- đường dẫn đầy đủ có underscore nên `sort -t_` trên nó sẽ lệch cột.
BEST=$(cd "$RUN_DIR" && ls | grep '^epoch_' | sort -t_ -k4 -g | tail -1 || true)
CHECKPOINT=${BEST:+$RUN_DIR/$BEST}
if [ -z "$CHECKPOINT" ]; then
    echo "Không thấy checkpoint trong $RUN_DIR" >&2
    exit 1
fi

echo
echo "== 4/5  eval: $CHECKPOINT =="

run_eval() {  # run_eval <seen 1|0> <split> <tên file log>
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

echo "-- seen (Base), đúng phần giữ lại lúc train --"
SEEN=$(run_eval 1 "$VAL_SPLIT" eval_seen.log)
echo "-- unseen (New), toàn bộ --"
UNSEEN=$(run_eval 0 0.0 eval_unseen.log)

"$PYTHON" - "$SEEN" "$UNSEEN" "$CHECKPOINT" <<'PY'
import sys

seen, unseen, ckpt = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
h = 0.0 if seen + unseen == 0 else 2 * seen * unseen / (seen + unseen)
print()
# Dòng máy đọc được, để script gộp bảng ablation lấy số (xem run_ablation_4gpu.sh).
print(f"RESULT\t{seen:.4f}\t{unseen:.4f}\t{h:.4f}")
print(f"checkpoint: {ckpt}")
print(f"{'':24} {'Seen':>7} {'Unseen':>7} {'H':>7}")
print(f"{'ours':24} {seen:7.3f} {unseen:7.3f} {h:7.3f}")
for name, s, u, hh in [("GR-ConvNet + CLIP †", 0.37, 0.18, 0.24),
                       ("CLIP-Fusion †", 0.40, 0.29, 0.33),
                       ("LGD †", 0.48, 0.42, 0.45)]:
    print(f"{name:24} {s:7.2f} {u:7.2f} {hh:7.2f}")
print("\n† Table 2 của paper (GA++ đầy đủ, split gốc theo nhãn LVIS). Split ở đây là bản tái")
print("  tạo protocol nên chỉ dùng làm mốc tham chiếu.")
PY

if [ "$SKIP_EXPORT" = "1" ]; then
    echo
    echo "== 5/5  bỏ qua export (SKIP_EXPORT=1) =="
    exit 0
fi

echo
echo "== 5/5  gom kết quả vào $RUN_RESULTS =="
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
echo "Xong. Kết quả ở $RUN_RESULTS"
