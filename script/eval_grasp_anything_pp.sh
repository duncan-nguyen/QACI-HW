#!/bin/bash
#
# Đánh giá một checkpoint trên Grasp-Anything++ theo protocol của paper LGD (§5.1):
# chạy cả tập seen (Base) lẫn unseen (New), rồi in harmonic mean H như Table 2.
#
# Metric do evaluate.py tính: IoU >= 0.25 và lệch góc <= 30 độ (utils/dataset_processing).
#
# Usage:
#   script/eval_grasp_anything_pp.sh <checkpoint> [dataset_dir] [split_frac]
#
#   split_frac: phần dữ liệu BỎ QUA ở đầu (evaluate.py đánh giá phần còn lại).
#               0.99 = chỉ lấy 1% cuối cho nhanh; 0.0 = toàn bộ tập.

set -euo pipefail

NETWORK="${1:?cần đường dẫn checkpoint}"
DATA="${2:-data/grasp-anything-pp}"
SPLIT="${3:-0.99}"

if [ ! -f "$NETWORK" ]; then
    echo "Không thấy checkpoint: $NETWORK" >&2
    exit 1
fi

run_split() {  # run_split <seen 1|0>
    python evaluate.py \
        --dataset grasp-anything-pp \
        --dataset-path "$DATA" \
        --network "$NETWORK" \
        --iou-eval \
        --use-depth 0 --use-rgb 1 \
        --seen "$1" \
        --split "$SPLIT" \
        --num-workers 4 2>&1 \
    | tee /dev/stderr \
    | awk '/IOU Results/ { print $NF }' | tail -1
}

echo "== seen (Base) =="
SEEN=$(run_split 1)
echo "== unseen (New) =="
UNSEEN=$(run_split 0)

python - "$SEEN" "$UNSEEN" <<'PY'
import sys

seen, unseen = float(sys.argv[1]), float(sys.argv[2])
h = 0.0 if seen + unseen == 0 else 2 * seen * unseen / (seen + unseen)
print()
print(f"{'Seen':>8} {'Unseen':>8} {'H':>8}")
print(f"{seen:8.4f} {unseen:8.4f} {h:8.4f}")
PY
