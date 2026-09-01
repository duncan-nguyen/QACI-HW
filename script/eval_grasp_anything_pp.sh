#!/bin/bash
#
# Evaluate one checkpoint on Grasp-Anything++ following the LGD paper's protocol (Sec. 5.1):
# run both the seen (Base) and unseen (New) sets, then print the harmonic mean H as in Table 2.
#
# The metric computed by evaluate.py: IoU >= 0.25 and angular error <= 30 degrees
# (utils/dataset_processing).
#
# Usage:
#   script/eval_grasp_anything_pp.sh <checkpoint> [dataset_dir] [split_frac]
#
#   split_frac: the leading fraction of the data to SKIP (evaluate.py scores the rest).
#               0.99 = only the last 1%, for speed; 0.0 = the whole set.

set -euo pipefail

NETWORK="${1:?a checkpoint path is required}"
DATA="${2:-data/grasp-anything-pp}"
SPLIT="${3:-0.99}"
PYTHON="${PYTHON:-python}"

if [ ! -f "$NETWORK" ]; then
    echo "Checkpoint not found: $NETWORK" >&2
    exit 1
fi

run_split() {  # run_split <seen 1|0>
    "$PYTHON" evaluate.py \
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

"$PYTHON" - "$SEEN" "$UNSEEN" <<'PY'
import sys

seen, unseen = float(sys.argv[1]), float(sys.argv[2])
h = 0.0 if seen + unseen == 0 else 2 * seen * unseen / (seen + unseen)
print()
print(f"{'Seen':>8} {'Unseen':>8} {'H':>8}")
print(f"{seen:8.4f} {unseen:8.4f} {h:8.4f}")
PY
