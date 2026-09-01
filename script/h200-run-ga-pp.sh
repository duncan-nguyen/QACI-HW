#!/usr/bin/env bash
#
# Train GA++ 200k trên H200, bản v2 -- có Update 1/2/3 và các fix sau lần chạy 260830.
#
# Khác bản cũ (results/h200-run-ga-pp-200k.sh) ở những điểm sau, mỗi điểm đều có lý do từ log
# của lần chạy trước:
#
#   * SPLIT MỚI. Split cũ chia 881.258 / 956 sample (99,89% / 0,11%) vì lấy 70% category
#     *đầu bảng* tần suất cho Base, nên New chỉ còn cái đuôi hiếm nhất. Bản này dùng
#     --split-mode stratified. Kết quả KHÔNG so sánh trực tiếp được với run cũ.
#   * ALIGN WEIGHTS. Tính trước d_i = 1 - IoU(part_mask_i, hợp mask cùng object) và nhân vào
#     L_align (Update 2). Dòng `mean_d_multi_part` ở đầu train.log là câu trả lời cho câu hỏi
#     "part_mask có thật sự ở mức part không".
#   * MASK ANGLE LOSS (Update 1). cos/sin/width chỉ tính trong vùng grasp.
#   * L_TOKEN (Update 3), chỉ bật ở RUN=C.
#   * WARM START từ weights/model_grasp_anything (GR-ConvNet3 đã train trên GA đầy đủ, cùng
#     kiến trúc, RGB 3 kênh, 115/115 tensor backbone khớp).
#   * EPOCHS=40 thay vì 100 -- xem phần "vì sao 40" bên dưới.
#   * Chọn checkpoint từ checkpoints.jsonl (có iou đầy đủ chữ số) thay vì từ tên file đã làm
#     tròn 2 chữ số.
#   * Validation KHÔNG augmentation (mặc định mới của train_network.py).
#
# ---------------------------------------------------------------------------
# VÌ SAO 40 EPOCH
#
# Từ đường cong validation của run 260830 (98 epoch, đọc bằng tensorboard events):
#
#     ep0-19   độ dốc +0,576 pp/epoch   <- học thật
#     ep20-39  độ dốc +0,089 pp/epoch   <- đã dưới mức nhiễu
#     ep40-59  độ dốc +0,025 pp/epoch
#     ep60-79  độ dốc -0,021 pp/epoch
#     ep80-97  độ dốc +0,019 pp/epoch
#     nhiễu val IoU trên plateau: std = 1,00 pp
#
# Từ epoch ~25 trở đi độ dốc nhỏ hơn nhiễu một bậc; 73 epoch cuối không mua được gì đo được.
# Và đó là run KHÔNG warm start: phần dốc ep0-19 chủ yếu là học hình học grasp cơ bản, thứ
# mà --init-from đã có sẵn. 40 nằm dư dả sau mốc hội tụ.
#
# KHÔNG mở rộng được sau khi chạy xong: cosine T_max = EPOCHS và code không có resume. Nếu
# 10 epoch cuối vẫn còn dốc (|slope| x 10 > 2 x std -- script in sẵn mean/std của 10 epoch
# cuối ở stage PICK_CHECKPOINT) thì phải chạy lại với EPOCHS lớn hơn, không vá được.
#
# 40 cho ba run so sánh A/B/C. Nếu sau đó cần MỘT run để báo cáo thì 60 (max val IoU của
# run cũ trong 60 epoch đầu là 0,3284 so với 0,3313 của cả 98 epoch -- đã bão hoà).
# Đừng đổi EPOCHS giữa A, B, C: khác ngân sách thì không so được với nhau.
#
# LƯU Ý khi đọc kết quả: max val IoU trong 50 epoch đầu của run cũ là 0,3165 còn của cả 98
# epoch là 0,3313. Chênh lệch đó gần như hoàn toàn là "max của nhiều lần rút thăm nhiễu",
# không phải model tốt hơn -- trung bình plateau ep40-97 chỉ là 0,3142. Vì vậy:
#   - đừng so max val IoU giữa hai run có số epoch khác nhau;
#   - so bằng trung bình 10 epoch cuối, hoặc so ở cùng mốc epoch.
# Tắt augmentation ở validation làm nhiễu này nhỏ đi đáng kể, nên hiệu ứng trên cũng nhỏ đi.
#
# EPOCHS ở đây đi kèm cosine T_max = EPOCHS. Đừng "dừng sớm" một lịch 100 epoch: làm thế thì
# learning rate chưa kịp anneal và kết quả sẽ tệ hơn hẳn chạy đủ lịch 50.
#
# ---------------------------------------------------------------------------
# ĐỔI SỐ SCENE (vd 100k thay vì 200k)
#
# Builder giữ 1 scene trong mỗi round(994.860 / SCENES):
#     SCENES=200000 -> 1/5  -> ~198.972 scene -> ~882.600 sample   (~41 GB)
#     SCENES=100000 -> 1/10 ->  ~99.486 scene -> ~441.300 sample   (~21 GB)
#
# Dựng dataset 100k (một lần, KHÔNG dùng script này):
#     SCENES=100000 DATA_DIR=$PROJECT/data/grasp-anything-pp-100k \
#     SKIP_TRAIN=1 SKIP_EVAL=1 bash script/run_paper_setting.sh
#
# Rồi chạy:
#     SCENES=100000 \
#     DATA_DIR=$PROJECT/data/grasp-anything-pp-100k \
#     SPLIT_DIR=$PROJECT/data/grasp-anything-pp-100k-split-v2 \
#     RUN=B bash script/h200-run-ga-pp-200k-v2.sh
#
# Cái gì KHÔNG đổi:
#   * EPOCHS, BATCHES_PER_EPOCH, BATCH_SIZE, LR, NUM_WORKERS. Ở repo này một "epoch" là
#     BATCHES_PER_EPOCH batch, KHÔNG phải một lượt qua hết dữ liệu -- nên **giảm dữ liệu
#     không làm giảm thời gian train**. 40 epoch vẫn là 5,12 triệu lượt sample, vẫn ~2,8 giờ.
#     Giữ nguyên để ba run A/B/C và hai kích thước dataset còn so sánh được với nhau.
#   * Thứ thay đổi là số lượt quét qua dữ liệu, tính trên tập seen của split stratified
#     (~70% dữ liệu): 8,3 lượt (200k) -> 16,6 lượt (100k).
#
# Cái gì tự đổi theo: VAL_SPLIT và EVAL_UNSEEN_SPLIT (mặc định `auto`, tính từ số sample
# thật của split để đạt VAL_TARGET / EVAL_UNSEEN_TARGET mẫu). Đừng hardcode chúng.
#
# Cái gì phải theo dõi thêm: overfitting. Run 260830 chạy 7,3 lượt và train ≈ val trên mọi
# thành phần (khoảng cách ~2%), tức underfitting. Ở 14,5 lượt thì khoảng cách đó có thể mở
# ra. Log đã in train và val riêng cho từng thành phần loss -- nếu khoảng cách mở ra trước
# epoch 40, đó là một *phát hiện* (model cuối cùng cũng bị ép về dung lượng), không phải sự cố.
#
# Có nên dùng 100k không: hợp lý cho ba run chẩn đoán. Nút thắt đo được ở run trước là thiết
# kế loss và chất lượng nhãn, không phải lượng dữ liệu -- model đang underfit chứ không thiếu
# dữ liệu. Cái tiết kiệm được là ~20 GB đĩa và một nửa thời gian build/precompute, KHÔNG phải
# thời gian GPU.
#
# ---------------------------------------------------------------------------
# CÁCH DÙNG
#
#     RUN=A bash script/h200-run-ga-pp-200k-v2.sh     # baseline no-text
#     RUN=B bash script/h200-run-ga-pp-200k-v2.sh     # full, Update 1+2
#     RUN=C bash script/h200-run-ga-pp-200k-v2.sh     # thêm Update 3 (L_token)
#
# Chạy B trước. Nếu `mean_d_multi_part` ≈ 0 thì part_mask ở mức object, L_token của run C
# cũng vô nghĩa -- dừng lại và đổi nguồn nhãn thay vì chạy tiếp.
#
# Ước tính thời gian: run cũ đạt 504 sample/s (GPU-bound, loader dư sức ở 16 worker), tức
# 4,25 phút/epoch. 40 epoch ≈ 2,8 giờ mỗi run; ba run ≈ 8,5 giờ.

set -euo pipefail

# ----------------------------------------------------------------- môi trường --
PROJECT=${PROJECT:-/home/tensara/projects/hw}
cd "${CODE_DIR:-$PROJECT/text-image-aware}"
export PATH="$PROJECT/.venv/bin:$PATH"
PYTHON=${PYTHON:-$PROJECT/.venv/bin/python}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export HF_HOME=${HF_HOME:-$PROJECT/.cache/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=60
export HF_HUB_ETAG_TIMEOUT=30

DATA_DIR=${DATA_DIR:-$PROJECT/data/grasp-anything-pp-200k}
# Split mới, thư mục mới: giữ nguyên split cũ để còn đối chiếu lịch sử.
SPLIT_DIR=${SPLIT_DIR:-$PROJECT/data/grasp-anything-pp-200k-split-v2}
LOGDIR=${LOGDIR:-$PROJECT/logs}
RESULTS_DIR=${RESULTS_DIR:-$PROJECT/results}
INIT_FROM=${INIT_FROM:-weights/model_grasp_anything}

# ------------------------------------------------------------ cấu hình chạy --
RUN=${RUN:-C}
EPOCHS=${EPOCHS:-40}
BATCHES_PER_EPOCH=${BATCHES_PER_EPOCH:-2000}
BATCH_SIZE=${BATCH_SIZE:-64}
INPUT_SIZE=${INPUT_SIZE:-224}
# VAL_SPLIT / EVAL_UNSEEN_SPLIT: `auto` = tính từ số sample thật sau khi dựng split, để đạt
# đúng VAL_TARGET / EVAL_UNSEEN_TARGET mẫu. Hardcode 0.998 chỉ đúng cho split CŨ (unseen 956
# mẫu, seen 881k); với split stratified và/hoặc SCENES khác thì nó sai hẳn -- xem ghi chú
# "ĐỔI SỐ SCENE" ở đầu file.
VAL_SPLIT=${VAL_SPLIT:-auto}
EVAL_UNSEEN_SPLIT=${EVAL_UNSEEN_SPLIT:-auto}
VAL_TARGET=${VAL_TARGET:-2000}
EVAL_UNSEEN_TARGET=${EVAL_UNSEEN_TARGET:-3000}
LR=${LR:-0.001}
LR_SCHEDULE=${LR_SCHEDULE:-cosine}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-3}
NUM_WORKERS=${NUM_WORKERS:-16}          # benchmark cũ: 8→489, 16→1433, 24→821 sample/s
NETWORK=${NETWORK:-grconvnet3_align}
MASK_ANGLE_LOSS=${MASK_ANGLE_LOSS:-1}
ALIGN_DIAG_SAMPLES=${ALIGN_DIAG_SAMPLES:-64}
SINGLE_PART_WEIGHT=${SINGLE_PART_WEIGHT:-1.0}

# W_ALIGN: bản cũ dùng 0,3 và L_align chiếm 63% tổng loss trong khi grasp chỉ 22%. Update 1
# đẩy cos/sin/width lên ~19-31 lần, Update 2 lại nhân L_align với mean(d) < 1, nên cán cân
# lật ngược. 3.0 là ước lượng để align về lại ~25%; ĐỌC dòng `epoch 0 train ... |` trong log
# rồi chỉnh lại cho đúng thay vì tin con số này.
case "$RUN" in
    A) USE_TEXT=0; W_ALIGN=0.0; W_AGNOSTIC=0.3; W_TOKEN=0.0
       DESC=${DESCRIPTION:-ga-pp-200k-v2-A-notext} ;;
    B) USE_TEXT=1; W_ALIGN=${W_ALIGN:-3.0}; W_AGNOSTIC=0.3; W_TOKEN=0.0
       DESC=${DESCRIPTION:-ga-pp-200k-v2-B-full} ;;
    C) USE_TEXT=1; W_ALIGN=${W_ALIGN:-3.0}; W_AGNOSTIC=0.3; W_TOKEN=${W_TOKEN:-0.5}
       DESC=${DESCRIPTION:-ga-pp-200k-v2-C-token} ;;
    *) echo "RUN phải là A, B hoặc C (đang là '$RUN')" >&2; exit 2 ;;
esac

RUN_STAMP=$(date +%y%m%d_%H%M)
RUN_RESULTS="$RESULTS_DIR/${RUN_STAMP}_${DESC}"
mkdir -p "$LOGDIR" "$RESULTS_DIR" "$RUN_RESULTS"

# Một GPU, một job. Bản cũ không có khoá nên hai lần chạy chồng nhau là im lặng hỏng cả hai.
exec 9>"$RESULTS_DIR/ga-pp-200k-train.lock"
if ! flock -n 9; then
    echo "Đã có một run khác đang giữ khoá $RESULTS_DIR/ga-pp-200k-train.lock" >&2
    exit 1
fi

trap 'rc=$?; echo "PIPELINE_EXIT_CODE=$rc AT=$(date --iso-8601=seconds) ELAPSED_SECONDS=$SECONDS"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# ------------------------------------------------------------------ preflight --
echo "PIPELINE_START=$(date --iso-8601=seconds) RUN=$RUN"
echo "CODE_REVISION=$(git rev-parse HEAD)"
echo "CODE_DIRTY=$(git status --porcelain | wc -l)"
echo "GPU_SELECTION=$CUDA_VISIBLE_DEVICES"
# Chỉ để ghi lại trạng thái GPU; máy không có nvidia-smi (dry-run trên laptop) vẫn chạy tiếp.
nvidia-smi --id=0 --query-gpu=name,memory.free,utilization.gpu --format=csv,noheader \
    || echo "nvidia-smi không có -- bỏ qua ghi nhận GPU"
test -d "$DATA_DIR" || { echo "Không thấy $DATA_DIR -- dựng dataset trước bằng script/run_paper_setting.sh" >&2; exit 1; }
test -x "$PYTHON" || { echo "Không thấy $PYTHON" >&2; exit 1; }
echo "DATA_DIR=$DATA_DIR SPLIT_DIR=$SPLIT_DIR"
echo "CONFIG RUN=$RUN USE_TEXT=$USE_TEXT W_ALIGN=$W_ALIGN W_AGNOSTIC=$W_AGNOSTIC W_TOKEN=$W_TOKEN"
echo "CONFIG EPOCHS=$EPOCHS BATCHES_PER_EPOCH=$BATCHES_PER_EPOCH BATCH_SIZE=$BATCH_SIZE"
echo "CONFIG BUDGET_SAMPLES=$(( EPOCHS * BATCHES_PER_EPOCH * BATCH_SIZE ))"

# ------------------------------------------------------- 1/5 split stratified --
echo "STAGE=SPLIT AT=$(date --iso-8601=seconds)"
if [ -f "$SPLIT_DIR/seen.obj" ] && [ -f "$SPLIT_DIR/unseen.obj" ]; then
    echo "split đã có ở $SPLIT_DIR, bỏ qua"
else
    "$PYTHON" split/build_grasp_anything_pp.py \
        --data-dir "$DATA_DIR" --out-dir "$SPLIT_DIR" \
        --split-mode stratified --base-frac 0.7 --seed 0 \
        2>&1 | tee "$RESULTS_DIR/ga-pp-200k-split-v2.log"
fi
cp -f "$SPLIT_DIR/split_stats.json" "$RUN_RESULTS/" 2>/dev/null || true

# Hai tỉ lệ dưới đây phụ thuộc kích thước tập, mà kích thước lại phụ thuộc SCENES và cách
# chia split. Tính từ số sample thật thay vì hardcode:
#   * VAL_SPLIT quá cao -> tập validation vài trăm mẫu, sai số ±4pp, chọn checkpoint bằng nhiễu.
#   * EVAL_UNSEEN_SPLIT=0 -> đánh giá TOÀN BỘ unseen. Với split cũ unseen chỉ 956 mẫu (21 giây),
#     nhưng split stratified cho unseen ~30% dữ liệu, tức hàng trăm nghìn mẫu và hàng giờ.
read -r N_SEEN N_UNSEEN <<EOF
$("$PYTHON" -c "
import pickle, sys
d = sys.argv[1]
print(len(pickle.load(open(d + '/seen.obj', 'rb'))),
      len(pickle.load(open(d + '/unseen.obj', 'rb'))))
" "$SPLIT_DIR")
EOF
frac_for() {   # frac_for <tong> <so mau muon giu lai o duoi>
    awk "BEGIN{v = 1 - $2/$1; if (v < 0) v = 0; printf \"%.6f\", v}"
}
[ "$VAL_SPLIT" = "auto" ] && VAL_SPLIT=$(frac_for "$N_SEEN" "$VAL_TARGET")
[ "$EVAL_UNSEEN_SPLIT" = "auto" ] && EVAL_UNSEEN_SPLIT=$(frac_for "$N_UNSEEN" "$EVAL_UNSEEN_TARGET")
echo "SPLIT_SIZES seen=$N_SEEN unseen=$N_UNSEEN"
echo "CONFIG VAL_SPLIT=$VAL_SPLIT (~$(awk "BEGIN{printf \"%d\", $N_SEEN*(1-$VAL_SPLIT)}") mẫu holdout)"
echo "CONFIG EVAL_UNSEEN_SPLIT=$EVAL_UNSEEN_SPLIT (~$(awk "BEGIN{printf \"%d\", $N_UNSEEN*(1-$EVAL_UNSEEN_SPLIT)}") mẫu)"

# ---------------------------------------------------- 2/5 trọng số alignment --
echo "STAGE=ALIGN_WEIGHTS AT=$(date --iso-8601=seconds)"
if [ -f "$DATA_DIR/align_weights.npz" ]; then
    echo "align_weights.npz đã có, bỏ qua (xoá file đó nếu muốn tính lại)"
else
    # Thuần CPU, đọc ~19 GB mask, chạy một lần rồi cache.
    "$PYTHON" script/build_align_weights.py --data-dir "$DATA_DIR" \
        2>&1 | tee "$RESULTS_DIR/ga-pp-200k-align-weights.log"
fi

# ---------------------------------------------------------------- 3/5 train --
echo "STAGE=TRAIN AT=$(date --iso-8601=seconds) WORKERS=$NUM_WORKERS"
printf "%s\n" \
    "run=$RUN" "data_dir=$DATA_DIR" "split_dir=$SPLIT_DIR" "split_mode=stratified" \
    "network=$NETWORK" "init_from=$INIT_FROM" "epochs=$EPOCHS" \
    "batches_per_epoch=$BATCHES_PER_EPOCH" "batch_size=$BATCH_SIZE" \
    "input_size=$INPUT_SIZE" "val_split=$VAL_SPLIT" "val_augment=0" \
    "eval_unseen_split=$EVAL_UNSEEN_SPLIT" "n_seen=$N_SEEN" "n_unseen=$N_UNSEEN" \
    "lr=$LR" "lr_schedule=$LR_SCHEDULE" "use_text=$USE_TEXT" \
    "w_align=$W_ALIGN" "w_agnostic=$W_AGNOSTIC" "w_token=$W_TOKEN" \
    "mask_angle_loss=$MASK_ANGLE_LOSS" "single_part_weight=$SINGLE_PART_WEIGHT" \
    "warmup_epochs=$WARMUP_EPOCHS" "num_workers=$NUM_WORKERS" \
    "budget_samples=$(( EPOCHS * BATCHES_PER_EPOCH * BATCH_SIZE ))" \
    "code_revision=$(git rev-parse HEAD)" \
    > "$RUN_RESULTS/config.txt"

TRAIN_ARGS=(
    --dataset grasp-anything-pp
    --dataset-path "$DATA_DIR"
    --split-path "$SPLIT_DIR"
    --network "$NETWORK"
    --use-depth 0 --use-rgb 1 --seen 1
    --input-size "$INPUT_SIZE"
    --split "$VAL_SPLIT"
    --val-augment 0
    --epochs "$EPOCHS"
    --batches-per-epoch "$BATCHES_PER_EPOCH"
    --batch-size "$BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --lr "$LR" --lr-schedule "$LR_SCHEDULE"
    --use-text "$USE_TEXT"
    --w-align "$W_ALIGN" --w-agnostic "$W_AGNOSTIC" --w-token "$W_TOKEN"
    --mask-angle-loss "$MASK_ANGLE_LOSS"
    --single-part-weight "$SINGLE_PART_WEIGHT"
    --align-diag-samples "$ALIGN_DIAG_SAMPLES"
    --warmup-epochs "$WARMUP_EPOCHS"
    --logdir "$LOGDIR" --description "$DESC"
)
[ -n "$INIT_FROM" ] && [ -f "$INIT_FROM" ] && TRAIN_ARGS+=(--init-from "$INIT_FROM")

"$PYTHON" train_network.py "${TRAIN_ARGS[@]}" 2>&1 | tee "$RUN_RESULTS/train.log"

RUN_DIR=$(find "$LOGDIR" -maxdepth 1 -type d -name "*${DESC}*" | sort | tail -1)
test -n "$RUN_DIR" || { echo "Không thấy thư mục log khớp *$DESC*" >&2; exit 1; }
echo "RUN_DIR=$RUN_DIR"
cp -f "$RUN_DIR/checkpoints.jsonl" "$RUN_RESULTS/" 2>/dev/null || true

# --------------------------------------------------- 4/5 chọn checkpoint ----
# Tên file `epoch_67_iou_0.33` làm tròn 2 chữ số nên không phân biệt được 0,3313 với 0,3296.
# checkpoints.jsonl giữ đủ chữ số, và in luôn top-5 theo hai tiêu chí để thấy mức nhiễu.
echo "STAGE=PICK_CHECKPOINT AT=$(date --iso-8601=seconds)"
CHECKPOINT=$("$PYTHON" - "$RUN_DIR/checkpoints.jsonl" <<'PY'
import json, statistics, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
saved = [r for r in rows if r.get("path")]
if not saved:
    sys.exit("checkpoints.jsonl không có checkpoint nào được lưu")
iou = {r["epoch"]: r["iou"] for r in rows}
def smooth(e):
    w = [iou[x] for x in (e - 1, e, e + 1) if x in iou]
    return sum(w) / len(w)
tail = [r["iou"] for r in rows[-10:]]
print("  10 epoch cuối: mean=%.4f std=%.4f" % (
    statistics.mean(tail), statistics.pstdev(tail)), file=sys.stderr)
for label, key in (("iou", lambda r: r["iou"]), ("iou trung bình trượt 3", lambda r: smooth(r["epoch"]))):
    top = sorted(saved, key=key, reverse=True)[:5]
    print("  top-5 theo %s: %s" % (
        label, ", ".join("ep%d=%.4f" % (r["epoch"], key(r)) for r in top)), file=sys.stderr)
best = max(saved, key=lambda r: r["iou"])
print("  chọn: epoch %d iou=%.6f sha256=%s" % (
    best["epoch"], best["iou"], best.get("sha256", "?")[:16]), file=sys.stderr)
print(best["path"])
PY
)
echo "CHECKPOINT=$CHECKPOINT"
test -f "$CHECKPOINT" || { echo "Không thấy $CHECKPOINT" >&2; exit 1; }

# ----------------------------------------------------------------- 5/5 eval --
echo "STAGE=EVAL AT=$(date --iso-8601=seconds)"
run_eval() {   # run_eval <seen 1|0> <split> <logfile> [extra args...]
    local seen=$1 split=$2 log=$3; shift 3
    "$PYTHON" evaluate.py \
        --dataset grasp-anything-pp \
        --dataset-path "$DATA_DIR" --split-path "$SPLIT_DIR" \
        --network "$CHECKPOINT" \
        --iou-eval --use-depth 0 --use-rgb 1 --input-size "$INPUT_SIZE" \
        --seen "$seen" --split "$split" --num-workers "$NUM_WORKERS" "$@" 2>&1 \
    | tee "$RUN_RESULTS/$log"
}

# --align-eval bật thêm chẩn đoán A_T: align_iou, fg vs bg, argmax_is_part, argmax_is_punct.
# fg ≈ bg nghĩa là gate vô dụng dù loss có giảm; argmax_is_part thấp nghĩa là nhánh ngôn ngữ
# chưa bám vào part. Chỉ bật ở run có text.
EXTRA=(); [ "$USE_TEXT" = "1" ] && EXTRA=(--align-eval)

echo "-- seen (Base), đúng phần holdout dùng để chọn checkpoint --"
run_eval 1 "$VAL_SPLIT" eval_seen.log "${EXTRA[@]}"
echo "-- unseen (New), mẫu con ~$EVAL_UNSEEN_TARGET để eval không kéo dài hàng giờ --"
run_eval 0 "$EVAL_UNSEEN_SPLIT" eval_unseen.log "${EXTRA[@]}"

SEEN=$(awk '/IOU Results:/ {print $NF}' "$RUN_RESULTS/eval_seen.log" | tail -1)
UNSEEN=$(awk '/IOU Results:/ {print $NF}' "$RUN_RESULTS/eval_unseen.log" | tail -1)
test -n "$SEEN" && test -n "$UNSEEN"
echo "RESULT_SEEN=$SEEN RESULT_UNSEEN=$UNSEEN"
"$PYTHON" -c "
s,u=float('$SEEN'),float('$UNSEEN')
print('RESULT_H=%.6f' % (0.0 if s+u==0 else 2*s*u/(s+u)))"

# ---------------------------------------------------------------- export ----
echo "STAGE=EXPORT AT=$(date --iso-8601=seconds)"
EXPORT_ARGS=(
    --checkpoint "$CHECKPOINT"
    --dataset-path "$DATA_DIR" --split-path "$SPLIT_DIR"
    --out "$RUN_RESULTS" --n-samples 4 --input-size "$INPUT_SIZE"
    --seen-split "$VAL_SPLIT"          # hình seen lấy từ holdout, không phải từ dữ liệu train
    --seen-acc "$SEEN" --unseen-acc "$UNSEEN"
    --train-log "$RUN_RESULTS/train.log" --config "$RUN_RESULTS/config.txt"
    --tensorboard-dir "$RUN_DIR"
)
"$PYTHON" script/export_results.py "${EXPORT_ARGS[@]}" 2>&1 | tee "$RUN_RESULTS/export.log"

echo "STAGE=COMPLETE AT=$(date --iso-8601=seconds) ELAPSED_SECONDS=$SECONDS"
echo "RESULTS=$RUN_RESULTS"
