#!/usr/bin/env bash
#
# Bảng ablation của idea.md §6 -- bốn arm chạy **tuần tự trên một GPU**.
#
# Tuần tự chứ không song song: train_network.py không có DataParallel/DDP nên mỗi run dùng
# đúng một GPU, và chỗ nghẽn thật là dataloader trên CPU. Nhét bốn tiến trình vào cùng một GPU
# chỉ khiến chúng giành nhau CPU và làm mỗi arm chậm đi gần bốn lần -- tổng thời gian như nhau,
# nhưng không có kết quả nào xong sớm để nhìn.
#
# Bốn arm, mỗi dòng bật thêm đúng MỘT thành phần so với dòng trên, nên chênh lệch giữa hai
# dòng liền nhau quy được cho thành phần đó:
#
#   notext  : GR-ConvNet thuần, không ngôn ngữ            (use_text=0)
#   hardmax : V1 -- max theo token + gate F*(1+A_T)       (align_mode=hard, fusion=gate)
#   soft    : softmax theo token + residual fusion        (align_mode=soft, fusion=residual)
#   full    : V2 đầy đủ, thêm region text R_T             (+ region_text=1)
#
# Hai arm phụ, chạy khi muốn tách riêng một biến (không nằm trong bảng §6):
#   noalign : có A_T nhưng KHÔNG có L_align -> đo riêng phần supervise mask
#   conv4   : như full nhưng tính A_T ở 113x113 thay vì 56x56
#
# Ngân sách train của mọi arm giống hệt nhau (EPOCHS x BATCHES_PER_EPOCH x BATCH_SIZE) -- điều
# kiện bắt buộc để bảng có nghĩa. Script in con số đó ra và ghi vào config.txt của từng arm.
#
#   bash script/run_ablation.sh                      # 4 arm, tuần tự, GPU 0
#   GPU=1 bash script/run_ablation.sh                # chọn GPU khác
#   ARMS="soft full" bash script/run_ablation.sh     # chạy lại vài arm
#   RESUME=1 bash script/run_ablation.sh             # bỏ qua arm đã có kết quả
#
# Đứt giữa chừng thì chạy lại với RESUME=1: arm nào đã có dòng RESULT trong log sẽ được bỏ qua.

set -euo pipefail
cd "$(dirname "$0")/.."

ARMS=${ARMS:-"notext soft full noalign conv4"}
GPU=${GPU:-0}
LOGDIR=${LOGDIR:-logs/}
DESCRIPTION=${DESCRIPTION:-ga-pp-abl}
RESUME=${RESUME:-0}

# Ngân sách train, giống hệt nhau giữa các arm. Đổi ở đây thì đổi cho cả bảng.
EPOCHS=${EPOCHS:-50}
BATCHES_PER_EPOCH=${BATCHES_PER_EPOCH:-2000}
BATCH_SIZE=${BATCH_SIZE:-64}
SCENES=${SCENES:-100000}
export EPOCHS BATCHES_PER_EPOCH BATCH_SIZE SCENES

# Một tiến trình train tại một thời điểm -> dùng được toàn bộ CPU cho dataloader.
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
        *) echo "arm không hợp lệ: $1" >&2; exit 1 ;;
    esac
}

N_ARMS=$(echo "$ARMS" | wc -w)
BUDGET=$(( EPOCHS * BATCHES_PER_EPOCH * BATCH_SIZE ))
ARM_LOGS="$LOGDIR/ablation"
mkdir -p "$ARM_LOGS"

cat <<EOF
== kế hoạch ==
  arm              : $ARMS  ($N_ARMS arm, chạy tuần tự)
  GPU              : $GPU
  scene            : $SCENES
  ngân sách mỗi arm: $EPOCHS x $BATCHES_PER_EPOCH x $BATCH_SIZE = $BUDGET lượt sample
  workers          : $NUM_WORKERS (máy có $NPROC core)
  log              : $ARM_LOGS/<arm>.log

Tổng thời gian ≈ $N_ARMS lần một run đơn. Theo dõi bằng: tail -f $ARM_LOGS/<arm>.log
EOF
echo

echo "== 1/3  chuẩn bị dữ liệu + kiểm dữ liệu (một lần, trước khi vào vòng arm) =="
# Làm xong hẳn ở đây để không arm nào phải tải/trích lại. Bước kiểm dữ liệu
# (script/check_dataset.py) cũng chỉ cần chạy một lần -- các arm sau đó dùng SKIP_CHECK=1.
GPU="$GPU" SKIP_TRAIN=1 SKIP_EVAL=1 bash script/run_paper_setting.sh

echo
echo "== 2/3  chạy $N_ARMS arm =="
fail=0
i=0
for arm in $ARMS; do
    i=$((i + 1))
    log="$ARM_LOGS/${arm}.log"
    if [ "$RESUME" = "1" ] && grep -q '^RESULT' "$log" 2>/dev/null; then
        echo "  [$i/$N_ARMS] bỏ qua $arm -- đã có kết quả trong $log (RESUME=1)"
        continue
    fi
    echo "  [$i/$N_ARMS] $arm  ($(date +%H:%M:%S))  -> $log"
    if env $(arm_flags "$arm") \
        GPU="$GPU" \
        DESCRIPTION="${DESCRIPTION}-${arm}" \
        SKIP_BUILD=1 SKIP_SPLIT=1 SKIP_CHECK=1 \
        bash script/run_paper_setting.sh > "$log" 2>&1
    then
        echo "        xong  ($(date +%H:%M:%S))  $(grep -h '^RESULT' "$log" | tail -1 || true)"
    else
        # Một arm hỏng không nên làm mất ba arm còn lại -- ghi nhận rồi chạy tiếp.
        echo "        LỖI -- xem $log" >&2
        tail -5 "$log" >&2 || true
        fail=1
    fi
done

echo
echo "== 3/3  bảng ablation =="
echo "ngân sách mỗi arm: $EPOCHS x $BATCHES_PER_EPOCH x $BATCH_SIZE = $BUDGET lượt sample · $SCENES scene"
echo
printf "%-10s %-62s %7s %7s %7s\n" "arm" "cấu hình" "Seen" "Unseen" "H"
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

Mốc tham chiếu (Table 2 của paper, GA++ đầy đủ 4,41M sample, 100 epoch, split gốc theo nhãn
LVIS). Ta chạy 10% dữ liệu và 50 epoch nên đây là mốc để đối chiếu, không phải để so hơn thua:
  GR-ConvNet + CLIP   0.37 / 0.18 / 0.24
  CLIP-Fusion         0.40 / 0.29 / 0.33
  LGD                 0.48 / 0.42 / 0.45

Sau bảng này, con số cần xem tiếp nằm ở results/<run>/audit_text_reliance.log:
Δ_shuffled ≈ 0 nghĩa là arm đó không thật sự dùng prompt, dù Seen/Unseen có cao.
EOF
exit "$fail"
