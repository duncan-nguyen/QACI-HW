#!/usr/bin/env bash
#
# Bốn arm ablation chạy song song trên bốn GPU -- KHÔNG phải một run nhanh gấp bốn.
#
# Vì sao: train_network.py không có DataParallel/DDP, nó chạy trên đúng một GPU
# (hardware/device.py trả về `cuda`, tức cuda:0). Thêm nữa GR-ConvNet chỉ 1,9M tham số --
# một H200 đã thừa sức, nghẽn nằm ở dataloader trên CPU. Nên cách dùng 4 GPU có ích nhất là
# chạy trọn bảng ablation của idea.md §6 cùng lúc: cùng wall-clock như một run, nhưng ra cả
# bốn dòng thay vì một.
#
#   notext : GR-ConvNet thuần, không ngôn ngữ
#   attn   : có A_T gate nhưng KHÔNG có L_align  -> đo riêng phần "text->spatial"
#   align  : thêm L_align(A_T, part_mask)
#   full   : thêm L_agnostic(Q_g, M_union)
#
# Ngân sách train của bốn arm giống hệt nhau -- điều kiện bắt buộc để bảng có nghĩa.
#
#   bash script/run_ablation_4gpu.sh
#   ARMS="notext full" GPUS="0 1" bash script/run_ablation_4gpu.sh

set -euo pipefail
cd "$(dirname "$0")/.."

ARMS=${ARMS:-"notext attn align full"}
GPUS=${GPUS:-"0 1 2 3"}
LOGDIR=${LOGDIR:-logs/}
DESCRIPTION=${DESCRIPTION:-ga-pp-abl}

# Bốn tiến trình train dùng chung CPU -> chia đều worker, chừa một core cho tiến trình chính.
NPROC=$(nproc)
N_ARMS=$(echo "$ARMS" | wc -w)
NUM_WORKERS=${NUM_WORKERS:-$(( NPROC / N_ARMS - 1 > 2 ? NPROC / N_ARMS - 1 : 2 ))}
export NUM_WORKERS

arm_flags() {
    case "$1" in
        notext) echo "USE_TEXT=0 W_ALIGN=0 W_AGNOSTIC=0" ;;
        attn)   echo "USE_TEXT=1 W_ALIGN=0 W_AGNOSTIC=0" ;;
        align)  echo "USE_TEXT=1 W_ALIGN=${W_ALIGN:-0.3} W_AGNOSTIC=0" ;;
        full)   echo "USE_TEXT=1 W_ALIGN=${W_ALIGN:-0.3} W_AGNOSTIC=${W_AGNOSTIC:-0.3}" ;;
        *) echo "arm không hợp lệ: $1" >&2; exit 1 ;;
    esac
}

echo "== 1/3  chuẩn bị dữ liệu (một lần, trước khi fan-out) =="
# Bốn tiến trình cùng trích dữ liệu sẽ giẫm lên nhau, nên làm xong hẳn ở đây.
SKIP_TRAIN=1 SKIP_EVAL=1 bash script/run_paper_setting.sh

echo
echo "== 2/3  chạy $N_ARMS arm trên GPU [$GPUS], mỗi arm $NUM_WORKERS worker =="
# Log của từng arm để riêng, tránh nằm lẫn với thư mục run mà train_network.py tạo.
ARM_LOGS="$LOGDIR/ablation"
mkdir -p "$ARM_LOGS"
pids=()
names=()
i=0
for arm in $ARMS; do
    gpu=$(echo "$GPUS" | cut -d' ' -f$((i + 1)))
    log="$ARM_LOGS/${arm}.log"
    echo "  GPU $gpu -> $arm  ($log)"
    env $(arm_flags "$arm") \
        CUDA_VISIBLE_DEVICES="$gpu" \
        DESCRIPTION="${DESCRIPTION}-${arm}" \
        SKIP_BUILD=1 SKIP_SPLIT=1 \
        bash script/run_paper_setting.sh > "$log" 2>&1 &
    pids+=($!)
    names+=("$arm")
    i=$((i + 1))
done

fail=0
for idx in "${!pids[@]}"; do
    if wait "${pids[$idx]}"; then
        echo "  xong: ${names[$idx]}"
    else
        echo "  LỖI: ${names[$idx]} -- xem $ARM_LOGS/${names[$idx]}.log" >&2
        fail=1
    fi
done

echo
echo "== 3/3  bảng ablation =="
printf "%-10s %-34s %7s %7s %7s\n" "arm" "cấu hình" "Seen" "Unseen" "H"
for arm in $ARMS; do
    log="$ARM_LOGS/${arm}.log"
    line=$(grep -h '^RESULT' "$log" 2>/dev/null | tail -1 || true)
    if [ -z "$line" ]; then
        printf "%-10s %-34s %7s %7s %7s\n" "$arm" "$(arm_flags "$arm")" "-" "-" "-"
    else
        printf "%-10s %-34s %7.3f %7.3f %7.3f\n" "$arm" "$(arm_flags "$arm")" \
            "$(echo "$line" | cut -f2)" "$(echo "$line" | cut -f3)" "$(echo "$line" | cut -f4)"
    fi
done
cat <<'EOF'

Mốc tham chiếu (Table 2 của paper, GA++ đầy đủ, split gốc theo nhãn LVIS):
  GR-ConvNet + CLIP   0.37 / 0.18 / 0.24
  CLIP-Fusion         0.40 / 0.29 / 0.33
  LGD                 0.48 / 0.42 / 0.45
EOF
exit "$fail"
