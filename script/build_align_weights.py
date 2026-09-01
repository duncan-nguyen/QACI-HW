"""Tính trước trọng số `d_i` cho `L_align` của Grasp-Anything++ (Update 2).

    d_i = 1 - IoU( part_mask_i , hợp part_mask của mọi part cùng object )

`d_i` đo đúng thứ cần đo: **biết part nào thì thu hẹp được vùng bao nhiêu**. Nếu mask của một
part bằng hợp của mọi part cùng object thì từ chỉ part không thu hẹp được gì -> d_i = 0.

Vì sao cần: `figures/parts_same_object.png` cho thấy bốn prompt trên cùng một quả táo --
"by its skin", "by its flesh", "by its seeds", "on its stem" -- có `part_mask` **giống hệt
nhau**, đều là toàn bộ quả táo. Với những sample như thế, `L_align` không chỉ vô dụng: nó ép
model *bất biến theo part*, vì bất kỳ `A_T` nào đặc thù theo part đều làm loss tăng. Nhân
`L_align` với d_i sẽ vô hiệu hoá đúng những sample đó và trả ngân sách gradient về nhánh grasp.

Và các prompt trong GA++ cho thấy đây là hỗn hợp chứ không phải toàn bộ: "clamping ends",
"webbed feet", "binding" là part thật; "skin", "texture", "fabric", "viscous liquid" thì
không. Vì vậy phải đánh trọng số theo từng sample, không tắt hẳn cũng không giảm đều.

Chỉ dùng numpy, không cần torch/GPU. Chạy một lần, kết quả cache lại:

    python script/build_align_weights.py --data-dir data/grasp-anything-pp-200k

Ghi ra `<data-dir>/align_weights.npz`; `GraspAnythingPPDataset` tự dò file này.
"""

import argparse
import collections
import glob
import os
import statistics
import time

import numpy as np

# Ảnh và part_mask của Grasp-Anything luôn là 416x416 (utils/data/grasp_anything_pp_data.py).
SOURCE_SIZE = 416
OUT_NAME = "align_weights.npz"


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data-dir", required=True, help="Thư mục GA++ (chứa part_mask/)")
    p.add_argument(
        "--out",
        default=None,
        help=f"Mặc định <data-dir>/{OUT_NAME} -- đúng chỗ dataset tự dò.",
    )
    p.add_argument(
        "--limit-objects",
        type=int,
        default=0,
        help="Chỉ xử lý N object đầu (0 = tất cả). Dùng để chạy thử cho nhanh.",
    )
    return p.parse_args()


def load_mask(path):
    """
    part_mask 416x416 uint8, chấp nhận cả bản đóng gói bit.

    Bản gốc: `GraspAnythingPPDataset._load_mask`. Nhân bản để script không cần torch.
    """
    mask = np.load(path)
    if mask.ndim == 1:
        mask = np.unpackbits(mask)[: SOURCE_SIZE * SOURCE_SIZE].reshape(
            SOURCE_SIZE, SOURCE_SIZE
        )
    return mask > 0


def main():
    args = parse_args()
    mask_dir = os.path.join(args.data_dir, "part_mask")
    if not os.path.isdir(mask_dir):
        raise SystemExit("Không thấy " + mask_dir)

    files = sorted(glob.glob(os.path.join(mask_dir, "*.npy")))
    if not files:
        raise SystemExit("Không thấy part_mask/*.npy trong " + args.data_dir)

    by_object = collections.defaultdict(list)
    for f in files:
        sid = os.path.splitext(os.path.basename(f))[0]
        by_object[sid.rsplit("_", 1)[0]].append((sid, f))

    keys = sorted(by_object)
    if args.limit_objects:
        keys = keys[: args.limit_objects]
    print(f"{len(files):,} part mask · {len(by_object):,} object" + (
        f" (xử lý {len(keys):,})" if args.limit_objects else ""))

    ids, d_values, n_parts_values = [], [], []
    multi_d = []
    n_single = 0
    t0 = time.time()

    for k, key in enumerate(keys):
        parts = sorted(by_object[key])
        if len(parts) < 2:
            # Object một part: hợp bằng chính nó nên d_i luôn 0 và không nói lên điều gì.
            # Ghi d=1 và n_parts=1; dataset xử lý riêng qua --single-part-weight.
            sid, _ = parts[0]
            ids.append(sid)
            d_values.append(1.0)
            n_parts_values.append(1)
            n_single += 1
            continue

        masks = [load_mask(f) for _, f in parts]
        union = np.logical_or.reduce(masks)
        u_sum = union.sum()
        for (sid, _), m in zip(parts, masks):
            inter = np.logical_and(m, union).sum()          # = m.sum(), m ⊆ union
            uni = np.logical_or(m, union).sum()             # = u_sum
            iou = float(inter / uni) if uni else 1.0
            d = 1.0 - iou
            ids.append(sid)
            d_values.append(d)
            n_parts_values.append(len(parts))
            multi_d.append(d)

        if (k + 1) % 20000 == 0:
            el = time.time() - t0
            print(f"  {k + 1:,}/{len(keys):,} object · {el:.0f}s · "
                  f"còn ~{el / (k + 1) * (len(keys) - k - 1):.0f}s")

    out = args.out or os.path.join(args.data_dir, OUT_NAME)
    np.savez_compressed(
        out,
        ids=np.array(ids),
        d=np.array(d_values, dtype=np.float32),
        n_parts=np.array(n_parts_values, dtype=np.int16),
    )

    print("\n" + "=" * 72)
    print(f"{len(ids):,} sample · {n_single:,} object chỉ có 1 part (d không xác định)")
    if multi_d:
        near_zero = sum(v < 0.05 for v in multi_d) / len(multi_d)
        print(f"trên object có ≥2 part ({len(multi_d):,} sample):")
        print(f"  mean d   = {statistics.mean(multi_d):.4f}   <-- CON SỐ QUYẾT ĐỊNH")
        print(f"  median d = {statistics.median(multi_d):.4f}")
        print(f"  d < 0.05 = {near_zero:.1%}  (mask bằng cả object, L_align vô hiệu)")
        print("=" * 72)
        m = statistics.mean(multi_d)
        if m < 0.1:
            print("\npart_mask ở mức OBJECT trên gần như toàn bộ dataset.")
            print("L_align không dạy được grounding mức part và đang dạy ngược lại; với")
            print("trọng số này nó sẽ tự teo về 0. Nguồn tín hiệu part-level duy nhất còn")
            print("lại là grasp_label_positive/ -- vốn đã theo từng part.")
        elif m < 0.4:
            print("\nHỗn hợp: một phần đáng kể part_mask ở mức object. Trọng số d_i sẽ lọc")
            print("đúng những sample đó, phần còn lại vẫn train bình thường.")
        else:
            print("\npart_mask phân biệt part tốt. Ca quả táo là cá biệt; nút thắt nằm ở")
            print("kiến trúc alignment chứ không phải ở nhãn -- Update 3 mới là chỗ đáng đầu tư.")
    print("\nGhi " + out)


if __name__ == "__main__":
    main()
