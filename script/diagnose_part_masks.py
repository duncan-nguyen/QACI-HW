"""Kiểm tra `part_mask` của Grasp-Anything++ có thật sự ở mức *part* hay chỉ ở mức *object*.

Lý do: `results/.../figures/parts_same_object.png` cho thấy bốn prompt khác nhau trên cùng
một quả táo -- "by its skin", "by its flesh", "by its seeds", "on its stem" -- có `part_mask`
**giống hệt nhau**, đều là toàn bộ quả táo. Nếu đó là hiện tượng phổ biến chứ không phải cá
biệt, thì `L_align(A_T, part_mask)` không thể dạy grounding mức part: tín hiệu giám sát không
chứa thông tin về part. Script này đo tỉ lệ đó trên toàn dataset.

Ba số cần đọc:

* `identical`  -- tỉ lệ object mà **mọi** part có mask trùng nhau (IoU ≥ --iou-identical).
                  Cao nghĩa là nhãn ở mức object, luận điểm part-level không giám sát được.
* `part≈union` -- tỉ lệ part mà mask của nó xấp xỉ hợp của mọi part cùng object. Cùng ý nghĩa,
                  đo trên từng part thay vì từng object.
* `fg_frac`    -- tỉ lệ diện tích foreground, để quy đổi giá trị `align_loss` quan sát được.

Chỉ dùng numpy, không cần torch/GPU:

    python script/diagnose_part_masks.py --data-dir data/grasp-anything-pp-200k --n-objects 2000
"""

import argparse
import collections
import glob
import json
import os
import pickle
import random
import statistics

import numpy as np

# Ảnh và part_mask của Grasp-Anything luôn là 416x416 (xem utils/data/grasp_anything_pp_data.py).
SOURCE_SIZE = 416


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data-dir", required=True, help="Thư mục GA++ (chứa part_mask/)")
    p.add_argument(
        "--n-objects",
        type=int,
        default=2000,
        help="Số object (có ≥2 part) lấy mẫu. 0 = tất cả.",
    )
    p.add_argument(
        "--iou-identical",
        type=float,
        default=0.95,
        help="Ngưỡng coi hai mask là trùng nhau.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="Ghi kết quả dạng JSON vào file này")
    p.add_argument(
        "--examples", type=int, default=5, help="Số ví dụ in kèm prompt để soi tay"
    )
    return p.parse_args()


def load_mask(path):
    """
    part_mask 416x416 uint8, chấp nhận cả bản đóng gói bit.

    Bản gốc: `GraspAnythingPPDataset._load_mask`. Nhân bản ở đây để script chạy được mà không
    cần import torch.
    """
    mask = np.load(path)
    if mask.ndim == 1:
        mask = np.unpackbits(mask)[: SOURCE_SIZE * SOURCE_SIZE].reshape(
            SOURCE_SIZE, SOURCE_SIZE
        )
    return mask > 0


def iou(a, b):
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    mask_dir = os.path.join(args.data_dir, "part_mask")
    if not os.path.isdir(mask_dir):
        raise SystemExit("Không thấy " + mask_dir)

    files = glob.glob(os.path.join(mask_dir, "*.npy"))
    if not files:
        raise SystemExit("Không thấy part_mask/*.npy trong " + args.data_dir)

    # Gom theo object "<scene>_<object>"; mỗi phần tử là một part.
    by_object = collections.defaultdict(list)
    for f in files:
        sid = os.path.splitext(os.path.basename(f))[0]
        by_object[sid.rsplit("_", 1)[0]].append(f)
    multi = {k: v for k, v in by_object.items() if len(v) >= 2}
    print(f"{len(files):,} part mask · {len(by_object):,} object · {len(multi):,} object có ≥2 part")
    if not multi:
        raise SystemExit("Không object nào có nhiều hơn một part -- không kiểm được gì.")

    keys = sorted(multi)
    if args.n_objects and args.n_objects < len(keys):
        keys = rng.sample(keys, args.n_objects)
    print(f"lấy mẫu {len(keys):,} object\n")

    n_identical = 0
    mean_ious, min_ious, part_vs_union, fg_fracs, n_parts = [], [], [], [], []
    examples = []

    for key in keys:
        paths = sorted(multi[key])
        masks = [load_mask(p) for p in paths]
        n_parts.append(len(masks))
        fg_fracs.extend(float(m.mean()) for m in masks)

        pairs = [
            iou(masks[i], masks[j])
            for i in range(len(masks))
            for j in range(i + 1, len(masks))
        ]
        mean_ious.append(statistics.mean(pairs))
        min_ious.append(min(pairs))
        if min(pairs) >= args.iou_identical:
            n_identical += 1
            if len(examples) < args.examples:
                examples.append((key, paths, min(pairs)))

        union = np.logical_or.reduce(masks)
        part_vs_union.extend(iou(m, union) for m in masks)

    def pct(values, q):
        return statistics.quantiles(values, n=100)[q - 1] if len(values) > 1 else values[0]

    frac_identical = n_identical / len(keys)
    frac_part_is_union = sum(v >= args.iou_identical for v in part_vs_union) / len(
        part_vs_union
    )

    print("=" * 72)
    print(f"identical   : {n_identical:,}/{len(keys):,} = {frac_identical:.1%} object có MỌI part trùng mask")
    print(f"              (ngưỡng IoU ≥ {args.iou_identical})")
    print(f"part≈union  : {frac_part_is_union:.1%} part có mask ≈ hợp của mọi part cùng object")
    print()
    print(f"IoU giữa các part cùng object: median {statistics.median(mean_ious):.3f} · "
          f"p10 {pct(mean_ious, 10):.3f} · p90 {pct(mean_ious, 90):.3f}")
    print(f"IoU nhỏ nhất trong mỗi object: median {statistics.median(min_ious):.3f}")
    print(f"số part mỗi object           : median {statistics.median(n_parts):.1f} · "
          f"max {max(n_parts)}")
    print(f"tỉ lệ diện tích foreground   : median {statistics.median(fg_fracs):.4f} · "
          f"p10 {pct(fg_fracs, 10):.4f} · p90 {pct(fg_fracs, 90):.4f}")
    print("=" * 72)

    if frac_identical > 0.5:
        print("\nKẾT LUẬN: part_mask chủ yếu ở mức OBJECT, không phải mức part.")
        print("L_align(A_T, part_mask) vì thế không thể dạy grounding mức part -- target không")
        print("chứa thông tin phân biệt part. Mọi cải tiến ở phía kiến trúc alignment đều bị")
        print("chặn trên bởi giới hạn này.")
    elif frac_identical > 0.15:
        print("\nKẾT LUẬN: một phần đáng kể part_mask ở mức object. Cần lọc bỏ những object đó")
        print("khỏi L_align, hoặc đánh trọng số theo mức phân biệt giữa các part.")
    else:
        print("\nKẾT LUẬN: part_mask phân biệt được part. Trường hợp quả táo là cá biệt;")
        print("nguyên nhân A_T không bám part nằm ở phía kiến trúc/đặc trưng thị giác.")

    # Ví dụ để soi tay, kèm prompt nếu có.
    prompt_dir = os.path.join(args.data_dir, "grasp_instructions")
    if examples and os.path.isdir(prompt_dir):
        print("\nVí dụ object có mọi part trùng mask:")
        for key, paths, v in examples:
            print(f"  {key}  (IoU nhỏ nhất {v:.3f})")
            for p in paths:
                sid = os.path.splitext(os.path.basename(p))[0]
                fp = os.path.join(prompt_dir, sid + ".pkl")
                prompt = "?"
                if os.path.isfile(fp):
                    with open(fp, "rb") as f:
                        prompt = pickle.load(f)
                    if not isinstance(prompt, str):
                        prompt = prompt[0]
                print(f'      {sid}: "{prompt}"')

    if args.out:
        with open(args.out, "w") as f:
            json.dump(
                {
                    "n_objects_sampled": len(keys),
                    "frac_objects_all_parts_identical": frac_identical,
                    "frac_parts_equal_union": frac_part_is_union,
                    "median_mean_iou": statistics.median(mean_ious),
                    "median_min_iou": statistics.median(min_ious),
                    "median_fg_frac": statistics.median(fg_fracs),
                    "iou_identical_threshold": args.iou_identical,
                },
                f,
                indent=2,
            )
        print("\nGhi " + args.out)


if __name__ == "__main__":
    main()
