"""Kiểm tra dữ liệu GA++ **trước** khi train -- chạy một lần, đọc kỹ, rồi mới tốn GPU.

Ba thứ có thể làm hỏng cả thí nghiệm mà không hề báo lỗi:

1. `part_mask` thật ra ở mức *object*: mọi part của cùng một object có mask giống hệt nhau.
   Khi đó `L_align` không chứa thông tin phân biệt part, và mọi cải tiến ở phía kiến trúc
   alignment đều bị chặn trên. Ngưỡng cảnh báo: phần lớn cặp part cùng object có IoU > 0.9.
2. Foreground quá nhỏ (hoặc quá lớn): quyết định việc Dice có cần thiết không, và giá trị
   `align_bce` quan sát được nên so với mốc nào ("đoán toàn 0" cho BCE bằng bao nhiêu).
3. Prompt còn lại bao nhiêu token sau khi loại SOT/EOT/pad/dấu câu -- đó chính là số candidate
   mà softmax theo token phải chọn giữa. Nếu trung bình chỉ 2-3 token thì entropy tối đa cũng
   chỉ ~1.0, đừng kỳ vọng attention "tập trung" mang nhiều ý nghĩa.

    python script/check_dataset.py --data-dir data/grasp-anything-pp-200k \\
        --split-path split/grasp-anything-pp --out results/dataset-check

Báo cáo sâu hơn về riêng câu hỏi (1), kèm ví dụ để soi tay:
`python script/diagnose_part_masks.py --data-dir <...>`.
"""
import argparse
import collections
import glob
import json
import os
import random
import statistics
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from script.diagnose_part_masks import iou, load_mask  # noqa: E402
from utils.data import get_dataset  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--split-path", default=None)
    p.add_argument("--seen", type=int, default=1)
    p.add_argument("--input-size", type=int, default=224)
    p.add_argument("--n-objects", type=int, default=1000,
                   help="Số object (có ≥2 part) lấy mẫu cho thống kê mask")
    p.add_argument("--n-prompts", type=int, default=2000,
                   help="Số prompt lấy mẫu để đếm token")
    p.add_argument("--n-grid", type=int, default=24, help="Số ô của lưới ảnh kiểm tra bằng mắt")
    p.add_argument("--no-clip", action="store_true",
                   help="Bỏ phần đếm token (khỏi phải nạp CLIP)")
    p.add_argument("--out", default=None, help="Thư mục ghi JSON + PNG")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def mask_report(data_dir, n_objects, seed, threshold=0.9):
    """Foreground, IoU giữa các part cùng object, tỉ lệ mask trùng nhau."""
    files = glob.glob(os.path.join(data_dir, "part_mask", "*.npy"))
    if not files:
        raise SystemExit(f"Không thấy part_mask/*.npy trong {data_dir}")

    by_object = collections.defaultdict(list)
    for f in files:
        sid = os.path.splitext(os.path.basename(f))[0]
        by_object[sid.rsplit("_", 1)[0]].append(f)
    multi = {k: v for k, v in by_object.items() if len(v) >= 2}

    keys = sorted(multi)
    rng = random.Random(seed)
    if n_objects and n_objects < len(keys):
        keys = rng.sample(keys, n_objects)

    fg, pair_ious, identical, n_pairs = [], [], 0, 0
    for key in keys:
        masks = [load_mask(p) for p in sorted(multi[key])]
        fg.extend(float(m.mean()) for m in masks)
        pairs = [iou(masks[i], masks[j])
                 for i in range(len(masks)) for j in range(i + 1, len(masks))]
        pair_ious.extend(pairs)
        n_pairs += len(pairs)
        identical += int(min(pairs) >= threshold)

    # Với subset nhỏ có thể không object nào có ≥2 part -- vẫn báo được foreground.
    if not fg:
        fg = [float(load_mask(f).mean()) for f in files[:n_objects or len(files)]]

    return {
        "n_masks": len(files),
        "n_objects": len(by_object),
        "n_objects_multi_part": len(multi),
        "n_objects_sampled": len(keys),
        "fg_frac_median": statistics.median(fg),
        "fg_frac_mean": statistics.mean(fg),
        "fg_frac_min": min(fg),
        "fg_frac_max": max(fg),
        "pair_iou_median": statistics.median(pair_ious) if pair_ious else None,
        "pair_iou_mean": statistics.mean(pair_ious) if pair_ious else None,
        "frac_pairs_above_0.9": (sum(v > 0.9 for v in pair_ious) / n_pairs) if n_pairs else None,
        "frac_objects_all_parts_identical": (identical / len(keys)) if keys else None,
    }


def token_report(dataset, n_prompts, seed):
    """Số token nội dung mỗi prompt (sau khi bỏ SOT/EOT/pad/dấu câu) + token hay gặp."""
    from inference.models.grconvnet3_align import CLIPTextEncoder

    rng = random.Random(seed)
    n = len(dataset.grasp_files)
    indices = rng.sample(range(n), min(n_prompts, n))
    prompts = [dataset.get_prompt(i) for i in indices]

    encoder = CLIPTextEncoder()
    counts, tally = [], collections.Counter()
    for start in range(0, len(prompts), 64):
        batch = prompts[start:start + 64]
        _, mask = encoder(batch)
        strings = encoder.token_strings(batch)
        for b in range(len(batch)):
            keep = mask[b].nonzero(as_tuple=True)[0].tolist()
            counts.append(len(keep))
            tally.update(strings[b][j].replace("</w>", "") for j in keep)

    return {
        "n_prompts": len(prompts),
        "tokens_per_prompt_mean": statistics.mean(counts),
        "tokens_per_prompt_median": statistics.median(counts),
        "tokens_per_prompt_min": min(counts),
        "tokens_per_prompt_max": max(counts),
        "n_empty_prompts": sum(c == 0 for c in counts),
        "distinct_tokens": len(tally),
        "most_common": tally.most_common(15),
        # log(L): mốc trên của token/attention_entropy trong TensorBoard.
        "max_entropy": float(np.log(max(1, statistics.mean(counts)))),
    }, prompts


def save_grid(dataset, n_grid, path, seed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from utils.visualisation.alignment import plot_dataset_grid

    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset.grasp_files)), min(n_grid, len(dataset.grasp_files)))
    fig = plot_dataset_grid(dataset, sorted(indices))
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    args = parse_args()
    kwargs = dict(output_size=args.input_size, include_depth=False, include_rgb=True,
                  seen=bool(args.seen))
    if args.split_path:
        kwargs["split_path"] = args.split_path
    dataset = get_dataset("grasp-anything-pp")(args.data_dir, **kwargs)

    print(f"split {'seen' if args.seen else 'unseen'}: {len(dataset.grasp_files):,} sample · "
          f"{len(dataset._files_by_object):,} object\n")

    print("== 1. part_mask ==")
    masks = mask_report(args.data_dir, args.n_objects, args.seed)
    for k, v in masks.items():
        print(f"  {k:34} {v if not isinstance(v, float) else f'{v:.4f}'}")

    identical = masks["frac_objects_all_parts_identical"]
    above = masks["frac_pairs_above_0.9"]
    if above is not None and above > 0.5:
        print("\n  CẢNH BÁO: hơn một nửa cặp part cùng object có IoU > 0.9 -- supervision thực")
        print("  tế gần mức OBJECT hơn mức part. L_align không dạy được grounding part-level,")
        print("  và Δ_prompt trong đối chứng sẽ nhỏ dù model không có lỗi gì.")
    elif identical is not None and identical > 0.15:
        print("\n  Lưu ý: một phần đáng kể object có mọi part trùng mask; cân nhắc lọc chúng")
        print("  khỏi L_align. Chi tiết + ví dụ: script/diagnose_part_masks.py")
    # BCE của nghiệm tầm thường "đoán toàn 0" -- mốc để biết align_bce quan sát được có nghĩa gì.
    fg = masks["fg_frac_median"]
    print(f"\n  Mốc tham chiếu: foreground {fg:.2%} -> 'đoán toàn 0' cho BCE ≈ {-np.log(1 - fg):.4f}"
          if fg < 1 else "")

    tokens = None
    if not args.no_clip:
        print("\n== 2. prompt ==")
        tokens, _ = token_report(dataset, args.n_prompts, args.seed)
        for k, v in tokens.items():
            if k == "most_common":
                print("  token hay gặp nhất:")
                print("    " + ", ".join(f"{t} {c}" for t, c in v))
            else:
                print(f"  {k:34} {v if not isinstance(v, float) else f'{v:.3f}'}")
        if tokens["n_empty_prompts"]:
            print(f"\n  CẢNH BÁO: {tokens['n_empty_prompts']} prompt không còn token nào sau khi")
            print("  lọc -- model sẽ nhận A_T = 0 cho chúng.")

    print("\n== 3. lưới ảnh ==")
    result = {"split": "seen" if args.seen else "unseen", "part_mask": masks, "prompt": tokens}
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        grid = save_grid(dataset, args.n_grid, os.path.join(args.out, "dataset_grid.png"),
                         args.seed)
        print(f"  {grid}")
        path = os.path.join(args.out, "dataset_check.json")
        with open(path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  {path}")
    else:
        print("  (bỏ qua -- truyền --out để ghi hình)")


if __name__ == "__main__":
    main()
