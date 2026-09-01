"""Dựng split seen/unseen cho Grasp-Anything++ theo protocol của paper LGD.

Paper (§5.1 Setup): "We categorize LVIS labels from Section 3.3 to form labels for our
experiment. In particular, we select 70% of these labels by frequency for 'Base' and assign
the remaining 30% to 'New'." Tức là chia theo **category object**, không phải chia ngẫu nhiên
theo sample -- unseen nghĩa là category chưa từng thấy lúc train.

Câu "70% by frequency" mơ hồ, và cách đọc theo nghĩa đen (lấy 70% category **đầu bảng** tần
suất cho Base) tạo ra một tập New chỉ gồm cái đuôi hiếm nhất: trên subset 200k scene nó cho
881.258 / 956 sample, tức 99,89% / 0,11%, không đo được gì. `--split-mode` chọn cách đọc:

    stratified   (mặc định) 70% category, bốc ngẫu nhiên trong từng khối cùng bậc tần suất.
                 Giữ đúng "70% nhãn" mà vẫn cho tỉ lệ sample gần 70/30.
    sample-mass  Lấy category đầu bảng cho tới khi đủ 70% *sample*. Base ít category, khó nhất.
    label-count  Cách đọc theo nghĩa đen, giữ lại để tái lập kết quả cũ. Có cảnh báo.

Category của một sample "<scene>_<object>_<part>" lấy từ scene_description của base
Grasp-Anything: file <scene>.pkl chứa (caption, [object_0, object_1, ...]), index theo
<object>.

Chạy:
    python split/build_grasp_anything_pp.py --data-dir data/grasp-anything-pp \
        --scene-description data/grasp-anything-pp/scene_description
"""

import argparse
import collections
import glob
import json
import os
import pickle
import random
import statistics


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--data-dir", required=True, help="Thư mục GA++ (chứa grasp_label_positive/)"
    )
    p.add_argument(
        "--scene-description",
        default=None,
        help="Thư mục scene_description/ (mặc định: <data-dir>/scene_description)",
    )
    p.add_argument("--out-dir", default=os.path.join("split", "grasp-anything-pp"))
    p.add_argument(
        "--base-frac",
        type=float,
        default=0.7,
        help="Tỉ lệ đưa vào tập seen/Base. Đơn vị (category hay sample) tuỳ "
        "--split-mode.",
    )
    p.add_argument(
        "--split-mode",
        default="stratified",
        choices=["stratified", "sample-mass", "label-count"],
        help='Cách diễn giải "70%% of labels by frequency" -- xem docstring.',
    )
    p.add_argument(
        "--strata-size",
        type=int,
        default=10,
        help="Kích thước khối tần suất khi --split-mode stratified.",
    )
    p.add_argument(
        "--exclude-shared-scenes",
        action="store_true",
        help="Bỏ khỏi unseen mọi sample nằm trên scene cũng xuất hiện ở seen. "
        'Ảnh đó đã được nhìn thấy lúc train nên không còn là "unseen" thật.',
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Giới hạn số sample mỗi tập (0 = không giới hạn). Dùng khi máy yếu.",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_categories(scene_dir, scenes):
    """{scene: [tên object theo index]} -- bỏ qua scene thiếu file."""
    out = {}
    for scene in scenes:
        path = os.path.join(scene_dir, scene + ".pkl")
        if not os.path.isfile(path):
            continue
        with open(path, "rb") as f:
            desc = pickle.load(f)
        # scene_description là tuple (caption, [object, ...]).
        objects = desc[1] if isinstance(desc, (tuple, list)) and len(desc) > 1 else []
        out[scene] = [str(o).strip().lower() for o in objects]
    return out


def assign_categories(ordered, freq, mode, base_frac, strata_size, rng):
    """
    Chia danh sách category (đã xếp theo tần suất giảm dần) thành Base / New.

    :param ordered: list[str] category, tần suất giảm dần
    :param freq: Counter category -> số sample
    :return: (set base, set new)
    """
    if mode == "label-count":
        n_base = int(round(len(ordered) * base_frac))
        return set(ordered[:n_base]), set(ordered[n_base:])

    if mode == "sample-mass":
        target = base_frac * sum(freq.values())
        base, acc = set(), 0
        for c in ordered:
            if acc >= target:
                break
            base.add(c)
            acc += freq[c]
        return base, set(ordered) - base

    # stratified: trong mỗi khối `strata_size` category cùng bậc tần suất, bốc ngẫu nhiên
    # base_frac cho Base. Cả hai tập vì thế có cùng dạng phân phối tần suất.
    base = set()
    for i in range(0, len(ordered), strata_size):
        block = ordered[i : i + strata_size]
        k = int(round(len(block) * base_frac))
        base.update(rng.sample(block, k))
    return base, set(ordered) - base


def describe(name, ids, sample_cat, freq, total_samples):
    """In thống kê của một tập: số sample, tỉ lệ, và phân bố sample/category."""
    cats = sorted({sample_cat[s] for s in ids})
    counts = sorted((freq[c] for c in cats), reverse=True)
    pct = 100.0 * len(ids) / total_samples if total_samples else 0.0
    print(f"{name:<6} {len(ids):>9,} sample ({pct:5.2f}%)  {len(cats):>4,} category")
    if counts:
        print(
            f"       sample/category: min {counts[-1]:,} · median {int(statistics.median(counts)):,} · max {counts[0]:,}"
        )
    return {
        "samples": len(ids),
        "pct": pct,
        "categories": len(cats),
        "per_category_min": counts[-1] if counts else 0,
        "per_category_median": int(statistics.median(counts)) if counts else 0,
        "per_category_max": counts[0] if counts else 0,
    }


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    scene_dir = args.scene_description or os.path.join(
        args.data_dir, "scene_description"
    )
    if not os.path.isdir(scene_dir):
        raise SystemExit(
            f"Không thấy {scene_dir}. Cần scene_description.zip của repo base Grasp-Anything -- "
            "script/download_grasp_anything_pp.sh tải sẵn."
        )

    files = glob.glob(os.path.join(args.data_dir, "grasp_label_positive", "*.pt"))
    if not files:
        raise SystemExit("Không thấy grasp_label_positive/*.pt trong " + args.data_dir)
    sample_ids = sorted(os.path.splitext(os.path.basename(f))[0] for f in files)
    print(f"{len(sample_ids):,} sample")

    scenes = {sid.rsplit("_", 2)[0] for sid in sample_ids}
    print(f"{len(scenes):,} scene, đang đọc scene_description...")
    catalog = load_categories(scene_dir, scenes)
    print(f"{len(catalog):,} scene có mô tả")

    # sample -> category
    sample_cat, skipped = {}, 0
    for sid in sample_ids:
        scene, obj_idx = sid.rsplit("_", 2)[0], int(sid.rsplit("_", 2)[1])
        objects = catalog.get(scene)
        if not objects or obj_idx >= len(objects):
            skipped += 1
            continue
        sample_cat[sid] = objects[obj_idx]
    print(
        f"{len(sample_cat):,} sample có category, bỏ {skipped:,} (thiếu mô tả hoặc index vượt danh sách)"
    )

    freq = collections.Counter(sample_cat.values())
    ordered = [c for c, _ in freq.most_common()]
    base, new = assign_categories(
        ordered, freq, args.split_mode, args.base_frac, args.strata_size, rng
    )
    print(
        f"{len(ordered):,} category -> Base {len(base):,} / New {len(new):,}   (mode={args.split_mode})"
    )

    seen = sorted(sid for sid, c in sample_cat.items() if c in base)
    unseen = sorted(sid for sid, c in sample_cat.items() if c in new)

    # ---------------------------------------------------------- rò rỉ scene --
    # Một scene chứa nhiều object khác category nên có thể rơi vào cả hai tập. Với *target*
    # thì vô hại (M_union chỉ gom trong split đang train), nhưng *ảnh đầu vào* thì đã bị nhìn
    # thấy lúc train -- phần unseen đó không còn đo được khả năng khái quát hoá.
    seen_scenes = {s.rsplit("_", 2)[0] for s in seen}
    shared = seen_scenes & {s.rsplit("_", 2)[0] for s in unseen}
    leaked = [s for s in unseen if s.rsplit("_", 2)[0] in shared]
    print(f"\n{len(shared):,} scene xuất hiện ở cả seen lẫn unseen")
    print(
        f"{len(leaked):,}/{len(unseen):,} sample unseen ({len(leaked) / len(unseen) if unseen else 0.0:.1%}) nằm trên scene đã có trong seen"
    )
    if args.exclude_shared_scenes:
        unseen = [s for s in unseen if s.rsplit("_", 2)[0] not in shared]
        print(f"--exclude-shared-scenes: unseen còn {len(unseen):,} sample")

    if args.max_samples:
        seen = sorted(rng.sample(seen, min(args.max_samples, len(seen))))
        unseen = sorted(rng.sample(unseen, min(args.max_samples, len(unseen))))

    # -------------------------------------------------------------- ghi ra --
    total = len(seen) + len(unseen)
    print()
    stats = {
        "mode": args.split_mode,
        "base_frac": args.base_frac,
        "seed": args.seed,
        "categories_total": len(ordered),
        "categories_base": len(base),
        "categories_new": len(new),
        "shared_scenes": len(shared),
        "leaked_unseen_samples": len(leaked),
        "excluded_shared_scenes": bool(args.exclude_shared_scenes),
    }
    os.makedirs(args.out_dir, exist_ok=True)
    for name, ids in (("seen", seen), ("unseen", unseen)):
        stats[name] = describe(name, ids, sample_cat, freq, total)
        with open(os.path.join(args.out_dir, name + ".obj"), "wb") as f:
            pickle.dump(ids, f)

    if unseen and len(unseen) / total < 0.05:
        print(
            f"\nCẢNH BÁO: unseen chỉ chiếm {len(unseen) / total:.2%} số sample. Tập test nhỏ như vậy có sai số "
            'lấy mẫu rất lớn và không đại diện cho "category mới". Cân nhắc '
            "--split-mode stratified."
        )

    with open(os.path.join(args.out_dir, "split_stats.json"), "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print("\nGhi vào " + args.out_dir + " (kèm split_stats.json)")


if __name__ == "__main__":
    main()
