"""Dựng split seen/unseen cho Grasp-Anything++ theo protocol của paper LGD.

Paper (§5.1 Setup): "We categorize LVIS labels from Section 3.3 to form labels for our
experiment. In particular, we select 70% of these labels by frequency for 'Base' and assign
the remaining 30% to 'New'." Tức là chia theo **category object**, không phải chia ngẫu nhiên
theo sample -- unseen nghĩa là category chưa từng thấy lúc train.

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
import os
import pickle
import random


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-dir', required=True,
                   help='Thư mục GA++ (chứa grasp_label_positive/)')
    p.add_argument('--scene-description', default=None,
                   help='Thư mục scene_description/ (mặc định: <data-dir>/scene_description)')
    p.add_argument('--out-dir', default=os.path.join('split', 'grasp-anything-pp'))
    p.add_argument('--base-frac', type=float, default=0.7,
                   help='Tỉ lệ category (theo tần suất giảm dần) đưa vào tập seen/Base')
    p.add_argument('--max-samples', type=int, default=0,
                   help='Giới hạn số sample mỗi tập (0 = không giới hạn). Dùng khi máy yếu.')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def load_categories(scene_dir, scenes):
    """{scene: [tên object theo index]} -- bỏ qua scene thiếu file."""
    out = {}
    for scene in scenes:
        path = os.path.join(scene_dir, scene + '.pkl')
        if not os.path.isfile(path):
            continue
        with open(path, 'rb') as f:
            desc = pickle.load(f)
        # scene_description là tuple (caption, [object, ...]).
        objects = desc[1] if isinstance(desc, (tuple, list)) and len(desc) > 1 else []
        out[scene] = [str(o).strip().lower() for o in objects]
    return out


def main():
    args = parse_args()
    scene_dir = args.scene_description or os.path.join(args.data_dir, 'scene_description')
    if not os.path.isdir(scene_dir):
        raise SystemExit(
            'Không thấy {}. Cần scene_description.zip của repo base Grasp-Anything -- '
            'script/download_grasp_anything_pp.sh tải sẵn.'.format(scene_dir))

    files = glob.glob(os.path.join(args.data_dir, 'grasp_label_positive', '*.pt'))
    if not files:
        raise SystemExit('Không thấy grasp_label_positive/*.pt trong ' + args.data_dir)
    sample_ids = sorted(os.path.splitext(os.path.basename(f))[0] for f in files)
    print('{:,} sample'.format(len(sample_ids)))

    scenes = {sid.rsplit('_', 2)[0] for sid in sample_ids}
    print('{:,} scene, đang đọc scene_description...'.format(len(scenes)))
    catalog = load_categories(scene_dir, scenes)
    print('{:,} scene có mô tả'.format(len(catalog)))

    # sample -> category
    sample_cat, skipped = {}, 0
    for sid in sample_ids:
        scene, obj_idx = sid.rsplit('_', 2)[0], int(sid.rsplit('_', 2)[1])
        objects = catalog.get(scene)
        if not objects or obj_idx >= len(objects):
            skipped += 1
            continue
        sample_cat[sid] = objects[obj_idx]
    print('{:,} sample có category, bỏ {:,} (thiếu mô tả hoặc index vượt danh sách)'.format(
        len(sample_cat), skipped))

    freq = collections.Counter(sample_cat.values())
    ordered = [c for c, _ in freq.most_common()]
    n_base = int(round(len(ordered) * args.base_frac))
    base, new = set(ordered[:n_base]), set(ordered[n_base:])
    print('{:,} category -> Base {:,} / New {:,}'.format(len(ordered), len(base), len(new)))

    seen = sorted(sid for sid, c in sample_cat.items() if c in base)
    unseen = sorted(sid for sid, c in sample_cat.items() if c in new)

    if args.max_samples:
        rng = random.Random(args.seed)
        seen = sorted(rng.sample(seen, min(args.max_samples, len(seen))))
        unseen = sorted(rng.sample(unseen, min(args.max_samples, len(unseen))))

    os.makedirs(args.out_dir, exist_ok=True)
    for name, ids in (('seen', seen), ('unseen', unseen)):
        with open(os.path.join(args.out_dir, name + '.obj'), 'wb') as f:
            pickle.dump(ids, f)
        print('{:<6} {:>9,} sample'.format(name, len(ids)))

    # Scene nằm ở cả hai tập là chuyện bình thường (một ảnh có nhiều object khác category),
    # nhưng phải biết con số: nó là lý do M_union chỉ được gom trong split đang train.
    shared = {s.rsplit('_', 2)[0] for s in seen} & {s.rsplit('_', 2)[0] for s in unseen}
    print('{:,} scene xuất hiện ở cả seen lẫn unseen'.format(len(shared)))
    print('\nGhi vào ' + args.out_dir)


if __name__ == '__main__':
    main()
