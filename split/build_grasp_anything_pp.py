"""Dựng split seen/unseen cho Grasp-Anything++ theo protocol của paper LGD.

Paper (§5.1 Setup): "We categorize LVIS labels from Section 3.3 to form labels for our
experiment. In particular, we select 70% of these labels by frequency for 'Base' and assign
the remaining 30% to 'New'." Tức là chia theo **category object**, không phải chia ngẫu nhiên
theo sample -- unseen nghĩa là category chưa từng thấy lúc train.

Hai chỗ khác với cách đọc đen của câu trên, vì cách đọc đen cho ra split không dùng được:

* `--split-mode balanced` (mặc định) chia category sao cho **số sample** đạt `--base-frac`,
  thay vì số category. Đọc đen -- lấy 70% category nhiều nhất cho Base -- trên phân bố lệch
  mạnh của GA++ chỉ để lại 956/882.214 sample (0,1%) cho New. Vẫn không có category nào dùng
  chung giữa hai tập, nên "unseen" giữ nguyên ý nghĩa. `--split-mode paper` khôi phục hành
  vi cũ.
* Scene chứa cả object Base lẫn New bị loại khỏi tập **seen** (`--drop-shared-from`): giữ
  nguyên là leak ở mức ảnh -- ảnh dùng để đánh giá đã được train trực tiếp, và ảnh train đã
  chứa pixel của category lẽ ra "chưa từng thấy".

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
                   help='Tỉ lệ đưa vào tập seen/Base. Ý nghĩa phụ thuộc --split-mode: '
                        '"balanced" tính theo số sample, "paper" tính theo số category.')
    p.add_argument('--split-mode', choices=('balanced', 'paper'), default='balanced',
                   help='Cách chia category. Xem assign_balanced(). Mặc định "balanced".')
    p.add_argument('--drop-shared-from', choices=('seen', 'unseen', 'none'), default='seen',
                   help='Scene chứa cả object Base lẫn New thì loại khỏi phía nào. "seen" '
                        '(mặc định) bỏ ảnh đó khỏi train: tập đánh giá giữ nguyên và category '
                        'New thật sự chưa từng xuất hiện lúc train. "unseen" bỏ khỏi tập đánh '
                        'giá. "none" giữ cả hai (có leak ở mức ảnh).')
    p.add_argument('--max-samples', type=int, default=0,
                   help='Giới hạn số sample mỗi tập (0 = không giới hạn). Dùng khi máy yếu.')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def assign_balanced(freq, base_frac):
    """
    Chia category thành Base/New sao cho *số sample* của Base xấp xỉ `base_frac`.

    Cách cũ (`--split-mode paper`) đọc protocol LGD theo nghĩa đen -- lấy 70% category
    *nhiều nhất* cho Base -- nhưng phân bố category của Grasp-Anything++ lệch cực mạnh, nên
    98 category hiếm nhất chỉ gom được 956/882.214 sample (0,1%). Tập New như vậy không đủ
    để đo bất cứ thứ gì: sai số chuẩn của một tỉ lệ trên 956 mẫu là ±1,3 điểm.

    Ở đây vẫn chia theo category (Base và New không bao giờ dùng chung category, nên "unseen"
    vẫn đúng nghĩa là category chưa từng thấy), nhưng duyệt category từ nhiều đến ít và đẩy
    mỗi category về phía đang *thiếu* nhiều nhất so với hạn ngạch sample của nó. Kết quả là
    cả hai tập đều có cả category phổ biến lẫn hiếm, và tỉ lệ sample bám sát `base_frac`.

    :param freq: collections.Counter {category: số sample}
    :param base_frac: tỉ lệ sample mong muốn cho Base
    :return: (set category Base, set category New, dict số sample thực tế)
    """
    total = sum(freq.values())
    target = {'base': base_frac * total, 'new': (1.0 - base_frac) * total}
    got = {'base': 0, 'new': 0}
    out = {'base': set(), 'new': set()}

    for cat, n in freq.most_common():
        # Hạn ngạch bằng 0 (base_frac 0 hoặc 1) -> coi như đã đầy, không bao giờ chọn.
        side = min(('base', 'new'),
                   key=lambda s: got[s] / target[s] if target[s] > 0 else float('inf'))
        out[side].add(cat)
        got[side] += n

    return out['base'], out['new'], got


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
    if args.split_mode == 'balanced':
        base, new, got = assign_balanced(freq, args.base_frac)
    else:
        ordered = [c for c, _ in freq.most_common()]
        n_base = int(round(len(ordered) * args.base_frac))
        base, new = set(ordered[:n_base]), set(ordered[n_base:])
        got = {'base': sum(freq[c] for c in base), 'new': sum(freq[c] for c in new)}
    total = got['base'] + got['new']
    print('{:,} category -> Base {:,} / New {:,}  (mode {})'.format(
        len(freq), len(base), len(new), args.split_mode))
    print('   sample: Base {:,} ({:.1%}) / New {:,} ({:.1%})'.format(
        got['base'], got['base'] / total, got['new'], got['new'] / total))

    seen = sorted(sid for sid, c in sample_cat.items() if c in base)
    unseen = sorted(sid for sid, c in sample_cat.items() if c in new)

    # Một ảnh chứa nhiều object khác category nên có thể rơi vào cả hai tập. Giữ nguyên là
    # leak ở mức ảnh: ảnh dùng để đánh giá đã được train trực tiếp, và ngược lại ảnh train đã
    # chứa pixel của category "chưa từng thấy".
    #
    # Mặc định loại khỏi *seen*: mất một ít dữ liệu train, đổi lại tập đánh giá giữ nguyên
    # kích thước và category New thật sự chưa xuất hiện lúc train -- đảm bảo mạnh hơn hẳn so
    # với việc cắt bớt tập đánh giá.
    shared = ({s.rsplit('_', 2)[0] for s in seen} & {s.rsplit('_', 2)[0] for s in unseen})
    print('{:,} scene xuất hiện ở cả seen lẫn unseen'.format(len(shared)))
    if shared and args.drop_shared_from != 'none':
        side = args.drop_shared_from
        ids = seen if side == 'seen' else unseen
        before = len(ids)
        kept = [s for s in ids if s.rsplit('_', 2)[0] not in shared]
        if side == 'seen':
            seen = kept
        else:
            unseen = kept
        print('   loại {:,} sample khỏi {} ({:.1%} của tập đó); --drop-shared-from để đổi'
              .format(before - len(kept), side, (before - len(kept)) / max(before, 1)))

    if args.max_samples:
        rng = random.Random(args.seed)
        seen = sorted(rng.sample(seen, min(args.max_samples, len(seen))))
        unseen = sorted(rng.sample(unseen, min(args.max_samples, len(unseen))))

    os.makedirs(args.out_dir, exist_ok=True)
    for name, ids in (('seen', seen), ('unseen', unseen)):
        with open(os.path.join(args.out_dir, name + '.obj'), 'wb') as f:
            pickle.dump(ids, f)
        print('{:<6} {:>9,} sample'.format(name, len(ids)))

    # Một tập đánh giá quá nhỏ thì mọi con số rút ra từ nó đều vô nghĩa -- nói thẳng ra ngay
    # lúc dựng split thay vì để phát hiện sau khi đã train xong.
    n_all = len(seen) + len(unseen)
    for name, ids in (('seen', seen), ('unseen', unseen)):
        if n_all and len(ids) / n_all < 0.05:
            se = (0.25 / max(len(ids), 1)) ** 0.5
            print('\nCẢNH BÁO: {} chỉ có {:,} sample ({:.2%} tổng số). Sai số chuẩn của một '
                  'tỉ lệ trên tập này là ±{:.1%} -- không đủ để so sánh model.'
                  .format(name, len(ids), len(ids) / n_all, se))
            if args.split_mode == 'paper':
                print('Thử --split-mode balanced để chia theo số sample thay vì số category.')

    print('\nGhi vào ' + args.out_dir)


if __name__ == '__main__':
    main()
