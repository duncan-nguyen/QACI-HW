"""Build the seen/unseen split for Grasp-Anything++ following the LGD paper's protocol.

The paper (Sec. 5.1 Setup): "We categorize LVIS labels from Section 3.3 to form labels for our
experiment. In particular, we select 70% of these labels by frequency for 'Base' and assign
the remaining 30% to 'New'." That is, the split is by **object category**, not a random split
over samples -- unseen means a category never encountered during training.

Two departures from the literal reading of that sentence, because the literal reading produces
an unusable split:

* `--split-mode balanced` (the default) assigns categories so that the **sample count** hits
  `--base-frac`, rather than the category count. The literal reading -- the 70% most frequent
  categories go to Base -- leaves only 956/882,214 samples (0.1%) for New on GA++'s heavily
  skewed distribution. No category is ever shared between the two sets, so "unseen" keeps its
  meaning. `--split-mode paper` restores the literal behaviour.
* Scenes containing both Base and New objects are dropped from the **seen** set
  (`--drop-shared-from`): keeping them is an image-level leak -- the evaluation image would
  have been trained on directly, and the training image would already contain pixels of a
  supposedly "never seen" category.

The category of a sample "<scene>_<object>_<part>" comes from the base Grasp-Anything
scene_description: the file <scene>.pkl holds (caption, [object_0, object_1, ...]), indexed by
<object>.

Run:
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
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--data-dir",
        required=True,
        help="GA++ directory (holding grasp_label_positive/)",
    )
    p.add_argument(
        "--scene-description",
        default=None,
        help="scene_description/ directory (default: <data-dir>/scene_description)",
    )
    p.add_argument("--out-dir", default=os.path.join("split", "grasp-anything-pp"))
    p.add_argument(
        "--base-frac",
        type=float,
        default=0.7,
        help="Fraction assigned to the seen/Base set. Its meaning depends on "
        '--split-mode: "balanced" counts samples, "paper" counts categories.',
    )
    p.add_argument(
        "--split-mode",
        choices=("balanced", "paper"),
        default="balanced",
        help='How categories are assigned. See assign_balanced(). Default "balanced".',
    )
    p.add_argument(
        "--drop-shared-from",
        choices=("seen", "unseen", "none"),
        default="seen",
        help="Which side loses scenes that contain both Base and New objects. "
        '"seen" (default) drops those images from training: the evaluation set '
        "keeps its size and New categories genuinely never appear during "
        'training. "unseen" drops them from the evaluation set. "none" keeps '
        "both (an image-level leak).",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Cap on samples per set (0 = no cap). Useful on small machines.",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def assign_balanced(freq, base_frac):
    """
    Assign categories to Base/New so that Base's *sample count* approximates `base_frac`.

    The literal reading of the LGD protocol (`--split-mode paper`) -- the 70% *most frequent*
    categories go to Base -- runs into Grasp-Anything++'s extremely skewed category
    distribution: the 98 rarest categories together hold only 956/882,214 samples (0.1%). Such
    a New set cannot measure anything: the standard error of a proportion over 956 samples is
    +/-1.3 points.

    This still splits by category (Base and New never share one, so "unseen" still means a
    never-seen category), but walks categories from most to least frequent and pushes each one
    to whichever side is furthest *below* its sample quota. Both sets then contain common and
    rare categories alike, and the sample ratio tracks `base_frac`.

    :param freq: collections.Counter {category: sample count}
    :param base_frac: desired sample fraction for Base
    :return: (set of Base categories, set of New categories, dict of actual sample counts)
    """
    total = sum(freq.values())
    target = {"base": base_frac * total, "new": (1.0 - base_frac) * total}
    got = {"base": 0, "new": 0}
    out = {"base": set(), "new": set()}

    for cat, n in freq.most_common():
        # A zero quota (base_frac 0 or 1) counts as full and is never selected.
        side = min(
            ("base", "new"),
            key=lambda s: got[s] / target[s] if target[s] > 0 else float("inf"),
        )
        out[side].add(cat)
        got[side] += n

    return out["base"], out["new"], got


def load_categories(scene_dir, scenes):
    """{scene: [object names by index]} -- scenes with a missing file are skipped."""
    out = {}
    for scene in scenes:
        path = os.path.join(scene_dir, scene + ".pkl")
        if not os.path.isfile(path):
            continue
        with open(path, "rb") as f:
            desc = pickle.load(f)
        # scene_description is a tuple (caption, [object, ...]).
        objects = desc[1] if isinstance(desc, (tuple, list)) and len(desc) > 1 else []
        out[scene] = [str(o).strip().lower() for o in objects]
    return out


def main():
    args = parse_args()
    scene_dir = args.scene_description or os.path.join(
        args.data_dir, "scene_description"
    )
    if not os.path.isdir(scene_dir):
        raise SystemExit(
            f"Could not find {scene_dir}. scene_description.zip from the base Grasp-Anything repo is "
            "required -- script/download_grasp_anything_pp.sh fetches it."
        )

    files = glob.glob(os.path.join(args.data_dir, "grasp_label_positive", "*.pt"))
    if not files:
        raise SystemExit("No grasp_label_positive/*.pt found in " + args.data_dir)
    sample_ids = sorted(os.path.splitext(os.path.basename(f))[0] for f in files)
    print(f"{len(sample_ids):,} sample")

    scenes = {sid.rsplit("_", 2)[0] for sid in sample_ids}
    print(f"{len(scenes):,} scenes, reading scene_description...")
    catalog = load_categories(scene_dir, scenes)
    print(f"{len(catalog):,} scenes have a description")

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
        f"{len(sample_cat):,} samples have a category, dropped {skipped:,} (missing description or index out of "
        "range)"
    )

    freq = collections.Counter(sample_cat.values())
    if args.split_mode == "balanced":
        base, new, got = assign_balanced(freq, args.base_frac)
    else:
        ordered = [c for c, _ in freq.most_common()]
        n_base = int(round(len(ordered) * args.base_frac))
        base, new = set(ordered[:n_base]), set(ordered[n_base:])
        got = {"base": sum(freq[c] for c in base), "new": sum(freq[c] for c in new)}
    total = got["base"] + got["new"]
    print(
        f"{len(freq):,} categories -> Base {len(base):,} / New {len(new):,}  (mode {args.split_mode})"
    )
    print(
        "   samples: Base {:,} ({:.1%}) / New {:,} ({:.1%})".format(
            got["base"], got["base"] / total, got["new"], got["new"] / total
        )
    )

    seen = sorted(sid for sid, c in sample_cat.items() if c in base)
    unseen = sorted(sid for sid, c in sample_cat.items() if c in new)

    # One image holds several objects of different categories, so it can land in both sets.
    # Keeping it is an image-level leak: the evaluation image would have been trained on
    # directly, and conversely the training image already contains pixels of a "never seen"
    # category.
    #
    # Dropped from *seen* by default: a little training data is lost, but the evaluation set
    # keeps its size and New categories genuinely never appear during training -- a much
    # stronger guarantee than shrinking the evaluation set.
    shared = {s.rsplit("_", 2)[0] for s in seen} & {s.rsplit("_", 2)[0] for s in unseen}
    print(f"{len(shared):,} scenes appear in both seen and unseen")
    if shared and args.drop_shared_from != "none":
        side = args.drop_shared_from
        ids = seen if side == "seen" else unseen
        before = len(ids)
        kept = [s for s in ids if s.rsplit("_", 2)[0] not in shared]
        if side == "seen":
            seen = kept
        else:
            unseen = kept
        print(
            f"   dropped {before - len(kept):,} samples from {side} ({(before - len(kept)) / max(before, 1):.1%} of that set); use --drop-shared-from "
            "to change"
        )

    if args.max_samples:
        rng = random.Random(args.seed)
        seen = sorted(rng.sample(seen, min(args.max_samples, len(seen))))
        unseen = sorted(rng.sample(unseen, min(args.max_samples, len(unseen))))

    os.makedirs(args.out_dir, exist_ok=True)
    for name, ids in (("seen", seen), ("unseen", unseen)):
        with open(os.path.join(args.out_dir, name + ".obj"), "wb") as f:
            pickle.dump(ids, f)
        print(f"{name:<6} {len(ids):>9,} samples")

    # Any number drawn from a too-small evaluation set is meaningless -- say so while building
    # the split rather than letting it be discovered after training finishes.
    n_all = len(seen) + len(unseen)
    for name, ids in (("seen", seen), ("unseen", unseen)):
        if n_all and len(ids) / n_all < 0.05:
            se = (0.25 / max(len(ids), 1)) ** 0.5
            print(
                f"\nWARNING: {name} holds only {len(ids):,} samples ({len(ids) / n_all:.2%} of the total). The standard "
                f"error of a proportion on this set is +/-{se:.1%} -- not enough to compare "
                "models."
            )
            if args.split_mode == "paper":
                print(
                    "Try --split-mode balanced to split by sample count rather than category count."
                )

    print("\nWritten to " + args.out_dir)


if __name__ == "__main__":
    main()
