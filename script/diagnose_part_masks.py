"""Check whether Grasp-Anything++'s `part_mask` is really at *part* level or only at *object* level.

The motivation: `results/.../figures/parts_same_object.png` shows four different prompts on the
same apple -- "by its skin", "by its flesh", "by its seeds", "on its stem" -- with **identical**
`part_mask`s, all covering the whole apple. If that is common rather than exceptional, then
`L_align(A_T, part_mask)` cannot teach part-level grounding: the supervision signal carries no
part information. This script measures that rate over the whole dataset.

Three numbers to read:

* `identical`  -- the fraction of objects where **every** part has the same mask
                  (IoU >= --iou-identical). High means the labels are at object level and the
                  part-level claim cannot be supervised.
* `part~union` -- the fraction of parts whose mask approximates the union of every part of the
                  same object. Same meaning, measured per part rather than per object.
* `fg_frac`    -- the foreground area fraction, for interpreting the observed `align_loss`.

numpy only, no torch/GPU required:

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

# Grasp-Anything images and part_masks are always 416x416 (see utils/data/grasp_anything_pp_data.py).
SOURCE_SIZE = 416


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data-dir", required=True, help="GA++ directory (holding part_mask/)")
    p.add_argument(
        "--n-objects",
        type=int,
        default=2000,
        help="How many objects (with >=2 parts) to sample. 0 = all of them.",
    )
    p.add_argument(
        "--iou-identical",
        type=float,
        default=0.95,
        help="IoU above which two masks count as identical.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="Write the results as JSON to this file")
    p.add_argument(
        "--examples", type=int, default=5, help="How many examples to print with their prompts"
    )
    return p.parse_args()


def load_mask(path):
    """
    416x416 uint8 part_mask, bit-packed form also accepted.

    The original is `GraspAnythingPPDataset._load_mask`. Duplicated here so the script runs
    without importing torch.
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
        raise SystemExit("Not found: " + mask_dir)

    files = glob.glob(os.path.join(mask_dir, "*.npy"))
    if not files:
        raise SystemExit("No part_mask/*.npy found in " + args.data_dir)

    # Group by object "<scene>_<object>"; each element is one part.
    by_object = collections.defaultdict(list)
    for f in files:
        sid = os.path.splitext(os.path.basename(f))[0]
        by_object[sid.rsplit("_", 1)[0]].append(f)
    multi = {k: v for k, v in by_object.items() if len(v) >= 2}
    print(f"{len(files):,} part masks · {len(by_object):,} objects · {len(multi):,} with >=2 parts")
    if not multi:
        raise SystemExit("No object has more than one part -- nothing to check.")

    keys = sorted(multi)
    if args.n_objects and args.n_objects < len(keys):
        keys = rng.sample(keys, args.n_objects)
    print(f"sampling {len(keys):,} objects\n")

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
    print(f"identical   : {n_identical:,}/{len(keys):,} = {frac_identical:.1%} of objects have "
          "ALL parts sharing one mask")
    print(f"              (IoU threshold >= {args.iou_identical})")
    print(f"part~union  : {frac_part_is_union:.1%} of parts have a mask ~ the union of all "
          "parts of the object")
    print()
    print(f"IoU between parts of one object: median {statistics.median(mean_ious):.3f} · "
          f"p10 {pct(mean_ious, 10):.3f} · p90 {pct(mean_ious, 90):.3f}")
    print(f"lowest IoU within each object  : median {statistics.median(min_ious):.3f}")
    print(f"parts per object               : median {statistics.median(n_parts):.1f} · "
          f"max {max(n_parts)}")
    print(f"foreground area fraction       : median {statistics.median(fg_fracs):.4f} · "
          f"p10 {pct(fg_fracs, 10):.4f} · p90 {pct(fg_fracs, 90):.4f}")
    print("=" * 72)

    if frac_identical > 0.5:
        print("\nCONCLUSION: part_mask is mostly at OBJECT level, not part level.")
        print("L_align(A_T, part_mask) therefore cannot teach part-level grounding -- the target")
        print("carries no information distinguishing parts. Every improvement on the alignment")
        print("architecture is capped by this limit.")
    elif frac_identical > 0.15:
        print("\nCONCLUSION: a sizeable fraction of part_masks is at object level. Those objects")
        print("should be filtered out of L_align, or weighted by how distinguishable their parts are.")
    else:
        print("\nCONCLUSION: part_masks do distinguish parts. The apple case is an exception;")
        print("if A_T does not track the part, the cause is on the architecture/visual-feature side.")

    # Examples for manual inspection, with their prompts where available.
    prompt_dir = os.path.join(args.data_dir, "grasp_instructions")
    if examples and os.path.isdir(prompt_dir):
        print("\nExample objects whose parts all share one mask:")
        for key, paths, v in examples:
            print(f"  {key}  (lowest IoU {v:.3f})")
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
