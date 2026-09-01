"""Inspect the GA++ data **before** training -- run it once, read it carefully, then spend GPU.

Three things can ruin the whole experiment without raising any error:

1. `part_mask` is actually at *object* level: every part of an object has the same mask. Then
   `L_align` carries no part-distinguishing information and every improvement on the alignment
   architecture is capped. Warning threshold: most same-object part pairs have IoU > 0.9.
2. The foreground is too small (or too large): this decides whether Dice is necessary, and what
   baseline the observed `align_bce` should be compared against (what BCE the all-zero
   prediction achieves).
3. How many tokens survive in a prompt after SOT/EOT/pad/punctuation are removed -- exactly the
   number of candidates the softmax over tokens must choose between. If the average is only 2-3
   tokens, the maximum entropy is only ~1.0, so do not read much into "concentrated" attention.

    python script/check_dataset.py --data-dir data/grasp-anything-pp-200k \\
        --split-path split/grasp-anything-pp --out results/dataset-check

A deeper report on question (1) alone, with examples for manual inspection:
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
                   help="Objects (with >=2 parts) to sample for the mask statistics")
    p.add_argument("--n-prompts", type=int, default=2000,
                   help="Prompts to sample for the token count")
    p.add_argument("--n-grid", type=int, default=24, help="Cells in the visual inspection grid")
    p.add_argument("--no-clip", action="store_true",
                   help="Skip the token count (avoids loading CLIP)")
    p.add_argument("--out", default=None, help="Directory for the JSON + PNG output")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def mask_report(data_dir, n_objects, seed, threshold=0.9):
    """Foreground, IoU between parts of one object, and the rate of identical masks."""
    files = glob.glob(os.path.join(data_dir, "part_mask", "*.npy"))
    if not files:
        raise SystemExit(f"No part_mask/*.npy found in {data_dir}")

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

    # On a small subset no object may have >=2 parts -- the foreground can still be reported.
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
    """Content tokens per prompt (after dropping SOT/EOT/pad/punctuation) + frequent tokens."""
    from inference.models.stag import CLIPTextEncoder

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
        # log(L): the upper bound on token/attention_entropy in TensorBoard.
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
        print("\n  WARNING: more than half of same-object part pairs have IoU > 0.9 -- the real")
        print("  supervision is closer to OBJECT level than part level. L_align cannot teach")
        print("  part-level grounding, and Delta_prompt in the counterfactual will be small even")
        print("  with a perfectly good model.")
    elif identical is not None and identical > 0.15:
        print("\n  Note: a sizeable fraction of objects have all parts sharing one mask; consider")
        print("  filtering them out of L_align. Details + examples: script/diagnose_part_masks.py")
    # The BCE of the trivial all-zero prediction -- the baseline that makes the observed
    # align_bce meaningful.
    fg = masks["fg_frac_median"]
    print(f"\n  Reference: foreground {fg:.2%} -> the all-zero prediction gives "
          f"BCE ~ {-np.log(1 - fg):.4f}"
          if fg < 1 else "")

    tokens = None
    if not args.no_clip:
        print("\n== 2. prompt ==")
        tokens, _ = token_report(dataset, args.n_prompts, args.seed)
        for k, v in tokens.items():
            if k == "most_common":
                print("  most frequent tokens:")
                print("    " + ", ".join(f"{t} {c}" for t, c in v))
            else:
                print(f"  {k:34} {v if not isinstance(v, float) else f'{v:.3f}'}")
        if tokens["n_empty_prompts"]:
            print(f"\n  WARNING: {tokens['n_empty_prompts']} prompts have no tokens left after")
            print("  filtering -- the model receives A_T = 0 for them.")

    print("\n== 3. image grid ==")
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
        print("  (skipped -- pass --out to write the figures)")


if __name__ == "__main__":
    main()
