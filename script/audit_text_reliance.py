"""Does the model *actually* read the prompt? Measured by pairing images with wrong prompts.

A language-driven model can score well while ignoring language entirely: GA++ has ~4.4 parts
per scene with overlapping grasp regions, so "the image's average grasp" already scores
reasonably under the IoU >= 0.25 metric. The Seen/Unseen/H table alone cannot tell those two
cases apart.

The counterfactual keeps the image and the ground truth and only substitutes the prompt. Four
conditions, same model, same samples:

    normal      the real prompt                          -- the reference result
    shuffled    another object's prompt                  -- is there any language dependence
    fixed       one sentence for every image             -- is text ignored entirely
    other_part  another part's prompt, *same object*     -- are parts distinguished at all

The number to read is Delta_prompt = S_normal - S_shuffled:

* large -> the language branch is genuinely steering the grasp.
* ~ 0   -> the model predicts from the image alone and the language is decorative, however low
           align_loss got.

`other_part` is harder than `shuffled`: both prompts name the same object and differ only in
the part. A large Delta on `shuffled` with `other_part` ~ 0 means the model has learned to
recognise the object, not the part.

Two grounding metrics on `A_T` come along with it (at the feature-map resolution, as in
training):

* `IoU(A_T > 0.5, part_mask)` -- is the right part region outlined
* `pointing` -- does the strongest `A_T` pixel land inside part_mask (threshold-free)

    python script/audit_text_reliance.py --checkpoint logs/.../epoch_67_iou_0.3313 \\
        --dataset-path data/grasp-anything-pp-200k --split-path split/grasp-anything-pp \\
        --n-samples 500

Companion: `script/diagnose_part_masks.py` checks whether `part_mask` itself is at part level
or is just the whole object's mask -- if the labels are already at object level, `L_align`
cannot teach part-level grounding and the table below can only show small deltas.
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.post_process import post_process_output  # noqa: E402
from utils.checkpoint import load_network  # noqa: E402
from utils.data import get_dataset  # noqa: E402
from utils.dataset_processing.grasp import detect_grasps  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--split-path", default=None)
    p.add_argument("--input-size", type=int, default=224)
    p.add_argument("--n-samples", type=int, default=500,
                   help="Samples per split (evenly spaced). 0 = all of them.")
    p.add_argument("--iou-threshold", type=float, default=0.25)
    p.add_argument("--n-grasps", type=int, default=1)
    p.add_argument("--seed", type=int, default=0, help="Seed for the prompt permutation")
    p.add_argument("--splits", nargs="+", default=["seen", "unseen"],
                   choices=["seen", "unseen"])
    p.add_argument("--fixed-prompt", default="Grasp the object.",
                   help="The sentence shared by every image in the 'fixed' condition")
    p.add_argument("--out", default=None, help="Write the results as JSON to this file")
    p.add_argument("--figures", default=None,
                   help="Directory for the four-class failure gallery of each split")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def load_dataset(args, seen):
    kwargs = dict(output_size=args.input_size, include_depth=False, include_rgb=True,
                  seen=seen)
    if args.split_path:
        kwargs["split_path"] = args.split_path
    return get_dataset("grasp-anything-pp")(args.dataset_path, **kwargs)


def grasp_metrics(net, x, prompt, gtbb, iou_threshold, n_grasps):
    """One forward pass with a single prompt -> (max IoU, pass/fail, Q, A_T)."""
    with torch.no_grad():
        pred = net.predict(x, [prompt]) if getattr(net, "use_text", False) else net.predict(x)
    q_img, ang_img, width_img = post_process_output(
        pred["pos"], pred["cos"], pred["sin"], pred["width"])
    grasps = detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=n_grasps)
    # `Grasp.max_iou` returns 0 when the angular error exceeds 30 degrees, so comparing against
    # the IoU threshold captures both conditions of the paper's metric.
    best = max((g.max_iou(gtbb) for g in grasps), default=0.0)
    return float(best), bool(best > iou_threshold), pred


def align_metrics(pred, part_mask):
    """
    IoU and pointing game of `A_T` against `part_mask`, at A_T's resolution (as in training).
    """
    if "align" not in pred:
        return None, None
    a = pred["align"][0, 0]
    target = F.adaptive_avg_pool2d(part_mask[None, None], a.shape[-2:])[0, 0] > 0.5
    if target.sum() == 0:
        return None, None

    hit = (a > 0.5) & target
    union = (a > 0.5) | target
    iou = float(hit.sum()) / float(union.sum()) if union.sum() else 0.0
    peak = int(a.argmax())
    return iou, bool(target.flatten()[peak])


CONDITIONS = ("normal", "shuffled", "fixed", "other_part")


def other_part_prompt(ds, idx):
    """The prompt of a *different* part of the same object, or None if it has only one."""
    object_id = ds._object_id(ds.sample_id(idx))
    for path in ds._files_by_object.get(object_id, []):
        other = ds.grasp_files.index(path)
        if other != idx:
            return ds.get_prompt(other, use_permutation=False)
    return None


def audit_split(net, ds, args, device):
    ds.shuffle_prompts(seed=args.seed)          # one permutation shared by the whole split
    n = len(ds.grasp_files)
    step = max(1, n // args.n_samples) if args.n_samples else 1
    indices = list(range(0, n, step))[: args.n_samples or n]

    keys = ("ok", "iou", "align_iou", "point")
    acc = {c: {k: 0.0 for k in keys} for c in CONDITIONS}
    seen = {c: 0 for c in CONDITIONS}
    n_align = {c: 0 for c in CONDITIONS}
    dq = 0.0

    for count, idx in enumerate(indices, 1):
        x, _, _, rot, zoom, extra = ds[idx]
        x = x.unsqueeze(0).to(device)
        gtbb = ds.get_gtbb(idx, rot, zoom)
        part_mask = extra["part_mask"][0].to(device)

        prompts = {
            "normal": ds.get_prompt(idx, use_permutation=False),
            "shuffled": ds.get_prompt(idx),
            "fixed": args.fixed_prompt,
            "other_part": other_part_prompt(ds, idx),
        }

        preds = {}
        for cond, prompt in prompts.items():
            if prompt is None:
                continue
            best, ok, pred = grasp_metrics(net, x, prompt, gtbb, args.iou_threshold,
                                           args.n_grasps)
            preds[cond] = pred
            seen[cond] += 1
            acc[cond]["ok"] += ok
            acc[cond]["iou"] += best

            a_iou, point = align_metrics(pred, part_mask)
            if a_iou is not None:
                n_align[cond] += 1
                acc[cond]["align_iou"] += a_iou
                acc[cond]["point"] += point

        if "shuffled" in preds:
            dq += float((preds["normal"]["pos"] - preds["shuffled"]["pos"]).abs().mean())

        if count % 100 == 0:
            print(f"    {count}/{len(indices)}", flush=True)

    out = {"n_samples": len(indices), "dq": dq / max(1, len(indices)), "conditions": {}}
    for cond in CONDITIONS:
        if not seen[cond]:
            continue
        row = {"n": seen[cond],
               "success": acc[cond]["ok"] / seen[cond],
               "iou": acc[cond]["iou"] / seen[cond]}
        if n_align[cond]:
            row["align_iou"] = acc[cond]["align_iou"] / n_align[cond]
            row["pointing"] = acc[cond]["point"] / n_align[cond]
        out["conditions"][cond] = row

    normal = out["conditions"].get("normal", {}).get("success")
    for cond in ("shuffled", "fixed", "other_part"):
        if normal is not None and cond in out["conditions"]:
            out[f"drop_{cond}"] = normal - out["conditions"][cond]["success"]
    return out


def save_failure_gallery(net, ds, args, folder, name):
    """The four-class failure gallery -- grounding fault or grasp-decoder fault."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from utils.visualisation.alignment import plot_failure_gallery

    ds.real_prompts()
    n = len(ds.grasp_files)
    step = max(1, n // min(args.n_samples or n, 200))
    indices = list(range(0, n, step))[:200]

    os.makedirs(folder, exist_ok=True)
    fig, counts = plot_failure_gallery(net, ds, indices, iou_threshold=args.iou_threshold)
    path = os.path.join(folder, f"failures_{name}.png")
    if fig is not None:
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"  failure gallery: {path}")
    return counts


def print_table(results):
    print()
    header = f"{'split':8} {'condition':11} {'n':>6} {'success':>9} {'IoU':>8} " \
             f"{'A_T IoU':>9} {'pointing':>9}"
    print(header)
    print("-" * len(header))
    for split, r in results.items():
        for cond in CONDITIONS:
            row = r["conditions"].get(cond)
            if row is None:
                continue
            align = f"{row['align_iou']:9.4f}" if "align_iou" in row else f"{'-':>9}"
            point = f"{row['pointing']:9.4f}" if "pointing" in row else f"{'-':>9}"
            print(f"{split:8} {cond:11} {row['n']:6d} {row['success']:9.4f} "
                  f"{row['iou']:8.4f} {align} {point}")
        drops = "  ".join(f"Δ_{c}={r[f'drop_{c}']:+.4f}" for c in
                          ("shuffled", "fixed", "other_part") if f"drop_{c}" in r)
        print(f"{split:8} {drops}   mean|ΔQ|={r['dq']:.2e}")
        print()

    print("Delta_shuffled ~ 0: the grasp does not depend on the prompt -- nothing can be")
    print("concluded about the language branch being useful, however low align_loss is. A large")
    print("Delta_shuffled with Delta_other_part ~ 0: the model recognises the *object* but not")
    print("the *part*. Before blaming the model, run script/check_dataset.py to see whether")
    print("part_mask really is at part level.")


def main():
    args = parse_args()
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    net = load_network(args.checkpoint, map_location=device)
    print(f"checkpoint: {args.checkpoint}  ({device})")
    print(f"use_text={getattr(net, 'use_text', False)}  "
          f"align_mode={getattr(net, 'align_mode', '-')}  "
          f"fusion={getattr(net, 'fusion', '-')}")

    results = {}
    for name in args.splits:
        print(f"\n== {name} ==")
        ds = load_dataset(args, seen=(name == "seen"))
        print(f"  {len(ds.grasp_files)} sample trong split")
        results[name] = audit_split(net, ds, args, device)
        if args.figures:
            results[name]["failure_classes"] = save_failure_gallery(
                net, ds, args, args.figures, name)

    print_table(results)

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"checkpoint": args.checkpoint, "seed": args.seed,
                       "iou_threshold": args.iou_threshold, "results": results}, f, indent=2)
        print(f"\nJSON: {args.out}")


if __name__ == "__main__":
    main()
