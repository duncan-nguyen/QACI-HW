"""Collect one run's results into results/: logs, numbers, and alignment figures.

The figures are the point -- they show which token of the prompt matches which region of the
image, i.e. exactly the method's claim. See utils/visualisation/alignment.py.

    python script/export_results.py --checkpoint logs/.../epoch_42_iou_0.31 \
        --dataset-path data/grasp-anything-pp-full --split-path split/grasp-anything-pp \
        --out results/my-run
"""

import argparse
import datetime
import json
import os
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.checkpoint import load_network
from utils.data import get_dataset
from utils.visualisation.alignment import (
    plot_grasp_prediction,
    plot_part_prompts,
    plot_prompt_comparison,
    plot_prompt_grid,
    plot_sample_diagnostics,
    plot_token_alignment,
)

REFERENCE = [
    ("GR-ConvNet + CLIP", 0.37, 0.18, 0.24),
    ("CLIP-Fusion", 0.40, 0.29, 0.33),
    ("LGD", 0.48, 0.42, 0.45),
]

FREE_PROMPTS = [
    "Grasp the object at its handle.",
    "Grasp the object at its top.",
    "Grasp the object at its bottom.",
]


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--split-path", default=None)
    p.add_argument("--out", required=True, help="Results directory")
    p.add_argument("--n-samples", type=int, default=4, help="Samples to draw per split")
    p.add_argument("--input-size", type=int, default=224)
    p.add_argument("--seen-acc", type=float, default=None)
    p.add_argument("--unseen-acc", type=float, default=None)
    p.add_argument("--train-log", default=None)
    p.add_argument("--eval-logs", nargs="*", default=[])
    p.add_argument("--tensorboard-dir", default=None)
    p.add_argument("--config", default=None, help="Config file to copy alongside")
    p.add_argument(
        "--copy-checkpoint",
        action="store_true",
        help="Also copy the checkpoint into results/",
    )
    p.add_argument(
        "--audit-json",
        default=None,
        help="JSON from script/audit_text_reliance.py, to include the "
        "counterfactual table in summary.md (default: "
        "<out>/audit_text_reliance.json if present)",
    )
    return p.parse_args()


def load_split(path, split_path, seen, input_size):
    kwargs = dict(
        output_size=input_size, include_depth=False, include_rgb=True, seen=seen
    )
    if split_path:
        kwargs["split_path"] = split_path
    return get_dataset("grasp-anything-pp")(path, **kwargs)


def save(fig, path):
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


def main():
    args = parse_args()
    figures = os.path.join(args.out, "figures")
    os.makedirs(figures, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = load_network(args.checkpoint, map_location=device)
    # The baseline arm (`--use-text 0`) has no language branch: every token/A_T figure must be
    # skipped, otherwise `net.token_alignment()` raises RuntimeError and kills the export of
    # exactly the arm used as the reference point.
    has_text = getattr(net, "use_text", False)
    print(f"checkpoint: {args.checkpoint}  ({device}, use_text={has_text})")

    # ---------------------------------------------------------- figures ----
    print("\ndrawing figures:")
    ranked_note, pred_note = {}, {}
    for name, seen in (("seen", True), ("unseen", False)):
        ds = load_split(args.dataset_path, args.split_path, seen, args.input_size)
        step = max(1, len(ds) // max(args.n_samples, 1))
        for k in range(args.n_samples):
            idx = min(k * step, len(ds) - 1)
            x, _, _, _, _, extra = ds[idx]

            # Predicted grasp beside the ground truth -- the end result, pass/fail at a glance.
            fig, info = plot_grasp_prediction(net, ds, idx)
            save(fig, os.path.join(figures, f"prediction_{name}_{k:02d}.png"))
            pred_note[f"{name}_{k:02d}"] = info

            if has_text:
                fig, ranked = plot_token_alignment(
                    net, x, extra["prompt"], part_mask=extra["part_mask"], max_tokens=7
                )
                save(fig, os.path.join(figures, f"tokens_{name}_{k:02d}.png"))
                ranked_note[f"{name}_{k:02d}"] = (extra["prompt"], ranked)

            # The full diagnostic row: adds A_T's error map (TP/FP/FN) and each token's
            # attention mass -- enough to say whether the fault is grounding or grasp decoding.
            fig, _ = plot_sample_diagnostics(net, ds, idx)
            save(fig, os.path.join(figures, f"diagnostic_{name}_{k:02d}.png"))

        if seen and has_text:
            # One object, several parts -> changing the prompt must move A_T.
            obj, files = max(ds._files_by_object.items(), key=lambda kv: len(kv[1]))
            idxs = [ds.grasp_files.index(f) for f in files][:4]
            if len(idxs) > 1:
                save(
                    plot_part_prompts(net, ds, idxs),
                    os.path.join(figures, "parts_same_object.png"),
                )

            x, _, _, _, _, extra = ds[0]
            save(
                plot_prompt_comparison(net, x, [extra["prompt"]] + FREE_PROMPTS),
                os.path.join(figures, "prompts_free_form.png"),
            )

            # Same image, different prompts -- the most important figure. Three identical rows
            # mean the model is ignoring the language.
            grid = [("real", extra["prompt"])] + [
                ("free-form", p) for p in FREE_PROMPTS
            ]
            save(
                plot_prompt_grid(net, ds, 0, grid),
                os.path.join(figures, "prompt_grid.png"),
            )

    # -------------------------------------------------------------- log ----
    print("\ncopying logs:")
    for src in filter(None, [args.train_log, args.config] + list(args.eval_logs)):
        if os.path.isfile(src):
            shutil.copy(src, os.path.join(args.out, os.path.basename(src)))
            print(f"  {os.path.basename(src)}")
    if args.tensorboard_dir and os.path.isdir(args.tensorboard_dir):
        dest = os.path.join(args.out, "tensorboard")
        shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(
            args.tensorboard_dir, dest, ignore=shutil.ignore_patterns("epoch_*")
        )
        print("  tensorboard/")
    if args.copy_checkpoint:
        shutil.copy(
            args.checkpoint, os.path.join(args.out, os.path.basename(args.checkpoint))
        )
        print(f"  {os.path.basename(args.checkpoint)}")

    # ---------------------------------------------------------- summary ----
    lines = [
        "# Results",
        "",
        f"- timestamp: {datetime.datetime.now():%Y-%m-%d %H:%M}",
        f"- checkpoint: `{args.checkpoint}`",
        f"- dataset: `{args.dataset_path}`",
        f"- split: `{args.split_path or "default (the loader's SPLIT_DIRS)"}`",
        "",
    ]

    if args.seen_acc is not None and args.unseen_acc is not None:
        s, u = args.seen_acc, args.unseen_acc
        h = 0.0 if s + u == 0 else 2 * s * u / (s + u)
        lines += [
            "## Scores",
            "",
            "| | Seen | Unseen | H |",
            "|---|---|---|---|",
            f"| **ours** | **{s:.3f}** | **{u:.3f}** | **{h:.3f}** |",
        ]
        lines += [
            f"| {n} † | {a:.2f} | {b:.2f} | {c:.2f} |" for n, a, b, c in REFERENCE
        ]
        lines += [
            "",
            "† Table 2 of the LGD paper (full GA++, the original split over LVIS "
            "labels). The split here reproduces the protocol, so these are reference "
            "points only.",
            "",
            "Metric: success when IoU >= 0.25 and angular error <= 30 deg.",
            "",
        ]

    audit_path = args.audit_json or os.path.join(args.out, "audit_text_reliance.json")
    if os.path.isfile(audit_path):
        with open(audit_path) as f:
            audit = json.load(f)
        lines += [
            "## Prompt counterfactual",
            "",
            "Same image, same ground truth, only the prompt changes "
            "(`script/audit_text_reliance.py`).",
            "",
            "| split | condition | n | success | IoU | A_T IoU | pointing |",
            "|---|---|---|---|---|---|---|",
        ]
        for split, r in audit.get("results", {}).items():
            for cond, row in r.get("conditions", {}).items():
                lines.append(
                    f"| {split} | {cond} | {row['n']} | {row['success']:.4f} | "
                    f"{row['iou']:.4f} | {row.get('align_iou', float('nan')):.4f} | "
                    f"{row.get('pointing', float('nan')):.4f} |"
                )
        drops = [
            f"{split}: "
            + ", ".join(
                f"Δ_{c} = {r[f'drop_{c}']:+.4f}"
                for c in ("shuffled", "fixed", "other_part")
                if f"drop_{c}" in r
            )
            for split, r in audit.get("results", {}).items()
        ]
        lines += (
            [
                "",
                "Delta = success(real prompt) - success(substituted prompt). "
                "Delta ~ 0 means the grasp does not depend on the prompt.",
                "",
            ]
            + [f"- {d}" for d in drops]
            + [""]
        )

    lines += [
        "## Figures",
        "",
        "- `figures/prediction_{seen,unseen}_*.png` — the predicted grasp (red) beside "
        "the ground truth (green), with the Q, angle and `A_T` maps. The title gives the "
        "best IoU and pass/fail under the metric (IoU >= 0.25 and angular error "
        "<= 30 deg).",
        "- `figures/tokens_{seen,unseen}_*.png` — the alignment map of each prompt token, "
        "beside the ground-truth `part_mask` and the aggregated `A_T`.",
        "- `figures/parts_same_object.png` — one object, changing the part in the "
        "instruction.",
        "- `figures/prompts_free_form.png` — free-form prompts on the same image.",
        "- `figures/diagnostic_{seen,unseen}_*.png` — RGB · part_mask · `A_T` · `A_T` "
        "error (green TP / red FP / yellow FN) · Q · grasp · strongest tokens with their "
        "attention mass.",
        "- `figures/prompt_grid.png` — one image, different prompts: `A_T`, tokens and "
        "grasp must all change.",
        "- `figures/failures_{seen,unseen}.png` — the four-class failure gallery (if the "
        "audit was run with `--figures`).",
        "",
        "### Per-sample predictions",
        "",
        "| sample | prompt | grasps detected | best IoU | outcome |",
        "|---|---|---|---|---|",
    ]
    for key, info in pred_note.items():
        # 0 grasps means Q never crosses peak_local_max's 0.2 threshold anywhere -- quite
        # different from "predicted but wrong", so it is reported separately.
        mark = (
            "pass"
            if info["success"]
            else ("no prediction" if not info["n_grasps"] else "fail")
        )
        lines.append(
            f'| {key} | "{info["prompt"]}" | {info["n_grasps"]} | '
            f"{info['iou']:.3f} | {mark} |"
        )

    if ranked_note:
        lines += ["", "### Strongest tokens per figure", ""]
        for key, (prompt, ranked) in ranked_note.items():
            top = ", ".join(f"`{t}` {v:.2f}" for t, v in ranked[:4])
            lines.append(f'- **{key}** — "{prompt}" → {top}')
    else:
        lines += [
            "",
            "This checkpoint has no language branch (`use_text=0`), so there are no "
            "token/alignment figures.",
            "",
        ]

    path = os.path.join(args.out, "summary.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n{path}")


if __name__ == "__main__":
    main()
