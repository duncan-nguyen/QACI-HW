"""Create clean same-image/different-part alignment figures for the paper.

Each output figure uses one object from the exact independent test subset.  Its two rows are
two real annotated parts of the same object, so the RGB image stays fixed.  Columns show the
target-part contour, the NoAlign prediction, and the Full prediction.  Prediction panels
overlay the learned alignment map and the top-1 grasp.

Candidate objects are selected only by the distance between the two ground-truth part-mask
centroids.  Model scores are never used for selection, which avoids choosing an example just
because Full happens to win on it.
"""

import argparse
import json
import os
import sys
import textwrap
from collections import defaultdict
from itertools import combinations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.post_process import post_process_output  # noqa: E402
from utils.checkpoint import load_network  # noqa: E402
from utils.data import get_dataset  # noqa: E402
from utils.data.index_split import index_splits, select_subset  # noqa: E402
from utils.dataset_processing.grasp import detect_grasps  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noalign-checkpoint", required=True)
    parser.add_argument("--full-checkpoint", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--split-path", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--split", type=float, default=0.995)
    parser.add_argument("--test-split", type=float, default=0.005)
    parser.add_argument("--subset", choices=("val", "test"), default="test")
    parser.add_argument("--seen", type=int, choices=(0, 1), default=0)
    parser.add_argument("--random-seed", type=int, default=123)
    parser.add_argument("--ds-shuffle", action="store_true")
    parser.add_argument(
        "--n-candidates",
        type=int,
        default=4,
        help="Number of independently selected candidate objects to render.",
    )
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.n_candidates < 1:
        parser.error("--n-candidates must be at least 1")
    return args


def load_dataset(args):
    kwargs = dict(
        output_size=args.input_size,
        include_depth=False,
        include_rgb=True,
        random_rotate=False,
        random_zoom=False,
        seen=bool(args.seen),
    )
    if args.split_path:
        kwargs["split_path"] = args.split_path
    return get_dataset("grasp-anything-pp")(args.dataset_path, **kwargs)


def mask_centroid(mask):
    points = np.argwhere(mask > 0.5)
    if not len(points):
        return None
    return points.mean(axis=0)


def candidate_pairs(dataset, indices):
    """Return object/pair candidates ordered only by GT part separation."""
    groups = defaultdict(list)
    for idx in indices:
        groups[dataset._object_id(dataset.sample_id(idx))].append(idx)

    candidates = []
    for object_id, object_indices in groups.items():
        if len(object_indices) < 2:
            continue
        masks = {idx: dataset.get_part_mask(idx) for idx in object_indices}
        centroids = {idx: mask_centroid(mask) for idx, mask in masks.items()}
        pairs = []
        for left, right in combinations(object_indices, 2):
            if centroids[left] is None or centroids[right] is None:
                continue
            distance = float(np.linalg.norm(centroids[left] - centroids[right]))
            pairs.append((distance, left, right))
        if pairs:
            distance, left, right = max(pairs)
            candidates.append((distance, object_id, left, right))

    # GT-only selection: visually separated parts make the same-image comparison readable.
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates


def rgb_for_show(dataset, idx):
    rgb = np.asarray(dataset.get_rgb(idx, normalise=False)).astype(np.float32)
    if rgb.max() > 1.0:
        rgb /= 255.0
    return np.clip(rgb, 0.0, 1.0)


def predict(net, x, prompt, output_size):
    device = next(net.parameters()).device
    with torch.no_grad():
        pred = net.predict(x.unsqueeze(0).to(device), [prompt])
    q_img, ang_img, width_img = post_process_output(
        pred["pos"], pred["cos"], pred["sin"], pred["width"]
    )
    grasps = detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=1)
    align = F.interpolate(
        pred["align"].float(),
        size=(output_size, output_size),
        mode="bilinear",
        align_corners=False,
    )[0, 0].cpu().numpy()
    return align, grasps


def draw_target(ax, rgb, mask, title):
    ax.imshow(rgb)
    ax.contour(mask, levels=[0.5], colors=["#00E5FF"], linewidths=1.8)
    ax.set_title(title, fontsize=9, weight="semibold")


def draw_prediction(ax, rgb, mask, align, grasps, title):
    ax.imshow(rgb)
    ax.imshow(align, cmap="magma", alpha=0.55, vmin=0.0, vmax=1.0)
    ax.contour(mask, levels=[0.5], colors=["#00E5FF"], linewidths=1.3)
    for grasp in grasps:
        grasp.plot(ax, color="#39FF14")
    ax.set_title(title, fontsize=9, weight="semibold")


def render_candidate(dataset, pair, nets, output_dir, rank, split_name):
    distance, object_id, left, right = pair
    indices = (left, right)
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 3.7), squeeze=False)
    row_metadata = []

    for row, idx in enumerate(indices):
        x, _, _, _, _, extra = dataset[idx]
        prompt = extra["prompt"]
        mask = dataset.get_part_mask(idx)
        rgb = rgb_for_show(dataset, idx)

        draw_target(axes[row, 0], rgb, mask, "Target part" if row == 0 else "")
        for col, (label, net) in enumerate(nets, start=1):
            align, grasps = predict(net, x, prompt, dataset.output_size)
            draw_prediction(
                axes[row, col],
                rgb,
                mask,
                align,
                grasps,
                label if row == 0 else "",
            )

        axes[row, 0].set_ylabel(
            textwrap.fill(prompt, width=34), fontsize=8, rotation=0, labelpad=70, va="center"
        )
        row_metadata.append({"index": idx, "sample_id": dataset.sample_id(idx), "prompt": prompt})

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)

    fig.suptitle(
        f"Same {split_name} image, two annotated parts",
        fontsize=10,
        weight="semibold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.012,
        "Cyan: target-part boundary     Magenta/yellow: learned alignment     Green: top grasp",
        ha="center",
        fontsize=7.5,
    )
    fig.subplots_adjust(left=0.23, right=0.995, top=0.91, bottom=0.08, wspace=0.035, hspace=0.16)

    stem = f"candidate_{rank:02d}_{object_id}"
    png_path = os.path.join(output_dir, stem + ".png")
    pdf_path = os.path.join(output_dir, stem + ".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    return {
        "rank": rank,
        "object_id": object_id,
        "gt_centroid_distance_px": distance,
        "rows": row_metadata,
        "png": png_path,
        "pdf": pdf_path,
    }


def main():
    args = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    dataset = load_dataset(args)
    splits = index_splits(
        dataset.length,
        train_frac=args.split,
        test_frac=args.test_split,
        shuffle=args.ds_shuffle,
        seed=args.random_seed,
    )
    indices = select_subset(splits, args.subset)
    candidates = candidate_pairs(dataset, indices)
    if not candidates:
        raise SystemExit(
            f"No object with at least two non-empty part masks in {args.subset!r} subset."
        )

    noalign = load_network(args.noalign_checkpoint, map_location=device)
    full = load_network(
        args.full_checkpoint,
        map_location=device,
        text_encoder=getattr(noalign, "text_encoder", None),
    )
    if not getattr(noalign, "use_text", False) or not getattr(full, "use_text", False):
        raise SystemExit("Both checkpoints must have use_text=1.")

    os.makedirs(args.out_dir, exist_ok=True)
    split_name = ("Seen" if args.seen else "Unseen") + f" {args.subset}"
    rendered = [
        render_candidate(
            dataset,
            pair,
            (("NoAlign", noalign), ("Full", full)),
            args.out_dir,
            rank,
            split_name,
        )
        for rank, pair in enumerate(candidates[: args.n_candidates], start=1)
    ]

    manifest = {
        "selection": "largest distance between GT part-mask centroids; model outputs unused",
        "category_split": "seen" if args.seen else "unseen",
        "subset": args.subset,
        "split": args.split,
        "test_split": args.test_split,
        "random_seed": args.random_seed,
        "n_subset_samples": len(indices),
        "noalign_checkpoint": args.noalign_checkpoint,
        "full_checkpoint": args.full_checkpoint,
        "figures": rendered,
    }
    manifest_path = os.path.join(args.out_dir, "manifest.json")
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    print(f"Exact subset: {split_name} ({len(indices):,} samples)")
    print(f"Eligible multi-part objects: {len(candidates):,}")
    for item in rendered:
        print(f"candidate {item['rank']}: {item['png']}")
        for row in item["rows"]:
            print(f"  - {row['prompt']}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
