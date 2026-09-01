"""Gom kết quả một lần chạy vào results/: log, số liệu, và hình alignment.

Hình là phần chính -- chúng cho thấy token nào trong prompt đang ăn vào vùng nào của ảnh,
tức là đúng luận điểm của method. Xem utils/visualisation/alignment.py.

    python script/export_results.py --checkpoint logs/.../epoch_42_iou_0.31 \
        --dataset-path data/grasp-anything-pp-full --split-path split/grasp-anything-pp \
        --out results/my-run
"""

import argparse
import datetime
import os
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data import get_dataset
from utils.visualisation.alignment import (
    grasp_probe,
    plot_grasp_prediction,
    plot_part_prompts,
    plot_prompt_comparison,
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
    p.add_argument("--out", required=True, help="Thư mục kết quả")
    p.add_argument(
        "--n-samples", type=int, default=4, help="Số sample vẽ cho mỗi split"
    )
    p.add_argument("--input-size", type=int, default=224)
    p.add_argument(
        "--seen-split",
        type=float,
        default=None,
        help="Giống --split của evaluate.py. Khi có, hình của split seen chỉ lấy từ phần "
        "holdout `indices[split:]`. Không có thì lấy trên toàn bộ seen -- tức phần lớn là "
        "dữ liệu model đã train, hình minh hoạ khi đó không nói lên điều gì về khái quát hoá.",
    )
    p.add_argument(
        "--pool-factor",
        type=int,
        default=8,
        help="Quét n_samples × pool_factor ứng viên rồi chọn xen kẽ mẫu ĐẠT và TRƯỢT. Bản cũ "
        "lấy stride cố định nên có thể ra toàn mẫu trượt và không đại diện.",
    )
    p.add_argument("--seen-acc", type=float, default=None)
    p.add_argument("--unseen-acc", type=float, default=None)
    p.add_argument("--train-log", default=None)
    p.add_argument("--eval-logs", nargs="*", default=[])
    p.add_argument("--tensorboard-dir", default=None)
    p.add_argument("--config", default=None, help="File cấu hình để chép kèm")
    p.add_argument(
        "--copy-checkpoint",
        action="store_true",
        help="Chép luôn checkpoint (~265 MB) vào results/",
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


def choose_samples(net, ds, name, args):
    """
    Chọn sample để vẽ: quét một pool rồi lấy xen kẽ mẫu ĐẠT và TRƯỢT.

    Hai vấn đề của cách cũ (`idx = k * (len(ds) // n_samples)`):

    * với split seen nó quét *toàn bộ* seen dataset, tức chủ yếu là dữ liệu train -- hình
      minh hoạ không nói được gì về khái quát hoá. `--seen-split` giới hạn về đúng holdout.
    * stride cố định không đảm bảo có mẫu đạt. Ở accuracy ~44% mà cả 4 hình đều trượt thì
      người đọc kết luận sai về model.

    :return: {"picked": [(idx, probe)], "note": str mô tả nguồn lấy mẫu}
    """
    pool_idx = list(range(len(ds)))
    note = f"toàn bộ split {name} ({len(ds):,} sample)"
    if name == "seen" and args.seen_split is not None:
        cut = int(args.seen_split * len(ds))
        pool_idx = pool_idx[cut:]
        note = (
            f"holdout của split seen, `indices[{cut}:]` ({len(pool_idx):,} sample) -- "
            f"cùng tập mà evaluate.py dùng với --split {args.seen_split}"
        )

    want = max(args.n_samples, 1)
    step = max(1, len(pool_idx) // max(want * args.pool_factor, 1))
    hits, misses = [], []
    for i in pool_idx[::step]:
        if len(hits) >= want and len(misses) >= want:
            break
        probe = grasp_probe(net, ds, i)
        (hits if probe["success"] else misses).append((i, probe))

    # Xen kẽ đạt/trượt để bộ hình phản ánh cả hai phía, rồi bù bằng phía nào còn dư.
    picked = []
    for a, b in zip(hits, misses):
        picked.extend([a, b])
    picked.extend(hits[len(misses) :] + misses[len(hits) :])
    picked = picked[:want]
    note += (
        f"; quét {len(hits) + len(misses)} ứng viên, "
        f"{len(hits)} đạt / {len(misses)} trượt"
    )
    return {"picked": picked, "note": note}


def main():
    args = parse_args()
    figures = os.path.join(args.out, "figures")
    os.makedirs(figures, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = torch.load(args.checkpoint, map_location=device, weights_only=False).eval()
    print(f"checkpoint: {args.checkpoint}  ({device})")

    # ------------------------------------------------------------- hình ----
    print("\nvẽ hình:")
    ranked_note, pred_note = {}, {}
    sample_note = {}
    for name, seen in (("seen", True), ("unseen", False)):
        ds = load_split(args.dataset_path, args.split_path, seen, args.input_size)
        pool = choose_samples(net, ds, name, args)
        sample_note[name] = pool["note"]
        for k, (idx, probe) in enumerate(pool["picked"]):
            x, _, _, _, _, extra = ds[idx]

            # Grasp dự đoán cạnh ground truth -- kết quả cuối, đọc được ngay đạt hay trượt.
            fig, info = plot_grasp_prediction(net, ds, idx, probe=probe)
            save(fig, os.path.join(figures, f"prediction_{name}_{k:02d}.png"))
            pred_note[f"{name}_{k:02d}"] = info

            fig, ranked = plot_token_alignment(
                net, x, extra["prompt"], part_mask=extra["part_mask"], max_tokens=7
            )
            save(fig, os.path.join(figures, f"tokens_{name}_{k:02d}.png"))
            ranked_note[f"{name}_{k:02d}"] = (extra["prompt"], ranked)

        if seen:
            # Cùng một object, nhiều part -> đổi prompt thì A_T phải dịch theo.
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

    # -------------------------------------------------------------- log ----
    print("\nchép log:")
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
        "# Kết quả",
        "",
        f"- thời điểm: {datetime.datetime.now():%Y-%m-%d %H:%M}",
        f"- checkpoint: `{args.checkpoint}`",
        f"- dataset: `{args.dataset_path}`",
        f"- split: `{args.split_path or 'mặc định (SPLIT_DIRS của loader)'}`",
        "",
    ]

    if args.seen_acc is not None and args.unseen_acc is not None:
        s, u = args.seen_acc, args.unseen_acc
        h = 0.0 if s + u == 0 else 2 * s * u / (s + u)
        lines += [
            "## Kết quả",
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
            "† Table 2 của paper LGD (GA++ đầy đủ, split gốc theo nhãn LVIS). Split ở đây "
            "là bản tái tạo protocol nên chỉ là mốc tham chiếu.",
            "",
            "Metric: success khi IoU ≥ 0.25 và lệch góc ≤ 30°.",
            "",
        ]

    lines += [
        "## Hình",
        "",
        "- `figures/prediction_{seen,unseen}_*.png` — grasp dự đoán (đỏ) cạnh ground "
        "truth (lục), kèm bản đồ Q, góc và `A_T`. Tiêu đề ghi IoU cao nhất và đạt/trượt "
        "theo đúng metric (IoU ≥ 0.25 và lệch góc ≤ 30°).",
        "- `figures/tokens_{seen,unseen}_*.png` — bản đồ alignment của từng token trong "
        "prompt, cạnh `part_mask` ground truth và `A_T` tổng hợp.",
        "- `figures/parts_same_object.png` — cùng một object, đổi part trong câu lệnh.",
        "- `figures/prompts_free_form.png` — prompt tự viết trên cùng một ảnh.",
        "",
        "### Dự đoán trên từng mẫu",
        "",
    ]
    lines += [f"- nguồn lấy mẫu `{k}`: {v}" for k, v in sample_note.items()]
    lines += [
        "",
        "| mẫu | prompt | grasp phát hiện | IoU cao nhất | kết quả |",
        "|---|---|---|---|---|",
    ]
    for key, info in pred_note.items():
        # 0 grasp nghĩa là Q không đâu vượt ngưỡng 0.2 của peak_local_max -- khác hẳn với
        # "có đoán nhưng đoán sai", nên tách ra.
        mark = (
            "đạt"
            if info["success"]
            else ("không đoán được" if not info["n_grasps"] else "trượt")
        )
        lines.append(
            f'| {key} | "{info["prompt"]}" | {info["n_grasps"]} | '
            f"{info['iou']:.3f} | {mark} |"
        )

    lines += ["", "### Token mạnh nhất theo từng hình", ""]
    for key, (prompt, ranked) in ranked_note.items():
        top = ", ".join(f"`{t}` {v:.2f}" for t, v in ranked[:4])
        lines.append(f'- **{key}** — "{prompt}" → {top}')

    path = os.path.join(args.out, "summary.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n{path}")


if __name__ == "__main__":
    main()
