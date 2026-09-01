"""Model có *thật sự* đọc prompt không? Đo bằng cách ghép ảnh với prompt sai.

Một model language-driven có thể đạt điểm cao mà hoàn toàn bỏ qua ngôn ngữ: GA++ có ~4,4
part cho mỗi scene, các vùng grasp của chúng chồng nhau, nên "grasp trung bình của ảnh" đã ăn
được kha khá theo metric IoU ≥ 0.25. Bảng Seen/Unseen/H một mình không phân biệt được hai
trường hợp đó.

Phép đối chứng giữ nguyên ảnh và ground truth, chỉ thay prompt. Bốn điều kiện, cùng model,
cùng sample:

    normal      prompt thật                                  -- kết quả chuẩn
    shuffled    prompt của một object khác                    -- có phụ thuộc ngôn ngữ không
    fixed       một câu duy nhất cho mọi ảnh                  -- có bỏ qua text hoàn toàn không
    other_part  prompt của part khác *cùng object*            -- có phân biệt được part không

Con số phải đọc là Δ_prompt = S_normal - S_shuffled:

* lớn   -> nhánh ngôn ngữ đang điều khiển grasp thật.
* ≈ 0   -> model đoán theo ảnh, phần ngôn ngữ chỉ là trang trí, bất kể align_loss thấp đến đâu.

`other_part` khó hơn `shuffled`: hai prompt cùng nói về một vật, chỉ khác part. Nếu Δ ở
`shuffled` lớn mà ở `other_part` ≈ 0 thì model mới chỉ học "nhận ra vật", chưa học part.

Kèm theo hai số đo grounding trên `A_T` (ở đúng resolution feature map, như lúc train):

* `IoU(A_T > 0.5, part_mask)` -- có khoanh đúng vùng part không
* `pointing` -- pixel `A_T` mạnh nhất có rơi vào part_mask không (không phụ thuộc ngưỡng)

    python script/audit_text_reliance.py --checkpoint logs/.../epoch_67_iou_0.3313 \\
        --dataset-path data/grasp-anything-pp-200k --split-path split/grasp-anything-pp \\
        --n-samples 500

Bổ trợ: `script/diagnose_part_masks.py` kiểm tra chính `part_mask` có ở mức part hay chỉ là
mask của cả object -- nếu nhãn đã ở mức object thì `L_align` không thể dạy grounding part-level
và bảng dưới đây cũng chỉ có thể ra Δ nhỏ.
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
                   help="Số sample mỗi split (lấy cách đều). 0 = toàn bộ.")
    p.add_argument("--iou-threshold", type=float, default=0.25)
    p.add_argument("--n-grasps", type=int, default=1)
    p.add_argument("--seed", type=int, default=0, help="Seed của phép hoán vị prompt")
    p.add_argument("--splits", nargs="+", default=["seen", "unseen"],
                   choices=["seen", "unseen"])
    p.add_argument("--fixed-prompt", default="Grasp the object.",
                   help="Câu dùng chung cho mọi ảnh ở điều kiện 'fixed'")
    p.add_argument("--out", default=None, help="Ghi kết quả JSON vào file này")
    p.add_argument("--figures", default=None,
                   help="Thư mục ghi gallery lỗi (bốn nhóm) cho mỗi split")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def load_dataset(args, seen):
    kwargs = dict(output_size=args.input_size, include_depth=False, include_rgb=True,
                  seen=seen)
    if args.split_path:
        kwargs["split_path"] = args.split_path
    return get_dataset("grasp-anything-pp")(args.dataset_path, **kwargs)


def grasp_metrics(net, x, prompt, gtbb, iou_threshold, n_grasps):
    """Chạy một forward với đúng một prompt, trả (max IoU, đạt/trượt, Q, A_T)."""
    with torch.no_grad():
        pred = net.predict(x, [prompt]) if getattr(net, "use_text", False) else net.predict(x)
    q_img, ang_img, width_img = post_process_output(
        pred["pos"], pred["cos"], pred["sin"], pred["width"])
    grasps = detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=n_grasps)
    # `Grasp.max_iou` trả 0 khi lệch góc > 30 độ, nên so với ngưỡng IoU là gói trọn cả hai
    # điều kiện của metric trong paper.
    best = max((g.max_iou(gtbb) for g in grasps), default=0.0)
    return float(best), bool(best > iou_threshold), pred


def align_metrics(pred, part_mask):
    """
    IoU và pointing-game của `A_T` so với `part_mask`, đo ở resolution của A_T (như lúc train).
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
    """Prompt của một part *khác* cùng object, hoặc None nếu object chỉ có một part."""
    object_id = ds._object_id(ds.sample_id(idx))
    for path in ds._files_by_object.get(object_id, []):
        other = ds.grasp_files.index(path)
        if other != idx:
            return ds.get_prompt(other, use_permutation=False)
    return None


def audit_split(net, ds, args, device):
    ds.shuffle_prompts(seed=args.seed)          # perm dùng chung cho cả split
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
    """Gallery lỗi bốn nhóm -- lỗi ở grounding hay ở grasp decoder."""
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
        print(f"  gallery lỗi: {path}")
    return counts


def print_table(results):
    print()
    header = f"{'split':8} {'điều kiện':11} {'n':>6} {'success':>9} {'IoU':>8} " \
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

    print("Δ_shuffled ≈ 0: grasp không phụ thuộc prompt -- không kết luận được nhánh ngôn ngữ")
    print("có ích, dù align_loss thấp. Δ_shuffled lớn nhưng Δ_other_part ≈ 0: model nhận ra")
    print("*vật* chứ chưa phân biệt *part*. Trước khi quy lỗi cho model, chạy")
    print("script/check_dataset.py xem part_mask có thật sự ở mức part không.")


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
