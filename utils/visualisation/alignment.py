"""Vẽ bản đồ alignment token-pixel của GenerativeResnetAlign.

Ý tưởng cốt lõi của method là *token văn bản nào* ăn với *vùng ảnh nào*. Module này lấy
`net.token_alignment(...)` rồi trải ra thành một hàng hình: ảnh gốc, part_mask ground truth,
A_T tổng hợp, và một ô cho mỗi token nội dung của prompt.
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")


def _to_hw(tensor, size):
    """(H, W) hoặc (1, H, W) -> ndarray (size, size) đã upsample bilinear."""
    t = tensor.detach().float()
    while t.dim() < 4:
        t = t.unsqueeze(0)
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t[0, 0].cpu().numpy()


def _rgb_for_show(rgb):
    """Ảnh của dataset đã normalise (zero-centred float) -> đưa về [0, 1] để imshow."""
    img = rgb.detach().cpu().numpy() if torch.is_tensor(rgb) else np.asarray(rgb)
    if img.ndim == 3 and img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    img = img.astype(np.float32)
    lo, hi = img.min(), img.max()
    return (img - lo) / (hi - lo + 1e-8)


def plot_token_alignment(net, x, prompt, part_mask=None, max_tokens=8, size=224, axes=None):
    """
    Vẽ alignment của một sample.

    :param net: GenerativeResnetAlign (đã .eval())
    :param x: (3, H, W) hoặc (1, 3, H, W) -- input đúng như dataset trả về
    :param prompt: câu lệnh gắp (str)
    :param part_mask: (1, H, W) ground truth, optional
    :param max_tokens: số token vẽ nhiều nhất (chọn theo activation cực đại giảm dần)
    :param axes: list axes có sẵn; None thì tự tạo figure
    :return: (fig, thứ tự token đã vẽ kèm activation cực đại)
    """
    if x.dim() == 3:
        x = x.unsqueeze(0)
    device = next(net.parameters()).device
    out = net.token_alignment(x.to(device), [prompt])

    tokens, maps, attention = out["tokens"][0], out["maps"][0], out["attention"][0]
    # Token nào "ăn" mạnh nhất thì vẽ trước -- đó chính là thứ ta muốn nhìn.
    peaks = maps.flatten(1).max(dim=1).values
    order = torch.argsort(peaks, descending=True)[:max_tokens].tolist()

    panels = [("image", None, None), ("A_T", _to_hw(attention, size), "jet")]
    if part_mask is not None:
        # GT là nhị phân -> phủ đặc vùng part thay vì heatmap, nếu không nó chìm hẳn.
        panels.insert(1, ("part_mask (GT)", _to_hw(part_mask, size), "mask"))
    panels += [(f"{tokens[i]}  ({peaks[i]:.2f})", _to_hw(maps[i], size), "jet") for i in order]

    if axes is None:
        ncol = min(len(panels), 6)
        nrow = int(np.ceil(len(panels) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(2.4 * ncol, 2.6 * nrow))
        axes = np.atleast_1d(axes).ravel().tolist()
    else:
        fig = axes[0].figure

    rgb = _rgb_for_show(x[0])
    for ax, (title, heat, style) in zip(axes, panels):
        ax.imshow(rgb)
        if style == "mask":
            ax.imshow(np.ma.masked_where(heat <= 0.5, heat), cmap="autumn", alpha=0.65,
                      vmin=0.0, vmax=1.0)
        elif heat is not None:
            ax.imshow(heat, cmap="jet", alpha=0.5, vmin=0.0, vmax=1.0)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    for ax in axes[len(panels):]:
        ax.axis("off")

    fig.suptitle(f'"{prompt}"', fontsize=11)
    fig.tight_layout()
    return fig, [(tokens[i], float(peaks[i])) for i in order]


def plot_prompt_comparison(net, x, prompts, size=224):
    """
    Cùng một ảnh, đổi prompt -> A_T phải dịch chuyển. Đây là cách nhìn trực tiếp nhất xem
    nhánh ngôn ngữ có thật sự điều khiển vùng chú ý hay không.
    """
    if x.dim() == 3:
        x = x.unsqueeze(0)
    device = next(net.parameters()).device
    rgb = _rgb_for_show(x[0])

    fig, axes = plt.subplots(1, len(prompts) + 1, figsize=(2.6 * (len(prompts) + 1), 3.0))
    axes[0].imshow(rgb)
    axes[0].set_title("image", fontsize=9)
    axes[0].axis("off")

    for ax, prompt in zip(axes[1:], prompts):
        out = net.token_alignment(x.to(device), [prompt])
        ax.imshow(rgb)
        ax.imshow(_to_hw(out["attention"][0], size), cmap="jet", alpha=0.5, vmin=0.0, vmax=1.0)
        ax.set_title("\n".join(_wrap(prompt, 22)), fontsize=8)
        ax.axis("off")

    fig.tight_layout()
    return fig


def plot_grasp_prediction(net, dataset, idx, no_grasps=1, iou_threshold=0.25, axes=None):
    """
    Grasp dự đoán cạnh ground truth, cùng bản đồ Q và A_T.

    Success được tính đúng như `evaluation.calculate_iou_match`: `Grasp.max_iou` trả 0 khi lệch
    góc quá 30 độ, nên `max_iou > 0.25` gói trọn cả hai điều kiện của metric trong paper.

    :param dataset: GraspAnythingPPDataset
    :param idx: index trong dataset
    :return: (fig, dict với iou/success/prompt)
    """
    from inference.post_process import post_process_output
    from utils.dataset_processing.grasp import detect_grasps

    x, _, _, rot, zoom, extra = dataset[idx]
    prompt = extra.get("prompt")
    device = next(net.parameters()).device

    with torch.no_grad():
        xc = x.unsqueeze(0).to(device)
        pred = net.predict(xc, [prompt]) if getattr(net, "use_text", False) else net.predict(xc)
    q_img, ang_img, width_img = post_process_output(
        pred["pos"], pred["cos"], pred["sin"], pred["width"])

    grasps = detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=no_grasps)
    gt = dataset.get_gtbb(idx, rot, zoom)
    best_iou = max((g.max_iou(gt) for g in grasps), default=0.0)
    success = best_iou > iou_threshold

    n_panel = 3 + int("align" in pred)
    if axes is None:
        fig, axes = plt.subplots(1, n_panel, figsize=(3.0 * n_panel, 3.4))
        axes = np.atleast_1d(axes).ravel().tolist()
    else:
        fig = axes[0].figure

    rgb = _rgb_for_show(x)
    axes[0].imshow(rgb)
    for gr in gt:
        gr.plot(axes[0], color="lime")
    for g in grasps:
        g.plot(axes[0], color="red")
    axes[0].set_title(f"GT (lục) vs dự đoán (đỏ)\nIoU={best_iou:.2f} "
                      f"{'ĐẠT' if success else 'TRƯỢT'}", fontsize=9)

    axes[1].imshow(rgb)
    axes[1].imshow(q_img, cmap="jet", alpha=0.5, vmin=0.0, vmax=1.0)
    axes[1].set_title("Q (quality)", fontsize=9)

    axes[2].imshow(ang_img, cmap="hsv", vmin=-np.pi / 2, vmax=np.pi / 2)
    axes[2].set_title("góc", fontsize=9)

    if "align" in pred:
        axes[3].imshow(rgb)
        axes[3].imshow(_to_hw(pred["align"][0], rgb.shape[0]), cmap="jet", alpha=0.5,
                       vmin=0.0, vmax=1.0)
        axes[3].set_title("A_T", fontsize=9)

    for ax in axes[:n_panel]:
        ax.axis("off")
    if prompt:
        fig.suptitle(f'"{prompt}"', fontsize=11)
    fig.tight_layout()
    return fig, {"iou": float(best_iou), "success": bool(success), "prompt": prompt,
                 "n_grasps": len(grasps)}


def plot_part_prompts(net, dataset, indices, size=224):
    """
    Cùng một object, nhiều part khác nhau -- mỗi hàng một prompt: ground truth cạnh `A_T`.

    Đây là hình nói đúng luận điểm của method: đổi part trong câu lệnh thì vùng chú ý phải
    dịch theo, dù ảnh không đổi.

    :param dataset: GraspAnythingPPDataset
    :param indices: các index cùng một object (xem `dataset._files_by_object`)
    """
    device = next(net.parameters()).device
    rows = len(indices)
    fig, axes = plt.subplots(rows, 3, figsize=(8.0, 2.7 * rows), squeeze=False)

    for r, idx in enumerate(indices):
        x, _, _, rot, zoom, extra = dataset[idx]
        rgb = _rgb_for_show(x)
        att = net.token_alignment(x.unsqueeze(0).to(device), [extra["prompt"]])["attention"][0]

        axes[r][0].imshow(rgb)
        axes[r][0].set_title(dataset.sample_id(idx).rsplit("_", 2)[-1:] and
                             f"part {dataset.sample_id(idx).rsplit('_', 1)[1]}", fontsize=9)

        gt = _to_hw(extra["part_mask"], size)
        axes[r][1].imshow(rgb)
        axes[r][1].imshow(np.ma.masked_where(gt <= 0.5, gt), cmap="autumn", alpha=0.65)
        axes[r][1].set_title("part_mask (GT)", fontsize=9)

        axes[r][2].imshow(rgb)
        axes[r][2].imshow(_to_hw(att, size), cmap="jet", alpha=0.5, vmin=0.0, vmax=1.0)
        axes[r][2].set_title("A_T", fontsize=9)

        axes[r][0].set_ylabel("\n".join(_wrap(extra["prompt"], 26)), fontsize=8)
        for ax in axes[r]:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.tight_layout()
    return fig


def _wrap(text, width):
    lines, line = [], ""
    for word in text.split():
        if len(line) + len(word) + 1 > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    return lines + [line] if line else lines


# ---------------------------------------------------------------- V2 debug --
# Ba câu hỏi khi soi một run: token nào được chọn, vùng chọn có đúng không, và grasp có đổi
# theo prompt không. Các hàm dưới đây trả lời lần lượt từng câu bằng hình.

TP_COLOR, FP_COLOR, FN_COLOR = (0.15, 0.85, 0.25), (0.90, 0.15, 0.15), (0.95, 0.85, 0.10)


def _error_overlay(pred, gt):
    """
    Mã màu sai/đúng của A_T: xanh = trúng, đỏ = báo nhầm, vàng = bỏ sót.

    :param pred, gt: (H, W) ndarray nhị phân cùng kích thước
    :return: (rgba (H, W, 4), chú thích)
    """
    pred, gt = pred > 0.5, gt > 0.5
    rgba = np.zeros(pred.shape + (4,), dtype=np.float32)
    for mask, color in ((pred & gt, TP_COLOR), (pred & ~gt, FP_COLOR), (~pred & gt, FN_COLOR)):
        rgba[mask, :3] = color
        rgba[mask, 3] = 0.75
    inter, union = (pred & gt).sum(), (pred | gt).sum()
    return rgba, f"IoU={inter / union:.2f}" if union else "mask rỗng"


def _top_tokens(out, b=0, k=4):
    """
    Token mạnh nhất theo *attention mass* (trung bình alpha trên toàn ảnh), không phải theo
    đỉnh similarity: mass mới là thứ quyết định R_T và là thứ so sánh được giữa các token.

    :return: list[(token, mass, map (H, W))]
    """
    tokens, weights, maps = out["tokens"][b], out["weights"][b], out["maps"][b]
    mass = weights.flatten(1).mean(dim=1)
    order = torch.argsort(mass, descending=True)[:k].tolist()
    return [(tokens[i], float(mass[i]), maps[i]) for i in order]


def _predict(net, x, prompt):
    """Một forward -> (pred dict, q_img, ang_img, width_img)."""
    from inference.post_process import post_process_output

    device = next(net.parameters()).device
    with torch.no_grad():
        xc = x.unsqueeze(0).to(device) if x.dim() == 3 else x.to(device)
        pred = net.predict(xc, [prompt]) if getattr(net, "use_text", False) else net.predict(xc)
    q, ang, width = post_process_output(pred["pos"], pred["cos"], pred["sin"], pred["width"])
    return pred, q, ang, width


def _draw_grasps(ax, rgb, gt, grasps, title):
    ax.imshow(rgb)
    for gr in gt:
        gr.plot(ax, color="lime")
    for g in grasps:
        g.plot(ax, color="red")
    ax.set_title(title, fontsize=9)


def plot_sample_diagnostics(net, dataset, idx, prompt=None, k_tokens=4, size=224,
                            iou_threshold=0.25):
    """
    Một hàng đầy đủ cho *một* sample:

        RGB | part_mask GT | A_T | sai số alignment | Q | GT+dự đoán | k token mạnh nhất

    Đây là hình để trả lời "lỗi nằm ở grounding hay ở grasp decoder": nếu A_T trúng mask mà
    grasp vẫn trượt thì decoder là chỗ hỏng, không phải nhánh ngôn ngữ.

    :param prompt: None = dùng prompt thật của sample
    :return: (fig, dict thông tin)
    """
    from utils.dataset_processing.grasp import detect_grasps

    x, _, _, rot, zoom, extra = dataset[idx]
    prompt = prompt if prompt is not None else extra["prompt"]
    pred, q_img, ang_img, width_img = _predict(net, x, prompt)

    grasps = detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=1)
    gt = dataset.get_gtbb(idx, rot, zoom)
    best_iou = max((g.max_iou(gt) for g in grasps), default=0.0)
    success = best_iou > iou_threshold

    gt_mask = _to_hw(extra["part_mask"], size)
    has_align = "align" in pred
    at = _to_hw(pred["align"][0], size) if has_align else None

    tokens = []
    if has_align and getattr(net, "use_text", False):
        tokens = _top_tokens(net.token_alignment(
            x.unsqueeze(0).to(next(net.parameters()).device), [prompt]), k=k_tokens)

    panels = 6 + len(tokens)
    fig, axes = plt.subplots(1, panels, figsize=(2.5 * panels, 3.1))
    axes = np.atleast_1d(axes).ravel().tolist()
    rgb = _rgb_for_show(x)

    axes[0].imshow(rgb)
    axes[0].set_title("RGB", fontsize=9)

    axes[1].imshow(rgb)
    axes[1].imshow(np.ma.masked_where(gt_mask <= 0.5, gt_mask), cmap="autumn", alpha=0.65)
    axes[1].set_title(f"part_mask GT ({gt_mask.mean():.1%})", fontsize=9)

    if has_align:
        axes[2].imshow(rgb)
        axes[2].imshow(at, cmap="jet", alpha=0.5, vmin=0.0, vmax=1.0)
        axes[2].set_title(f"A_T (max {at.max():.2f})", fontsize=9)

        overlay, note = _error_overlay(at, gt_mask)
        axes[3].imshow(rgb)
        axes[3].imshow(overlay)
        axes[3].set_title(f"sai số: xanh TP / đỏ FP / vàng FN\n{note}", fontsize=8)
    else:
        for ax in axes[2:4]:
            ax.imshow(rgb)
            ax.set_title("(không có nhánh text)", fontsize=9)

    axes[4].imshow(rgb)
    axes[4].imshow(q_img, cmap="jet", alpha=0.5, vmin=0.0, vmax=1.0)
    axes[4].set_title(f"Q (max {q_img.max():.2f})", fontsize=9)

    _draw_grasps(axes[5], rgb, gt, grasps,
                 f"GT (lục) / dự đoán (đỏ)\nIoU={best_iou:.2f} {'ĐẠT' if success else 'TRƯỢT'}")

    for ax, (token, mass, tmap) in zip(axes[6:], tokens):
        ax.imshow(rgb)
        ax.imshow(_to_hw(tmap, size), cmap="jet", alpha=0.5, vmin=0.0, vmax=1.0)
        ax.set_title(f"{token} — {mass:.2f}", fontsize=9)

    for ax in axes:
        ax.axis("off")
    fig.suptitle(f'"{prompt}"', fontsize=11)
    fig.tight_layout()
    return fig, {"iou": float(best_iou), "success": bool(success), "prompt": prompt,
                 "n_grasps": len(grasps),
                 "tokens": [(t, m) for t, m, _ in tokens]}


def plot_prompt_grid(net, dataset, idx, prompts, size=224, k_tokens=3):
    """
    Cùng một ảnh, nhiều prompt -- hình quan trọng nhất của idea này.

    Mỗi hàng: prompt | part_mask GT | A_T | top token | Q | grasp. Nếu đổi prompt mà A_T, top
    token và grasp gần như không đổi thì model đang bỏ qua ngôn ngữ, bất kể align_loss thấp
    đến đâu.

    :param prompts: list[(nhãn, prompt)] hoặc list[str]
    """
    from utils.dataset_processing.grasp import detect_grasps

    prompts = [(p, p) if isinstance(p, str) else p for p in prompts]
    x, _, _, rot, zoom, extra = dataset[idx]
    rgb = _rgb_for_show(x)
    gt = dataset.get_gtbb(idx, rot, zoom)
    gt_mask = _to_hw(extra["part_mask"], size)
    device = next(net.parameters()).device

    fig, axes = plt.subplots(len(prompts), 5, figsize=(13.5, 2.8 * len(prompts)), squeeze=False)
    for r, (label, prompt) in enumerate(prompts):
        pred, q_img, ang_img, width_img = _predict(net, x, prompt)
        grasps = detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=1)

        axes[r][0].imshow(rgb)
        axes[r][0].imshow(np.ma.masked_where(gt_mask <= 0.5, gt_mask), cmap="autumn", alpha=0.6)
        axes[r][0].set_title("part_mask GT" if r == 0 else "", fontsize=9)
        axes[r][0].set_ylabel("\n".join(_wrap(label, 24)), fontsize=8)

        if "align" in pred:
            axes[r][1].imshow(rgb)
            axes[r][1].imshow(_to_hw(pred["align"][0], size), cmap="jet", alpha=0.5,
                              vmin=0.0, vmax=1.0)
        axes[r][1].set_title("A_T" if r == 0 else "", fontsize=9)

        tokens = _top_tokens(net.token_alignment(x.unsqueeze(0).to(device), [prompt]),
                             k=k_tokens) if getattr(net, "use_text", False) else []
        axes[r][2].axis("off")
        axes[r][2].text(0.02, 0.95, "\n".join(f"{t:<10s} {m:.2f}" for t, m, _ in tokens),
                        family="monospace", fontsize=9, va="top")
        axes[r][2].set_title("token mạnh nhất" if r == 0 else "", fontsize=9)

        axes[r][3].imshow(rgb)
        axes[r][3].imshow(q_img, cmap="jet", alpha=0.5, vmin=0.0, vmax=1.0)
        axes[r][3].set_title("Q" if r == 0 else "", fontsize=9)

        best = max((g.max_iou(gt) for g in grasps), default=0.0)
        _draw_grasps(axes[r][4], rgb, gt, grasps,
                     ("grasp  " if r == 0 else "") + f"IoU={best:.2f}")

        for c, ax in enumerate(axes[r]):
            ax.set_xticks([])
            ax.set_yticks([])
            if c != 2:
                ax.set_frame_on(False)

    fig.tight_layout()
    return fig


FAILURE_CLASSES = ("align đúng / grasp đúng", "align đúng / grasp sai",
                   "align sai / grasp sai", "không phát hiện grasp")


def classify_failure(align_iou, grasp_iou, n_grasps, align_threshold=0.25,
                     grasp_threshold=0.25):
    """Bốn nhóm để biết lỗi nằm ở grounding hay ở grasp decoder."""
    if n_grasps == 0:
        return 3
    if grasp_iou > grasp_threshold:
        return 0 if align_iou > align_threshold else 2
    return 1 if align_iou > align_threshold else 2


def plot_failure_gallery(net, dataset, indices, per_class=2, size=224, iou_threshold=0.25):
    """
    Gallery lỗi chia bốn nhóm (xem FAILURE_CLASSES).

    Nhóm 2 nhiều -> nhánh grounding hỏng. Nhóm 1 nhiều -> grounding ổn mà decoder không dùng
    được nó; đó là hai hướng sửa hoàn toàn khác nhau.

    :return: (fig hoặc None, đếm theo nhóm)
    """
    from utils.dataset_processing.grasp import detect_grasps

    buckets = {i: [] for i in range(len(FAILURE_CLASSES))}
    counts = {i: 0 for i in range(len(FAILURE_CLASSES))}

    for idx in indices:
        x, _, _, rot, zoom, extra = dataset[idx]
        prompt = extra["prompt"]
        pred, q_img, ang_img, width_img = _predict(net, x, prompt)
        grasps = detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=1)
        gt = dataset.get_gtbb(idx, rot, zoom)
        grasp_iou = max((g.max_iou(gt) for g in grasps), default=0.0)

        gt_mask = _to_hw(extra["part_mask"], size)
        align_iou = 0.0
        if "align" in pred:
            at = _to_hw(pred["align"][0], size) > 0.5
            union = (at | (gt_mask > 0.5)).sum()
            align_iou = float((at & (gt_mask > 0.5)).sum()) / float(union) if union else 0.0

        cls = classify_failure(align_iou, grasp_iou, len(grasps), grasp_threshold=iou_threshold)
        counts[cls] += 1
        if len(buckets[cls]) < per_class:
            buckets[cls].append((idx, prompt, x, gt, grasps, q_img, pred, gt_mask,
                                 align_iou, grasp_iou))

    rows = [(c, item) for c in buckets for item in buckets[c]]
    if not rows:
        return None, counts

    fig, axes = plt.subplots(len(rows), 3, figsize=(8.5, 2.8 * len(rows)), squeeze=False)
    for r, (cls, item) in enumerate(rows):
        _, prompt, x, gt, grasps, q_img, pred, gt_mask, a_iou, g_iou = item
        rgb = _rgb_for_show(x)

        axes[r][0].imshow(rgb)
        axes[r][0].imshow(np.ma.masked_where(gt_mask <= 0.5, gt_mask), cmap="autumn", alpha=0.6)
        axes[r][0].set_title(f"[{cls + 1}] {FAILURE_CLASSES[cls]}", fontsize=9)
        axes[r][0].set_ylabel("\n".join(_wrap(prompt, 24)), fontsize=8)

        if "align" in pred:
            overlay, note = _error_overlay(_to_hw(pred["align"][0], size), gt_mask)
            axes[r][1].imshow(rgb)
            axes[r][1].imshow(overlay)
            axes[r][1].set_title(f"A_T vs GT — {note}", fontsize=9)

        _draw_grasps(axes[r][2], rgb, gt, grasps, f"IoU={g_iou:.2f}")
        for ax in axes[r]:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle("gallery lỗi: " + " · ".join(
        f"[{i + 1}] {FAILURE_CLASSES[i]} {counts[i]}" for i in counts), fontsize=10)
    fig.tight_layout()
    return fig, counts


def plot_dataset_grid(dataset, indices, size=224, ncol=5):
    """
    Lưới ảnh + prompt + part_mask + grasp GT -- kiểm tra *dữ liệu* trước khi train.

    Sai lệch dễ thấy nhất bằng mắt ở đây: mask lệch khỏi vật, mask phủ cả object thay vì một
    part, hoặc grasp GT không nằm trong mask.
    """
    nrow = int(np.ceil(len(indices) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.9 * ncol, 3.2 * nrow), squeeze=False)
    axes = axes.ravel()

    for ax, idx in zip(axes, indices):
        x, _, _, rot, zoom, extra = dataset[idx]
        rgb = _rgb_for_show(x)
        mask = _to_hw(extra["part_mask"], size)
        ax.imshow(rgb)
        ax.imshow(np.ma.masked_where(mask <= 0.5, mask), cmap="autumn", alpha=0.55)
        for gr in dataset.get_gtbb(idx, rot, zoom):
            gr.plot(ax, color="lime")
        ax.set_title("\n".join(_wrap(extra["prompt"], 30)) + f"\nfg={mask.mean():.1%}",
                     fontsize=7)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    return fig
