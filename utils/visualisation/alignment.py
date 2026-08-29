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
