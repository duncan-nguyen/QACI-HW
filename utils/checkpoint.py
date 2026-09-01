"""Lưu / nạp checkpoint mà không kéo theo cả CLIP text tower.

Repo gốc lưu bằng `torch.save(net)` -- pickle nguyên object. Với `grconvnet3_align` thì object
đó chứa luôn CLIP text encoder (63M tham số đóng băng), nên mỗi checkpoint nặng ~265 MB dù
phần *học được* chỉ ~2 MB. Train 100 epoch và lưu mỗi lần cải thiện là hàng chục GB cho một
thứ tải lại được từ HuggingFace trong vài giây.

Định dạng mới là dict thuần:

    {'format': 2, 'network': 'grconvnet3_align', 'kwargs': {...}, 'state_dict': {...}}

`state_dict` bỏ mọi key `text_encoder.*`; lúc load thì `CLIPTextEncoder` được dựng lại từ
cache HuggingFace. Model nào không tự mô tả được (`init_kwargs`) thì vẫn pickle như cũ, nên
cornell/jacquard/ggcnn... không đổi hành vi.

`load_network` đọc được cả hai định dạng, kể cả checkpoint V1 pickle từ code cũ.
"""

import os

import torch

FORMAT_VERSION = 2
_TEXT_PREFIX = "text_encoder."


def is_state_dict_checkpoint(obj):
    return isinstance(obj, dict) and "state_dict" in obj and "network" in obj


def checkpoint_state(net):
    """
    Dict tự mô tả của model, hoặc None nếu model không khai báo `init_kwargs`/`network_name`.
    """
    name = getattr(net, "network_name", None)
    kwargs = getattr(net, "init_kwargs", None)
    if not name or not kwargs:
        return None
    state = {
        k: v for k, v in net.state_dict().items() if not k.startswith(_TEXT_PREFIX)
    }
    return {
        "format": FORMAT_VERSION,
        "network": name,
        "kwargs": dict(kwargs),
        "state_dict": state,
    }


def save_checkpoint(net, path):
    """
    Ghi checkpoint (atomic: file tạm rồi `os.replace`, để Ctrl-C giữa chừng không để lại file cụt).

    :return: đường dẫn đã ghi
    """
    payload = checkpoint_state(net)
    if payload is None:
        payload = net  # fallback: pickle cả module như repo gốc
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def load_network(path, map_location="cpu", text_encoder=None, strict=True):
    """
    Nạp checkpoint ở *bất kỳ* định dạng nào của repo này.

    :param text_encoder: dùng lại một CLIPTextEncoder có sẵn thay vì dựng bản mới (hữu ích khi
                         đánh giá nhiều checkpoint trong cùng một tiến trình).
    :param strict: báo lỗi nếu state_dict thiếu/thừa key ngoài phần `text_encoder.*`.
    :return: nn.Module ở chế độ eval()
    """
    try:
        obj = torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        # Checkpoint cũ là module đã pickle -> bắt buộc weights_only=False. Ta chỉ load file
        # do chính mình tạo ra nên chấp nhận được.
        obj = torch.load(path, map_location=map_location, weights_only=False)

    if not is_state_dict_checkpoint(obj):
        return obj.eval()

    from inference.models import get_network

    kwargs = dict(obj["kwargs"])
    if text_encoder is not None and kwargs.get("use_text", False):
        kwargs["text_encoder"] = text_encoder
    net = get_network(obj["network"])(**kwargs)
    missing, unexpected = net.load_state_dict(obj["state_dict"], strict=False)

    missing = [k for k in missing if not k.startswith(_TEXT_PREFIX)]
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"state_dict không khớp model dựng từ kwargs của checkpoint {path}:\n"
            f"  thiếu:  {missing}\n  thừa:   {list(unexpected)}"
        )
    if isinstance(map_location, (str, torch.device)):
        net = net.to(map_location)
    return net.eval()
