import torch
from skimage.filters import gaussian


def post_process_output(q_img, cos_img, sin_img, width_img, width_scale=None):
    """
    Post-process the raw output of the network, convert to numpy arrays, apply filtering.
    :param q_img: Q output of network (as torch Tensors)
    :param cos_img: cos output of network
    :param sin_img: sin output of network
    :param width_img: Width output of network
    :param width_scale: Hằng số giải chuẩn hoá width. None = suy ra từ kích thước ảnh.
    :return: Filtered Q output, Filtered Angle output, Filtered Width output
    """
    # Nhãn width được chuẩn hoá bằng `output_size / 2` (utils/data/grasp_data.py), nên decode
    # phải dùng đúng hằng số đó. Hằng 150 cũ chính là `output_size / 2` của GR-ConvNet gốc
    # (300x300); giữ nguyên nó với input 224 làm mọi grasp dài hơn thật 150/112 = 1.34 lần.
    if width_scale is None:
        width_scale = width_img.shape[-1] / 2.0

    q_img = q_img.cpu().numpy().squeeze()
    ang_img = (torch.atan2(sin_img, cos_img) / 2.0).cpu().numpy().squeeze()
    width_img = width_img.cpu().numpy().squeeze() * width_scale

    q_img = gaussian(q_img, 2.0, preserve_range=True)
    ang_img = gaussian(ang_img, 2.0, preserve_range=True)
    width_img = gaussian(width_img, 1.0, preserve_range=True)

    return q_img, ang_img, width_img
