import torch
from skimage.filters import gaussian


def post_process_output(q_img, cos_img, sin_img, width_img, width_scale=None):
    """
    Post-process the raw output of the network, convert to numpy arrays, apply filtering.
    :param q_img: Q output of network (as torch Tensors)
    :param cos_img: cos output of network
    :param sin_img: sin output of network
    :param width_img: Width output of network
    :param width_scale: Width de-normalisation constant. None = infer it from the image size.
    :return: Filtered Q output, Filtered Angle output, Filtered Width output
    """
    # Width labels are normalised by `output_size / 2` (utils/data/grasp_data.py), so decoding
    # must use that same constant. The old constant 150 is `output_size / 2` of the original
    # 300x300 GR-ConvNet; keeping it at input 224 makes every grasp 150/112 = 1.34x too long.
    if width_scale is None:
        width_scale = width_img.shape[-1] / 2.0

    q_img = q_img.cpu().numpy().squeeze()
    ang_img = (torch.atan2(sin_img, cos_img) / 2.0).cpu().numpy().squeeze()
    width_img = width_img.cpu().numpy().squeeze() * width_scale

    q_img = gaussian(q_img, 2.0, preserve_range=True)
    ang_img = gaussian(ang_img, 2.0, preserve_range=True)
    width_img = gaussian(width_img, 1.0, preserve_range=True)

    return q_img, ang_img, width_img
