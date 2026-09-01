import copy
import random

import numpy as np
import torch
import torch.utils.data


class GraspDatasetBase(torch.utils.data.Dataset):
    """
    An abstract dataset for training networks in a common format.
    """

    def __init__(
        self,
        output_size=224,
        include_depth=True,
        include_rgb=False,
        random_rotate=False,
        random_zoom=False,
        input_only=False,
        seen=True,
    ):
        """
        :param output_size: Image output size in pixels (square)
        :param include_depth: Whether depth image is included
        :param include_rgb: Whether RGB image is included
        :param random_rotate: Whether random rotations are applied
        :param random_zoom: Whether random zooms are applied
        :param input_only: Whether to return only the network input (no labels)
        """
        self.output_size = output_size
        self.random_rotate = random_rotate
        self.random_zoom = random_zoom
        self.input_only = input_only
        self.include_depth = include_depth
        self.include_rgb = include_rgb

        self.grasp_files = []

        if include_depth is False and include_rgb is False:
            raise ValueError("At least one of Depth or RGB must be specified.")

    def eval_view(self):
        """
        A *shallow* copy of the dataset with augmentation disabled, for validation/test.

        Previously `train_network.py` handed the same dataset (random_rotate/random_zoom on) to
        both training and validation. With `zoom ~ U(0.5, 1.0)` the crop cuts away up to half
        the frame, and `get_gtbb` does not drop grasps that fall outside -- the target ends up
        empty and the sample automatically counts as a failure. The checkpoint-selection metric
        was therefore measured on corrupted data.

        The shallow copy (not `deepcopy`) is deliberate: `grasp_files` and the auxiliary index
        tables are shared by reference, so it costs no extra memory and no re-globbing of 880k
        files, while each DataLoader worker still sees the copy's own augmentation flags.
        """
        view = copy.copy(self)
        view.random_rotate = False
        view.random_zoom = False
        return view

    @staticmethod
    def numpy_to_torch(s):
        if len(s.shape) == 2:
            return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
        else:
            return torch.from_numpy(s.astype(np.float32))

    def get_gtbb(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()

    def get_depth(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()

    def get_rgb(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()

    def __getitem__(self, idx):
        if self.random_rotate:
            rotations = [0, np.pi / 2, 2 * np.pi / 2, 3 * np.pi / 2]
            # The `float(...)` is not redundant: the first element of `rotations` is an *int*
            # 0, so when it lands first in a batch `default_collate` takes the int branch and
            # builds the tensor as `torch.tensor([0, 1.5707963267948966, ...])` -> float32,
            # losing 8 digits of the other angles. At batch_size = 1 the bug is invisible; once
            # validation is batched, the `rot` used to build the GT drifts from the `rot` the
            # loader used to rotate the image.
            rot = float(random.choice(rotations))
        else:
            rot = 0.0

        if self.random_zoom:
            zoom_factor = float(np.random.uniform(0.5, 1.0))
        else:
            zoom_factor = 1.0

        # Load the depth image
        if self.include_depth:
            depth_img = self.get_depth(idx, rot, zoom_factor)

        # Load the RGB image
        if self.include_rgb:
            rgb_img = self.get_rgb(idx, rot, zoom_factor)

        # Load the grasps
        bbs = self.get_gtbb(idx, rot, zoom_factor)

        pos_img, ang_img, width_img = bbs.draw((self.output_size, self.output_size))
        width_img = np.clip(width_img, 0.0, self.output_size / 2) / (
            self.output_size / 2
        )

        if self.include_depth and self.include_rgb:
            x = self.numpy_to_torch(
                np.concatenate((np.expand_dims(depth_img, 0), rgb_img), 0)
            )
        elif self.include_depth:
            x = self.numpy_to_torch(depth_img)
        elif self.include_rgb:
            x = self.numpy_to_torch(rgb_img)

        pos = self.numpy_to_torch(pos_img)
        cos = self.numpy_to_torch(np.cos(2 * ang_img))
        sin = self.numpy_to_torch(np.sin(2 * ang_img))
        width = self.numpy_to_torch(width_img)

        return x, (pos, cos, sin, width), idx, rot, zoom_factor

    def __len__(self):
        return len(self.grasp_files)
