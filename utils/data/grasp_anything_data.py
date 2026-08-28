import glob
import os
import pickle
import re

from utils.dataset_processing import grasp, image, mask
from .grasp_data import GraspDatasetBase

# Grasp files are named "<scene_id>_<object_idx>.pt". The object index is not limited
# to a single digit, so scenes with 10 or more objects must still match.
GRASP_SUFFIX_RE = re.compile(r"_\d+\.pt$")


class GraspAnythingDataset(GraspDatasetBase):
    """
    Dataset wrapper for the Grasp-Anything dataset.

    One image yields several samples -- one per annotated object -- so `grasp_files` is
    the single source of truth for indexing and every other path is derived from it.
    """

    def __init__(self, file_path, ds_rotate=0, **kwargs):
        """
        :param file_path: Grasp-Anything Dataset directory.
        :param ds_rotate: If splitting the dataset, rotate the list of items by this fraction first
        :param kwargs: kwargs for GraspDatasetBase
        """
        super(GraspAnythingDataset, self).__init__(**kwargs)

        self.file_path = file_path

        # Do NOT glob the image/scene_description folders into parallel lists: they do
        # not line up with `grasp_files` (the seen/unseen filter below applies to grasp
        # files only, and several grasp files share one image). Derive those paths from
        # `grasp_files[idx]` instead -- see `get_rgb_file` / `get_prompt_file`.
        self.grasp_files = glob.glob(os.path.join(file_path, 'grasp_label_positive', '*.pt'))

        split_file = 'seen.obj' if kwargs.get("seen", True) else 'unseen.obj'
        with open(os.path.join('split', 'grasp-anything', split_file), 'rb') as f:
            idxs = pickle.load(f)

        self.grasp_files = list(filter(lambda x: self._sample_id(x) in idxs, self.grasp_files))
        self.grasp_files.sort()

        self.length = len(self.grasp_files)

        if self.length == 0:
            raise FileNotFoundError('No dataset files found. Check path: {}'.format(file_path))

        if ds_rotate:
            self.grasp_files = self.grasp_files[int(self.length * ds_rotate):] + self.grasp_files[
                                                                                 :int(self.length * ds_rotate)]

    @staticmethod
    def _sample_id(grasp_file):
        """
        :return: "<scene_id>_<object_idx>", identifying one (image, object) pair.
        """
        return os.path.splitext(os.path.basename(grasp_file))[0]

    def _scene_id(self, idx):
        """
        :return: "<scene_id>", identifying the image shared by every object of a scene.
        """
        return GRASP_SUFFIX_RE.sub('', os.path.basename(self.grasp_files[idx]))

    def get_rgb_file(self, idx):
        return os.path.join(self.file_path, 'image', self._scene_id(idx) + '.jpg')

    def get_prompt_file(self, idx):
        return os.path.join(self.file_path, 'scene_description', self._scene_id(idx) + '.pkl')

    def _get_crop_attrs(self, idx):
        gtbbs = grasp.GraspRectangles.load_from_grasp_anything_file(self.grasp_files[idx])
        center = gtbbs.center
        left = max(0, min(center[1] - self.output_size // 2, 416 - self.output_size))
        top = max(0, min(center[0] - self.output_size // 2, 416 - self.output_size))
        return center, left, top

    def get_gtbb(self, idx, rot=0, zoom=1.0):
        # Jacquard try
        gtbbs = grasp.GraspRectangles.load_from_grasp_anything_file(self.grasp_files[idx], scale=self.output_size / 416.0)

        c = self.output_size // 2
        gtbbs.rotate(rot, (c, c))
        gtbbs.zoom(zoom, (c, c))

        # Cornell try
        # gtbbs = grasp.GraspRectangles.load_from_grasp_anything_file(self.grasp_files[idx])
        # center, left, top = self._get_crop_attrs(idx)
        # gtbbs.rotate(rot, center)
        # gtbbs.offset((-top, -left))
        # gtbbs.zoom(zoom, (self.output_size // 2, self.output_size // 2))
        return gtbbs

    def get_depth(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError(
            'Grasp-Anything ships RGB only; train with --use-depth 0.'
        )

    def get_rgb(self, idx, rot=0, zoom=1.0, normalise=True):
        # mask_file = self.grasp_files[idx].replace("grasp_label_positive", "mask").replace(".pt", ".npy")
        # mask_img = mask.Mask.from_file(mask_file)
        rgb_img = image.Image.from_file(self.get_rgb_file(idx))
        # rgb_img = image.Image.mask_out_image(rgb_img, mask_img)

        # Jacquard try
        rgb_img.rotate(rot)
        rgb_img.zoom(zoom)
        rgb_img.resize((self.output_size, self.output_size))
        if normalise:
            rgb_img.normalise()
            rgb_img.img = rgb_img.img.transpose((2, 0, 1))
        return rgb_img.img

        # Cornell try
        # center, left, top = self._get_crop_attrs(idx)
        # rgb_img.rotate(rot, center)
        # rgb_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
        # rgb_img.zoom(zoom)
        # rgb_img.resize((self.output_size, self.output_size))
        # if normalise:
        #     rgb_img.normalise()
        #     rgb_img.img = rgb_img.img.transpose((2, 0, 1))
        # return rgb_img.img
