import glob
import os
import pickle
from collections import defaultdict

import numpy as np

from utils.dataset_processing import grasp, image

from .grasp_data import GraspDatasetBase

# Grasp-Anything images and part_masks are always 416x416.
SOURCE_SIZE = 416

# Search order when no split is given: the GA++-specific split first, otherwise fall back to
# the object-level split of the base GA.
SPLIT_DIRS = (
    os.path.join("split", "grasp-anything-pp"),
    os.path.join("split", "grasp-anything"),
)


class GraspAnythingPPDataset(GraspDatasetBase):
    """
    Dataset wrapper for Grasp-Anything++ (language-driven grasping).

    Three differences from the original Grasp-Anything:

    * a sample is (scene, object, part), with filename `<scene_id>_<object_idx>_<part_idx>` --
      *three* parts, not two. `scene_id` is the SHA-256 of the image, so it contains no '_' and
      stripping suffixes with `rsplit` is safe.
    * prompts live in `grasp_instructions/`, one `str` per file ("Lift apple by its skin.").
      This is not the base GA's `scene_description/`, which is a tuple (caption, [object names])
      describing the whole scene and targeting no particular part.
    * there is an extra `part_mask/`: a 416x416 uint8 binary mask of exactly the part the
      prompt refers to.

    The `grasp_label_positive/*.pt` layout is identical to the base GA -- float32 (N, 6), each
    row `[q, x, y, w, h, theta_deg]` -- so `GraspRectangles.load_from_grasp_anything_file` is
    reused as is. The `q` column is the antipodal score T~ = (cos a1 + cos a2) / R of the LGD
    paper (Sec. 3.2); `q > 0` marks a positive grasp, the rest live in `grasp_label_negative/`.
    `_grasp_anything_format` ignores this column.
    """

    def __init__(
        self,
        file_path,
        ds_rotate=0,
        seen=True,
        include_prompt=True,
        include_mask=True,
        include_union=False,
        split_path=None,
        prompt_tokenizer=None,
        **kwargs,
    ):
        """
        :param file_path: Grasp-Anything++ directory (holding image/, grasp_instructions/,
                          grasp_label_positive/, part_mask/).
        :param ds_rotate: Rotate the sample list by this fraction before splitting.
        :param seen: Use the seen (True) or unseen (False) split.
        :param include_prompt: Also return the grasp instruction.
        :param include_mask: Also return part_mask (under the same rot/zoom as the image).
        :param include_union: Also return M_union -- the union of grasps over all parts of the
                              same object. Off by default: the model has no unconditional
                              graspability branch, and building M_union means reading and
                              re-drawing the grasps of *every* part of the object (~4.4x the
                              work) for a target nobody consumes. Turn it back on for analysis.
        :param split_path: Custom split directory; defaults to searching SPLIT_DIRS.
        :param prompt_tokenizer: callable str -> dict of tensors (`PromptTokenizer`). When set,
                                 prompts are tokenized inside the worker and `extra["prompt"]`
                                 additionally carries a dict -- taking 6.3 ms/batch off the
                                 main process. None = return plain strings.
        :param kwargs: kwargs of GraspDatasetBase.
        """
        super().__init__(seen=seen, **kwargs)

        self.file_path = file_path
        self.prompt_tokenizer = prompt_tokenizer
        self.include_prompt = include_prompt
        self.include_mask = include_mask
        self.include_union = include_union

        required = ["image", "grasp_label_positive"]
        if include_prompt:
            required.append("grasp_instructions")
        if include_mask:
            required.append("part_mask")
        missing = [d for d in required if not os.path.isdir(os.path.join(file_path, d))]
        if missing:
            raise FileNotFoundError(
                "Missing directory {} in {}. Download it with "
                "script/download_grasp_anything_pp.sh.".format(
                    ", ".join(missing), file_path
                )
            )

        # grasp_files is the single source of indexing; every other path is derived from it,
        # because one image is shared by several part-level samples and the split filter is
        # applied to grasp_files only.
        self.grasp_files = sorted(
            glob.glob(os.path.join(file_path, "grasp_label_positive", "*.pt"))
        )
        if not self.grasp_files:
            raise FileNotFoundError(f"No .pt files found. Check the path: {file_path}")

        split_ids = self._load_split(split_path, seen)
        self.grasp_files = [
            f for f in self.grasp_files if self._in_split(self._sample_id(f), split_ids)
        ]
        if not self.grasp_files:
            raise FileNotFoundError(
                'Split {} matched no samples. GA++ ids are "<scene>_<object>_<part>" while '
                'split/grasp-anything/*.obj holds "<scene>_<object>" -- check split_path.'.format(
                    "seen" if seen else "unseen"
                )
            )

        self.length = len(self.grasp_files)

        # Group samples by object to build M_union. Only self.grasp_files (already filtered by
        # split) is scanned, so the union never pulls a category from the other split into the
        # target -- unioning every file on disk would be a leak.
        self._files_by_object = defaultdict(list)
        for f in self.grasp_files:
            self._files_by_object[self._object_id(self._sample_id(f))].append(f)

        if ds_rotate:
            split = int(self.length * ds_rotate)
            self.grasp_files = self.grasp_files[split:] + self.grasp_files[:split]

        # None = real prompts. shuffle_prompts()/set_fixed_prompt() change these two flags.
        self._prompt_perm = None
        self._fixed_prompt = None

    # ------------------------------------------------------------------ split --
    @staticmethod
    def _load_split(split_path, seen):
        split_file = "seen.obj" if seen else "unseen.obj"
        for directory in [split_path] if split_path else SPLIT_DIRS:
            path = os.path.join(directory, split_file)
            if os.path.isfile(path):
                with open(path, "rb") as f:
                    return set(pickle.load(f))
        raise FileNotFoundError(
            "Could not find {} in {}".format(
                split_file, split_path or " or ".join(SPLIT_DIRS)
            )
        )

    @classmethod
    def _in_split(cls, sample_id, split_ids):
        """
        GA++ splits may be part-level while base GA splits are object-level -- accept both.
        """
        return sample_id in split_ids or cls._object_id(sample_id) in split_ids

    # --------------------------------------------------------------------- id --
    @staticmethod
    def _sample_id(grasp_file):
        """ "<scene_id>_<object_idx>_<part_idx>" -- one (image, object, part) triple."""
        return os.path.splitext(os.path.basename(grasp_file))[0]

    @staticmethod
    def _object_id(sample_id):
        """ "<scene_id>_<object_idx>" -- the key used by the base GA split."""
        return sample_id.rsplit("_", 1)[0]

    @staticmethod
    def _scene_id(sample_id):
        """ "<scene_id>" -- the key into image/; all parts of a scene share this image."""
        return sample_id.rsplit("_", 2)[0]

    def sample_id(self, idx):
        return self._sample_id(self.grasp_files[idx])

    def scene_id(self, idx):
        return self._scene_id(self.sample_id(idx))

    # ------------------------------------------------------------------ paths --
    def get_rgb_file(self, idx):
        return os.path.join(self.file_path, "image", self.scene_id(idx) + ".jpg")

    def get_prompt_file(self, idx):
        return os.path.join(
            self.file_path, "grasp_instructions", self.sample_id(idx) + ".pkl"
        )

    def get_mask_file(self, idx):
        return os.path.join(self.file_path, "part_mask", self.sample_id(idx) + ".npy")

    # ------------------------------------------------------------------- data --
    def get_gtbb(self, idx, rot=0, zoom=1.0):
        # validate() in train_network.py passes rot/zoom straight from the batch (1-element
        # tensors); np.cos() of a tensor yields an array and corrupts the rotation matrix ->
        # coerce to float.
        rot, zoom = float(rot), float(zoom)
        gtbbs = grasp.GraspRectangles.load_from_grasp_anything_file(
            self.grasp_files[idx], scale=self.output_size / float(SOURCE_SIZE)
        )

        c = self.output_size // 2
        gtbbs.rotate(rot, (c, c))
        gtbbs.zoom(zoom, (c, c))
        return gtbbs

    def get_depth(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError(
            "Grasp-Anything++ is RGB only; train with --use-depth 0."
        )

    def _augment(self, img, rot, zoom):
        """
        crop(zoom) -> resize(output_size) -> rotate.

        The upstream order is rotate -> zoom -> resize, i.e. the two most expensive
        interpolations run at 416x416 before dropping to 224 -- ~3.4x the pixels for an
        identical result. For rotations by multiples of 90° (exactly what `random_rotate`
        produces), the centre crop and the rotation commute, so reordering is geometrically
        equivalent.

        :param img: image.Image, modified in place
        """
        if zoom != 1.0:
            h, w = img.img.shape[0], img.img.shape[1]
            sr, sc = int(h * (1 - zoom)) // 2, int(w * (1 - zoom)) // 2
            img.img = img.img[sr : h - sr, sc : w - sc]
        img.resize((self.output_size, self.output_size))
        if rot != 0.0:
            img.rotate(rot)
        return img

    def get_rgb(self, idx, rot=0, zoom=1.0, normalise=True):
        rot, zoom = float(rot), float(zoom)
        rgb_img = self._augment(
            image.Image.from_file(self.get_rgb_file(idx)), rot, zoom
        )
        if normalise:
            rgb_img.normalise()
            rgb_img.img = rgb_img.img.transpose((2, 0, 1))
        return rgb_img.img

    def get_prompt(self, idx, use_permutation=True):
        """
        The grasp instruction of this sample, e.g. "Lift apple by its skin.".

        After `shuffle_prompts()` this returns the prompt of a *different* sample -- the image
        is unchanged, the instruction is wrong. See that docstring.

        :param use_permutation: False to get the real prompt even while the permutation is
                                active (to compare both on the same sample, see
                                script/audit_text_reliance.py).
        """
        if self._fixed_prompt is not None:
            return self._fixed_prompt
        if use_permutation and self._prompt_perm is not None:
            idx = int(self._prompt_perm[idx])
        with open(self.get_prompt_file(idx), "rb") as f:
            prompt = pickle.load(f)
        # GA++ stores a bare str, but a few older files wrap it in a list/tuple.
        if not isinstance(prompt, str):
            prompt = prompt[0]
        return prompt

    def shuffle_prompts(self, seed=0):
        """
        Pair each image with another sample's prompt -- the counterfactual for "does the model
        *actually* read the prompt".

        If accuracy barely moves when prompts are permuted, the language branch contributes
        nothing: the model is just predicting the image's average grasp. This is the mandatory
        negative control for a language-driven method, and far cheaper than retraining an
        image-only arm.

        The permutation is patched so that no sample keeps its own prompt *and* none receives
        the prompt of another part of the same object (a same-object prompt still names the
        right object, which weakens the control).

        `part_mask` and the grasp labels are **not** permuted: they remain the image's ground
        truth, so the `align_loss` measured here is exactly the error caused by the wrong prompt.

        :param seed: seed that makes the permutation reproducible
        :return: self (for chaining)
        """
        rng = np.random.default_rng(seed)
        n = len(self.grasp_files)
        perm = rng.permutation(n)
        objects = [self._object_id(self._sample_id(f)) for f in self.grasp_files]

        # One repair pass: wherever a sample is paired with its own object, swap it with a
        # random other position. Real datasets have hundreds of thousands of objects, so
        # collisions are rare.
        for i in range(n):
            if objects[perm[i]] != objects[i]:
                continue
            for _ in range(8):
                j = int(rng.integers(n))
                if objects[perm[j]] != objects[i] and objects[perm[i]] != objects[j]:
                    perm[i], perm[j] = perm[j], perm[i]
                    break

        self._prompt_perm = perm
        return self

    def set_fixed_prompt(self, prompt):
        """
        A single prompt for *every* image -- the second counterfactual.

        A permuted prompt still hands the model a grammatical, in-distribution sentence; a
        fixed prompt removes both the information and the variety. If accuracy does not drop
        even here, the language branch certainly contributes nothing.

        :return: self
        """
        self._fixed_prompt = prompt
        return self

    def real_prompts(self):
        """Drop the permutation / fixed prompt and go back to the real prompts."""
        self._prompt_perm = None
        self._fixed_prompt = None
        return self

    def get_part_mask(self, idx, rot=0, zoom=1.0):
        """
        Binary mask of the part the prompt refers to, put through *exactly* the same transform
        chain as get_rgb (rotate -> zoom -> resize) so the alignment loss is not learned
        misaligned.
        """
        rot, zoom = float(rot), float(zoom)
        mask_img = image.Image(self._load_mask(idx).astype(np.float32))
        self._augment(mask_img, rot, zoom)
        # Interpolation makes the mask non-binary -> threshold it again.
        return (mask_img.img > 0.5).astype(np.float32)

    def _load_mask(self, idx):
        """
        416x416 uint8 part_mask. Also accepts the bit-packed form (a flat 21,632-byte array)
        written by `script/build_ga_pp_subset.py --pack-masks`, 8x smaller on disk.
        """
        mask = np.load(self.get_mask_file(idx))
        if mask.ndim == 1:
            mask = np.unpackbits(mask)[: SOURCE_SIZE * SOURCE_SIZE].reshape(
                SOURCE_SIZE, SOURCE_SIZE
            )
        return mask

    def get_union_gtbb(self, idx, rot=0, zoom=1.0):
        """
        Union of the grasp rectangles of *every* part of the same object -- the target for a
        text-independent graspability branch (answering "where can this be held", regardless of
        which part the prompt names).
        """
        rot, zoom = float(rot), float(zoom)
        grs = []
        for path in self._files_by_object[self._object_id(self.sample_id(idx))]:
            gtbbs = grasp.GraspRectangles.load_from_grasp_anything_file(
                path, scale=self.output_size / float(SOURCE_SIZE)
            )
            grs.extend(gtbbs.grs)

        gtbbs = grasp.GraspRectangles(grs)
        c = self.output_size // 2
        gtbbs.rotate(rot, (c, c))
        gtbbs.zoom(zoom, (c, c))
        return gtbbs

    def get_union_pos(self, idx, rot=0, zoom=1.0):
        """M_union as a [0, 1] map at the output resolution."""
        pos_img, _, _ = self.get_union_gtbb(idx, rot, zoom).draw(
            (self.output_size, self.output_size)
        )
        return np.clip(pos_img, 0.0, 1.0).astype(np.float32)

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        if not (self.include_prompt or self.include_mask or self.include_union):
            return sample

        # rot/zoom come from the sample just returned, so prompt/mask match the image and labels.
        _, _, _, rot, zoom_factor = sample
        extra = {}
        if self.include_prompt:
            extra["prompt"] = self.get_prompt(idx)
            # Tokenizing here means tokenizing in the worker. `default_collate` stacks the
            # tensor dicts into (B, L) and `CLIPTextEncoder` takes them directly. The raw
            # string stays under the "prompt" key, because utils/visualisation/alignment.py and
            # evaluate.py read it as a str.
            if self.prompt_tokenizer is not None:
                extra["prompt_tokens"] = self.prompt_tokenizer(extra["prompt"])
        if self.include_mask:
            extra["part_mask"] = self.numpy_to_torch(
                self.get_part_mask(idx, rot, zoom_factor)
            )
        if self.include_union:
            extra["union_pos"] = self.numpy_to_torch(
                self.get_union_pos(idx, rot, zoom_factor)
            )
        return sample + (extra,)
