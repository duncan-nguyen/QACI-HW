import glob
import os
import pickle
from collections import defaultdict

import numpy as np

from utils.dataset_processing import grasp, image

from .grasp_data import GraspDatasetBase

# Ảnh và part_mask của Grasp-Anything luôn là 416x416.
SOURCE_SIZE = 416

# Thứ tự tìm split nếu người dùng không chỉ định: split riêng cho GA++ trước, không có thì
# dùng tạm split object-level của base GA.
SPLIT_DIRS = (
    os.path.join("split", "grasp-anything-pp"),
    os.path.join("split", "grasp-anything"),
)


class GraspAnythingPPDataset(GraspDatasetBase):
    """
    Dataset wrapper cho Grasp-Anything++ (language-driven grasping).

    Khác Grasp-Anything gốc ở ba chỗ:

    * một sample là (scene, object, part), tên file `<scene_id>_<object_idx>_<part_idx>` --
      *ba* phần, không phải hai. `scene_id` là SHA-256 của ảnh nên không chứa '_', bóc hậu tố
      bằng `rsplit` là an toàn.
    * prompt nằm ở `grasp_instructions/`, mỗi file là một `str` ("Lift apple by its skin.").
      Đây không phải `scene_description/` của base GA -- cái đó là tuple (caption, [tên object])
      mô tả cả scene, không nhắm vào part nào.
    * có thêm `part_mask/`: mask nhị phân 416x416 uint8 của đúng part mà prompt nói tới.

    Layout `grasp_label_positive/*.pt` thì giống hệt base GA -- float32 (N, 6) mỗi hàng là
    `[q, x, y, w, h, theta_deg]` -- nên `GraspRectangles.load_from_grasp_anything_file` dùng lại
    được nguyên vẹn. Cột `q` là điểm antipodal T~ = (cos a1 + cos a2) / R của paper LGD (§3.2);
    `q > 0` là grasp dương, phần còn lại nằm ở `grasp_label_negative/`. `_grasp_anything_format`
    bỏ qua cột này.
    """

    def __init__(
        self,
        file_path,
        ds_rotate=0,
        seen=True,
        include_prompt=True,
        include_mask=True,
        include_union=True,
        split_path=None,
        align_weight_path=None,
        single_part_weight=1.0,
        **kwargs,
    ):
        """
        :param file_path: Thư mục Grasp-Anything++ (chứa image/, grasp_instructions/,
                          grasp_label_positive/, part_mask/).
        :param ds_rotate: Xoay vòng danh sách sample theo tỉ lệ này trước khi chia tập.
        :param seen: Lấy split seen (True) hay unseen (False).
        :param include_prompt: Trả kèm prompt gắp.
        :param include_mask: Trả kèm part_mask (đã chịu cùng rot/zoom với ảnh).
        :param include_union: Trả kèm M_union -- bản đồ graspability hợp mọi part của object.
        :param split_path: Thư mục split tự chọn; mặc định dò theo SPLIT_DIRS.
        :param kwargs: kwargs của GraspDatasetBase.
        """
        super().__init__(seen=seen, **kwargs)

        self.file_path = file_path
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
                "Thiếu thư mục {} trong {}. Tải bằng script/download_grasp_anything_pp.sh.".format(
                    ", ".join(missing), file_path
                )
            )

        # grasp_files là nguồn duy nhất để đánh index; mọi path khác derive từ nó, vì một ảnh
        # dùng chung cho nhiều sample part-level và bộ lọc split chỉ áp lên grasp_files.
        self.grasp_files = sorted(
            glob.glob(os.path.join(file_path, "grasp_label_positive", "*.pt"))
        )
        if not self.grasp_files:
            raise FileNotFoundError(
                f"Không thấy file .pt nào. Kiểm tra path: {file_path}"
            )

        split_ids = self._load_split(split_path, seen)
        self.grasp_files = [
            f for f in self.grasp_files if self._in_split(self._sample_id(f), split_ids)
        ]
        if not self.grasp_files:
            raise FileNotFoundError(
                'Split {} không khớp sample nào. Id GA++ là "<scene>_<object>_<part>" còn '
                'split/grasp-anything/*.obj là "<scene>_<object>" -- kiểm tra lại split_path.'.format(
                    "seen" if seen else "unseen"
                )
            )

        self.length = len(self.grasp_files)

        # Nhóm sample theo object để dựng M_union. Chỉ gom từ self.grasp_files (đã lọc split),
        # nên union không bao giờ kéo category của split kia vào target -- union mọi file trên
        # đĩa mới là leak.
        self._files_by_object = defaultdict(list)
        for f in self.grasp_files:
            self._files_by_object[self._object_id(self._sample_id(f))].append(f)

        if ds_rotate:
            split = int(self.length * ds_rotate)
            self.grasp_files = self.grasp_files[split:] + self.grasp_files[:split]

        self.align_weights, self.align_weight_summary = self._load_align_weights(
            align_weight_path, single_part_weight
        )

    # --------------------------------------------------- trọng số alignment --
    DEFAULT_ALIGN_WEIGHTS = "align_weights.npz"

    def _load_align_weights(self, path, single_part_weight):
        """
        Nạp d_i = 1 - IoU(part_mask_i, hợp mask mọi part cùng object), do
        `script/build_align_weights.py` tính sẵn.

        d_i đo "biết part nào thì thu hẹp được vùng bao nhiêu". Bằng 0 nghĩa là mask của part
        bằng cả object -- target không phân biệt part, nên `L_align` trên sample đó dạy model
        *bất biến theo part*, ngược hẳn luận điểm của method.

        Trả về mảng float32 xếp theo đúng thứ tự self.grasp_files (đã lọc split), không giữ
        dict: dict 880k khoá sẽ phá copy-on-write khi fork DataLoader worker và ngốn vài GB.

        :return: (np.ndarray hoặc None, dict tóm tắt để log)
        """
        path = path or os.path.join(self.file_path, self.DEFAULT_ALIGN_WEIGHTS)
        if not os.path.isfile(path):
            return None, {"source": None}

        blob = np.load(path, allow_pickle=False)
        lookup = dict(zip(blob["ids"].tolist(), blob["d"].tolist()))
        n_parts = dict(zip(blob["ids"].tolist(), blob["n_parts"].tolist()))

        weights = np.ones(len(self.grasp_files), dtype=np.float32)
        multi, single, missing = [], 0, 0
        for i, f in enumerate(self.grasp_files):
            sid = self._sample_id(f)
            if sid not in lookup:
                missing += 1
                continue
            if n_parts[sid] < 2:
                # Object một part: d_i không xác định (hợp bằng chính nó). Không có bằng chứng
                # để loại, nên giữ nguyên trọng số và đếm riêng.
                weights[i] = single_part_weight
                single += 1
            else:
                weights[i] = lookup[sid]
                multi.append(lookup[sid])

        summary = {
            "source": path,
            "n_samples": len(self.grasp_files),
            "n_missing": missing,
            "n_single_part": single,
            "n_multi_part": len(multi),
            # Con số quyết định: trung bình d_i trên các object có ≥2 part. Gần 0 nghĩa là
            # part_mask của dataset ở mức object và L_align không dạy được grounding part.
            "mean_d_multi_part": float(np.mean(multi)) if multi else float("nan"),
            "mean_weight_all": float(weights.mean()),
            "single_part_weight": single_part_weight,
        }
        return weights, summary

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
            "Không tìm thấy {} trong {}".format(
                split_file, split_path or " hoặc ".join(SPLIT_DIRS)
            )
        )

    @classmethod
    def _in_split(cls, sample_id, split_ids):
        """
        Split của GA++ có thể là part-level, của base GA thì là object-level -- nhận cả hai.
        """
        return sample_id in split_ids or cls._object_id(sample_id) in split_ids

    # --------------------------------------------------------------------- id --
    @staticmethod
    def _sample_id(grasp_file):
        """ "<scene_id>_<object_idx>_<part_idx>" -- một cặp (ảnh, object, part)."""
        return os.path.splitext(os.path.basename(grasp_file))[0]

    @staticmethod
    def _object_id(sample_id):
        """ "<scene_id>_<object_idx>" -- khoá của split base GA."""
        return sample_id.rsplit("_", 1)[0]

    @staticmethod
    def _scene_id(sample_id):
        """ "<scene_id>" -- khoá của image/, mọi part của cùng scene dùng chung ảnh này."""
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
        # validate() trong train_network.py truyền rot/zoom lấy thẳng từ batch (tensor 1 phần
        # tử), np.cos() của tensor sẽ ra mảng và làm hỏng ma trận xoay -> ép về float.
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
            "Grasp-Anything++ chỉ có RGB; train với --use-depth 0."
        )

    def _augment(self, img, rot, zoom):
        """
        crop(zoom) -> resize(output_size) -> rotate.

        Thứ tự gốc của repo là rotate -> zoom -> resize, tức là hai phép nội suy đắt nhất chạy
        ở 416x416 rồi mới hạ xuống 224 -- tốn gấp ~3,4 lần số pixel mà kết quả không khác.
        Với góc xoay bội số 90° (đúng những góc `random_rotate` sinh ra) thì crop tâm và xoay
        giao hoán, nên đảo thứ tự là tương đương về hình học.

        :param img: image.Image, sẽ bị sửa tại chỗ
        """
        if zoom != 1.0:
            h, w = img.img.shape[0], img.img.shape[1]
            sr, sc = int(h * (1 - zoom)) // 2, int(w * (1 - zoom)) // 2
            img.img = img.img[sr:h - sr, sc:w - sc]
        img.resize((self.output_size, self.output_size))
        if rot != 0.0:
            img.rotate(rot)
        return img

    def get_rgb(self, idx, rot=0, zoom=1.0, normalise=True):
        rot, zoom = float(rot), float(zoom)
        rgb_img = self._augment(image.Image.from_file(self.get_rgb_file(idx)), rot, zoom)
        if normalise:
            rgb_img.normalise()
            rgb_img.img = rgb_img.img.transpose((2, 0, 1))
        return rgb_img.img

    def get_prompt(self, idx):
        """Câu lệnh gắp của sample này, ví dụ "Lift apple by its skin."."""
        with open(self.get_prompt_file(idx), "rb") as f:
            prompt = pickle.load(f)
        # GA++ lưu một str thuần, nhưng vài file cũ gói trong list/tuple.
        if not isinstance(prompt, str):
            prompt = prompt[0]
        return prompt

    def get_part_mask(self, idx, rot=0, zoom=1.0):
        """
        Mask nhị phân của part được prompt nhắc tới, đã qua *đúng* chuỗi biến đổi của get_rgb
        (rotate -> zoom -> resize) để alignment loss không học lệch.
        """
        rot, zoom = float(rot), float(zoom)
        mask_img = image.Image(self._load_mask(idx).astype(np.float32))
        self._augment(mask_img, rot, zoom)
        # Nội suy làm mask hết nhị phân -> ngưỡng lại.
        return (mask_img.img > 0.5).astype(np.float32)

    def _load_mask(self, idx):
        """
        part_mask 416x416 uint8. Chấp nhận cả bản đóng gói bit (mảng 1 chiều 21.632 byte) --
        `script/build_ga_pp_subset.py --pack-masks` lưu kiểu đó, nhỏ hơn 8 lần trên đĩa.
        """
        mask = np.load(self.get_mask_file(idx))
        if mask.ndim == 1:
            mask = np.unpackbits(mask)[:SOURCE_SIZE * SOURCE_SIZE].reshape(SOURCE_SIZE,
                                                                          SOURCE_SIZE)
        return mask

    def get_union_gtbb(self, idx, rot=0, zoom=1.0):
        """
        Hợp grasp rectangle của *mọi* part thuộc cùng object -- target cho nhánh graspability
        không điều kiện text (trả lời "cầm được ở đâu", bất kể prompt nói part nào).
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
        """M_union dưới dạng bản đồ [0, 1] cùng kích thước output."""
        pos_img, _, _ = self.get_union_gtbb(idx, rot, zoom).draw(
            (self.output_size, self.output_size)
        )
        return np.clip(pos_img, 0.0, 1.0).astype(np.float32)

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        if not (self.include_prompt or self.include_mask or self.include_union):
            return sample

        # rot/zoom lấy từ chính sample vừa trả về, để prompt/mask khớp với ảnh và grasp label.
        _, _, _, rot, zoom_factor = sample
        extra = {}
        if self.include_prompt:
            extra["prompt"] = self.get_prompt(idx)
        if self.include_mask:
            extra["part_mask"] = self.numpy_to_torch(
                self.get_part_mask(idx, rot, zoom_factor)
            )
        if self.include_union:
            extra["union_pos"] = self.numpy_to_torch(
                self.get_union_pos(idx, rot, zoom_factor)
            )
        if self.align_weights is not None:
            extra["align_weight"] = float(self.align_weights[idx])
        return sample + (extra,)
