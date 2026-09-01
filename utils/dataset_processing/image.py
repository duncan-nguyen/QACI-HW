import warnings

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
from imageio import imread
from skimage.transform import resize, rotate

warnings.filterwarnings("ignore", category=UserWarning)

# OpenCV tự mở thread pool riêng cho từng phép. Trong DataLoader mỗi worker đã là một tiến
# trình, nên để cv2 đẻ thêm thread chỉ khiến N worker giành nhau cùng số core và chậm hơn.
cv2.setNumThreads(0)

# resize/warpAffine của OpenCV chỉ nhận các depth này; dtype khác (vd int32) rơi về skimage.
_CV_DTYPES = (np.uint8, np.uint16, np.int16, np.float32, np.float64)


def _cv_ok(img):
    return img.dtype.type in _CV_DTYPES and (img.ndim == 2 or img.shape[2] <= 4)



# `skimage.transform.resize(mode=...)` đặt border cho *bộ lọc anti-alias*, và tên của skimage
# không trùng tên của cv2: skimage 'reflect' -> scipy 'mirror' -> cv2 BORDER_REFLECT_101, còn
# skimage 'symmetric' -> scipy 'reflect' -> cv2 BORDER_REFLECT. Ánh xạ sai ở đây thì ảnh lệch
# ở vài pixel viền -- với đúng ánh xạ này kết quả trùng khít từng bit với bản cũ.
_SKIMAGE_BORDER = {
    "reflect": cv2.BORDER_REFLECT_101,
    "symmetric": cv2.BORDER_REFLECT,
    "edge": cv2.BORDER_REPLICATE,
    "constant": cv2.BORDER_CONSTANT,
}


def _gauss_kernel(sigma):
    """Kernel Gauss 1 chiều theo đúng công thức ksize của cv2; sigma <= 0 -> kernel đơn vị."""
    if sigma <= 0:
        return np.ones((1, 1), dtype=np.float64)
    ksize = int(round(sigma * 8 + 1)) | 1
    return cv2.getGaussianKernel(ksize, sigma)

class Image:
    """
    Wrapper around an image with some convenient functions.
    """

    def __init__(self, img):
        self.img = img

    def __getattr__(self, attr):
        # Pass along any other methods to the underlying ndarray
        return getattr(self.img, attr)

    @classmethod
    def from_file(cls, fname):
        """
        Đọc ảnh màu. cv2.imread nhanh hơn imageio ~25% trên JPEG (1,51 ms vs 1,94 ms cho
        416x416) vì đi thẳng libjpeg-turbo, không qua lớp plugin của imageio.

        cv2 trả BGR nên phải đổi về RGB -- mọi caller của hàm này (cornell, jacquard, vmrd,
        ocid, grasp-anything) đều coi kênh là RGB.
        """
        img = cv2.imread(str(fname), cv2.IMREAD_COLOR)
        if img is None:
            # cv2.imread nuốt lỗi và trả None (file thiếu, quyền, hoặc format lạ); imread của
            # imageio thì ném exception. Giữ hành vi "nổ ngay" để không train trên ảnh đen.
            raise FileNotFoundError(f"Không đọc được ảnh: {fname}")
        return cls(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def copy(self):
        """
        :return: Copy of self.
        """
        return self.__class__(self.img.copy())

    @classmethod
    def mask_out_image(cls, image, mask):
        # Apply the mask to the image
        masked_image = np.array(image)
        masked_image[:, :, 0] = masked_image[:, :, 0] * mask + 255 * (1 - mask)
        masked_image[:, :, 1] = masked_image[:, :, 1] * mask + 255 * (1 - mask)
        masked_image[:, :, 2] = masked_image[:, :, 2] * mask + 255 * (1 - mask)

        return cls(imageio.core.util.Array(masked_image))

    def crop(self, top_left, bottom_right, resize=None):
        """
        Crop the image to a bounding box given by top left and bottom right pixels.
        :param top_left: tuple, top left pixel.
        :param bottom_right: tuple, bottom right pixel
        :param resize: If specified, resize the cropped image to this size
        """
        self.img = self.img[
            top_left[0] : bottom_right[0], top_left[1] : bottom_right[1]
        ]
        if resize is not None:
            self.resize(resize)

    def cropped(self, *args, **kwargs):
        """
        :return: Cropped copy of the image.
        """
        i = self.copy()
        i.crop(*args, **kwargs)
        return i

    def normalise(self):
        """
        Normalise the image by converting to float [0,1] and zero-centering
        """
        self.img = self.img.astype(np.float32) / 255.0
        self.img -= self.img.mean()

    def resize(self, shape, mode="reflect"):
        """
        Resize image to shape.

        Thay `skimage.transform.resize` bằng đúng hai bước mà skimage làm bên trong, chỉ khác
        là chạy bằng cv2: lọc Gauss chống răng cưa với `sigma = (tỉ lệ thu nhỏ - 1) / 2`, rồi
        nội suy song tuyến. 12,5 ms -> 0,40 ms cho một ảnh 416x416x3 -> 224x224.

        Sai khác so với bản cũ chỉ là làm tròn: ảnh uint8 lệch 1/255 trên 0,02-0,4% pixel (do
        tính ở float32 thay vì float64 -- ép về float64 thì trùng khít từng bit nhưng mất
        3,2 ms), mask float32 lệch 2e-7. Xem `_SKIMAGE_BORDER` và nhánh phóng to bên dưới:
        hai chỗ đó mới là chỗ dễ lệch *thật*, không phải làm tròn.

        Đừng thay bằng INTER_AREA: nó cũng chống răng cưa nhưng là box filter nên cho ảnh khác
        skimage tới 0,74/255 mỗi pixel -- mà hoá ra còn *chậm hơn* (0,69 ms).

        :param shape: New shape.
        :param mode: Xử lý biên, cùng tên với skimage ("reflect" hoặc "symmetric").
        """
        if self.img.shape == shape:
            return
        h, w = int(shape[0]), int(shape[1])
        if not _cv_ok(self.img):
            self.img = resize(self.img, shape, mode=mode, preserve_range=True).astype(
                self.img.dtype
            )
            return

        out_dtype = self.img.dtype
        # skimage tính toàn bộ ở dấu phẩy động rồi mới ép về dtype cũ. Làm y hệt: nội suy
        # fixed-point của cv2 trên uint8 sẽ lệch 1 LSB so với đường float.
        work = np.float64 if out_dtype == np.float64 else np.float32
        src = np.ascontiguousarray(self.img, dtype=work)

        sigma_y = max(0.0, (src.shape[0] / h - 1) / 2.0)
        sigma_x = max(0.0, (src.shape[1] / w - 1) / 2.0)
        if sigma_y > 0 or sigma_x > 0:
            src = cv2.sepFilter2D(
                src,
                -1,
                _gauss_kernel(sigma_x),
                _gauss_kernel(sigma_y),
                borderType=_SKIMAGE_BORDER.get(mode, cv2.BORDER_REFLECT_101),
            )

        border = _SKIMAGE_BORDER.get(mode, cv2.BORDER_REFLECT_101)
        if src.shape[0] >= h and src.shape[1] >= w:
            # Thu nhỏ: toạ độ nguồn `(i + 0.5) * tỉ_lệ - 0.5` luôn nằm trong ảnh, nên không
            # có pixel nào lấy mẫu ngoài biên và `cv2.resize` (không cho chọn borderMode)
            # cho đúng kết quả -- lại nhanh gấp ba warpAffine.
            out = cv2.resize(src, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            # Phóng to: hàng/cột ngoài cùng *có* lấy mẫu ngoài biên (dst=0 ứng với src=-0.05),
            # mà `cv2.resize` luôn nhân bản biên còn skimage thì phản chiếu -> lệch tới 13/255
            # ở khung viền. warpAffine nhận borderMode nên khớp lại được.
            sy, sx = src.shape[0] / h, src.shape[1] / w
            matrix = np.array([[sx, 0.0, 0.5 * sx - 0.5],
                               [0.0, sy, 0.5 * sy - 0.5]], dtype=np.float64)
            out = cv2.warpAffine(src, matrix, (w, h),
                                 flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                 borderMode=border)
        self.img = out.astype(out_dtype)


    def resized(self, *args, **kwargs):
        """
        :return: Resized copy of the image.
        """
        i = self.copy()
        i.resize(*args, **kwargs)
        return i

    def rotate(self, angle, center=None):
        """
        Rotate the image.

        Ba đường, nhanh dần:

        * góc là bội số của 90 độ quanh tâm ảnh -> `np.rot90`, tức hoán vị chỉ số thuần tuý:
          0,22 ms thay vì 2,32 ms, và *chính xác tuyệt đối* (skimage vẫn nội suy nên làm tròn
          sai lệch ~0,3/255 mỗi pixel). Đây đúng là mọi góc mà `GraspDatasetBase.__getitem__`
          sinh ra, nên đường này gánh gần như toàn bộ lượt gọi lúc train.
        * góc bất kỳ -> `cv2.warpAffine` với BORDER_REFLECT (tương đương `mode='symmetric'`
          của skimage; đã đối chiếu: lệch tối đa 1/255).
        * dtype cv2 không nhận -> skimage như cũ.

        :param angle: Angle (in radians) to rotate by.
        :param center: Center pixel to rotate if specified, otherwise image center is used.
        """
        h, w = self.img.shape[0], self.img.shape[1]
        k = angle / (np.pi / 2)
        k_int = int(round(k))
        # rot90 quanh tâm ((h-1)/2, (w-1)/2) -- đúng tâm mà skimage dùng khi center=None.
        # k lẻ đổi chiều cao/rộng nên chỉ dùng được cho ảnh vuông; k=2 (lật cả hai trục) thì
        # đúng với mọi kích thước.
        if center is None and abs(k - k_int) < 1e-9 and (h == w or k_int % 2 == 0):
            k_int %= 4
            if k_int:
                self.img = np.ascontiguousarray(np.rot90(self.img, k_int))
            return

        if center is not None:
            center = (center[1], center[0])
        if not _cv_ok(self.img):
            self.img = rotate(
                self.img,
                angle / np.pi * 180,
                center=center,
                mode="symmetric",
                preserve_range=True,
            ).astype(self.img.dtype)
            return
        # skimage lấy tâm ((cols-1)/2, (rows-1)/2) khi center=None; cv2 cũng nhận (x, y).
        c = (
            ((w - 1) / 2.0, (h - 1) / 2.0)
            if center is None
            else (float(center[0]), float(center[1]))
        )
        M = cv2.getRotationMatrix2D(c, angle / np.pi * 180, 1.0)
        self.img = cv2.warpAffine(
            np.ascontiguousarray(self.img),
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        ).astype(self.img.dtype)

    def rotated(self, *args, **kwargs):
        """
        :return: Rotated copy of image.
        """
        i = self.copy()
        i.rotate(*args, **kwargs)
        return i

    def show(self, ax=None, **kwargs):
        """
        Plot the image
        :param ax: Existing matplotlib axis (optional)
        :param kwargs: kwargs to imshow
        """
        if ax:
            ax.imshow(self.img, **kwargs)
        else:
            plt.imshow(self.img, **kwargs)
            plt.show()

    def zoom(self, factor):
        """
        "Zoom" the image by cropping and resizing.
        :param factor: Factor to zoom by. e.g. 0.5 will keep the center 50% of the image.
        """
        sr = int(self.img.shape[0] * (1 - factor)) // 2
        sc = int(self.img.shape[1] * (1 - factor)) // 2
        orig_shape = self.img.shape
        self.img = self.img[
            sr : self.img.shape[0] - sr, sc : self.img.shape[1] - sc
        ].copy()
        self.resize(orig_shape, mode="symmetric")

    def zoomed(self, *args, **kwargs):
        """
        :return: Zoomed copy of the image.
        """
        i = self.copy()
        i.zoom(*args, **kwargs)
        return i


class DepthImage(Image):
    def __init__(self, img):
        super().__init__(img)

    @classmethod
    def from_pcd(cls, pcd_filename, shape, default_filler=0, index=None):
        """
        Create a depth image from an unstructured PCD file.
        If index isn't specified, use euclidean distance, otherwise choose x/y/z=0/1/2
        """
        img = np.zeros(shape)
        if default_filler != 0:
            img += default_filler

        with open(pcd_filename) as f:
            for l in f:
                ls = l.split()

                if len(ls) != 5:
                    # Not a point line in the file.
                    continue
                try:
                    # Not a number, carry on.
                    float(ls[0])
                except ValueError:
                    continue

                i = int(ls[4])
                r = i // shape[1]
                c = i % shape[1]

                if index is None:
                    x = float(ls[0])
                    y = float(ls[1])
                    z = float(ls[2])

                    img[r, c] = np.sqrt(x**2 + y**2 + z**2)

                else:
                    img[r, c] = float(ls[index])

        return cls(img / 1000.0)

    @classmethod
    def from_tiff(cls, fname):
        return cls(imread(fname))

    def inpaint(self, missing_value=0):
        """
        Inpaint missing values in depth image.
        :param missing_value: Value to fill in teh depth image.
        """
        # cv2 inpainting doesn't handle the border properly
        # https://stackoverflow.com/questions/25974033/inpainting-depth-map-still-a-black-image-border
        self.img = cv2.copyMakeBorder(self.img, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
        mask = (self.img == missing_value).astype(np.uint8)

        # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
        scale = np.abs(self.img).max()
        self.img = (
            self.img.astype(np.float32) / scale
        )  # Has to be float32, 64 not supported.
        self.img = cv2.inpaint(self.img, mask, 1, cv2.INPAINT_NS)

        # Back to original size and value range.
        self.img = self.img[1:-1, 1:-1]
        self.img = self.img * scale

    def gradients(self):
        """
        Compute gradients of the depth image using Sobel filtesr.
        :return: Gradients in X direction, Gradients in Y diretion, Magnitude of XY gradients.
        """
        grad_x = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, borderType=cv2.BORDER_DEFAULT)
        grad_y = cv2.Sobel(self.img, cv2.CV_64F, 0, 1, borderType=cv2.BORDER_DEFAULT)
        grad = np.sqrt(grad_x**2 + grad_y**2)

        return DepthImage(grad_x), DepthImage(grad_y), DepthImage(grad)

    def normalise(self):
        """
        Normalise by subtracting the mean and clippint [-1, 1]
        """
        self.img = np.clip((self.img - self.img.mean()), -1, 1)


class WidthImage(Image):
    """
    A width image is one that describes the desired gripper width at each pixel.
    """

    def zoom(self, factor):
        """
        "Zoom" the image by cropping and resizing.  Also scales the width accordingly.
        :param factor: Factor to zoom by. e.g. 0.5 will keep the center 50% of the image.
        """
        super().zoom(factor)
        self.img = self.img / factor

    def normalise(self):
        """
        Normalise by mapping [0, 150] -> [0, 1]
        """
        self.img = np.clip(self.img, 0, 150.0) / 150.0
