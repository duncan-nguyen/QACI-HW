import warnings

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
from imageio import imread
from skimage.transform import resize, rotate

warnings.filterwarnings("ignore", category=UserWarning)

# OpenCV spawns its own thread pool per operation. Inside a DataLoader each worker is already a
# process, so letting cv2 add threads only makes N workers fight over the same cores and run
# slower.
cv2.setNumThreads(0)

# OpenCV resize/warpAffine only accepts these depths; other dtypes (e.g. int32) fall back to skimage.
_CV_DTYPES = (np.uint8, np.uint16, np.int16, np.float32, np.float64)


def _cv_ok(img):
    return img.dtype.type in _CV_DTYPES and (img.ndim == 2 or img.shape[2] <= 4)


# `skimage.transform.resize(mode=...)` sets the border for the *anti-alias filter*, and
# skimage's names do not match cv2's: skimage 'reflect' -> scipy 'mirror' -> cv2
# BORDER_REFLECT_101, while skimage 'symmetric' -> scipy 'reflect' -> cv2 BORDER_REFLECT. Get
# this mapping wrong and the image differs on a few border pixels -- with the mapping below the
# result is bit-identical to the original.
_SKIMAGE_BORDER = {
    "reflect": cv2.BORDER_REFLECT_101,
    "symmetric": cv2.BORDER_REFLECT,
    "edge": cv2.BORDER_REPLICATE,
    "constant": cv2.BORDER_CONSTANT,
}


def _gauss_kernel(sigma):
    """1-D gaussian kernel following cv2's ksize formula; sigma <= 0 -> identity kernel."""
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
        Read a colour image. cv2.imread is ~25% faster than imageio on JPEG (1.51 ms vs 1.94 ms
        for 416x416) because it goes straight to libjpeg-turbo without imageio's plugin layer.

        cv2 returns BGR, so it must be converted to RGB -- every caller of this function
        (cornell, jacquard, vmrd, ocid, grasp-anything) treats the channels as RGB.
        """
        img = cv2.imread(str(fname), cv2.IMREAD_COLOR)
        if img is None:
            # cv2.imread swallows errors and returns None (missing file, permissions, or an
            # unusual format), whereas imageio's imread raises. Keep the fail-fast behaviour so
            # training never runs on black images.
            raise FileNotFoundError(f"Could not read image: {fname}")
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

        Replaces `skimage.transform.resize` with exactly the two steps skimage performs
        internally, only run through cv2: an anti-alias gaussian filter with
        `sigma = (downscale factor - 1) / 2`, then bilinear interpolation. 12.5 ms -> 0.40 ms
        for a 416x416x3 -> 224x224 image.

        The only difference from the original is rounding: uint8 images differ by 1/255 on
        0.02-0.4% of pixels (because the work is done in float32 rather than float64 -- forcing
        float64 is bit-identical but costs 3.2 ms), float32 masks differ by 2e-7. See
        `_SKIMAGE_BORDER` and the upscaling branch below: those are where *real* divergence can
        creep in, not rounding.

        Do not swap in INTER_AREA: it also anti-aliases, but as a box filter, so it differs from
        skimage by up to 0.74/255 per pixel -- and turns out to be *slower* (0.69 ms).

        :param shape: New shape.
        :param mode: Border handling, using skimage's names ("reflect" or "symmetric").
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
        # skimage does all the work in floating point and casts back at the end. Do the same:
        # cv2's fixed-point interpolation on uint8 would differ by 1 LSB from the float path.
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
            # Downscaling: the source coordinate `(i + 0.5) * scale - 0.5` always lies inside
            # the image, so no pixel samples outside the border and `cv2.resize` (which offers
            # no borderMode) gives the right result -- and is three times faster than
            # warpAffine.
            out = cv2.resize(src, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            # Upscaling: the outermost rows/columns *do* sample outside the border (dst=0 maps
            # to src=-0.05), and `cv2.resize` always replicates the border while skimage
            # reflects it -> up to 13/255 of difference along the frame. warpAffine accepts a
            # borderMode, so it can match.
            sy, sx = src.shape[0] / h, src.shape[1] / w
            matrix = np.array(
                [[sx, 0.0, 0.5 * sx - 0.5], [0.0, sy, 0.5 * sy - 0.5]], dtype=np.float64
            )
            out = cv2.warpAffine(
                src,
                matrix,
                (w, h),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=border,
            )
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

        Three paths, increasingly fast:

        * an angle that is a multiple of 90 degrees about the image centre -> `np.rot90`, i.e.
          pure index permutation: 0.22 ms instead of 2.32 ms, and *exact* (skimage still
          interpolates, so it rounds off by ~0.3/255 per pixel). These are exactly the angles
          `GraspDatasetBase.__getitem__` generates, so this path carries almost every call
          during training.
        * an arbitrary angle -> `cv2.warpAffine` with BORDER_REFLECT (equivalent to skimage's
          `mode='symmetric'`; verified: at most 1/255 of difference).
        * a dtype cv2 does not accept -> skimage, as before.

        :param angle: Angle (in radians) to rotate by.
        :param center: Center pixel to rotate if specified, otherwise image center is used.
        """
        h, w = self.img.shape[0], self.img.shape[1]
        k = angle / (np.pi / 2)
        k_int = int(round(k))
        # rot90 is about the centre ((h-1)/2, (w-1)/2) -- exactly the centre skimage uses when
        # center=None. Odd k swaps height and width, so it only applies to square images; k=2
        # (flip both axes) is correct at any size.
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
        # skimage takes the centre ((cols-1)/2, (rows-1)/2) when center=None; cv2 also takes (x, y).
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
