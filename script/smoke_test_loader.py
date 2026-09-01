"""
Check the image/label transforms after the move from skimage to cv2 -- no dataset required.

Run before spending GPU time. Three groups, matching the three places that can fail *silently*:

1. `Image.resize` must match `skimage.transform.resize` at every ratio the loaders use,
   including the **upscaling** path (GA++ with zoom < 0.538 crops to less than 224) -- where
   `cv2.resize` replicates the border while skimage reflects it, differing by up to 13/255
   along the frame.
2. `Image.rotate` at multiples of 90 degrees must be an *exact* index permutation, and at
   arbitrary angles must match skimage to within 1/255.
3. `GraspRectangles.draw` must preserve the `angle`/`length` values and the overwrite order;
   the `pos` area may differ by a few percent because cv2 and skimage fill borders differently.

    python script/smoke_test_loader.py
"""
import os
import sys

import cv2
import numpy as np
from skimage.draw import polygon
from skimage.transform import resize as sk_resize
from skimage.transform import rotate as sk_rotate

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataset_processing.grasp import Grasp, GraspRectangle, GraspRectangles  # noqa: E402
from utils.dataset_processing.image import Image  # noqa: E402

cv2.setNumThreads(0)
RNG = np.random.default_rng(0)
FAILS = []

# Allowed difference: rounding level only, per dtype.
TOL = {np.uint8: 1.0, np.float32: 1e-6, np.float64: 1e-12}


def check(name, ok, detail=""):
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILS.append(name)


def scene(h, w, channels=3, dtype=np.uint8):
    """A structured image (not white noise), so interpolation has something to get wrong."""
    img = cv2.resize(RNG.random((13, 13, 3)) * 255, (w, h), interpolation=cv2.INTER_CUBIC)
    if channels == 1:
        img = img[:, :, 0]
    return img.astype(dtype) if dtype == np.uint8 else (img / 255.0).astype(dtype)


def old_resize(img, shape, mode="reflect"):
    return sk_resize(img, shape, mode=mode, preserve_range=True).astype(img.dtype)


def old_rotate(img, angle, center=None):
    c = None if center is None else (center[1], center[0])
    return sk_rotate(img, angle / np.pi * 180, center=c, mode="symmetric",
                     preserve_range=True).astype(img.dtype)


def test_resize():
    print("\n1. Image.resize matches skimage (downscale, upscale, and different ratios per axis)")
    cases = [(416, 416, 3, np.uint8), (416, 416, 1, np.float32), (1024, 1024, 1, np.float64),
             (480, 640, 3, np.uint8), (230, 230, 3, np.uint8), (200, 200, 3, np.uint8),
             (150, 300, 3, np.uint8)]
    for h, w, ch, dtype in cases:
        img = scene(h, w, ch, dtype)
        view = Image(img.copy())
        view.resize((224, 224))
        diff = np.abs(view.img.astype(np.float64) - old_resize(img, (224, 224)).astype(np.float64))
        direction = "downscale" if h >= 224 and w >= 224 else "upscale"
        check(f"{h}x{w}x{ch} {np.dtype(dtype).name} ({direction})", diff.max() <= TOL[dtype],
              f"max|d|={diff.max():g}, {100 * (diff > 0).mean():.2f}% of pixels differ")


def test_zoom():
    print("\n2. Image.zoom (crop then scale back, skimage mode='symmetric')")
    for factor in (0.5, 0.75, 0.95):
        img = scene(416, 416)
        view = Image(img.copy())
        view.zoom(factor)
        cut = int(416 * (1 - factor)) // 2
        cropped = img[cut:416 - cut, cut:416 - cut].copy()
        diff = np.abs(view.img.astype(float) - old_resize(cropped, img.shape, "symmetric").astype(float))
        check(f"factor={factor}", diff.max() <= 1.0, f"max|Δ|={diff.max():g}")


def test_rotate():
    print("\n3. Image.rotate")
    for k in range(4):
        img = scene(224, 224)
        view = Image(img.copy())
        view.rotate(k * np.pi / 2)
        exact = np.rot90(img, k)
        drift = np.abs(view.img.astype(int) - old_rotate(img, k * np.pi / 2).astype(int)).max()
        check(f"{k * 90} deg is exactly np.rot90", np.array_equal(view.img, exact),
              f"difference from the old skimage path (which interpolated): {drift}/255")

    # 180 deg still holds for non-square images; 90 deg does not (it swaps the axes) and must
    # fall back to warpAffine.
    tall = scene(300, 416)
    view = Image(tall.copy())
    view.rotate(np.pi)
    check("180 deg on a non-square image", np.array_equal(view.img, np.rot90(tall, 2)))
    view = Image(tall.copy())
    view.rotate(np.pi / 2)
    check("90 deg on a non-square image keeps the shape", view.img.shape == tall.shape,
          str(view.img.shape))

    for angle, centre in ((np.pi / 2, (110, 120)), (0.37, (112, 112)), (np.pi, (90, 150))):
        img = scene(300, 300)
        view = Image(img.copy())
        view.rotate(angle, centre)
        diff = np.abs(view.img.astype(int) - old_rotate(img, angle, centre).astype(int)).max()
        check(f"angle {angle:.3f} about {centre} (the cornell/vmrd/ocid path)", diff <= 1,
              f"max|d|={diff}/255")


def old_draw(rects, shape):
    pos, ang, wid = np.zeros(shape), np.zeros(shape), np.zeros(shape)
    for gr in rects.grs:
        rr, cc = gr.compact_polygon_coords(shape)
        pos[rr, cc] = 1.0
        ang[rr, cc] = gr.angle
        wid[rr, cc] = gr.length
    return pos, ang, wid


def random_rects(n, as_int=False):
    grs = []
    for _ in range(n):
        centre = RNG.uniform(15, 205, 2)
        points = Grasp(centre, RNG.uniform(-np.pi / 2, np.pi / 2),
                       RNG.uniform(10, 50), RNG.uniform(5, 30)).as_gr.points
        grs.append(GraspRectangle(points.astype(int) if as_int else points))
    return GraspRectangles(grs)


def test_draw():
    print("\n4. GraspRectangles.draw (cv2.fillConvexPoly instead of skimage.draw.polygon)")
    area_diff = area_total = 0.0
    lit = wrong_winner = 0
    for _ in range(30):
        rects = random_rects(int(RNG.integers(5, 60)))
        old = old_draw(rects, (224, 224))
        new = rects.draw((224, 224))
        area_diff += np.abs(old[0] - new[0]).sum()
        area_total += old[0].sum()
        both = (old[0] > 0) & (new[0] > 0)
        lit += int(both.sum())
        wrong_winner += int((np.abs(old[1] - new[1])[both] > 1e-9).sum())
    check("pos area differs by under 5%", 100 * area_diff / area_total < 5.0,
          f"{100 * area_diff / area_total:.2f}%")
    check("under 1% of pixels change winning rect in angle", 100 * wrong_winner / lit < 1.0,
          f"{100 * wrong_winner / lit:.2f}%")

    rects = random_rects(20, as_int=True)
    old, new = old_draw(rects, (224, 224)), rects.draw((224, 224))
    check("integer coordinates (after gtbbs.zoom) still work",
          100 * np.abs(old[0] - new[0]).sum() / old[0].sum() < 5.0)

    empty = GraspRectangles([]).draw((224, 224))
    check("an empty rect list -> three zero arrays", all(m.shape == (224, 224) and m.sum() == 0
                                                  for m in empty))

    # The filled values must be the *original* rect's angle/length, not the 1/3-shrunk one's.
    one = random_rects(1)
    pos, ang, wid = one.draw((224, 224))
    gr = one.grs[0]
    on = pos > 0
    check("angle/length come from the original rect",
          bool(on.any()) and np.allclose(ang[on], gr.angle) and np.allclose(wid[on], gr.length))


def main():
    print(__doc__.strip().split("\n")[0])
    test_resize()
    test_zoom()
    test_rotate()
    test_draw()
    print()
    if FAILS:
        print(f"{len(FAILS)} check(s) FAILED: {', '.join(FAILS)}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
