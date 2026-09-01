import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.draw import polygon
from skimage.feature import peak_local_max

cv2.setNumThreads(0)

# `cv2.fillConvexPoly` fills every pixel an edge passes through, while `skimage.draw.polygon`
# only takes pixels whose *centre* lies inside the polygon -- exactly half a pixel along each
# edge. Insetting the rectangle by 0.5 px per side before filling compensates: the difference
# drops to 2.7% of the area (from 16%), a mean offset of -1.2 px on a ~165 px rect. Measured on
# 1,500 random GA++-sized rectangles.
_FILL_INSET = 0.5
# Coordinates are handed to cv2 as 1/32-px fixed point; without this the vertices are rounded
# to integers and the difference doubles.
_FILL_SHIFT = 5


def _gr_text_to_no(l, offset=(0, 0)):
    """
    Transform a single point from a Cornell file line to a pair of ints.
    :param l: Line from Cornell grasp file (str)
    :param offset: Offset to apply to point positions
    :return: Point [y, x]
    """
    x, y = l.split()
    return [int(round(float(y))) - offset[0], int(round(float(x))) - offset[1]]


def _grasp_anything_format(grasp: list):
    _, x, y, w, h, theta = grasp
    # index based on row, column (y,x), and the Grasp-Anything dataset's angles are flipped around an axis.
    return Grasp(np.array([y, x]), -theta / 180.0 * np.pi, w, h).as_gr


class GraspRectangles:
    """
    Convenience class for loading and operating on sets of Grasp Rectangles.
    """

    def __init__(self, grs=None):
        if grs:
            self.grs = grs
        else:
            self.grs = []

    def __getitem__(self, item):
        return self.grs[item]

    def __iter__(self):
        return self.grs.__iter__()

    def __getattr__(self, attr):
        """
        Test if GraspRectangle has the desired attr as a function and call it.
        """
        # Fuck yeah python.
        if hasattr(GraspRectangle, attr) and callable(getattr(GraspRectangle, attr)):
            return lambda *args, **kwargs: list(
                map(lambda gr: getattr(gr, attr)(*args, **kwargs), self.grs)
            )
        else:
            raise AttributeError(
                "Couldn't find function %s in BoundingBoxes or BoundingBox" % attr
            )

    @classmethod
    def load_from_array(cls, arr):
        """
        Load grasp rectangles from numpy array.
        :param arr: Nx4x2 array, where each 4x2 array is the 4 corner pixels of a grasp rectangle.
        :return: GraspRectangles()
        """
        grs = []
        for i in range(arr.shape[0]):
            grp = arr[i, :, :].squeeze()
            if grp.max() == 0:
                break
            else:
                grs.append(GraspRectangle(grp))
        return cls(grs)

    @classmethod
    def load_from_cornell_file(cls, fname):
        """
        Load grasp rectangles from a Cornell dataset grasp file.
        :param fname: Path to text file.
        :return: GraspRectangles()
        """
        grs = []
        with open(fname) as f:
            while True:
                # Load 4 lines at a time, corners of bounding box.
                p0 = f.readline()
                if not p0:
                    break  # EOF
                p1, p2, p3 = f.readline(), f.readline(), f.readline()
                try:
                    gr = np.array(
                        [
                            _gr_text_to_no(p0),
                            _gr_text_to_no(p1),
                            _gr_text_to_no(p2),
                            _gr_text_to_no(p3),
                        ]
                    )

                    grs.append(GraspRectangle(gr))

                except ValueError:
                    # Some files contain weird values.
                    continue

        return cls(grs)

    @classmethod
    def load_from_ocid_grasp_file(cls, fname):
        """
        Load grasp rectangles from a Cornell dataset grasp file.
        :param fname: Path to text file.
        :return: GraspRectangles()
        """
        grs = []
        with open(fname) as f:
            while True:
                # Load 4 lines at a time, corners of bounding box.
                p0 = f.readline()
                if not p0:
                    break  # EOF
                p1, p2, p3 = f.readline(), f.readline(), f.readline()
                try:
                    gr = np.array(
                        [
                            _gr_text_to_no(p0),
                            _gr_text_to_no(p1),
                            _gr_text_to_no(p2),
                            _gr_text_to_no(p3),
                        ]
                    )

                    grs.append(GraspRectangle(gr))

                except ValueError:
                    # Some files contain weird values.
                    continue

        return cls(grs)

    @classmethod
    def load_from_vmrd_file(cls, fname):
        """
        Load grasp rectangles from a VMRD dataset grasp file.
        :param fname: Path to text file.
        :return: GraspRectangles()
        """
        grs = []
        with open(fname) as f:
            grasp_lines = f.readlines()
            for grasp_line in grasp_lines:
                # x1, y1, x2, y2, x3, y3, x4, y4 = list(map(lambda x: int(round(float(x))), grasp_line.split(' ')[:8]))
                y1, x1, y2, x2, y3, x3, y4, x4 = list(
                    map(lambda x: int(round(float(x))), grasp_line.split(" ")[:8])
                )
                try:
                    gr = np.array(
                        [
                            [x1, y1],
                            [x2, y2],
                            [x3, y3],
                            [x4, y4],
                        ]
                    )

                    grs.append(GraspRectangle(gr))

                except ValueError:
                    # Some files contain weird values.
                    continue

        return cls(grs)

    @classmethod
    def load_from_jacquard_file(cls, fname, scale=1.0):
        """
        Load grasp rectangles from a Jacquard dataset file.
        :param fname: Path to file.
        :param scale: Scale to apply (e.g. if resizing images)
        :return: GraspRectangles()
        """
        grs = []
        with open(fname) as f:
            for l in f:
                x, y, theta, w, h = [float(v) for v in l[:-1].split(";")]
                # index based on row, column (y,x), and the Jacquard dataset's angles are flipped around an axis.
                grs.append(Grasp(np.array([y, x]), -theta / 180.0 * np.pi, w, h).as_gr)
        grs = cls(grs)
        grs.scale(scale)
        return grs

    @classmethod
    def load_from_grasp_anything_file(cls, fname, scale=1.0):
        """
        Load grasp rectangles from a Grasp-Anything dataset grasp file.
        :param fname: Path to text file.
        :return: GraspRectangles()
        """
        grs = None
        with open(fname, "rb") as f:
            pos_grasps = torch.load(f)
        # add_fn = fname.replace("positive_grasp", "negative_grasp")
        # with open(fname, 'rb') as f:
        #     neg_grasps = torch.load(f)
        # grasps = torch.cat((pos_grasps, neg_grasps), dim=0).tolist()
        grasps = pos_grasps.tolist()
        grs = list(map(lambda x: _grasp_anything_format(x), grasps))

        grs = cls(grs)
        grs.scale(scale)
        return grs

    def append(self, gr):
        """
        Add a grasp rectangle to this GraspRectangles object
        :param gr: GraspRectangle
        """
        self.grs.append(gr)

    def copy(self):
        """
        :return: A deep copy of this object and all of its GraspRectangles.
        """
        new_grs = GraspRectangles()
        for gr in self.grs:
            new_grs.append(gr.copy())
        return new_grs

    def show(self, ax=None, shape=None):
        """
        Draw all GraspRectangles on a matplotlib plot.
        :param ax: (optional) existing axis
        :param shape: (optional) Plot shape if no existing axis
        """
        if ax is None:
            f = plt.figure()
            ax = f.add_subplot(1, 1, 1)
            ax.imshow(np.zeros(shape))
            ax.axis([0, shape[1], shape[0], 0])
            self.plot(ax)
            plt.show()
        else:
            self.plot(ax)

    def _compact_geometry(self):
        """
        A vectorised form of `GraspRectangle.{center, angle, length, width}` for the *whole* list.

        The original reads those four properties through a Python `for` loop, each property
        calling arctan2/sqrt on two scalars -- that alone accounted for ~40% of `draw()`. The
        formulas are preserved literally, including `center`'s `.astype(int)` (truncation
        towards zero, not rounding).

        :return: (poly, angle, length) -- `poly` is (N, 4, 2) of [y, x] coordinates for the
                 shrunk rectangle (1/3 of the length) with `_FILL_INSET` subtracted;
                 `angle`/`length` are those of the original rect, since those are the values
                 that go into ang_out/width_out.
        """
        pts = np.stack([gr.points for gr in self.grs]).astype(float)

        centre = pts.mean(axis=1).astype(int)
        dy = pts[:, 1, 0] - pts[:, 0, 0]
        dx = pts[:, 1, 1] - pts[:, 0, 1]
        angle = (np.arctan2(-dy, dx) + np.pi / 2) % np.pi - np.pi / 2
        length = np.sqrt(dx**2 + dy**2)
        wdy = pts[:, 2, 1] - pts[:, 1, 1]
        wdx = pts[:, 2, 0] - pts[:, 1, 0]
        rect_width = np.sqrt(wdx**2 + wdy**2)

        # `Grasp(centre, angle, length / 3, width).as_gr`, vectorised.
        half_l = np.maximum(length / 3.0 - 2 * _FILL_INSET, 1e-3) / 2.0
        half_w = np.maximum(rect_width - 2 * _FILL_INSET, 1e-3) / 2.0
        xo, yo = np.cos(angle), np.sin(angle)
        cy, cx = centre[:, 0].astype(float), centre[:, 1].astype(float)
        y1, x1 = cy + half_l * yo, cx - half_l * xo
        y2, x2 = cy - half_l * yo, cx + half_l * xo
        poly = np.stack(
            [
                np.stack([y1 - half_w * xo, x1 - half_w * yo], axis=-1),
                np.stack([y2 - half_w * xo, x2 - half_w * yo], axis=-1),
                np.stack([y2 + half_w * xo, x2 + half_w * yo], axis=-1),
                np.stack([y1 + half_w * xo, x1 + half_w * yo], axis=-1),
            ],
            axis=1,
        )
        return poly, angle, length

    def draw(self, shape, position=True, angle=True, width=True):
        """
        Plot all GraspRectangles as solid rectangles in a numpy array, e.g. as network training data.

        Filled with `cv2.fillConvexPoly` instead of `skimage.draw.polygon`: 0.005 ms versus
        0.034 ms per rect. At ~30 rects per sample, `draw()` goes from 1.55 ms to ~0.3 ms --
        once `Image.resize/rotate` moved to cv2, this was the loader's largest remaining cost.

        The overwrite order is preserved: later rects paint over earlier ones, as in the
        original loop.

        :param shape: output shape
        :param position: If True, Q output will be produced
        :param angle: If True, Angle output will be produced
        :param width: If True, Width output will be produced
        :return: Q, Angle, Width outputs (or None)
        """
        pos_out = np.zeros(shape) if position else None
        ang_out = np.zeros(shape) if angle else None
        width_out = np.zeros(shape) if width else None

        if not self.grs:
            return pos_out, ang_out, width_out

        poly, gr_angle, gr_length = self._compact_geometry()
        # cv2 takes (x, y), so swap the axes, then convert to fixed point.
        quads = np.round(poly[:, :, ::-1] * (1 << _FILL_SHIFT)).astype(np.int32)

        for i in range(len(quads)):
            q = quads[i]
            if position:
                cv2.fillConvexPoly(pos_out, q, 1.0, cv2.LINE_8, _FILL_SHIFT)
            if angle:
                cv2.fillConvexPoly(
                    ang_out, q, float(gr_angle[i]), cv2.LINE_8, _FILL_SHIFT
                )
            if width:
                cv2.fillConvexPoly(
                    width_out, q, float(gr_length[i]), cv2.LINE_8, _FILL_SHIFT
                )

        return pos_out, ang_out, width_out

    def to_array(self, pad_to=0):
        """
        Convert all GraspRectangles to a single array.
        :param pad_to: Length to 0-pad the array along the first dimension
        :return: Nx4x2 numpy array
        """
        a = np.stack([gr.points for gr in self.grs])
        if pad_to:
            if pad_to > len(self.grs):
                a = np.concatenate((a, np.zeros((pad_to - len(self.grs), 4, 2))))
        return a.astype(int)

    @property
    def center(self):
        """
        Compute mean center of all GraspRectangles
        :return: float, mean centre of all GraspRectangles
        """
        points = [gr.points for gr in self.grs]
        return np.mean(np.vstack(points), axis=0).astype(int)


class GraspRectangle:
    """
    Representation of a grasp in the common "Grasp Rectangle" format.
    """

    def __init__(self, points):
        self.points = points

    def __str__(self):
        return str(self.points)

    @property
    def angle(self):
        """
        :return: Angle of the grasp to the horizontal.
        """
        dx = self.points[1, 1] - self.points[0, 1]
        dy = self.points[1, 0] - self.points[0, 0]
        return (np.arctan2(-dy, dx) + np.pi / 2) % np.pi - np.pi / 2

    @property
    def as_grasp(self):
        """
        :return: GraspRectangle converted to a Grasp
        """
        return Grasp(self.center, self.angle, self.length, self.width)

    @property
    def center(self):
        """
        :return: Rectangle center point
        """
        return self.points.mean(axis=0).astype(int)

    @property
    def length(self):
        """
        :return: Rectangle length (i.e. along the axis of the grasp)
        """
        dx = self.points[1, 1] - self.points[0, 1]
        dy = self.points[1, 0] - self.points[0, 0]
        return np.sqrt(dx**2 + dy**2)

    @property
    def width(self):
        """
        :return: Rectangle width (i.e. perpendicular to the axis of the grasp)
        """
        dy = self.points[2, 1] - self.points[1, 1]
        dx = self.points[2, 0] - self.points[1, 0]
        return np.sqrt(dx**2 + dy**2)

    def polygon_coords(self, shape=None):
        """
        :param shape: Output Shape
        :return: Indices of pixels within the grasp rectangle polygon.
        """
        return polygon(self.points[:, 0], self.points[:, 1], shape)

    def compact_polygon_coords(self, shape=None):
        """
        :param shape: Output shape
        :return: Indices of pixels within the centre thrid of the grasp rectangle.
        """
        return Grasp(
            self.center, self.angle, self.length / 3, self.width
        ).as_gr.polygon_coords(shape)

    def iou(self, gr, angle_threshold=np.pi / 6):
        """
        Compute IoU with another grasping rectangle
        :param gr: GraspingRectangle to compare
        :param angle_threshold: Maximum angle difference between GraspRectangles
        :return: IoU between Grasp Rectangles
        """
        if (
            abs((self.angle - gr.angle + np.pi / 2) % np.pi - np.pi / 2)
            > angle_threshold
        ):
            return 0

        rr1, cc1 = self.polygon_coords()
        rr2, cc2 = polygon(gr.points[:, 0], gr.points[:, 1])

        try:
            r_max = max(rr1.max(), rr2.max()) + 1
            c_max = max(cc1.max(), cc2.max()) + 1
            # Grasps at the image border (or pushed outside by zoom) have negative
            # coordinates. Indexing the canvas with a negative value wraps around to the end of
            # the array and creates a phantom intersection, so both polygons must be shifted to
            # origin 0 before drawing.
            r_min = min(rr1.min(), rr2.min(), 0)
            c_min = min(cc1.min(), cc2.min(), 0)
        except:
            return 0

        canvas = np.zeros((r_max - r_min, c_max - c_min))
        canvas[rr1 - r_min, cc1 - c_min] += 1
        canvas[rr2 - r_min, cc2 - c_min] += 1
        union = np.sum(canvas > 0)
        if union == 0:
            return 0
        intersection = np.sum(canvas == 2)
        return intersection / union

    def copy(self):
        """
        :return: Copy of self.
        """
        return GraspRectangle(self.points.copy())

    def offset(self, offset):
        """
        Offset grasp rectangle
        :param offset: array [y, x] distance to offset
        """
        self.points += np.array(offset).reshape((1, 2))

    def rotate(self, angle, center):
        """
        Rotate grasp rectangle
        :param angle: Angle to rotate (in radians)
        :param center: Point to rotate around (e.g. image center)
        """
        R = np.array(
            [
                [np.cos(-angle), np.sin(-angle)],
                [-1 * np.sin(-angle), np.cos(-angle)],
            ]
        )
        c = np.array(center).reshape((1, 2))
        self.points = ((np.dot(R, (self.points - c).T)).T + c).astype(int)

    def scale(self, factor):
        """
        :param factor: Scale grasp rectangle by factor
        """
        if factor == 1.0:
            return
        self.points *= factor

    def plot(self, ax, color=None):
        """
        Plot grasping rectangle.
        :param ax: Existing matplotlib axis
        :param color: matplotlib color code (optional)
        """
        points = np.vstack((self.points, self.points[0]))
        ax.plot(points[:, 1], points[:, 0], color=color)

    def zoom(self, factor, center):
        """
        Zoom grasp rectangle by given factor.
        :param factor: Zoom factor
        :param center: Zoom zenter (focus point, e.g. image center)
        """
        T = np.array([[1 / factor, 0], [0, 1 / factor]])
        c = np.array(center).reshape((1, 2))
        self.points = ((np.dot(T, (self.points - c).T)).T + c).astype(int)


class Grasp:
    """
    A Grasp represented by a center pixel, rotation angle and gripper width (length)
    """

    def __init__(self, center, angle, length=60, width=30):
        self.center = center
        self.angle = (
            angle  # Positive angle means rotate anti-clockwise from horizontal.
        )
        self.length = length
        self.width = width

    @property
    def as_gr(self):
        """
        Convert to GraspRectangle
        :return: GraspRectangle representation of grasp.
        """
        xo = np.cos(self.angle)
        yo = np.sin(self.angle)

        y1 = self.center[0] + self.length / 2 * yo
        x1 = self.center[1] - self.length / 2 * xo
        y2 = self.center[0] - self.length / 2 * yo
        x2 = self.center[1] + self.length / 2 * xo

        return GraspRectangle(
            np.array(
                [
                    [y1 - self.width / 2 * xo, x1 - self.width / 2 * yo],
                    [y2 - self.width / 2 * xo, x2 - self.width / 2 * yo],
                    [y2 + self.width / 2 * xo, x2 + self.width / 2 * yo],
                    [y1 + self.width / 2 * xo, x1 + self.width / 2 * yo],
                ]
            ).astype(float)
        )

    def max_iou(self, grs):
        """
        Return maximum IoU between self and a list of GraspRectangles
        :param grs: List of GraspRectangles
        :return: Maximum IoU with any of the GraspRectangles
        """
        self_gr = self.as_gr
        max_iou = 0
        for gr in grs:
            iou = self_gr.iou(gr)
            max_iou = max(max_iou, iou)
        return max_iou

    def plot(self, ax, color=None):
        """
        Plot Grasp
        :param ax: Existing matplotlib axis
        :param color: (optional) color
        """
        self.as_gr.plot(ax, color)

    def to_jacquard(self, scale=1):
        """
        Output grasp in "Jacquard Dataset Format" (https://jacquard.liris.cnrs.fr/database.php)
        :param scale: (optional) scale to apply to grasp
        :return: string in Jacquard format
        """
        # Output in jacquard format.
        return "%0.2f;%0.2f;%0.2f;%0.2f;%0.2f" % (
            self.center[1] * scale,
            self.center[0] * scale,
            -1 * self.angle * 180 / np.pi,
            self.length * scale,
            self.width * scale,
        )


def detect_grasps(q_img, ang_img, width_img=None, no_grasps=1):
    """
    Detect grasps in a network output.
    :param q_img: Q image network output
    :param ang_img: Angle image network output
    :param width_img: (optional) Width image network output
    :param no_grasps: Max number of grasps to return
    :return: list of Grasps
    """
    local_max = peak_local_max(
        q_img, min_distance=20, threshold_abs=0.2, num_peaks=no_grasps
    )

    grasps = []
    for grasp_point_array in local_max:
        grasp_point = tuple(grasp_point_array)

        grasp_angle = ang_img[grasp_point]

        g = Grasp(grasp_point, grasp_angle)
        if width_img is not None:
            g.length = width_img[grasp_point]
            g.width = g.length / 2

        grasps.append(g)

    return grasps
