import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:  # pragma: no cover - only needed on the real-robot path
    rs = None


class RealSenseCamera:
    """
    Thin wrapper around an Intel RealSense RGB-D camera.

    Only used by the real-robot path (``inference/grasp_generator.py``); training and
    evaluation never touch this module.
    """

    def __init__(self, device_id, width=640, height=480, fps=30):
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps

        self.pipeline = None
        self.scale = None
        self.intrinsics = None

    def connect(self):
        if rs is None:
            raise ImportError(
                "pyrealsense2 is required to use RealSenseCamera. "
                "Install it with `pip install pyrealsense2`."
            )

        # Start the stream
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(str(self.device_id))
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        cfg = self.pipeline.start(config)

        # Determine intrinsics
        rgb_profile = cfg.get_stream(rs.stream.color)
        self.intrinsics = rgb_profile.as_video_stream_profile().get_intrinsics()

        # Determine depth scale
        self.scale = cfg.get_device().first_depth_sensor().get_depth_scale()

    def get_image_bundle(self):
        """
        :return: dict with the RGB frame and the depth frame aligned to it (H x W x 1)
        """
        frames = self.pipeline.wait_for_frames()

        align = rs.align(rs.stream.color)
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.first(rs.stream.color)
        aligned_depth_frame = aligned_frames.get_depth_frame()

        depth_image = np.asarray(aligned_depth_frame.get_data(), dtype=np.float32)
        depth_image *= self.scale
        color_image = np.asanyarray(color_frame.get_data())

        depth_image = np.expand_dims(depth_image, axis=2)

        return {
            'rgb': color_image,
            'aligned_depth': depth_image,
        }

    def plot_image_bundle(self):
        import matplotlib.pyplot as plt

        images = self.get_image_bundle()
        rgb = images['rgb']
        depth = images['aligned_depth']

        fig, ax = plt.subplots(1, 2, squeeze=False)
        ax[0][0].imshow(rgb)
        m, s = np.nanmean(depth), np.nanstd(depth)
        ax[0][1].imshow(depth.squeeze(axis=2), vmin=m - s, vmax=m + s, cmap='gray')
        ax[0][0].set_title('rgb')
        ax[0][1].set_title('aligned_depth')

        plt.show()
