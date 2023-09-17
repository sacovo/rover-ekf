import cv2
from joblib import os

from ekf.sensors.camera import CameraConfig

from .tag_positions import TagSensor


class FolderTagSensor(TagSensor):
    def __init__(
        self,
        camera_config: CameraConfig,
        tag_size,
        tag_positions,
        **kwargs,
    ) -> None:
        super().__init__(camera_config, tag_size, tag_positions, **kwargs)
        self.images = [
            os.path.join(self.url, path) for path in sorted(os.listdir(self.url))
        ]

        self.counter = 0

    def next_frame(self):
        img = cv2.imread(self.images[self.counter])  # type: ignore
        self.counter += 1
        return img
