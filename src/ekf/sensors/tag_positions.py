from functools import partial
from urllib import request

import cv2
import numpy as np
from erctag import TagDetector
from jax import jit
from jax import numpy as jnp
from jax import vmap

from ekf.measurements import TagMeasurement
from ekf.sensors.camera import CameraConfig
from ekf.tag_calculations import ypr_to_rotation_matrix

from .sensor import Sensor


class TagSensor(Sensor):
    name = "tags"

    def __init__(
        self, camera_config: CameraConfig, tag_size, tag_positions, **kwargs
    ) -> None:
        super().__init__(
            timeout=kwargs.pop("timeout", None), name=kwargs.pop("name", self.name)
        )
        self.url = camera_config.url
        self.camera = camera_config

        self.detector = TagDetector(
            calibration=camera_config.calibration,
            distortion=camera_config.distortion,
        )
        self.tag_size = tag_size
        self.tag_positions = tag_positions
        self.total_tags = len(tag_positions)

    def get_tag_positions(self, image, total_tags=10):
        result = self.detector.detect_tags(image)
        # Prepare output arrays
        positions = jnp.zeros((total_tags, 3))
        uncertainties = jnp.full((total_tags, 3), jnp.inf)

        # Extract tag information
        for tag in result:
            if tag.tag_id >= total_tags:
                continue  # Ignore tags with IDs outside our expected range

            positions[tag.tag_id] = tag.t

            uncertainties[tag.tag_id] = 1.0

        return positions, uncertainties

    def next_frame(self):
        response = request.urlopen(self.url)
        img_array = jnp.asarray(bytearray(response.read()), dtype=jnp.uint8)

        img_array = np.asarray(bytearray(response.read()), dtype=np.uint8)
        img = cv2.imdecode(img_array, -1)

        return img

    @staticmethod
    def calculate_tag_position(tag_position, camera_position, camera_orientation):
        # Calculate the position of the tag in the camera frame
        tag_position_camera_frame = jnp.dot(
            ypr_to_rotation_matrix(camera_orientation),
            (tag_position - camera_position),
        )
        return tag_position_camera_frame

    def get_reading(self, img):
        positions, uncertainties = self.get_tag_positions(img)

        return TagMeasurement(
            data=positions.flatten(),
            R=uncertainties,
        )

    def measure(self):
        img = self.next_frame()

        # R: für alle gleich
        return self.get_reading(img)

    @partial(jit, static_argnums=0)
    def H(self, state):
        x, y, z, _, _, _, theta, pitch, roll = state
        rover_position = jnp.array([x, y, z])
        rover_orientation = jnp.array([theta, pitch, roll])

        # Compute camera position and orientation in global frame
        camera_position = rover_position + self.camera.position
        camera_orientation = rover_orientation + self.camera.orientation

        # Vector from camera to each tag in camera frame

        tag_vectors_camera_frame = vmap(
            partial(
                self.calculate_tag_position,
                camera_position=camera_position,
                camera_orientation=camera_orientation,
            )
        )(self.tag_positions)

        return tag_vectors_camera_frame.flatten()
