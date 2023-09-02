from collections import defaultdict
from functools import partial
from urllib import request

import cv2
import numpy as np
from erctag import TagDetector
from jax import jit
from jax import numpy as jnp
from jax import vmap
from jax.scipy.spatial.transform import Rotation

from ekf.measurements import TagMeasurement
from ekf.sensors.camera import CameraConfig
from ekf.tag_calculations import (
    apply_distortion,
    project_to_image,
    tag_to_camera_coordinates,
)

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
            n_jobs=1,
        )
        self.tag_size = tag_size
        self.tag_positions = tag_positions
        self.total_tags = len(tag_positions)

    def get_tag_positions(self, image, total_tags=10):
        result = self.detector.detect_tags(image)
        # Prepare output arrays
        positions = np.zeros((total_tags, 2))
        uncertainties = np.full((total_tags, 2), 1.0)

        tag_centers = defaultdict(list)
        for tag in result:
            if tag.distance > 7:
                continue

            if tag.tag_id >= total_tags:
                continue

            corners = np.array(tag.corners)
            center = corners.mean(axis=0)
            tag_centers[tag.tag_id].append(center)

        if len(tag_centers) == 0:
            return None, None
        print(tag_centers)
        # Extract tag information
        for tag_id, centers in tag_centers.items():
            positions[tag_id] = np.mean(centers, axis=0)

            uncertainties[tag_id] = 0.1

        return positions, uncertainties

    def next_frame(self):
        response = request.urlopen(self.url)
        img_array = np.asarray(bytearray(response.read()), dtype=np.uint8)
        img = cv2.imdecode(img_array, -1)

        return img

    @staticmethod
    def calculate_tag_position(
        tag_position, camera_position, camera_orientation, calibration, distortion
    ):
        # Calculate the position of the tag in the camera frame
        tag_in_camera_coordinates = tag_to_camera_coordinates(
            tag_position, camera_position, camera_orientation
        )
        projected_position = project_to_image(tag_in_camera_coordinates, calibration)
        distorted_position = apply_distortion(
            projected_position, calibration, distortion
        )

        return distorted_position

    def get_reading(self, img):
        positions, uncertainties = self.get_tag_positions(img, self.total_tags)
        if positions is None or uncertainties is None:
            return None

        return TagMeasurement(
            data=positions.flatten(),
            R=uncertainties.flatten(),
        )

    def measure(self):
        img = self.next_frame()

        # R: für alle gleich
        return self.get_reading(img)

    @partial(jit, static_argnums=0)
    def H(self, state):
        x, y, z, _, _, _, theta, pitch, roll = state
        rover_position = jnp.array([x, y, z])
        rover_orientation = Rotation.from_euler("ZYX", jnp.array([theta, pitch, roll]))

        # Compute camera position and orientation in global frame
        camera_position = rover_position + self.camera.position
        camera_orientation = rover_orientation * self.camera.orientation

        # Vector from camera to each tag in camera frame

        tag_vectors_camera_frame = vmap(
            partial(
                self.calculate_tag_position,
                camera_position=camera_position,
                camera_orientation=camera_orientation,
                calibration=self.camera.calibration,
                distortion=self.camera.distortion,
            )
        )(self.tag_positions)

        return tag_vectors_camera_frame.flatten()
