from collections import defaultdict
from datetime import datetime
from functools import partial
from typing import Optional
from urllib import request

import cv2
import numpy as np
from erctag import TagDetector
from jax import jit
from jax import numpy as jnp
from jax import vmap
from jax.scipy.spatial.transform import Rotation

from ekf.config import DEBUG
from ekf.measurements import Measurement
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
        self,
        camera_config: CameraConfig,
        tag_size,
        tag_positions,
        confidence_factor=0.1,
        confidence_add=0.1,
        **kwargs,
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
        self.tag_positions = jnp.array(tag_positions)
        self.total_tags = len(tag_positions)
        self.confidence_factor = confidence_factor
        self.confidence_add = confidence_add

    def get_tag_positions(self, image, total_tags=10):
        result = self.detector.detect_tags(image)
        # Prepare output arrays
        positions = np.zeros((total_tags, 2))
        uncertainties = np.full((total_tags, 2), 1e32)

        tag_centers = defaultdict(list)
        tag_distances = defaultdict(lambda: 1000.0)

        for tag in result:
            if tag.distance > 6:
                continue

            if tag.tag_id >= total_tags:
                continue

            corners = np.array(tag.corners)
            center = corners.mean(axis=0)
            tag_centers[tag.tag_id].append(center)
            tag_distances[tag.tag_id] = tag.distance

        if len(tag_centers) == 0:
            return None, None

        # Extract tag information
        for tag_id, centers in tag_centers.items():
            # Take the one further down to avoid getting the
            # tags on top of the pole
            positions[tag_id] = np.max(centers, axis=0)

            uncertainties[tag_id] = (
                tag_distances[tag_id] * self.confidence_factor + self.confidence_add
            )

        if DEBUG:
            print("Tag", positions, uncertainties, datetime.now())

        return positions, uncertainties

    def next_frame(self):
        response = request.urlopen(self.url)
        img_array = np.asarray(bytearray(response.read()), dtype=np.uint8)
        img = cv2.imdecode(img_array, -1)  # type: ignore

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

    def get_reading(self, img) -> Optional[Measurement]:
        positions, uncertainties = self.get_tag_positions(img, self.total_tags)
        if positions is None or uncertainties is None:
            return None

        return Measurement(
            data=positions.flatten(),
            R=jnp.diagflat(uncertainties.flatten()),
        )

    def measure(self):
        img = self.next_frame()

        # R: für alle gleich
        measurement = self.get_reading(img)

        if measurement is not None and self.verbose:
            measurement.meta = img

        return measurement

    @partial(jit, static_argnums=0)
    def H(self, state):
        x, y, z, theta, pitch, roll = state
        rover_position = jnp.array([x, y, z])
        rover_orientation = Rotation.from_euler("zyx", jnp.array([theta, pitch, roll]))

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
