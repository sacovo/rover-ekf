from functools import partial
from urllib import request

import cv2
import numpy as np
from jax import jit
from jax import numpy as jnp
from jax import vmap

from build.lib.ekf.measurements import Measurement

from .sensor import Sensor
from .tag_positions import CameraConfig


def ypr_to_rotation_matrix(angles):
    yaw, pitch, roll = angles  # convert degrees to radian

    # rotation matrix for yaw
    R_yaw = jnp.array(
        [
            [jnp.cos(yaw), -jnp.sin(yaw), 0],
            [jnp.sin(yaw), jnp.cos(yaw), 0],
            [0, 0, 1],
        ]
    )

    # rotation matrix for pitch
    R_pitch = jnp.array(
        [
            [jnp.cos(pitch), 0, jnp.sin(pitch)],
            [0, 1, 0],
            [-jnp.sin(pitch), 0, jnp.cos(pitch)],
        ]
    )

    # rotation matrix for roll
    R_roll = jnp.array(
        [
            [1, 0, 0],
            [0, jnp.cos(roll), -jnp.sin(roll)],
            [0, jnp.sin(roll), jnp.cos(roll)],
        ]
    )

    # Combined rotation matrix is R = R_yaw * R_pitch * R_roll
    R = jnp.dot(R_yaw, jnp.dot(R_pitch, R_roll))

    return R


def euler_to_rotation_matrix(theta):
    R_x = jnp.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, jnp.cos(theta[0]), -jnp.sin(theta[0])],
            [0.0, jnp.sin(theta[0]), jnp.cos(theta[0])],
        ]
    )

    R_y = jnp.array(
        [
            [jnp.cos(theta[1]), 0, jnp.sin(theta[1])],
            [0, 1, 0],
            [-jnp.sin(theta[1]), 0, jnp.cos(theta[1])],
        ]
    )

    R_z = jnp.array(
        [
            [jnp.cos(theta[2]), -jnp.sin(theta[2]), 0],
            [jnp.sin(theta[2]), jnp.cos(theta[2]), 0],
            [0, 0, 1],
        ]
    )

    R = jnp.dot(R_z, jnp.dot(R_y, R_x))

    return R


class ArucoTagSensor(Sensor):
    name = "tags"

    def __init__(
        self, camera_config: CameraConfig, tag_size, tag_positions, **kwargs
    ) -> None:
        super().__init__(timeout=kwargs.pop("timeout", None), name=kwargs.pop("name"))

        dictionary = cv2.aruco.getPredefinedDictionary(  # type: ignore
            cv2.aruco.DICT_ARUCO_ORIGINAL,  # type: ignore
        )
        parameters = cv2.aruco.DetectorParameters()  # type: ignore
        self.detector = cv2.aruco.ArucoDetector(dictionary, parameters)  # type: ignore

        self.url = camera_config.url
        self.camera_parameters = camera_config
        self.tag_size = tag_size
        self.total_tags = 10
        self.tag_positions = tag_positions

    def get_tag_positions(self, image, total_tags=10):
        (corners, ids, rejected) = self.detector.detectMarkers(image)

        positions = jnp.zeros((total_tags, 3))
        uncertainties = jnp.full((total_tags, 3), jnp.inf)

        if ids is None:
            return positions, uncertainties

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(  # type: ignore
            corners,
            0.05,
            self.camera_parameters.calibration,
            self.camera_parameters.distortion,
        )

        # Extract tag information
        for rvec, tvec, tag_id in zip(rvecs, tvecs, ids):
            if tag_id > self.total_tags:
                continue  # Ignore tags with IDs outside our expected range

            positions[tag_id] = tvec

            uncertainties[tag_id] = 0.0

        return positions, uncertainties

    @staticmethod
    def calculate_tag_position(tag_position, camera_position, camera_orientation):
        # Calculate the position of the tag in the camera frame
        tag_position_camera_frame = jnp.dot(
            ypr_to_rotation_matrix(camera_orientation),
            (tag_position - camera_position),
        )
        return tag_position_camera_frame

    def measure(self):
        response = request.urlopen(self.url)
        img_array = np.asarray(bytearray(response.read()), dtype=np.uint8)
        img = cv2.imdecode(img_array, -1)  # type: ignore
        cv2.imshow("", img)  # type: ignore
        cv2.waitKey(0)  # type: ignore

        # R: für alle gleich
        positions, uncertainties = self.get_tag_positions(img)

        return Measurement(
            data=positions.flatten(),
            R=uncertainties,
        )

    @partial(jit, static_argnums=0)
    def H(self, state):
        x, y, z, _, _, _, theta, pitch, roll = state
        rover_position = jnp.array([x, y, z])
        rover_orientation = jnp.array([theta, pitch, roll])

        # Compute camera position and orientation in global frame
        camera_position = rover_position + self.camera_parameters.position
        camera_orientation = rover_orientation + self.camera_parameters.orientation

        # Vector from camera to each tag in camera frame

        tag_vectors_camera_frame = vmap(
            partial(
                self.calculate_tag_position,
                camera_position=camera_position,
                camera_orientation=camera_orientation,
            )
        )(self.tag_positions)

        return tag_vectors_camera_frame.flatten()
