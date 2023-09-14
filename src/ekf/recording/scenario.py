import os
from dataclasses import dataclass
from typing import List, Tuple

import cv2
from jax import numpy as jnp
from scipy.spatial.transform import Rotation

from ekf.sensors.camera import CameraConfig
from ekf.sensors.motor_state import to_left_and_right_speed
from ekf.sensors.tag_positions import TagSensor
from ekf.sensors.yocto_3d import RotationSensor


@dataclass
class Camera:
    folder: str
    config: CameraConfig


@dataclass
class RecordingScenario:
    gyro_path: str
    gyro_orientation: Rotation
    gyro_initial: Rotation
    motor_path: str
    cameras: List[Camera]
    tag_positions: List[Tuple[float, float]]
    tag_size: float = 20


class GyroLoader(RotationSensor):
    def __init__(self, path, sensor_orientation, initial_orientation, **kwargs) -> None:
        self.fp = open(path, "r")
        # Skip header
        self.fp.readline()

        super().__init__(
            sensor_orientation=sensor_orientation,
            initial_orientation=initial_orientation,
            **kwargs
        )
        self.sensor = self
        self.step()

    def step(self):
        self.value = self.measure()

    def _get_quaternion(self):
        line = self.fp.readline().strip()
        line = line.split(",")
        if len(line) == 1:
            return self.quat

        self.timestamp = float(line[0])
        x, z, y, w = [float(x) for x in line[1:]]
        quaternion = jnp.array([x, y, z, w])
        self.quat = quaternion
        return quaternion


class MotorLoader:
    def __init__(self, path) -> None:
        self.fp = open(path, "r")
        self.step()
        self.done = False

    def step(self):
        line = self.fp.readline().strip()
        if line == "":
            self.done = True
            return
        timestamp, *line = line.split(",")
        self.timestamp = float(timestamp)
        self.value = to_left_and_right_speed(line)


class CameraLoader:
    def __init__(self, path, camera_config, tag_positions, tag_size) -> None:
        self.path = path
        self.images = sorted(filter(lambda x: x.endswith(".jpg"), os.listdir(path)))
        self.sensor = TagSensor(
            camera_config=camera_config,
            tag_positions=tag_positions,
            tag_size=tag_size,
        )
        self.idx = 0
        self.value = None
        self.timestamp = 0
        self.step()

    def step(self):
        if self.idx >= len(self.images):
            return
        img_path = self.images[self.idx]
        img = cv2.imread(os.path.join(self.path, img_path))
        reading = self.sensor.get_reading(img)

        self.value = reading
        # img_000727_1691417746026.jpg
        self.timestamp = int(img_path.split("_")[-1].split(".")[0]) / 1000

        self.idx += 1


class SensorLoader:
    def __init__(self, scenario: RecordingScenario) -> None:
        self.scenario = scenario
        self.gyro_loader = GyroLoader(
            scenario.gyro_path, scenario.gyro_orientation, scenario.gyro_initial
        )
        self.motor_loader = MotorLoader(scenario.motor_path)
        self.cameras: List[CameraLoader] = []

        for camera in scenario.cameras:
            self.cameras.append(
                CameraLoader(
                    camera.folder,
                    camera.config,
                    scenario.tag_positions,
                    scenario.tag_size,
                )
            )

    def next_reading(self):
        loader = self.gyro_loader

        for camera in self.cameras:
            if camera.timestamp < loader.timestamp:
                loader = camera

        value = loader.value

        timestamp = loader.timestamp
        motor = self.motor_loader.value
        sensor = loader.sensor

        loader.step()

        while (
            loader.timestamp > self.motor_loader.timestamp
            and not self.motor_loader.done
        ):
            self.motor_loader.step()
        motor = self.motor_loader.value

        return timestamp, value, motor, sensor
