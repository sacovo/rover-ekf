import os
from dataclasses import dataclass
from typing import List, Tuple

import cv2
from jax import numpy as jnp

from ekf.measurements import OrientationMeasurement
from ekf.sensors.camera import CameraConfig
from ekf.sensors.motor_state import to_left_and_right_speed
from ekf.sensors.tag_positions import TagSensor
from ekf.sensors.yocto_3d import multiply, quaternion_to_euler


@dataclass
class Camera:
    folder: str
    config: CameraConfig


@dataclass
class RecordingScenario:
    gyro_path: str
    motor_path: str
    cameras: List[Camera]
    tag_positions: List[Tuple[float, float]]
    tag_size: float = 20


class GyroLoader:
    def __init__(self, path) -> None:
        self.fp = open(path, "r")
        # Skip header
        self.fp.readline()
        # First line is initial orientation of the sensor
        line = self.fp.readline().strip().split(",")
        self.initial_quaternion = jnp.array([float(x) for x in line[1:]])
        self.step()

    def step(self):
        line = self.fp.readline().strip()

        if line == "":
            return

        line = line.split(",")
        self.timestamp = float(line[0])
        quaternion = jnp.array([float(x) for x in line[1:]])
        relative = multiply(quaternion, self.initial_quaternion)

        euler = quaternion_to_euler(*relative)

        self.value = OrientationMeasurement(jnp.array(euler), None)


class MotorLoader:
    def __init__(self, path) -> None:
        self.fp = open(path, "r")
        self.step()

    def step(self):
        line = self.fp.readline().strip()
        if line == "":
            return
        line = line.split()
        self.timestamp = float(line[0])
        self.value = to_left_and_right_speed(line[1:])


class CameraLoader:
    def __init__(self, path, camera_config, tag_positions, tag_size) -> None:
        self.path = path
        self.images = sorted(os.listdir(path))
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
        self.gyro_loader = GyroLoader(scenario.gyro_path)
        self.motor_loader = MotorLoader(scenario.motor_path)
        self.cameras = []

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

        if loader.timestamp > self.motor_loader.timestamp:
            self.motor_loader.step()

        return timestamp, value, motor, sensor
