import pickle
from io import BytesIO
from pathlib import Path

from cv2 import os
from erctag.detection import cv2
from jax import numpy as jnp

from ekf.measurements import Measurement
from ekf.sensors.sensor import Sensor
from ekf.sensors.tag_positions import CameraConfig, TagSensor

source_path = Path(__file__).resolve()

source_dir = source_path.parent
url = "file://" + os.path.join(source_dir, "imgs/test_tag.jpg")


def test_pickling():
    measurement = Measurement(jnp.zeros((9, 1)), jnp.identity(5))
    measurement.meta = cv2.imread(url)  # type: ignore
    sensor = Sensor()

    io = BytesIO()

    pickle.dump((measurement, sensor), io)
    io.seek(0)

    read_measurement, read_sensor = pickle.load(io)

    assert read_measurement is not None
    assert read_sensor is not None


def test_pickling_tag():
    config = CameraConfig(position=[0, 0, 0], orientation=[0, 0, 0])
    sensor = TagSensor(config, 20, [[0, 1, 2]])

    io = BytesIO()
    pickle.dump(sensor.config, io)
    io.seek(0)

    assert pickle.load(io) is not None
