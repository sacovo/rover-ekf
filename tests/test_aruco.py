import os
from pathlib import Path
from urllib import request

import cv2
import numpy as np

from ekf.sensors.tag_positions import CameraConfig, TagSensor

source_path = Path(__file__).resolve()
source_dir = source_path.parent
url = "file://" + os.path.join(source_dir, "imgs/test_tag.jpg")
calibration = os.path.join(source_dir, "calibration.pkl")


def test_load_image():
    response = request.urlopen(url)
    img_array = np.asarray(bytearray(response.read()), dtype=np.uint8)
    img = cv2.imdecode(img_array, -1)

    assert img.shape


def test_with_calibration():
    config = CameraConfig(
        url=url,
        calibration=calibration,
        position=[0, 0, 0],
        orientation=[0, 0, 0],
    )
    sensor = TagSensor(
        camera_config=config, tag_size=22, tag_positions=[[1, 1, 1]], name="right"
    )

    result = sensor.measure()

    print(result.data)
    assert len(result.data) == 3
