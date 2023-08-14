import os
import pickle
from pathlib import Path
from urllib import request

import cv2
import numpy as np
from pyapriltags import Detector

source_path = Path(__file__).resolve()

source_dir = source_path.parent
url = "file://" + os.path.join(source_dir, "imgs/test_tag.jpg")
calibration = os.path.join(source_dir, "calibration.pkl")

with open(calibration, "rb") as fp:
    (cameraMatrix, dist) = pickle.load(fp)
    camera_matrix = cameraMatrix
    distortion = dist

    fx, fy = camera_matrix[0][0], camera_matrix[1][1]
    cx, cy = camera_matrix[0][2], camera_matrix[1][2]


def test_tag_detection():
    response = request.urlopen(url)
    img_array = np.asarray(bytearray(response.read()), dtype=np.uint8)

    img = cv2.imdecode(img_array, -1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detector = Detector(families="tag16h5", quad_decimate=1)

    detections = detector.detect(gray, False)

    print(detections)

    for detection in detections:
        corners = detection.corners.astype(np.int32)
        print(corners)
        cv2.polylines(gray, [corners], True, (0, 255, 255))
        print(detection)

    cv2.imshow("img", gray)
    cv2.waitKey(0)
    assert False
