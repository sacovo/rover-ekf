import pickle
import queue
import time
from datetime import datetime
from threading import Thread
from typing import List, Optional, Tuple

from cv2 import os
from jax import Array

from ekf.filter import ExtendedKalmanFilter
from ekf.measurements import Measurement
from ekf.sensors import Sensor
from ekf.sensors.motor_state import MotorControlState


class EKFTracker:
    def __init__(
        self,
        sensors: List[Sensor],
        filter: ExtendedKalmanFilter,
        motor_control: MotorControlState,
        output_path: Optional[str],
        verbose: bool = False,
    ) -> None:
        self.sensors = sensors
        self.filter = filter
        self.initial_covariance = filter.covariance.copy()
        self.motor_control = motor_control
        self.last = time.time()
        self.queue: queue.Queue[
            Tuple[Array, float, Measurement, Sensor]
        ] = queue.Queue()

        self.stopping = False
        self.callbacks = []
        self._listeners_setup = False
        self.set_state = None

        self.verbose = verbose

        self.recording = False

        if output_path is not None:
            self.output_path = os.path.join(
                output_path, datetime.now().strftime("%Y-%d-%mT%H:%I") + ".pkl"
            )
            self.output = open(self.output_path, "wb")
            self.recording = True

    def _setup_listeners(self):
        if not self._listeners_setup:
            for sensor in self.sensors:
                sensor.verbose = self.verbose
                sensor.subscribe(self._sensor_callback)
            self._listeners_setup = True

        for sensor in self.sensors:
            sensor.start()

        self.motor_control.verbose = self.verbose
        self.motor_control.start()

    def subscribe(self, callback):
        self.callbacks.append(callback)

    def start(self):
        self._setup_listeners()
        self.thread = Thread(target=self._filter_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.stopping = True

        for sensor in self.sensors:
            sensor.stop()

        self.queue.join()
        self.thread.join()
        self.stopping = False

    def _sensor_callback(self, measurement: Measurement, sensor: Sensor):
        control = self.motor_control.get_current_state()
        now = time.time()
        dt = now - self.last
        self.last = now
        self.queue.put((control, dt, measurement, sensor))

    def store_reading(self, reading):
        pickle.dump(reading, self.output)

    def _filter_loop(self):
        while not self.stopping:
            if self.set_state is not None:
                self.filter.state = self.set_state
                self.filter.covariance = self.initial_covariance
                self.set_state = None

            try:
                control, dt, measurement, sensor = self.queue.get(timeout=0.1)
                self.filter.step(control, dt, measurement, sensor)
                self.call_callbacks(self.filter.state)
                if self.recording:
                    self.store_reading((control, dt, measurement, sensor.config))
            except queue.Empty:
                continue

    def get_state(self):
        return self.filter.state

    def call_callbacks(self, state):
        for callback in self.callbacks:
            callback(state)
