import queue
import time
from threading import Thread
from typing import List

from ekf.filter import ExtendedKalmanFilter
from ekf.sensors import Sensor
from ekf.sensors.motor_state import MotorControlState


class EKFTracker:
    def __init__(
        self,
        sensors: List[Sensor],
        filter: ExtendedKalmanFilter,
        motor_control: MotorControlState,
    ) -> None:
        self.sensors = sensors
        self.filter = filter
        self.motor_control = motor_control
        self.last = time.time()
        self.queue = queue.Queue()

        self.stopping = False
        self.callbacks = []
        self._listeners_setup = False

    def _setup_listeners(self):
        if not self._listeners_setup:
            for sensor in self.sensors:
                sensor.subscribe(self._sensor_callback)
            self._listeners_setup = True

        for sensor in self.sensors:
            sensor.start()

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

    def _sensor_callback(self, measurement, sensor):
        control = self.motor_control.get_current_state()
        now = time.time()
        dt = now - self.last
        self.queue.put((control, dt, measurement, sensor))

    def _filter_loop(self):
        while not self.stopping:
            control, dt, measurement, sensor = self.queue.get()
            self.filter.step(control, dt, measurement, sensor)
            self.call_callbacks(self.filter.state)

    def get_state(self):
        return self.filter.state

    def call_callbacks(self, state):
        for callback in self.callbacks:
            callback(state)
