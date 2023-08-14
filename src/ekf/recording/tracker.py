from ekf.filter import ExtendedKalmanFilter
from ekf.recording.scenario import SensorLoader


class EKFTracker:
    def __init__(self, scenario, filter: ExtendedKalmanFilter) -> None:
        self.loader = SensorLoader(scenario)
        self.filter = filter
        timestamp = self.loader.next_reading()[0]
        self.last_timestamp = timestamp

    def step(self):
        timestamp, reading, motor, sensor = self.loader.next_reading()
        dt = timestamp - self.last_timestamp
        self.last_timestamp = timestamp

        self.filter.step(motor, dt, reading, sensor)
