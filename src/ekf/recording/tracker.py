from ekf.filter import ExtendedKalmanFilter
from ekf.recording.scenario import SensorLoader


class EKFTracker:
    def __init__(
        self, scenario, filter: ExtendedKalmanFilter, verbose: bool = False
    ) -> None:
        self.loader = SensorLoader(scenario)
        self.filter = filter
        timestamp = self.loader.next_reading()[0]
        self.last_timestamp = timestamp
        self.current_step = None
        self.verbose = verbose

    def step(self):
        timestamp, reading, motor, sensor = self.loader.next_reading()
        if reading is None:
            return
        dt = timestamp - self.last_timestamp
        self.last_timestamp = timestamp
        self.current_step = (reading, motor, sensor)

        prediction = self.filter.F(self.filter.state, motor, dt)
        if self.verbose:
            print(f"Delta Time: {dt}")
            print(f"Motor speed: {motor} => Predicted State: \n{prediction}")

            print(f"Sensor reading: \n{reading.data}")
            print(f"Sensor prediction: \n{sensor.H(prediction)}")

        self.filter.step(motor, dt, reading, sensor)
        if self.verbose:
            print(f"State after update:\n{self.filter.state}")
