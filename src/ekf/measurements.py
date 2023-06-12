from jax import numpy as jnp


class Measurement:
    def __init__(self, data, R):
        self.data = data
        self.R = R


class VisualOdometryMeasurement(Measurement):
    def __init__(self, data, R, previous_state):
        super().__init__(data, R)
        self.previous_state = previous_state

    def H(self, x):
        # Return change in position and orientation elements of the state
        delta_position = x[:3] - self.previous_state[:3]
        delta_orientation = x[3:6] - self.previous_state[3:6]
        return jnp.concatenate((delta_position, delta_orientation))


class OrientationMeasurement(Measurement):
    def __init__(self, data, R):
        """
        data: the measurement data, expected to be [yaw, pitch, roll]
        R: noise matrix for this measurement
        """
        super().__init__(data, R)


class TagMeasurement(Measurement):
    def __init__(self, data, R):
        super().__init__(data, R)
