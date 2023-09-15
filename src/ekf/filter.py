import jax.numpy as jnp
from jax import jacfwd

from ekf.measurements import Measurement
from ekf.sensors import Sensor


class ExtendedKalmanFilter:
    def __init__(self, F, Q, state, P0):
        self.F = F  # Transition model
        self.Q = Q  # Process noise covariance
        self.state_dim = len(state)

        self.state = state
        self.covariance = P0
        self.I = jnp.eye(self.covariance.shape[0])  # Identity matrix

    def predict(self, control, dt):
        # Propagate state forward
        self.state = self.F(self.state, control, dt)

        # Calculate Jacobian of F at the current state
        F_jacobian = jacfwd(self.F, argnums=0)(self.state, control, dt)

        self.F_jacobian = F_jacobian
        # Propagate covariance forward
        self.covariance = F_jacobian @ self.covariance @ F_jacobian.T + self.Q

    def update(self, measurement: Measurement, sensor: Sensor):
        # Calculate Jacobian of H at the current state
        y = measurement.data - sensor.H(self.state)

        H_jacobian = jacfwd(sensor.H, argnums=0)(self.state)

        # Calculate Kalman gain
        S = H_jacobian @ self.covariance @ H_jacobian.T + measurement.R
        self.S = S
        self.H = H_jacobian

        inv = jnp.linalg.inv(S)

        if jnp.isnan(inv).any():
            print("Warning: Could not calculate K-matrix, skipping")
            return

        K = self.covariance @ H_jacobian.T @ inv

        self.K = K
        self.y = y
        # Update the state estimate

        self.state = self.state + K @ y
        # Update the covariance estimate
        self.covariance = (self.I - K @ H_jacobian) @ self.covariance

    def step(self, control, dt: float, measurement: Measurement, sensor: Sensor):
        self.predict(control, dt)
        self.update(measurement, sensor)
