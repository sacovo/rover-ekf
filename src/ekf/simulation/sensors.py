from functools import partial
from typing import Optional

from jax import jit
from jax import numpy as jnp

from ekf.measurements import OrientationMeasurement, TagMeasurement
from ekf.sensors import Sensor
from ekf.sensors.tag_positions import CameraConfig, TagSensor
from ekf.simulation.rover import Rover

TAG_POSITIONS = jnp.array(
    [
        [20, 10, 2.0],
        [30, 20, 1.3],
        [25, 30, 1.5],
        [21, 40, 1.3],
        [21, 90, 0.5],
        [33, 20, 0.8],
        [41, 30, 1.1],
        [22, 14, 1.3],
        [10, 39, 1.7],
    ]
)


class SimulatedTagSensor(TagSensor):
    def __init__(
        self,
        camera_parameters=None,
        tag_size=1,
        tag_positions=TAG_POSITIONS,
        rover: Optional[Rover] = None,
    ):
        super().__init__(
            None,
            camera_parameters=camera_parameters
            if camera_parameters
            else CameraConfig(),
            tag_size=tag_size,
            tag_positions=tag_positions,
        )
        self.rover = rover if rover else Rover(d=2)

    def measure(self):
        data = super().H(self.rover.state)
        return TagMeasurement(data, jnp.eye(len(self.tag_positions) * 3) * 0.001)


class SimulatedGyroSensor(Sensor):
    def __init__(self, rover: Rover, **kwargs) -> None:
        super().__init__(kwargs.pop("timeout", None))
        self.rover = rover

    def measure(self):
        yaw, pitch, roll = self.rover.state[-3:]
        # Return estimated yaw, pitch, roll from state
        data = jnp.array([yaw, pitch, roll])
        return OrientationMeasurement(data, jnp.eye(3) * 0.01)

    @partial(jit, static_argnums=0)
    def H(self, x):
        """
        x: state vector
        """

        yaw, pitch, roll = x[-3:]
        # Return estimated yaw, pitch, roll from state
        return jnp.array([yaw, pitch, roll])
