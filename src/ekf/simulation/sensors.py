from functools import partial
from typing import Optional

from jax import jit
from jax import numpy as jnp
from jax import random

from ekf.measurements import Measurement
from ekf.sensors import Sensor
from ekf.sensors.tag_positions import TagSensor
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
        camera_config,
        tag_size=1,
        tag_positions=TAG_POSITIONS,
        rover: Optional[Rover] = None,
        **kwargs,
    ):
        super().__init__(
            camera_config,
            tag_size=tag_size,
            tag_positions=tag_positions,
            name=kwargs.pop("name", "tags"),
        )
        self.rover = rover if rover else Rover(d=2)
        self.key = random.PRNGKey(12)

    def measure(self):
        data = super().H(self.rover.state)

        self.key, subkey = random.split(self.key)
        data = (random.uniform(subkey, shape=(len(data),)) - 0.5) * 10.0
        return Measurement(data, jnp.eye(len(self.tag_positions) * 3) * 0.001)


class SimulatedGyroSensor(Sensor):
    def __init__(self, rover: Rover, **kwargs) -> None:
        super().__init__(kwargs.pop("timeout", None), name="gyro")
        self.rover = rover
        self.key = random.PRNGKey(758493)  # Random seed is explicit in JAX

    def measure(self):
        self.key, subkey = random.split(self.key)
        # Return estimated yaw, pitch, roll from state
        #
        data = (
            self.H(self.rover.state)
            + (random.uniform(subkey, shape=(3,)) - 0.5) * 0.001
        )

        return Measurement(data, jnp.eye(3) * 0.000001)

    @partial(jit, static_argnums=0)
    def H(self, x):
        """
        x: state vector
        """

        yaw, pitch, roll = x[-3:]
        # Return estimated yaw, pitch, roll from state
        return jnp.array([yaw, pitch, roll])
