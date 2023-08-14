from time import time

from jax import numpy as jnp
from jax import random

from ekf.state import RoverModel


class Rover:
    def __init__(self, d) -> None:
        self.actor = RoverModel(d)
        self.state = jnp.zeros((9,))
        self.P0 = jnp.zeros((9, 9))
        self.control = jnp.array((0, 0))
        self.last_time = time()
        self.key = random.PRNGKey(51)

    def step(self):
        now = time()
        dt = now - self.last_time
        self.step_dt(dt)
        self.last_time = now

    def step_dt(self, dt):
        self.key, subkey = random.split(self.key)
        noise = (random.uniform(subkey, shape=(2,)) - 0.5) * 0.01
        self.state = self.actor.F(self.state, self.control + noise, dt)

    def set_control(self, speeds):
        self.control = jnp.array(speeds)
