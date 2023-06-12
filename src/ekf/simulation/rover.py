from time import time

from jax import numpy as jnp

from ekf.state import Actor3D


class Rover:
    def __init__(self, d) -> None:
        self.actor = Actor3D(d)
        self.state = jnp.zeros((9,))
        self.P0 = jnp.zeros((9,))
        self.control = jnp.array((0, 0))
        self.last_time = time()

    def step(self):
        now = time()
        dt = now - self.last_time
        self.step_dt(dt)
        self.last_time = now

    def step_dt(self, dt):
        self.state = self.actor.F(self.state, self.control, dt)

    def set_control(self, speeds):
        self.control = jnp.array(speeds)
