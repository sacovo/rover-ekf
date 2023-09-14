from functools import partial

import jax.numpy as jnp
from jax import jit


@jit
def normalize_angle(yaw):
    normalized_yaw = yaw % (2 * jnp.pi)  # restricts yaw to [0, 2pi]
    normalized_yaw = jnp.where(
        normalized_yaw > jnp.pi, normalized_yaw - 2 * jnp.pi, normalized_yaw
    )
    return normalized_yaw


class RoverModel:
    def __init__(self, d):
        self.d = d
        dim = 6
        self.state_dim = dim
        self.Q = (
            jnp.array(
                [
                    [1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1],
                ],
            )
            * 0.1
        )
        self.initial_state = jnp.zeros((dim,))
        self.P0 = jnp.zeros(((dim, dim)))

    @partial(jit, static_argnums=0)
    def F(self, state, control, dt):
        x, y, z, yaw, pitch, roll = state
        v_left, v_right = control

        # Calculate average velocity adjusted for yaw and decompose it
        v_avg = (v_left + v_right) / 2
        v_xy = v_avg * jnp.cos(pitch)
        v_z = v_avg * jnp.sin(pitch)  # adjust vertical velocity based on pitch

        # Incorporate pitch and roll in horizontal velocities
        x_new = x + v_xy * dt * jnp.sin(yaw)  # * jnp.cos(roll)
        y_new = y + v_xy * dt * jnp.cos(yaw)  # * jnp.cos(roll)
        z_new = z + v_z * dt

        return jnp.array(
            [
                x_new,
                y_new,
                z_new,
                yaw,
                pitch,
                roll,
            ]
        )
