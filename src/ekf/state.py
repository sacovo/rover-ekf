from functools import partial

import jax.numpy as jnp
from jax import jit, lax


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
                    [1, 1, 1, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1],
                ],
            )
            * 0.1
        )
        self.initial_state = jnp.zeros((dim,))
        self.P0 = jnp.zeros(((dim, dim)))

    @partial(jit, static_argnums=0)
    def F(self, state, control, dt):
        d = self.d

        x, y, z, yaw, pitch, roll = state
        v_left, v_right = control

        def if_equal(carry):
            x, y, z, yaw, v_left, v_right, _, _, dt = carry

            # Calculate average velocity adjusted for yaw and decompose it
            v_avg = (v_left + v_right) / 2
            v_xy = v_avg * jnp.cos(pitch)
            v_z = v_avg * jnp.sin(pitch)  # adjust vertical velocity based on pitch

            # Incorporate pitch and roll in horizontal velocities
            x_new = x + v_xy * dt * jnp.sin(yaw) * jnp.cos(roll)
            y_new = y + v_xy * dt * jnp.cos(yaw) * jnp.cos(roll)
            z_new = z + v_z * dt
            yaw_new = yaw
            pitch_new = pitch  # no change
            roll_new = roll  # no change

            return x_new, y_new, z_new, yaw_new, pitch_new, roll_new

        def if_not_equal(carry):
            x, y, z, yaw, v_left, v_right, _, d, dt = carry
            R = d / 2 * (v_left + v_right) / (v_right - v_left)
            omega = (v_right - v_left) / d

            ICC_x = x - R * jnp.cos(yaw)
            ICC_y = y + R * jnp.sin(yaw)

            rotation_matrix = jnp.array(
                [
                    [jnp.cos(omega * dt), -jnp.sin(omega * dt), 0],
                    [jnp.sin(omega * dt), jnp.cos(omega * dt), 0],
                    [0, 0, 1],
                ]
            )

            # Calculate average velocity adjusted for yaw and decompose it
            v_avg = (v_left + v_right) / 2
            v_xy = v_avg * jnp.cos(pitch)
            v_z = v_avg * jnp.sin(pitch)

            # Incorporate pitch and roll in the movement
            x_movement = v_xy * jnp.sin(yaw)  # * jnp.cos(roll)
            y_movement = v_xy * jnp.cos(yaw)  # * jnp.cos(roll)

            new_pos = jnp.dot(
                rotation_matrix,
                jnp.array(
                    [x - ICC_x + x_movement * dt, y - ICC_y + y_movement * dt, z]
                ),
            ) + jnp.array([ICC_x, ICC_y, v_z * dt])
            x_new, y_new, z_new = new_pos

            yaw_new = normalize_angle(yaw + omega * dt)
            pitch_new = pitch  # no change
            roll_new = roll  # no change
            return x_new, y_new, z_new, yaw_new, pitch_new, roll_new

        x_new, y_new, z_new, yaw_new, pitch_new, roll_new = lax.cond(
            jnp.isclose(v_right, v_left),
            if_equal,
            if_not_equal,
            (x, y, z, yaw, v_left, v_right, 0, d, dt),
        )

        return jnp.array(
            [
                x_new,
                y_new,
                z_new,
                yaw_new,
                pitch_new,
                roll_new,
            ]
        )
