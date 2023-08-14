from jax import numpy as jnp


def ypr_to_rotation_matrix(angles):
    yaw, pitch, roll = angles  # convert degrees to radian

    # rotation matrix for yaw
    R_yaw = jnp.array(
        [
            [jnp.cos(yaw), -jnp.sin(yaw), 0],
            [jnp.sin(yaw), jnp.cos(yaw), 0],
            [0, 0, 1],
        ]
    )

    # rotation matrix for pitch
    R_pitch = jnp.array(
        [
            [jnp.cos(pitch), 0, jnp.sin(pitch)],
            [0, 1, 0],
            [-jnp.sin(pitch), 0, jnp.cos(pitch)],
        ]
    )

    # rotation matrix for roll
    R_roll = jnp.array(
        [
            [1, 0, 0],
            [0, jnp.cos(roll), -jnp.sin(roll)],
            [0, jnp.sin(roll), jnp.cos(roll)],
        ]
    )

    # Combined rotation matrix is R = R_yaw * R_pitch * R_roll
    R = jnp.dot(R_yaw, jnp.dot(R_pitch, R_roll))

    return R


def euler_to_rotation_matrix(theta):
    R_x = jnp.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, jnp.cos(theta[0]), -jnp.sin(theta[0])],
            [0.0, jnp.sin(theta[0]), jnp.cos(theta[0])],
        ]
    )

    R_y = jnp.array(
        [
            [jnp.cos(theta[1]), 0, jnp.sin(theta[1])],
            [0, 1, 0],
            [-jnp.sin(theta[1]), 0, jnp.cos(theta[1])],
        ]
    )

    R_z = jnp.array(
        [
            [jnp.cos(theta[2]), -jnp.sin(theta[2]), 0],
            [jnp.sin(theta[2]), jnp.cos(theta[2]), 0],
            [0, 0, 1],
        ]
    )

    R = jnp.dot(R_z, jnp.dot(R_y, R_x))

    return R
