import math

from jax import numpy as jnp
from numpy import testing

from ekf.state import Actor3D


def test_no_movement():
    actor = Actor3D(d=2)
    state = jnp.zeros((9,))

    control = jnp.zeros((2,))

    state1 = actor.F(state, control, 1)
    assert jnp.all(state1 == state)


def test_movement_ahead():
    actor = Actor3D(d=2)
    state = jnp.zeros((9,))

    control = jnp.array((100, 100))

    state1 = actor.F(state, control, 1)

    testing.assert_allclose(state1[:3], jnp.array([100, 0, 0]))


def test_rotation_left():
    actor = Actor3D(d=2)
    state = jnp.zeros((9,))

    control = jnp.array((50, -50))

    state1 = actor.F(state, control, 1)

    testing.assert_allclose(state1[:3], jnp.array([0, 0, 0]))

    yaw, pitch, roll = state1[-3:]

    assert yaw > 0
    assert pitch == 0
    assert roll == 0


def test_rotation_right():
    actor = Actor3D(d=2)
    state = jnp.zeros((9,))

    control = jnp.array((-50, 50))

    state1 = actor.F(state, control, 1)

    testing.assert_allclose(state1[:3], jnp.array([0, 0, 0]))

    yaw, pitch, roll = state1[-3:]

    assert yaw < 0
    assert pitch == 0
    assert roll == 0


def test_rotated_drive():
    actor = Actor3D(d=2)

    state = jnp.zeros((9,))

    control = jnp.array((100, 100))

    state1 = actor.F(state.at[-3].set(jnp.pi / 2), control, 1)
    testing.assert_allclose(
        state1[:3],
        jnp.array([0.0, 100.0, 0.0]),
        atol=5e-6,
    )

    state2 = actor.F(state.at[-3].set(jnp.pi / 4), control, 1)

    testing.assert_allclose(
        state2[:3],
        jnp.array(
            [
                math.cos(math.pi / 4) * 100,
                math.sin(math.pi / 4) * 100,
                0.0,
            ]
        ),
        atol=5e-6,
    )
