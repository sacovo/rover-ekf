from functools import partial

from jax import numpy as jnp
from jax import vmap
from numpy import testing
from numpy.lib import math

from ekf.sensors.tag_positions import CameraConfig, TagSensor, euler_to_rotation_matrix
from ekf.simulation.rover import Rover


def test_euler_to_rotation():
    R = euler_to_rotation_matrix(jnp.array([0, 0, 0]))
    testing.assert_allclose(R, jnp.eye(3))

    R = euler_to_rotation_matrix(jnp.array([1.5, 0, 0]))
    testing.assert_allclose(
        R,
        jnp.array(
            [
                [1, 0, 0],
                [0, 0.0707372, -0.9974950],
                [0.0000000, 0.9974950, 0.0707372],
            ]
        ),
    )


def test_euler_vmap():
    camera_position = jnp.array([0, 0, 0])
    camera_orientation = jnp.array([0, 0, 0])
    tag_positions = jnp.array([[23, 10, 10], [12, 5, 6]])

    vmap(
        partial(
            TagSensor.calculate_tag_position,
            camera_position=camera_position,
            camera_orientation=camera_orientation,
        )
    )(tag_positions)


def test_H_call():
    rover = Rover(d=2)
    camera = CameraConfig()
    sensor = TagSensor(
        url="",
        camera_parameters=camera,
        tag_positions=jnp.array([[23, 10, 12], [12, 5, 6]]),
        tag_size=1,
    )

    measurement = sensor.H(rover.state)

    assert len(measurement) == 2 * 3

    assert measurement[0] == 23
    assert measurement[1] == 10
    assert measurement[2] == 12


def test_H_moved():
    rover = Rover(d=2)
    camera = CameraConfig()
    rover.state = rover.state.at[0].set(23).at[1].set(10).at[2].set(12)
    sensor = TagSensor(
        url="",
        camera_parameters=camera,
        tag_positions=jnp.array([[23, 10, 12], [12, 5, 6]]),
        tag_size=1,
    )

    measurement = sensor.H(rover.state)

    assert measurement[0] == 0
    assert measurement[1] == 0
    assert measurement[2] == 0


def test_tag_position():
    tag_position = jnp.array([1, 0, 0])

    relative_position = TagSensor.calculate_tag_position(
        tag_position,
        jnp.zeros(3),
        jnp.array((math.pi / 2, 0, 0)),
    )

    testing.assert_allclose(relative_position, [0, 1, 0], atol=5e-6)

    relative_position = TagSensor.calculate_tag_position(
        tag_position,
        jnp.zeros(3),
        jnp.zeros(3),
    )

    testing.assert_allclose(tag_position, relative_position)


def test_H_call_rotated():
    rover = Rover(d=2)
    rover.state = rover.state.at[-3].set(math.pi / 2)

    camera = CameraConfig()
    sensor = TagSensor(
        url="",
        camera_parameters=camera,
        tag_positions=jnp.array([[1, 0, 0]]),
        tag_size=1,
    )

    measurement = sensor.H(rover.state)

    testing.assert_allclose(measurement, jnp.array([0, 1, 0]), atol=5e-6)
