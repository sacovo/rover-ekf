from jax import numpy as jnp
from jax import random
from numpy import testing

from ekf.filter import ExtendedKalmanFilter
from ekf.sensors.tag_positions import CameraConfig
from ekf.simulation.rover import Rover
from ekf.simulation.sensors import SimulatedGyroSensor, SimulatedTagSensor


def test_single_call_zero():
    rover = Rover(d=2)

    ekf = ExtendedKalmanFilter(rover.actor.F, rover.actor.Q, rover.state, rover.P0)
    control = jnp.zeros((2,))
    sensor = SimulatedGyroSensor(rover=rover)

    ekf.step(control, 1, sensor.measure(), sensor)
    rover.step_dt(1)

    testing.assert_allclose(ekf.state, rover.state, atol=0.01)


def test_single_call():
    rover = Rover(d=2)

    ekf = ExtendedKalmanFilter(rover.actor.F, rover.actor.Q, rover.state, rover.P0)
    control = jnp.array([100, 20])
    sensor = SimulatedGyroSensor(rover=rover)

    rover.set_control(control)
    rover.step_dt(1)

    ekf.step(control, 1, sensor.measure(), sensor)

    testing.assert_allclose(ekf.state, rover.state, atol=0.1)


def test_random_stream():
    rover = Rover(d=2)

    ekf = ExtendedKalmanFilter(rover.actor.F, rover.actor.Q, rover.state, rover.P0)

    key = random.PRNGKey(12)

    gyro = SimulatedGyroSensor(rover=rover)

    for _ in range(100):
        key, subkey = random.split(key)
        control = random.randint(subkey, shape=(2,), minval=-10, maxval=10)

        rover.set_control(control)
        rover.step_dt(1)

        ekf.predict(control, 1)
        ekf.update(gyro.measure(), gyro)

    testing.assert_allclose(ekf.state, rover.state, atol=0.1)


def test_single_call_tag():
    rover = Rover(d=2)

    ekf = ExtendedKalmanFilter(rover.actor.F, rover.actor.Q, rover.state, rover.P0)
    control = jnp.zeros((2,))

    sensor = SimulatedTagSensor(
        camera_config=CameraConfig(
            orientation=jnp.array([0, 0, 0]),
            position=jnp.array(
                [0, 0, 0],
            ),
        ),
        tag_size=1,
        tag_positions=jnp.array(
            [
                [1, 2, 3],
                [3, 4, 5],
            ]
        ),
        rover=rover,
    )

    ekf.step(control, 1, sensor.measure(), sensor)
