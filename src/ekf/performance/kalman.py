from time import time
from typing import Literal

from jax import numpy as jnp
from tqdm import trange

from ekf.filter import ExtendedKalmanFilter
from ekf.simulation.rover import Rover
from ekf.simulation.sensors import SimulatedGyroSensor, SimulatedTagSensor
from ekf.state import RoverModel


def measure_actor(iterations: int = 1000):
    actor = RoverModel(d=2)
    state = jnp.zeros((9,))
    control = jnp.zeros((2,))

    # Warmup
    for _ in range(10):
        state = actor.F(state, control, 1)

    start = time()

    for _ in trange(iterations):
        state = actor.F(state, control, 1)

    return time() - start


SENSORS = Literal["gyro", "tag"]


def measure_kalman(iterations: int = 1000, sensor_kind: SENSORS = "gyro"):
    rover = Rover(d=2)

    filter = ExtendedKalmanFilter(rover.actor.F, rover.actor.Q, rover.state, rover.P0)
    control = jnp.zeros((2,))
    if sensor_kind == "gyro":
        sensor = SimulatedGyroSensor(rover=rover)
    elif sensor_kind == "tag":
        sensor = SimulatedTagSensor(rover=rover)
    else:
        raise ValueError(f"Invalid sensor kind provided: {sensor_kind}")

    # Warmup
    for _ in range(10):
        filter.step(control, 1.0, sensor.measure(), sensor)

    start = time()

    for _ in trange(iterations):
        filter.step(control, 1.0, sensor.measure(), sensor)

    return time() - start


METHODS = Literal["actor", "ekf"]


def benchmark(method: METHODS, iterations: int, sensor: SENSORS = "gyro"):
    t = -1

    print(f"Starting '{method}' with {iterations} its.:")

    if method == "actor":
        t = measure_actor(iterations)

    elif method == "ekf":
        t = measure_kalman(iterations, sensor)

    print(
        f"Running {iterations} its. took: {t:.3f} seconds ({iterations / t:.3f} it/s)"
    )
