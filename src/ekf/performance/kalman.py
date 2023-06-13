from time import time

from jax import numpy as jnp
from tqdm import trange

from ekf.filter import ExtendedKalmanFilter
from ekf.simulation.rover import Rover
from ekf.simulation.sensors import SimulatedGyroSensor, SimulatedTagSensor
from ekf.state import RoverModel


def measure_actor(iterations=1000):
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


def measure_kalman(iterations=1000, sensor="gyro"):
    rover = Rover(d=2)

    filter = ExtendedKalmanFilter(rover.actor.F, rover.actor.Q, rover.state, rover.P0)
    control = jnp.zeros((2,))
    if sensor == "gyro":
        sensor = SimulatedGyroSensor(rover=rover)
    else:
        sensor = SimulatedTagSensor(rover=rover)

    # Warmup
    for _ in range(10):
        filter.step(control, 1.0, sensor.measure(), sensor)

    start = time()

    for _ in trange(iterations):
        filter.step(control, 1.0, sensor.measure(), sensor)

    return time() - start


def benchmark(method, iterations, sensor="gyro"):
    t = -1

    print(f"Starting '{method}' with {iterations} its.:")

    if method == "actor":
        t = measure_actor(iterations)

    elif method == "ekf":
        t = measure_kalman(iterations, sensor)

    print(
        f"Running {iterations} its. took: {t:.3f} seconds ({iterations / t:.3f} it/s)"
    )
