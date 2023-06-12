import pygame
from jax import numpy as jnp
from pygame.event import Event

from ekf.filter import ExtendedKalmanFilter
from ekf.sensors.tag_positions import CameraConfig
from ekf.simulation.motor import SimulatedMotorControlState
from ekf.simulation.rover import Rover
from ekf.simulation.sensors import SimulatedGyroSensor, SimulatedTagSensor
from ekf.tracker import EKFTracker


def setup_simulation():
    print("Setup Rover:")
    rover = Rover(d=2)

    print("Setup EKF")
    ekf = ExtendedKalmanFilter(
        rover.actor.F,
        rover.actor.Q,
        rover.state,
        rover.P0,
    )

    print("Setup sensors")
    sensors = [
        SimulatedTagSensor(
            camera_parameters=CameraConfig(),
            tag_size=0,
            tag_positions=jnp.array([[1, 2, 3]]),
            rover=rover,
        ),
        SimulatedGyroSensor(rover=rover),
    ]
    motor_control = SimulatedMotorControlState(rover)

    tracker = EKFTracker(sensors, ekf, motor_control)

    return rover, tracker


class Simulation:
    def __init__(self):
        self.rover, self.tracker = setup_simulation()
        self.tracker.start()
        self._running = True

    def on_init(self):
        pass

    def on_event(self, event: Event):
        if event.type == pygame.QUIT:
            self._running = False

    def on_loop(self):
        v_left, v_right = 10, 20

        self.rover.set_control((v_left, v_right))

    def on_render(self):
        pass

    def on_cleanup(self):
        self.tracker.stop()

    def on_execute(self):
        if self.on_init() is False:
            self._running = False

        while self._running:
            self.on_loop()
            self.on_render()
        self.on_cleanup()


def simulation():
    print("Loading simulation")
    simulation = Simulation()
    print("Starting simulation GUI:")
    simulation.on_execute()
