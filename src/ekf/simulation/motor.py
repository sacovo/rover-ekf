from ekf.sensors.motor_state import MotorControlState
from ekf.simulation.rover import Rover


class SimulatedMotorControlState(MotorControlState):
    def __init__(self, rover: Rover) -> None:
        self.rover = rover

    def get_current_state(self):
        return self.rover.control
