import socket
import time
from threading import Lock, Thread
from typing import Sequence

from jax import Array


def chunker(seq: Sequence, size: int):
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


def to_left_and_right_speed(received):
    left_front, left_rear, right_front, right_rear = [int(x) for x in received[1:5]]

    return (left_front + left_rear) / 2 / 1000, -(right_front + right_rear) / 2 / 1000


def main(host: str, port: int):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        while True:
            # Receive data from the server
            recv = s.recv(15 * 4)
            chunked = chunker(recv, 4)
            received = []
            for chunk in chunked:
                received.append(int.from_bytes(chunk, "big", signed=True))
            print(time.time(), received)


class MotorControlState:
    verbose: bool = False

    def get_current_state(self) -> Array:
        raise NotImplementedError()

    def start(self):
        pass

    def stop(self):
        pass


class MotorControlStateTCP(MotorControlState):
    def __init__(self, host: str = "172.16.10.77", port: int = 3002) -> None:
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.host = host
        self.port = port
        self.stopping: bool = False

        self.state = (0, 0)
        self.state_lock = Lock()
        self.thread = None

    def connect(self):
        self.socket.connect((self.host, self.port))

    def start(self):
        if self.thread is None:
            self.thread = Thread(target=self._run)

        if self.thread.is_alive():
            return

        self.thread.start()

    def stop(self):
        if self.thread is None:
            return

        self.stopping = True
        self.thread.join()
        self.stopping = False
        self.thread = None

    def _run(self):
        self.connect()
        while not self.stopping:
            # Receive data from the server
            recv = self.socket.recv(15 * 4)
            chunked = chunker(recv, 4)
            received = []

            for chunk in chunked:
                received.append(int.from_bytes(chunk, "big", signed=True))

            with self.state_lock:
                self.state = to_left_and_right_speed(received)

    def get_current_state(self):
        with self.state_lock:
            return self.state
