import time
from threading import Thread
from typing import Callable

from ekf.measurements import Measurement


class Sensor:
    name = "Sensor"

    def __init__(self, timeout=None) -> None:
        self.listeners = []
        self.stopping = False
        self.timeout = timeout
        self.thread = None

    def start(self):
        if self.thread is None:
            self.thread = Thread(target=self._run, daemon=True)

        if self.thread.is_alive():
            return
        self.thread.start()

    def stop(self):
        if self.thread is None or not self.thread.is_alive():
            return

        self.stopping = True
        self.thread.join()
        self.stopping = False

    def _run(self):
        while not self.stopping:
            value = self.measure()
            self._call_callbacks(value)

            if self.timeout:
                time.sleep(self.timeout)

    def measure(self) -> Measurement:
        raise NotImplementedError("This needs to be implmeneted by a subclass!")

    def H(self, state):
        pass

    def subscribe(self, callback: Callable[[Measurement, "Sensor"], None]):
        self.listeners.append(callback)

    def _call_callbacks(self, reading: Measurement):
        for listener in self.listeners:
            listener(reading, self)
