import json
import pickle
import threading
from queue import Queue
from time import time
from typing import Tuple

import click

from ekf.measurements import Measurement
from ekf.sensors.motor_state import MotorControlStateTCP
from ekf.sensors.sensor import Sensor
from ekf.sensors.tag_positions import TagSensor
from ekf.sensors.yocto_3d import Yocto3DSensor


@click.group()
def cli():
    """Tracking rover position using onboard sensors"""
    pass


@cli.command("benchmark")
@click.argument("method", default="actor", type=click.Choice(["actor", "ekf"]))
@click.option("--iterations", default=100_000)
@click.option("--sensor", default="gyro", type=click.Choice(["gyro", "tag"]))
def cmd_benchmark(method, iterations, sensor):
    from ekf.performance.kalman import benchmark

    benchmark(method, iterations, sensor)


@cli.command("motor")
@click.option("--host", default="172.16.10.77")
@click.option("--port", default=3002)
def cmd_motor(host, port):
    from ekf.sensors.motor_state import main

    main(host, port)


@cli.command("record")
@click.option(
    "--tag_config",
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
)
@click.option(
    "--camera_config",
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
)
@click.option("--motor-host")
@click.option("--motor-port")
@click.argument("path", type=click.Path(dir_okay=False))
def cmd_record(path, tag_config, camera_config, motor_host, motor_port):
    with open(tag_config) as fp:
        json.load(fp)

    with open(camera_config) as fp:
        cameras = json.load(fp)

    tag_sensors = []

    motor = MotorControlStateTCP(motor_host, motor_port)
    motor.connect()
    motor.start()

    fifo: Queue[Tuple[float, Measurement, str]] = Queue()

    start = time()

    def add_to_fifo(measurement: Measurement, sensor: Sensor):
        return fifo.put((time() - start, measurement, sensor.name or ""))

    gyro = Yocto3DSensor(timeout=0.5, name="gyro")
    gyro.subscribe(add_to_fifo)
    gyro.start()

    for camera in cameras:
        tag_sensor = TagSensor(timeout=0.5, **camera)
        tag_sensor.subscribe(add_to_fifo)
        tag_sensor.start()
        tag_sensors.append(tag_sensor)

    def write_output():
        with open(path, "wb") as out:
            while True:
                t, m, s = fifo.get()
                pickle.dump((t, m, s, motor.get_current_state()), out)

    t = threading.Thread(target=write_output)
    t.start()
