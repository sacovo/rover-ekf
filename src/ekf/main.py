import click


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


@cli.command("simulation")
def cmd_simulation():
    from ekf.simulation import simulation

    simulation()


@cli.command("motor")
@click.option("--host", default="172.16.10.77")
@click.option("--port", default=3002)
def cmd_motor(host, port):
    from ekf.sensors.motor_state import main

    main(host, port)
