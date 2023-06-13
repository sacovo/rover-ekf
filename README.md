# Extended Kalman Filter

This package implements an extended Kalman filter for usage with the FHNW Rover system.

## Overview

The filter is implemented in `src/ekf/filter.py`, a wrapper for usage with tracking is implemented in `src/ekf/tracker.py`.

The tracker is initialized with a list of sensors that provide input to the filter. Additionally the tracked needs a motor control instance, that provides the filter with the current control input.

```python
from ekf.tracker import EKFTracker

from ekf.sensors.yocto_3d import Yocto3DSensor
from ekf.sensors.motor_state import MotorControlStateTCP
from ekf.filter import ExtendedKalmanFilter

ekf = ExtendedKalmanFilter(
  F=F, # transition function
  Q=Q, # Process noise covariance
  state=state, # initial state
  P0=P0, # initial uncertainty
)

tracker = EKFTracker(
  sensors=[Yocto3DSensor()],
  motor_control=MotorControlStateTCP(host, port),
  filter=ekf,
)
```

The transition model for the rover is implemented in `src/ekf/state.py`, you can use this to initialize the EKF:

```python
from jax import numpy as jnp

from ekf.state import Rovermodel

rover = RoverModel(d=2) # distance between the left and right wheels
ekf = ExtendedKalmanFilter(
  F=rover.F,
  Q=rover.Q,
  state=rover.initial_state,
  P0=rover.P0,
)
```

Additional sensors can be implemented by extending the subclass `Sensor`:

```python

from functools import partial
from jax import jit

from ekf.sensors.sensor import Sensor
from ekf.measurements import Measurement

class MySensor(Sensor):

  def measure(self):
    # data should be a jnp array with the same dimension as the
    # return value of MySensor.H
    data = ...
    R = ...
    return Measurement(data=data, R=R)

  # Adding jit allows for faster execution and calculation of the Jacobian
  @partial(jit, static_argnums=0)
  def H(self, state):
    return ...
```



## Install

Install into a python virtual environment:

```bash
pip install -e .
```

Additionally you need to install a backend for `jax`, you can either install the cpu based one for testing:
```bash
pip install --upgrade "jax[cpu]"
```

Or, if you want to use GPU/TPU for autograd, you need to install jax with the cuda dependencies, see here: https://github.com/google/jax#installation

```bash
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Tests

Tests are located in `tests`, run them with pytest

```bash
pip install pytest
pytest
```

## Performance

You can run performance benchmarks using the rover_ekf script:

```bash
rover_ekv benchmark --help
```


## Notes

Kalman Filter Ressources:


Tutorial: https://www.kalmanfilter.net/alphabeta.html

Intro: http://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf



https://github.com/commaai/rednose

The kalman filter framework described here is an incredibly powerful tool for any optimization problem, but particularly for visual odometry, sensor fusion localization or SLAM. It is designed to provide very accurate results, work online or offline, be fairly computationally efficient, be easy to design filters with in python.


Visual Aided Position estimate:

https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.437.1085&rep=rep1&type=pdf

https://web.casadi.org/


https://en.wikipedia.org/wiki/Extended_Kalman_filter



https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/263423/1/ROVIO.pdf

https://www.researchgate.net/profile/Udo-Frese/publication/289663789_A_Kalman_filter_for_odometry_using_a_wheel_mounted_inertial_sensor/links/5dc82ad7a6fdcc57503dd1aa/A-Kalman-filter-for-odometry-using-a-wheel-mounted-inertial-sensor.pdf?origin=publication_detail
