from functools import partial

import jax.numpy as jnp
from jax import jit
from scipy.spatial.transform import Rotation
from yoctopuce.yocto_api import YAPI, YRefParam
from yoctopuce.yocto_gyro import YGyro
from yoctopuce.yocto_tilt import YTilt

from ekf.config import DEBUG
from ekf.measurements import Measurement
from ekf.sensors import Sensor


class RotationSensor(Sensor):
    def __init__(
        self,
        initial_orientation: Rotation = Rotation.identity(),
        confidence=0.000001,
        angles="ZYX",
        signs=[1, -1, -1],
        **kwargs
    ) -> None:
        self.initial_orientation = initial_orientation

        initial_measurement = self._next_orientation()
        self.rot_initial_inv = (initial_measurement).inv()
        super().__init__(**kwargs)
        self.confidence = confidence
        self.angles = angles
        self.signs = jnp.array(signs)
        self.config.update(
            {
                "signs": self.signs,
                "angles": self.angles,
                "confidence": self.confidence,
                "initial_orientation": initial_orientation,
                "initial_measurement": initial_measurement,
            }
        )

    def measure(self) -> Measurement:
        orientation = self._next_orientation()

        target = self.initial_orientation * self.rot_initial_inv * orientation
        z, y, x = target.as_euler(self.angles)
        euler = jnp.array([z, y, x]) * self.signs

        if DEBUG:
            print("Gyro [QUAT]", target.as_quat())
            print("Gyro [Euler]", euler)

        measurement = Measurement(euler, jnp.eye(3) * self.confidence)

        if self.verbose:
            measurement.meta = orientation

        return measurement

    def _next_orientation(self):
        return Rotation.from_quat(self._get_quaternion())

    def _get_quaternion(self):
        raise NotImplementedError()

    @partial(jit, static_argnums=0)
    def H(self, x):
        """
        x: state vector
        """

        yaw, pitch, roll = x[-3:]
        # Return estimated yaw, pitch, roll from state
        return jnp.array([yaw, pitch, roll])


class Yocto3DSensor(RotationSensor):
    def __init__(self, target="any", host="127.0.0.1", **kwargs) -> None:
        errmsg = YRefParam()
        YAPI.RegisterHub(host, errmsg)

        if target == "any":
            # retreive any tilt sensor
            anytilt = YTilt.FirstTilt()
            if anytilt is None:
                raise ValueError("No tilt sensor found")
            m = anytilt.get_module()
            target = m.get_serialNumber()
        else:
            anytilt = YTilt.FindTilt(target + ".tilt1")

        serial = anytilt.get_module().get_serialNumber()
        gyro = YGyro.FindGyro(serial + ".gyro")
        self.gyro = gyro
        super().__init__(**kwargs)

    def _get_quaternion(self):
        x, y, z, w = (
            self.gyro.get_quaternionX(),
            self.gyro.get_quaternionY(),
            self.gyro.get_quaternionZ(),
            self.gyro.get_quaternionW(),
        )

        return jnp.array((x, y, z, w))
