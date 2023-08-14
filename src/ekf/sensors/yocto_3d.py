import math
from functools import partial

from jax import jit
from jax import numpy as jnp
from yoctopuce.yocto_api import YAPI, YRefParam
from yoctopuce.yocto_gyro import YGyro
from yoctopuce.yocto_tilt import YTilt

from ekf.measurements import Measurement, OrientationMeasurement
from ekf.sensors import Sensor


@jit
def conjugate(w, x, y, z):
    """Returns the conjugate of a quaternion"""
    return (w, -x, -y, -z)


@jit
def multiply(quaternion1, quaternion2):
    """Multiplies two quaternions"""
    w1, x1, y1, z1 = quaternion1
    w2, x2, y2, z2 = quaternion2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return w, x, y, z


@jit
def quaternion_to_euler(w, x, y, z):
    """
    Converts quaternions with components w, x, y, z into a tuple (yaw, pitch, roll).
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return yaw_z, pitch_y, roll_x  # in radians


class Yocto3DSensor(Sensor):
    def __init__(self, timeout=None, target="any", **kwargs) -> None:
        super().__init__(timeout, **kwargs)

        errmsg = YRefParam()
        YAPI.RegisterHub("127.0.0.1", errmsg)

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
        self.inital_orientation = self._get_quaternion()

    def _get_quaternion(self):
        w, x, y, z = (
            self.gyro.get_quaternionW(),
            self.gyro.get_quaternionX(),
            self.gyro.get_quaternionY(),
            self.gyro.get_quaternionZ(),
        )

        return jnp.array((w, x, y, z))

    def measure(self) -> Measurement:
        quaternion = self._get_quaternion()
        relative = multiply(quaternion, self.inital_orientation)
        euler = quaternion_to_euler(*relative)

        return OrientationMeasurement(jnp.array(euler), None)

    @partial(jit, static_argnums=0)
    def H(self, x):
        """
        x: state vector
        """

        yaw, pitch, roll = x[-3:]
        # Return estimated yaw, pitch, roll from state
        return jnp.array([yaw, pitch, roll])
