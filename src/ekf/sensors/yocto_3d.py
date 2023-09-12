from functools import partial

import jax.numpy as jnp
from jax import jit, lax
from scipy.spatial.transform import Rotation
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
def quaternion_inverse(q):
    w, x, y, z = q
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


def rotation_to_euler(r):
    return r.as_euler("YXZ")


@jit
def quaternion_to_euler(w, x, y, z):
    """
    Converts quaternions with components w, x, y, z into a tuple (yaw, pitch, roll).
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = jnp.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)

    # Handle t2 saturation with lax.cond
    def clamp_upper(dummy):
        return +1.0

    def clamp_lower(dummy):
        return -1.0

    def identity(t2):
        return t2

    t2 = lax.cond(
        t2 > +1.0,
        t2,
        clamp_upper,
        t2,
        lambda t2: lax.cond(t2 < -1.0, t2, clamp_lower, t2, identity),
    )

    pitch_y = jnp.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = jnp.arctan2(t3, t4)

    return yaw_z, pitch_y, roll_x  # in radians


class RotationSensor(Sensor):
    def __init__(
        self,
        sensor_orientation: Rotation = Rotation.identity(),
        initial_orientation: Rotation = Rotation.identity(),
        **kwargs
    ) -> None:
        self.rot_sensor_inv = sensor_orientation.inv()
        self.initial_orientation = initial_orientation

        initial_measurement = self._next_orientation()
        self.rot_initial_inv = (initial_measurement * self.rot_sensor_inv).inv()
        super().__init__(**kwargs)
        self.confidence = 0.0000001

    def measure(self) -> Measurement:
        measurement = self._next_orientation()

        target = (
            self.initial_orientation
            * self.rot_initial_inv
            * measurement
            * self.rot_sensor_inv
        )
        euler = target.as_euler("ZYX")

        return OrientationMeasurement(jnp.array(euler), jnp.eye(3) * self.confidence)

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
        w, x, y, z = (
            self.gyro.get_quaternionW(),
            self.gyro.get_quaternionX(),
            self.gyro.get_quaternionY(),
            self.gyro.get_quaternionZ(),
        )

        return jnp.array((w, x, y, z))
