from jax import numpy as jnp
from jax.scipy.spatial.transform import Rotation


class CameraConfig:
    def __init__(self, **kwargs) -> None:
        self.url = kwargs.pop("url", "")

        self.calibration = jnp.array(
            kwargs.pop(
                "calibration",
                [
                    [100, 0, 50],
                    [0, 100, 50],
                    [0, 0, 1],
                ],
            )
        )

        distortion = kwargs.pop("distortion", None)
        if distortion is not None:
            self.distortion = jnp.array(*distortion)
        else:
            self.distortion = None

        self.position = jnp.array(kwargs.pop("position"))
        orientation: Rotation | list[float] = kwargs.pop("orientation")

        if isinstance(orientation, list):
            self.orientation = Rotation.from_euler(
                "ZYX", jnp.array(orientation), degrees=True
            )
        else:
            self.orientation = orientation
