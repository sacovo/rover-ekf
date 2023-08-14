from jax import numpy as jnp


class CameraConfig:
    def __init__(self, **kwargs) -> None:
        self.url = kwargs.pop("url")

        self.calibration = jnp.array(kwargs.pop("calibration"))

        if distortion := kwargs.pop("distortion", None):
            self.distortion = jnp.array(distortion)

        self.position = jnp.array(kwargs.pop("position"))
        self.orientation = jnp.array(kwargs.pop("orientation"))
