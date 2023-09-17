from typing import Any, Optional


class Measurement:
    def __init__(self, data, R):
        self.data = data
        self.R = R
        self.meta: Optional[Any] = None
