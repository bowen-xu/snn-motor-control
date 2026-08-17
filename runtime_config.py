"""Read and validate simulation runtime environment settings."""

from __future__ import annotations

import math
import os


def read_nonnegative_ms(name: str) -> float:
    """Return a non-negative millisecond value from the environment.

    Zero means that the corresponding override is disabled.
    """

    raw_value = os.environ.get(name, "0")
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a finite non-negative number") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return value
