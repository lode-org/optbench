"""Measured times and step counts read back from a start record.

A zero on a row that crashed, or that never spent a force call, was not
a measurement. A zero on a successful row was.
"""

from __future__ import annotations

import math


def false_zero(row: dict, value) -> bool:
    """True when ``value`` is a stored zero the row did not measure."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if not math.isfinite(float(value)) or float(value) != 0.0:
        return False
    if row.get("success") is True:
        return False
    if row.get("failure") == "crash":
        return True
    calls = row.get("force_calls")
    return row.get("success") is False and calls in (None, 0, 0.0)
