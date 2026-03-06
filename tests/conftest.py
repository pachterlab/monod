"""
Pytest configuration and shared utilities for monod tests.

Adds src/monod to sys.path so that the bare-import style used in the source
files (e.g. `from extract_data import log`) works without an installed package.

Snapshot helper
---------------
`check_snapshot(name, value)` compares `value` against a saved .npy file in
tests/snapshots/.  On the first run the file does not exist, so it is created
and the comparison is skipped (the test still passes).  On every subsequent run
the saved array is loaded and compared with np.testing.assert_allclose.

To regenerate all snapshots, delete the tests/snapshots/ directory and run
pytest once.
"""

import os
import sys

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_SRC_MONOD = os.path.join(os.path.dirname(__file__), "..", "src", "monod")
if _SRC_MONOD not in sys.path:
    sys.path.insert(0, os.path.abspath(_SRC_MONOD))

# ---------------------------------------------------------------------------
# Snapshot directory
# ---------------------------------------------------------------------------

SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), "snapshots")


def check_snapshot(name: str, value, rtol: float = 1e-5, atol: float = 1e-8):
    """Compare *value* against the saved snapshot named *name*.

    Parameters
    ----------
    name:
        Unique identifier for this snapshot (used as the filename stem).
    value:
        Array-like to compare.
    rtol, atol:
        Tolerances passed to ``np.testing.assert_allclose``.

    Behaviour
    ---------
    * If the snapshot file does not exist it is created and the function
      returns without asserting.  The test that called this function passes.
    * If the snapshot file exists the stored array is loaded and compared
      against *value*.
    """
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    path = os.path.join(SNAPSHOT_DIR, f"{name}.npy")
    value = np.asarray(value)
    if not os.path.exists(path):
        np.save(path, value)
        return  # snapshot created; nothing to compare yet
    expected = np.load(path, allow_pickle=True)
    np.testing.assert_allclose(
        value,
        expected,
        rtol=rtol,
        atol=atol,
        err_msg=f"Snapshot mismatch for '{name}'",
    )
