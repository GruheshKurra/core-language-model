import numpy as np

from mla import backend, to_device, to_numpy


def test_default_dtype_is_float64_on_numpy():
    if backend.NAME == "numpy":
        assert backend.DTYPE == np.dtype("float64")


def test_roundtrip_preserves_values():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = to_numpy(to_device(a))
    assert np.allclose(b, a)
