import numpy as np
from embedml.tensor import Tensor


def test_addition():
    xnp = np.ones((3, 3))
    ynp = np.eye(3)
    znp = xnp + ynp

    x = Tensor.ones((3, 3))
    y = Tensor.eye(3)
    z = x + y
    assert np.allclose(z.cpu(), znp)

