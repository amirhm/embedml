import numpy as np
from embedml.tensor import Tensor


def test_mult():
    xnp = np.ones((3, 3))
    ynp = np.eye(3)
    znp = 2 * xnp + ynp * 4

    x = Tensor.ones((3, 3))
    y = Tensor.eye(3)
    z = 2 * x + y * 4
    assert np.allclose(z.cpu(), znp)