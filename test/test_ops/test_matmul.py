import numpy as np
from embedml.tensor import Tensor


def test_simple():
    xnp = np.eye(3)
    ynp = np.ones((3, 8))
    znp = np.matmul(xnp, ynp).sum()

    x = Tensor.eye(3)
    y = Tensor.ones((3, 8))
    z = x.matmul(y).sum()

    x = x.cpu()
    y = y.cpu()

    assert np.allclose(x, xnp)
    assert np.allclose(y, ynp)
    assert np.allclose(z.cpu(), znp)
