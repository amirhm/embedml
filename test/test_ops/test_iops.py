import numpy as np
from embedml.tensor import Tensor


def test_iaddition():
    xnp = np.ones((3, 3))
    ynp = xnp
    xnp += 10

    x = Tensor.ones((3, 3))
    y = x
    x += 10

    assert xnp is ynp
    assert x is y


def test_isub():
    xnp = np.ones((3, 3))
    ynp = xnp
    xnp -= 10

    x = Tensor.ones((3, 3))
    y = x
    x -= 10

    assert xnp is ynp
    assert x is y
