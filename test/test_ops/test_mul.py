import numpy as np
from embedml.tensor import Tensor
import torch


def test_mult():
    xnp = np.ones((3, 3))
    ynp = np.eye(3)
    znp = 2 * xnp + ynp * 4

    x = Tensor.ones((3, 3))
    y = Tensor.eye(3)
    z = 2 * x + y * 4
    assert np.allclose(z.cpu(), znp)


def test_mat_mul():
    xnp = np.random.randn(5, 10)
    ynp = np.random.randn(10)
    znp = xnp * ynp

    x = Tensor(xnp)
    y = Tensor(ynp)
    z = x * y

    assert np.allclose(z.cpu(), znp)


def test_mat_mul_grad():
    xnp = np.random.randn(5, 3, 10)
    ynp = np.random.randn(10)

    x = torch.tensor(xnp, requires_grad=True)
    y = torch.tensor(ynp, requires_grad=True)
    z = x * y
    list(map(lambda x: x.retain_grad(), [x, y, z]))

    z.sum().backward()

    xt = Tensor(xnp)
    yt = Tensor(ynp)
    zt = xt * yt
    zt.sum().backward()

    assert np.array_equal(zt.cpu(), z.detach().numpy())
    assert np.array_equal(zt.grad.cpu(), z.grad)
    assert np.array_equal(xt.grad.cpu(), x.grad)
    assert xt.grad.shape == (x.grad.shape)
    assert yt.grad.shape == (y.grad.shape)
    assert np.allclose(yt.grad.cpu(), y.grad)
