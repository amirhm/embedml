import numpy as np
from embedml.tensor import Tensor
import torch


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


def test_matmul():
    xp = Tensor.ones((10, 3, 4, 5))
    yp = Tensor.ones((10, 3, 5, 6))
    zp = xp.matmul(yp)
    assert zp.shape == (10, 3, 4, 6)


def test_matmul_nd():
    xt = torch.randn(3, 2, 4, 5)
    yt = torch.randn(3, 2, 5, 6)
    zt = xt.matmul(yt)

    x = Tensor(xt)
    y = Tensor(yt)
    z = x.matmul(y)

    assert np.allclose(z.data, zt, rtol=1e-5, atol=1e-5)


def test_matmul_nd_grad():
    xt = torch.randn(3, 2, 4, 5).float().requires_grad_(True)
    yt = torch.randn(3, 2, 5, 6).float().requires_grad_(True)

    zt = xt.matmul(yt).sum()
    zt.backward()

    x = Tensor(xt.detach().numpy())
    y = Tensor(yt.detach().numpy())
    z = x.matmul(y).sum()
    z.backward()
    assert np.allclose(x.grad.data, xt.grad, rtol=1e-5, atol=1e-5)
    assert np.allclose(y.grad.data, yt.grad, rtol=1e-5, atol=1e-5)
    assert not xt.grad.requires_grad
    assert not yt.grad.requires_grad
