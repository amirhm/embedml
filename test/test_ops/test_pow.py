import torch
import numpy as np
from embedml.tensor import Tensor


def test_pow():
    x = torch.tensor(np.random.randn(10))
    y = x.pow(2)

    xt = Tensor(np.array(x))

    yt = xt**2

    assert np.allclose(xt.cpu(), x.detach())
    assert np.allclose(yt.cpu(), y.detach())


def test_pow_xint_grad():
    x = torch.tensor(np.random.randn(10), requires_grad=True)
    y = x.pow(2)
    y.retain_grad()
    y.sum().backward()

    xt = Tensor(np.array(x.detach()))
    yt = xt ** 2
    yt.sum().backward()

    assert np.array_equal(xt.grad.cpu(), x.grad)
    assert np.array_equal(yt.grad.cpu(), y.grad)


def test_pow_xy_grad():
    x = torch.tensor(np.random.randint(1, 10, 10).astype(float), requires_grad=True)
    y = torch.tensor(np.random.randn(10), requires_grad=True)
    z = x.pow(y)
    y.retain_grad()
    z.sum().backward()

    xt = Tensor(np.array(x.detach()))
    yt = Tensor(np.array(y.detach()))
    zt = xt ** yt
    zt.sum().backward()

    assert np.array_equal(zt.cpu(), z.detach().numpy())
    assert np.array_equal(xt.grad.cpu(), x.grad)
    assert np.array_equal(yt.grad.cpu(), y.grad)
