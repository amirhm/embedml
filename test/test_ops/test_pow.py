import torch
import numpy as np
from embedml.tensor import Tensor
import pytest

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


@pytest.mark.parametrize('x,y', [
    (
        torch.empty((5, 10)).random_().requires_grad_(True),
        torch.tensor(np.random.randn(5, 1), requires_grad=True)
    ),
    (
        torch.zeros((5, 10), requires_grad=True),
        torch.tensor(np.random.randn(5, 1), requires_grad=True)
    )
    ])
def test_div_xy_grad_1(x, y):
    z = x / y
    y.retain_grad()
    z.sum().backward()

    xt = Tensor(np.array(x.detach()))
    yt = Tensor(np.array(y.detach()))
    zt = xt.div(yt)
    zt.sum().backward()

    assert np.allclose(zt.cpu(), z.detach().numpy())
    assert np.allclose(xt.grad.cpu(), x.grad)
    assert np.allclose(yt.grad.cpu(), y.grad)


def test_div_xy_grad_4():
    r = torch.randn((5, 10), requires_grad=True)
    x = (r - r.max(dim=1, keepdim=True).values).requires_grad_(True)
    y = torch.ones((5, 1), requires_grad=True)
    z = x / y
    y.retain_grad()
    x.retain_grad()
    z.sum().backward()

    xt = Tensor(np.array(x.detach()))
    yt = Tensor(np.array(y.detach()))
    zt = xt.div(yt)
    zt.sum().backward()
    assert np.allclose(zt.cpu(), z.detach().numpy())
    assert np.allclose(xt.grad.cpu(), x.grad)
    assert np.allclose(yt.grad.cpu(), y.grad)
