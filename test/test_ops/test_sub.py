from embedml.tensor import Tensor
import numpy as np


def test_sub():
    x = Tensor(np.array(3))
    assert np.allclose((1 - x).cpu(), -2)
    assert np.allclose((x - 1).cpu(), 2)


def test_sub_mat():
    x = np.random.randn(5, 10)
    y = np.random.randn(5, 10)

    xt = Tensor(x)
    yt = Tensor(y)
    assert np.allclose((xt - yt).cpu(), x - y)
    assert np.allclose((yt - xt).cpu(), y - x)


def test_sub_mat_grad():
    xn = np.random.randn(2, 3)
    yn = np.random.randn(2, 3)
    import torch
    x = torch.tensor(xn, requires_grad=True)
    y = torch.tensor(yn, requires_grad=True)
    z = (x - y).sum()
    z.retain_grad()
    z.backward()

    xt = Tensor(xn)
    yt = Tensor(yn)
    zt = (xt - yt).sum()
    zt.backward()
    assert np.allclose(xt.grad.cpu(), x.grad)
    assert np.allclose(yt.grad.cpu(), y.grad)


def test_sub_mat_grad_broadcast():
    xn = np.random.randn(2, 3)
    yn = np.random.randn(1, 3)
    import torch
    x = torch.tensor(xn, requires_grad=True)
    y = torch.tensor(yn, requires_grad=True)
    z = (x - y).sum()
    z.retain_grad()
    z.backward()

    xt = Tensor(xn)
    yt = Tensor(yn)
    zt = (xt - yt).sum()
    zt.backward()
    assert np.allclose(xt.grad.cpu(), x.grad)
    assert np.allclose(yt.grad.cpu(), y.grad)
