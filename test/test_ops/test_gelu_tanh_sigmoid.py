import torch
from embedml.tensor import Tensor
import numpy as np


def test_sigmoid():

    x = torch.arange(-2, 2, 0.1)
    y = x.sigmoid()

    xe = Tensor(np.array(x))
    ye = xe.sigmoid()
    assert np.allclose(y, ye.data)


def test_tanh():
    x = torch.arange(-2, 2, 0.1)
    y = x.tanh()

    xe = Tensor(np.array(x))
    ye = xe.tanh()
    assert np.allclose(y, ye.data)


def test_gelu():
    xt = torch.arange(-2, 2, 0.1).float()
    lt = torch.nn.GELU()
    yt = lt(xt)

    xe = Tensor(np.array(xt))
    ye = xe.gelu()

    assert np.allclose(yt, ye.data, atol=1e-3, rtol=1e-5)


def test_gelu_grad():
    xt = torch.arange(-10, 10).float().requires_grad_(True)
    lt = torch.nn.GELU()
    yt = lt(xt)
    lt = yt.sum()
    lt.backward()
    xe = Tensor(np.array(xt.detach()))
    ye = xe.gelu()
    le = ye.sum()
    le.backward()

    assert np.allclose(lt.detach().numpy(), le.data, atol=1e-3, rtol=1e-5)
    assert np.allclose(xt.grad, xe.grad.data, atol=1e-3, rtol=1e-5)
