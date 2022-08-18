import torch
import numpy as np
from embedml.tensor import Tensor


def test_log():
    x = torch.tensor(np.random.rand(100) + 1e-6)
    y = x.log()

    xt = Tensor(np.array(x))
    yt = xt.log()

    assert np.allclose(xt.cpu(), x.detach())
    assert np.allclose(yt.cpu(), y.detach())


def test_log_xint_grad():
    x = torch.tensor(np.random.rand(100) + 1e-6, requires_grad=True)
    y = x.log()
    y.retain_grad()
    y.sum().backward()

    xt = Tensor(np.array(x.detach()))
    yt = xt.log()
    yt.sum().backward()

    assert np.allclose(xt.grad.cpu(), x.grad)
    assert np.allclose(yt.grad.cpu(), y.grad)


def test_log_xnd_grad():
    x = torch.tensor(np.random.rand(100, 20) + 1e-6, requires_grad=True)
    y = x.log()
    y.retain_grad()
    y.sum().backward()

    xt = Tensor(np.array(x.detach()))
    yt = xt.log()
    yt.sum().backward()

    assert np.allclose(xt.grad.cpu(), x.grad)
    assert np.allclose(yt.grad.cpu(), y.grad)
