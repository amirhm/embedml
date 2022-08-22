import torch
import numpy as np
import pytest
from embedml.tensor import Tensor


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
def test_div_xy_grad(x, y):
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


def test_div_xy_grad_com_graph():
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
