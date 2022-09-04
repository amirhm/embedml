import torch
import numpy as np
from embedml.tensor import Tensor
import pytest


@pytest.mark.parametrize('dim', [0, 1])
def test_mean(dim):
    x = torch.empty((5, 20, 10)).random_(10).requires_grad_()
    y = x.mean(dim=dim) * 10
    z = y.sum()
    y.retain_grad()
    z.retain_grad()
    z.backward()

    xt = Tensor(x.detach().numpy(), dtype=torch.float32, requires_grad=True)
    yt = xt.mean(axis=dim) * 10
    zt = yt.sum()
    zt.backward()
    assert np.allclose(z.detach(), zt.data)
    assert np.allclose(x.grad, xt.grad.data)
    assert np.allclose(y.grad, yt.grad.data)
    assert np.allclose(z.grad, zt.grad.data)
