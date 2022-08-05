
import numpy as np
from embedml.tensor import Tensor
import torch


def test_exp():
    xt = Tensor(np.array(1))

    def module(x):
        y = (-2 * x).exp()
        return (1 - y)

    zt = module(xt)
    zt.backward()

    x = torch.tensor(1, dtype=torch.float32, requires_grad=True)
    z = module(x)
    z.backward()

    assert np.allclose(zt.cpu(), z.detach())


def test_exp_grad():
    x = torch.empty((5, 10)).random_(10).requires_grad_()
    e = x.exp()
    z = e.sum()
    e.retain_grad()
    z.retain_grad()
    z.backward()

    xt = Tensor(x.detach().numpy())
    et = xt.exp()
    zt = et.sum()
    zt.backward()

    assert np.allclose(x.detach().numpy(), xt.data)
    assert np.allclose(e.detach().numpy(), et.data)
    assert np.allclose(e.grad, et.grad.data)
