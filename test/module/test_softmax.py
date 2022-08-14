import torch
import numpy as np
from embedml.nn import Softmax
from embedml.tensor import Tensor
import pytest


@pytest.mark.parametrize('dim', [(0), (1)])
def test_softmax(dim):
    from torch import nn
    sm = nn.Softmax(dim=dim)
    x = torch.empty((5, 10)).random_().requires_grad_(True)
    s = sm(x)

    xt = Tensor(x.detach().numpy())
    st = Softmax(dim=dim)
    smt = st(xt)

    assert np.allclose(s.detach().numpy(), smt.data)


@pytest.mark.parametrize('dim', [(0), (1)])
def test_softmax_grad(dim):
    from torch import nn
    t_x = torch.empty((5, 10)).random_().requires_grad_(True)
    sm = nn.Softmax(dim=dim)
    t_s = sm(t_x)
    t_s.sum().backward()

    xt = Tensor(t_x.detach().numpy())
    st = Softmax(dim=dim)
    smt = st(xt)
    smt.sum().backward()

    assert np.allclose(t_s.detach().numpy(), smt.data)
    assert np.allclose(t_x.grad, xt.grad.data)
