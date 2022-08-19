
import torch
import numpy as np
from embedml.nn import LogSoftmax
from embedml.tensor import Tensor
import pytest


@pytest.mark.parametrize('dim', [(0), (1)])
def test_logsoftmax(dim):
    from torch import nn
    sm = nn.LogSoftmax(dim=dim)
    x = torch.rand((5, 10)).requires_grad_(True)
    s = sm(x)

    xt = Tensor(x.detach().numpy())
    st = LogSoftmax(dim=dim)
    smt = st(xt)
    ss = s.detach().numpy()

    assert np.allclose(ss, smt.data, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize('dim', [(0), (1)])
def test_logsoftmax_zeros(dim):
    from torch import nn
    sm = nn.LogSoftmax(dim=dim)
    x = torch.empty((5, 10)).random_().requires_grad_(True)
    s = sm(x)

    xt = Tensor(x.detach().numpy())
    st = LogSoftmax(dim=dim)
    smt = st(xt)
    ss = s.detach().numpy()

    assert np.allclose(ss[ss > -10], smt.data[ss > -10], rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize('dim', [(0), (1)])
def test_logsoftmax_grad(dim):
    from torch import nn
    t_x = torch.rand((5, 10)).requires_grad_(True)
    sm = nn.LogSoftmax(dim=dim)
    t_s = sm(t_x)
    t_s.sum().backward()

    xt = Tensor(t_x.detach().numpy())
    st = LogSoftmax(dim=dim)
    smt = st(xt)
    smt.sum().backward()

    assert np.allclose(t_s.detach().numpy(), smt.data)
    assert np.allclose(t_x.grad, xt.grad.data, rtol=1e-3)
