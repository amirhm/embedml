import torch
import torch.nn as nn
from embedml.tensor import Tensor
from embedml.nn import LayerNorm
import numpy as np
import pytest


@pytest.mark.parametrize('num', [2, 3])
def test_layernorm(num):

    x = torch.rand(3, 5, num)
    lt = nn.LayerNorm(num)
    yt = lt(x)

    xe = Tensor(np.array(x), requires_grad=False)
    le = LayerNorm(num)
    ye = le(xe)

    assert np.allclose(yt.detach().numpy(), ye.data, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize('num', [2, 6])
def test_module_trainable_params(num):
    le = LayerNorm(num, eps=1e-5)
    params = le.get_parameters()
    assert len(params) == 2
    assert params[0].shape[0] == num
    assert params[1].shape[0] == num


@pytest.mark.parametrize('num', [2, 3, 10])
def test_layernorm_grad(num):
    x = torch.rand(3, 5, num)
    lt = nn.LayerNorm(num, eps=0)
    yt = lt(x)
    lt.weight.retain_grad()
    yt.sum().backward()
    xr = Tensor(np.array(x), requires_grad=False)
    le = LayerNorm(num, eps=0)
    ye = le(xr)
    ye.sum().backward()
    assert np.allclose(yt.detach().numpy(), ye.data, rtol=1e-5, atol=1e-5)
    assert np.allclose(lt.weight.grad, le.weight.grad.data, rtol=1e-5, atol=1e-5)
    assert np.allclose(lt.bias.grad, le.bias.grad.data, rtol=1e-5, atol=1e-5)
