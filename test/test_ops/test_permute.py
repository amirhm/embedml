import pytest
import torch
from embedml.tensor import Tensor
import numpy as np


@pytest.mark.parametrize('ndim', [3, 2, 5])
def test_permute(ndim):
    sh = np.random.randint(1, 5, ndim)
    xe = Tensor.ones(sh)
    order = tuple(np.random.permutation(ndim))
    ye = xe.permute(order)
    sh_new = tuple(sh[x] for x in order)

    assert ye.shape == sh_new


@pytest.mark.parametrize('ndim', [3, 2, 5])
def test_permute_grad(ndim):
    sh = np.random.randint(1, 5, ndim)
    x = torch.randn(*sh).float().requires_grad_(True)
    orders = np.random.permutation(ndim)
    r = x.permute(*orders)
    y = torch.randn(r.shape).float().requires_grad_(True)
    z = r * y

    x.retain_grad()
    lt = z.sum()
    lt.backward()

    xe = Tensor(np.array(x.detach().numpy()))
    re = xe.permute(orders)
    ye = Tensor(np.array(y.detach().numpy()))
    ze = re * ye
    le = ze.sum()
    le.backward()
    assert np.allclose(x.grad, xe.grad.data, atol=1e-4, rtol=1e-4)
