import torch
import numpy as np
from embedml.tensor import Tensor
import pytest


@pytest.mark.parametrize('axis', [(0), (1), (None)])
def test_max(axis):
    x = torch.empty((5, 10)).random_().requires_grad_(True)
    m = x.max(dim=axis, keepdim=True).values if axis is not None else x.max()
    r = (x - m)
    z = r.sum()
    m.retain_grad()
    r.retain_grad()
    z.retain_grad()
    z.backward()

    xt = Tensor(x.detach().numpy())
    mt = xt.max(axis=axis, keepdims=True) if axis is not None else xt.max()
    rt = (xt - mt)
    zt = rt.sum()
    zt.backward()
    assert np.allclose(zt.data, z.detach().numpy())


@pytest.mark.parametrize('axis', [(0), (1), (2), (None)])
def test_max_grad0(axis):
    x = torch.empty((5, 20, 10)).random_().requires_grad_(True)
    m = x.max(dim=axis, keepdim=True).values if axis is not None else x.max()
    z = m.sum() * 123
    m.retain_grad()
    z.retain_grad()
    z.backward()

    xt = Tensor(x.detach().numpy())
    mt = xt.max(axis=axis, keepdims=True) if axis is not None else xt.max()

    zt = mt.sum() * 123
    zt.backward()
    assert np.allclose(zt.data, z.detach().numpy())
    assert np.allclose(xt.grad.data, x.grad)
    assert np.allclose(mt.grad.data, m.grad)

    assert not xt.grad.requires_grad
    assert not mt.grad.requires_grad


@pytest.mark.parametrize('axis', [(0), (1)])
def test_max_grad(axis):
    x = torch.empty((5, 10)).random_().requires_grad_(True)
    m = x.max(dim=axis, keepdim=True).values
    r = (x - m)
    z = r.sum() * 10
    m.retain_grad()
    r.retain_grad()
    z.retain_grad()
    z.backward()

    xt = Tensor(x.detach().numpy())
    mt = xt.max(axis=axis, keepdims=True)
    rt = (xt - mt)
    zt = rt.sum() * 10
    zt.backward()
    assert np.allclose(zt.data, z.detach().numpy())
    assert np.allclose(xt.grad.data, x.grad)
    assert np.allclose(mt.grad.data, m.grad)
