import torch
from embedml.tensor import Tensor
import numpy as np
import pytest


def test_simple_grad_th():
    x = torch.tensor(2, dtype=torch.float32, requires_grad=True)
    y = 4 + 2 * x + 3
    y.backward()
    assert y == 11
    assert x.grad == 2


def test_simple_grad():
    x = Tensor(np.array(2), dtype=torch.float32, requires_grad=True)
    y = 4 + x * 2 + 3
    y.backward()
    assert y.cpu() == 11
    assert x.grad.cpu() == 2


def test_simple_grad_positions_1():
    x = Tensor(np.array(2), dtype=torch.float32, requires_grad=True)
    y = 8 + x + 2 * 3
    y.backward()
    assert y.cpu() == 16
    assert x.grad.cpu() == 1


def test_simple_grad_positions_2():
    x = Tensor(np.array(2), dtype=torch.float32, requires_grad=True)
    y = 8 * x * 2 * 3
    y.backward()
    assert y.cpu() == 96
    assert x.grad.cpu() == 48


def test_grad_linear():
    seed = np.random.get_state()

    x = torch.tensor(np.ones((300, 2)), dtype=torch.float32, requires_grad=True)
    y = torch.tensor(np.random.randn(2, 2), dtype=torch.float32, requires_grad=True)
    b = torch.tensor(np.random.randn(2), dtype=torch.float32, requires_grad=True)
    z = torch.matmul(x, y)
    a = z + b
    c = a.sum() * 10

    z.retain_grad()
    a.retain_grad()
    c.retain_grad()
    c.backward()

    np.random.set_state(seed)
    xx = Tensor(np.ones((300, 2)), dtype=torch.float32, requires_grad=True)
    yy = Tensor(np.random.randn(2, 2), dtype=torch.float32, requires_grad=True)
    bb = Tensor(np.random.randn(2), dtype=torch.float32, requires_grad=True)
    zz = xx.matmul(yy)
    aa = zz + bb
    cc = aa.sum() * 10

    cc.backward()

    assert np.allclose(z.grad, zz.grad.cpu())
    assert np.allclose(y.grad, yy.grad.cpu())
    assert np.allclose(x.grad, xx.grad.cpu())
    assert np.allclose(a.grad, aa.grad.cpu())
    assert np.allclose(b.grad, bb.grad.cpu())
    assert np.allclose(c.grad, cc.grad.cpu())


@pytest.mark.parametrize("bs,feature_in,feature_out", [(2, 10, 2)])
def test_linear_module(bs, feature_in, feature_out):
    from torch import nn
    inp = torch.randn(bs, feature_in)
    l1 = nn.Linear(feature_in, feature_out, bias=True)
    w = np.random.randn(feature_out, feature_in)
    b = np.random.randn(feature_out)
    l1.weight = nn.Parameter(torch.tensor(w, dtype=torch.float32, requires_grad=True))
    l1.bias = nn.Parameter(torch.tensor(b, dtype=torch.float32, requires_grad=True))

    c0 = l1(inp)
    c = c0.sum()
    c.retain_grad()
    c0.retain_grad()
    c.backward()

    x, W, b = Tensor(inp), Tensor(w.T), Tensor(b)

    tc0 = x.matmul(W)
    tc = tc0 + b

    assert np.allclose(c0.detach().numpy(), tc.data)
    tc1 = tc.sum()
    assert np.allclose(c.detach().numpy(), tc1.data)
    tc1.backward()
    assert np.allclose(l1.weight.grad, W.grad.data.T)
    assert np.allclose(l1.bias.grad, b.grad.data)


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


def test_max_grad():
    x = torch.empty((5, 10)).random_().requires_grad_(True)
    m = x.max(dim=1, keepdim=True).values
    r = (x - m)
    z = r.sum()
    m.retain_grad()
    r.retain_grad()
    z.retain_grad()
    z.backward()

    xt = Tensor(x.detach().numpy())
    mt = xt.max(axis=1, keepdims=True)
    rt = (xt - mt)
    zt = rt.sum()
    zt.backward()
    assert np.allclose(zt.data, z.detach().numpy())
