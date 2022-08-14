import torch
import numpy as np
import pytest
from embedml.tensor import Tensor
from embedml.nn import Linear


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


@pytest.mark.parametrize("bs,feature_in,feature_out", [(2, 10, 2), (109, 510, 2)])
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

    assert np.allclose(c0.detach().numpy(), tc.data, rtol=1e-3, atol=1e-3)
    tc1 = tc.sum()
    assert np.allclose(c.detach().numpy(), tc1.data, rtol=1e-3)
    tc1.backward()
    assert np.allclose(l1.weight.grad, W.grad.data.T, rtol=1e-1)
    assert np.allclose(l1.bias.grad, b.grad.data, rtol=1e-3)


@pytest.mark.parametrize("bs,feature_in,feature_out", [(2, 10, 2)])
def test_nn_linear(bs, feature_in, feature_out):
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

    x = Tensor(inp)

    lt = Linear(feature_in, feature_out)
    lt.weight = Tensor(w.T)
    lt.bias = Tensor(b)
    tc = lt(x)

    limit = {"rtol":1e-3, "atol":1e-3}
    assert np.allclose(c0.detach().numpy(), tc.data, **limit)
    tc1 = tc.sum()
    assert np.allclose(c.detach().numpy(), tc1.data, **limit)
    tc1.backward()
    assert np.allclose(l1.weight.grad, lt.weight.grad.data.T, **limit)
    assert np.allclose(l1.bias.grad, lt.bias.grad.data, **limit)
    print(lt.weight.shape)
