import torch
from embedml.tensor import Tensor
import numpy as np
import pytest


def test_simple_grad_th():
    x = torch.tensor(2, dtype=torch.float32, requires_grad=True)
    y = 4 + 2 * x + 3
    y.backward()
    assert x.grad == 2


def test_simple_grad():
    x = Tensor(np.array(2), dtype=torch.float32, requires_grad=True)
    y = 4 + x * 2 + 3
#    y.backward()
#    assert y.cpu() == 11
#    assert x.grad == 2



def test_grad_linear():
    seed = np.random.get_state()

    x = torch.tensor(np.ones((300, 2)), dtype=torch.float32, requires_grad=True)
    y = torch.tensor(np.random.randn(2, 2), dtype=torch.float32, requires_grad=True)
    z = torch.matmul(x, y).sum()


    z.retain_grad()
    z.backward()

    np.random.set_state(seed)
    xx = Tensor(np.ones((300, 2)), dtype=torch.float32, requires_grad=True)
    yy = Tensor(np.random.randn(2, 2), dtype=torch.float32, requires_grad=True)
    zz = xx.matmul(yy).sum()
    zz.backward()

    assert np.allclose(z.grad, zz.grad.cpu())
    assert np.allclose(y.grad, yy.grad.cpu())
    assert np.allclose(x.grad, xx.grad.cpu())


@pytest.mark.parametrize("bs,feature_in,feature_out", [(2, 10, 2)])
def test_linear_module(bs, feature_in, feature_out):
    from torch import nn
    inp = torch.randn(bs, feature_in)    
    l1 = nn.Linear(feature_in, feature_out, bias=False)
    w = np.random.randn(feature_out, feature_in)
    b = np.random.randn(feature_out)
    l1.weight = nn.Parameter(torch.tensor(w, dtype=torch.float32, requires_grad=True))
    l1.bias = nn.Parameter(torch.tensor(b, dtype=torch.float32, requires_grad=True))

    c = l1(inp).sum()
    c.retain_grad()
    c.backward()

    x, W, b = Tensor(inp), Tensor(w.T), Tensor(b)

    tl1 = x.matmul(W) 
    tc = tl1.sum()
    tc.backward()
    assert np.allclose(l1.weight.grad, W.grad.data.T)








