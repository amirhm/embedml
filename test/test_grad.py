import torch
from embedml.tensor import Tensor
import numpy as np


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

    x = torch.tensor(np.ones((3, 2)), dtype=torch.float32, requires_grad=True)
    y = torch.tensor(np.random.randn(2, 2), dtype=torch.float32, requires_grad=True)
    z = torch.matmul(x, y)
    r = z.sum()


    z.retain_grad()
    r.retain_grad()
    r.backward()

    np.random.set_state(seed)
    xx = Tensor(np.ones((3, 2)), dtype=torch.float32, requires_grad=True)
    yy = Tensor(np.random.randn(2, 2), dtype=torch.float32, requires_grad=True)
    zz = xx.matmul(yy)
    rr = zz.sum()
    rr.backward()


    assert np.allclose(z.grad, zz.grad.cpu())
    assert np.allclose(y.grad, yy.grad.cpu())
    assert np.allclose(x.grad, xx.grad.cpu())
