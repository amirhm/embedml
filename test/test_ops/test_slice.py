import pytest
from embedml.tensor import Tensor
import numpy as np
import torch


def test_slicing():
    x = Tensor(np.array([[1, 2, 3], [4, 5, 6]]))
    assert x[0].shape == (3,)
    assert x[0, 2:].shape == (1,)
    assert x[:, 0].shape == (2,)


def module_1(x):
    y = x[0, 2:] * 10
    return y.sum()


def module_2(x):
    y = x[0] * 10
    z = x[1] * 20
    r = y + z
    return r.sum()


@pytest.mark.parametrize('module,', [module_1, module_2])
def test_slicing_grad(module):
    x = Tensor(np.array([[1, 2, 3], [4, 5, 6]]))
    le = module(x)
    le.backward()

    xt = torch.tensor(x.data, dtype=torch.float, requires_grad=True)
    lt = module(xt)
    lt.backward()
    assert np.allclose(lt.detach().numpy(), le.data)
    assert np.allclose(xt.grad, xt.grad.data)
