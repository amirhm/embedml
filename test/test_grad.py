import torch
from embedml.tensor import Tensor
import numpy as np


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
