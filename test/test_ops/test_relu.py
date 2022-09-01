import torch
from embedml.tensor import Tensor
import numpy as np


def test_relu_forward():

    xt = torch.randn(10, 20, 30, 40)
    yt = xt.relu()

    xe = Tensor(xt.detach().numpy())
    ye = xe.relu()

    np.allclose(ye.data, yt)


def test_relu():
    def func(x):
        y = x.relu()
        z = x + y
        return z.sum()

    x = torch.randn(10, 20, 30, 40).requires_grad_(True)
    xe = Tensor(x.detach().numpy())

    loss = list(map(func, [x, xe]))
    list(map(lambda x: x.backward(), loss))
    np.allclose(x.grad, xe.grad.data)
    np.allclose(loss[0].detach().numpy(), loss[1].data)
