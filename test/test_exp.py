
import numpy as np
from embedml.tensor import Tensor
import torch


def test_exp():
    xt = Tensor(np.array(1))
    def module(x):
        y = (-2 * x ).exp()
        return (1 - y)

    zt = module(xt)
    zt.backward()

    x = torch.tensor(1, dtype=torch.float32, requires_grad=True)
    z = module(x)
    z.backward()

    assert np.allclose(zt.cpu(), z.detach())