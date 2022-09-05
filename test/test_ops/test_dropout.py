import torch
from embedml.tensor import Tensor
import numpy as np


def test_droupout():

    x = torch.arange(-10, 10, 2).float()
    lt = torch.nn.Dropout(0.5)
    yt = lt(x)
    assert np.allclose(yt[yt != 0], x[yt != 0] * 2)

    xe = Tensor(np.array(x))
    ye = xe.dropout(0.5)
    print(ye[yt.data != 0].data, xe[yt.data != 0].data * 2)
    y, x = ye.data, xe.data * 2
    assert np.allclose(y[y != 0], x[y != 0])
    print(ye.data, xe.data)
