import torch
from embedml.tensor import Tensor
import numpy as np


def test_droupout():

    x = torch.arange(-10, 10, 0.01).float()
    lt = torch.nn.Dropout(0.5)
    yt = lt(x)

    xe = Tensor(np.array(x))
    ye = xe.dropout(0.5)
