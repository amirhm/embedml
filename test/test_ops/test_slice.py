import pytest
from embedml.tensor import Tensor
import numpy as np


def test_slicing():

    x = Tensor(np.array([[1, 2, 3], [4, 5, 6]]))
    assert x[0].shape == (3,)
    assert x[:, 0].shape == (2,)
