import torch
import numpy as np
from embedml.tensor import Tensor


def test_max_grad():
    x = torch.empty((5, 10)).random_().requires_grad_(True)
    m = x.max(dim=1, keepdim=True).values
    r = (x - m)
    z = r.sum()
    m.retain_grad()
    r.retain_grad()
    z.retain_grad()
    z.backward()

    xt = Tensor(x.detach().numpy())
    mt = xt.max(axis=1, keepdims=True)
    rt = (xt - mt)
    zt = rt.sum()
    zt.backward()
    assert np.allclose(zt.data, z.detach().numpy())