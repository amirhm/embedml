import torch
import numpy as np
from embedml.tensor import Tensor


def test_softmax():
    from torch import nn
    sm = nn.Softmax(dim=1)
    x = torch.empty((5, 10)).random_().requires_grad_(True)
    sm_out = sm(x)

    # torch imp:
    r = (x - x.max(dim=1, keepdim=True).values)
    e = r.exp()
    s = e.sum(dim=1, keepdim=True)
    sm_th = e / s

    assert np.allclose(sm_th.detach().numpy(), sm_out.detach().numpy())

    xt = Tensor(x.detach().numpy())
    rt = xt - xt.max(axis=1, keepdims=True)
    et = rt.exp()
    st = et.sum(axis=1)
    # sm_t = e / s

    assert np.allclose(r.detach().numpy(), rt.data)
    assert np.allclose(e.detach().numpy(), et.data)
    assert np.allclose(s.detach().numpy(), st.data)