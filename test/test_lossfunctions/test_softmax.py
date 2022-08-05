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
    sm = e / s

    assert np.allclose(sm.detach().numpy(), sm_out.detach().numpy())

    xt = Tensor(x.detach().numpy())
    rt = xt - xt.max(axis=1, keepdims=True)
    et = rt.exp()
    st = et.sum(axis=1)
    smt = et.div(st)

    assert np.allclose(r.detach().numpy(), rt.data)
    assert np.allclose(e.detach().numpy(), et.data)
    assert np.allclose(s.detach().numpy(), st.data)
    assert np.allclose(sm.detach().numpy(), smt.data)


def test_softmax_grad():
    x = torch.empty((5, 10)).random_().requires_grad_(True)
    r = (x - x.max(dim=1, keepdim=True).values)
    e = r.exp()
    s = e.sum(dim=1, keepdim=True)
    sm = e / s

    list(map(lambda x: x.retain_grad(), [r, e, s, sm]))
    sm.sum().backward()

    xt = Tensor(x.detach().numpy())
    rt = xt - xt.max(axis=1, keepdims=True)
    et = rt.exp()
    st = et.sum(axis=1)
    # pt = st ** -1
    smt = et.div(st)

    smt.sum().backward()
    assert np.allclose(sm.grad, smt.grad.data)
    assert np.allclose(x.grad, xt.grad.data)
#    assert np.allclose(s.grad, st.grad.data)
#    assert np.allclose(r.grad, rt.grad.data)
#    assert np.allclose(e.grad, et.grad.data)
