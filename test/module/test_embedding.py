import torch
from torch import nn
import numpy as np
from embedml.tensor import Tensor
from embedml.nn import Embedding


def test_embedding():
    em = nn.Embedding(5, 10)

    w = em.weight.clone()
    idx = torch.LongTensor([1, 2, 3, 4])
    out = em(idx)
    assert np.array_equal(out.detach(), w[idx].detach())


def test_embedding_grad():
    em = nn.Embedding(6, 10)
    w = em.weight.clone()
    idx = torch.LongTensor([[1, 2, 3, 4], [0, 1, 3, 2], [3, 2, 4, 3]])
    out = em(idx)
    y = (out * 2).sum()
    y.retain_grad()
    em.weight.retain_grad()
    out.retain_grad()
    y.backward()

    et = Embedding(6, 10)
    et.weight = Tensor(w.detach())
    nidx = Tensor(idx, requires_grad=False)
    out_t = et(nidx)
    yt = (out_t * 2).sum()
    yt.backward()
    assert np.allclose(yt.data, y.detach())
    assert np.allclose(em.weight.grad, et.weight.grad.data)
