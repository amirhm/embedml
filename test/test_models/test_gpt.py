from collections import namedtuple
from models.gpt import GPT
from embedml.tensor import Tensor
from embedml.nn import SGD
import numpy as np
import pytest


@pytest.fixture
def model():
    config = namedtuple(
        'config',
        ['n_layer', 'n_embd', 'n_head', 'resid_pdrop', 'attn_pdrop', 'vocab_size', 'block_size', 'embd_pdrop'],
        defaults=[1, 48, 3, 0.1, 0.1, 10, 19, 0.1]
    )
    return GPT(config)


def test_gpt(model):
    bs, T = 5, 4
    inp = Tensor(np.ones((bs, T), dtype=int))
    logits = model(inp)
    assert logits.shape == (bs, T, model.cfg.vocab_size)
    logits.backward()
    assert model.transformer["wte"].weight.shape == (model.cfg.vocab_size, model.cfg.n_embd)
    assert model.transformer["wte"].weight.grad.shape == (model.cfg.vocab_size, model.cfg.n_embd)

    # steps = list(map(lambda x: (x.ctx, x.ctx.parents[0].shape, x.ctx.parents[1].shape if len(x.ctx.parents) > 1 else None), logits.get_topo_graph()))
    # print(*steps, sep="\n")


def test_learnable_params(model):
    m = model.get_parameters()
    assert len(m) == 54


def test_backward_path(model):
    m = model.get_parameters()

    bs, T = 5, 4
    inp = Tensor(np.ones((bs, T), dtype=int))
    logits = model(inp)
    loss = logits.sum()
    loss.backward()
    assert sum(map(lambda x: hasattr(x, 'grad'), m)) == len(m)

    def nodes(node):
        return [id(n) for n in node.ctx.parents]

    param = list(map(lambda x: id(x), m))
    bparam = list(map(nodes, loss.get_topo_graph()))
    t1 = list(filter(lambda x: isinstance(x, list), bparam))
    bparam = list(filter(lambda x: not isinstance(x, list), bparam))
    while t1 and (d := t1.pop()):
        bparam.extend(d)
    assert all(list(map(lambda x: x in bparam, param)))


def test_training(model):
    m = model.get_parameters()
    optim = SGD(m, lr=0.0001)
    bs, T = 5, 4
    for i in range(10):
        inp = Tensor(np.ones((bs, T), dtype=int))
        logits = model(inp)
        loss = logits.sum()
        loss.backward()
        optim.step()
        optim.zero_grad()
