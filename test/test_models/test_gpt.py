from collections import namedtuple
from models.gpt import GPT
from embedml.tensor import Tensor
import numpy as np


def test_gpt():
    config = namedtuple(
        'config',
        ['n_layer', 'n_embd', 'n_head', 'resid_pdrop', 'attn_pdrop', 'vocab_size', 'block_size', 'embd_pdrop'],
        defaults=[1, 48, 3, 0.1, 0.1, 10, 19, 0.1]
    )
    gpt = GPT(config)
    bs, T = 5, 4
    inp = Tensor(np.ones((bs, T), dtype=int))
    logits = gpt(inp)
    assert logits.shape == (bs, T, gpt.cfg.vocab_size)
    logits.backward()
    assert gpt.transformer["wte"].weight.shape == (gpt.cfg.vocab_size, gpt.cfg.n_embd)
    assert gpt.transformer["wte"].weight.grad.shape == (gpt.cfg.vocab_size, gpt.cfg.n_embd)

    # steps = list(map(lambda x: (x.ctx, x.ctx.parents[0].shape, x.ctx.parents[1].shape if len(x.ctx.parents) > 1 else None), logits.get_topo_graph()))
    # print(*steps, sep="\n")


def test_learnable_params():
    config = namedtuple(
        'config',
        ['n_layer', 'n_embd', 'n_head', 'resid_pdrop', 'attn_pdrop', 'vocab_size', 'block_size', 'embd_pdrop'],
        defaults=[1, 48, 3, 0.1, 0.1, 10, 19, 0.1]
    )
    gpt = GPT(config)
    m = gpt.get_parameters()
    assert len(m) == 54


def test_backward_path():
    config = namedtuple(
        'config',
        ['n_layer', 'n_embd', 'n_head', 'resid_pdrop', 'attn_pdrop', 'vocab_size', 'block_size', 'embd_pdrop'],
        defaults=[1, 48, 3, 0.1, 0.1, 10, 19, 0.1]
    )
    gpt = GPT(config)
    m = gpt.get_parameters()

    bs, T = 5, 4
    inp = Tensor(np.ones((bs, T), dtype=int))
    logits = gpt(inp)
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
    a = (list(map(lambda x: x in bparam, param)))
