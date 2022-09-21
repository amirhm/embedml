from embedml.nn import Module
from embedml.nn import Linear
from embedml.nn import LogSoftmax
from embedml.tensor import Tensor
import numpy as np

class simple(Module):
    def __init__(self):
        super().__init__()
        self.l1 = Linear(28 * 28, 64)
        self.l2 = Linear(64, 10)
        self.ac = LogSoftmax(dim=1)

    def forward(self, data):
        y0 = self.l1(data).relu()
        y1 = self.ac(self.l2(y0))
        return y1
    

def test_model():
    gpt = simple()
    bs, T = 5, 28 * 28
    inp = Tensor(np.ones((bs, T), dtype=int))
    logits = gpt(inp)
    assert logits.shape == (bs, 10)
    logits.backward()

    # steps = list(map(lambda x: (x.ctx, x.ctx.parents[0].shape, x.ctx.parents[1].shape if len(x.ctx.parents) > 1 else None), logits.get_topo_graph()))
    # print(*steps, sep="\n")


def test_learnable_params():
    gpt = simple()
    m = gpt.get_parameters()
    assert len(m) == 4


def test_backward_path():
    gpt = simple()
    m = gpt.get_parameters()

    bs, T = 5, 28 * 28
    inp = Tensor(np.ones((bs, T), dtype=int))
    logits = gpt(inp)
    loss = logits.sum()
    loss.backward()
    assert sum(map(lambda x: hasattr(x, 'grad'), m)) == len(m)
    param = list(map(lambda x: id(x), m))

    def nodes(node):
        return [id(n) for n in node.ctx.parents]


    param = list(map(lambda x: id(x), m))
    bparam = list(map(nodes, loss.get_topo_graph()))
    t1 = list(filter(lambda x: isinstance(x, list), bparam))
    bparam = list(filter(lambda x: not isinstance(x, list), bparam))
    while t1 and (d := t1.pop()):
        bparam.extend(d)
    assert all(list(map(lambda x: x in bparam, param)))
