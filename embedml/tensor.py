import numpy as np
import torch


class Function:
    def __init__(self, function):
        self.func = function

    def __call__(self, *args, **kwargs):
        self.parents = args
        return self.forward(*args, **kwargs)

#    def __get__(self, instance, owner):
#        from functools import partial
#        return partial(self.__call__, instance)
#
    def forward(ctx, *args, **kwargs): raise NotImplemented
    def backward(ctx, *args, **kwargs): raise NotImplemented


class MatMul(Function):
    def forward(ctx, x1, x2):
        return Tensor(np.matmul(x1.data, x2.data), ctx=ctx)

    def backward(ctx, grad_out):
        x1, x2 = ctx.parents
        x1gard = Tensor(np.matmul(grad_out, x2.T), require_grad=False) if ctx.parents[0].requires_grad else None
        x2gard = Tensor(np.matmul(x1.T, grad_out), require_grad=False) if ctx.parents[1].requires_grad else None
        return x1gard, x2gard

class SUM(Function):
    def forward(ctx, x1):
        return Tensor(np.sum(x1.data), ctx=ctx)

    def backward(ctx, grad_out):
        return Tensor(Tensor.ones(ctx.parents.shape), requires_grad=False) if ctx.parents[0].requires_grad else None 
        

class Tensor:
    def __init__(self, data, dtype=np.float32, requires_grad=True, ctx=None):
        self.dtype = dtype
        self.shape = data.shape
        self.data = np.array(data)
        self.ctx = ctx
        self.requires_grad = requires_grad

    @classmethod
    def eye(cls, val):
        return cls(np.eye(val), dtype=np.float32)

    @classmethod
    def ones(cls, shape):
        return cls(np.ones(shape, dtype=np.float32))

    @classmethod
    def zeros(cls, shape):
        return cls(np.zeros(shape, dtype=np.float32))

    def cpu(self):
        return np.array(self.data)


    @MatMul
    def _matmul(x1, x2): pass

    @SUM
    def _sum(x1): pass


    def matmul(self, y):
        return self._matmul(self, y)

    def sum(self):
        return self._sum(self)

    def __add__(self, other):
        if isinstance(other, int):
            return type(self)(self.data + other)
        else:
            return type(self)(self.data + other.data)

    def __radd__(self, other):
        if isinstance(other, int):
            return type(self)(self.data + other)
        else:
            return type(self)(self.data + other.data)

    def __mul__(self, other):
        if isinstance(other, int):
            return type(self)(self.data + other)
        else:
            return type(self)(self.data + other.data)

    def __rmul__(self, other):
        if isinstance(other, int):
            return type(self)(self.data + other)
        else:
            return type(self)(self.data + other.data)




    def get_topo_graph(self):
        topological = []
        def _backward(node, visited, topological):
            if node.ctx is None:
                return
            for n in node.ctx.parents:
                if n not in visited:
                    visited.add(n)
                    topological.append(n)
                    _backward(n, visited, topological)
        _backward(self, set(), topological)
        return topological

    def backward(self):
        # Only placeholder to be implemented
        graph = self.get_topo_graph()
        for node in reversed(graph):
            node.grad = Tensor.zeros(node.data.shape)






seed = np.random.get_state()

x = torch.tensor(np.ones((3, 2)), dtype=torch.float32, requires_grad=True)
y = torch.tensor(np.random.randn(2, 2), dtype=torch.float32, requires_grad=True)
z = torch.matmul(x, y)
r = z.sum()


z.retain_grad()
r.retain_grad()
r.backward()
print(r.grad)

assert r.grad == 1
assert torch.allclose(z.grad, torch.ones(z.shape))
assert torch.allclose(x.grad, torch.matmul(z.grad, y.T))
assert torch.allclose(y.grad, torch.matmul(x.T, z.grad))


np.random.set_state(seed)
xx = Tensor(np.ones((3, 2)), dtype=torch.float32, requires_grad=True)
yy = Tensor(np.random.randn(2, 2), dtype=torch.float32, requires_grad=True)
zz = xx.matmul(yy)
rr = zz.sum()



# x = Tensor(np.array(2), dtype=np.float32, requires_grad=True)

# y = Tensor(np.array(3), dtype=np.float32, requires_grad=True)
# z = y + x
