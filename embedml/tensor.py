import numpy as np
import torch


def broadcast(data, shape):
    src_shape = data.shape
    src_dims = len(src_shape)
    dims = len(shape)
    if src_shape == shape:
        return data
    if dims < src_dims:
        if shape[0] == src_shape[0]:
            return data.sum(axis=-1)
        elif shape[0] == src_shape[-1]:
            return data.sum(axis=0)


class Function:
    def __init__(self, function):
        self.func = function

    def __call__(self, *args, **kwargs):
        self.parents = args
        return self.forward(*args, **kwargs)

    def forward(ctx, *args, **kwargs): raise NotImplementedError
    def backward(ctx, *args, **kwargs): raise NotImplementedError


class MatMul(Function):
    def forward(ctx, x1, x2):
        return Tensor(np.matmul(x1.data, x2.data), ctx=ctx)

    def backward(ctx, grad_out):
        x1, x2 = ctx.parents
        x1gard = Tensor(np.matmul(grad_out.cpu(), x2.cpu().T), requires_grad=False) if x1.requires_grad else None
        x2gard = Tensor(np.matmul(x1.cpu().T, grad_out.cpu()), requires_grad=False) if x2.requires_grad else None
        x1.grad = x1gard
        x2.grad = x2gard


class SUM(Function):
    def forward(ctx, x1):
        return Tensor(np.sum(x1.data), ctx=ctx)

    def backward(ctx, grad_out):
        ctx.parents[0].grad = Tensor.ones(ctx.parents[0].shape, requires_grad=False) if ctx.parents[0].requires_grad else None 


class ADD(Function):
    def forward(ctx, x1, x2):
        return Tensor(x1.data + x2.data, ctx=ctx)

    def backward(ctx, grad_out):
        x1, x2 = ctx.parents
        x1.grad = Tensor(broadcast(grad_out.data, x1.shape), requires_grad=False) if x1.requires_grad else None
        x2.grad = Tensor(broadcast(grad_out.data, x2.shape), requires_grad=False) if x2.requires_grad else None


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
    def ones(cls, shape, requires_grad=False):
        return cls(np.ones(shape, dtype=np.float32), requires_grad=requires_grad)

    @classmethod
    def zeros(cls, shape):
        return cls(np.zeros(shape, dtype=np.float32))

    def cpu(self):
        return np.array(self.data)


    @MatMul
    def _matmul(x1, x2): pass

    @SUM
    def _sum(x1): pass

    @ADD
    def _add(x1, x2): pass

    def matmul(self, y):
        return self._matmul(self, y)

    def sum(self):
        return self._sum(self)

    def __add__(self, other):
        if isinstance(other, int):
            other = Tensor(np.array(other))
        return self._add(self, other)

    def __radd__(self, other):
        if isinstance(other, int):
            return type(self)(self.data + other)
        else:
            return type(self)(self.data + other.data)

    def __mul__(self, other):
        if isinstance(other, int):
            return type(self)(self.data * other)
        else:
            return type(self)(self.data * other.data)

    def __rmul__(self, other):
        if isinstance(other, int):
            return type(self)(self.data * other)
        else:
            return type(self)(self.data * other.data)

    def get_topo_graph(self):
        topological = [self]

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
        self.grad = Tensor(np.array(1), requires_grad=False)
        graph = self.get_topo_graph()
        for node in graph:
            if node.ctx:
                node.ctx.backward(node.grad)
