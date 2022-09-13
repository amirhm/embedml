import numpy as np
from collections import deque
import sys
from functools import partialmethod


def max_broad(max_idx, grad_out, axis, grad):
    shape = max_idx.shape
    dims_strides = np.cumprod((1, *shape[::-1]))[::-1]
    dims = np.zeros_like(shape)
    if axis is None:
        grad0 = grad.reshape(-1)
        grad0[max_idx] = grad_out.data
    else:
        for v, (i, d) in enumerate(zip(max_idx.flatten(), grad_out.flatten())):
            for j in range(len(shape)):
                dims[j] = (v // dims_strides[j + 1]) % shape[j]
            dims[axis] = i
            grad[tuple(dims)] = d


def extend(data, shape, axis):
    ext_shape = tuple(shape[i] if axis in (i, -1) else 1 for i in range(len(shape)))
    return Tensor(np.tile(data.data, ext_shape), requires_grad=data.requires_grad)


def broadcast(data, shape):
    src_shape = data.shape
    if src_shape == shape:
        return data
    if len(shape) != len(src_shape):
        brd = tuple(range(len(src_shape) - len(shape)))
    else:
        brd = tuple(idx for idx, j in enumerate(zip(src_shape, shape)) if j[0] != j[1])
    return data.sum(axis=brd, keepdims=True)


class Function:
    def __init__(self):
        self.outs = []

    @classmethod
    def call(cls, *args, **kwargs):
        ctx = cls()
        ctx.parents = args
        ctx.requires_grad = any(ar.requires_grad for ar in args if isinstance(ar, Tensor))
        return ctx.forward(*args, **kwargs)

    def forward(ctx, *args, **kwargs): raise NotImplementedError
    def backward(ctx, *args, **kwargs): raise NotImplementedError


class MATMUL(Function):
    def forward(ctx, x1, x2):
        return Tensor(np.matmul(x1.data, x2.data), requires_grad=ctx.requires_grad, ctx=ctx)

    def backward(ctx, grad_out):
        x1, x2 = ctx.parents
        if x1.requires_grad:
            x1.grad += grad_out.matmul(x2.transpose())
        if x2.requires_grad:
            x2.grad += x1.transpose().matmul(grad_out)


class SUM(Function):
    def forward(ctx, x1, **kwargs):
        ctx.axis = kwargs.get("axis", -1)
        kwargs["keepdims"] = True
        return Tensor(np.sum(x1.data, **kwargs), requires_grad=ctx.requires_grad, ctx=ctx)

    def backward(ctx, grad_out):
        if ctx.parents[0].requires_grad:
            ctx.parents[0].grad += extend(grad_out, ctx.parents[0].shape, ctx.axis)


class ADD(Function):
    def forward(ctx, x1, x2):
        return Tensor(x1.data + x2.data, requires_grad=ctx.requires_grad, ctx=ctx)

    def backward(ctx, grad_out):
        x1, x2 = ctx.parents
        if x1.requires_grad:
            x1.grad += broadcast(grad_out, x1.shape)
        if x2.requires_grad:
            x2.grad += broadcast(grad_out, x2.shape)


class MAX(Function):
    def forward(ctx, x1, **kwargs):
        tmp = np.max(x1.data, **kwargs)
        ctx.max_idx = np.argmax(x1.data, **kwargs)
        ctx.axis = kwargs.get("axis", None)
        return Tensor(tmp, requires_grad=ctx.requires_grad, ctx=ctx)

    def backward(ctx, grad_out):
        if ctx.parents[0].requires_grad:
            tmp = np.zeros(ctx.parents[0].shape)
            max_broad(ctx.max_idx, grad_out, ctx.axis, tmp)
            tmp_g = Tensor(tmp, requires_grad=False)
            ctx.parents[0].grad += tmp_g


class EXP(Function):
    def forward(ctx, x1):
        ret = Tensor(np.exp(x1.data), requires_grad=ctx.requires_grad, ctx=ctx)
        ctx.outs.append(ret)
        return ret

    def backward(ctx, grad_out):
        if ctx.parents[0].requires_grad:
            ctx.parents[0].grad += grad_out * ctx.outs[0]


class SUB(Function):
    def forward(ctx, x1, x2):
        return Tensor(x1.data - x2.data, requires_grad=ctx.requires_grad, ctx=ctx)

    def backward(ctx, grad_out):
        x1, x2 = ctx.parents
        if x1.requires_grad:
            x1.grad += broadcast(grad_out.data, x1.shape)
        if x2.requires_grad:
            x2.grad += broadcast(-1 * grad_out, x2.shape)


class MUL(Function):
    def forward(ctx, x1, x2):
        return Tensor(x1.data * x2.data, requires_grad=ctx.requires_grad, ctx=ctx)

    def backward(ctx, grad_out):
        x1, x2 = ctx.parents
        if x1.requires_grad:
            x1.grad += grad_out * x2
        if x2.requires_grad:
            x2.grad += broadcast(grad_out * x1, x2.shape)


class RELU(Function):
    def forward(ctx, x1):
        ctx.outs.append(x1.data > 0)
        return Tensor(x1.data * (x1.data > 0), requires_grad=ctx.requires_grad, ctx=ctx)

    def backward(ctx, grad_out):
        idx, x = ctx.outs[0], ctx.parents[0]
        if x.requires_grad:
            x.grad[idx] += grad_out[idx]


class POW(Function):
    def forward(ctx, x1, x2):
        ret = Tensor(np.power(x1.data, x2.data), requires_grad=ctx.requires_grad, ctx=ctx)
        ctx.outs.append(ret)
        return ret

    def backward(ctx, grad_out):
        if ctx.parents[0].requires_grad:
            ctx.parents[0].grad += grad_out * (ctx.parents[1] * ctx.parents[0] ** (ctx.parents[1] - 1))
        if ctx.parents[1].requires_grad:
            ctx.parents[1].grad += broadcast(grad_out * ctx.parents[0].log() * ctx.outs[0], ctx.parents[1].shape)


class LOG(Function):
    def forward(ctx, x):
        x.data[x.data <= 1e-7] = 1e-7
        ret = Tensor(np.log(x.data), requires_grad=ctx.requires_grad, ctx=ctx)
        return ret

    def backward(ctx, grad_out):
        if ctx.parents[0].requires_grad:
            ctx.parents[0].grad += grad_out * (ctx.parents[0] ** -1)


class SLC(Function):
    def forward(ctx, x, *args):
        ctx.outs.append(*args)
        return Tensor(x.data.__getitem__(*args), requires_grad=ctx.requires_grad, ctx=ctx)

    def backward(ctx, grad_out):
        args = ctx.outs.pop()
        ctx.parents[0].grad[args] += grad_out


class PERMUTE(Function):
    def forward(ctx, x, order):
        ctx.outs.append(np.argsort(order))
        return Tensor(np.moveaxis(x.data, order, tuple(range(x.data.ndim))), requires_grad=ctx.requires_grad, ctx=ctx)

    def backward(ctx, grad_out):
        order = ctx.outs.pop()
        ctx.parents[0].grad += grad_out.permute(order)


class RESHAPE(Function):
    def forward(ctx, x, shape):
        ctx.outs.append((shape, x.shape))
        return Tensor(np.reshape(x.data, shape), requires_grad=ctx.requires_grad, ctx=ctx)

    def backward(ctx, grad_out):
        _, oshape = ctx.outs.pop()
        ctx.parents[0].grad += grad_out.reshape(oshape)


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
    def zeros(cls, shape, requires_grad=False):
        return cls(np.zeros(shape, dtype=np.float32), requires_grad=requires_grad)

    @staticmethod
    def totensor(func):

        def inner(*args, **kwargs):
            vals = (Tensor(np.array(val), dtype=np.float32, requires_grad=False) if isinstance(val, int) or isinstance(val, float) else val for val in args)
            return func(*vals, **kwargs)
        return inner

    def flatten(self):
        return self.data.flatten()

    def cpu(self):
        return np.array(self.data)

    def matmul(self, y):
        return self._matmul(y)

    def mean(self, axis=None):
        out = self.sum(axis=axis)
        return out * (np.prod(out.shape) / np.prod(self.shape))

    def relu(self, **kwargs): return self._relu(**kwargs)
    def sigmoid(self): return (1.0 + (-1 * self).exp()) ** -1
    def tanh(self): return 2.0 * ((2.0 * self).sigmoid()) - 1.0
    def gelu(self, **kwargs): return 0.5 * self * (1 + (0.7978845608028654 * (self + 0.044715 * self ** 3)).tanh())
    def sum(self, **kwargs): return self._sum(**kwargs)

    def exp(self): return self._exp()
    def pow(self, other): return self._pow(other)
    def log(self): return self._log()
    def max(self, **kwargs): return self._max(**kwargs)

    def div(self, other): return self * (other ** -1)

    def add(self, other): return self._add(other)
    def mul(self, other): return self._mul(other)
    def sub(self, other): return self._sub(other)
    def permute(self, orders): return self._permute(orders)
    def reshape(self, shape): return self._reshape(shape)

    def transpose(self, dim0=-2, dim1=-1):
        shape = list(range(len(self.shape)))
        shape[dim0], shape[dim1] = shape[dim1], shape[dim0]
        return self.permute(tuple(shape))

    def dropout(self, p):
        _mask = np.random.binomial(1, 1.0 - p, size=self.shape)
        return self * Tensor(_mask, requires_grad=False) * (1 / (1.0 - p))

    def get_topo_graph(self):
        topological = []
        self.grad = Tensor(np.array(1), requires_grad=False)

        def _backward(node, visited, topological):
            visited.add(node)
            if node.ctx:
                for n in node.ctx.parents:
                    if isinstance(n, Tensor) and n not in visited:
                        n.grad = Tensor.zeros(n.shape)
                        _backward(n, visited, topological)
                topological.append(node)
        _backward(self, set(), topological)
        return reversed(topological)

    def backward(self):
        # Only placeholder to be implemented
        self.grad = Tensor(np.array(1), requires_grad=False)
        graph = self.get_topo_graph()
        for node in graph:
            if node.ctx:
                node.ctx.backward(node.grad)

    def __repr__(self):
        return f"{self.data}"

    def __getitem__(self, slc): return self._slc(slc)

    def __setitem__(self, slc, x): self.data[slc] = x.data

    def move(self, x):
        self.data = x.data
        return self


for func in ['MATMUL', 'SUM', 'ADD', 'EXP', 'MAX', 'SUB', 'MUL', 'POW', 'LOG', 'SLC', 'RELU', 'PERMUTE', 'RESHAPE']:
    setattr(Tensor, f'_{func.lower()}', partialmethod(getattr(sys.modules[__name__], func).call))


def add_method(name, method):
    setattr(Tensor, f"__{name}__", lambda self, x: Tensor.totensor(method)(self, x))
    setattr(Tensor, f"__r{name}__", lambda self, x: Tensor.totensor(method)(x, self))
    setattr(Tensor, f"__i{name}__", lambda self, x: self.move(Tensor.totensor(method)(self, x)))


deque((add_method(func, getattr(Tensor, func)) for func in ['add', 'mul', 'sub', 'pow']), maxlen=0)
