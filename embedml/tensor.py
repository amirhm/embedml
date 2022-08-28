import numpy as np
from collections import deque


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
    return np.tile(data, ext_shape)


def broadcast(data, shape):
    src_shape = data.shape
    if src_shape == shape:
        return data
    elif shape == ():
        return data.sum()
    if shape[0] == src_shape[0]:
        return data.sum(axis=-1, keepdims=True)
    elif shape[0] == src_shape[-1]:
        return data.sum(axis=0, keepdims=True)
    elif shape[1] == src_shape[1]:
        return data.sum(axis=0, keepdims=True)


class Function:
    def __init__(self):
        self.outs = []

    def __call__(self, *args, **kwargs):
        ctx = type(self)()
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
            x1.grad += Tensor(np.matmul(grad_out.cpu(), x2.cpu().T), requires_grad=False)
        if x2.requires_grad:
            x2.grad += Tensor(np.matmul(x1.cpu().T, grad_out.cpu()), requires_grad=False)


class SUM(Function):
    def forward(ctx, x1, **kwargs):
        ctx.axis = kwargs.get("axis", -1)
        return Tensor(np.sum(x1.data, **kwargs, keepdims=True), requires_grad=ctx.requires_grad, ctx=ctx)

    def backward(ctx, grad_out):
        if ctx.parents[0].requires_grad:
            ctx.parents[0].grad += Tensor(extend(grad_out.data, ctx.parents[0].shape, ctx.axis), requires_grad=False)


class ADD(Function):
    def forward(ctx, x1, x2):
        return Tensor(x1.data + x2.data, requires_grad=ctx.requires_grad, ctx=ctx)

    def backward(ctx, grad_out):
        x1, x2 = ctx.parents
        if x1.requires_grad:
            x1.grad += Tensor(broadcast(grad_out.data, x1.shape), requires_grad=False)
        if x2.requires_grad:
            x2.grad += Tensor(broadcast(grad_out.data, x2.shape), requires_grad=False)


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
        ctx.parents[0].grad += (grad_out * ctx.outs[0]) if ctx.parents[0].requires_grad else None


class SUB(Function):
    def forward(ctx, x1, x2):
        return Tensor(x1.data - x2.data, requires_grad=ctx.requires_grad, ctx=ctx)

    def backward(ctx, grad_out):
        x1, x2 = ctx.parents
        if x1.requires_grad:
            x1.grad += Tensor(broadcast(grad_out.data, x1.shape), requires_grad=False)
        if x2.requires_grad:
            x2.grad += Tensor(broadcast(-grad_out.data, x2.shape), requires_grad=False)


class MUL(Function):
    def forward(ctx, x1, x2):
        return Tensor(x1.data * x2.data, requires_grad=ctx.requires_grad, ctx=ctx)

    def backward(ctx, grad_out):
        x1, x2 = ctx.parents
        if x1.requires_grad:
            x1.grad += Tensor(grad_out.cpu() * x2.data, requires_grad=False)
        if x2.requires_grad:
            x2.grad += Tensor(broadcast(grad_out.cpu() * x1.data, x2.shape), requires_grad=False)


class RELU(Function):
    pass


class POW(Function):
    def forward(ctx, x1, x2):
        ret = Tensor(np.power(x1.data, x2.data), requires_grad=ctx.requires_grad, ctx=ctx)
        ctx.outs.append(ret)
        return ret

    def backward(ctx, grad_out):
        if ctx.parents[0].requires_grad:
            tmp = ctx.parents[1] * np.power(ctx.parents[0].data, ctx.parents[1].data - 1)
            ctx.parents[0].grad += (grad_out * tmp)
        if ctx.parents[1].requires_grad:
            tmp = Tensor(np.log(ctx.parents[0].data) * ctx.outs[0].data)
            ctx.parents[1].grad += broadcast(grad_out * tmp, ctx.parents[1].shape)


class LOG(Function):
    def forward(ctx, x):
        x.data[x.data <= 1e-7] = 1e-7
        ret = Tensor(np.log(x.data), requires_grad=ctx.requires_grad, ctx=ctx)
        return ret

    def backward(ctx, grad_out):
        if ctx.parents[0].requires_grad:
            ctx.parents[0].grad += (grad_out * (ctx.parents[0] ** -1))


class SLC(Function):
    def forward(ctx, x, *args):
        ctx.outs.append(*args)
        return Tensor(x.data.__getitem__(*args), requires_grad=ctx.requires_grad, ctx=ctx)

    def backward(ctx, grad_out):
        args = ctx.outs.pop()
        ctx.parents[0].grad[args] += grad_out.data


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
            vals = (Tensor(np.array(val), dtype=np.float32, requires_grad=False) if isinstance(val, int) else val for val in args)
            return func(*vals, **kwargs)
        return inner

    def flatten(self):
        return self.data.flatten()

    def cpu(self):
        return np.array(self.data)

    def matmul(self, y):
        return self._matmul(self, y)

    def sum(self, **kwargs): return self._sum(self, **kwargs)

    def exp(self): return self._exp(self)
    def pow(self, p): return self._pow(self, p)
    def log(self): return self._log(self)
    def max(self, **kwargs): return self._max(self, **kwargs)

    def div(self, other): return self * (other ** -1)

    def add(self, other): return self._add(self, other)
    def mul(self, other): return self._mul(self, other)
    def sub(self, other): return self._sub(self, other)

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

    def __getitem__(self, slc): return self._slc(self, slc)

    def __setitem__(self, slc, x): self.data[slc] = x.data

    def move(self, x):
        self.data = x.data
        return self


for func in ['MATMUL', 'SUM', 'ADD', 'EXP', 'MAX', 'SUB', 'MUL', 'POW', 'LOG', 'SLC']:
    setattr(Tensor, f'_{func.lower()}', eval(f"{func}()"))


def add_method(name, method):
    setattr(Tensor, f"__{name}__", lambda self, x: Tensor.totensor(method)(self, x))
    setattr(Tensor, f"__r{name}__", lambda self, x: Tensor.totensor(method)(x, self))
    setattr(Tensor, f"__i{name}__", lambda self, x: self.move(Tensor.totensor(method)(self, x)))


deque((add_method(func, getattr(Tensor, func)) for func in ['add', 'mul', 'sub', 'pow']), maxlen=0)
