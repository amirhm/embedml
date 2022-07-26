import numpy as np
import torch


def broadcast(data, shape):
    src_shape = data.shape
    src_dims = len(src_shape)
    dims = len(shape)
    if src_shape == shape:
        return data
    if shape[0] == src_shape[0]:
        return data.sum(axis=-1)
    elif shape[0] == src_shape[-1]:
        return data.sum(axis=0)

class Function:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        self.parents = args
        return self.forward(*args, **kwargs)

    def forward(ctx, *args, **kwargs): raise NotImplementedError
    def backward(ctx, *args, **kwargs): raise NotImplementedError


class MATMUL(Function):
    def forward(ctx, x1, x2):
        return Tensor(np.matmul(x1.data, x2.data), ctx=ctx)

    def backward(ctx, grad_out):
        x1, x2 = ctx.parents
        x1.grad = Tensor(np.matmul(grad_out.cpu(), x2.cpu().T), requires_grad=False) if x1.requires_grad else None
        x2.grad = Tensor(np.matmul(x1.cpu().T, grad_out.cpu()), requires_grad=False) if x2.requires_grad else None


class SUM(Function):
    def forward(ctx, x1, dim=None):
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


class MAX(Function):
    def forward(ctx, x1, **kwargs):
        return Tensor(np.max(x1.data, **kwargs), ctx=ctx)

    def backward(ctx, grad_out):
        ctx.parents[0].grad = Tensor.ones(ctx.parents[0].shape, requires_grad=False) if ctx.parents[0].requires_grad else None 

class EXP(Function):
    def forward(ctx, x1):
        return Tensor(np.exp(x1.data), ctx=ctx)

    def backward(ctx, grad_out):
        ctx.parents[0].grad = Tensor.ones(ctx.parents[0].shape, requires_grad=False) if ctx.parents[0].requires_grad else None 



class SUB(Function):
    def forward(ctx, x1, x2):
        return Tensor(x1.data - x2.data, ctx=ctx)

    def backward(ctx, grad_out):
        x1, x2 = ctx.parents
        x1.grad = Tensor(broadcast(grad_out.data, x1.shape), requires_grad=False) if x1.requires_grad else None
        x2.grad = Tensor(broadcast(grad_out.data, x2.shape), requires_grad=False) if x2.requires_grad else None

class RELU(Function):pass


class POW(Function): pass


class LOG(Function): pass


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

    @staticmethod
    def totensor(func):

        def inner(*args, **kwargs):
            vals = (Tensor(np.array(val),dtype=np.float32) if isinstance(val, int) else val for val in args)
            return func(*vals)
        return inner


    def cpu(self):
        return np.array(self.data)

    def matmul(self, y):
        return self._matmul(self, y)

    def sum(self): return self._sum(self)

    def exp(self): return self._exp(self)
    def max(self, **kwargs): return self._max(self, **kwargs)

    def add(self, other): return self._add(self, other)
    def mul(self, other): return type(self)(self.data * other.data)
    def sub(self, other): return self._sub(self, other)

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

    def __repr__(self):
        return f"{self.data}"

for func in ['MATMUL', 'SUM', 'ADD', 'EXP', 'MAX', 'SUB']:
    setattr(Tensor, f'_{func.lower()}', eval(f"{func}()"))



def add_method(name, method):
    setattr(Tensor, f"__{name}__" , Tensor.totensor(method))
    setattr(Tensor, f"__r{name}__" , lambda self, x: Tensor.totensor(method)(x, self))

for func in ['add', 'mul', 'sub']: add_method(func, getattr(Tensor, func))

a = Tensor(np.array(10))

b = Tensor(np.array(23))


print((a + b).data )
print((a + 2).data)
print(a.__add__)
print(a.__sub__)
print(a.__mul__)