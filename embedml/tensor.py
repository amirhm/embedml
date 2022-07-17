
import numpy as np


class Function:
    def __init__(self, function):
        self.func = function
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
        pass

    def __get__(self, instance, owner):
        from functools import partial
        return partial(self.__call__, instance)

    @staticmethod
    def forward(args, kwargs):
        ctx = args[0]
        ctx.saved_vectors = args
        print(args)
        pass

    def backward():
        pass


class Tensor:
    def __init__(self, data, dtype=np.float32, requires_grad=True):
        self.dtype = dtype
        self.shape = data.shape
        self.data = np.array(data)
        self.ctx = None
        self.requires_grad = requires_grad

    @classmethod
    def eye(cls, val):
        return cls(np.eye(val), dtype=np.float32)

    @classmethod
    def ones(cls, shape):
        return cls(np.ones(shape, dtype=np.float32))

    def cpu(self):
        return np.array(self.data)

    def matmul(self, y):
        cx = np.matmul(self.data, y.data)
        return type(self)(cx)

    def sum(self):
        cx = np.sum(self.data)
        return type(self)(cx)

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

    @Function
    def _add(self, x):
        pass



import torch
import numpy as np

x = torch.tensor(np.ones((3, 2)), dtype=torch.float32, requires_grad=True)
y = torch.tensor(np.random.randn(2, 2), dtype=torch.float32, requires_grad=True)
z = torch.matmul(x , y)
r = z.sum()

z.retain_grad()
r.backward()
print(y.grad)
print(x.grad)

# x = Tensor(np.array(2), dtype=np.float32, requires_grad=True)
# y = Tensor(np.array(3), dtype=np.float32, requires_grad=True)
# z = y + x