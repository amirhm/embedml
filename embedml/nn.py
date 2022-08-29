from embedml.tensor import Tensor
import numpy as np


class Module:
    def __init__(self):
        self.outs = []

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(ctx, *args, **kwargs): raise NotImplementedError  # noqa: E704
    def backward(ctx, *args, **kwargs): raise NotImplementedError  # noqa: E704

    def get_parameters(self):
        params = []

        def parameters(obj):
            nonlocal params
            if isinstance(obj, Tensor) and obj.requires_grad:
                params += [obj]
            if hasattr(obj, "__dict__"):
                for k, v in obj.__dict__.items():
                    parameters(v)
        parameters(self)
        return params


class Linear(Module):
    def __init__(self, feature_in, feature_out):
        self.fin = feature_in
        self.fout = feature_out
        stdev = 1.0 / np.sqrt(self.fin)
        self.weight = Tensor(np.random.uniform(-stdev, stdev, (self.fin * self.fout)).reshape(self.fin, self.fout), requires_grad=True)
        self.bias = Tensor(np.random.uniform(-stdev, stdev, self.fout).reshape(self.fout), requires_grad=True)

    def forward(self, x):
        return x.matmul(self.weight) + self.bias


class Softmax(Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        e = (x - x.max(axis=self.dim, keepdims=True)).exp()
        s = e.sum(axis=self.dim)
        sm = e.div(s)
        return sm


class LogSoftmax(Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        e = (x - x.max(axis=self.dim, keepdims=True)).exp()
        s = e.sum(axis=self.dim)
        sm = e.log() - s.log()
        return sm


class Optimizer:
    def __init__(self, params):
        self.params = params

    def zero_grad(self):
        for param in self.params:
            param.grad = param.grad * 0


class SGD(Optimizer):
    def __init__(self, params, lr):
        super().__init__(params)
        self.lr = lr

    def step(self):
        for param in self.params:
            param -= param.grad * self.lr


def one_hot(label, num_classes):
    shape = label.shape[0], num_classes
    y = np.zeros(shape)
    y_ptr = y.reshape((-1,))
    idx = label.flatten() + np.arange(0, (np.prod(shape)), shape[1])
    y_ptr[idx] = 1
    return y


class CrossEntropy(Module):
    def __init__(self, num_class):
        super().__init__()
        self.num_class = num_class

    def forward(self, y, target):
        T = Tensor(one_hot(target, self.num_class), requires_grad=False)
        return (y * T).sum() * -1
