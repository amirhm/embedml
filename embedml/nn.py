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
        self.weight = Tensor(np.random.randn(self.fin, self.fout), requires_grad=True)
        self.bias = Tensor(np.random.randn(self.fout), requires_grad=True)

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
