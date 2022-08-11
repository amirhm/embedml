from embedml.tensor import Tensor


class Module:
    def __init__(self):
        self.outs = []

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(ctx, *args, **kwargs): raise NotImplementedError
    def backward(ctx, *args, **kwargs): raise NotImplementedError


class Linear(Module):
    def __init__(self, feature_in, feature_out):
        self.fin = feature_in
        self.fout = feature_out
        self.weight = Tensor.ones((self.fin, self.fout), requires_grad=True)
        self.bias = Tensor.ones((self.fout), requires_grad=True)

    def forward(self, x):
        return x.matmul(self.weight) + self.bias
