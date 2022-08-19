from embedml.tensor import Tensor
from embedml.nn import LogSoftmax, Linear, Module
import numpy as np


def test_train_linear():
    bs = 5
    num_classes = 10
    in_f = 768
    out_f = num_classes

    class model(Module):
        def __init__(self, in_f, out_f):
            super().__init__()
            self.num_classes = out_f
            self.l1 = Linear(feature_in=in_f, feature_out=100)
            self.l2 = Linear(feature_in=100, feature_out=out_f)
            self.ac = LogSoftmax(dim=1)

        def forward(self, x):
            ret = self.ac(self.l2(self.l1(x)))
            return ret

    m = model(in_f, out_f)
    x = Tensor(np.random.randn(bs, 768))
    yt = np.zeros(bs * num_classes)
    idx = np.random.randint(0, num_classes, (bs)) + np.arange(0, (bs * num_classes), num_classes)
    yt[idx] = 1
    target = Tensor(yt.reshape((bs, num_classes)), requires_grad=False)
    y = m(x)
    loss = (y * target).sum()

    loss.backward()


if __name__ == "__main__":
    test_train_linear()
