


from re import I


def test_f(self):
    return self

class test:
    def __init__(self, c2 ):
        self.func = c2
        pass 
    def __call__(self, *args, **kwds):
        breakpoint()
        return self.forward(self, args, kwds)

    def __get__(self, instance, owner):
        breakpoint()
        from functools import partial
        return partial(self.__call__, instance)


    def forward(self, *data, **kwargs):
        self.saved_vectors = data
        breakpoint()
        return data[1] *2
    def backward(self):
        pass

class data:
    def __init__(self):
        self.require_grad = True
        pass


    @test
    def sum(self, x):
        return 10 + x



d = data()
y = d.sum()