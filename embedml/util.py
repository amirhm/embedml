from abc import ABC
from abc import abstractmethod
import importlib
import numpy as np


class Dataset(ABC):
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass


class DataLoader:
    def __init__(self, dataset, batch_size=8, shuffle=False):
        self.bs = batch_size
        self.dataset = dataset
        self.shuffle = shuffle
        self.idx = np.arange(len(self.dataset))
        self.make_indexs()

    def make_indexs(self):
        self.index = 0
        if self.shuffle:
            np.random.shuffle(self.idx)

    def __len__(self):
        return (len(self.dataset) // self.bs)

    def __next__(self):
        if self.index < (len(self.dataset) // self.bs):
            slc = slice(self.index * self.bs, (self.index + 1) * self.bs)
            self.index += 1
            return self.dataset[self.idx[slc]]
        else:
            self.make_indexs()
            raise StopIteration

    def __iter__(self):
        return self


def draw(root):
    graph = importlib.import_module('graphviz')

    def get_name():
        num = 0
        while True:
            yield f"tensor_{num}"
            num += 1

    g = graph.Digraph(format='svg', graph_attr={'rankdir': 'TB'})
    tensor_dict = dict()
    r = get_name()
    for node in reversed(list(root.get_topo_graph())):
        for p in node.ctx.parents:
            if tensor_dict.get(id(p), 0) == 0:
                tensor_dict[id(p)] = next(r)
                color = 'green' if p.requires_grad else 'black'
                g.node(tensor_dict[id(p)], color=color)
            g.edge(tensor_dict[id(p)], str((node.ctx)))
        if tensor_dict.get(id(node), 0) == 0:
            tensor_dict[id(node)] = next(r)
            g.node(str(node.ctx), label=node.ctx.__class__.__name__, color='red')
        g.edge(str((node.ctx)), tensor_dict[id(node)])
    return g
