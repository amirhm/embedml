from embedml.util import DataLoader, Dataset
from embedml.tensor import Tensor
import numpy as np
import pytest


class mydataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.length = data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.length


def test_dataset():
    mdata = mydataset(Tensor(np.arange(15).reshape((5, 3))))

    assert len(mdata) == 5
    assert np.array_equal(mdata[0].data, np.array([0, 1, 2]))


@pytest.mark.parametrize('bs', [2, 4, 8])
@pytest.mark.parametrize('shuffle', [True, False])
def test_dataloader(bs, shuffle):
    mdata = mydataset(Tensor(np.arange(24).reshape((8, 3))))
    dloader = DataLoader(mdata, batch_size=bs, shuffle=True)

    assert len(mdata) // bs == len(list(dloader))
    li = list(dloader)

    acc = sum([it.sum() for it in li])
    assert np.allclose(acc.data, sum(range(24)))
    for d in dloader:
        assert d.shape == (bs, 3)
