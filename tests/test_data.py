import torch

from sysml_gpt.data import get_batch, train_val_split


def test_train_val_split():
    data = torch.arange(10)

    train_data, val_data = train_val_split(data, train_ratio=0.8)

    assert train_data.tolist() == list(range(8))
    assert val_data.tolist() == [8, 9]


def test_get_batch_shapes():
    data = torch.arange(100)

    x, y = get_batch(data, batch_size=4, block_size=8)

    assert x.shape == (4, 8)
    assert y.shape == (4, 8)


def test_get_batch_targets_are_shifted():
    data = torch.arange(100)

    x, y = get_batch(data, batch_size=4, block_size=8)

    assert torch.equal(x[:, 1:], y[:, :-1])