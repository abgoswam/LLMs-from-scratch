import torch
from torch.utils.data import DataLoader

from toy_dataset import ToyDataset

# --- A.6 Setting up efficient data loaders --------------------------------


def test_toy_dataset_len_and_item(X_train, y_train):
    train_ds = ToyDataset(X_train, y_train)
    assert len(train_ds) == 5
    x, y = train_ds[3]
    assert torch.equal(x, X_train[3])
    assert torch.equal(y, y_train[3])


def test_toy_dataset_shuffled_batches(X_train, y_train):
    train_ds = ToyDataset(X_train, y_train)
    torch.manual_seed(123)
    train_loader = DataLoader(dataset=train_ds, batch_size=2, shuffle=True, num_workers=0)

    batches = [(x, y.tolist()) for x, y in train_loader]
    assert [y for _, y in batches] == [[1, 0], [0, 0], [1]]
    assert torch.allclose(batches[0][0], torch.tensor([[2.3, -1.1], [-0.9, 2.9]]))
    assert torch.allclose(batches[2][0], torch.tensor([[2.7, -1.5]]))
