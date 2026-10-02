"""Section A.6: ToyDataset."""

from torch.utils.data import Dataset


class ToyDataset(Dataset):
    """Map-style dataset over in-memory feature and label tensors.

    __init__ stores X (features, one row per example) and y (labels).
    __getitem__(index) returns the tuple (X[index], y[index]).
    __len__ returns the number of labels.
    """

    def __init__(self, X, y):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
