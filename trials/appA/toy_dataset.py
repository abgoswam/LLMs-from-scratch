"""Section A.6: ToyDataset."""

from torch.utils.data import Dataset


class ToyDataset(Dataset):
    def __init__(self, X, y):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
