import pytest
import torch


@pytest.fixture
def X_train():
    return torch.tensor([
        [-1.2, 3.1],
        [-0.9, 2.9],
        [-0.5, 2.6],
        [2.3, -1.1],
        [2.7, -1.5]
    ])


@pytest.fixture
def y_train():
    return torch.tensor([0, 0, 0, 1, 1])


@pytest.fixture
def X_test():
    return torch.tensor([
        [-0.8, 2.8],
        [2.6, -1.6],
    ])


@pytest.fixture
def y_test():
    return torch.tensor([0, 1])
