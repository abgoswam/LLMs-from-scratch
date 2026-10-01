import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from accuracy import compute_accuracy
from neural_network import NeuralNetwork
from toy_dataset import ToyDataset

# --- A.7 A typical training loop ------------------------------------------
# Same training loop as the notebook, run on my NeuralNetwork and ToyDataset.


@pytest.fixture
def train_loader(X_train, y_train):
    return DataLoader(dataset=ToyDataset(X_train, y_train), batch_size=2, shuffle=True, num_workers=0, drop_last=True)


@pytest.fixture
def test_loader(X_test, y_test):
    return DataLoader(dataset=ToyDataset(X_test, y_test), batch_size=2, shuffle=False, num_workers=0)


@pytest.fixture
def trained_model(train_loader):
    torch.manual_seed(123)
    model = NeuralNetwork(num_inputs=2, num_outputs=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    num_epochs = 3

    for epoch in range(num_epochs):

        model.train()
        for batch_idx, (features, labels) in enumerate(train_loader):

            logits = model(features)

            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()

    return model


def test_trained_model_outputs(trained_model, X_train):
    trained_model.eval()
    with torch.no_grad():
        outputs = trained_model(X_train)
    expected = torch.tensor([
        [2.8569, -4.1618],
        [2.5382, -3.7548],
        [2.0944, -3.1820],
        [-1.4814, 1.4816],
        [-1.7176, 1.7342],
    ])
    assert torch.allclose(outputs, expected, atol=1e-4)


def test_compute_accuracy(trained_model, train_loader, test_loader):
    assert compute_accuracy(trained_model, train_loader) == 1.0
    assert compute_accuracy(trained_model, test_loader) == 1.0
