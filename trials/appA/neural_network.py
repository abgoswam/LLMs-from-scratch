"""Section A.5: NeuralNetwork."""

import torch


class NeuralNetwork(torch.nn.Module):
    """Multilayer perceptron with two hidden layers.

    self.layers is a torch.nn.Sequential, built in this order:
    Linear(num_inputs, 30) -> ReLU -> Linear(30, 20) -> ReLU -> Linear(20, num_outputs).
    The order matters: weights are initialized from the seed as each layer is created.
    forward(x) passes x through self.layers and returns the raw logits (no softmax).
    """

    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
