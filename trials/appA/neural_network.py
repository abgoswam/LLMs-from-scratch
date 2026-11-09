"""Section A.5: NeuralNetwork."""

import torch
import torch.nn as nn

class NeuralNetwork(torch.nn.Module):
    """Multilayer perceptron with two hidden layers.

    self.layers is a torch.nn.Sequential, built in this order:
    Linear(num_inputs, 30) -> ReLU -> Linear(30, 20) -> ReLU -> Linear(20, num_outputs).
    The order matters: weights are initialized from the seed as each layer is created.
    forward(x) passes x through self.layers and returns the raw logits (no softmax).
    """

    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        # raise NotImplementedError
        self.layers = nn.Sequential(
            nn.Linear(num_inputs, 30),
            nn.ReLU(),
            nn.Linear(30, 20),
            nn.ReLU(),
            nn.Linear(20, num_outputs)
        )

    def forward(self, x):
        # raise NotImplementedError

        assert self.num_inputs == x.shape[1]

        logits = self.layers(x)

        assert self.num_outputs == logits.shape[1]
        return logits