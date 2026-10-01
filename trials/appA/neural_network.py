"""Section A.5: NeuralNetwork."""

import torch


class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
