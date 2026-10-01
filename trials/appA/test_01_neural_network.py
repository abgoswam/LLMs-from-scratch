import torch

from neural_network import NeuralNetwork

# --- A.5 Implementing multilayer neural networks --------------------------


def test_neural_network_trainable_parameter_count():
    model = NeuralNetwork(50, 3)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert num_params == 2213


def test_neural_network_seeded_forward_pass():
    torch.manual_seed(123)
    model = NeuralNetwork(50, 3)
    assert model.layers[0].weight.shape == (30, 50)

    torch.manual_seed(123)
    X = torch.rand((1, 50))
    out = model(X)
    assert torch.allclose(out, torch.tensor([[-0.1262, 0.1080, -0.1792]]), atol=1e-4)
