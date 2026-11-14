import pytest
import torch

from broadcasting import closest_code, closest_codes, scale_channels

# --- NumPy docs: scaling the colors of an image ------------------------------


def test_scale_channels():
    image = torch.arange(12).reshape(2, 2, 3)  # 2x2 pixels, 3 channels
    scale = torch.tensor([1, 10, 100])
    expected = torch.tensor([
        [[0, 10, 200], [3, 40, 500]],
        [[6, 70, 800], [9, 100, 1100]],
    ])
    assert torch.equal(scale_channels(image, scale), expected)


# --- NumPy docs: a practical example, vector quantization --------------------


@pytest.fixture
def codes():
    return torch.tensor([
        [102.0, 203.0],
        [132.0, 193.0],
        [45.0, 155.0],
        [57.0, 173.0],
    ])


def test_closest_code(codes):
    observation = torch.tensor([111.0, 188.0])
    assert closest_code(observation, codes) == 0


def test_closest_codes_batched(codes):
    observations = torch.tensor([
        [111.0, 188.0],
        [130.0, 190.0],
        [50.0, 160.0],
        [60.0, 175.0],
    ])
    assert closest_codes(observations, codes).tolist() == [0, 1, 2, 3]
