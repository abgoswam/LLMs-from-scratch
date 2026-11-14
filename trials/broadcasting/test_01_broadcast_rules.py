import pytest
import torch

from broadcasting import broadcast_shape

# --- The broadcasting rules (NumPy and PyTorch docs) ------------------------

COMPATIBLE = [
    ((256, 256, 3), (3,), (256, 256, 3)),
    ((8, 1, 6, 1), (7, 1, 5), (8, 7, 6, 5)),
    ((5, 4), (1,), (5, 4)),
    ((5, 4), (4,), (5, 4)),
    ((15, 3, 5), (15, 1, 5), (15, 3, 5)),
    ((15, 3, 5), (3, 5), (15, 3, 5)),
    ((15, 3, 5), (3, 1), (15, 3, 5)),
    ((5, 1, 4, 1), (3, 1, 1), (5, 3, 4, 1)),
    ((1,), (3, 1, 7), (3, 1, 7)),
    ((6, 2), (6, 1), (6, 2)),
]

INCOMPATIBLE = [
    ((3,), (4,)),
    ((2, 1), (8, 4, 3)),
    ((5, 2, 4, 1), (3, 1, 1)),
    ((6, 2), (6,)),
]


def test_broadcast_shape_compatible():
    for shape_a, shape_b, expected in COMPATIBLE:
        assert broadcast_shape(shape_a, shape_b) == expected
        assert broadcast_shape(shape_a, shape_b) == tuple(torch.broadcast_shapes(shape_a, shape_b))


def test_broadcast_shape_incompatible():
    for shape_a, shape_b in INCOMPATIBLE:
        with pytest.raises(ValueError):
            broadcast_shape(shape_a, shape_b)
