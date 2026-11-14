import pytest
import torch
import torch.nn.functional as F

from broadcasting import apply_linear, attention_scores, matmul_shape

# --- The @ shape rule: matrix math on the last two dims, broadcasting on the rest

COMPATIBLE = [
    ((2, 2), (2, 2), (2, 2)),
    ((2, 6, 3), (2, 3, 4), (2, 6, 4)),
    ((2, 6, 3), (3, 4), (2, 6, 4)),
    ((5, 1, 6, 3), (4, 3, 2), (5, 4, 6, 2)),
    ((2, 2), (2,), (2,)),
    ((2,), (2, 2), (2,)),
    ((3,), (3,), ()),
    ((2, 6, 3), (3,), (2, 6)),
    ((3,), (2, 3, 4), (2, 4)),
]

INCOMPATIBLE = [
    ((2, 6, 3), (2, 4, 3)),
    ((2, 3), (2, 3)),
    ((2, 6, 3), (3, 3, 4)),
    ((3,), (4,)),
]


def test_matmul_shape_compatible():
    for shape_a, shape_b, expected in COMPATIBLE:
        assert matmul_shape(shape_a, shape_b) == expected
        assert matmul_shape(shape_a, shape_b) == tuple((torch.ones(shape_a) @ torch.ones(shape_b)).shape)


def test_matmul_shape_incompatible():
    for shape_a, shape_b in INCOMPATIBLE:
        with pytest.raises(ValueError):
            matmul_shape(shape_a, shape_b)


# --- Batched matmul in attention and nn.Linear -------------------------------


def test_attention_scores_2d():
    queries = torch.tensor([[1, 2], [3, 4]])
    keys = torch.tensor([[5, 6], [7, 8]])
    assert torch.equal(attention_scores(queries, keys), torch.tensor([[17, 23], [39, 53]]))


def test_attention_scores_batch_and_heads():
    torch.manual_seed(123)
    queries = torch.rand(2, 3, 6, 4)  # (batch, heads, num_tokens, head_dim)
    keys = torch.rand(2, 3, 5, 4)
    scores = attention_scores(queries, keys)
    assert scores.shape == torch.Size([2, 3, 6, 5])
    assert torch.allclose(scores[1, 2], queries[1, 2] @ keys[1, 2].T)
    assert torch.isclose(scores[0, 1, 4, 3], torch.dot(queries[0, 1, 4], keys[0, 1, 3]))


def test_apply_linear_broadcasts_weight():
    torch.manual_seed(123)
    x = torch.rand(2, 6, 3)  # (batch, num_tokens, d_in)
    weight = torch.rand(4, 3)  # (d_out, d_in), stored like nn.Linear
    out = apply_linear(x, weight)
    assert out.shape == torch.Size([2, 6, 4])
    assert torch.allclose(out, F.linear(x, weight))
