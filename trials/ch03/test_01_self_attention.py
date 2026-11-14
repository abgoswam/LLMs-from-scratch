import pytest
import torch

from self_attention import SelfAttention_v1, SelfAttention_v2, softmax_naive

# --- 3.3.1 A simple self-attention mechanism without trainable weights ------


@pytest.fixture
def attn_scores_2(inputs):
    query = inputs[1]  # 2nd input token is the query

    attn_scores_2 = torch.empty(inputs.shape[0])
    for i, x_i in enumerate(inputs):
        attn_scores_2[i] = torch.dot(x_i, query)
    return attn_scores_2


def test_softmax_naive_attention_weights(attn_scores_2):
    attn_weights_2_naive = softmax_naive(attn_scores_2)
    expected = torch.tensor([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
    assert torch.allclose(attn_weights_2_naive, expected, atol=1e-4)
    assert torch.isclose(attn_weights_2_naive.sum(), torch.tensor(1.0))


def test_softmax_2d():
    x = torch.tensor([
        [0.2996, 0.8053],
        [0.3061, 0.8210],
        [0.3058, 0.8203],
        [0.2948, 0.7939],
        [0.2927, 0.7891],
        [0.2990, 0.8040],
    ])
    out = softmax_naive(x)
    assert out.shape == x.shape
    assert torch.allclose(out, torch.softmax(x, dim=-1))
    assert torch.allclose(out.sum(dim=-1), torch.ones(6))


# --- 3.4.2 Implementing a compact SelfAttention class -------------------------


def test_self_attention_v1_outputs(inputs):
    torch.manual_seed(123)
    sa_v1 = SelfAttention_v1(d_in=3, d_out=2)
    expected = torch.tensor([
        [0.2996, 0.8053],
        [0.3061, 0.8210],
        [0.3058, 0.8203],
        [0.2948, 0.7939],
        [0.2927, 0.7891],
        [0.2990, 0.8040],
    ])
    assert torch.allclose(sa_v1(inputs), expected, atol=1e-4)


def test_self_attention_v2_outputs(inputs):
    torch.manual_seed(789)
    sa_v2 = SelfAttention_v2(d_in=3, d_out=2)
    expected = torch.tensor([
        [-0.0739, 0.0713],
        [-0.0748, 0.0703],
        [-0.0749, 0.0702],
        [-0.0760, 0.0685],
        [-0.0763, 0.0679],
        [-0.0754, 0.0693],
    ])
    assert torch.allclose(sa_v2(inputs), expected, atol=1e-4)


# --- Exercise 3.1 Comparing SelfAttention_v1 and SelfAttention_v2 -----------


def test_v1_with_transposed_v2_weights_matches_v2(inputs):
    torch.manual_seed(123)
    sa_v1 = SelfAttention_v1(d_in=3, d_out=2)
    torch.manual_seed(123)
    sa_v2 = SelfAttention_v2(d_in=3, d_out=2)

    sa_v1.W_query = torch.nn.Parameter(sa_v2.W_query.weight.T)
    sa_v1.W_key = torch.nn.Parameter(sa_v2.W_key.weight.T)
    sa_v1.W_value = torch.nn.Parameter(sa_v2.W_value.weight.T)

    expected = torch.tensor([
        [-0.5337, -0.1051],
        [-0.5323, -0.1080],
        [-0.5323, -0.1079],
        [-0.5297, -0.1076],
        [-0.5311, -0.1066],
        [-0.5299, -0.1081],
    ])
    assert torch.allclose(sa_v2(inputs), expected, atol=1e-4)
    assert torch.allclose(sa_v1(inputs), sa_v2(inputs), atol=1e-6)
