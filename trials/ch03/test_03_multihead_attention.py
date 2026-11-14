import torch

from multihead_attention import MultiHeadAttention, MultiHeadAttentionWrapper

# --- 3.6.1 Stacking multiple single-head attention layers -------------------


def test_multihead_wrapper_outputs(batch):
    torch.manual_seed(123)
    mha = MultiHeadAttentionWrapper(
        d_in=3, d_out=2, context_length=batch.shape[1], dropout=0.0, num_heads=2
    )
    context_vecs = mha(batch)

    expected = torch.tensor([
        [-0.4519, 0.2216, 0.4772, 0.1063],
        [-0.5874, 0.0058, 0.5891, 0.3257],
        [-0.6300, -0.0632, 0.6202, 0.3860],
        [-0.5675, -0.0843, 0.5478, 0.3589],
        [-0.5526, -0.0981, 0.5321, 0.3428],
        [-0.5299, -0.1081, 0.5077, 0.3493],
    ])
    assert context_vecs.shape == torch.Size([2, 6, 4])
    assert torch.allclose(context_vecs[0], expected, atol=1e-4)
    assert torch.allclose(context_vecs[1], expected, atol=1e-4)


# --- 3.6.2 Implementing multi-head attention with weight splits -------------


def test_multihead_attention_outputs(batch):
    torch.manual_seed(123)
    mha = MultiHeadAttention(
        d_in=3, d_out=2, context_length=batch.shape[1], dropout=0.0, num_heads=2
    )
    context_vecs = mha(batch)

    expected = torch.tensor([
        [0.3190, 0.4858],
        [0.2943, 0.3897],
        [0.2856, 0.3593],
        [0.2693, 0.3873],
        [0.2639, 0.3928],
        [0.2575, 0.4028],
    ])
    assert context_vecs.shape == torch.Size([2, 6, 2])
    assert torch.allclose(context_vecs[0], expected, atol=1e-4)
    assert torch.allclose(context_vecs[1], expected, atol=1e-4)
