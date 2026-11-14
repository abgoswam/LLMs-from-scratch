import torch

from causal_attention import CausalAttention

# --- 3.5.3 Implementing a compact causal self-attention class ---------------


def test_causal_attention_outputs(batch):
    torch.manual_seed(123)
    ca = CausalAttention(d_in=3, d_out=2, context_length=batch.shape[1], dropout=0.0)
    context_vecs = ca(batch)

    expected = torch.tensor([
        [-0.4519, 0.2216],
        [-0.5874, 0.0058],
        [-0.6300, -0.0632],
        [-0.5675, -0.0843],
        [-0.5526, -0.0981],
        [-0.5299, -0.1081],
    ])
    assert context_vecs.shape == torch.Size([2, 6, 2])
    assert torch.allclose(context_vecs[0], expected, atol=1e-4)
    assert torch.allclose(context_vecs[1], expected, atol=1e-4)


def test_causal_attention_ignores_future_tokens(batch):
    torch.manual_seed(123)
    ca = CausalAttention(d_in=3, d_out=2, context_length=batch.shape[1], dropout=0.0)

    changed = batch.clone()
    changed[:, -1] = torch.tensor([9.0, -9.0, 9.0])  # replace the last token ("step")

    original_out = ca(batch)
    changed_out = ca(changed)
    assert torch.allclose(original_out[:, :-1], changed_out[:, :-1])
    assert not torch.allclose(original_out[:, -1], changed_out[:, -1])
