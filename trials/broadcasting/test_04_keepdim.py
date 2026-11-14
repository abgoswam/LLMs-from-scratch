import torch

from broadcasting import normalize_columns, normalize_rows

# --- keepdim: reductions that must broadcast back ----------------------------


def test_normalize_rows():
    x = torch.tensor([[1.0, 3.0], [2.0, 2.0], [0.0, 4.0]])
    expected = torch.tensor([[0.25, 0.75], [0.5, 0.5], [0.0, 1.0]])
    assert torch.allclose(normalize_rows(x), expected)


def test_normalize_columns():
    x = torch.tensor([[1.0, 3.0], [2.0, 2.0], [0.0, 4.0]])
    expected = torch.tensor([[1 / 3, 3 / 9], [2 / 3, 2 / 9], [0.0, 4 / 9]])
    assert torch.allclose(normalize_columns(x), expected)
