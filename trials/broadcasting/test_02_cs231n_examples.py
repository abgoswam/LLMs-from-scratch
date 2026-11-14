import torch

from broadcasting import add_vector_to_columns, add_vector_to_rows, outer_product

# --- CS231n Python/NumPy tutorial: Broadcasting ------------------------------


def test_add_vector_to_rows():
    x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    v = torch.tensor([1, 0, 1])
    expected = torch.tensor([[2, 2, 4], [5, 5, 7], [8, 8, 10], [11, 11, 13]])
    assert torch.equal(add_vector_to_rows(x, v), expected)


def test_add_vector_to_columns():
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    w = torch.tensor([4, 5])
    expected = torch.tensor([[5, 6, 7], [9, 10, 11]])
    assert torch.equal(add_vector_to_columns(x, w), expected)


def test_outer_product():
    v = torch.tensor([1, 2, 3])
    w = torch.tensor([4, 5])
    expected = torch.tensor([[4, 5], [8, 10], [12, 15]])
    assert torch.equal(outer_product(v, w), expected)
