# Broadcasting practice

Not from the book. Built after hitting `[6, 2] / [6]` in `ch03/softmax_naive`.

Sources the examples follow:
- CS231n Python/NumPy tutorial, Broadcasting: https://cs231n.github.io/python-numpy-tutorial/#broadcasting
- NumPy, Broadcasting: https://numpy.org/doc/stable/user/basics.broadcasting.html
- PyTorch, Broadcasting semantics: https://docs.pytorch.org/docs/stable/notes/broadcasting.html

| File | Idea |
|---|---|
| `test_01_broadcast_rules.py` | The rules by hand: align from the right, pad with 1s, 1 stretches |
| `test_02_cs231n_examples.py` | Vector to rows, vector to columns (reshape to `(n, 1)`), outer product |
| `test_03_numpy_examples.py` | RGB channel scaling, vector quantization (single and batched) |
| `test_04_keepdim.py` | Reductions that must broadcast back: rows need `keepdim`, columns don't |
| `test_05_matmul.py` | `@`: matrix rule on the last two dims, broadcasting on the batch dims, 1-D special cases |

Not tested, worth knowing (PyTorch notes): an in-place op such as `x.add_(y)` can never change
`x`'s shape, so `x` of shape `(1, 3, 1)` plus `y` of shape `(3, 1, 7)` works out of place but
fails in place.

Run: `pytest trials/broadcasting -vv`
