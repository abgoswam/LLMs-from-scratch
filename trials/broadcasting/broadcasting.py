"""Broadcasting practice (not from the book).

Examples follow the CS231n NumPy tutorial and the NumPy / PyTorch broadcasting docs (see README.md).
Every tensor function must use broadcasting: no Python loops, no torch.tile / repeat / expand.
"""
import torch


def broadcast_shape(shape_a, shape_b):
    """Apply the broadcasting rules by hand, in plain Python (no torch, no numpy).

    shape_a and shape_b are tuples of ints. Line the shapes up from the right; the shorter one
    is treated as if padded with 1s on the left. Each pair of sizes must be equal or contain a 1;
    the result takes the larger size. Returns the result shape as a tuple.
    Raises ValueError if any pair is incompatible.
    """
    raise NotImplementedError


def add_vector_to_rows(x, v):
    """CS231n: add v, shape (num_cols,), to every row of x, shape (num_rows, num_cols)."""
    raise NotImplementedError


def add_vector_to_columns(x, w):
    """CS231n: add w, shape (num_rows,), to every column of x, shape (num_rows, num_cols).

    w[i] is added to every entry of row i. Returns shape (num_rows, num_cols).
    """
    raise NotImplementedError


def outer_product(v, w):
    """CS231n: outer product of v, shape (n,), and w, shape (m,).

    Returns shape (n, m) with entry [i, j] equal to v[i] * w[j]. Use broadcasting, not torch.outer.
    """
    raise NotImplementedError


def scale_channels(image, scale):
    """NumPy docs: multiply each color channel of an image by its own factor.

    image has shape (height, width, 3); scale has shape (3,). Returns shape (height, width, 3).
    """
    raise NotImplementedError


def closest_code(observation, codes):
    """NumPy docs (vector quantization): index of the code closest to one observation.

    observation has shape (d,), codes has shape (num_codes, d). Distance is Euclidean.
    Returns a Python int.
    """
    raise NotImplementedError


def closest_codes(observations, codes):
    """Batched closest_code: for each observation, the index of its closest code.

    observations has shape (num_obs, d), codes has shape (num_codes, d). Compute all
    differences in one broadcast (insert a dim so the two inputs line up as
    (num_obs, 1, d) and (num_codes, d)), then the Euclidean distance over the last dim.
    Returns a tensor of shape (num_obs,) with the index of the closest code per observation.
    """
    raise NotImplementedError


def normalize_rows(x):
    """Divide each row of x, shape (num_rows, num_cols), by that row's sum, so every row sums to 1."""
    raise NotImplementedError


def normalize_columns(x):
    """Divide each column of x, shape (num_rows, num_cols), by that column's sum, so every column sums to 1."""
    raise NotImplementedError


def matmul_shape(shape_a, shape_b):
    """Result shape of a @ b, worked out by hand in plain Python (no torch, no numpy).

    The last two dims follow the matrix rule: a's last dim must equal b's second-to-last dim,
    and the result has a's second-to-last and b's last. All dims before those (batch dims)
    follow the broadcasting rules, so you may reuse broadcast_shape.
    A 1-D a is treated as a row (1, k) and a 1-D b as a column (k, 1); the added dim is
    removed from the result again (so two 1-D inputs give the empty shape ()).
    Returns a tuple; raises ValueError if the inner dims or the batch dims do not match.
    """
    raise NotImplementedError


def attention_scores(queries, keys):
    """Dot product of every query with every key, as in ch03 (queries times keys transposed).

    queries has shape (..., num_queries, d), keys has shape (..., num_keys, d), with the same
    leading dims (e.g. batch, or batch and heads). Returns (..., num_queries, num_keys) where entry
    [..., i, j] is query i dotted with key j. One @, no loops; swap only the last two dims of keys,
    so it works for any number of leading dims.
    """
    raise NotImplementedError


def apply_linear(x, weight):
    """What nn.Linear without bias does: x of shape (..., d_in), weight of shape (d_out, d_in).

    Returns (..., d_out). One @; the 2-D weight broadcasts over all leading dims of x.
    """
    raise NotImplementedError
