"""Section A.7: compute_accuracy."""


def compute_accuracy(model, dataloader):
    """Fraction of correctly classified examples over the whole dataloader.

    Puts the model in eval mode and runs every batch without gradients.
    The prediction is the argmax of the logits over dim=1, compared with the labels.
    Returns correct / total as a Python float (e.g. 1.0), not a tensor.
    """
    raise NotImplementedError
