"""Section A.7: compute_accuracy."""
import torch

def compute_accuracy(model, dataloader):
    """Fraction of correctly classified examples over the whole dataloader.

    Puts the model in eval mode and runs every batch without gradients.
    The prediction is the argmax of the logits over dim=1, compared with the labels.
    Returns correct / total as a Python float (e.g. 1.0), not a tensor.
    """
    # raise NotImplementedError

    model.eval()

    logit_batches = []
    label_batches = []

    for batch_idx, (features, labels) in enumerate(dataloader):
        print(f"batch: {batch_idx}, shapes: {features.shape}:{labels}")

        with torch.no_grad():
            logit_batch = model(features)
            logit_batches.append(logit_batch)
            label_batches.append(labels)


    logits = torch.cat(logit_batches)
    labels = torch.cat(label_batches)

    y_pred = torch.argmax(logits, dim=1)

    accuracy = torch.sum(y_pred == labels) / len(labels)

    print("done")
    return accuracy


def compute_accuracy_v2(model, dataloader):
    """Same result as compute_accuracy, but keeps running counts instead of storing all logits."""
    model.eval()
    correct = 0
    total_examples = 0

    with torch.no_grad():
        for features, labels in dataloader:
            logits = model(features)
            predictions = torch.argmax(logits, dim=1)
            correct += torch.sum(predictions == labels).item()
            total_examples += len(labels)

    return correct / total_examples
