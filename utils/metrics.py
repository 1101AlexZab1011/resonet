import torch

def accuracy(predictions, targets):
    """
    Computes the accuracy metric for multiclass tasks with one-hot encoded labels.

    Args:
    - predictions: A PyTorch tensor of shape (batch_size, num_classes) representing the model's predictions.
    - targets: A PyTorch tensor of shape (batch_size, num_classes) representing the true labels in one-hot encoded format.

    Returns:
    - accuracy: A float value representing the accuracy metric.
    """

    # Convert one-hot encoded labels to class indices
    targets = torch.argmax(targets, dim=1)

    # Compute predicted class indices
    _, predicted = torch.max(predictions, dim=1)

    # Compute accuracy
    correct = torch.sum(predicted == targets)
    total = targets.shape[0]
    accuracy = correct / total

    return accuracy