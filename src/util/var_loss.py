import torch
import torch.nn.functional as F


def variance_loss(x, flatten=True):
    """
    Computes the variance loss to maintain the variance of
    flattened subjet features across a batch.

    Args:
        x (torch.Tensor): Tensor of shape [batch_size, num_subjets, num_features].

    Returns:
        torch.Tensor: Scalar tensor representing the variance loss.
    """
    batch_size, num_subjets, num_features = x.size()
    if flatten:
        x = x.view(batch_size, -1)  # Shape: [batch_size, num_subjets * num_features]
    std_x = torch.sqrt(x.var(dim=1) + 0.0001)  # Shape: [num_subjets * num_features]
    std_loss = torch.mean(F.relu(1 - std_x))
    return std_loss
