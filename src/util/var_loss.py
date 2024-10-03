import torch
import torch.nn.functional as F


def variance_loss(masked_reps, mask):
    """
    Computes the variance loss to maintain the variance of
    flattened subjet features across a batch.

    Args:
        masked_reps (torch.Tensor): Tensor of shape [batch_size, num_subjets, num_features].
        mask (torch.Tensor): Tensor of shape [batch_size, num_subjets] with 1s for valid， 0s for invalid.

    Returns:
        torch.Tensor: Scalar tensor representing the variance loss.
    """
    # Convert mask to float for arithmetic operations
    mask = mask.float()

    # Small epsilon to prevent division by zero
    epsilon = 1e-8

    # Expand the mask to match the dimensions of masked_reps
    mask_expanded = mask.unsqueeze(-1)  # Shape: [batch_size, seq_len, 1]

    # Compute the number of valid entries per batch item
    valid_counts = mask.sum(dim=1).unsqueeze(-1) + epsilon  # Shape: [batch_size, 1]

    # Compute the masked mean per batch item and hidden unit
    mean_vals = (masked_reps * mask_expanded).sum(
        dim=1
    ) / valid_counts  # Shape: [batch_size, hidden_size]

    # Compute the squared differences from the mean
    sq_diffs = (
        (masked_reps - mean_vals.unsqueeze(1)) ** 2
    ) * mask_expanded  # Shape: [batch_size, seq_len, hidden_size]

    # Compute the masked variance per batch item and hidden unit
    variance = sq_diffs.sum(dim=1) / valid_counts  # Shape: [batch_size, hidden_size]

    # Compute the mean variance per batch item across hidden units
    mean_variance_per_batch = variance.mean(dim=1)  # Shape: [batch_size]

    # Compute the final mean across the batch
    final_mean_variance = mean_variance_per_batch.mean()

    std_x = torch.sqrt(final_mean_variance + 0.0001)
    std_loss = torch.mean(F.relu(1 - std_x))
    return std_loss
