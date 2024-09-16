import torch


def covariance_loss(x):
    """
    Computes the covariance loss to reduce the covariance between different features.

    Args:
        x (torch.Tensor): Tensor of shape [batch_size, num_subjets, num_features].

    Returns:
        torch.Tensor: Scalar tensor representing the covariance loss.
    """
    batch_size, num_subjets, num_features = x.size()

    # Center the representations over the num_subjets dimension
    x_centered = x - x.mean(
        dim=1, keepdim=True
    )  # Shape: [batch_size, num_subjets, num_features]

    # Compute covariance matrices for each sample in the batch
    # Covariance is computed over the num_subjets dimension
    cov_x = torch.matmul(
        x_centered.transpose(1, 2),  # Shape: [batch_size, num_features, num_subjets]
        x_centered,  # Shape: [batch_size, num_subjets, num_features]
    )  # Shape: [batch_size, num_features, num_features]

    # Compute the covariance loss
    # Sum the squared off-diagonal elements of the covariance matrices
    def off_diagonal_loss(cov):
        # Sum of squares of all elements in the covariance matrices
        cov_frobenius_squared = cov.pow(2).sum(dim=(1, 2))  # Shape: [batch_size]

        # Sum of squares of the diagonal elements
        cov_diag_squared = (
            torch.diagonal(cov, dim1=1, dim2=2).pow(2).sum(dim=1)
        )  # Shape: [batch_size]

        # Sum of squares of off-diagonal elements
        off_diag_sum = cov_frobenius_squared - cov_diag_squared  # Shape: [batch_size]

        # Normalize by the number of features
        off_diag_loss = off_diag_sum / num_features  # Shape: [batch_size]

        # Average over the batch
        return off_diag_loss.mean()

    # Compute the total covariance loss
    cov_loss = off_diagonal_loss(cov_x)

    return cov_loss
