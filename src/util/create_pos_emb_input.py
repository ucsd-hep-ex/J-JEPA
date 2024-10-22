import torch


def create_pos_emb_input(x, stats, mask):
    """
    Create the input to the positional embedding layer
    Args:
        x: input tensor of shape (B, N, 4) [eta, phi, log_pt, log_energy]
        stats: dictionary of statistics for the input features
        mask: mask tensor of shape (B, N) -- real particle = 1, padded = 0
    Returns:
        pos_emb_input: input tensor for the positional embedding layer
                     last dimension: [pt, eta, phi, E]
    """
    pos_emb_input = torch.empty_like(x)
    pos_emb_input[:, :, 0] = (
        torch.exp(x[:, :, 2] * stats["part_pt_log"][1] + stats["part_pt_log"][0])
    ) * mask
    pos_emb_input[:, :, 1] = (
        x[:, :, 0] * stats["part_deta"][1] + stats["part_deta"][0]
    ) * mask
    pos_emb_input[:, :, 2] = (
        x[:, :, 1] * stats["part_dphi"][1] + stats["part_dphi"][0]
    ) * mask
    pos_emb_input[:, :, 3] = (
        torch.exp(x[:, :, 3] * stats["part_e_log"][1] + stats["part_e_log"][0])
    ) * mask
    return pos_emb_input
