import torch


def get_1d_sincos_pos_emb(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (B, S)
    out: (B, S, D)
    """
    assert embed_dim % 2 == 0

    omega = torch.arange(embed_dim // 2, device=pos.device, dtype=torch.float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    out = torch.einsum("bs,d->bsd", pos, omega)  # (B, S, D/2), outer product

    emb_sin = torch.sin(out)  # (B, S, D/2)
    emb_cos = torch.cos(out)  # (B, S, D/2)

    # Interleave sine and cosine embeddings
    emb = torch.stack((emb_sin, emb_cos), dim=-1)  # (B, S, D/2, 2)
    emb = emb.view(emb.shape[0], emb.shape[1], -1)  # (B, S, D)

    return emb


def create_pt_pos_emb_fn(emb_dim):
    """
    Input:
        emb_dim: Integer
    Return:
        a function that calculates the positional embedding based on
        the pT ranks of subjets.
    """

    def calc_pos_emb(subjet_ftrs):
        """
        Input:
            subjet_ftrs: torch tensor of shape (bs, N_subjets, N_sj_ftrs)
                last dimension: [pt, eta, phi, E]
        Output:
            ranks: torch tensor of shape (bs, N_subjets)
        """
        bs, N_subjets, _ = subjet_ftrs.shape
        device = subjet_ftrs.device
        dtype = torch.float

        # Since subjets are already sorted by pT from high to low,
        # we can assign ranks directly.
        ranks = (
            torch.arange(N_subjets, device=device, dtype=dtype)
            .unsqueeze(0)
            .expand(bs, -1)
        )

        # Generate positional embeddings using the ranks
        emb = get_1d_sincos_pos_emb(emb_dim, ranks)  # (bs, N_subjets, emb_dim)

        return emb

    return calc_pos_emb
