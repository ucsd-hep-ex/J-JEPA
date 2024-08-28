import torch

def get_1d_sincos_pos_emb(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega   # (D/2,)

    pos = pos.reshape(-1)   # (M,)
    out = torch.einsum('m,d->md', pos, omega)   # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def create_pos_emb_fn(emb_dim):
    """
    Input:
        emb_dim: Integer
    Return:
        a function that calculate the positional embeding by
        subjet eta and phi
    """
    def calc_pos_emb(subjet_ftrs):
        """
        Input:
            subjets_ftrs: torch tensor of shape (bs, N_subjets, N_sj_ftrs)
                last dimension: [pt, eta, phi, E]
        """
        sj_eta = subjet_ftrs[:, :, 1]
        sj_phi = subjet_ftrs[:, :, 2]

        # process phi to impose physical distance
        sj_phi_star = torch.sin(sj_phi/2)

        # shift eta to avoid negative positions
        sj_eta += 3

        # calculate embedding
        emb_eta = get_1d_sincos_pos_emb(emb_dim//2, sj_eta)
        emb_phi_star = get_1d_sincos_pos_emb(emb_dim//2, sj_phi_star)

        emb = torch.cat([emb_eta, emb_phi_star], axis=1)

        return emb

    return calc_pos_emb
