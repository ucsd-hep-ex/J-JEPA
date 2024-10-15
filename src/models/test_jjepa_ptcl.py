import sys
import torch
from src.models.ParT.ParticleTransformerEncoder import ParTEncoder, ParTPredictor
from src.models.jjepa import JJEPA
from src.options import Options


if __name__ == "__main__":
    options = Options()
    batch_size = 16
    N_ctxt = 50
    N_trgt = 30
    options.batch_size = batch_size
    options.emb_dim = 1024
    options.embed_dims[-1] = options.emb_dim
    # options.fc_params = [(128, 0), (256, 0), (512, 0), (options.emb_dim, 0)]
    options.fc_params = None
    options.use_parT = True
    options.display()

    x = v = torch.rand((batch_size, N_ctxt + N_trgt, 4))
    full_mask_shape = (batch_size, N_ctxt + N_trgt)
    full_particle_mask = torch.rand(full_mask_shape) > 0.5
    ctxt_split_mask = torch.cat(
        (torch.ones((batch_size, N_ctxt)), torch.zeros((batch_size, N_trgt))), dim=-1
    )
    trgt_split_mask = torch.cat(
        (torch.zeros((batch_size, N_ctxt)), torch.ones((batch_size, N_trgt))), dim=-1
    )

    jjepa_model = JJEPA(options)
    print("jjepa_model", jjepa_model)
    """
    context = {
        p4_spatial: torch.Tensor of shape [B, N_ctxt, 4],
        particle_mask: torch.Tensor of shape [B, N_ctxt],
        split_mask: torch.Tensor of shape [B, N_ctxt],
    }
    target = {
        p4_spatial: torch.Tensor of shape [B, N_trgt, 4],
        particle_mask: torch.Tensor of shape [B, N_trgt],
        split_mask: torch.Tensor of shape [B, N_trgt],
    }
    full_jet = {
        p4: torch.Tensor of shape [B, N, 4],
        p4_spatial: torch.Tensor of shape [B, N, 4],
        particle_mask: torch.Tensor of shape [B, N],
    }
    """
    context = {
        "p4_spatial": v[:, :N_ctxt, :],
        "particle_mask": full_particle_mask[:, :N_ctxt],
        "split_mask": ctxt_split_mask,
    }
    target = {
        "p4_spatial": v[:, N_ctxt:, :],
        "particle_mask": full_particle_mask[:, N_ctxt:],
        "split_mask": trgt_split_mask,
    }
    full_jet = {"p4": x, "p4_spatial": v, "particle_mask": full_particle_mask}
    pred_repr, target_repr, context_repr = jjepa_model(context, target, full_jet)
    print("pred_repr", pred_repr.shape)
    print("target_repr", target_repr.shape)
    print("context_repr", context_repr.shape)
    print("options.emb_dim", options.emb_dim)
