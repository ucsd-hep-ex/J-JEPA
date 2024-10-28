from src.options import Options
from src.util import create_pos_emb_fn

import torch

if __name__ == "__main__":
    options = Options()
    options.pos_emb_type = "Learnable_Space"
    emb_layer = create_pos_emb_fn(options, 256)

    B = 100
    N_ctxt = 8
    N_trgt = 2
    N_ftr = 4  # [pt, eta, phi, E]

    target_subjet_ftrs = torch.rand(B, N_trgt, N_ftr, dtype=torch.float)
    pos_emb = emb_layer(target_subjet_ftrs)

    print(pos_emb.shape)
