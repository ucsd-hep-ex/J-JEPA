import torch
import torch.nn as nn

import sys
import os

sys.path.insert(0, "../")
from src.options import Options
from src.models.jjepa import JetsTransformerPredictor

if __name__ == "__main__":
    print("Testing JetTransformerPredictor")

    options = Options()
    options.predictor_embedding_layers_type = 'EmbeddingStack'
    options.pos_emb_type = "Learnable_Space"
    options.initial_embedding_dim = 128
    options.emb_dim = 128
    options.predictor_emb_dim = 64
    options.display()

    jetT = JetsTransformerPredictor(options)
    """
        Inputs:
            x: context subjet representations
                shape: [B, N_ctxt, emb_dim]
            subjet_mask: mask for zero-padded subjets
                shape: [B, N_ctxt]
            target_subjet_ftrs: target subjet features
                shape: [B, N_trgt, N_ftr]
            context_subjet_ftrs: context subjet features
                shape: [B, N_ctxt, N_ftr]
        Output:
            predicted target subjet representations
                shape: [B, N_trgt, predictor_output_dim]
    """

    # set a pseudo input
    # of shape (bs, N_sj, N_part, N_part_ftr)
    B = 100
    N_ctxt = 8
    N_trgt = 2
    N_ftr = 4  # [pt, eta, phi, E]

    x = torch.rand(B, N_ctxt, options.emb_dim, dtype=torch.float)
    subjet_mask = torch.randint(0, 2, (B, N_ctxt), dtype=torch.bool)
    target_subjet_ftrs = torch.rand(B, N_trgt, N_ftr, dtype=torch.float)
    context_subjet_ftrs = torch.rand(B, N_ctxt, N_ftr, dtype=torch.float)
    result = jetT(x, subjet_mask, target_subjet_ftrs, context_subjet_ftrs)
    print(result.shape)
