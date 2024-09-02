import torch
import torch.nn as nn

import sys
import os

sys.path.insert(0, "../")
from src.options import Options
from src.models.jjepa import JetsTransformer

if __name__ == "__main__":
    print("Testing embedding stack layer")

    options = Options()
    options.load("src/test_options.json")
    options.display()

    jetT = JetsTransformer(
        options,
        depth=12,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
    )

    # set a pseudo input
    # of shape (bs, N_sj, N_part, N_part_ftr)
    subjets = torch.rand(100, 20, 30, 4, dtype=torch.float)
    subjets_meta = torch.rand(100, 20, 5, dtype=torch.float)
    result = jetT(subjets, subjets_meta)

    print(result.shape)
