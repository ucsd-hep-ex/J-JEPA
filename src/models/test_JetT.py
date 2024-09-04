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

    jetT = JetsTransformer(options)

    # set a pseudo input
    # of shape (bs, N_sj, N_part, N_part_ftr)
    subjets = torch.rand(100, 20, 30, 4, dtype=torch.float)
    subjets_meta = torch.rand(100, 20, 5, dtype=torch.float)
    split_mask = torch.cat(
        (torch.zeros(100, 20 - 9), torch.ones(100, 9)), dim=1
    )  # Random boolean mask for example
    split_mask = split_mask.bool()
    result = jetT(subjets, subjets_meta, split_mask)

    print(result.shape)
