import torch
import torch.nn as nn

import sys
import os

sys.path.insert(0, "../")
from src.options import Options
from src.models.jjepa import JJEPA

if __name__ == "__main__":
    print("Testing full JJEPA model")

    options = Options()
    options.load("src/test_options.json")
    options.display()

    jjepa = JJEPA(options)

    # set a pseudo input
    """
    context = {
        subjets: torch.Tensor,
        particle_mask: torch.Tensor,
        subjet_mask: torch.Tensor,
        split_mask: torch.Tensor,
    }
    target = {
        subjets: torch.Tensor,
        particle_mask: torch.Tensor,
        subjet_mask: torch.Tensor,
        split_mask: torch.Tensor,
    }
    full_jet = {
        particles: torch.Tensor,
        particle_mask: torch.Tensor,
        subjet_mask: torch.Tensor,
    }
    """

    result = jjepa(context, target, full_jet)

    print(result.shape)
