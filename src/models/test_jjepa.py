import torch
import torch.nn as nn

import sys
import os

sys.path.insert(0, "../")
from options import Options
from models.jjepa import JJEPA

if __name__ == "__main__":
    print("Testing full JJEPA model")

    options = Options()
    options.load("test_options.json")
    options.display()

    jjepa = JJEPA(options)

    # Generate random data for testing
    batch_size = 32
    num_subjets = 20
    num_particles = 30
    num_features = 4

    # Create random data for context and target
    context_subjets = torch.randn(batch_size, num_subjets, num_particles, num_features)
    context_particle_mask = torch.randint(0, 2, (batch_size, num_particles)).bool()
    context_subjet_mask = torch.randint(0, 2, (batch_size, num_subjets)).bool()
    context_split_mask = torch.randint(0, 2, (batch_size, num_subjets)).bool()

    target_subjets = torch.randn(batch_size, num_subjets, num_particles, num_features)
    target_particle_mask = torch.randint(0, 2, (batch_size, num_particles)).bool()
    target_subjet_mask = torch.randint(0, 2, (batch_size, num_subjets)).bool()
    target_split_mask = torch.randint(0, 2, (batch_size, num_subjets)).bool()

    # Create random data for full_jet
    full_jet_particles = torch.randn(batch_size, num_particles, num_features)
    full_jet_particle_mask = torch.randint(0, 2, (batch_size, num_particles)).bool()
    full_jet_subjet_mask = torch.randint(0, 2, (batch_size, num_subjets)).bool()

    # Create input dictionaries
    context = {
        "subjets": context_subjets,
        "particle_mask": context_particle_mask,
        "subjet_mask": context_subjet_mask,
        "split_mask": context_split_mask,
    }

    target = {
        "subjets": target_subjets,
        "particle_mask": target_particle_mask,
        "subjet_mask": target_subjet_mask,
        "split_mask": target_split_mask,
    }

    full_jet = {
        "particles": full_jet_particles,
        "particle_mask": full_jet_particle_mask,
        "subjet_mask": full_jet_subjet_mask,
    }

    # Run the model
    result = jjepa(context, target, full_jet)

    print("Input shapes:")
    print(f"Context subjets: {context_subjets.shape}")
    print(f"Target subjets: {target_subjets.shape}")
    print(f"Full jet particles: {full_jet_particles.shape}")
    print(f"Result shape: {result.shape}")