import torch
import torch.nn as nn

import sys
import os

sys.path.insert(0, "../")
from src.options import Options
from src.models.jjepa import JetsTransformer

if __name__ == "__main__":
    print("Testing JetsTransformer with AttentionEmbeddingStack")

    options = Options()

    options.embedding_layers_type = "PlainAttentionEmbeddingStack"
    options.attention_embedding_block_type = "ParticleAttentionBlock"

    options.display()

    jetT = JetsTransformer(options)
    print(jetT)

    # Generate random data for testing
    batch_size = 32
    num_subjets = 20
    num_ctxt_subjets = 15
    num_trgt_subjets = num_subjets - num_ctxt_subjets
    num_particles = 30
    num_features = 4
    num_subjet_features = 5

    for i in range(5):
        print(f"Test {i+1}")
        # Create random data for context and target
        context_subjets = torch.randn(batch_size, num_ctxt_subjets, num_subjet_features)
        context_particle_mask = torch.randint(0, 2, (batch_size, num_particles)).bool()
        context_subjet_mask = torch.randint(0, 2, (batch_size, num_subjets)).bool()
        context_split_mask = torch.cat(
            (
                torch.zeros(batch_size, num_subjets - num_ctxt_subjets),
                torch.ones(batch_size, num_ctxt_subjets),
            ),
            dim=1,
        ).bool()  # Random boolean mask for example

        # Create random data for full_jet
        full_jet_particles = torch.randn(
            batch_size, num_subjets, num_particles * num_features
        )
        full_jet_subjets = torch.randn(batch_size, num_subjets, num_subjet_features)
        full_jet_particle_mask = torch.randint(0, 2, (batch_size, num_subjets, num_particles)).bool()
        full_jet_subjet_mask = torch.randint(0, 2, (batch_size, num_subjets)).bool()

        # Create input dictionaries
        context = {
            "subjets": context_subjets,
            "particle_mask": context_particle_mask,
            "subjet_mask": context_subjet_mask,
            "split_mask": context_split_mask,
        }

        full_jet = {
            "particles": full_jet_particles,
            "subjets": full_jet_subjets,
            "particle_mask": full_jet_particle_mask,
            "subjet_mask": full_jet_subjet_mask,
        }

        # Run the model
        jetT(full_jet, full_jet["subjet_mask"], full_jet["subjets"], context["split_mask"], full_jet["particle_mask"])
