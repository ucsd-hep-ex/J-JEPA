import torch

from src.options import Options
from src.layers.embedding_stack import PlainAttentionEmbeddingStack

if __name__ == "__main__":
    print("Testing attention embedding stack layer")

    options = Options()
    options.display()

    N_part = 30
    N_part_ftr = 4

    options.particle_emb_dim: int = 64
    options.num_particle_embedding_layers: int = 10
    options.initial_particle_embedding_dim: int = 8
    options.initial_particle_embedding_skip_connections: bool = False
    options.particle_embedding_skip_connections: bool = True
    options.num_heads_in_subjet_embedding_blocks: int = 2
    options.num_particle_attention_blocks_in_embedding: int = 4
    options.num_class_attention_blocks_in_embedding: int = 2


    emb = PlainAttentionEmbeddingStack(options, N_part_ftr)
    print(emb)

    # set a pseudo input
    # of shape (bs, N_sj, N_part, N_part_ftr)
    bs = options.batch_size
    N_sj = options.num_subjets
    N_part = options.num_particles
    N_part_ftr = options.num_part_ftr

    has_nan = False
    for i in range(10):
        print(f'Testing trial {i+1}')
        x = torch.rand(bs, N_sj, N_part, N_part_ftr, dtype=torch.float)*6-3
        particle_masks = torch.zeros(bs, N_sj, N_part).bool()
        particle_masks[:, 0:5, 0:20] = True
        particle_masks[:, 5:10, 0:12] = True
        particle_masks[:, 10:15, 0:5] = True
        x[particle_masks==0] = 0.000
        result = emb(x, particle_masks)

        has_nan = has_nan or torch.isnan(result).any()

    print(x)
    print(particle_masks==0)
    print(result)
    print(result.shape)
    print("has_nan", has_nan)
