import torch

from src.options import Options
from src.layers.attention_block import create_particle_attention_block

if __name__ == "__main__":
    print("Testing ParticleAttentionBlock")

    options = Options()
    options.particle_attention_block_type = "ParticleAttentionBlock"
    options.display()

    blk = create_particle_attention_block(options=options,
                                    input_dim=128,
                                    output_dim=128,
                                    n_heads=8
                                )

    # set a pseudo input
    # of shape (bs, N_sj, N_part, N_part_ftr)
    bs = 200
    N_sj = 20
    N_part = 30
    N_part_ftr = 128

    random_tensor = torch.rand(bs,
                                N_sj,
                                N_part,
                                N_part_ftr,
                                dtype=torch.float
                                )
    particle_masks = torch.randint(low=0, high=2, size=(bs, N_sj, N_part)).bool()
    result = blk(random_tensor, particle_masks)

    print(result.shape)
