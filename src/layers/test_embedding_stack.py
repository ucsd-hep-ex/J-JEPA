import torch

from src.options import Options
from src.layers.embedding_stack import EmbeddingStack

if __name__ == "__main__":
    print("Testing embedding stack layer")

    options = Options()
    options.load("test_embedding_stack.json")
    options.display()

    N_part = 30
    N_part_ftr = 4
    emb = EmbeddingStack(options, N_part * N_part_ftr)

    # set a pseudo input
    # of shape (bs, N_sj, N_part, N_part_ftr)
    bs = options.batch_size
    N_sj = options.num_subjets
    N_part = options.num_particles
    N_part_ftr = options.num_part_ftr

    random_tensor = torch.rand(100, 20, 30, 4, dtype=torch.float)
    x = random_tensor.flatten(2)
    result = emb(x)

    print(result.shape)


