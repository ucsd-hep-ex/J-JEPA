from src.layers.attention_block.particle_attention_block import ParticleAttentionBlock
from src.layers.attention_block.class_attention_block import ClassAttentionBlock
from src.options import Options


def create_attention_block(
        options: Options,
        input_dim: int,
        output_dim: int,
        n_heads: int,
):
    if options.attention_embedding_block_type is None:
        raise Exception("You didn't set particle attention block type. Do you need a particle attention block?")

    linear_block_type = options.attention_embedding_block_type.lower().replace("_", "").replace(" ", "")

    if linear_block_type == "particleattentionblock":
        return ParticleAttentionBlock(options, input_dim, output_dim, n_heads)
    else:
        raise Exception("Sorry, not implemented yet")

def create_particle_attention_block(
        options: Options,
        input_dim: int,
        output_dim: int,
        n_heads: int,
):
    return ParticleAttentionBlock(options, input_dim, output_dim, n_heads)

def create_class_attention_block(
        options: Options,
        input_dim: int,
        output_dim: int,
        n_heads: int,
):
    return ClassAttentionBlock(options, input_dim, output_dim, n_heads)
