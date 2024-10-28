from src.util.positional_embedding import (
    create_space_pos_emb_fn,
    create_phase_space_pos_emb_fn,
    create_learnable_space_emb_fn,
)

from src.options import Options

def create_pos_emb_fn(options: Options, emb_dim: int):
    pos_emb_type = options.pos_emb_type.lower().replace("_", "").replace(" ", "")
    if pos_emb_type == 'space':
        return create_space_pos_emb_fn(emb_dim)
    elif pos_emb_type == "phasespace":
        return create_phase_space_pos_emb_fn(emb_dim)
    elif pos_emb_type == "learnablespace":
        return create_learnable_space_emb_fn(emb_dim)
