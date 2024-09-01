import torch
import torch.nn as nn
from .PairEmbed import Embed, PairEmbed
from .Block import Block

class ParticleTransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        embed_dims=[128, 512, 128],
        pair_input_dim=4,
        pair_embed_dims=[64, 64, 64],
        num_heads=8,
        num_layers=8,
        block_params=None,
        activation="gelu",
        **kwargs
    ):
        super().__init__(**kwargs)
        embed_dim = embed_dims[-1] if len(embed_dims) > 0 else input_dim
        default_cfg = dict(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ffn_ratio=4,
            dropout=0.1,
            attn_dropout=0.1,
            activation_dropout=0.1,
            add_bias_kv=False,
            activation=activation,
            scale_fc=True,
            scale_attn=True,
            scale_heads=True,
            scale_resids=True,
        )
        cfg_block = default_cfg.copy()
        if block_params is not None:
            cfg_block.update(block_params)

        self.embed = (
            Embed(input_dim, embed_dims, activation=activation)
            if len(embed_dims) > 0
            else nn.Identity()
        )
        self.pair_embed = (
            PairEmbed(
                pair_input_dim,
                0,
                pair_embed_dims + [cfg_block["num_heads"]],
                remove_self_pair=True,
                use_pre_activation_pair=True,
                for_onnx=False,
            )
            if pair_embed_dims is not None and pair_input_dim > 0
            else None
        )
        self.blocks = nn.ModuleList([Block(**cfg_block) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, v=None, mask=None, uu=None):
        padding_mask = ~mask.squeeze(1)  # (N, P)
        x = self.embed(x).masked_fill(~mask.permute(2, 0, 1), 0)  # (P, N, C)
        attn_mask = None
        if (v is not None or uu is not None) and self.pair_embed is not None:
            attn_mask = self.pair_embed(v, uu).view(
                -1, v.size(-1), v.size(-1)
            )  # (N*num_heads, P, P)

        for block in self.blocks:
            x = block(x, padding_mask=padding_mask, attn_mask=attn_mask)

        x = x.sum(dim=0)
        x = self.norm(x)  # (batch_size, embed_dim)
        return x


"""
Usage of the ParticleTransformerEncoder
encoder = ParticleTransformerEncoder(input_dim=4)
batch_size = 16
N_ptcls = 50
x = v = torch.rand((batch_size, 4, N_ptcls))

mask_shape = (batch_size, 1, N_ptcls)
mask = (torch.rand(mask_shape) > 0.5)
encoded_features = encoder(x, v, mask)
"""
