import copy
import torch
import torch.nn as nn
from .PairEmbed import Embed, PairEmbed
from .Block import Block
from .utils import trunc_normal_


class ParTEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        pair_input_dim=4,
        embed_dims=[128, 512, 128],
        pair_embed_dims=[64, 64, 64],
        num_heads=8,
        num_layers=8,
        num_cls_layers=2,
        block_params=None,
        cls_block_params={"dropout": 0, "attn_dropout": 0, "activation_dropout": 0},
        fc_params=[],
        activation="gelu",
        use_amp=False,
        for_inference=False,
    ):
        super().__init__()
        self.use_amp = use_amp
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

        cfg_cls_block = copy.deepcopy(default_cfg)
        if cls_block_params is not None:
            cfg_cls_block.update(cls_block_params)

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
                for_onnx=for_inference,
            )
            if pair_embed_dims is not None and pair_input_dim > 0
            else None
        )
        self.blocks = nn.ModuleList([Block(**cfg_block) for _ in range(num_layers)])
        self.cls_blocks = (
            nn.ModuleList([Block(**cfg_cls_block) for _ in range(num_cls_layers)])
            if num_cls_layers > 0
            else None
        )
        self.norm = nn.LayerNorm(embed_dim)
        if fc_params is not None:
            fcs = []
            in_dim = embed_dim
            for out_dim, drop_rate in fc_params:
                fcs.append(
                    nn.Sequential(
                        nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(drop_rate)
                    )
                )
                in_dim = out_dim
            self.fc = nn.Sequential(*fcs)
        else:
            self.fc = None
        # init class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        trunc_normal_(self.cls_token, std=0.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            "cls_token",
        }

    def forward(self, x, v=None, mask=None, uu=None):
        # x: (N, C, P)
        # v: (N, 4, P) [px,py,pz,energy]
        # mask: (N, 1, P) -- real particle = 1, padded = 0
        padding_mask = ~mask.squeeze(1)  # (N, P)
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            # input embedding
            x = self.embed(x).masked_fill(~mask.permute(2, 0, 1), 0)  # (P, N, C)
            attn_mask = None
            if (v is not None or uu is not None) and self.pair_embed is not None:
                attn_mask = self.pair_embed(v, uu).view(
                    -1, v.size(-1), v.size(-1)
                )  # (N*num_heads, P, P)

            for block in self.blocks:
                x = block(x, x_cls=None, padding_mask=padding_mask, attn_mask=attn_mask)

            if self.cls_blocks is not None:
                # extract class token
                cls_tokens = self.cls_token.expand(1, x.size(1), -1)  # (1, N, C)
                for block in self.cls_blocks:
                    cls_tokens = block(x, x_cls=cls_tokens, padding_mask=padding_mask)
                x_cls = self.norm(cls_tokens).squeeze(0)
            else:
                x = x.sum(dim=0)
                x_cls = self.norm(x)  # (batch_size, embed_dim)
            if self.fc is None:
                return x_cls
            output = self.fc(x_cls)
            return output


"""
Usage of the ParTEncoder
encoder = ParTEncoder(input_dim=4)
batch_size = 16
N_ptcls = 50
x = v = torch.rand((batch_size, 4, N_ptcls))

mask_shape = (batch_size, 1, N_ptcls)
mask = (torch.rand(mask_shape) > 0.5)
encoded_features = encoder(x, v, mask)
"""
