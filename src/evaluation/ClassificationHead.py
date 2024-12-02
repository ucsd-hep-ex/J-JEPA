from src.models.ParT.Block import Block
import torch
import torch.nn as nn
import copy
import math
from src.models.ParT.utils import trunc_normal_


class ClassificationHead(nn.Module):
    def __init__(self, embed_dim, num_heads=8, num_cls_layers=2):
        super(ClassificationHead, self).__init__()

        cfg_cls_block = dict(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ffn_ratio=4,
            dropout=0.1,
            attn_dropout=0.1,
            activation_dropout=0.1,
            add_bias_kv=False,
            activation="gelu",
            scale_fc=True,
            scale_attn=True,
            scale_heads=True,
            scale_resids=True,
        )

        self.cls_blocks = nn.ModuleList(
            [Block(**cfg_cls_block) for _ in range(num_cls_layers)]
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        trunc_normal_(self.cls_token, std=0.02)
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, 2)

    def forward(self, x, padding_mask=None):
        """
        x: Tensor of shape (seq_len, batch_size, embed_dim)
        padding_mask: Optional Tensor of shape (batch_size, seq_len)
        output: Tensor of shape (batch_size, 2) for binary classification
        """
        # Expand cls_token to match the batch size
        cls_tokens = self.cls_token.expand(
            1, x.size(1), -1
        )  # (1, batch_size, embed_dim)

        for block in self.cls_blocks:
            cls_tokens = block(x, x_cls=cls_tokens, padding_mask=padding_mask)

        x_cls = self.norm(cls_tokens).squeeze(0)  # Remove sequence dimension
        out = self.proj(x_cls)  # Shape: (batch_size, num_classes)
        return out


# Example usage:
"""
embed_dim = 128
batch_size = 32
num_subjets = 20
x = torch.randn((batch_size, num_subjets, embed_dim)).transpose(0,1)
padding_mask = torch.randn((batch_size, num_subjets))
classification_head = ClassificationHead(embed_dim=embed_dim)
output = classification_head(x, padding_mask=padding_mask)
"""
