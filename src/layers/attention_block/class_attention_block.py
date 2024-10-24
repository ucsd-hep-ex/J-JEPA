import torch
from torch import nn

from src.layers.linear_block.activations import create_activation
from src.options import Options

class ClassAttention(nn.Module):
    def __init__(
        self,
        options: Options,
        dim: int,
        n_heads: int,
    ):
        super().__init__()

        assert dim % n_heads == 0, "The particle attention dimension must be divisible by the number of heads."

        self.options = options
        self.dim = dim
        self.num_heads = n_heads

        self.proj_drop = nn.Dropout(options.proj_drop)

        self.multihead_attn = nn.MultiheadAttention(
            self.dim, self.num_heads, batch_first=True
        )


    def forward(self, u, x_cls, seq_masks):
        bs, SJ, Seq, C = u.shape

        # flatten x into specific shape for pytorch MHA
        # BMSJ: Batch_size Multiply n_Subjet
        # Seq: length of sequence = N_particle + 1
        # C: n_channel
        x_cls = x_cls.flatten(start_dim=0, end_dim=1)
        u = u.flatten(start_dim=0, end_dim=1)
        seq_masks = seq_masks.flatten(start_dim=0, end_dim=1)
        BMSJ, P, C = u.shape


        u, _ = self.multihead_attn(x_cls, u, u, key_padding_mask=seq_masks==0)  # (1, batch, embed_dim)

        # reshape x back to preserve the bs, n_sj dimensions
        u = u.reshape(bs, SJ, 1, C)

        return u


class ClassMLP(nn.Module):
    def __init__(self, options: Options, input_dim: int, output_dim: int):
        super().__init__()
        self.options = options
        hidden_features = input_dim * 4
        self.fc1 = nn.Linear(input_dim, hidden_features)
        self.act = create_activation(options.activation, None)
        self.fc2 = nn.Linear(hidden_features, output_dim)
        self.drop = nn.Dropout(options.drop_mlp)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class ClassAttentionBlock(nn.Module):
    def __init__(self, options: Options, input_dim: int, output_dim: int, n_heads: int):
        super().__init__()

        norm_layer = nn.LayerNorm
        self.norm1 = norm_layer(input_dim)
        self.attn = ClassAttention(options, input_dim, n_heads)
        self.drop_path = (
            nn.Identity() if options.drop_path <= 0 else nn.Dropout(options.drop_path)
        )
        self.norm2 = norm_layer(input_dim)
        self.mlp = ClassMLP(options, input_dim, output_dim)

    def forward(self, x, x_cls, particle_masks):
        with torch.no_grad():
            # prepend one element for x_cls: -> (batch, subjet, 1+part_len)
            particle_masks = torch.cat((torch.ones_like(particle_masks[:, :, :1]), particle_masks), dim=2)

        u = torch.cat((x_cls, x), dim=2)  # (batch, subjet, 1+part_len, embed_dim)

        y = self.attn(self.norm1(u), x_cls, particle_masks)
        x = x_cls + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
