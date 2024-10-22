from torch import nn

from src.layers.linear_block.activations import create_activation
from src.options import Options

class ParticleAttention(nn.Module):
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


    def forward(self, x, particle_masks):
        bs, SJ, P, C = x.shape

        # flatten x into specific shape for pytorch MHA
        # BMSJ: Batch_size Multiply n_Subjet
        # P: n_Particle in a subjet
        # C: n_channel
        x = x.flatten(start_dim=0, end_dim=1)
        particle_masks = particle_masks.flatten(start_dim=0, end_dim=1)
        BMSJ, P, C = x.shape

        x, _ = self.multihead_attn(x, x, x, key_padding_mask=particle_masks==0)
        x = self.proj_drop(x)

        # reshape x back to preserve the bs, n_sj dimensions
        x = x.reshape(bs, SJ, P, C)

        return x


class ParticleMLP(nn.Module):
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


class ParticleAttentionBlock(nn.Module):
    def __init__(self, options: Options, input_dim: int, output_dim: int, n_heads: int):
        super().__init__()

        norm_layer = nn.LayerNorm
        self.norm1 = norm_layer(input_dim)
        self.attn = ParticleAttention(options, input_dim, n_heads)
        self.drop_path = (
            nn.Identity() if options.drop_path <= 0 else nn.Dropout(options.drop_path)
        )
        self.norm2 = norm_layer(input_dim)
        self.mlp = ParticleMLP(options, input_dim, output_dim)

    def forward(self, x, particle_masks):
        y = self.attn(self.norm1(x), particle_masks)
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
