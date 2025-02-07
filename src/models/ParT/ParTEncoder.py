import copy
import torch
import torch.nn as nn

from src.layers.linear_block.activations import create_activation
from src.util import create_pos_emb_fn
from src.options import Options
from src.util.tensors import trunc_normal_
from src.util.create_pos_emb_input import create_pos_emb_input

from src.models.ParT.PairEmbed import PairEmbed
from src.util.positional_embedding import create_space_pos_emb_fn

# A dictionary for normalization layers
NORM_LAYERS = {
    "None": None,
    "BatchNorm": nn.BatchNorm1d,
    "LayerNorm": nn.LayerNorm,
    "MaskedBatchNorm": None,
}

# A dictionary for activation functions
ACTIVATION_LAYERS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "LeakyRelU": nn.LeakyReLU,
    "SELU": nn.SELU,
}


class Attention(nn.Module):
    def __init__(
        self,
        options: Options,
    ):
        super().__init__()
        self.options = options
        if options.debug:
            print("Initializing Attention module")
        self.num_heads = options.num_heads
        self.dim = options.attn_dim
        self.head_dim = self.dim // options.num_heads
        self.scale = options.qk_scale or self.head_dim**-0.5
        self.W_qkv = nn.Linear(self.dim, self.dim * 3, bias=options.qkv_bias)
        self.attn_drop = nn.Dropout(options.attn_drop)
        self.proj = nn.Linear(self.dim, self.dim)
        self.activation = create_activation(options.activation, self.dim)
        self.proj_drop = nn.Dropout(options.proj_drop)

        self.multihead_attn = nn.MultiheadAttention(
            self.dim, self.num_heads, batch_first=True
        )

    def forward(self, x, key_padding_mask=None, attn_mask=None):
        if self.options.debug:
            print(f"Attention forward pass with input shape: {x.shape}")
        B, N, C = x.shape
        assert C % self.num_heads == 0
        if self.options.debug:
            print("num_heads: ", self.num_heads)
        # qkv = (
        #     self.W_qkv(x)
        #     .reshape(B, N, 3, self.num_heads, C // self.num_heads)
        #     .permute(2, 0, 3, 1, 4)
        # )
        qkv = self.W_qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # attn = (q @ k.transpose(-2, -1)) * self.scale
        # attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)
        # x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x, _ = self.multihead_attn(
            q, k, v, key_padding_mask=key_padding_mask, attn_mask=attn_mask
        )

        x = self.proj(x)
        x = self.activation(x)
        x = self.proj_drop(x)
        if self.options.debug:
            print(f"Attention output shape: {x.shape}")

        return x


class MLP(nn.Module):
    def __init__(self, options: Options):
        super().__init__()
        self.options = options
        if self.options.debug:
            print("Initializing MLP module")
        act_layer = ACTIVATION_LAYERS.get(options.activation, nn.GELU)
        self.pre_fc_norm = nn.LayerNorm(options.in_features)
        self.fc1 = nn.Linear(options.in_features, options.in_features * 4)
        self.act = act_layer()
        self.norm = nn.LayerNorm(options.in_features * 4)
        self.fc2 = nn.Linear(options.in_features * 4, options.in_features)
        self.post_fc_norm = nn.LayerNorm(options.in_features)
        self.drop = nn.Dropout(options.drop_mlp)

    def forward(self, x):
        if self.options.debug:
            print(f"MLP forward pass with input shape: {x.shape}")
        x = self.pre_fc_norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.post_fc_norm(x)
        if self.options.debug:
            print(f"MLP output shape: {x.shape}")
        return x


class Block(nn.Module):
    def __init__(self, options: Options):
        super().__init__()
        self.options = options
        if self.options.debug:
            print("Initializing Block module")
        act_layer = ACTIVATION_LAYERS.get(options.activation, nn.GELU)
        norm_layer = NORM_LAYERS.get(options.normalization, nn.LayerNorm)
        self.norm1 = norm_layer(options.repr_dim)
        self.dim = options.attn_dim
        self.attn = Attention(options)

        self.drop_path = (
            nn.Identity() if options.drop_path <= 0 else nn.Dropout(options.drop_path)
        )
        self.norm2 = norm_layer(self.dim)
        mlp_hidden_dim = int(self.dim * options.mlp_ratio)
        options.hidden_features = mlp_hidden_dim
        self.mlp = MLP(options)

    def forward(self, x, padding_mask=None, attn_mask=None):
        if self.options.debug:
            print(f"Block forward pass with input shape: {x.shape}")
        if attn_mask is not None:
            expected_shape = (x.size(0) * self.options.num_heads, x.size(1), x.size(1))
            if attn_mask.shape != expected_shape:
                raise ValueError(
                    f"Attention mask shape {attn_mask.shape} does not match expected shape {expected_shape}"
                )
        y = self.attn(self.norm1(x), key_padding_mask=padding_mask, attn_mask=attn_mask)
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if self.options.debug:
            print(f"Block output shape: {x.shape}")
        return x


class Embed(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=1024):
        super().__init__()
        self.input_bn = nn.LayerNorm(input_dim)
        self.embed = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, output_dim),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.input_bn(x)
        return self.embed(x)


class ParTEncoder(nn.Module):
    def __init__(self, options: Options):
        super().__init__()
        self.options = options
        if self.options.debug:
            print("Initializing JetsTransformer module")
        norm_layer = NORM_LAYERS.get(options.normalization, nn.LayerNorm)
        self.num_part_ftr = options.num_part_ftr
        self.embed_dim = options.emb_dim
        self.calc_pos_emb = create_pos_emb_fn(options, options.emb_dim)

        self.pair_embed = (
            PairEmbed(
                self.options.pair_input_dim,
                0,
                self.options.pair_embed_dims + [options.num_heads],
                remove_self_pair=True,
                use_pre_activation_pair=True,
                for_onnx=False,
            )
            if self.options.pair_embed_dims is not None
            and self.options.pair_input_dim > 0
            else None
        )

        print("num_particles", options.num_particles)
        print("num_part_ftr", options.num_part_ftr)

        self.particle_emb = Embed(
            input_dim=options.num_part_ftr, hidden_dim=128, output_dim=options.emb_dim
        )

        options.repr_dim = options.emb_dim
        options.attn_dim = options.repr_dim
        options.in_features = options.repr_dim
        options.out_features = options.repr_dim
        self.blocks = nn.ModuleList(
            [Block(options=options) for _ in range(options.encoder_depth)]
        )
        self.norm = norm_layer(options.emb_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, v, particle_masks, split_mask=None, stats=None):
        # x: (B, P, 4) [eta, phi, log_pt, log_energy]
        # v: (B, P, 4) [px,py,pz,energy]
        # particle_masks: (B, P) -- real particle = 1, padded = 0
        # split_mask: (B, P) -- keep in output = 1, ignore in output = 0
        if self.options.debug:
            print(f"JetsTransformer forward pass with input shape: {x.shape}")

        B, P, F = x.shape

        # process the masks
        particle_masks = particle_masks.bool()
        particle_masks = (
            particle_masks.unsqueeze(1) if particle_masks is not None else None
        )  # (B, 1, P)
        split_mask = (
            split_mask.unsqueeze(1) if split_mask is not None else None
        )  # (B, 1, P)

        padding_mask = ~particle_masks.squeeze(1)  # (B, P)
        embed_mask = ~particle_masks.transpose(1, 2)  # (B, P, 1)

        attn_mask = None

        # Reshape x to (B*P, F) for particle embedding
        x = x.view(B * P, F)

        # Embed each particle
        x = self.particle_emb(x)

        # Reshape back to (B, P, embed_dim)
        x = x.view(B, P, -1)

        # apply the embed mask
        x = x.masked_fill(embed_mask, 0)  # (B, P, options.emb_dim)

        # Add positional embeddings
        pos_emb_input = create_pos_emb_input(x, stats, particle_masks)
        pos_emb = self.calc_pos_emb(pos_emb_input)
        x = x + pos_emb

        if self.pair_embed is not None:
            if v is None:
                raise ValueError(
                    "Four-momentum tensor 'v' is required when using pair embedding"
                )
            v = v.transpose(1, 2)  # (B, 4, P)
            attn_mask = self.pair_embed(v, None).view(
                -1, v.size(-1), v.size(-1)
            )  # (B*num_heads, P, P)
            if self.options.debug:
                print(f"Pairwise attention mask shape: {attn_mask.shape}")

        # Pass through transformer blocks
        for blk in self.blocks:
            x = blk(x, padding_mask=padding_mask, attn_mask=attn_mask)

        x = self.norm(x)

        if split_mask is not None:
            # Convert split_mask to boolean if it's not already
            if split_mask.dtype != torch.bool:
                split_mask = split_mask.bool()

            # Ensure split_mask has the right shape (B, P)
            if split_mask.dim() == 3:
                split_mask = split_mask.squeeze(
                    1
                )  # Remove middle dimension if (B, 1, P)

            # Create an index tensor for batch dimension
            batch_size = x.size(0)

            # Get selected indices
            selected_indices = split_mask.nonzero(as_tuple=True)

            # Select the particles
            x = x[selected_indices[0], selected_indices[1]]

            # Reshape to (B, num_selected, embed_dim)
            num_selected = split_mask.sum(dim=1).min().item()
            x = x.view(batch_size, num_selected, -1)

        if self.options.debug:
            print(f"JetsTransformer output shape: {x.shape}")

        return x


class ParTPredictor(nn.Module):
    def __init__(
        self,
        options=None,
    ):
        super().__init__()
        self.options = options
        embed_dim = options.embed_dims[-1]

        default_cfg = dict(
            embed_dim=options.predictor_embed_dims[-1],
            num_heads=options.num_heads,
            ffn_ratio=4,
            dropout=options.dropout,
            attn_dropout=options.attn_drop,
            activation_dropout=options.dropout,
            add_bias_kv=False,
            activation=options.activation,
            scale_fc=True,
            scale_attn=True,
            scale_heads=True,
            scale_resids=True,
        )
        cfg_block = default_cfg.copy()
        if options.block_params is not None:
            cfg_block.update(options.block_params)

        # Embedding layers
        pred_emb_input_dim = (
            options.fc_params[-1][0] if options.fc_params else embed_dim
        )
        self.embed = (
            Embed(
                pred_emb_input_dim,
                self.options.predictor_embed_dims,
                activation=self.options.activation,
            )
            if len(self.options.embed_dims) > 0
            else nn.Identity()
        )
        self.calc_predictor_pos_emb = create_space_pos_emb_fn(
            options.predictor_embed_dims[-1]
        )

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [Block(**cfg_block) for _ in range(options.pred_depth)]
        )

        # Normalization and projection layers
        self.norm = nn.LayerNorm(options.predictor_embed_dims[-1])
        self.predictor_proj = nn.Linear(
            options.predictor_embed_dims[-1], options.emb_dim, bias=True
        )

        # Mask token initialization
        self.init_std = options.init_std
        self.mask_token = nn.Parameter(
            torch.zeros(1, 1, options.predictor_embed_dims[-1]), requires_grad=True
        )
        trunc_normal_(self.mask_token, std=self.init_std)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            "mask_token",
        }

    def forward(
        self,
        x,
        ctxt_particle_mask,
        trgt_particle_mask,
        target_particle_ftrs,
        context_particle_ftrs,
        stats,
    ):
        """
        new inputs:
            x: (B, N_ctxt, emb_dim)
            ctxt_particle_mask: (B,  N_ctxt) -- real particle = 1, padded = 0
            trgt_particle_mask: (B,  N_trgt) -- real particle = 1, padded = 0
            target_particle_ftrs: (B, N_trgt, 4) [eta, phi, log_pt, log_energy]
            context_particle_ftrs: (B, N_ctxt, 4) [eta, phi, log_pt, log_energy]
            stats: dictionary of statistics for the input features

        Old Inputs:
            x: context particle representations
                shape: [B, N_ctxt, emb_dim]
            ctxt_particle_mask: mask for zero-padded context particles
                shape: [B, 1, N_ctxt]
            trgt_particle_mask: mask for zero-padded target particles
                shape: [B, 1, N_trgt]
            target_particle_ftrs: target particle 4-vector [eta, phi, log_pt, log_energy]
                shape: [B, N_trgt, 4]
            context_particle_ftrs: context particle 4-vector [eta, phi, log_pt, log_energy]
                shape: [B, N_ctxt, 4]
        Output:
            predicted target particle representations
                shape: [B, N_trgt, predictor_output_dim]
        """
        # unsqueeze the masks
        ctxt_particle_mask = ctxt_particle_mask.unsqueeze(1)  # (B, 1, N_ctxt)
        trgt_particle_mask = trgt_particle_mask.unsqueeze(1)  # (B, 1, N_trgt)
        if self.options.debug:
            print(f"ParTPredictor forward pass with input shape: {x.shape}")

        x = self.embed(x)
        ctxt_pos_emb_input = create_pos_emb_input(
            context_particle_ftrs, stats, ctxt_particle_mask.squeeze(1)
        )
        pos_emb = self.calc_predictor_pos_emb(ctxt_pos_emb_input)
        x += pos_emb

        B, N_ctxt, D = x.shape
        _, N_trgt, _ = target_particle_ftrs.shape

        # Prepare position embeddings for target particles
        trgt_pos_emb_input = create_pos_emb_input(
            target_particle_ftrs, stats, trgt_particle_mask.squeeze(1)
        )
        trgt_pos_emb = self.calc_predictor_pos_emb(trgt_pos_emb_input)
        assert trgt_pos_emb.shape[2] == D

        # Create prediction tokens
        trgt_pos_emb = trgt_pos_emb.view(B * N_trgt, 1, D)
        pred_token = self.mask_token.expand(
            trgt_pos_emb.size(0), trgt_pos_emb.size(1), D
        )
        pred_token = (
            pred_token + trgt_pos_emb
        )  # avoid in-place operation for parameters that require grad

        # Repeat context embeddings for each target
        x = x.repeat_interleave(N_trgt, dim=0)

        # Concatenate context embeddings and prediction tokens
        x = torch.cat([x, pred_token], dim=1)

        # Prepare masks
        # Expand context subjet masks
        ctxt_particle_mask_expanded = ctxt_particle_mask.repeat(1, N_trgt, 1)
        ctxt_particle_mask_expanded = ctxt_particle_mask_expanded.view(
            B * N_trgt, N_ctxt
        )

        # Expand target particle masks
        trgt_particle_mask = trgt_particle_mask.contiguous()
        trgt_particle_mask_expanded = trgt_particle_mask.view(B * N_trgt, 1)

        # Combine masks
        combined_masks = torch.cat(
            [ctxt_particle_mask_expanded, trgt_particle_mask_expanded], dim=1
        )

        # Create padding mask for transformer (True for positions to be masked)
        padding_mask = ~combined_masks.bool()
        # Transpose for transformer input (Seq_len, Batch, Embedding)
        x = x.transpose(0, 1)  # Shape: (N_ctxt + 1, B * N_trgt, D)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, x_cls=None, padding_mask=padding_mask)

        # Transpose back to (Batch, Seq_len, Embedding)
        x = x.transpose(0, 1)
        x = self.norm(x)

        # Extract predictions for target particles
        x = x[:, N_ctxt:, :]  # Shape: (B * N_trgt, 1, D)
        x = self.predictor_proj(x)

        if self.options.debug:
            print(f"ParTPredictor output shape: {x.shape}")

        # Reshape to (B, N_trgt, predictor_output_dim)
        return x.view(B, N_trgt, -1)
