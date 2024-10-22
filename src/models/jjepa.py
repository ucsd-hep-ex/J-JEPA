import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.layers import create_embedding_layers, create_predictor_embedding_layers
from src.layers.linear_block.activations import create_activation
from src.layers.embedding_stack import EmbeddingStack, PredictorEmbeddingStack
from src.util import create_pos_emb_fn
from src.options import Options
from src.util.tensors import trunc_normal_
from src.util.DimensionCheckLayer import DimensionCheckLayer

from src.models.ParT.ParticleTransformerEncoder import ParTEncoder, ParTPredictor

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

    def forward(self, x, particle_masks):
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

        x, _ = self.multihead_attn(q, k, v, key_padding_mask=particle_masks)

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
        out_features = options.out_features or options.in_features
        hidden_features = options.hidden_features or options.in_features
        self.fc1 = nn.Linear(options.in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(options.drop_mlp)

    def forward(self, x):
        if self.options.debug:
            print(f"MLP forward pass with input shape: {x.shape}")
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
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

    def forward(self, x, particle_masks):
        if self.options.debug:
            print(f"Block forward pass with input shape: {x.shape}")
        y = self.attn(self.norm1(x), particle_masks)
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if self.options.debug:
            print(f"Block output shape: {x.shape}")
        return x


class JetsTransformer(nn.Module):
    def __init__(self, options: Options):
        super().__init__()
        self.options = options
        if self.options.debug:
            print("Initializing JetsTransformer module")
        norm_layer = NORM_LAYERS.get(options.normalization, nn.LayerNorm)
        self.num_part_ftr = options.num_part_ftr
        self.embed_dim = options.emb_dim
        self.calc_pos_emb = create_pos_emb_fn(options, options.emb_dim)

        print("num_particles", options.num_particles)
        print("num_part_ftr", options.num_part_ftr)

        self.particle_emb = create_embedding_layers(options, options.num_part_ftr)

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

    def forward(self, x, particle_masks, split_mask):
        if self.options.debug:
            print(f"JetsTransformer forward pass with input shape: {x.shape}")

        B, N, F = x.shape

        # Reshape x to (B*N, F) for particle embedding
        x = x.view(B * N, F)

        # Embed each particle
        x = self.particle_emb(x)

        # Reshape back to (B, N, embed_dim)
        x = x.view(B, N, -1)

        # Add positional embeddings
        pos_emb = self.calc_pos_emb(x)
        x = x + pos_emb

        # Pass through transformer blocks
        for blk in self.blocks:
            x = blk(x, particle_masks)

        x = self.norm(x)

        if split_mask is not None:
            # Convert split_mask to boolean if it's not already
            if split_mask.dtype != torch.bool:
                split_mask = split_mask.bool()

            # Apply split mask to select specific particles
            x = x[split_mask]

            # Reshape to (B, num_selected, embed_dim)
            num_selected = split_mask.sum(dim=1).min().item()
            x = x.view(B, num_selected, -1)

        if self.options.debug:
            print(f"JetsTransformer output shape: {x.shape}")

        return x


class JetsTransformerPredictor(nn.Module):
    def __init__(self, options: Options):
        super().__init__()
        self.options = options
        if self.options.debug:
            print("Initializing JetsTransformerPredictor module")
        norm_layer = NORM_LAYERS.get(options.normalization, nn.LayerNorm)
        self.init_std = options.init_std
        self.predictor_embed = create_predictor_embedding_layers(
            options, input_dim=options.emb_dim
        )
        self.calc_predictor_pos_emb = create_pos_emb_fn(
            options, options.predictor_emb_dim
        )
        options.repr_dim = options.predictor_emb_dim
        options.attn_dim = options.repr_dim
        options.in_features = options.repr_dim
        options.out_features = options.repr_dim
        self.predictor_blocks = nn.ModuleList(
            [Block(options=options) for _ in range(options.pred_depth)]
        )
        self.predictor_norm = norm_layer(options.predictor_emb_dim)
        self.predictor_proj = nn.Linear(
            options.predictor_emb_dim, options.emb_dim, bias=True
        )
        self.apply(self._init_weights)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, options.predictor_emb_dim))
        trunc_normal_(self.mask_token, std=self.init_std)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(
        self, context_repr, context_mask, target_mask, context_p4=None, target_p4=None
    ):
        if self.options.debug:
            print(f"JetsTransformerPredictor forward pass")
            print(f"  context_repr shape: {context_repr.shape}")
            print(f"  context_mask shape: {context_mask.shape}")
            print(f"  target_mask shape: {target_mask.shape}")

        x = self.predictor_embed(context_repr)

        B, N_ctxt, D = x.shape
        N_trgt = target_mask.shape[1]

        # Create prediction tokens for target particles
        pred_token = self.mask_token.expand(B, N_trgt, -1)

        # Concatenate context representations and prediction tokens
        x = torch.cat([x, pred_token], dim=1)

        # Add positional embeddings
        if context_p4 is not None and target_p4 is not None:
            full_p4 = torch.cat([context_p4, target_p4], dim=1)
            pos_emb = self.calc_predictor_pos_emb(full_p4)
            x = x + pos_emb

        # Create full particle mask
        full_mask = torch.cat([context_mask, target_mask], dim=1)

        if self.options.debug:
            print(f"  After concatenation:")
            print(f"    x shape: {x.shape}")
            print(f"    full_mask shape: {full_mask.shape}")

        # Ensure the full_mask matches the input sequence length
        if full_mask.shape[1] != x.shape[1]:
            print(
                f"Warning: Mask shape mismatch. Adjusting mask from {full_mask.shape} to match input {x.shape}"
            )
            full_mask = F.pad(
                full_mask, (0, x.shape[1] - full_mask.shape[1]), value=False
            )

        # Pass through predictor blocks
        for blk in self.predictor_blocks:
            x = blk(x, full_mask)

        x = self.predictor_norm(x)

        # Return the predictions for target particles
        x = x[:, N_ctxt:, :]
        x = self.predictor_proj(x)

        if self.options.debug:
            print(f"JetsTransformerPredictor output shape: {x.shape}")

        return x


class JJEPA(nn.Module):
    def __init__(self, options: Options):
        super(JJEPA, self).__init__()
        self.options = options
        if self.options.debug:
            print("Initializing JJEPA module")
        self.use_predictor = options.use_predictor
        self.use_parT = options.use_parT

        if self.use_parT:
            self.context_transformer = ParTEncoder(options=options)
        else:
            self.context_transformer = JetsTransformer(options)

        self.target_transformer = copy.deepcopy(self.context_transformer)
        for param in self.target_transformer.parameters():
            param.requires_grad = False

        if self.use_predictor:
            if self.use_parT:
                self.predictor_transformer = ParTPredictor(options=options)
            else:
                self.predictor_transformer = JetsTransformerPredictor(options)

        if self.options.debug:
            self.input_check = DimensionCheckLayer("Model Input", 3)
            self.context_check = DimensionCheckLayer("After Context Transformer", 3)
            self.predictor_check = DimensionCheckLayer("After Predictor", 3)

    def forward(self, context, target, full_jet, stats):
        if self.options.debug:
            print(f"JJEPA forward pass")
            print(f"Context shape: {context['p4_spatial'].shape}")
            print(f"Target shape: {target['p4_spatial'].shape}")
            print(f"Full jet shape: {full_jet['p4'].shape}")
            print(f"Context particle mask shape: {context['particle_mask'].shape}")
            print(f"Target particle mask shape: {target['particle_mask'].shape}")
            print(f"Full jet particle mask shape: {full_jet['particle_mask'].shape}")

        context_split_mask = (
            context["split_mask"].bool() if context["split_mask"] is not None else None
        )
        target_split_mask = (
            target["split_mask"].bool() if target["split_mask"] is not None else None
        )

        if self.use_parT:
            context_repr = self.context_transformer(
                full_jet["p4"],
                full_jet["p4_spatial"],
                full_jet["particle_mask"],
                context_split_mask,
                stats=stats,
            )
            target_repr = self.target_transformer(
                full_jet["p4"],
                full_jet["p4_spatial"],
                full_jet["particle_mask"],
                target_split_mask,
                stats=stats,
            )
        else:
            context_repr = self.context_transformer(
                full_jet["p4"],
                full_jet["particle_mask"],
                context_split_mask,
                stats=stats,
            )
            target_repr = self.target_transformer(
                full_jet["p4"],
                full_jet["particle_mask"],
                target_split_mask,
                stats=stats,
            )
        if self.options.debug:
            print(f"Context repr shape: {context_repr.shape}")
            print(f"Target repr shape: {target_repr.shape}")

        if self.use_predictor:
            # pred_repr = self.predictor_transformer(
            #     context_repr, context["particle_mask"], target["particle_mask"]
            # )
            pred_repr = self.predictor_transformer(
                context_repr,
                context["particle_mask"],
                target["particle_mask"],
                target["p4"],
                context["p4"],
                stats=stats,
            )
            if self.options.debug:
                pred_repr = self.predictor_check(pred_repr)
                print(f"Predictor output shape: {pred_repr.shape}")
        else:
            pred_repr = None
            if self.options.debug:
                print("Predictor not used in forward pass")

        if self.options.debug:
            print(f"JJEPA output shapes:")
            print(f"  pred_repr: {pred_repr.shape if pred_repr is not None else None}")
            print(f"  target_repr: {target_repr.shape}")
            print(f"  context_repr: {context_repr.shape}")

        return pred_repr, target_repr, context_repr
