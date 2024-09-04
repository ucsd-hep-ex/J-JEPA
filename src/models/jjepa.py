import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.embedding_stack import EmbeddingStack
from util.positional_embedding import create_pos_emb_fn
from options import Options
from util.tensors import trunc_normal_
from util.DimensionCheckLayer import DimensionCheckLayer

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
        print("Initializing Attention module")
        self.num_heads = options.num_heads
        self.dim = options.attn_dim
        self.head_dim = self.dim // options.num_heads
        self.scale = options.qk_scale or self.head_dim**-0.5
        self.W_qkv = nn.Linear(self.dim, self.dim * 3, bias=options.qkv_bias)
        self.attn_drop = nn.Dropout(options.attn_drop)
        self.proj = nn.Linear(self.dim, self.dim)
        self.proj_drop = nn.Dropout(options.proj_drop)

    def forward(self, x):
        print(f"Attention forward pass with input shape: {x.shape}")
        B, N, C = x.shape
        assert C % self.num_heads == 0
        print("num_heads: ", self.num_heads)
        qkv = (
            self.W_qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        print(f"Attention output shape: {x.shape}")
        return x


class MLP(nn.Module):
    def __init__(self, options: Options):
        super().__init__()
        print("Initializing MLP module")
        act_layer = ACTIVATION_LAYERS.get(options.activation, nn.GELU)
        out_features = options.out_features or options.in_features
        hidden_features = hidden_features or options.in_features
        self.fc1 = nn.Linear(options.in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(options.drop_mlp)

    def forward(self, x):
        print(f"MLP forward pass with input shape: {x.shape}")
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        print(f"MLP output shape: {x.shape}")
        return x


class Block(nn.Module):
    def __init__(self, options: Options):
        super().__init__()
        print("Initializing Block module")
        act_layer = ACTIVATION_LAYERS.get(options.activation, nn.GELU)
        norm_layer = NORM_LAYERS.get(options.normalization, nn.LayerNorm)
        self.norm1 = norm_layer(options.attn_dim)
        self.dim = options.attn_dim
        self.attn = Attention(
            self.dim,
            num_heads=options.num_heads,
            qkv_bias=options.qkv_bias,
            qk_scale=options.qk_scale,
            attn_drop=options.attn_drop,
            proj_drop=options.proj_drop,
        )
        self.drop_path = (
            nn.Identity() if options.drop_path <= 0 else nn.Dropout(options.drop_path)
        )
        self.norm2 = norm_layer(self.dim)
        mlp_hidden_dim = int(self.dim * options.mlp_ratio)
        self.mlp = MLP(
            in_features=self.dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=options.drop_mlp,
        )

    def forward(self, x):
        print(f"Block forward pass with input shape: {x.shape}")
        y = self.attn(self.norm1(x))
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        print(f"Block output shape: {x.shape}")
        return x


class JetsTransformer(nn.Module):
    def __init__(self, options: Options):
        super().__init__()
        print("Initializing JetsTransformer module")
        norm_layer = NORM_LAYERS.get(options.normalization, nn.LayerNorm)
        self.num_part_ftr = options.num_part_ftr
        self.embed_dim = options.emb_dim
        self.calc_pos_emb = create_pos_emb_fn(options.emb_dim)

        # Adjust the input dimensions based on the new input shape
        self.subjet_emb = nn.Linear(
            options.num_particles * options.num_part_ftr, options.emb_dim
        )  # num_features * subjet_length

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=options.emb_dim,
                    num_heads=options.num_heads,
                    mlp_ratio=options.mlp_ratio,
                    qkv_bias=options.qkv_bias,
                    qk_scale=options.qk_scale,
                    drop=options.dropout,
                    attn_drop=options.attn_drop,
                    drop_path=options.drop_path,
                    norm_layer=norm_layer,
                )
                for i in range(options.encoder_depth)
            ]
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

    def forward(self, subjets, subjets_meta):
        """
        Inputs:
            x: particles of subjets
                shape: [B, N_sj, N_part, N_part_ftr]
            subjet_meta: 4 vec of subjets
                shape: [B, N_sj, N_sj_ftr=5]
                N_sj_ftr: pt, eta, phi, E, num_part
        Return:
            subjet representations
        """
        # Flatten last two dimensions to [B, SJ, P*DP]
        B, SJ, P, DP = subjets.shape
        x = subjets.view(B, SJ, -1)

        # subjet emb
        x = self.subjet_emb(x)

        # pos emb
        pos_emb = self.calc_pos_emb(subjets_meta)
        print(pos_emb.shape)
        x += pos_emb

        # forward prop
        for blk in self.blocks:
            x = blk(x)

        # norm
        x = self.norm(x)

        return x


class JetsTransformerPredictor(nn.Module):
    def __init__(self, options: Options):
        super().__init__()
        print("Initializing JetsTransformerPredictor module")
        norm_layer = NORM_LAYERS.get(options.normalization, nn.LayerNorm)
        self.init_std = options.init_std
        self.predictor_embed = nn.Linear(options.repr_dim, options.repr_dim, bias=True)
        self.calc_predictor_pos_emb = create_pos_emb_fn(options.repr_dim)
        self.predictor_blocks = nn.ModuleList(
            [
                Block(
                    dim=options.repr_dim,
                    num_heads=options.num_heads,
                    mlp_ratio=options.mlp_ratio,
                    qkv_bias=options.qkv_bias,
                    qk_scale=options.qk_scale,
                    drop=options.dropout,
                    attn_drop=options.attn_drop,
                    drop_path=options.drop_path,
                    norm_layer=norm_layer,
                )
                for _ in range(options.pred_depth)
            ]
        )
        self.predictor_norm = norm_layer(options.repr_dim)
        # TODO: figure out predictor_output_dim
        self.predictor_proj = nn.Linear(
            options.repr_dim, options.repr_dim, bias=True
        )  # Match target dimensions
        self.apply(self._init_weights)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, options.repr_dim))
        trunc_normal_(self.mask_token, std=self.init_std)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, subjet_mask, target_subjet_ftrs, context_subjet_ftrs):
        """
        Inputs:
            x: context subjet representations
                shape: [B, N_ctxt, emb_dim]
            subjet_mask: mask for zero-padded subjets
                shape: [B, N_ctxt]
            target_subjet_ftrs: target subjet features
                shape: [B, N_trgt, N_ftr]
            context_subjet_ftrs: context subjet features
                shape: [B, N_ctxt, N_ftr]
        Output:
            predicted target subjet representations
                shape: [B, N_trgt, predictor_output_dim]
        """
        print(f"JetsTransformerPredictor forward pass with input shape: {x.shape}")
        # calcualte context positional embedding
        x = self.predictor_embed(x)
        x += self.calc_predictor_pos_emb(context_subjet_ftrs)

        B, N_ctxt, D = x.shape
        _, N_trgt, _ = target_subjet_ftrs.shape
        # prepare position embeddings for target subjets
        # (B, N_trgt, N_ftr) -> (B, N_trgt, D)
        trgt_pos_emb = self.calc_predictor_pos_emb(target_subjet_ftrs)
        assert trgt_pos_emb.shape[2] == D
        # (B, N_trgt, D) -> (B*N_trgt, 1, D) following FAIR_src
        trgt_pos_emb = trgt_pos_emb.view(B * N_trgt, 1, D)
        # TODO: add an learnable token
        pred_token = self.mask_token.repeat(
            trgt_pos_emb.size(0), trgt_pos_emb.size(1), 1
        )
        pred_token += trgt_pos_emb

        # (B, N_ctxt, D) -> (B * N_trgt, N_ctxt, D)
        x = x.repeat(N_trgt, 1, 1)

        x = torch.cat([x, pred_token], axis=1)

        for blk in self.predictor_blocks:
            x = blk(x)
        x = self.predictor_norm(x)

        # -- return the preds for target subjets
        x = x[:, N_ctxt:, :]
        x = self.predictor_proj(x)
        print(f"JetsTransformerPredictor output shape: {x.shape}")
        return x.view(B, N_trgt, -1)


class JJEPA(nn.Module):
    def __init__(self, options: Options):
        super(JJEPA, self).__init__()
        print("Initializing JJEPA module")
        self.use_predictor = options.use_predictor
        self.context_transformer = JetsTransformer(options=options)
        self.target_transformer = copy.deepcopy(self.context_transformer)
        if self.use_predictor:
            self.predictor_transformer = JetsTransformerPredictor(options=options)

        # Debug Statement
        self.input_check = DimensionCheckLayer("Model Input", 3)
        self.context_check = DimensionCheckLayer("After Context Transformer", 3)
        self.predictor_check = DimensionCheckLayer("After Predictor", 3)

    """
    context = {
        particles: torch.Tensor,
        subjets: torch.Tensor,
        particle_mask: torch.Tensor,
        subjet_mask: torch.Tensor,
    }
    """

    def forward(self, context, target):
        print(
            f"JJEPA forward pass with context shape: {context.shape} and target shape: {target.shape}"
        )
        # TODO: update the input to the model
        context_repr = self.context_transformer(context)
        # Debug Statement
        context_repr = self.context_check(context_repr)
        target_repr = self.target_transformer(target)
        if self.use_predictor:
            # TODO: update the input to the model
            pred_repr = self.predictor_transformer(context_repr, context, target)
            pred_repr = self.predictor_check(pred_repr)
            print(
                f"JJEPA output - pred_repr shape: {pred_repr.shape}, context_repr shape: {context_repr.shape}"
            )
            return pred_repr, target_repr

        print(
            f"JJEPA output - context_repr shape: {context_repr.shape}, target shape: {target_repr.shape}"
        )
        return context_repr, target
