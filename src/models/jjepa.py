import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.layers import create_embedding_layers, create_predictor_embedding_layers
from src.layers.linear_block.activations import create_activation
from src.layers.embedding_stack import EmbeddingStack, PredictorEmbeddingStack, LearnableEmbeddingStacks
from src.util import create_pos_emb_fn
from src.util.pt_pos_emb import create_pt_pos_emb_fn
from src.options import Options
from src.util.tensors import trunc_normal_
from src.util.DimensionCheckLayer import DimensionCheckLayer

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
        self.proj_drop = nn.Dropout(options.proj_drop)

        self.multihead_attn = nn.MultiheadAttention(
            self.dim, self.num_heads, batch_first=True
        )

    def forward(self, x, subjet_masks):
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
        x, _ = self.multihead_attn(q, k, v, key_padding_mask=subjet_masks == 0)

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

    def forward(self, x, subjet_masks):
        if self.options.debug:
            print(f"Block forward pass with input shape: {x.shape}")
        y = self.attn(self.norm1(x), subjet_masks)
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
        self.use_learnable_space_emb = False
        if options.pos_emb_type == "pt":
            self.calc_pos_emb = create_pt_pos_emb_fn(options.emb_dim)
        elif options.pos_emb_type == "Learnable_Space":
            self.use_learnable_space_emb = True
            emb_dim = options.emb_dim
            self.eta_emb_layers = LearnableEmbeddingStacks(input_dim=1, output_dim=emb_dim // 4)
            self.phi_emb_layers = LearnableEmbeddingStacks(input_dim=1, output_dim=emb_dim // 4)
            self.eta_phi_low_level_emb_layers = LearnableEmbeddingStacks(input_dim=2, output_dim=emb_dim // 4)
            self.eta_phi_high_level_emb_layers = LearnableEmbeddingStacks(input_dim=emb_dim // 2, output_dim=emb_dim // 4)
        else:
            self.calc_pos_emb = create_pos_emb_fn(options, options.emb_dim)

        # Adjust the input dimensions based on the new input shape
        print("num_particles", options.num_particles)
        print("num_part_ftr", options.num_part_ftr)
        self.subjet_emb = create_embedding_layers(
            options, options.num_particles * options.num_part_ftr
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

    def calc_learnable_space_embedding(self, subjet_ftrs):
        sj_eta = subjet_ftrs[:, :, 1].unsqueeze(-1)
        sj_phi = subjet_ftrs[:, :, 2].unsqueeze(-1)

        # process phi to impose physical distance
        sj_phi_star = torch.sin(sj_phi / 2)

        # shift eta to avoid negative positions
        sj_eta += 3

        # calculate embedding
        emb_phi = self.phi_emb_layers(sj_phi_star)
        emb_eta = self.eta_emb_layers(sj_eta)
        emb_low_level = self.eta_phi_low_level_emb_layers(
                            torch.cat([sj_phi_star, sj_eta], axis=2)
                        )
        emb_high_level = self.eta_phi_high_level_emb_layers(
                            torch.cat([emb_phi, emb_eta], axis=2)
                        )

        # print(emb_phi_star.shape)
        emb = torch.cat([emb_phi, emb_eta, emb_low_level, emb_high_level], axis=2)
        return emb

    def forward(self, x, subjet_masks, subjets_meta, split_mask, particle_masks=None):
        """
        Inputs:
            x: particles of subjets
                shape: [B, N_sj, N_part * N_part_ftr]
            subjet_meta: 4 vec of subjets
                shape: [B, N_sj, N_sj_ftr=5]
                N_sj_ftr: pt, eta, phi, E, num_part
            split_mask: mask out certain subjet representations, depending on context/target
                shape: [B, N_sj_to_keep]
        Return:
            subjet representations
                shape: [B, N_sj, emb_dim]
        """
        # Flatten last two dimensions to [B, SJ, P*DP]
        x = x["particles"]
        B, SJ, _ = x.shape

        # subjet emb
        # if not use attention blks to create subjet emb
        if particle_masks is None:
            x = x.view(B, SJ, -1)
            x = self.subjet_emb(x)
        # use attention blks to create subjet emb
        else:
            x = x.view(B, SJ, -1, self.num_part_ftr)
            x = self.subjet_emb(x, particle_masks)

        # pos emb
        if self.options.encoder_pos_emb:
            if self.use_learnable_space_emb:
                pos_emb = self.calc_learnable_space_embedding(subjets_meta)
            else:
                pos_emb = self.calc_pos_emb(subjets_meta)
            if self.options.debug:
                print(pos_emb.shape)
            x += pos_emb

        # forward prop
        for blk in self.blocks:
            x = blk(x, subjet_masks)

        # norm
        x = self.norm(x)
        if split_mask != None:
            # select indices of certain subjet representations from split_mask
            selected_subjets = x[split_mask.unsqueeze(-1).expand(-1, -1, x.shape[-1])]

            num_selected = split_mask.sum(
                dim=1
            ).min()  # Minimum to handle potentially non-uniform selections
            selected_subjets = selected_subjets.view(B, num_selected, x.shape[-1])
            return selected_subjets
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
        self.use_learnable_space_emb = False
        if options.pos_emb_type == "pt":
            self.calc_predictor_pos_emb = create_pt_pos_emb_fn(
                options.predictor_emb_dim
            )
        elif options.pos_emb_type == "Learnable_Space":
            self.use_learnable_space_emb = True
            emb_dim = options.predictor_emb_dim
            self.eta_emb_layers = LearnableEmbeddingStacks(input_dim=1, output_dim=emb_dim // 4)
            self.phi_emb_layers = LearnableEmbeddingStacks(input_dim=1, output_dim=emb_dim // 4)
            self.eta_phi_low_level_emb_layers = LearnableEmbeddingStacks(input_dim=2, output_dim=emb_dim // 4)
            self.eta_phi_high_level_emb_layers = LearnableEmbeddingStacks(input_dim=emb_dim // 2, output_dim=emb_dim // 4)
        else:
            self.calc_predictor_pos_emb = create_pos_emb_fn(
                options,
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
        )  # Match target dimensions
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

    def calc_learnable_space_embedding(self, subjet_ftrs):
        sj_eta = subjet_ftrs[:, :, 1].unsqueeze(-1)
        sj_phi = subjet_ftrs[:, :, 2].unsqueeze(-1)

        # process phi to impose physical distance
        sj_phi_star = torch.sin(sj_phi / 2)

        # shift eta to avoid negative positions
        sj_eta += 3

        # calculate embedding
        emb_phi = self.phi_emb_layers(sj_phi_star)
        emb_eta = self.eta_emb_layers(sj_eta)
        emb_low_level = self.eta_phi_low_level_emb_layers(
                            torch.cat([sj_phi_star, sj_eta], axis=2)
                        )
        emb_high_level = self.eta_phi_high_level_emb_layers(
                            torch.cat([emb_phi, emb_eta], axis=2)
                        )

        # print(emb_phi_star.shape)
        emb = torch.cat([emb_phi, emb_eta, emb_low_level, emb_high_level], axis=2)
        return emb

    def forward(self, x, subjet_masks, target_subjet_ftrs, context_subjet_ftrs):
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
        if self.options.debug:
            print(f"JetsTransformerPredictor forward pass with input shape: {x.shape}")



        x = self.predictor_embed(x)

        B, N_ctxt, D = x.shape
        _, N_trgt, _ = target_subjet_ftrs.shape

        if self.use_learnable_space_emb:
            # calcualte context positional embedding
            x += self.calc_learnable_space_embedding(context_subjet_ftrs)
            # prepare position embeddings for target subjets
            # (B, N_trgt, N_ftr) -> (B, N_trgt, D)
            trgt_pos_emb = self.calc_learnable_space_embedding(target_subjet_ftrs)
        else:
            # calcualte context positional embedding
            x += self.calc_predictor_pos_emb(context_subjet_ftrs)
            # prepare position embeddings for target subjets
            # (B, N_trgt, N_ftr) -> (B, N_trgt, D)
            trgt_pos_emb = self.calc_predictor_pos_emb(target_subjet_ftrs)

        assert trgt_pos_emb.shape[2] == D
        # (B, N_trgt, D) -> (B*N_trgt, 1, D) following FAIR_src
        trgt_pos_emb = trgt_pos_emb.view(B * N_trgt, 1, D)
        pred_token = self.mask_token.repeat(
            trgt_pos_emb.size(0), trgt_pos_emb.size(1), 1
        )
        pred_token += trgt_pos_emb

        # (B, N_ctxt, D) -> (B * N_trgt, N_ctxt, D)
        x = x.repeat_interleave(N_trgt, dim=0)

        x = torch.cat([x, pred_token], axis=1)

        subjet_masks = torch.cat(
            [subjet_masks, torch.ones((B, 1)).to(subjet_masks.device)], axis=1
        )

        subjet_masks = subjet_masks.repeat(N_trgt, 1)

        for blk in self.predictor_blocks:
            x = blk(x, subjet_masks)
        x = self.predictor_norm(x)

        # -- return the preds for target subjets
        x = x[:, N_ctxt:, :]
        x = self.predictor_proj(x)
        if self.options.debug:
            print(f"JetsTransformerPredictor output shape: {x.shape}")
        return x.view(B, N_trgt, -1)


class JJEPA(nn.Module):
    def __init__(self, options: Options):
        super(JJEPA, self).__init__()
        self.options = options
        if self.options.debug:
            print("Initializing JJEPA module")
        self.use_predictor = options.use_predictor
        self.context_transformer = JetsTransformer(options)
        self.target_transformer = copy.deepcopy(self.context_transformer)
        self.need_particle_masks = "att" in options.embedding_layers_type.lower()
        for param in self.target_transformer.parameters():
            param.requires_grad = False
        if self.use_predictor:
            self.predictor_transformer = JetsTransformerPredictor(options)

        # Debug Statement
        if self.options.debug:
            self.input_check = DimensionCheckLayer("Model Input", 3)
            self.context_check = DimensionCheckLayer("After Context Transformer", 3)
            self.predictor_check = DimensionCheckLayer("After Predictor", 3)

    """
    context = {
        subjets: torch.Tensor,
        particle_mask: torch.Tensor,
        subjet_mask: torch.Tensor,
        split_mask: torch.Tensor,
    }
    target = {
        subjets: torch.Tensor,
        particle_mask: torch.Tensor,
        subjet_mask: torch.Tensor,
        split_mask: torch.Tensor,
    }
    full_jet = {
        particles: torch.Tensor,
        particle_mask: torch.Tensor,
        subjet_mask: torch.Tensor,
    }
    """

    def forward(self, context, target, full_jet):
        if self.options.debug:
            print(f"JJEPA forward pass")

        if self.need_particle_masks:
            context_repr = self.context_transformer(
                full_jet,
                full_jet["subjet_mask"],
                full_jet["subjets"],
                context["split_mask"],
                full_jet["particle_mask"],
            )

            # Debug Statement
            if self.options.debug:
                context_repr = self.context_check(context_repr)

            target_repr = self.target_transformer(
                full_jet,
                full_jet["subjet_mask"],
                full_jet["subjets"],
                target["split_mask"],
                full_jet["particle_mask"],
            )
        else:
            context_repr = self.context_transformer(
                full_jet,
                full_jet["subjet_mask"],
                full_jet["subjets"],
                context["split_mask"],
            )

            # Debug Statement
            if self.options.debug:
                context_repr = self.context_check(context_repr)

            target_repr = self.target_transformer(
                full_jet,
                full_jet["subjet_mask"],
                full_jet["subjets"],
                target["split_mask"],
            )
        if self.use_predictor:
            # TODO: update the input to the model x, subjet_mask, target_subjet_ftrs, context_subjet_ftrs):
            pred_repr = self.predictor_transformer(
                context_repr,
                context["subjet_mask"],
                target["subjets"],
                context["subjets"],
            )
            if self.options.debug:
                pred_repr = self.predictor_check(pred_repr)
            if self.options.debug:
                print(
                    f"JJEPA output - pred_repr shape: {pred_repr.shape}, context_repr shape: {context_repr.shape}"
                )
            return pred_repr, target_repr, context_repr

        if self.options.debug:
            print(
                f"JJEPA output - context_repr shape: {context_repr.shape}, target shape: {target_repr.shape}"
            )
        return context_repr, target_repr
