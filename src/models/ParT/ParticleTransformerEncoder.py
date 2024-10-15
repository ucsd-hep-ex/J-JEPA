import copy
import torch
import torch.nn as nn
from src.models.ParT.PairEmbed import Embed, PairEmbed
from src.models.ParT.Block import Block
from src.models.ParT.utils import trunc_normal_
from src.layers import create_embedding_layers, create_predictor_embedding_layers
from src.util.positional_embedding import create_space_pos_emb_fn


class ParTEncoder(nn.Module):
    def __init__(
        self,
        for_inference=False,
        aggregate_ptcl_features=False,  # not needed for J-JEPA
        options=None,
    ):
        super().__init__()
        self.options = options
        self.aggregate_ptcl_features = aggregate_ptcl_features
        embed_dim = (
            self.options.embed_dims[-1]
            if len(self.options.embed_dims) > 0
            else self.options.input_dim
        )
        default_cfg = dict(
            embed_dim=embed_dim,
            num_heads=self.options.num_heads,
            ffn_ratio=4,
            dropout=0.1,
            attn_dropout=0.1,
            activation_dropout=0.1,
            add_bias_kv=False,
            activation=self.options.activation,
            scale_fc=True,
            scale_attn=True,
            scale_heads=True,
            scale_resids=True,
        )
        self.calc_pos_emb = create_space_pos_emb_fn(embed_dim)
        cfg_block = default_cfg.copy()
        if self.options.block_params is not None:
            cfg_block.update(self.options.block_params)

        cfg_cls_block = copy.deepcopy(default_cfg)
        if self.options.cls_block_params is not None:
            cfg_cls_block.update(self.options.cls_block_params)

        self.embed = (
            Embed(
                self.options.input_dim,
                self.options.embed_dims,
                activation=self.options.activation,
            )
            if len(self.options.embed_dims) > 0
            else nn.Identity()
        )
        self.pair_embed = (
            PairEmbed(
                self.options.pair_input_dim,
                0,
                self.options.pair_embed_dims + [cfg_block["num_heads"]],
                remove_self_pair=True,
                use_pre_activation_pair=True,
                for_onnx=for_inference,
            )
            if self.options.pair_embed_dims is not None
            and self.options.pair_input_dim > 0
            else None
        )
        self.blocks = nn.ModuleList(
            [Block(**cfg_block) for _ in range(self.options.num_layers)]
        )
        self.cls_blocks = (
            nn.ModuleList(
                [Block(**cfg_cls_block) for _ in range(self.options.num_cls_layers)]
            )
            if self.options.num_cls_layers > 0
            else None
        )
        self.norm = nn.LayerNorm(embed_dim)
        if self.options.fc_params is not None:
            fcs = []
            in_dim = embed_dim
            for out_dim, drop_rate in self.options.fc_params:
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

    def forward(self, x, v=None, mask=None, split_mask=None, uu=None):
        # new shapes:
        # x: (N, P, 4) [eta, phi, log_pt, log_energy]
        # v: (N, P, 4) [px,py,pz,energy]
        # mask: (N, P) -- real particle = 1, padded = 0
        # split_mask: (N, P) -- keep in output = 1, ignore in output = 0
        # old shapes:
        # x: (N, C, P)
        # v: (N, 4, P) [px,py,pz,energy]
        # mask: (N, 1, P) -- real particle = 1, padded = 0
        # split_mask: (N, 1, P) -- keep in output = 1, ignore in output = 0
        pos_emb_input = torch.clone(x)  # (N, P, 4)
        x = x.transpose(1, 2)  # (N, 4, P)
        v = v.transpose(1, 2) if v is not None else None  # (N, 4, P)
        mask = mask.unsqueeze(1) if mask is not None else None  # (N, 1, P)
        split_mask = (
            split_mask.unsqueeze(1) if split_mask is not None else None
        )  # (N, 1, P)

        padding_mask = ~mask.squeeze(1)  # (N, P)
        with torch.cuda.amp.autocast(enabled=self.options.use_amp):
            # input embedding
            x = self.embed(x).masked_fill(~mask.permute(2, 0, 1), 0)  # (P, N, emb_dim)
            if self.options.encoder_pos_emb:
                pos_emb = self.calc_pos_emb(pos_emb_input).transpose(
                    0, 1
                )  # (P, N, emb_dim)
                print(f"pos_emb shape: {pos_emb.shape}")
                print(f"x shape: {x.shape}")
                x += pos_emb
            attn_mask = None
            if (v is not None or uu is not None) and self.pair_embed is not None:
                attn_mask = self.pair_embed(v, uu).view(
                    -1, v.size(-1), v.size(-1)
                )  # (N*num_heads, P, P)

            for block in self.blocks:
                x = block(x, x_cls=None, padding_mask=padding_mask, attn_mask=attn_mask)
            x_cls = x
            if self.aggregate_ptcl_features:
                if self.cls_blocks is not None:
                    # extract class token
                    cls_tokens = self.cls_token.expand(1, x.size(1), -1)  # (1, N, C)
                    for block in self.cls_blocks:
                        cls_tokens = block(
                            x, x_cls=cls_tokens, padding_mask=padding_mask
                        )
                    x_cls = self.norm(cls_tokens).squeeze(0)
                else:
                    x = x.sum(dim=0)
                    x_cls = self.norm(x)  # (batch_size, embed_dim)

            # apply split mask
            # print("x_cls shape:", x_cls.shape)
            x_cls = x_cls.transpose(0, 1)  # (N, P, emb_dim)
            if split_mask is not None:
                split_mask = split_mask.bool()
                # 'output' is the output tensor of shape (batch_size, N, emb_dim)
                # 'split_mask' is the mask tensor of shape (batch_size, N)

                # Example tensors
                B, N, emb_dim = (
                    x_cls.shape
                )  # Batch size, sequence length, embedding dimension

                # Step 1: Find indices where 'split_mask' is True
                indices = split_mask.nonzero(as_tuple=False)  # Shape: (num_selected, 2)
                batch_indices = indices[:, 0]  # Batch indices
                seq_indices = indices[:, 1]  # Sequence indices

                # Step 2: Gather selected particles
                selected_particles = x_cls[
                    batch_indices, seq_indices, :
                ]  # Shape: (num_selected, emb_dim)

                # Step 3: Compute the number of selected particles per batch
                lengths = (
                    split_mask.squeeze(dim=1).sum(dim=1).to(torch.long)
                )  # Shape: (B,)

                # Step 4: Compute cumulative lengths and positions within each batch
                cum_lengths = torch.cat(
                    [torch.tensor([0], device=lengths.device), lengths.cumsum(0)[:-1]]
                )
                positions_in_batch = torch.arange(
                    lengths.sum(), device=lengths.device
                ) - torch.repeat_interleave(cum_lengths, lengths)

                # Step 5: Determine the maximum number of selections
                max_length = lengths.max().item()

                # Step 6: Initialize a padded tensor
                selected_particles_padded = torch.zeros(
                    B, max_length, emb_dim, device=x_cls.device, dtype=x_cls.dtype
                )

                # Step 7: Assign selected particles to the padded tensor
                selected_particles_padded[batch_indices, positions_in_batch] = (
                    selected_particles
                )
                x_cls = selected_particles_padded  # (B, max_length, emb_dim)
            if self.fc is None:
                print(f"encoder output shape: {x_cls.transpose(0, 1).shape}")
                return x_cls.transpose(0, 1)
            output = self.fc(x_cls).transpose(0, 1)
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
        print(f"pred_emb_input_dim: {pred_emb_input_dim}")
        self.embed = (
            Embed(
                pred_emb_input_dim,
                self.options.predictor_embed_dims,
                activation=self.options.activation,
            )
            if len(self.options.embed_dims) > 0
            else nn.Identity()
        )
        print("embedding layers of ParTPredictor")
        print(self.embed)
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
    ):
        """
        new inputs:
            x: (B, N_ctxt, 4) [eta, phi, log_pt, log_energy]
            ctxt_particle_mask: (B,  N_ctxt) -- real particle = 1, padded = 0
            trgt_particle_mask: (B,  N_trgt) -- real particle = 1, padded = 0
            target_particle_ftrs: (B, N_trgt, 4) [eta, phi, log_pt, log_energy]
            context_particle_ftrs: (B, N_ctxt, 4) [eta, phi, log_pt, log_energy]

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

        print(f"ParTPredictor embedding with input x shape: {x.shape}")
        x = self.embed(x.transpose(1, 2))
        print(f"ParTPredictor embedding output shape: {x.shape}")
        pos_emb = self.calc_predictor_pos_emb(context_particle_ftrs)
        print(f"pos_emb shape: {pos_emb.shape}")
        print(f"x shape: {x.shape}")
        x += pos_emb

        B, N_ctxt, D = x.shape
        _, N_trgt, _ = target_particle_ftrs.shape

        # Prepare position embeddings for target particles
        trgt_pos_emb = self.calc_predictor_pos_emb(target_particle_ftrs)
        print(f"trgt_pos_emb shape: {trgt_pos_emb.shape}")
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
