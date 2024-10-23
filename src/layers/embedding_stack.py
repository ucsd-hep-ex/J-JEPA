from typing import List
from copy import deepcopy

import torch
from torch import Tensor, nn

from src.layers.linear_block.basic_block import BasicBlock
from src.layers.linear_block import create_linear_block
from src.layers.linear_block.activations import create_activation

from src.layers.attention_block import create_particle_attention_block, create_class_attention_block

from src.options import Options

from src.util.tensors import trunc_normal_


class EmbeddingStack(nn.Module):
    __constants__ = ["input_dim"]

    def __init__(self, options: Options, input_dim: int):
        super(EmbeddingStack, self).__init__()

        self.input_dim = input_dim
        self.embedding_layers = nn.ModuleList(self.create_embedding_layers(options, input_dim))

    @staticmethod
    def create_embedding_layers(options: Options, input_dim: int) -> List[BasicBlock]:
        """ Create a stack of linear layer with increasing emb dimensions.

        Each emb layer will have double the dimensions as the previous, beginning with the
        size of the feature-space and ending with the emb_dim specified in options.
        """

        # Initial embedding layer to just project to our shared first dimension.
        embedding_layers = [create_linear_block(
            options,
            input_dim,
            options.initial_embedding_dim,
            options.initial_embedding_skip_connections
        )]
        current_embedding_dim = options.initial_embedding_dim

        # Keep doubling dimensions until we reach the desired latent dimension.
        for i in range(options.num_embedding_layers):
            next_embedding_dim = 2 * current_embedding_dim
            if next_embedding_dim >= options.emb_dim:
                break

            embedding_layers.append(create_linear_block(
                options,
                current_embedding_dim,
                next_embedding_dim,
                options.embedding_skip_connections
            ))
            current_embedding_dim = next_embedding_dim

        # Final embedding layer to ensure proper size on output.
        embedding_layers.append(create_linear_block(
            options,
            current_embedding_dim,
            options.emb_dim,
            options.embedding_skip_connections
        ))

        return embedding_layers

    def forward(self, vectors: Tensor) -> Tensor:
        """ Embed a sequence of vectors through a series of doubling linear layers.

        Parameters
        ----------
        vectors: [*, input_dim]
            Original vectors to embed.

        Returns
        -------
        embeddings: [*, output_dim]
            Output embeddings.
        """

        embeddings = vectors

        for layer in self.embedding_layers:
            embeddings = layer(embeddings)

        return embeddings

class LinearEmbeddingStack(nn.Module):
    __constants__ = ["input_dim"]

    def __init__(self, options: Options, input_dim: int):
        super(LinearEmbeddingStack, self).__init__()
        self.options = deepcopy(options)
        self.options.activation = 'linear'
        self.input_dim = input_dim
        self.embedding_layers = nn.ModuleList(self.create_embedding_layers(self.options, input_dim))

    @staticmethod
    def create_embedding_layers(options: Options, input_dim: int) -> List[BasicBlock]:
        """ Create a stack of linear layer with increasing emb dimensions.

        Each emb layer will have double the dimensions as the previous, beginning with the
        size of the feature-space and ending with the emb_dim specified in options.
        """

        # Initial embedding layer to just project to our shared first dimension.
        embedding_layers = [create_linear_block(
            options,
            input_dim,
            options.initial_embedding_dim,
            options.initial_embedding_skip_connections
        )]
        current_embedding_dim = options.initial_embedding_dim

        # Keep doubling dimensions until we reach the desired latent dimension.
        for i in range(options.num_embedding_layers):
            next_embedding_dim = 2 * current_embedding_dim
            if next_embedding_dim >= options.emb_dim:
                break

            embedding_layers.append(create_linear_block(
                options,
                current_embedding_dim,
                next_embedding_dim,
                options.embedding_skip_connections
            ))
            current_embedding_dim = next_embedding_dim

        # Final embedding layer to ensure proper size on output.
        embedding_layers.append(create_linear_block(
            options,
            current_embedding_dim,
            options.emb_dim,
            options.embedding_skip_connections
        ))

        return embedding_layers

    def forward(self, vectors: Tensor) -> Tensor:
        """ Embed a sequence of vectors through a series of doubling linear layers.

        Parameters
        ----------
        vectors: [*, input_dim]
            Original vectors to embed.

        Returns
        -------
        embeddings: [*, output_dim]
            Output embeddings.
        """

        embeddings = vectors

        for layer in self.embedding_layers:
            embeddings = layer(embeddings)

        return embeddings

# TODO: This class is to enable faster development for downstream tasks
# should delete it after the deadline chasing
class PredictorEmbeddingStack(nn.Module):
    __constants__ = ["input_dim"]

    def __init__(self, options: Options, input_dim: int):
        super(PredictorEmbeddingStack, self).__init__()

        self.input_dim = input_dim
        self.embedding_layers = nn.ModuleList(self.create_embedding_layers(options, input_dim))

    @staticmethod
    def create_embedding_layers(options: Options, input_dim: int) -> List[BasicBlock]:
        """ Create a stack of linear layer with increasing emb dimensions.

        Each emb layer will have double the dimensions as the previous, beginning with the
        size of the feature-space and ending with the emb_dim specified in options.
        """

        # Initial embedding layer to just project to our shared first dimension.
        embedding_layers = [create_linear_block(
            options,
            input_dim,
            options.initial_embedding_dim,
            options.initial_embedding_skip_connections
        )]
        current_embedding_dim = options.initial_embedding_dim

        # Keep doubling dimensions until we reach the desired latent dimension.
        for i in range(options.num_embedding_layers):
            next_embedding_dim = 2 * current_embedding_dim
            if next_embedding_dim >= options.predictor_emb_dim:
                break

            embedding_layers.append(create_linear_block(
                options,
                current_embedding_dim,
                next_embedding_dim,
                options.embedding_skip_connections
            ))
            current_embedding_dim = next_embedding_dim

        # Final embedding layer to ensure proper size on output.
        embedding_layers.append(create_linear_block(
            options,
            current_embedding_dim,
            options.predictor_emb_dim,
            options.embedding_skip_connections
        ))

        return embedding_layers

    def forward(self, vectors: Tensor) -> Tensor:
        """ Embed a sequence of vectors through a series of doubling linear layers.

        Parameters
        ----------
        vectors: [*, input_dim]
            Original vectors to embed.

        Returns
        -------
        embeddings: [*, output_dim]
            Output embeddings.
        """

        embeddings = vectors

        for i, layer in enumerate(self.embedding_layers):
            embeddings = layer(embeddings)

        return embeddings

class PredictorLinearEmbeddingStack(nn.Module):
    __constants__ = ["input_dim"]

    def __init__(self, options: Options, input_dim: int):
        super(PredictorLinearEmbeddingStack, self).__init__()
        self.options = deepcopy(options)
        self.options.activation = 'linear'

        self.input_dim = input_dim
        self.embedding_layers = nn.ModuleList(self.create_embedding_layers(self.options, input_dim))

    @staticmethod
    def create_embedding_layers(options: Options, input_dim: int) -> List[BasicBlock]:
        """ Create a stack of linear layer with increasing emb dimensions.

        Each emb layer will have double the dimensions as the previous, beginning with the
        size of the feature-space and ending with the emb_dim specified in options.
        """

        # Initial embedding layer to just project to our shared first dimension.
        embedding_layers = [create_linear_block(
            options,
            input_dim,
            options.initial_embedding_dim,
            options.initial_embedding_skip_connections
        )]
        current_embedding_dim = options.initial_embedding_dim

        # Keep doubling dimensions until we reach the desired latent dimension.
        for i in range(options.num_embedding_layers):
            next_embedding_dim = 2 * current_embedding_dim
            if next_embedding_dim >= options.predictor_emb_dim:
                break

            embedding_layers.append(create_linear_block(
                options,
                current_embedding_dim,
                next_embedding_dim,
                options.embedding_skip_connections
            ))
            current_embedding_dim = next_embedding_dim

        # Final embedding layer to ensure proper size on output.
        embedding_layers.append(create_linear_block(
            options,
            current_embedding_dim,
            options.predictor_emb_dim,
            options.embedding_skip_connections
        ))

        return embedding_layers

    def forward(self, vectors: Tensor) -> Tensor:
        """ Embed a sequence of vectors through a series of doubling linear layers.

        Parameters
        ----------
        vectors: [*, input_dim]
            Original vectors to embed.

        Returns
        -------
        embeddings: [*, output_dim]
            Output embeddings.
        """

        embeddings = vectors

        for layer in self.embedding_layers:
            embeddings = layer(embeddings)

        return embeddings

class PlainAttentionEmbeddingStack(nn.Module):
    __constants__ = ["input_dim"]

    def __init__(self, options: Options, input_dim: int):
        super(PlainAttentionEmbeddingStack, self).__init__()

        self.has_cls_attn_blks = options.num_class_attention_blocks_in_embedding > 0

        self.particle_embedding_layers = nn.ModuleList(self.create_particle_embedding_layers(options, options.num_part_ftr))
        self.particle_attention_blocks = nn.ModuleList(self.create_particle_attention_blocks(options))
        if self.has_cls_attn_blks:
            self.init_std = options.init_std
            self.cls_token = nn.Parameter(torch.zeros(1, 1, 1, options.particle_emb_dim), requires_grad=True)
            trunc_normal_(self.mask_token, std=self.init_std)
            self.class_attention_blocks = nn.ModuleList(self.create_class_attention_blocks(options))

        # pre-act norm
        norm_layer = nn.LayerNorm
        self.norm = norm_layer(options.emb_dim * 4)

        # if use cls attn block, output size is (bs, n_sj, particle_emb_dim)
        if self.has_cls_attn_blks:
            self.fc1 = nn.Linear(options.particle_emb_dim, options.emb_dim * 4)
        # if not use cls attn block, output of part attn block will be flatten
        else:
            self.fc1 = nn.Linear(options.particle_emb_dim * options.num_particles, options.emb_dim * 4)

        self.act = create_activation(options.activation, None)
        self.fc2 = nn.Linear(options.emb_dim * 4, options.emb_dim)
        self.drop = nn.Dropout(options.drop_mlp)

        self.softmax = nn.Softmax(-1)

    @staticmethod
    def create_particle_embedding_layers(options: Options, input_dim: int) -> List[BasicBlock]:
        """ Create a stack of particle embeddings layers increasing emb dimensions.

        Each emb layer will have double the dimensions as the previous, beginning with the
        size of the feature-space and ending with the emb_dim specified in options.
        """

        # Initial embedding layer to just project to our shared first dimension.
        embedding_layers = [create_linear_block(
            options,
            input_dim,
            options.initial_particle_embedding_dim,
            options.initial_particle_embedding_skip_connections
        )]
        current_embedding_dim = options.initial_particle_embedding_dim

        # Keep doubling dimensions until we reach the desired latent dimension.
        for i in range(options.num_particle_embedding_layers):
            next_embedding_dim = 2 * current_embedding_dim
            if next_embedding_dim >= options.particle_emb_dim:
                break

            embedding_layers.append(create_linear_block(
                options,
                current_embedding_dim,
                next_embedding_dim,
                options.particle_embedding_skip_connections
            ))
            current_embedding_dim = next_embedding_dim

        # Final embedding layer to ensure proper size on output.
        embedding_layers.append(create_linear_block(
            options,
            current_embedding_dim,
            options.particle_emb_dim,
            options.particle_embedding_skip_connections
        ))

        return embedding_layers

    @staticmethod
    def create_particle_attention_blocks(options: Options) -> List[BasicBlock]:
        """ Create a stack of particle attention blocks keeping emb dimensions.

        """

        # Initial embedding layer to just project to our shared first dimension.
        part_att_blks = [create_particle_attention_block(options=options,
                                        input_dim=options.particle_emb_dim,
                                        output_dim=options.particle_emb_dim,
                                        n_heads=options.num_heads_in_subjet_embedding_blocks,
                                    ) for _ in range(options.num_particle_attention_blocks_in_embedding)]

        return part_att_blks

    @staticmethod
    def create_class_attention_blocks(options: Options) -> List[BasicBlock]:
        """ Create a stack of class attention blocks keeping emb dimensions.

        """

        # Initial embedding layer to just project to our shared first dimension.
        part_att_blks = [create_class_attention_block(options=options,
                                        input_dim=options.particle_emb_dim,
                                        output_dim=options.particle_emb_dim,
                                        n_heads=options.num_heads_in_subjet_embedding_blocks,
                                    ) for _ in range(options.num_class_attention_blocks_in_embedding)]

        return part_att_blks

    def forward(self, vectors: Tensor, particle_masks: Tensor) -> Tensor:
        """ Embed each particle in the subjet,
            then let particles exchange information via attention blocks.
            finally use a mlp and softmax to extract a token

        Parameters
        ----------
        vectors: [*, input_dim]
            Original vectors to embed.

        Returns
        -------
        embeddings: [*, output_dim]
            Output embeddings.
        """
        bs, N_sj, N_ptcl, N_ptcl_ftr = vectors.shape

        embeddings = vectors
        for layer in self.particle_embedding_layers:
            embeddings = layer(embeddings)

        repr = embeddings
        for blk in self.particle_attention_blocks:
            repr = blk(repr, particle_masks)


        if self.has_cls_attn_blks:
            cls_tokens = self.cls_token.expand(bs, N_sj, 1, -1)
            for blk in self.class_attention_blocks:
                cls_tokens = blk(repr, cls_tokens, particle_masks)
                x_cls = cls_tokens.squeeze(2)
            repr = x_cls
        else:
            # flatten the particle dimension to create subjet token
            repr = repr.flatten(start_dim=-2, end_dim=-1)

        repr = self.fc1(repr)
        repr = self.norm(repr)
        repr = self.act(repr)
        repr = self.fc2(repr)
        repr = self.drop(repr)

        # token = self.softmax(repr)
        token = repr

        return token
