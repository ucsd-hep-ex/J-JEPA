from typing import List

import torch
from torch import Tensor, nn

from src.layers.linear_block.basic_block import BasicBlock
from src.layers.linear_block import create_linear_block

from src.options import Options


def get_1d_sincos_pos_emb(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (B,S,1)
    out: (B, S, D)
    """
    assert embed_dim % 2 == 0

    omega = torch.arange(embed_dim // 2, device=pos.device, dtype=torch.float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    out = torch.einsum("bs,d->bsd", pos, omega)  # (B, S, D/2), outer product

    emb_sin = torch.sin(out)  # (B, S, D/2)
    emb_cos = torch.cos(out)  # (B, S, D/2)

    emb = torch.cat([emb_sin, emb_cos], axis=2)  # (B, S, D)
    return emb


def create_space_pos_emb_fn(emb_dim):
    """
    Input:
        emb_dim: Integer
    Return:
        a function that calculate the positional embeding by
        subjet eta and phi
    """

    def calc_pos_emb(subjet_ftrs):
        """
        Input:
            subjets_ftrs: torch tensor of shape (bs, N_subjets, N_sj_ftrs)
                last dimension: [pt, eta, phi, E]
        """
        sj_eta = subjet_ftrs[:, :, 1]
        sj_phi = subjet_ftrs[:, :, 2]

        # process phi to impose physical distance
        sj_phi_star = torch.sin(sj_phi / 2)

        # shift eta to avoid negative positions
        sj_eta += 3

        # calculate embedding
        emb_eta = get_1d_sincos_pos_emb(emb_dim // 2, sj_eta)
        emb_phi_star = get_1d_sincos_pos_emb(emb_dim // 2, sj_phi_star)

        # print(emb_phi_star.shape)
        emb = torch.cat([emb_eta, emb_phi_star], axis=2)

        return emb

    return calc_pos_emb

def create_phase_space_pos_emb_fn(emb_dim):
    """
    Input:
        emb_dim: Integer
    Return:
        a function that calculate the positional embeding by
        subjet eta and phi
    """

    def calc_pos_emb(subjet_ftrs):
        """
        Input:
            subjets_ftrs: torch tensor of shape (bs, N_subjets, N_sj_ftrs)
                last dimension: [pt, eta, phi, E]
        """
        sj_pt = subjet_ftrs[:, :, 0]
        sj_eta = subjet_ftrs[:, :, 1]
        sj_phi = subjet_ftrs[:, :, 2]
        sj_E = subjet_ftrs[:, :, 3]

        # process phi to impose physical distance
        sj_phi_star = torch.sin(sj_phi / 2)

        # shift eta to avoid negative positions
        sj_eta += 3

        # calculate embedding
        emb_pt = get_1d_sincos_pos_emb(emb_dim // 4, sj_pt)
        emb_eta = get_1d_sincos_pos_emb(emb_dim // 4, sj_eta)
        emb_phi_star = get_1d_sincos_pos_emb(emb_dim // 4, sj_phi_star)
        emb_E = get_1d_sincos_pos_emb(emb_dim // 4, sj_E)

        # print(emb_phi_star.shape)
        emb = torch.cat([emb_pt, emb_eta, emb_phi_star, emb_E], axis=2)

        return emb

    return calc_pos_emb

class LearnableEmbeddingStacks(nn.ModuleList):
    def __init__(self, input_dim: int, output_dim: int):
        super(LearnableEmbeddingStacks, self).__init__()

        self.embedding_layers = nn.ModuleList(self.create_embedding_layers(input_dim, output_dim))

    @staticmethod
    def create_embedding_layers(input_dim: int, output_dim: int) -> List[BasicBlock]:
        """ Create a stack of linear layer with increasing emb dimensions.

        Each emb layer will have double the dimensions as the previous, beginning with the
        size of the feature-space and ending with the emb_dim specified in options.
        """

        # use a default option to recycle code
        # the default option should use a gelu act
        options = Options()

        if input_dim <= output_dim:
            # Initial embedding layer to just project to our shared first dimension.
            embedding_layers = [create_linear_block(
                options,
                input_dim,
                input_dim * 2,
                None
            )]
            current_embedding_dim = input_dim * 2

            # Keep doubling dimensions until we reach the desired latent dimension.
            for i in range(100):
                next_embedding_dim = 2 * current_embedding_dim
                if next_embedding_dim >= output_dim:
                    break

                embedding_layers.append(create_linear_block(
                    options,
                    current_embedding_dim,
                    next_embedding_dim,
                    True,
                ))
                current_embedding_dim = next_embedding_dim
        else:
            # Initial embedding layer to just project to our shared first dimension.
            embedding_layers = [create_linear_block(
                options,
                input_dim,
                input_dim // 2,
                None
            )]
            current_embedding_dim = input_dim // 2

            # Keep doubling dimensions until we reach the desired latent dimension.
            for i in range(100):
                next_embedding_dim = current_embedding_dim // 2
                if next_embedding_dim <= output_dim:
                    break

                embedding_layers.append(create_linear_block(
                    options,
                    current_embedding_dim,
                    next_embedding_dim,
                    True,
                ))
                current_embedding_dim = next_embedding_dim

        # Final embedding layer to ensure proper size on output.
        embedding_layers.append(create_linear_block(
            options,
            current_embedding_dim,
            output_dim,
            True
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

def create_learnable_space_emb_fn(emb_dim):
    """
    Input:
        emb_dim: Integer
    Return:
        a function that calculate the positional embeding by
        subjet eta and phi
    """

    eta_emb_layers = LearnableEmbeddingStacks(input_dim=1, output_dim=emb_dim // 4)
    phi_emb_layers = LearnableEmbeddingStacks(input_dim=1, output_dim=emb_dim // 4)
    eta_phi_low_level_emb_layers = LearnableEmbeddingStacks(input_dim=2, output_dim=emb_dim // 4)
    eta_phi_high_level_emb_layers = LearnableEmbeddingStacks(input_dim=emb_dim // 2, output_dim=emb_dim // 4)

    print("Creating learnable spacial embedding")

    def calc_pos_emb(subjet_ftrs):
        """
        Input:
            subjets_ftrs: torch tensor of shape (bs, N_subjets, N_sj_ftrs)
                last dimension: [pt, eta, phi, E]
        """
        sj_eta = subjet_ftrs[:, :, 1].unsqueeze(-1)
        sj_phi = subjet_ftrs[:, :, 2].unsqueeze(-1)

        # process phi to impose physical distance
        sj_phi_star = torch.sin(sj_phi / 2)

        # shift eta to avoid negative positions
        sj_eta += 3

        # calculate embedding
        emb_phi = phi_emb_layers(sj_phi_star)
        emb_eta = eta_emb_layers(sj_eta)
        emb_low_level = eta_phi_low_level_emb_layers(
                            torch.cat([sj_phi_star, sj_eta], axis=2)
                        )
        emb_high_level = eta_phi_high_level_emb_layers(
                            torch.cat([emb_phi, emb_eta], axis=2)
                        )

        # print(emb_phi_star.shape)
        emb = torch.cat([emb_phi, emb_eta, emb_low_level, emb_high_level], axis=2)

        return emb

    return calc_pos_emb
