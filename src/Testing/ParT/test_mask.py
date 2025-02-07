import torch

if __name__ == "__main__":
    # Sample data
    B = 2  # Batch size
    N = 4  # Sequence length
    emb_dim = 3  # Embedding dimension

    # Create a sample output tensor
    output = torch.arange(B * N * emb_dim).view(B, N, emb_dim).float()
    print("Output shape:\n", output.shape)
    print("Output:\n", output)

    # Create a sample split_mask with varying True values
    split_mask = torch.tensor(
        [
            [True, False, True, False],  # Batch 0: selects positions 0 and 2
            [False, True, False, True],  # Batch 1: selects positions 1 and 3
        ]
    )
    print("Split Mask:\n", split_mask)

    # Apply the optimized approach
    indices = split_mask.nonzero(as_tuple=False)
    batch_indices = indices[:, 0]
    seq_indices = indices[:, 1]

    selected_particles = output[batch_indices, seq_indices, :]

    lengths = split_mask.sum(dim=1)
    cum_lengths = torch.cat([torch.tensor([0]), lengths.cumsum(0)[:-1]])

    positions_in_batch = torch.arange(lengths.sum()) - torch.repeat_interleave(
        cum_lengths, lengths
    )

    max_length = lengths.max().item()

    selected_particles_padded = torch.zeros(B, max_length, emb_dim)
    selected_particles_padded[batch_indices, positions_in_batch] = selected_particles

    mask = torch.arange(max_length).unsqueeze(0).expand(
        B, max_length
    ) < lengths.unsqueeze(1)

    # Output the results
    print("Selected Particles Padded:\n", selected_particles_padded)
    print("Selected Particles Padded shape:\n", selected_particles_padded.shape)
    print("Mask:\n", mask)
