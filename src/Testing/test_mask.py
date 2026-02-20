import torch


def main():
    # Set random seed for reproducibility
    torch.manual_seed(0)

    # Define sample dimensions
    B = 3  # Batch size (number of jets)
    N = 6  # Total number of particles per jet
    F = 4  # Number of features per particle

    # Define fixed number of context and target particles
    N_ctxt = 4  # Number of context particles per jet
    N_trgt = 2  # Number of target particles per jet

    # Ensure that N_ctxt + N_trgt == N
    assert N_ctxt + N_trgt == N, "N_ctxt + N_trgt must equal N to cover all particles."

    # Generate random particle data
    particles = torch.randn(B, N, F)  # Shape: [3, 6, 4]
    print("particles", particles)

    # Split the particles into context and target particles
    context_masks = torch.tensor(
        [[1, 1, 1, 1, 0, 0], [0, 1, 1, 0, 1, 1], [1, 0, 0, 1, 1, 1]]
    )
    target_masks = 1 - context_masks  # Shape: [3, 6]

    context_masks = context_masks.bool()
    target_masks = target_masks.bool()

    context_masks = context_masks.unsqueeze(-1)  # Shape: [3, 6, 1]

    # Broadcast the mask across the channel dimension
    context_masks = context_masks.expand(-1, -1, 4)  # Shape: [3, 6, 4]
    print("context masks", context_masks)

    target_masks = target_masks.unsqueeze(-1)  # Shape: [3, 6, 1]

    # Broadcast the mask across the channel dimension
    target_masks = target_masks.expand(-1, -1, 4)  # Shape: [3, 6, 4]
    print("target masks", target_masks)

    # Apply the masks to the particles
    print("particles[context_masks]", particles[context_masks])
    print("particles[target_masks]", particles[target_masks])
    print(
        f"particles shape: {particles.shape}"
    )  # Expected: [B, C, D, ...] (e.g., [3, 4, 128])
    print(
        f"context_masks shape: {context_masks.shape}"
    )  # Expected: [B, ...] (e.g., [3, 128])
    print(f"Number of True in context_masks: {context_masks.sum().item()}")

    p4_spatial_context = particles[context_masks].view(B, N_ctxt, 4)
    p4_spatial_target = particles[target_masks].view(B, N_trgt, 4)

    print("p4_spatial_context", p4_spatial_context)
    print("p4_spatial_target", p4_spatial_target)


if __name__ == "__main__":
    main()
