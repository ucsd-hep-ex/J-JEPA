import torch


def main():
    # Set random seed for reproducibility
    torch.manual_seed(0)

    # Define sample dimensions
    B = 3  # Batch size (number of jets)
    N = 4  # Total number of particles per jet
    F = 4  # Number of features per particle

    # Define fixed number of context and target particles
    N_ctxt = 2  # Number of context particles per jet
    N_trgt = 2  # Number of target particles per jet

    # Ensure that N_ctxt + N_trgt == N
    assert N_ctxt + N_trgt == N, "N_ctxt + N_trgt must equal N to cover all particles."

    # Initialize p4_spatial with -1 for all entries (optional, since all particles are either context or target)
    p4_spatial = torch.full((B, N, F), -1.0)

    # Define context_masks and target_masks of shape (B, N)
    # Ensure that each jet has exactly N_ctxt context and N_trgt target particles
    # and that there is no overlap between context_masks and target_masks

    # Corrected Masks: Complementary and Non-Overlapping
    context_masks = torch.tensor(
        [
            [True, False, True, False],  # Jet 0: Particles 0 and 2 are context
            [False, True, False, True],  # Jet 1: Particles 1 and 3 are context
            [True, True, False, False],  # Jet 2: Particles 0 and 1 are context
        ]
    )

    # Since masks are complementary, target_masks can be defined as the inverse of context_masks
    target_masks = ~context_masks

    # Alternatively, define target_masks manually ensuring no overlap and complete coverage
    # target_masks = torch.tensor([
    #     [False, True, False, True],   # Jet 0: Particles 1 and 3 are target
    #     [True, False, True, False],   # Jet 1: Particles 0 and 2 are target
    #     [False, False, True, True]    # Jet 2: Particles 2 and 3 are target
    # ])

    # **Important**: Verify that masks are complementary and non-overlapping
    overlap = torch.logical_and(context_masks, target_masks)
    coverage = torch.logical_or(context_masks, target_masks)

    if overlap.any():
        overlapping_indices = overlap.nonzero(as_tuple=True)
        print(f"Overlapping Masks at indices: {overlapping_indices}")
    assert (
        not overlap.any()
    ), "Masks overlap! A particle cannot be both context and target."

    if not coverage.all():
        uncovered_indices = (~coverage).nonzero(as_tuple=True)
        print(f"Uncovered Particles at indices: {uncovered_indices}")
    assert coverage.all(), "Some particles are neither context nor target."

    # Update p4_spatial based on masks
    # Set context particles to 1
    p4_spatial[context_masks] = 1.0

    # Set target particles to 0
    p4_spatial[target_masks] = 0.0

    # Define ptcl_mask of shape (B, N)
    # This could represent additional particle properties; for testing, use binary masks
    ptcl_mask = torch.tensor(
        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],  # Jet 0  # Jet 1  # Jet 2
        dtype=torch.float,
    )

    print("=== Sample Input ===")
    print("p4_spatial (B, N, 4):")
    print(p4_spatial)
    print("\ncontext_masks (B, N):")
    print(context_masks)
    print("\ntarget_masks (B, N):")
    print(target_masks)
    print("\nptcl_mask (B, N):")
    print(ptcl_mask)

    print(f"\nNumber of context particles per jet (N_ctxt): {N_ctxt}")
    print(f"Number of target particles per jet (N_trgt): {N_trgt}")

    # Select context particles
    p4_spatial_context = p4_spatial[context_masks].view(B, N_ctxt, F)
    p4_spatial_target = p4_spatial[target_masks].view(B, N_trgt, F)

    # Select corresponding ptcl_mask
    ctxt_particle_mask = ptcl_mask[context_masks].view(B, N_ctxt)
    trgt_particle_mask = ptcl_mask[target_masks].view(B, N_trgt)

    print("\n=== Selected Context Particles ===")
    print("p4_spatial_context (B, N_ctxt, 4):")
    print(p4_spatial_context)
    print("\nctxt_particle_mask (B, N_ctxt):")
    print(ctxt_particle_mask)

    print("\n=== Selected Target Particles ===")
    print("p4_spatial_target (B, N_trgt, 4):")
    print(p4_spatial_target)
    print("\ntrgt_particle_mask (B, N_trgt):")
    print(trgt_particle_mask)

    # Verification
    # For each jet, verify that the selected particles match the masks
    for b in range(B):
        # Context verification
        ctxt_indices = context_masks[b].nonzero(as_tuple=True)[0]
        expected_context = torch.ones((N_ctxt, F))
        actual_context = p4_spatial_context[b]
        assert torch.allclose(
            expected_context, actual_context
        ), f"Context selection mismatch in jet {b}"

        # Target verification
        trgt_indices = target_masks[b].nonzero(as_tuple=True)[0]
        expected_target = torch.zeros((N_trgt, F))
        actual_target = p4_spatial_target[b]
        assert torch.allclose(
            expected_target, actual_target
        ), f"Target selection mismatch in jet {b}"

        # Particle mask verification
        expected_ctxt_mask = ptcl_mask[b, ctxt_indices]
        actual_ctxt_mask = ctxt_particle_mask[b]
        assert torch.allclose(
            expected_ctxt_mask, actual_ctxt_mask
        ), f"Context particle mask mismatch in jet {b}"

        expected_trgt_mask = ptcl_mask[b, trgt_indices]
        actual_trgt_mask = trgt_particle_mask[b]
        assert torch.allclose(
            expected_trgt_mask, actual_trgt_mask
        ), f"Target particle mask mismatch in jet {b}"

    print("\nAll tests passed successfully!")


if __name__ == "__main__":
    main()
