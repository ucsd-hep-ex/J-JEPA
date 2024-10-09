import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

def create_random_masks(p4_spatial, ratio, max_targets):
    """
    Creates context and target masks for a batch of jets based on the provided ratio and max_targets.

    Parameters:
    - p4_spatial: Torch tensor of shape (batch_size, 4, total_num_particles_padded) containing px, py, pz, 
        and e (energy) of a batch of jets for input into get_subjets(px, py, pz, e, JET_ALGO="CA", jet_radius=0.2)
    - ratio: Float between 0 and 1, specifying the ratio of target particles to total non-padded particles.
    - max_targets: Integer, maximum number of target particles each jet should have.

    Returns:
    - context_masks: Torch tensor of shape (batch_size, total_num_particles_padded), with 1s for context particles and 0s elsewhere.
    - target_masks: Torch tensor of shape (batch_size, total_num_particles_padded), with 1s for target particles and 0s elsewhere.
    """
    batch_size, _, total_num_particles_padded = p4_spatial.shape

    context_masks = torch.zeros((batch_size, total_num_particles_padded), dtype=torch.float32)
    target_masks = torch.zeros((batch_size, total_num_particles_padded), dtype=torch.float32)

    for i in tqdm(range(batch_size)):
        # Extract px, py, pz, e for this jet
        px = p4_spatial[i, 0, :]  # Shape: (total_num_particles_padded,)
        py = p4_spatial[i, 1, :]
        pz = p4_spatial[i, 2, :]
        e  = p4_spatial[i, 3, :]

        # Get N_non_padded by counting non-zero entries in e
        N_non_padded = torch.count_nonzero(e)

        # Call get_subjets
        subjets_info_sorted = get_subjets(px, py, pz, e, JET_ALGO="CA", jet_radius=0.2)

        # Create masks for this jet
        context_mask, target_mask = create_random_masks_single(
            subjets_info_sorted, N_non_padded.item(), total_num_particles_padded, ratio, max_targets
        )

        # Assign to the batch masks
        context_masks[i] = context_mask
        target_masks[i] = target_mask

    return context_masks, target_masks

def create_random_masks_single(subjets_info_sorted, N_non_padded, total_num_particles_padded, ratio, max_targets):
    """
    Creates context and target masks for a single jet.

    Parameters:
    - subjets_info_sorted: List of dictionaries, each representing a subjet as returned by get_subjets().
    - N_non_padded: Integer, number of non-padded particles in the jet.
    - total_num_particles_padded: Integer, total number of particles per jet after padding (e.g., 128).
    - ratio: Float between 0 and 1, specifying the ratio of target particles to total non-padded particles.
    - max_targets: Integer, maximum number of target particles the jet should have.

    Returns:
    - context_mask: Torch tensor of shape (total_num_particles_padded,), with 1s for context particles and 0s elsewhere.
    - target_mask: Torch tensor of shape (total_num_particles_padded,), with 1s for target particles and 0s elsewhere.
    """
    # Calculate the initial number of target particles based on the ratio
    num_targets_real = min(int(round(ratio * N_non_padded)), max_targets)
    num_targets = max_targets

    target_mask = torch.zeros(total_num_particles_padded, dtype=torch.float32)
    selected_indices = []

    # Collect particles from subjets starting from the highest pT subjet
    for subjet in subjets_info_sorted:
        indices = subjet['indices']  # Indices of particles in the subjet
        for index in indices:
            if index not in selected_indices:
                selected_indices.append(index)
                if len(selected_indices) >= num_targets_real:
                    break
        if len(selected_indices) >= num_targets_real:
            break

    # If not enough target particles have been selected, select from padded particles to reach max_targets
    if len(selected_indices) < num_targets:
        num_needed = num_targets - len(selected_indices)
        padded_indices = list(range(N_non_padded, total_num_particles_padded))
        num_padded_available = len(padded_indices)
        num_padded_to_select = min(num_needed, num_padded_available)
        if num_padded_to_select > 0:
            additional_padded_indices = torch.tensor(
                torch.multinomial(torch.ones(num_padded_available), num_padded_to_select, replacement=False)
            )
            selected_indices.extend(padded_indices[i] for i in additional_padded_indices.tolist())

    # Set target_mask for selected indices
    target_mask[selected_indices] = 1.0

    # Context mask is the complement of target mask
    context_mask = 1.0 - target_mask

    return context_mask, target_mask

def plot_particles(eta, phi, pT, context_mask, target_mask, valid_mask, jet_idx, save_dir, min_size=10, max_size=200):
    """
    Plot the eta and phi of particles, color coding them based on context/target masks,
    and vary the size of each point based on the transverse momentum (pT).
    
    Parameters:
    - eta: Torch tensor of pseudorapidity values.
    - phi: Torch tensor of azimuthal angle values.
    - pT: Torch tensor of transverse momentum values.
    - context_mask: Torch tensor of shape (total_num_particles,) with 1s for context particles and 0s elsewhere.
    - target_mask: Torch tensor of shape (total_num_particles,) with 1s for target particles and 0s elsewhere.
    - valid_mask: Torch tensor of shape (total_num_particles,) with 1s for valid (non-padded) particles and 0s elsewhere.
    - min_size: Minimum size for the points in the scatter plot.
    - max_size: Maximum size for the points in the scatter plot.
    """
    os.makedirs(save_dir, exist_ok=True)
    # Ensure all tensors are on CPU and convert to NumPy arrays
    eta = eta.cpu().numpy()
    phi = phi.cpu().numpy()
    pT = pT.cpu().numpy()
    context_mask = context_mask.cpu().numpy()
    target_mask = target_mask.cpu().numpy()
    valid_mask = valid_mask.cpu().numpy()

    # Apply valid mask to filter out padded particles
    valid_indices = valid_mask.astype(bool)
    eta = eta[valid_indices]
    phi = phi[valid_indices]
    pT = pT[valid_indices]
    context_mask = context_mask[valid_indices]
    target_mask = target_mask[valid_indices]

    # Normalize pT for point sizes
    pT_min = pT.min()
    pT_max = pT.max()
    if pT_max > pT_min:
        pT_normalized = (pT - pT_min) / (pT_max - pT_min)
    else:
        pT_normalized = np.ones_like(pT) * 0.5  # Default normalization if pT is constant

    sizes = min_size + pT_normalized * (max_size - min_size)

    # Plot context particles
    context_indices = (context_mask == 1)
    plt.scatter(
        eta[context_indices],
        phi[context_indices],
        s=sizes[context_indices],
        c='blue',
        alpha=0.7,
        label='Context'
    )

    # Plot target particles
    target_indices = (target_mask == 1)
    plt.scatter(
        eta[target_indices],
        phi[target_indices],
        s=sizes[target_indices],
        c='red',
        alpha=0.7,
        label='Target'
    )

    # Set plot labels and title
    plt.xlabel('Eta')
    plt.ylabel('Phi')
    plt.title(f'Particle Distribution: Context vs Target for Jet {jet_idx}')

    # Adjust axes limits for better visualization
    plt.xlim(eta.min() - 0.1, eta.max() + 0.1)
    plt.ylim(phi.min() - 0.1, phi.max() + 0.1)

    # Add legend
    plt.legend()

    # Show grid
    plt.grid(False)

    plt.savefig(f"{save_dir}/jet_{jet_idx}")

    # Display the plot
    plt.show()
