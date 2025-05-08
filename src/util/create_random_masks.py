import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import numpy as np
import awkward as ak
import fastjet
import vector


def get_subjets(px, py, pz, e, JET_ALGO="CA", jet_radius=0.2, return_sorted=True):
    """
    Clusters particles into subjets using the specified jet clustering algorithm and jet radius,
    then returns information about the subjets sorted by their transverse momentum (pT) in descending order.

    Each particle is represented by its momentum components (px, py, pz) and energy (e). The function
    filters out zero-momentum particles, clusters the remaining particles into jets using the specified
    jet algorithm and radius, and then retrieves each subjet's pT, eta, and phi, along with the indices
    of the original particles that constitute each subjet.

    Parameters:
    - px (np.ndarray): NumPy array containing the x-component of momentum for each particle.
    - py (np.ndarray): NumPy array containing the y-component of momentum for each particle.
    - pz (np.ndarray): NumPy array containing the z-component of momentum for each particle.
    - e (np.ndarray): NumPy array containing the energy of each particle.
    - JET_ALGO (str, optional): The jet clustering algorithm to use. Choices are "CA" (Cambridge/Aachen), "kt", and "antikt".
      The default is "CA".
    - jet_radius (float, optional): The radius parameter for the jet clustering algorithm. The default is 0.2.

    Returns:
    - List[Dict]: A list of dictionaries, one for each subjet. Each dictionary contains two keys:
        "features", mapping to another dictionary with keys "pT", "eta", and "phi" representing the subjet's
        kinematic properties, and "indices", mapping to a list of indices corresponding to the original
        particles that make up the subjet. The list is sorted by the subjets' pT in descending order.

    Example:
    >>> px = np.array([...])
    >>> py = np.array([...])
    >>> pz = np.array([...])
    >>> e = np.array([...])
    >>> subjets_info_sorted = get_subjets(px, py, pz, e, JET_ALGO="kt", jet_radius=0.2)
    >>> print(subjets_info_sorted[0])  # Access the leading subjet information
    """

    if JET_ALGO == "kt":
        JET_ALGO = fastjet.kt_algorithm
    elif JET_ALGO == "antikt":
        JET_ALGO = fastjet.antikt_algorithm
    else:  # Default to "CA" if not "kt" or "antikt"
        JET_ALGO = fastjet.cambridge_algorithm

    jetdef = fastjet.JetDefinition(JET_ALGO, jet_radius)

    # Ensure px, py, pz, and e are filtered arrays of non-zero values
    px_nonzero = px[px != 0]
    py_nonzero = py[py != 0]
    pz_nonzero = pz[pz != 0]
    e_nonzero = e[e != 0]

    jet = ak.zip(
        {
            "px": px_nonzero,
            "py": py_nonzero,
            "pz": pz_nonzero,
            "E": e_nonzero,
        },
        with_name="MomentumArray4D",
    )

    # Create PseudoJet objects for non-zero particles
    pseudojets = []
    for i in range(len(px_nonzero)):
        particle = jet[i]
        pj = fastjet.PseudoJet(
            particle.px.item(),
            particle.py.item(),
            particle.pz.item(),
            particle.E.item(),
        )
        pj.set_user_index(i)
        pseudojets.append(pj)

    cluster = fastjet.ClusterSequence(pseudojets, jetdef)

    subjets = cluster.inclusive_jets()  # Get the jets from the clustering

    subjets_info = []  # List to store dictionaries for each subjet

    for subjet in subjets:
        # Extract features
        features = {
            "pT": subjet.pt(),
            "eta": subjet.eta(),
            "phi": subjet.phi(),
            "num_ptcls": 0,
        }

        # Extract indices, sort by pT
        indices = [constituent.user_index() for constituent in subjet.constituents()]
        indices = sorted(
            indices
        )  # since the original particles were already sorted by pT
        features["num_ptcls"] = len(indices)

        # Create dictionary for the current subjet and append to the list
        subjet_dict = {"features": features, "indices": indices}
        subjets_info.append(subjet_dict)

    # subjets_info now contains the required dictionaries for each subjet
    subjets_info_sorted = subjets_info
    if return_sorted:
        subjets_info_sorted = sorted(
            subjets_info, key=lambda x: x["features"]["pT"], reverse=True
        )

    # subjets_info_sorted now contains the subjets sorted by pT in descending order
    return subjets_info_sorted
from concurrent.futures import ThreadPoolExecutor, as_completed
def create_random_masks(p4_spatial, ratio, max_targets, return_sorted=True, max_workers = None):
    """
    Creates context and target masks for a batch of jets based on the provided ratio and max_targets.

    Parameters:
    - p4_spatial: Torch tensor of shape (batch_size, total_num_particles_padded, 4) containing px, py, pz,
        and e (energy) of a batch of jets for input into get_subjets(px, py, pz, e, JET_ALGO="CA", jet_radius=0.2)
    - ratio: Float between 0 and 1, specifying the ratio of target particles to total non-padded particles.
    - max_targets: Integer, maximum number of target particles each jet should have.

    Returns:
    - context_masks: Torch tensor of shape (batch_size, total_num_particles_padded), with 1s for context particles and 0s elsewhere.
    - target_masks: Torch tensor of shape (batch_size, total_num_particles_padded), with 1s for target particles and 0s elsewhere.
    """
    p4_spatial = p4_spatial.transpose(1, 2)
    batch_size, _, total_num_particles_padded = p4_spatial.shape

    # prepare per-jet inputs
    jobs = [
        (p4_spatial[i], ratio, max_targets, return_sorted)
        for i in range(batch_size)
    ]
    
    context_list = [None] * batch_size
    target_list = [None] * batch_size

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(_make_masks_for_jet, *job): idx
            for idx, job in enumerate(jobs)
        }
        # store result
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            context_list[idx], target_list[idx] = future.result()

    # stack back into torch tensors
    context_masks = torch.stack(context_list).bool()
    target_masks = torch.stack(target_list).bool()
    return context_masks, target_masks

def _make_masks_for_jet(p4_jet, ratio, max_targets, return_sorted):
    # p4_jet shape: (4, N_padded)
    px, py, pz, e = p4_jet
    N_non_padded = int(torch.count_nonzero(e).item())
    subjets = get_subjets(px, py, pz, e,
                         JET_ALGO="CA",
                         jet_radius=0.2,
                         return_sorted=return_sorted)
    context_mask, target_mask = create_random_masks_single(
        subjets,
        N_non_padded,
        p4_jet.shape[-1],
        ratio,
        max_targets
    )
    return context_mask, target_mask

def create_random_masks_single(
    subjets_info_sorted, N_non_padded, total_num_particles_padded, ratio, max_targets
):
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
        indices = subjet["indices"]  # Indices of particles in the subjet
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
                torch.multinomial(
                    torch.ones(num_padded_available),
                    num_padded_to_select,
                    replacement=False,
                )
            )
            selected_indices.extend(
                padded_indices[i] for i in additional_padded_indices.tolist()
            )

    # Set target_mask for selected indices
    target_mask[selected_indices] = 1.0

    # Context mask is the complement of target mask
    context_mask = 1.0 - target_mask

    return context_mask, target_mask


def plot_particles(
    eta,
    phi,
    pT,
    context_mask,
    target_mask,
    valid_mask,
    jet_idx,
    save_dir,
    min_size=10,
    max_size=200,
):
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
        pT_normalized = (
            np.ones_like(pT) * 0.5
        )  # Default normalization if pT is constant

    sizes = min_size + pT_normalized * (max_size - min_size)

    # Plot context particles
    context_indices = context_mask == 1
    plt.scatter(
        eta[context_indices],
        phi[context_indices],
        s=sizes[context_indices],
        c="blue",
        alpha=0.7,
        label="Context",
    )

    # Plot target particles
    target_indices = target_mask == 1
    plt.scatter(
        eta[target_indices],
        phi[target_indices],
        s=sizes[target_indices],
        c="red",
        alpha=0.7,
        label="Target",
    )

    # Set plot labels and title
    plt.xlabel("Eta")
    plt.ylabel("Phi")
    plt.title(f"Particle Distribution: Context vs Target for Jet {jet_idx}")

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
