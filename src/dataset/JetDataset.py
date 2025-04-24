from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# HDF5 handling
import h5py


def check_raw_data(subjets_data, jet_index=0):
    print(f"\n--- Checking Raw Data for Jet {jet_index} ---")
    print(f"Number of subjets: {subjets_data['subjet_pt'].shape[-1]}")
    print(f"Subjet features: {list(subjets_data.keys())}")
    print(f"Number of indices per subjet: {subjets_data['particle_indices'].shape[-1]}")
    print(f"Sample subjet feature values: ")
    for name in subjets_data.keys():
        print(name, subjets_data[name][jet_index])


def check_model_input(model_input, batch_index=0):
    print(f"\n--- Checking Model Input for Batch Item {batch_index} ---")
    print(f"Input shape: {model_input.shape}")
    if len(model_input.shape) == 3:
        batch_size, num_subjets, feature_dim = model_input.shape
        print(f"Batch size: {batch_size}")
        print(f"Number of subjets: {num_subjets}")
        print(f"Feature dimension: {feature_dim}")
        print("\nFirst few values of the first subjet:")
        print(model_input[0, 0, :10])
    else:
        print("Unexpected shape for model input")


def inspect_indices(subjets, num_samples=5):
    print("\n--- Inspecting Subjet Indices ---")
    for i in range(min(num_samples, len(subjets["subjet_eta"]))):
        print(f"\nSubjet {i}:")
        print(f"  pT: {subjets['subjet_pt'][i]:.2f}")
        print(f"  eta: {subjets['subjet_eta'][i]:.2f}")
        print(f"  phi: {subjets['subjet_phi'][i]:.2f}")
        print(f"  num_ptcls: {subjets['subjet_num_ptcls'][i]}")
        print(
            f"  Indices: {subjets['particle_indices'][i, :10]}..."
        )  # Print first 10 indices
        print(f"  Number of indices: {len(subjets['particle_indices'][i])}")


class DimensionCheckLayer(torch.nn.Module):
    def __init__(self, name, expected_dims):
        super().__init__()
        self.name = name
        self.expected_dims = expected_dims

    def forward(self, x):
        if len(x.shape) != self.expected_dims:
            print(
                f"WARNING: {self.name} has {len(x.shape)} dimensions, expected {self.expected_dims}"
            )
        return x


class JetDataset(Dataset):
    """JetDataset class for loading and processing jet data from HDF5 files.

    Args:
        Dataset (_type_): _description_
    """

    def __init__(
        self,
        file_path,
        num_jets=None,
        transform=None,
        config=None,
        debug=False,
        labels=False,
    ):
        print(f"Initializing JetDataset with file: {file_path}")

        with h5py.File(file_path, "r") as hdf:
            print("Loading features and subjets from HDF5 file")
            self.particles = {
                name: hdf["particles"][name][:] for name in hdf["particles"]
            }
            self.subjets = {name: hdf["subjets"][name][:] for name in hdf["subjets"]}
            self.labels = hdf["labels"][:]

        # calculate subjet energy from E = pT * cosh(eta)
        self.subjets["subjet_E"] = self.subjets["subjet_pt"] * np.cosh(
            self.subjets["subjet_eta"]
        )
        self.transform = transform
        self.config = config
        self.debug = debug
        self.return_labels = labels
        if self.debug:
            print(f"Raw dataset size: {self.labels.shape[0]} jets")
            print("Particle features shapes")
            for name in self.particles.keys():
                print(f"shape of {name}: {self.particles[name].shape}")
            print("Subjet features shapes")
            for name in self.subjets.keys():
                print(f"shape of {name}: {self.subjets[name].shape}")

        # Normalize each particle feature
        # For each key, the array is of shape (num_jets, num_particles)
        # We compute the mean and std over both dimensions (axes 0 and 1)
        for key in self.particles:
            data = self.particles[key]
            mean = np.mean(data, axis=(0, 1), keepdims=True)
            std = np.std(data, axis=(0, 1), keepdims=True)
            # Prevent division by zero: if std == 0, set it to 1.0
            std[std == 0] = 1.0
            self.particles[key] = (data - mean) / std

        self.filter_good_jets()

        if num_jets is not None:
            if self.debug:
                print(f"Reducing number of jets to: {num_jets}")
            for name in self.particles.keys():
                self.particles[name] = self.particles[name][:num_jets]
                if self.debug:
                    print(f"shape of {name}: {self.particles[name].shape}")
            for name in self.subjets.keys():
                self.subjets[name] = self.subjets[name][:num_jets]
                if self.debug:
                    print(f"shape of {name}: {self.subjets[name].shape}")
        self.labels = self.labels[:num_jets]
        print(f"Final dataset size: {self.labels.shape[0]} jets")
        if self.return_labels:
            print(
                "__getitem__ returns (x, particle_features, subjets, indices, subjet_mask, particle_mask, labels)"
            )
        else:
            print(
                "__getitem__ returns (x, particle_features, subjets, indices, subjet_mask, particle_mask)"
            )

    def filter_good_jets(self):
        """
        Filters jets based on the number of real subjets and updates the dataset to only include 'good' jets.

        Modifies:
        - Updates self.particles, self.subjets, and self.labels to only include jets with at least 10 real subjets.

        Prints:
        - The shape of particles and subjets after filtering.
        - The total number of good jets remaining after filtering.
        """
        if self.debug:
            print("Filtering good jets...")
        good_jet_indices = []

        for jet_idx in range(self.labels.shape[0]):
            num_real_subjets = self.get_num_real_subjets(jet_idx)
            # print(f"Jet {jet_idx}: {num_real_subjets} real subjets")
            if num_real_subjets >= 10:
                good_jet_indices.append(jet_idx)

        for name in self.particles.keys():
            self.particles[name] = self.particles[name][good_jet_indices]
            if self.debug:
                print(f"shape of {name}: {self.particles[name].shape}")
        for name in self.subjets.keys():
            self.subjets[name] = self.subjets[name][good_jet_indices]
            if self.debug:
                print(f"shape of {name}: {self.subjets[name].shape}")
        self.labels = self.labels[good_jet_indices]
        print(f"Filtered to {self.labels.shape[0]} good jets")

    def get_num_real_subjets(self, jet_idx):
        """
        Returns the number of real subjets in a given jet.
        Parameters:
        - jet_idx: the index of the jet to check
        Returns:
        - int: The number of real subjets.
        Example:
        >>> self.subjets['subjet_num_ptcls'][jet_idx] = [15, 10, 10, 9, 8, 0, 0, 0, ..., 0]
        >>> get_num_real_subjets(jet_idx)
        5
        """
        return sum(
            1
            for subjet_num_ptcls in self.subjets["subjet_num_ptcls"][jet_idx]
            if subjet_num_ptcls > 0
        )

    def __len__(self):
        """
        Returns the length of the jets in the dataset.
        Returns:
            int: Number of jets in the dataset.
        """

        return self.labels.shape[0]

    def __getitem__(self, idx):
        """
        Retrievs the features and subjets for a given index and processes them.
        Args:
            idx (int): The index of the item to fetch.
        Returns:
            tuple: A tuple containing the following elements:
                - features (numpy.ndarray): The normalized particle features of the item.
                - subjets (numpy.ndarray): The subjets data of the item.
                - subjet_mask (numpy.ndarray): The subjet mask data of the item.
                - particle_mask (numpy.ndarray): The particle mask data of the item.
        """
        if self.debug:
            check_raw_data(self.subjets, jet_index=idx)  # -- Debug statment

        if self.debug:
            print(f"\nFetching item {idx} from dataset")
        particle_feature_names = ["part_deta", "part_dphi", "part_pt_log", "part_e_log"]
        particle_features = np.stack(
            [self.particles[name][idx] for name in particle_feature_names]
        )
        particle_features = particle_features.transpose()
        if self.debug:
            print("particle features shape", particle_features.shape)
        subjets = {name: self.subjets[name][idx] for name in self.subjets.keys()}
        if self.debug:
            print("subjets", subjets)

        if self.debug:
            inspect_indices(subjets)  # -- Debug statment

        subjets, indices, subjet_mask, particle_mask = self.process_subjets(subjets)

        # feature_names = ['pT', 'eta', 'phi']
        # print("Normalizing features")
        # features = normalize_features(features, feature_names, self.config, jet_type='Jets')

        if self.transform:
            print("Applying transform to features")
            features = self.transform(features)

        # print(f"Returning data for item {idx}")
        # print(f"Features shape: {features.shape}")
        # print(f"Subjets shape: {subjets.shape}")
        # print(f"Subjet mask shape: {subjet_mask.shape}")
        # print(f"Particle mask shape: {particle_mask.shape}")
        # print(f"Subjets data: {subjets}")
        # print(f"Subjet mask data: {subjet_mask}")
        # print(f"Particle mask data: {particle_mask}")

        particle_features = torch.from_numpy(particle_features)
        # # Need x to be of shape (N_subjets, N_part, N_part_ftr)
        # # particle_features: (N_part, N_part_ftr)
        N_part_per_jet, N_part_ftr = particle_features.shape  # Example: 128, 4
        N_part_per_subjet = indices.shape[-1]
        N_subjets = subjets.shape[0]  # Example: 20
        num_real_ptcls = subjets[:, -1]

        # Prepare x tensor
        x = torch.zeros((N_subjets, N_part_per_subjet, N_part_ftr))

        # Create a mask for valid indices
        mask = torch.arange(N_part_per_subjet).expand(
            N_subjets, N_part_per_subjet
        ) < num_real_ptcls.unsqueeze(1)

        # Gather indices within bounds
        all_real_indices = indices[:, :N_part_per_subjet].long()

        valid_features = particle_features[all_real_indices]
        # print("valid features", valid_features.shape)

        # Expand mask to feature dimensions
        expanded_mask = mask.unsqueeze(-1).expand(-1, -1, N_part_ftr)
        # print("expanded mask", expanded_mask.shape)

        # Apply expanded mask
        x[expanded_mask] = valid_features[expanded_mask].to(torch.float)

        return_tuple = (
            x,
            particle_features,
            subjets,
            indices,
            subjet_mask,
            particle_mask,
        )
        if self.return_labels:
            return_tuple = (
                x,
                particle_features,
                subjets,
                indices,
                subjet_mask,
                particle_mask,
                self.labels[idx],
            )

        return return_tuple

    def process_subjets(self, subjets):
        """
        Processes subjets to create tensor representations and masks.

        Parameters:
        - subjets (dictionary): a dictionary with keys ['subjet_pt', 'subjet_eta', 'subjet_phi', 'subjet_E, 'subjet_num_ptcls', 'particle_indices'].
            each item is a numpy array of shape (N_subjets, ) or (N_subjets, N_particles) for the last key

        Returns:
        - tuple: (subjets, subjet_mask, particle_mask)
            where `subjets` is the tensor representation of subjets,
            `subjet_mask` is the mask for subjets,
            and `particle_mask` is the mask for particles.
        """
        if self.debug:
            print("Processing subjets")

        max_len = subjets["particle_indices"].shape[-1]
        if self.debug:
            print(f"Max length of indices in subjets: {max_len}")
        subjet_mask = torch.tensor(
            subjets["subjet_num_ptcls"] != 0, dtype=torch.float32
        )

        particle_mask = torch.tensor(
            subjets["particle_indices"] != -1, dtype=torch.float32
        )

        features = torch.stack(
            [
                torch.from_numpy(subjets[name])
                for name in [
                    "subjet_pt",
                    "subjet_eta",
                    "subjet_phi",
                    "subjet_E",
                    "subjet_num_ptcls",
                ]
            ],
            dim=0,
        )
        features = features.t()

        indices = torch.from_numpy(subjets["particle_indices"])
        if self.debug:
            print(f"Subjet indices shape: {indices.shape}")  # (N_subjets, N_particles)

            print(f"Final processed subjets shape: {features.shape}")  # (N_subjets, 4)
            print(
                f"Final subjet mask shape: {subjet_mask.shape}"
            )  # (N_subjets, ) same as 'subjet_num_ptcls'
            print(
                f"Final particle mask shape: {particle_mask.shape}"
            )  # (N_subjets, N_particles) same as 'particle_indices'

        return features, indices, subjet_mask, particle_mask
