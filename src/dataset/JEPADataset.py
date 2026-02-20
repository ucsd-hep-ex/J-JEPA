from torch.utils.data import Dataset
import torch
import h5py
import os
import numpy as np

class JEPADataset(Dataset):
    def __init__(self, directory_path, num_jets=None):
        # Initialize empty lists to store the datasets
        x_list, particle_features_list, subjets_list = [], [], []
        particle_indices_list, subjet_mask_list, particle_mask_list = [], [], []

        # Loop through each file in the directory
        for filename in os.listdir(directory_path):
            if filename.endswith(".hdf5") or filename.endswith(".h5"):
                file_path = os.path.join(directory_path, filename)
                with h5py.File(file_path, 'r') as file:
                    # Append each dataset to the corresponding list
                    print(f"Loading {filename}")
                    x_list.append(file['x'][:])
                    particle_features_list.append(file['particle_features'][:])
                    subjets_list.append(file['subjets'][:])
                    particle_indices_list.append(file['particle_indices'][:])
                    subjet_mask_list.append(file['subjet_mask'][:])
                    particle_mask_list.append(file['particle_mask'][:])
                print(f"Loaded {filename}")

        # Concatenate all datasets from all files
        self.x = np.concatenate(x_list, axis=0)
        self.particle_features = np.concatenate(particle_features_list, axis=0)
        self.subjets = np.concatenate(subjets_list, axis=0)
        self.particle_indices = np.concatenate(particle_indices_list, axis=0)
        self.subjet_mask = np.concatenate(subjet_mask_list, axis=0)
        self.particle_mask = np.concatenate(particle_mask_list, axis=0)

        if num_jets:
            self.x = self.x[:num_jets]
            self.particle_features = self.particle_features[:num_jets]
            self.subjets = self.subjets[:num_jets]
            self.particle_indices = self.particle_indices[:num_jets]
            self.subjet_mask = self.subjet_mask[:num_jets]
            self.particle_mask = self.particle_mask[:num_jets]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return (self.x[idx], self.particle_features[idx], self.subjets[idx],
                self.particle_indices[idx], self.subjet_mask[idx], self.particle_mask[idx])