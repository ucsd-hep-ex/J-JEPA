from torch.utils.data import Dataset
import torch
import h5py

class JEPADataset(Dataset):
    def __init__(self, file_path, num_jets=None):
        self.num_jets = num_jets if num_jets else -1
        with h5py.File(file_path, 'r') as self.file:
            print(f"Loading file from {file_path}")
            self.x = self.file['x'][:self.num_jets]  # Load entire dataset into memory
            self.particle_features = self.file['particle_features'][:self.num_jets]
            self.subjets = self.file['subjets'][:self.num_jets]
            self.particle_indices = self.file['particle_indices'][:self.num_jets]
            self.subjet_mask = self.file['subjet_mask'][:self.num_jets]
            self.particle_mask = self.file['particle_mask'][:self.num_jets]
        print(f"number of jets: {len(self)}")

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        # Directly return the data from memory
        return (self.x[idx], self.particle_features[idx], self.subjets[idx],
                self.particle_indices[idx], self.subjet_mask[idx], self.particle_mask[idx])