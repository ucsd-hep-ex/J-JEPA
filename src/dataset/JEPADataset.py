from torch.utils.data import Dataset
import torch
import h5py
import os
import numpy as np


class JEPADataset(Dataset):
    def __init__(self, directory_path, num_jets=None):
        self.directory_path = directory_path
        self.file_list = []
        self.file_sizes = []
        self.cumulative_sizes = []
        self.total_samples = 0
        self.num_jets = num_jets

        print(
            f"Initialized JEPADataset with {self.total_samples} samples from {len(self.file_list)} files."
        )

        # Gather all HDF5 files in the directory
        for filename in os.listdir(directory_path):
            if filename.endswith(".hdf5") or filename.endswith(".h5"):
                file_path = os.path.join(directory_path, filename)
                with h5py.File(file_path, "r") as file:
                    num_samples = file["x"].shape[0]  # Adjust 'x' to your dataset's key
                self.file_list.append(file_path)
                self.file_sizes.append(num_samples)
                self.total_samples += num_samples
                self.cumulative_sizes.append(self.total_samples)

        # If num_jets is specified, adjust the total number of samples and cumulative sizes
        if num_jets is not None:
            self.total_samples = min(self.total_samples, num_jets)

            # Adjust cumulative sizes to reflect num_jets
            adjusted_cumulative_sizes = []
            cumulative = 0
            for size in self.file_sizes:
                if cumulative + size >= num_jets:
                    adjusted_cumulative_sizes.append(num_jets)
                    break
                else:
                    cumulative += size
                    adjusted_cumulative_sizes.append(cumulative)
            self.cumulative_sizes = adjusted_cumulative_sizes
            # Adjust file lists and sizes
            last_idx = len(self.cumulative_sizes)
            self.file_list = self.file_list[:last_idx]
            self.file_sizes = self.file_sizes[:last_idx]
        else:
            self.total_samples = total_samples

        # Initialize file handles dictionary
        self.file_handles = None
        print(f"total samples: {self.total_samples}")
        print(f"number of files: {len(self.file_list)}")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total_samples:
            raise IndexError("Index out of range")

        # Initialize file handles per worker
        if self.file_handles is None:
            self.file_handles = {}
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                self.worker_id = worker_info.id
            else:
                self.worker_id = 0  # Single-process data loading
            # Open all files for this worker
            for file_path in self.file_list:
                self.file_handles[file_path] = h5py.File(file_path, "r")

        # Find the file index and local index within the file
        file_idx = np.searchsorted(self.cumulative_sizes, idx + 1)
        if file_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cumulative_sizes[file_idx - 1]

        file_path = self.file_list[file_idx]
        file = self.file_handles[file_path]

        # Retrieve the data
        x = file["x"][local_idx]
        particle_features = file["particle_features"][local_idx]
        subjets = file["subjets"][local_idx]
        particle_indices = file["particle_indices"][local_idx]
        subjet_mask = file["subjet_mask"][local_idx]
        particle_mask = file["particle_mask"][local_idx]

        return (
            x,
            particle_features,
            subjets,
            particle_indices,
            subjet_mask,
            particle_mask,
        )

    def close(self):
        if self.file_handles is not None:
            for f in self.file_handles.values():
                f.close()
        self.file_handles = None

    def __del__(self):
        self.close()
