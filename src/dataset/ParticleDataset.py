from torch.utils.data import Dataset
import torch
import h5py
import os
import numpy as np
from collections import namedtuple
import gc

DataSample = namedtuple('DataSample', ['p3', 'p4', 'mask'])
DataSample_label = namedtuple('DataSample_label', ['p3', 'p4', 'mask', 'labels'])

class ParticleDataset(Dataset):
    def __init__(self, directory_path, num_jets=None, return_labels=False):
        self.return_labels = return_labels
        # Initialize data containers
        particles_dict = {}  # Will hold lists of particle data arrays
        labels_list = []
        mask_list = []
        stats_dict = {}  # Will hold lists of stats data arrays

        # Loop through files in the directory
        for filename in os.listdir(directory_path):
            if filename.endswith(".hdf5") or filename.endswith(".h5"):
                file_path = os.path.join(directory_path, filename)
                with h5py.File(file_path, 'r') as hdf:
                    print(f"Loading {directory_path}/{filename}")
                    particles = {name: hdf["particles"][name][:] for name in hdf["particles"]}
                    labels = hdf['labels'][:]
                    mask = hdf['mask'][:]
                    stats = {name: hdf["stats"][name][:] for name in hdf["stats"]}

                # Append data to lists/dicts
                for key in particles:
                    if key not in particles_dict:
                        particles_dict[key] = []
                    particles_dict[key].append(particles[key])

                labels_list.append(labels)
                mask_list.append(mask)

                for key in stats:
                    if key not in stats_dict:
                        stats_dict[key] = []
                    stats_dict[key].append(stats[key])

                print(f"Loaded {directory_path}/{filename}")

        # Concatenate all the data
        self.particles = {key: np.concatenate(particles_dict[key], axis=0) for key in particles_dict}
        self.labels = np.concatenate(labels_list, axis=0)
        self.mask = np.concatenate(mask_list, axis=0)
        self.stats = {key: np.concatenate(stats_dict[key], axis=0) for key in stats_dict}

        part_features = particles_dict.keys() # [part_px, part_py, part_pz, part_deta, part_dphi, part_pt_log, part_e_log]
        self.p3 = np.stack([self.particles[key] for key in ["part_px", "part_py", "part_pz"]], axis=1) # for pos emb
        self.p4 = np.stack([self.particles[key] for key in ["part_deta", "part_dphi", "part_pt_log", "part_e_log"]], axis=1) # for input to J-JEPA

        # Limit to num_jets if specified
        if num_jets:
            self.labels = self.labels[:num_jets]
            self.mask = self.mask[:num_jets]
            self.p3 = self.p3[:num_jets]
            self.p4 = self.p4[:num_jets]
            for key in self.stats:
                self.stats[key] = self.stats[key][:num_jets]
        if self.return_labels:
            print("__getitem__ returns", ['p3 (px, py, pz)', 'p4 (eta, phi, log_pt, log_e)', 'mask', 'labels'])
        else:
            print("__getitem__ returns", ['p3 (px, py, pz)', 'p4 (eta, phi, log_pt, log_e)', 'mask'])
        del self.particles
        gc.collect()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.return_labels:
            sample = DataSample_label(
        p3 = torch.from_numpy(self.p3[idx]),
        p4 = torch.from_numpy(self.p4[idx]),
        mask = torch.from_numpy(self.mask[idx]),
        labels = self.labels[idx]
    )
        else:
            sample = DataSample(
        p3 = torch.from_numpy(self.p3[idx]),
        p4 = torch.from_numpy(self.p4[idx]),
        mask = torch.from_numpy(self.mask[idx]),
    )

        return sample
