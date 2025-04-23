from torch.utils.data import Dataset
import torch
import h5py
import os
import numpy as np
from collections import namedtuple
from functools import lru_cache

DataSample       = namedtuple("DataSample",       ["p4_spatial", "p4", "mask"])
DataSample_label = namedtuple("DataSample_label", ["p4_spatial", "p4", "mask", "labels"])

class ParticleDataset(Dataset):
    """
    description:
        this version of ParticleDataset contains the following features:
            - Content cache:  preloads small files into CPU RAM up to cache_size_gb
            - LRU file cache: keeps up to 8 HDF5 files open to avoid reopening
            - For uncached files: reads only the single requested jet 
    """
    def __init__(
        self,
        directory_path,
        num_jets=None,
        return_labels=False,
        cache_size_gb= 128.0,
        size_multiplier=1.2,
    ):
        self.return_labels     = return_labels

        # sort all hfd5 files in directory, treating each file as a long concatenated dataset of jets
        self.files = sorted(
            os.path.join(directory_path, f)
            for f in os.listdir(directory_path)
            if f.endswith((".h5", ".hdf5"))
        )
        if not self.files:
            raise ValueError(f"No HDF5 files in {directory_path!r}")

        # read stats once 
        with h5py.File(self.files[0], "r") as f0:
            stats = {k: f0["stats"][k][:] for k in f0["stats"]}
        self.mean_log_e, self.std_log_e = stats["part_e_log"]
        self.stats = stats

        # compute number of jets per file
        lengths = []
        for fn in self.files:
            with h5py.File(fn, "r") as f:
                lengths.append(f["labels"].shape[0])
                
        # truncate files based on num_jets ( if specified )        
        if num_jets is not None and num_jets < sum(lengths):
            capped, cum = [], 0
            for fn, L in zip(self.files, lengths):
                if cum + L < num_jets:
                    capped.append(L); cum += L
                else:
                    capped.append(num_jets - cum)
                    break
            lengths = capped
            self.files = self.files[: len(lengths)]
        
        # convert to np.array and build cumulative sum
        self.file_lengths = np.array(lengths, dtype=int)
        self.cum_lengths  = np.concatenate([[0], np.cumsum(self.file_lengths)])
        self._total       = int(self.cum_lengths[-1])

        # content cache: uses sorted order of file sizes and preloads into CPU ram up to cache_size_gb
        self.cache_size_bytes = int(cache_size_gb * 1024**3)
        self.size_multiplier  = size_multiplier
        self.content_cache    = {}
        self.total_cached     = 0
        self._preload_content()

    def _estimate_size(self, path: str) -> int:
        """
        helper function for calculating how many bytes the file will occupy in CPU ram
        """
        return int(os.path.getsize(path) * self.size_multiplier)

    def _preload_content(self):
        """
        walks through the files in sorted order, and checks if adding the estimated size will exceed the cache limit.
        if not, then all the datasets are read into np.arrays depending on if it will fit into the cache.
        """
        # go thru files in sorted order
        for fn in sorted(self.files, key=self._estimate_size):
            est = self._estimate_size(fn)
            # check if the estimated cache size is larger than the cache limit
            if self.total_cached + est > self.cache_size_bytes:
                break
            # read file into memory
            with h5py.File(fn, "r") as f:
                data = {k: f["particles"][k][:] for k in f["particles"]}
                data["mask"]   = f["mask"][:]
                data["labels"] = f["labels"][:]
            actual = sum(arr.nbytes for arr in data.values())
            if self.total_cached + actual <= self.cache_size_bytes:
                self.content_cache[fn] = data
                self.total_cached     += actual
            else:
                break  # exceeds cache limit

    @lru_cache(maxsize=8)
    def _get_file_handle(self, fn: str) -> h5py.File:
        """
        opens and caches up to 8 HDF5 files to avoid reopening memory cost
        """
        return h5py.File(fn, "r")

    def __len__(self):
        return self._total

    def __getitem__(self, idx):
        """
        
        """
        # Map global idx → file and local index
        file_idx  = int(np.searchsorted(self.cum_lengths, idx, side="right") - 1)
        local_idx = int(idx - self.cum_lengths[file_idx])
        fn        = self.files[file_idx]

        if fn in self.content_cache:
            # fast path: slice from cache arrays
            data = self.content_cache[fn]
            p_px, p_py, p_pz = data["part_px"][local_idx], data["part_py"][local_idx], data["part_pz"][local_idx]
            p_deta, p_dphi  = data["part_deta"][local_idx], data["part_dphi"][local_idx]
            p_ptl, p_el     = data["part_pt_log"][local_idx], data["part_e_log"][local_idx]
            mask            = data["mask"][local_idx]
            labels          = data["labels"][local_idx] if self.return_labels else None
        else:
            # slow path: read one jet from disk via cached file handle
            f = self._get_file_handle(fn)
            p_px   = f["particles"]["part_px"][local_idx]
            p_py   = f["particles"]["part_py"][local_idx]
            p_pz   = f["particles"]["part_pz"][local_idx]
            p_deta = f["particles"]["part_deta"][local_idx]
            p_dphi = f["particles"]["part_dphi"][local_idx]
            p_ptl  = f["particles"]["part_pt_log"][local_idx]
            p_el   = f["particles"]["part_e_log"][local_idx]
            mask   = f["mask"][local_idx]
            labels = f["labels"][local_idx] if self.return_labels else None

        # reconstruct p4, p4_spatial
        log_e      = p_el * self.std_log_e + self.mean_log_e
        norm_e     = np.exp(log_e) * mask
        p4_spatial = np.stack([p_px, p_py, p_pz, norm_e], axis=1)
        p4         = np.stack([p_deta, p_dphi, p_ptl, p_el], axis=1)

        p_spatial = torch.from_numpy(p4_spatial)
        p4      = torch.from_numpy(p4)
        p_mask    = torch.from_numpy(mask).unsqueeze(1)

        if self.return_labels:
            return DataSample_label(p_spatial, p4, p_mask, labels)
        else:
            return DataSample(p_spatial, p4, p_mask)
