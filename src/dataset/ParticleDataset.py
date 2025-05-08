from torch.utils.data import Dataset
import torch
import h5py
import os
import numpy as np
from collections import namedtuple
from functools import lru_cache

DataSample = namedtuple("DataSample", ["p4_spatial", "p4", "mask"])
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
        cache_size_gb=64.0,
        size_multiplier=1.0,
    ):
        self.return_labels = return_labels
        self.size_multiplier = size_multiplier

         # sort all hfd5 files in directory, treating each file as a long concatenated dataset of jets
        self.files = sorted(
            os.path.join(directory_path, f)
            for f in os.listdir(directory_path)
            if f.endswith((".h5", ".hdf5"))
        )
        if not self.files:
            raise ValueError(f"No HDF5 files in {directory_path!r}")

         # read stats once 
        with h5py.File(self.files[0], 'r') as f0:
            stats = {k: f0['stats'][k][:] for k in f0['stats']}
        self.mean_log_e, self.std_log_e = stats['part_e_log']
        self.stats = stats

         # compute number of jets per file
        lengths = []
        for fn in self.files:
            with h5py.File(fn, 'r') as f:
                lengths.append(int(f['labels'].shape[0]))

         # truncate files based on num_jets ( if specified )
        if num_jets is not None and num_jets < sum(lengths):
            capped, total = [], 0
            for fn, L in zip(self.files, lengths):
                if total + L < num_jets:
                    capped.append(L)
                    total += L
                else:
                    capped.append(num_jets - total)
                    break
            lengths = capped
            self.files = self.files[:len(lengths)]

        # convert to np.array and build cumulative sum
        self.file_lengths = np.array(lengths, dtype=int)
        self.cum_lengths = np.concatenate([[0], np.cumsum(self.file_lengths)])
        self._total = int(self.cum_lengths[-1])

        # content cache: uses sorted order of file sizes and preloads into CPU ram up to cache_size_gb
        self.cache_size_bytes = int(cache_size_gb * 1024**3)
        self.content_cache = {}
        self.total_cached = 0
        self._preload_content()

    def _estimate_size(self, path: str) -> int:
        """
        helper function for calculating how many bytes the file will occupy in CPU ram
        """
        # estimate bytes of tensors cached per file
        with h5py.File(path, 'r') as f:
            n_jets, n_parts = f['labels'].shape[0], f['mask'].shape[1]
            bytes_p4_spatial = n_jets * n_parts * 4 * 4
            bytes_p4         = n_jets * n_parts * 4 * 4
            bytes_mask       = n_jets * n_parts * 1 * 4
            bytes_labels     = n_jets * f['labels'].dtype.itemsize
        return int((bytes_p4_spatial + bytes_p4 + bytes_mask + bytes_labels) * self.size_multiplier)

    def _preload_content(self):
        """
        walks through the files in sorted order, and checks if adding the estimated size will exceed the cache limit.
        if not, then all the datasets are read into np.arrays depending on if it will fit into the cache.
        """
        for fn in sorted(self.files, key=self._estimate_size):
            est = self._estimate_size(fn)
            # check if the estimated cache size is larger than the cache limit
            if self.total_cached + est > self.cache_size_bytes:
                break

            # read file into memory
            with h5py.File(fn, 'r') as f:
                parts = {k: f['particles'][k][:] for k in f['particles']}
                mask_np = f['mask'][:]
                labels_np = f['labels'][:] if self.return_labels else None

            # compute numpy arrays
            for k, arr in parts.items():
                parts[k] = arr.astype(np.float32)
            mask_np = mask_np.astype(np.float32)
            log_e = parts['part_e_log'] * self.std_log_e + self.mean_log_e
            norm_e = (np.exp(log_e) * mask_np).astype(np.float32)
            p4_spatial_np = np.stack([parts['part_px'], parts['part_py'], parts['part_pz'], norm_e], axis=-1)
            p4_np         = np.stack([parts['part_deta'], parts['part_dphi'], parts['part_pt_log'], parts['part_e_log']], axis=-1)

            # convert to torch tensors once
            p4_spatial = torch.from_numpy(p4_spatial_np)
            p4         = torch.from_numpy(p4_np)
            mask       = torch.from_numpy(mask_np).unsqueeze(-1)
            labels     = torch.from_numpy(labels_np) if self.return_labels else None

            data = {'p4_spatial': p4_spatial, 'p4': p4, 'mask': mask, 'labels': labels}
            actual = sum(t.element_size() * t.numel() for t in data.values() if t is not None)
            self.content_cache[fn] = data
            self.total_cached += actual

    @lru_cache(maxsize=8)
    def _get_file_handle(self, fn: str) -> h5py.File:
        """
        opens and caches up to 8 HDF5 files to avoid reopening memory cost
        """
        return h5py.File(fn, 'r', rdcc_nbytes=512*1024**2, rdcc_nslots=1_000_000, rdcc_w0=0.9)

    def __len__(self):
        return self._total

    def __getitem__(self, idx):
        # Map global idx → file and local index
        file_idx = int(np.searchsorted(self.cum_lengths, idx, side='right') - 1)
        local_idx = int(idx - self.cum_lengths[file_idx])
        fn = self.files[file_idx]

        if fn in self.content_cache:
            # fast path: slice from cache arrays
            d = self.content_cache[fn]
            p_spatial = d['p4_spatial'][local_idx]
            p4_tensor = d['p4'][local_idx]
            p_mask     = d['mask'][local_idx]
            labels     = d['labels'][local_idx] if self.return_labels else None
        else:
            # slow path: read one jet from disk via cached file handle
            f = self._get_file_handle(fn)
            px = f['particles']['part_px'][local_idx].astype(np.float32)
            py = f['particles']['part_py'][local_idx].astype(np.float32)
            pz = f['particles']['part_pz'][local_idx].astype(np.float32)
            deta = f['particles']['part_deta'][local_idx].astype(np.float32)
            dphi = f['particles']['part_dphi'][local_idx].astype(np.float32)
            ptl = f['particles']['part_pt_log'][local_idx].astype(np.float32)
            elog = f['particles']['part_e_log'][local_idx].astype(np.float32)
            mask_np = f['mask'][local_idx].astype(np.float32)
            labels = f['labels'][local_idx] if self.return_labels else None
            log_e = elog * self.std_log_e + self.mean_log_e
            norm_e = (np.exp(log_e) * mask_np).astype(np.float32)
            p_spatial = torch.from_numpy(np.stack([px, py, pz, norm_e], axis=-1))
            p4_tensor = torch.from_numpy(np.stack([deta, dphi, ptl, elog], axis=-1))
            p_mask     = torch.from_numpy(mask_np).unsqueeze(1)

        if self.return_labels:
            return DataSample_label(p_spatial, p4_tensor, p_mask, labels)
        return DataSample(p_spatial, p4_tensor, p_mask)