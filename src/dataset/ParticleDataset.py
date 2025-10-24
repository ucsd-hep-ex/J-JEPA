from torch.utils.data import Dataset
import torch
import h5py
import os
import numpy as np
from collections import namedtuple
from functools import lru_cache

#from src.util.create_random_masks import get_subjets

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
        compute_subjets=False
    ):
        self.return_labels = return_labels
        self.size_multiplier = size_multiplier
        self.compute_subjets = compute_subjets
        self.subjets_cache = {}
        self.files = sorted(
            os.path.join(directory_path, f)
            for f in os.listdir(directory_path)
            if f.endswith((".h5", ".hdf5"))
        )
        if not self.files:
            raise ValueError(f"No HDF5 files in {directory_path!r}")
        with h5py.File(self.files[0], 'r') as f0:
            stats = {k: f0['stats'][k][:] for k in f0['stats']}
        self.mean_log_e, self.std_log_e = stats['part_e_log']
        self.stats = stats
        lengths = []
        for fn in self.files:
            with h5py.File(fn, 'r') as f:
                lengths.append(int(f['labels'].shape[0]))
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
        self.file_lengths = np.array(lengths, dtype=int)
        self.cum_lengths = np.concatenate([[0], np.cumsum(self.file_lengths)])
        self._total = int(self.cum_lengths[-1])
        self.cache_size_bytes = int(cache_size_gb * 1024**3)
        self.content_cache = {}
        self.total_cached = 0
        self._cache_order = []
        self._preload_content()

    def _estimate_size(self, path: str) -> int:
        with h5py.File(path, 'r') as f:
            n_jets, n_parts = f['labels'].shape[0], f['mask'].shape[1]
            bytes_p4_spatial = n_jets * n_parts * 4 * 4
            bytes_p4         = n_jets * n_parts * 4 * 4
            bytes_mask       = n_jets * n_parts * 1 * 4
            bytes_labels     = n_jets * f['labels'].dtype.itemsize
        return int((bytes_p4_spatial + bytes_p4 + bytes_mask + bytes_labels) * self.size_multiplier)

    def _preload_content(self):
        for fn in sorted(self.files, key=self._estimate_size):
            est = self._estimate_size(fn)
            if self.total_cached + est > self.cache_size_bytes:
                break
            with h5py.File(fn, 'r') as f:
                parts = {k: f['particles'][k][:] for k in f['particles']}
                mask_np = f['mask'][:]
                labels_np = f['labels'][:] if self.return_labels else None
            for k, arr in parts.items():
                parts[k] = arr.astype(np.float32)
            mask_np = mask_np.astype(np.float32)
            log_e = parts['part_e_log'] * self.std_log_e + self.mean_log_e
            norm_e = (np.exp(log_e) * mask_np).astype(np.float32)
            p4_spatial_np = np.stack([parts['part_px'], parts['part_py'], parts['part_pz'], norm_e], axis=-1)
            p4_np         = np.stack([parts['part_deta'], parts['part_dphi'], parts['part_pt_log'], parts['part_e_log']], axis=-1)
            p4_spatial = torch.from_numpy(p4_spatial_np)
            p4         = torch.from_numpy(p4_np)
            mask       = torch.from_numpy(mask_np).unsqueeze(-1)
            labels     = torch.from_numpy(labels_np) if self.return_labels else None
            data = {'p4_spatial': p4_spatial, 'p4': p4, 'mask': mask, 'labels': labels}
            actual = sum(t.element_size() * t.numel() for t in data.values() if t is not None)
            self.content_cache[fn] = data
            self.total_cached += actual
            self._cache_order.append(fn)

    @lru_cache(maxsize=8)
    def _get_file_handle(self, fn: str) -> h5py.File:
        return h5py.File(fn, 'r', rdcc_nbytes=512*1024**2, rdcc_nslots=1_000_000, rdcc_w0=0.9)

    def _prefetch_file(self, fn: str):
        if fn in self.content_cache:
            return
        with h5py.File(fn, 'r') as f:
            parts = {k: f['particles'][k][:] for k in f['particles']}
            mask_np = f['mask'][:]
            labels_np = f['labels'][:] if self.return_labels else None
        for k, arr in parts.items():
            parts[k] = arr.astype(np.float32)
        mask_np = mask_np.astype(np.float32)
        log_e = parts['part_e_log'] * self.std_log_e + self.mean_log_e
        norm_e = (np.exp(log_e) * mask_np).astype(np.float32)
        p4_spatial_np = np.stack([parts['part_px'], parts['part_py'], parts['part_pz'], norm_e], axis=-1)
        p4_np         = np.stack([parts['part_deta'], parts['part_dphi'], parts['part_pt_log'], parts['part_e_log']], axis=-1)
        p4_spatial = torch.from_numpy(p4_spatial_np)
        p4         = torch.from_numpy(p4_np)
        mask       = torch.from_numpy(mask_np).unsqueeze(-1)
        labels     = torch.from_numpy(labels_np) if self.return_labels else None
        data = {'p4_spatial': p4_spatial, 'p4': p4, 'mask': mask, 'labels': labels}
        need = sum(t.element_size() * t.numel() for t in data.values() if t is not None)
        while self._cache_order and self.total_cached + need > self.cache_size_bytes:
            victim = self._cache_order.pop(0)
            ev = self.content_cache.pop(victim, None)
            if ev is not None:
                self.total_cached -= sum(t.element_size() * t.numel() for t in ev.values() if t is not None)
        self.content_cache[fn] = data
        self.total_cached += need
        self._cache_order.append(fn)

    def __len__(self):
        return self._total

    def __getitem__(self, idx):
        file_idx = int(np.searchsorted(self.cum_lengths, idx, side='right') - 1)
        local_idx = int(idx - self.cum_lengths[file_idx])
        fn = self.files[file_idx]
        if fn in self.content_cache:
            d = self.content_cache[fn]
            p_spatial = d['p4_spatial'][local_idx]
            p4_tensor = d['p4'][local_idx]
            p_mask    = d['mask'][local_idx]
            labels    = d['labels'][local_idx] if self.return_labels else None
        else:
            self._prefetch_file(fn)
            if fn in self.content_cache:
                d = self.content_cache[fn]
                p_spatial = d['p4_spatial'][local_idx]
                p4_tensor = d['p4'][local_idx]
                p_mask    = d['mask'][local_idx]
                labels    = d['labels'][local_idx] if self.return_labels else None
            else:
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
                p_mask    = torch.from_numpy(mask_np).unsqueeze(-1)
        p_spatial = p_spatial * p_mask
        p4_tensor = p4_tensor * p_mask
        subjets_info_sorted = None
        if self.compute_subjets:
            if idx not in self.subjets_cache:
                valid = p_mask.squeeze(-1).bool().numpy()
                arr   = p_spatial[valid].numpy()
                if arr.size == 0:
                    self.subjets_cache[idx] = None
                else:
                    px, py, pz, e = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
                    subjets = get_subjets(px, py, pz, e, JET_ALGO="CA", jet_radius=0.2)
                    self.subjets_cache[idx] = subjets
            subjets_info_sorted = self.subjets_cache[idx]
        if self.return_labels:
            return p_spatial, p4_tensor, p_mask, subjets_info_sorted, labels
        return p_spatial, p4_tensor, p_mask, subjets_info_sorted
