# -*- coding: utf-8 -*-
import logging
from pathlib import Path
import os
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
import torch
import random

import awkward as ak
import fastjet
import vector

from copy import deepcopy

import argparse
import awkward as ak
import numpy as np
import pandas as pd
import vector
import torch
import os
import os.path as osp

import tqdm

import h5py


def zero_pad_jets(arr, max_nconstit=128):
    """
    arr: numpy array of shape (num_ptcls, 1)
    """
    arr = arr[:max_nconstit]
    if arr.shape[0] < max_nconstit:
        zeros = np.zeros([max_nconstit - arr.shape[0], 1])
        padded_arr = np.concatenate((arr, zeros), axis=0)
        return padded_arr
    else:
        return arr


def normalize(arr):
    mean = ak.mean(arr)
    std = ak.std(arr)
    norm_arr = (arr - mean) / std
    return norm_arr, mean, std
    
def main(args):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")
    label = args.label
    hdf5_file = f"/j-jepa-vol/TopTagging/raw/{label}.h5"
    vector.register_awkward()

    print("reading h5 file")
    df = pd.read_hdf(hdf5_file, key="table")
    num_jets = len(df)
    print("finished reading h5 file")

    print("-------------------------------")
    print("Obtain (pT, eta_rel, phi_rel, E)")
    print("-------------------------------\n")
    
    def _col_list(prefix, max_particles=128):
        return ["%s_%d" % (prefix, i) for i in range(max_particles)]
    
    _px = df[_col_list("PX")].values
    _py = df[_col_list("PY")].values
    _pz = df[_col_list("PZ")].values
    _e = df[_col_list("E")].values
    
    mask = _e > 0
    n_particles = np.sum(mask, axis=1)
    
    px = ak.unflatten(_px[mask], n_particles)
    py = ak.unflatten(_py[mask], n_particles)
    pz = ak.unflatten(_pz[mask], n_particles)
    energy = ak.unflatten(_e[mask], n_particles)
    
    p4 = ak.zip(
        {
            "px": px,
            "py": py,
            "pz": pz,
            "energy": energy,
        },
        with_name="Momentum4D",
    )

    jet_p4 = ak.sum(p4, axis=-1)

    # outputs
    v = {}
    v["label"] = df["is_signal_new"].values
    
    v["jet_pt"] = jet_p4.pt.to_numpy()
    v["jet_eta"] = jet_p4.eta.to_numpy()
    v["jet_phi"] = jet_p4.phi.to_numpy()
    v["jet_energy"] = jet_p4.energy.to_numpy()
    v["jet_mass"] = jet_p4.mass.to_numpy()
    v["jet_nparticles"] = n_particles
    
    v["part_px"] = px
    v["part_py"] = py
    v["part_pz"] = pz
    v["part_energy"] = energy
    
    # dim 1 ordering: 'part_deta','part_dphi','part_pt_log', 'part_e_log', 'part_logptrel', 'part_logerel','part_deltaR'
    v["part_deta"] = p4.eta - v["jet_eta"]
    v["part_dphi"] = p4.phi - v["jet_phi"]
    v["part_pt"] = np.hypot(v["part_px"], v["part_py"])
    v["part_pt_log"] = np.log(v["part_pt"])
    v["part_e_log"] = np.log(v["part_energy"])
    
    # normalize
    print("Begin normalization across dataset")
    v["part_deta"], deta_mean, deta_std = normalize(v["part_deta"])
    v["part_dphi"], dphi_mean, dphi_std = normalize(v["part_dphi"])
    v["part_pt_log"], pt_log_mean, pt_log_std = normalize(v["part_pt_log"])
    v["part_e_log"], e_log_mean, e_log_std = normalize(v["part_e_log"])
    
    stats = {
        "part_deta": [deta_mean, deta_std],
        "part_dphi": [dphi_mean, dphi_std],
        "part_pt_log": [pt_log_mean, pt_log_std],
        "part_e_log": [e_log_mean, e_log_std]
    }
    
    print("Finished normalization across dataset")
    print(stats)
    
    features = []
    labels = []
    
    # particle mask
    max_particles = 128
    
    # Create an empty mask array with shape (n_jets, max_particles)
    mask = np.zeros((len(n_particles), max_particles), dtype=int)
    
    # For each jet, mark the real particles as 1, and leave the rest as 0
    for i, num_particles in enumerate(n_particles):
        mask[i, :num_particles] = 1
    
    assert (np.array([sum(mask[i] == 1) for i in range(len(mask))]) == n_particles).all()
    
    for jet_index in tqdm.tqdm(range(num_jets)):
        # dim 1 ordering: 'part_eta','part_phi','part_pt_log', 'part_e_log', 'part_logptrel', 'part_logerel','part_deltaR'
        part_deta = zero_pad_jets(v["part_deta"][jet_index].to_numpy().reshape(-1, 1))
        part_dphi = zero_pad_jets(v["part_dphi"][jet_index].to_numpy().reshape(-1, 1))
        part_pt_log = zero_pad_jets(
            v["part_pt_log"][jet_index].to_numpy().reshape(-1, 1)
        )
        part_e_log = zero_pad_jets(v["part_e_log"][jet_index].to_numpy().reshape(-1, 1))
    
        part_px = zero_pad_jets(v["part_px"][jet_index].to_numpy().reshape(-1, 1))
        part_py = zero_pad_jets(v["part_py"][jet_index].to_numpy().reshape(-1, 1))
        part_pz = zero_pad_jets(v["part_pz"][jet_index].to_numpy().reshape(-1, 1))
    
        jet = np.concatenate(
            [part_px, part_py, part_pz, part_deta, part_dphi, part_pt_log, part_e_log], axis=1
        ).transpose()
        y = v["label"][jet_index]
    
        features.append(jet)
        labels.append(y)
    
    features_array = np.stack(features)
    labels_array = np.stack(labels)
    print(f"features array shape: {features_array.shape}")
    print(f"labels array shape: {labels_array.shape}")
    print(f"mask array shape: {mask.shape}")
    
    part_feature_names = ["part_px", "part_py", "part_pz", "part_deta", "part_dphi", "part_pt_log", "part_e_log"]
    part_features = {
        name: features_array[:, i, :] for i, name in enumerate(part_feature_names)
    }
    print(part_features["part_deta"].shape)
    
    save_path = f"/j-jepa-vol/J-JEPA/data/top/ptcl/{label}"
    os.system(f"mkdir -p {save_path}")  # -p: create parent dirs if needed, exist_ok
    
    print("-------------------------------")
    print("Save to h5")
    print("-------------------------------\n")
    
    with h5py.File(
        f"{save_path}/{label}{args.tag}.h5", "w"
    ) as hdf:
        particles_group = hdf.create_group("particles")
        for name in part_features.keys():
            particles_group.create_dataset(name, data=part_features[name])
        hdf.create_dataset("labels", data=labels_array)
        hdf.create_dataset("mask", data=mask)
        stats_group = hdf.create_group("stats")
        for name in stats.keys():
            stats_group.create_dataset(name, data=stats[name])

    # load saved file and inspect
    with h5py.File(f"{save_path}/{label}{args.tag}.h5", 'r') as hdf:
        particles = {
                    name: hdf["particles"][name][:] for name in hdf["particles"]
                }
        labels = hdf['labels'][:]
        mask = hdf['mask'][:]
        stats = {
                    name: hdf["stats"][name][:] for name in hdf["stats"]
                }
    print(particles)
    print("labels shape:", labels.shape)
    print("mask shape:", mask.shape)
    print(stats)

if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--label",
        type=str,
        action="store",
        default="train",
        help="train/val/test",
    )
    parser.add_argument(
        "--tag",
        type=str,
        action="store",
        default="",
        help="a tag for the dataset, e.g. _no_json",
    )

    args = parser.parse_args()
    main(args)
