# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
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
import mplhep as hep
import torch
import os
import os.path as osp

import tqdm

import h5py

def zero_pad_jets(arr, max_nconstit=128):
    """
    arr: numpy array
    """
    arr = arr[:max_nconstit]
    if arr.shape[0] < max_nconstit:
        zeros = np.zeros([max_nconstit - arr.shape[0],1])
        padded_arr = np.concatenate((arr, zeros), axis=0)
        return padded_arr
    else:
        return arr

def zero_pad(subjets_info, n_subjets, n_ptcls_per_subjet):
    """
    Pads the subjets and their constituent particle indices to fixed sizes.

    Args:
        subjets_info (list of dicts): List of dictionaries containing subjet features and particle indices.
        n_subjets (int): Fixed number of subjets to pad to.
        n_ptcls_per_subjet (int): Fixed number of particle indices per subjet to pad to.

    Returns:
        list of dicts: The padded list of subjet information.

    Each dictionary in the input list represents a subjet and contains:
    - "features": a dictionary with keys "pT", "eta", and "phi" for subjet features.
    - "indices": a list of indices for particles constituting the subjet.
    Padded subjets will have 0 for all features and -1 for all particle indices.
    """
    # Create a deep copy of subjets_info to avoid modifying the original list
    padded_subjets_info = deepcopy(subjets_info)
    # Pad subjets to have a fixed number of subjets
    while len(padded_subjets_info) < n_subjets:
        padded_subjets_info.append({"features": {"pT": 0, "eta": 0, "phi": 0, "num_ptcls": 0}, "indices": [-1] * n_ptcls_per_subjet})

    # Pad each subjet to have a fixed number of particle indices
    for subjet in padded_subjets_info:
        subjet["indices"] += [-1] * (n_ptcls_per_subjet - len(subjet["indices"]))
        subjet["indices"] = subjet["indices"][:n_ptcls_per_subjet]  # Ensure not to exceed the fixed size if already larger
        # subjet["indices"] = np.array(subjet["indices"])

    return padded_subjets_info


def get_subjets(px, py, pz, e, JET_ALGO="CA", jet_radius=0.2):
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
        pj = fastjet.PseudoJet(particle.px.item(), particle.py.item(), particle.pz.item(), particle.E.item())
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
            "num_ptcls": 0
        }
        
        # Extract indices, sort by pT
        indices = [constituent.user_index() for constituent in subjet.constituents()]
        indices = sorted(indices)  # since the original particles were already sorted by pT
        features["num_ptcls"] = len(indices)
    
        # Create dictionary for the current subjet and append to the list
        subjet_dict = {"features": features, "indices": indices}
        subjets_info.append(subjet_dict)
    
    # subjets_info now contains the required dictionaries for each subjet
    subjets_info_sorted = sorted(subjets_info, key=lambda x: x["features"]["pT"], reverse=True)
    
    # subjets_info_sorted now contains the subjets sorted by pT in descending order
    return subjets_info_sorted



def main(args):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    label = args.label
    hdf5_file = f"/ssl-jet-vol-v3/toptagging/{label}/raw/{label}.h5"
    vector.register_awkward()
    
    print("reading h5 file")
    df = pd.read_hdf(hdf5_file, key="table")
    print("finished reading h5 file")

    print("-------------------------------")
    print("Obtain (pT, eta_rel, phi_rel, E)")
    print("-------------------------------\n")
    def _col_list(prefix, max_particles=200):
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

    features = []
    labels = []
    
    for jet_index in tqdm.tqdm(range(len(df))):
    # for jet_index in tqdm.tqdm(range(10)):  
        # dim 1 ordering: 'part_eta','part_phi','part_pt_log', 'part_e_log', 'part_logptrel', 'part_logerel','part_deltaR'
        part_deta = zero_pad_jets(
            v["part_deta"][jet_index].to_numpy().reshape(-1, 1)
        )
        part_dphi = zero_pad_jets(
            v["part_dphi"][jet_index].to_numpy().reshape(-1, 1)
        )
        part_pt_log = zero_pad_jets(
            v["part_pt_log"][jet_index].to_numpy().reshape(-1, 1)
        )
        part_e_log = zero_pad_jets(
            v["part_e_log"][jet_index].to_numpy().reshape(-1, 1)
        )
        
        jet = np.concatenate([part_deta, part_dphi, part_pt_log, part_e_log], axis=1).transpose()
        y = v["label"][jet_index]
    
        features.append(jet)
        labels.append(y)
    features_array = np.stack(features)
    labels_array = np.stack(labels)
    print(f"features array shape: {features_array.shape}")
    print(f"labels array shape: {labels_array.shape}")

    save_path = f"/ssl-jet-vol-v3/I-JEPA-Jets/data/{label}"
    os.system(f"mkdir -p {save_path}")  # -p: create parent dirs if needed, exist_ok
    
    print("-------------------------------")
    print("Save to h5")
    print("-------------------------------\n")

    n_subjets = args.n_subjets
    n_ptcls_per_subjet = args.n_ptcls_per_subjet
    with h5py.File(f'{save_path}/{label}_{n_subjets}_{n_ptcls_per_subjet}_no_json.h5', 'w') as hdf:
        # Create group for particles
        particles_group = hdf.create_group("particles")
        # Storing the particles features array directly
        particles_group.create_dataset("features", data=features_array)
        particles_group.create_dataset("labels", data=labels_array)
        
        # Initialize an empty list to store serialized subjets information
        subjets = []
        
    
        for jet_idx in tqdm.tqdm(range(len(df))):  
        # for jet_idx in tqdm.tqdm(range(10)):  
            subjets_info = get_subjets(_px[jet_idx], _py[jet_idx], _pz[jet_idx], _e[jet_idx])

            subjets_padded = zero_pad(subjets_info, n_subjets, n_ptcls_per_subjet)
        
            subjets.append(subjets_padded)

        jets_group = hdf.create_group('jets')
        for jet_idx, jet in enumerate(subjets):
            
            jet_group = jets_group.create_group(f'jet_{jet_idx}')
            for subjet_idx, subjet in enumerate(jet):
                subjet_group = jet_group.create_group(f'subjet_{subjet_idx}')
                features_group = subjet_group.create_group('features')
                indices_ds = subjet_group.create_dataset('indices', data=subjet['indices'])
                
                for k in subjet['features'].keys():
                    features_group.create_dataset(k, data=subjet['features'][k])
            
        # # Convert list of JSON strings to numpy object array for storage
        # subjets_array = np.concatenate(subjets, axis=0)
        
        # # Create dataset for serialized subjets using variable-length strings
        # ds = hdf.create_dataset("subjets", data=subjets_array)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--label",
        type=str,
        action="store",
        default="train",
        help="train/val/test",
    )
    parser.add_argument(
        "--n-subjets",
        type=int,
        action="store",
        help="number of subjets per jet",
    )
    parser.add_argument(
        "--n-ptcls-per-subjet",
        type=int,
        action="store",
        help="number of particles per subjet",
    )
    args = parser.parse_args()
    main(args)
