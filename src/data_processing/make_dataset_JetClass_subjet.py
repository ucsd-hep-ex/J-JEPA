import os
import os.path as osp
import glob
import random
import pickle as pkl
import logging
import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import awkward as ak
import torch
import h5py
import uproot
import tqdm
import fastjet
import vector

from dotenv import find_dotenv, load_dotenv

# Register vector for awkward usage
vector.register_awkward()


def zero_pad_jets(arr, max_nconstit=128):
    """
    arr: numpy array
    """
    arr = arr[:max_nconstit]
    if arr.shape[0] < max_nconstit:
        zeros = np.zeros([max_nconstit - arr.shape[0], 1])
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
        padded_subjets_info.append(
            {
                "features": {"pT": 0, "eta": 0, "phi": 0, "num_ptcls": 0},
                "indices": [-1] * n_ptcls_per_subjet,
            }
        )

    # Pad each subjet to have a fixed number of particle indices
    for subjet in padded_subjets_info:
        subjet["indices"] += [-1] * (n_ptcls_per_subjet - len(subjet["indices"]))
        subjet["indices"] = subjet["indices"][
            :n_ptcls_per_subjet
        ]  # Ensure not to exceed the fixed size if already larger
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
        pj = fastjet.PseudoJet(
            particle.px.item(),
            particle.py.item(),
            particle.pz.item(),
            particle.E.item(),
        )
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
            "num_ptcls": 0,
        }

        # Extract indices, sort by pT
        indices = [constituent.user_index() for constituent in subjet.constituents()]
        indices = sorted(
            indices
        )  # since the original particles were already sorted by pT
        features["num_ptcls"] = len(indices)

        # Create dictionary for the current subjet and append to the list
        subjet_dict = {"features": features, "indices": indices}
        subjets_info.append(subjet_dict)

    # subjets_info now contains the required dictionaries for each subjet
    subjets_info_sorted = sorted(
        subjets_info, key=lambda x: x["features"]["pT"], reverse=True
    )

    # subjets_info_sorted now contains the subjets sorted by pT in descending order
    return subjets_info_sorted


def normalize(arr):
    mean = ak.mean(arr)
    std = ak.std(arr)
    norm_arr = (arr - mean) / std
    return norm_arr, mean, std


def _pad(a, maxlen, value=0, dtype="float32"):
    if isinstance(a, np.ndarray) and a.ndim >= 2 and a.shape[1] == maxlen:
        return a
    elif isinstance(a, ak.Array):
        if a.ndim == 1:
            a = ak.unflatten(a, 1)
        a = ak.fill_none(ak.pad_none(a, maxlen, clip=True), value)
        return ak.values_astype(a, dtype)
    else:
        x = (np.ones((len(a), maxlen)) * value).astype(dtype)
        for idx, s in enumerate(a):
            if not len(s):
                continue
            trunc = s[:maxlen].astype(dtype)
            x[idx, : len(trunc)] = trunc
        return x


def _clip(a, a_min, a_max):
    try:
        return np.clip(a, a_min, a_max)
    except ValueError:
        return ak.unflatten(np.clip(ak.flatten(a), a_min, a_max), ak.num(a))


def build_features_and_labels(tree, transform_features=True):
    # load arrays from the tree
    # Construct a Lorentz 4-vector from the (px, py, pz, energy) arrays
    a = tree.arrays(filter_name=["part_*", "jet_pt", "jet_energy", "label_*"])
    p4 = vector.zip(
        {
            "px": a["part_px"],
            "py": a["part_py"],
            "pz": a["part_pz"],
            "energy": a["part_energy"],
        }
    )
    jet_p4 = ak.sum(p4, axis=-1)

    # compute new features
    a["part_mask"] = ak.ones_like(a["part_energy"])
    a["part_pt"] = np.hypot(a["part_px"], a["part_py"])
    a["part_pt_log"] = np.log(a["part_pt"])
    a["part_e_log"] = np.log(a["part_energy"])
    a["part_logptrel"] = np.log(a["part_pt"] / a["jet_pt"])
    a["part_logerel"] = np.log(a["part_energy"] / a["jet_energy"])
    a["part_eta"] = p4.eta
    a["part_phi"] = p4.phi
    a["part_deta"] = p4.eta - jet_p4.eta
    a["part_dphi"] = p4.phi - jet_p4.phi

    # apply standardization
    if transform_features:
        a["part_pt_log"] = (a["part_pt_log"] - 1.7) * 0.7
        a["part_e_log"] = (a["part_e_log"] - 2.0) * 0.7
        a["part_logptrel"] = (a["part_logptrel"] - (-4.7)) * 0.7
        a["part_logerel"] = (a["part_logerel"] - (-4.7)) * 0.7

    out = {}
    out["pf_features"] = {}
    out["pf_mask"] = {}
    feature_list = {
        "pf_features": [
            "part_px",
            "part_py",
            "part_pz",
            "part_deta",
            "part_dphi",
            "part_pt_log",
            "part_e_log",
            "part_energy",
        ],
        "pf_mask": ["part_mask"],
    }

    stats = {"part_pt_log": [1.7, 0.58823529], "part_e_log": [2.0, 0.58823529]}
    out["stats"] = stats
    for k, names in feature_list.items():
        for name in names:
            out[k][name] = _pad(a[name], maxlen=128).to_numpy()

    label_list = [
        "label_QCD",
        "label_Hbb",
        "label_Hcc",
        "label_Hgg",
        "label_H4q",
        "label_Hqql",
        "label_Zqq",
        "label_Wqq",
        "label_Tbqq",
        "label_Tbl",
    ]
    out["label"] = np.stack([a[n].to_numpy().astype("int") for n in label_list], axis=1)

    return out


def main(args):
    """Runs data processing scripts to turn raw data from (/j-jepa-vol/JetClass/Pythia/) into
    cleaned data ready to be analyzed (saved in /j-jepa-vol/JetClass/processed).
    Convert root to pt files, each containing 1M zero-padded jets cropped to 128 constituents
    Only contains kinematic features
    Shape: (100k, 7, 128)
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")
    label = args.label
    if label == "train":
        label += "_100M"
    elif label == "val":
        label += "_5M"
    elif label == "test":
        label += "_20M"
    data_dir = f"/j-jepa-vol/JetClass/Pythia/{label}"
    data_files = glob.glob(f"{data_dir}/*")
    label_orig = label.split("_")[0]  # without _100M, _5M, _20M
    processed_dir = f"/j-jepa-vol/J-JEPA/data/JetClass/subjet/{label_orig}"
    os.system(f"mkdir -p {processed_dir}")

    for i, file in enumerate(data_files):
        tree = uproot.open(file)["tree"]
        file_name = file.split("/")[-1].split(".")[0]
        print(f"--- loaded data file {i} {file_name} from `{label}` directory")
        f_dict = build_features_and_labels(tree)
        num_jets = f_dict["pf_features"]["part_px"].shape[0]
        # Initialize an empty list to store serialized subjets information
        subjet_feature_names = [
            "subjet_pt",
            "subjet_eta",
            "subjet_phi",
            "subjet_num_ptcls",
        ]
        subjets = {
            name: np.zeros((num_jets, args.n_subjets))
            for i, name in enumerate(subjet_feature_names)
        }
        subjets["particle_indices"] = np.zeros(
            (num_jets, args.n_subjets, args.n_ptcls_per_subjet)
        )
        _px = f_dict["pf_features"]["part_px"]
        _py = f_dict["pf_features"]["part_py"]
        _pz = f_dict["pf_features"]["part_pz"]
        _e = f_dict["pf_features"]["part_energy"]
        for jet_idx in tqdm.tqdm(range(num_jets)):
            subjets_info = get_subjets(
                _px[jet_idx], _py[jet_idx], _pz[jet_idx], _e[jet_idx]
            )
            subjets_info_padded = zero_pad(
                subjets_info, args.n_subjets, args.n_ptcls_per_subjet
            )
            subjets_padded = subjets_info_padded[: args.n_subjets]
            for subjet_idx, subjet in enumerate(subjets_padded):
                subjets["subjet_pt"][jet_idx, subjet_idx] = subjet["features"]["pT"]
                subjets["subjet_eta"][jet_idx, subjet_idx] = subjet["features"]["eta"]
                subjets["subjet_phi"][jet_idx, subjet_idx] = subjet["features"]["phi"]
                subjets["subjet_num_ptcls"][jet_idx, subjet_idx] = subjet["features"][
                    "num_ptcls"
                ]
                subjets["particle_indices"][jet_idx, subjet_idx] = subjet["indices"]
                # if subjet["features"]["num_ptcls"] > 0:
                #     for ptcl_idx in subjet["indices"]:
                #         if ptcl_idx != -1 and ptcl_idx < 128:
                #             f_dict["pf_features"]["subjet_indices"][
                #                 jet_idx, ptcl_idx
                #             ] = int(subjet_idx)
        with h5py.File(f"{processed_dir}/{file_name}.h5", "w") as hdf:
            particles_group = hdf.create_group("particles")
            for name in f_dict["pf_features"].keys():
                particles_group.create_dataset(name, data=f_dict["pf_features"][name])
            hdf.create_dataset("labels", data=f_dict["label"])
            hdf.create_dataset("mask", data=f_dict["pf_mask"]["part_mask"])
            stats_group = hdf.create_group("stats")
            for name in f_dict["stats"].keys():
                stats_group.create_dataset(name, data=f_dict["stats"][name])
            subjets_group = hdf.create_group("subjets")
            for name in subjets.keys():
                subjets_group.create_dataset(name, data=subjets[name])
        print(f"--- saved data file {i} {file_name} to `{processed_dir}` directory")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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
        "--tag",
        type=str,
        action="store",
        default="JetCLR",
        help="JetCLR/JJEPA",
    )
    parser.add_argument(
        "--n-subjets",
        type=int,
        action="store",
        default=20,
        help="number of subjets per jet",
    )
    parser.add_argument(
        "--n-ptcls-per-subjet",
        type=int,
        action="store",
        default=30,
        help="number of particles per subjet",
    )

    args = parser.parse_args()
    main(args)
