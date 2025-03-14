# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import argparse
import numpy as np
import awkward as ak
import uproot
import vector

vector.register_awkward()
import torch
import os
import os.path as osp
import glob
import h5py
from collections import defaultdict


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

    # compute new features
    a["part_mask"] = ak.ones_like(a["part_energy"])
    a["part_pt"] = np.hypot(a["part_px"], a["part_py"])
    a["part_pt_log"] = np.log(a["part_pt"])
    a["part_e_log"] = np.log(a["part_energy"])
    a["part_logptrel"] = np.log(a["part_pt"] / a["jet_pt"])
    a["part_logerel"] = np.log(a["part_energy"] / a["jet_energy"])
    a["part_deltaR"] = np.hypot(a["part_deta"], a["part_dphi"])
    a["part_eta"] = p4.eta
    a["part_phi"] = p4.phi

    # Add impact parameter features as shown in JetClass101
    a["part_d0"] = np.tanh(a["part_d0val"])
    a["part_dz"] = np.tanh(a["part_dzval"])

    # Compute or ensure required features are present
    a["part_deta"] = a.get("part_deta", a["part_eta"])  # Use if exists or calculate
    a["part_dphi"] = a.get("part_dphi", a["part_phi"])  # Use if exists or calculate

    # Stats collection for normalization
    stats = {}

    # Apply standardization if requested
    if transform_features:
        # Calculate statistics before transformation
        mean_log_e = np.mean(ak.flatten(a["part_e_log"]))
        std_log_e = np.std(ak.flatten(a["part_e_log"]))

        # Store statistics
        stats["part_e_log"] = np.array([mean_log_e, std_log_e])

        # Apply normalization
        a["part_pt_log"] = (a["part_pt_log"] - 1.7) / 0.7
        a["part_e_log"] = (a["part_e_log"] - mean_log_e) / std_log_e
        a["part_logptrel"] = (a["part_logptrel"] - (-4.7)) / 0.7
        a["part_logerel"] = (a["part_logerel"] - (-4.7)) / 0.7
        a["part_deltaR"] = (a["part_deltaR"] - 0.2) / 4.0

        # Clip impact parameter errors as in JetClass101
        a["part_d0err"] = _clip(a["part_d0err"], 0, 1)
        a["part_dzerr"] = _clip(a["part_dzerr"], 0, 1)

    # Required by ParticleDataset
    particle_features = {
        # Core 4-vector components needed by ParticleDataset
        "part_px": a["part_px"],
        "part_py": a["part_py"],
        "part_pz": a["part_pz"],
        "part_deta": a["part_deta"],
        "part_dphi": a["part_dphi"],
        "part_pt_log": a["part_pt_log"],
        "part_e_log": a["part_e_log"],
        # Additional features - particle IDs
        "part_charge": a["part_charge"],
        "part_isChargedHadron": a["part_isChargedHadron"],
        "part_isNeutralHadron": a["part_isNeutralHadron"],
        "part_isPhoton": a["part_isPhoton"],
        "part_isElectron": a["part_isElectron"],
        "part_isMuon": a["part_isMuon"],
        # Additional features - impact parameters
        "part_d0": a["part_d0"],
        "part_d0err": a["part_d0err"],
        "part_dz": a["part_dz"],
        "part_dzerr": a["part_dzerr"],
        # Additional derived features
        "part_logptrel": a["part_logptrel"],
        "part_logerel": a["part_logerel"],
        "part_deltaR": a["part_deltaR"],
    }

    # Padding all features to the same length
    maxlen = 128
    out = {}
    out["particles"] = {
        name: _pad(feature, maxlen).to_numpy()
        for name, feature in particle_features.items()
    }
    out["mask"] = _pad(a["part_mask"], maxlen).to_numpy()

    # Labels
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
    out["labels"] = np.stack(
        [a[n].to_numpy().astype("int") for n in label_list], axis=1
    )

    # Stats
    out["stats"] = stats

    return out


def main(args):
    """Runs data processing scripts to turn raw data from (/ssl-jet-vol-v2/JetClass/Pythia/) into
    cleaned data ready to be analyzed (saved in /ssl-jet-vol-v2/JetClass/processed).
    Convert root to h5 files, each containing structured data as required by ParticleDataset
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
    processed_dir = f"/j-jepa-vol/J-JEPA/data/JetClass/ptcl/{label_orig}"
    os.system(f"mkdir -p {processed_dir}")  # -p: create parent dirs if needed, exist_ok

    # Calculate global statistics across all files if needed
    if args.global_stats:
        logger.info("Calculating global statistics across all files...")
        all_stats = defaultdict(list)

        for i, file in enumerate(data_files):
            tree = uproot.open(file)["tree"]
            file_name = file.split("/")[-1].split(".")[0]
            print(
                f"--- processing file {i} {file_name} from `{label}` directory for statistics"
            )

            # Get arrays needed for statistics
            arrays = tree.arrays(filter_name=["part_energy"])
            e_log = np.log(ak.flatten(arrays["part_energy"]))

            # Store statistics from this file
            all_stats["part_e_log"].extend(e_log)

        # Calculate global statistics
        global_stats = {}
        global_stats["part_e_log"] = np.array(
            [np.mean(all_stats["part_e_log"]), np.std(all_stats["part_e_log"])]
        )
        logger.info(f"Global statistics: {global_stats}")
    else:
        global_stats = None

    # Process each file
    for i, file in enumerate(data_files):
        tree = uproot.open(file)["tree"]
        file_name = file.split("/")[-1].split(".")[0]
        print(f"--- processing data file {i} {file_name} from `{label}` directory")

        # Pass global stats to use consistent normalization
        f_dict = build_features_and_labels(tree, transform_features=True)

        # Save as H5 file
        h5_path = osp.join(processed_dir, f"{file_name}.h5")
        with h5py.File(h5_path, "w") as h5f:
            # Create groups
            particles_group = h5f.create_group("particles")
            stats_group = h5f.create_group("stats")

            # Save particles data
            for feature_name, feature_data in f_dict["particles"].items():
                particles_group.create_dataset(feature_name, data=feature_data)

            # Save mask, labels, and stats
            h5f.create_dataset("mask", data=f_dict["mask"])
            h5f.create_dataset("labels", data=f_dict["labels"])

            # Save stats
            for stat_name, stat_data in f_dict["stats"].items():
                stats_group.create_dataset(stat_name, data=stat_data)

        print(f"--- saved h5 file {h5_path}")


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
        "--global_stats",
        action="store_true",
        help="Calculate global statistics across all files for consistent normalization",
    )
    args = parser.parse_args()
    main(args)
