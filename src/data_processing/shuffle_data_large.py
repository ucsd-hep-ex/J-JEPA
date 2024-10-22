#!/bin/env python3.7

# load standard python modules
import sys
import logging
from pathlib import Path

# sys.path.insert(0, "../src")
import os
import numpy as np
import random
import glob
import argparse
import gc

# load torch modules
import torch

import h5py

# Seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


def get_data_file_paths(flag, tag, percent=1):
    data_files = []
    if tag == "JetCLR":
        data_files = glob.glob(
            f"/j-jepa-vol/JetClass/processed/JetCLR/{percent}%/{flag}/data/*"
        )
    elif tag == "JJEPA":
        data_files = glob.glob(
            f"/j-jepa-vol/J-JEPA/data/JetClass/ptcl/{percent}%/{flag}/*"
        )
    return data_files


def load_data_torch(data_files):
    data = []
    for file in data_files:
        tensor = torch.load(file)
        data.append(tensor)
        print(f"--- loaded {file}")
    return data


part_feature_names = [
    "part_px",
    "part_py",
    "part_pz",
    "part_deta",
    "part_dphi",
    "part_pt_log",
    "part_e_log",
]


def load_data_h5(data_files):
    all_particles = {name: [] for name in part_feature_names}
    all_labels = []
    all_mask = []
    with h5py.File(data_files[0], "r") as hdf:
        stats = {name: hdf["stats"][name][:] for name in hdf["stats"]}

    for file in data_files:
        with h5py.File(file, "r") as hdf:
            for name in all_particles.keys():
                all_particles[name].append(hdf["particles"][name][:])
            all_labels.append(hdf["labels"][:])
            all_mask.append(hdf["mask"][:])

    # Concatenate the lists into arrays
    for name in all_particles.keys():
        all_particles[name] = np.concatenate(all_particles[name], axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_mask = np.concatenate(all_mask, axis=0)
    return all_particles, all_labels, all_mask, stats


def shuffle_and_save_torch(
    data_file_paths, label_file_paths, save_dir_data, save_dir_label
):
    assert len(data_file_paths) == len(
        label_file_paths
    ), "Data and label files count must be equal."

    indices = list(range(len(data_file_paths)))
    random.shuffle(indices)

    for j in range(0, len(indices), 10):
        print(j)
        batch_indices = indices[j : j + 10]
        # print(batch_indices)
        data_batch_paths = [data_file_paths[index] for index in batch_indices]
        label_batch_paths = [label_file_paths[index] for index in batch_indices]

        data_content = load_data_torch(data_batch_paths)
        label_content = load_data_torch(label_batch_paths)
        data_torch = torch.cat(data_content)
        labels_torch = torch.cat(label_content)

        # Generate shuffled indices
        jet_indices = torch.randperm(data_torch.size(0))
        # Apply shuffled indices to both data and labels
        data_torch_shuffled = data_torch[jet_indices]
        labels_torch_shuffled = labels_torch[jet_indices]

        # To ensure continuous indices for saved files
        save_indices = list(range(j, j + 10))
        save_tensors_in_chunks(
            data_torch_shuffled,
            labels_torch_shuffled,
            save_dir_data,
            save_dir_label,
            save_indices,
        )

        # Explicitly delete objects to free up memory
        del (
            data_batch_paths,
            label_batch_paths,
            data_content,
            label_content,
            data_torch,
            labels_torch,
            data_torch_shuffled,
            labels_torch_shuffled,
        )
        gc.collect()  # Force garbage collection


def shuffle_and_save_h5(data_file_paths, save_dir):
    indices = list(range(len(data_file_paths)))
    random.shuffle(indices)

    for j in range(0, len(indices), 10):
        print(j)
        batch_indices = indices[j : j + 10]
        # print(batch_indices)
        data_batch_paths = [data_file_paths[index] for index in batch_indices]

        part_batch, labels_batch, mask_batch, stats = load_data_h5(data_batch_paths)
        # shuffle data
        jet_indices = np.random.permutation(part_batch["part_px"].shape[0])
        for name in part_batch.keys():
            part_batch[name] = part_batch[name][jet_indices]
        labels_batch = labels_batch[jet_indices]
        mask_batch = mask_batch[jet_indices]

        save_indices = list(range(j, j + 10))
        save_h5_in_chunks(
            part_batch, labels_batch, mask_batch, stats, save_dir, save_indices
        )

        # Explicitly delete objects to free up memory
        del part_batch, labels_batch, mask_batch
        gc.collect()  # Force garbage collection


def save_h5_in_chunks(
    part_batch, labels_batch, mask_batch, stats, save_dir, save_indices
):
    os.makedirs(save_dir, exist_ok=True)

    # Calculate the number of samples per file
    total_samples = part_batch["part_px"].shape[0]
    samples_per_file = 100000
    num_files = total_samples // samples_per_file

    if total_samples < samples_per_file:
        samples_per_file = total_samples
        num_files = 1

    for i in range(num_files):
        start_index = i * samples_per_file
        # Handle the last file which might have more samples due to rounding
        end_index = (i + 1) * samples_per_file if i < num_files - 1 else total_samples

        # Extract the current chunk for data and labels
        current_samples = end_index - start_index
        part_chunk = {
            name: np.empty((current_samples, 128)) for name in part_batch.keys()
        }
        label_chunk = np.empty((current_samples, 10))
        mask_chunk = np.empty((current_samples, 128))

        for name in part_batch.keys():
            part_chunk[name][:] = part_batch[name][start_index:end_index]
        label_chunk[:] = labels_batch[start_index:end_index]
        mask_chunk[:] = mask_batch[start_index:end_index]

        # Save the current chunk to file
        with h5py.File(f"{save_dir}/data_{save_indices[i]}.h5", "w") as hdf:
            particles_group = hdf.create_group("particles")
            for name in part_chunk.keys():
                particles_group.create_dataset(name, data=part_chunk[name])
            hdf.create_dataset("labels", data=label_chunk)
            hdf.create_dataset("mask", data=mask_chunk)
            stats_group = hdf.create_group("stats")
            for name in stats.keys():
                stats_group.create_dataset(name, data=stats[name])

        print(f"saved file {save_indices[i]}")
        del part_chunk, label_chunk, mask_chunk
        gc.collect()
    del part_batch, labels_batch, mask_batch
    gc.collect()


def save_tensors_in_chunks(
    data_tensor, label_tensor, save_dir_data, save_dir_label, save_indices
):
    # print("shape of data tensor", data_tensor.shape)
    # Ensure the save directories exist
    os.makedirs(save_dir_data, exist_ok=True)
    os.makedirs(save_dir_label, exist_ok=True)

    # Calculate the number of samples per file
    total_samples = data_tensor.shape[0]
    samples_per_file = 100000
    num_files = total_samples // samples_per_file

    if total_samples < samples_per_file:
        samples_per_file = total_samples
        num_files = 1

    for i in range(num_files):
        start_index = i * samples_per_file
        # Handle the last file which might have more samples due to rounding
        end_index = (i + 1) * samples_per_file if i < num_files - 1 else total_samples
        # print("samples_per_file", samples_per_file)
        # print("start", start_index)
        # print("end", end_index)
        current_samples = end_index - start_index
        # Extract the current chunk for data and labels
        data_chunk = torch.empty((current_samples, 6, 128))
        label_chunk = torch.empty((current_samples, 10))
        data_chunk[:] = data_tensor[start_index:end_index]
        label_chunk[:] = label_tensor[start_index:end_index]

        # Save the current chunk to file
        torch.save(
            data_chunk, os.path.join(save_dir_data, f"data_{save_indices[i]}.pt")
        )
        torch.save(
            label_chunk, os.path.join(save_dir_label, f"label_{save_indices[i]}.pt")
        )
        print(f"saved file {save_indices[i]}")
        del data_chunk, label_chunk
        gc.collect()
    del data_tensor, label_tensor
    gc.collect()


def main(args):
    flag = args.flag
    for percent in [1, 5, 10, 50, 100]:
        print(f"Processing {percent}% of data")
        data_file_paths = get_data_file_paths(flag, args.tag, percent)
        print(f"Number of files: {len(data_file_paths)}")
        if args.tag == "JetCLR":
            label_file_paths = [
                path.replace("data/data", "label/labels") for path in data_file_paths
            ]
            if percent == 100:
                label_file_paths = [
                    path.replace("data/", "label/labels_") for path in data_file_paths
                ]

        if args.tag == "JetCLR":
            save_dir = (
                f"/j-jepa-vol/JetClass/processed/JetCLR/shuffled/{percent}%/{flag}/"
            )
            save_dir_data = f"{save_dir}/data"
            save_dir_label = f"{save_dir}/label"
            os.makedirs(save_dir_data, exist_ok=True)
            os.makedirs(save_dir_label, exist_ok=True)
            shuffle_and_save_torch(
                data_file_paths, label_file_paths, save_dir_data, save_dir_label
            )
        elif args.tag == "JJEPA":
            save_dir = (
                f"/j-jepa-vol/J-JEPA/data/JetClass/ptcl/shuffled/{percent}%/{flag}/"
            )
            os.makedirs(save_dir, exist_ok=True)
            shuffle_and_save_h5(data_file_paths, save_dir)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--flag",
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
    args = parser.parse_args()
    main(args)
