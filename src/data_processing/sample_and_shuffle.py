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
import os
import gc

import h5py
import random
import shutil
from tqdm import tqdm

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def modify_path(path):
    """
    Given a path for a data file, e.g. path = '/j-jepa-vol/JetClass/processed/val/data/HToBB_123.pt',
    Constructs the path for the corresponding label file, e.g. new_path = '/j-jepa-vol/JetClass/processed/val/label/labels_HToBB_123.pt'
    """
    # Split the string into parts
    parts = path.split("/")

    # Replace 'data' with 'label' in the second-to-last part
    parts[-2] = "label"

    # Insert 'labels_' before the last part
    parts[-1] = "labels_" + parts[-1]

    # Join the parts back together
    new_path = "/".join(parts)
    return new_path


def load_data_h5(data_files):
    """Load a batch of H5 files into memory"""
    # First file to determine available features and structure
    with h5py.File(data_files[0], "r") as hdf:
        part_feature_names = list(hdf["particles"].keys())
        # Check if stats group exists, handle case where it might not
        stats = {}
        if "stats" in hdf:
            stats = {name: hdf["stats"][name][:] for name in hdf["stats"]}

    # If stats not found in first file, check other files
    if not stats:
        for file in data_files[1:]:
            try:
                with h5py.File(file, "r") as hdf:
                    if "stats" in hdf:
                        stats = {name: hdf["stats"][name][:] for name in hdf["stats"]}
                        print(f"Found stats in {file}")
                        break
            except Exception as e:
                print(f"Error checking stats in {file}: {e}")

    # Initialize containers for data
    all_particles = {name: [] for name in part_feature_names}
    all_labels = []
    all_mask = []

    # Load each file
    for file in tqdm(data_files, desc="Loading files"):
        with h5py.File(file, "r") as hdf:
            for name in part_feature_names:
                all_particles[name].append(hdf["particles"][name][:])
            all_labels.append(hdf["labels"][:])
            all_mask.append(hdf["mask"][:])

    # Concatenate the lists into arrays
    for name in all_particles.keys():
        all_particles[name] = np.concatenate(all_particles[name], axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_mask = np.concatenate(all_mask, axis=0)

    return all_particles, all_labels, all_mask, stats, part_feature_names


def save_h5_in_chunks(
    part_batch, labels_batch, mask_batch, stats, save_dir, batch_size=100000
):
    """Save data in chunks of specified size"""
    os.makedirs(save_dir, exist_ok=True)

    # Calculate the number of samples per file
    total_samples = labels_batch.shape[0]
    samples_per_file = batch_size
    num_files = (
        total_samples + samples_per_file - 1
    ) // samples_per_file  # Ceiling division

    for i in range(num_files):
        start_index = i * samples_per_file
        # Handle the last file which might have fewer samples
        end_index = min((i + 1) * samples_per_file, total_samples)

        # Extract the current chunk
        current_samples = end_index - start_index

        # Create the output file
        with h5py.File(f"{save_dir}/data_{i}.h5", "w") as hdf:
            # Create groups
            particles_group = hdf.create_group("particles")

            # Save particle features for this chunk
            for name in part_batch.keys():
                particles_group.create_dataset(
                    name, data=part_batch[name][start_index:end_index]
                )

            # Save labels and mask
            hdf.create_dataset("labels", data=labels_batch[start_index:end_index])
            hdf.create_dataset("mask", data=mask_batch[start_index:end_index])

            # Save stats in every file if they exist
            if stats:
                stats_group = hdf.create_group("stats")
                for name in stats.keys():
                    stats_group.create_dataset(name, data=stats[name])

        print(f"Saved chunk {i+1}/{num_files} with {current_samples} samples")


def shuffle_dataset(input_dir, batch_size=10):
    """Shuffle a dataset by loading in batches and shuffling"""
    print(f"Shuffling dataset in {input_dir}")

    # Get all data files
    data_files = sorted(glob.glob(f"{input_dir}/*.h5"))
    if not data_files:
        print(f"No .h5 files found in {input_dir}")
        return

    # Create a temporary directory for shuffled results
    temp_dir = f"{input_dir}_temp_shuffle"
    os.makedirs(temp_dir, exist_ok=True)

    # Track original file indices to maintain the same numbering
    file_indices = [
        int(os.path.basename(f).split("_")[1].split(".")[0]) for f in data_files
    ]

    # Process files in batches to manage memory
    for batch_start in range(0, len(data_files), batch_size):
        batch_end = min(batch_start + batch_size, len(data_files))
        batch_files = data_files[batch_start:batch_end]

        print(
            f"Processing batch {batch_start//batch_size + 1}, files {batch_start} to {batch_end-1}"
        )

        # Load batch data
        part_batch, labels_batch, mask_batch, stats, part_feature_names = load_data_h5(
            batch_files
        )

        # Shuffle all data with the same random indices
        shuffle_indices = np.random.permutation(labels_batch.shape[0])
        for name in part_batch.keys():
            part_batch[name] = part_batch[name][shuffle_indices]
        labels_batch = labels_batch[shuffle_indices]
        mask_batch = mask_batch[shuffle_indices]

        # Save shuffled data - use original file indices to maintain numbering
        save_batch_with_indices(
            part_batch,
            labels_batch,
            mask_batch,
            stats,
            temp_dir,
            file_indices[batch_start:batch_end],
            batch_size=100000,
        )

        # Free memory
        del part_batch, labels_batch, mask_batch
        gc.collect()

    # Remove original directory and rename temp directory
    shutil.rmtree(input_dir)
    os.rename(temp_dir, input_dir)
    print(f"Shuffling complete, replaced {input_dir} with shuffled data")


def save_batch_with_indices(
    part_batch,
    labels_batch,
    mask_batch,
    stats,
    save_dir,
    file_indices,
    batch_size=100000,
):
    """Save data in chunks, preserving original file indices"""
    os.makedirs(save_dir, exist_ok=True)

    # Calculate the number of samples per file
    total_samples = labels_batch.shape[0]
    samples_per_file = batch_size
    num_files = len(file_indices)

    # Calculate samples per original file (approximately even distribution)
    samples_per_original_file = total_samples // num_files
    remainder = total_samples % num_files

    start_index = 0
    for i, file_idx in enumerate(file_indices):
        # Calculate how many samples go in this file (distribute remainder if any)
        this_file_samples = samples_per_original_file + (1 if i < remainder else 0)
        end_index = min(start_index + this_file_samples, total_samples)

        # Skip if no samples for this file
        if end_index <= start_index:
            continue

        # Create the output file
        with h5py.File(f"{save_dir}/data_{file_idx}.h5", "w") as hdf:
            # Create groups
            particles_group = hdf.create_group("particles")

            # Save particle features for this chunk
            for name in part_batch.keys():
                particles_group.create_dataset(
                    name, data=part_batch[name][start_index:end_index]
                )

            # Save labels and mask
            hdf.create_dataset("labels", data=labels_batch[start_index:end_index])
            hdf.create_dataset("mask", data=mask_batch[start_index:end_index])

            # Save stats in every file if they exist
            if stats:
                stats_group = hdf.create_group("stats")
                for name in stats.keys():
                    stats_group.create_dataset(name, data=stats[name])

        print(f"Saved file data_{file_idx}.h5 with {end_index - start_index} samples")
        start_index = end_index


def main(args):
    """
    Samples a fraction of jets from all data files, then shuffles the data.
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    label = args.label
    if args.tag == "JetCLR":
        data_dir = f"/j-jepa-vol/JetClass/processed/JetCLR/{label}"
        data_files = glob.glob(f"{data_dir}/data/*")
    elif args.tag == "JJEPA":
        data_dir = f"/j-jepa-vol/J-JEPA/data/JetClass/ptcl/{label}"
        data_files = glob.glob(f"{data_dir}/*.h5")

    frac_lst = [1, 5, 10, 50]
    total_samples = 100000  # 100k jets per file

    # Initialize file_counter outside the frac loop to maintain it across fractions
    file_counter = 0

    for frac in frac_lst:
        file_counter = 0
        print(f"Sampling {frac}% of data from `{label}` directory")
        if args.tag == "JetCLR":
            processed_data_dir = (
                f"/j-jepa-vol/JetClass/processed/JetCLR/{frac}%/{label}/data"
            )
            processed_label_dir = (
                f"/j-jepa-vol/JetClass/processed/JetCLR/{frac}%/{label}/label"
            )
            os.system(
                f"mkdir -p {processed_data_dir} {processed_label_dir}"
            )  # -p: create parent dirs if needed, exist_ok

            data_shape = (total_samples, 6, 128)  # Example data shape
            label_shape = (total_samples, 10)  # Example label shape, adjust as needed

            # Pre-allocate tensors
            temp_sampled_data = torch.zeros(
                data_shape, dtype=torch.float32
            )  # Adjust dtype as needed
            temp_sampled_labels = torch.zeros(
                label_shape, dtype=torch.int64
            )  # Adjust dtype as needed

            for i, file in enumerate(data_files):
                data = torch.load(file)  # Assume shape is (N, 7, 128) for each file
                data_file_name = file.split("/")[-1].split(".")[0]
                print(
                    f"--- loaded data file {i} {data_file_name} from `{label}` directory"
                )

                label_path = modify_path(file)
                labels = torch.load(label_path)  # Assume shape is (N,) for each file

                # Calculate number of samples to take
                num_samples = int(frac / 100 * data.shape[0])
                # Generate random indices
                indices = torch.randperm(data.shape[0])[:num_samples]

                # Fill pre-allocated tensors
                # Calculate new end index based on current batch
                end_index = current_index + num_samples

                # Check if the temp tensors can accommodate this batch; if not, save and reset
                if end_index > temp_sampled_data.shape[0]:
                    # Save the filled portion of the tensors
                    torch.save(
                        temp_sampled_data[:current_index],
                        os.path.join(processed_data_dir, f"data_{file_counter}.pt"),
                    )
                    torch.save(
                        temp_sampled_labels[:current_index],
                        os.path.join(processed_label_dir, f"labels_{file_counter}.pt"),
                    )
                    file_counter += 1  # Increment file counter after saving
                    print(f"----finished creating {file_counter} files")

                    # Reset current index for the next batch of tensors
                    current_index = 0
                    end_index = num_samples  # As we are starting from 0 again

                    # Reset current index for the next batch of tensors, only if not at the last file
                    if i != len(data_files) - 1:
                        print("resetting  temp storage tensors")
                        temp_sampled_data.fill_(0)  # Resetting tensors for reuse
                        temp_sampled_labels.fill_(0)

                # Fill pre-allocated tensors with the current batch
                temp_sampled_data[current_index:end_index] = data[indices]
                temp_sampled_labels[current_index:end_index] = labels[indices]

                current_index = end_index  # Update current index

                # Condition to save at the last file
                if i == len(data_files) - 1:
                    # Save the filled portion of the tensors
                    torch.save(
                        temp_sampled_data[:current_index],
                        os.path.join(processed_data_dir, f"data_{file_counter}.pt"),
                    )
                    torch.save(
                        temp_sampled_labels[:current_index],
                        os.path.join(processed_label_dir, f"labels_{file_counter}.pt"),
                    )
                    file_counter += 1  # Increment file counter after saving
                # delete the data and labels to free up memory
                del data, labels
                gc.collect()

            # TODO: Add JetCLR shuffling if needed

        elif args.tag == "JJEPA":
            processed_dir = f"/j-jepa-vol/J-JEPA/data/JetClass/ptcl/{frac}%/{label}"
            os.system(f"mkdir -p {processed_dir}")

            # Determine feature names and stats from the first file
            first_file = data_files[0]
            with h5py.File(first_file, "r") as hdf:
                part_feature_names = list(hdf["particles"].keys())
                stats = {name: hdf["stats"][name][:] for name in hdf["stats"]}

            print(
                f"Found {len(part_feature_names)} particle features: {part_feature_names}"
            )

            data_shape = (total_samples, 128)  # Base shape for particle features
            label_shape = (total_samples, 10)  # Shape for labels

            # Initialize temporary arrays for all features
            temp_particles = {name: np.zeros(data_shape) for name in part_feature_names}
            temp_labels = np.zeros(label_shape)
            temp_mask = np.zeros(data_shape)

            # Initialize counters once per fraction so that file numbering starts at 0 and increments
            file_counter = 0
            current_index = 0

            for i, file in enumerate(data_files):
                with h5py.File(file, "r") as hdf:
                    particles = {
                        name: hdf["particles"][name][:] for name in part_feature_names
                    }
                    labels = hdf["labels"][:]
                    mask = hdf["mask"][:]

                data_file_name = file.split("/")[-1].split(".")[0]
                print(
                    f"--- loaded data file {i} {data_file_name} from `{label}` directory"
                )

                num_samples = int(frac / 100 * mask.shape[0])
                indices = np.random.choice(mask.shape[0], num_samples, replace=False)
                end_index = current_index + num_samples

                # If the current batch overflows the preallocated space, save the filled portion to disk.
                if end_index > temp_mask.shape[0]:
                    with h5py.File(
                        f"{processed_dir}/data_{file_counter}.h5", "w"
                    ) as hdf:
                        particles_group = hdf.create_group("particles")
                        stats_group = hdf.create_group("stats")
                        for name in part_feature_names:
                            particles_group.create_dataset(
                                name, data=temp_particles[name][:current_index]
                            )
                        hdf.create_dataset("labels", data=temp_labels[:current_index])
                        hdf.create_dataset("mask", data=temp_mask[:current_index])
                        for name in stats:
                            stats_group.create_dataset(name, data=stats[name])
                    print(
                        f"----finished creating file data_{file_counter}.h5 with {current_index} samples"
                    )
                    file_counter += 1
                    # Reset for the next batch within the same fraction
                    current_index = 0
                    end_index = num_samples
                    if i != len(data_files) - 1:
                        print("resetting temporary storage tensors")
                        temp_particles = {
                            name: np.zeros(data_shape) for name in part_feature_names
                        }
                        temp_labels = np.zeros(label_shape)
                        temp_mask = np.zeros(data_shape)

                # Fill the preallocated arrays with the current file's sampled data
                for name in part_feature_names:
                    temp_particles[name][current_index:end_index] = particles[name][
                        indices
                    ]
                temp_labels[current_index:end_index] = labels[indices]
                temp_mask[current_index:end_index] = mask[indices]
                current_index = end_index

                # At the very end, save any remaining data in the temporary arrays
                if i == len(data_files) - 1:
                    with h5py.File(
                        f"{processed_dir}/data_{file_counter}.h5", "w"
                    ) as hdf:
                        particles_group = hdf.create_group("particles")
                        stats_group = hdf.create_group("stats")
                        for name in part_feature_names:
                            particles_group.create_dataset(
                                name, data=temp_particles[name][:current_index]
                            )
                        hdf.create_dataset("labels", data=temp_labels[:current_index])
                        hdf.create_dataset("mask", data=temp_mask[:current_index])
                        for name in stats:
                            stats_group.create_dataset(name, data=stats[name])
                    print(
                        f"----finished creating file data_{file_counter}.h5 with {current_index} samples"
                    )
                    file_counter += 1

                del particles, labels, mask
                gc.collect()

            print(f"Now shuffling the sampled {frac}% data...")
            shuffle_dataset(processed_dir, batch_size=5)

        # Reset for the next fraction
        current_index = 0
        file_counter = 0  # Reset file counter for the next fraction
        print(f"Finished sampling, shuffling, and saving {frac}% of data")


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
        default="JJEPA",
        help="JetCLR/JJEPA",
    )
    args = parser.parse_args()
    main(args)
