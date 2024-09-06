""" 
The full evaluation pipeline for JJEPA.
1. Plot training and validation losses for the trained model.
2. Generate histograms for the learned representations using PCA.
4. Generate t-SNE plots for the learned representations.
"""

# load standard python modules
import argparse
from datetime import datetime
import copy
import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import tqdm
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.manifold import TSNE
from pathlib import Path

# load torch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.jjepa import JJEPA
from src.options import Options

project_dir = Path(__file__).resolve().parents[2]
print(f"project_dir: {project_dir}")  # /ssl-jet-vol-v3/I-JEPA-Jets


# load the data files and the label files from the specified directory
# TODO: generate 4 feature dataset
def load_data(dataset_path, flag, n_files=-1):
    data_dir = f"{dataset_path}/{flag}/processed/4_features"
    data_files = glob.glob(f"{data_dir}/*")

    data = []
    for i, file in enumerate(data_files):
        data += torch.load(f"{dataset_path}/{flag}/processed/4_features/data_{i}.pt")
        print(f"--- loaded data file {i} from `{flag}` directory")
        if n_files != -1 and i == n_files - 1:
            break

    return data


def load_labels(dataset_path, flag, n_files=-1):
    data_dir = f"{dataset_path}/{flag}/processed/4_features"
    data_files = glob.glob(f"{data_dir}/*")

    data = []
    for i, file in enumerate(data_files):
        data += torch.load(f"{dataset_path}/{flag}/processed/4_features/labels_{i}.pt")
        print(f"--- loaded label file {i} from `{flag}` directory")
        if n_files != -1 and i == n_files - 1:
            break

    return data


def load_model(model_path=None, device="cpu"):
    options = Options()
    model = JJEPA(options).to(device)
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device))
    return model


def plot_losses(args):
    pass


if __name__ == "__main__":
    """This is executed when run from the command line"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-path",
        type=str,
        action="store",
        default=f"{project_dir}/data",
        help="Input directory with the dataset for evaluation",
    )
    parser.add_argument(
        "--eval-path",
        type=str,
        action="store",
        default=f"{project_dir}/models/model_performances/Top_Tagging/",
        help="the evaluation results will be saved at eval-path/label",
    )
    parser.add_argument(
        "--load-jjepa-path",
        type=str,
        action="store",
        default=f"{project_dir}/models/trained_models/Top_Tagging/",
        help="Load weights from JJEPA model if enabled",
    )
    parser.add_argument(
        "--num-test-files",
        type=int,
        default=1,
        help="Number of files to use for testing",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        action="store",
        dest="outdir",
        default=f"{project_dir}/models/",
        help="Output directory",
    )
    parser.add_argument(
        "--label",
        type=str,
        action="store",
        dest="label",
        default="new",
        help="a label for the model used for inference",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        action="store",
        dest="batch_size",
        default=2048,
        help="batch_size",
    )

    args = parser.parse_args()
    main(args)
