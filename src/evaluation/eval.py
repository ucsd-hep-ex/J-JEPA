""" 
The full evaluation pipeline for JJEPA.
1. Plot training and validation losses for the trained model.
2. Generate histograms for the learned representations using PCA.
3. Generate t-SNE plots for the learned representations.
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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pathlib import Path

# load torch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.models.jjepa import JJEPA
from src.options import Options
from src.dataset.JetDataset import JetDataset

project_dir = Path(__file__).resolve().parents[2]
print(f"project_dir: {project_dir}")  # /ssl-jet-vol-v3/I-JEPA-Jets


# load the data files and the label files from the specified directory
def load_data(dataset_path):
    # data_dir = f"{dataset_path}/{flag}/processed/4_features"
    datset = JetDataset(dataset_path, labels=True)
    dataloader = DataLoader(datset, batch_size=args.batch_size, shuffle=False)
    return dataloader


def load_model(model_path=None, device="cpu"):
    options = Options.load("/mnt/d/physic/billy_output/mock_config.json")
    model = JJEPA(options).to(device)
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device))
    print(model)
    return model


def plot_losses(args):
    try:
        losses_train = np.load(f"train_losses.npy")
        losses_val = np.load(f"val_losses.npy")
    except:
        print("No losses found")
        return
    plt.plot(losses_train, label="train")
    plt.xlabel("Epoch")  # x-axis label
    plt.ylabel("Loss")  # y-axis label
    plt.legend()
    plt.savefig(f"{args.eval_path}/train_losses.png", dpi=300)
    plt.close()
    plt.plot(losses_val, label="val")
    plt.xlabel("Epoch")  # x-axis label
    plt.ylabel("Loss")  # y-axis label
    plt.legend()
    plt.savefig(f"{args.eval_path}/val_losses.png", dpi=300)


def obtain_reps(net, dataloader, args):
    with torch.no_grad():
        net.eval()
        all_reps = []
        pbar = tqdm.tqdm(dataloader)
        for i, (x, _, subjets, _, subjet_mask, _, labels) in enumerate(pbar):
            x = x.view(x.shape[0], x.shape[1], -1).to(args.device)
            batch = {"particles": x.to(torch.float32)}
            reps = net(
                batch,
                subjet_mask.to(args.device),
                subjets_meta=subjets.to(args.device),
                split_mask=None,
            )
            all_reps.append(reps)
            pbar.set_description(f"{i}")
        all_reps = torch.concatenate(all_reps)
        if args.flatten:
            all_reps = all_reps.view(all_reps.shape[0], -1)
        elif args.sum:
            all_reps = all_reps.sum(dim=1)
        else:
            raise ValueError("No aggregation method specified")
        print(all_reps.shape)
    return all_reps


def plot_tsne(args, net, label, dataloader, n_pca=None):
    # Assuming your data tensor is named 'data' and has a shape of [num_samples, emb_dim]
    # Flatten the data if needed and convert it to numpy
    data_test = obtain_reps(net, dataloader, args)
    if n_pca is not None:
        data_test = do_pca(data_test, n_components=n_pca)
        
    data_test = obtain_reps(net, dataloader, args)
    data_numpy = data_test.to("cpu").numpy()
    labels_numpy = torch.cat([batch[-1] for batch in dataloader], dim=0).numpy()

    for perp in reversed(range(5, 60, 5)):
        for n_iter in [5000]:
            print(f"perp: {perp}, n_iter: {n_iter}")
            # Apply t-SNE
            tsne = TSNE(
                n_components=2, perplexity=perp, n_iter=n_iter
            )  # you can change these hyperparameters as needed
            tsne_results = tsne.fit_transform(data_numpy)

            # tsne_results now has a shape of [num_samples, 2], and you can plot it

            # Use boolean indexing to separate points for each label
            top_points = tsne_results[labels_numpy == 1]
            qcd_points = tsne_results[labels_numpy == 0]

            plt.figure(figsize=(10, 10))
            ax = plt.gca()
            # Plot each class with a different color and label
            plt.scatter(
                top_points[:, 0],
                top_points[:, 1],
                color="b",
                alpha=0.5,
                label="top",
                s=1,
            )
            plt.scatter(
                qcd_points[:, 0],
                qcd_points[:, 1],
                color="y",
                alpha=0.5,
                label="QCD",
                s=1,
            )
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            plt.title("t-SNE visualization of jet features")
            plt.legend(loc="upper right")  # place the legend at the upper right corner
            plt.savefig(f"{args.eval_path}/tsne_{label}_{perp}_{n_iter}.png", dpi=300)
            # plt.show()
            plt.close()


def plot_top_and_qcd_features(data_top, data_qcd, args, label):
    """
    Generates histograms for each feature from two datasets, 'top' and 'QCD',
    on the same canvas for direct comparison.

    Parameters:
    - data_top: A numpy array for the 'top' dataset where rows represent samples and columns represent features.
    - data_qcd: A numpy array for the 'QCD' dataset where rows represent samples and columns represent features.
    - args
    """
    # logfile = args.logfile
    num_feats = data_qcd.shape[1]
    num_columns = 2
    num_rows = int(np.ceil(num_feats / num_columns))

    # Adjust the figure size
    subplot_width = 10
    subplot_height = 4
    fig, axs = plt.subplots(
        num_rows,
        num_columns,
        figsize=(subplot_width * num_columns, subplot_height * num_rows),
    )

    # Style settings
    sns.set_style("whitegrid")
    colors = ["blue", "red"]

    for i in range(num_feats):
        row = i // num_columns
        col = i % num_columns
        ax = axs[row, col] if num_rows > 1 else axs[col]

        sns.histplot(
            data_top[:, i], bins=50, ax=ax, color=colors[0], label="Top", alpha=0.6
        )
        sns.histplot(
            data_qcd[:, i], bins=50, ax=ax, color=colors[1], label="QCD", alpha=0.6
        )

        ax.set_title(f"PCA Feature {i}")
        ax.legend()

        sns.despine(ax=ax)

    # If the number of features isn't a multiple of the columns, remove unused subplots
    if num_feats % num_columns != 0:
        for j in range(num_feats, num_rows * num_columns):
            axs.flatten()[j].axis("off")

    fig.suptitle("Top and QCD Feature Comparison", y=1.02)
    plt.tight_layout(pad=2.0)
    save_path = f"{args.eval_path}/Top_and_QCD_Feature_Comparison_{label}.png"
    plt.savefig(save_path)
    # plt.show()
    plt.close()


def split_data(dataloader):
    """
    Splits the dataset from the dataloader into two dataloaders: one for signal and one for background.
    The dataloader returns: (x, particle_features, subjets, indices, subjet_mask, particle_mask, labels).
    """

    # Initialize lists to hold data for signal and background
    signal_data = []
    background_data = []

    print("Splitting dataset into signal and background", flush=True)

    # Iterate over the dataloader and split based on labels
    for batch in dataloader:
        # Unpack the batch (x, particle_features, subjets, indices, subjet_mask, particle_mask, labels)
        x, particle_features, subjets, indices, subjet_mask, particle_mask, labels = (
            batch
        )

        # Split based on the labels (1 for signal, 0 for background)
        signal_mask = labels == 1
        background_mask = labels == 0

        # Append signal and background batches
        signal_data.append(
            (
                x[signal_mask],
                particle_features[signal_mask],
                subjets[signal_mask],
                indices[signal_mask],
                subjet_mask[signal_mask],
                particle_mask[signal_mask],
                labels[signal_mask],
            )
        )

        background_data.append(
            (
                x[background_mask],
                particle_features[background_mask],
                subjets[background_mask],
                indices[background_mask],
                subjet_mask[background_mask],
                particle_mask[background_mask],
                labels[background_mask],
            )
        )

    # Combine signal and background batches into tensors
    signal_tensors = [torch.cat(data, dim=0) for data in zip(*signal_data)]
    background_tensors = [torch.cat(data, dim=0) for data in zip(*background_data)]

    # Create datasets for signal and background
    signal_dataset = TensorDataset(*signal_tensors)
    background_dataset = TensorDataset(*background_tensors)

    # Create new dataloaders for signal and background
    signal_dataloader = DataLoader(
        signal_dataset, batch_size=dataloader.batch_size, shuffle=True
    )
    background_dataloader = DataLoader(
        background_dataset, batch_size=dataloader.batch_size, shuffle=True
    )

    return signal_dataloader, background_dataloader


def do_pca(X, n_components=8):
    # Assuming X is your 1000-dimensional dataset with shape (n_samples, 1000)
    # Step 1: Standardize the data
    print(f"Shape of X: {X.shape}")
    print("started PCA")
    scaler = StandardScaler()
    X = X.cpu()
    X_standardized = scaler.fit_transform(X)

    # Step 2: Apply PCA to reduce dimensionality to 8 components
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_standardized)
    # Now X_pca has the transformed data in the reduced dimensionality space
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Explained variance: {pca.explained_variance_}")
    return X_pca


def plot_pca(args, net, label, dataloader):
    # split the data into top and qcd
    signal_dataloader, background_dataloader = split_data(dataloader)
    # obtain representations for top and qcd
    reps_top = obtain_reps(net, signal_dataloader, args).detach().cpu().numpy()
    reps_qcd = obtain_reps(net, background_dataloader, args).detach().cpu().numpy()
    # do pca
    reps_top_pca = do_pca(reps_top)
    reps_qcd_pca = do_pca(reps_qcd)
    # plot top and qcd features
    plot_top_and_qcd_features(reps_top_pca, reps_qcd_pca, args, label)


def main(args):
    print(f"args.flatten: {args.flatten}")
    print(f"args.sum: {args.sum}")
    world_size = torch.cuda.device_count()
    if world_size:
        device = torch.device("cuda:0")
        for i in range(world_size):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        device = "cpu"
        print("Device: CPU")
    args.device = device

    dataloader = load_data(args.dataset_path)
    model = load_model(args.load_jjepa_path, args.device)
    context_encoder = model.context_transformer
    target_encoder = model.target_transformer
    print("-------------------")
    print("Plotting losses")
    plot_losses(args)
    print("-------------------")
    # plot t-SNE
    print("Making t-SNE plots")
    plot_tsne(args, context_encoder, "context", dataloader, n_pca=50)
    plot_tsne(args, target_encoder, "target", dataloader, n_pca=50)
    print("-------------------")
    # plot PCA
    print("Making PCA plots")
    plot_pca(args, context_encoder, "context")
    plot_pca(args, target_encoder, "target")
    print("-------------------")
    print("Evaluation complete")


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
    parser.add_argument(
        "--flatten",
        type=int,
        action="store",
        default=0,
        help="flatten the representation",
    )
    parser.add_argument(
        "--sum",
        type=int,
        action="store",
        default=1,
        help="sum the representation",
    )

    args = parser.parse_args()
    main(args)
