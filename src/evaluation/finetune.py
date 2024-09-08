#!/bin/env python3.7

# load standard python modules
import sys

sys.path.insert(0, "../src")
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import glob
import argparse
import copy
import tqdm
import gc
from pathlib import Path
import math
from tqdm import tqdm

# load torch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

from sklearn.metrics import accuracy_score
from sklearn import metrics


from src.models.jjepa import JJEPA
from src.options import Options
from src.dataset.JetDataset import JetDataset


# set the number of threads that pytorch will use
torch.set_num_threads(2)


# the MLP projector
def Projector(mlp, embedding):
    mlp_spec = f"{embedding}-{mlp}"
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)


# load data
def load_data(dataset_path, tag=None):
    # data_dir = f"{dataset_path}/{flag}/processed/4_features"
    if tag == "train":
        num_jets = 100 * 1000
    elif tag == "val":
        num_jets = 10 * 1000
    else:
        raise ValueError("tag must be either train or val")
    datset = JetDataset(dataset_path, labels=True, num_jets=num_jets)
    dataloader = DataLoader(datset, batch_size=args.batch_size, shuffle=True)
    return dataloader


def load_model(model_path=None, device="cpu"):
    options = Options(args.option_file)
    model = JJEPA(options).to(device)
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device))
    print(model)
    return model


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def get_perf_stats(labels, measures):
    measures = np.nan_to_num(measures)
    auc = metrics.roc_auc_score(labels, measures)
    fpr, tpr, _ = metrics.roc_curve(labels, measures)
    fpr2 = [fpr[i] for i in range(len(fpr)) if tpr[i] >= 0.5]
    tpr2 = [tpr[i] for i in range(len(tpr)) if tpr[i] >= 0.5]
    try:
        imtafe = np.nan_to_num(
            1 / fpr2[list(tpr2).index(find_nearest(list(tpr2), 0.5))]
        )
    except:
        imtafe = 1
    return auc, imtafe


def main(args):
    t0 = time.time()
    # set up results directory
    if args.flatten:
        args.output_dim *= 20
    out_dir = args.out_dir
    args.opt = "adam"
    args.learning_rate = 0.00005 * args.batch_size / 128

    # check if experiment already exists and is not empty

    if os.path.isdir(out_dir) and os.listdir(out_dir):
        sys.exit(
            "ERROR: experiment already exists and is not empty, don't want to overwrite it by mistake"
        )
    else:
        # This will create the directory if it does not exist or if it is empty
        os.makedirs(out_dir, exist_ok=True)

    # initialise logfile
    args.logfile = out_dir + "logfile.txt"
    logfile = open(args.logfile, "a")
    print("logfile initialised", file=logfile, flush=True)

    # define the global base device
    world_size = torch.cuda.device_count()
    if world_size:
        device = torch.device("cuda:0")
        for i in range(world_size):
            print(
                f"Device {i}: {torch.cuda.get_device_name(i)}", file=logfile, flush=True
            )
    else:
        device = torch.device("cpu")
        print("Device: CPU", file=logfile, flush=True)
    args.device = device

    # print purpose of experiment
    if "from_scratch" in args.label:
        print("training from scratch", file=logfile, flush=True)
        args.finetune = 1
    print(f"finetune: {args.finetune}", file=logfile, flush=True)

    print("loading data")
    train_dataloader = load_data(args.train_dataset_path, "train")
    val_dataloader = load_data(args.val_dataset_path, "val")

    t1 = time.time()

    print(
        "time taken to load and preprocess data: "
        + str(np.round(t1 - t0, 2))
        + " seconds",
        flush=True,
        file=logfile,
    )

    # initialise the network
    model = load_model(args.load_jjepa_path, args.device)
    net = model.context_transformer

    # initialize the MLP projector
    finetune_mlp_dim = args.output_dim
    if args.finetune_mlp:
        finetune_mlp_dim = f"{args.output_dim}-{args.finetune_mlp}"
    proj = Projector(2, finetune_mlp_dim).to(args.device)
    print(f"finetune mlp: {proj}", flush=True, file=logfile)
    if args.finetune:
        optimizer = optim.Adam(
            [{"params": proj.parameters()}, {"params": net.parameters(), "lr": 1e-6}],
            lr=1e-4,
        )
        net.train()
    else:
        net.eval()
        optimizer = optim.Adam(proj.parameters(), lr=1e-4)

    loss = nn.CrossEntropyLoss(reduction="mean")

    l_val_best = 99999
    acc_val_best = 0
    rej_val_best = 0

    softmax = torch.nn.Softmax(dim=1)
    loss_train_all = []
    loss_val_all = []
    acc_val_all = []

    for epoch in range(args.n_epochs):
        # re-batch the data on each epoch

        # initialise timing stats
        te_start = time.time()
        te0 = time.time()

        # initialise lists to store batch stats
        losses_e = []
        losses_e_val = []
        predicted_e = []  # store the predicted labels by batch
        correct_e = []  # store the true labels by batch

        # the inner loop goes through the dataset batch by batch
        proj.train()
        pbar = tqdm(train_dataloader)
        for i, (x, _, subjets, _, subjet_mask, _, labels) in enumerate(pbar):
            optimizer.zero_grad()

            y = labels.to(args.device)
            x = x.view(x.shape[0], x.shape[1], -1)
            batch = {"particles": x.to(torch.float32)}
            reps = net(
                batch,
                subjet_mask.to(args.device),
                subjets_meta=subjets.to(args.device),
                split_mask=None,
            )
            if args.flatten:
                reps = reps.view(reps.shape[0], -1)
            elif args.sum:
                reps = reps.sum(dim=1)
            else:
                raise ValueError("No aggregation method specified")
            out = proj(reps)
            batch_loss = loss(out, y.long()).to(args.device)
            batch_loss.backward()
            optimizer.step()
            batch_loss = batch_loss.detach().cpu().item()
            losses_e.append(batch_loss)
        loss_e = np.mean(np.array(losses_e))
        loss_train_all.append(loss_e)

        te1 = time.time()
        print(f"Training done in {te1-te0} seconds", flush=True, file=logfile)
        te0 = time.time()

        # validation
        with torch.no_grad():
            proj.eval()
            pbar = tqdm(val_dataloader)
            for i, (x, _, subjets, _, subjet_mask, _, labels) in enumerate(pbar):
                y = labels.to(args.device)
            x = x.view(x.shape[0], x.shape[1], -1)
            batch = {"particles": x.to(torch.float32)}
            reps = net(
                batch,
                subjet_mask.to(args.device),
                subjets_meta=subjets.to(args.device),
                split_mask=None,
            )
            if args.flatten:
                reps = reps.view(reps.shape[0], -1)
            elif args.sum:
                reps = reps.sum(dim=1)
            else:
                raise ValueError("No aggregation method specified")
            out = proj(reps)
            batch_loss = loss(out, y.long()).detach().cpu().item()
            losses_e_val.append(batch_loss)
            predicted_e.append(softmax(out).cpu().data.numpy())
            correct_e.append(y.cpu().data)
            loss_e_val = np.mean(np.array(losses_e_val))
            loss_val_all.append(loss_e_val)

        te1 = time.time()
        print(
            f"validation done in {round(te1-te0, 1)} seconds", flush=True, file=logfile
        )

        print(
            "epoch: "
            + str(epoch)
            + ", loss: "
            + str(round(loss_train_all[-1], 5))
            + ", val loss: "
            + str(round(loss_val_all[-1], 5)),
            flush=True,
            file=logfile,
        )

        # get the predicted labels and true labels
        predicted = np.concatenate(predicted_e)
        target = np.concatenate(correct_e)

        # get the accuracy
        accuracy = accuracy_score(target, predicted[:, 1] > 0.5)
        print(
            "epoch: " + str(epoch) + ", accuracy: " + str(round(accuracy, 5)),
            flush=True,
            file=logfile,
        )
        acc_val_all.append(accuracy)

        # save the latest model
        if args.finetune:
            torch.save(net.state_dict(), out_dir + "jjepa_finetune_last" + ".pt")
        torch.save(proj.state_dict(), out_dir + "projector_finetune_last" + ".pt")

        # save the model if lowest val loss is achieved
        if loss_val_all[-1] < l_val_best:
            # print("new lowest val loss", flush=True, file=logfile)
            l_val_best = loss_val_all[-1]
            if args.finetune:
                torch.save(
                    net.state_dict(), out_dir + "jjepa_finetune_best_loss" + ".pt"
                )
            torch.save(
                proj.state_dict(), out_dir + "projector_finetune_best_loss" + ".pt"
            )
            np.save(
                f"{out_dir}validation_target_vals_loss.npy",
                target,
            )
            np.save(
                f"{out_dir}validation_predicted_vals_loss.npy",
                predicted,
            )
        # also save the model if highest val accuracy is achieved
        if acc_val_all[-1] > acc_val_best:
            print("new highest val accuracy", flush=True, file=logfile)
            acc_val_best = acc_val_all[-1]
            if args.finetune:
                torch.save(
                    net.state_dict(), out_dir + "jjepa_finetune_best_acc" + ".pt"
                )
            torch.save(
                proj.state_dict(), out_dir + "projector_finetune_best_acc" + ".pt"
            )
            np.save(
                f"{out_dir}validation_target_vals_acc.npy",
                target,
            )
            np.save(
                f"{out_dir}validation_predicted_vals_acc.npy",
                predicted,
            )
        # calculate the AUC and imtafe and output to the logfile
        auc, imtafe = get_perf_stats(target, predicted[:, 1])

        if imtafe > rej_val_best:
            print("new highest val rejection", flush=True, file=logfile)
            print(
                f"epoch: {epoch}, AUC: {auc}, IMTAFE: {imtafe}",
                flush=True,
                file=logfile,
            )
            rej_val_best = imtafe
            if args.finetune:
                torch.save(
                    net.state_dict(), out_dir + "jjepa_finetune_best_rej" + ".pt"
                )
            torch.save(
                proj.state_dict(), out_dir + "projector_finetune_best_rej" + ".pt"
            )
            np.save(
                f"{out_dir}validation_target_vals_rej.npy",
                target,
            )
            np.save(
                f"{out_dir}validation_predicted_vals_rej.npy",
                predicted,
            )

        # save all losses and accuracies
        np.save(
            f"{out_dir}loss_train.npy",
            np.array(loss_train_all),
        )
        np.save(
            f"{out_dir}loss_val.npy",
            np.array(loss_val_all),
        )
        np.save(
            f"{out_dir}acc_val.npy",
            np.array(acc_val_all),
        )
        te_end = time.time()
        print(
            f"epoch {epoch} done in {round(te_end - te_start, 1)} seconds",
            flush=True,
            file=logfile,
        )

    # Training done
    print("Training done", flush=True, file=logfile)


if __name__ == "__main__":
    """This is executed when run from the command line"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        default="",
        action="store",
        dest="out_dir",
        type=str,
        help="output directory",
    )
    parser.add_argument(
        "--option-file",
        default="",
        action="store",
        dest="option_file",
        type=str,
        help="option file for initializing JJEPA model",
    )
    parser.add_argument(
        "--finetune-mlp",
        default="",
        action="store",
        dest="finetune_mlp",
        type=str,
        help="Size and number of layers of the MLP finetuning head following output_dim of model, e.g. 512-256-128",
    )
    parser.add_argument(
        "--finetune",
        type=int,
        action="store",
        help="keep the transformer frozen and only train the MLP head",
    )
    parser.add_argument(
        "--train-dataset-path",
        type=str,
        action="store",
        default="/ssl-jet-vol-v3/toptagging/processed",
        help="Input directory with the dataset",
    )
    parser.add_argument(
        "--val-dataset-path",
        type=str,
        action="store",
        default="/ssl-jet-vol-v3/toptagging/processed",
        help="Input directory with the dataset",
    )
    parser.add_argument(
        "--load-jjepa-path",
        type=str,
        action="store",
        default=None,
        help="Load weights from JJEPA model if enabled",
    )
    parser.add_argument(
        "--output-dim",
        type=int,
        action="store",
        dest="output_dim",
        default=1024,
        help="dimension of the output of transformer-encoder",
    )
    parser.add_argument(
        "--n-epoch",
        type=int,
        action="store",
        dest="n_epochs",
        default=300,
        help="Epochs",
    )
    parser.add_argument(
        "--label",
        type=str,
        action="store",
        dest="label",
        default="new",
        help="label of the model to load",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        action="store",
        dest="batch_size",
        default=32,
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
