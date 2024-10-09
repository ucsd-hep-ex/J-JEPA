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