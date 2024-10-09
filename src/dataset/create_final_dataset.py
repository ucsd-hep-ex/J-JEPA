import os
import random
import h5py
from src.dataset.JetDataset import JetDataset
from src.dataset.JEPADataset import JEPADataset
import torch
from torch.utils.data import Dataset, DataLoader, Subset

from tqdm import tqdm
import numpy as np


# for tag in ["train", "val", "test"]:
for tag in ["val", "test"]:
    file_path = f"/j-jepa-vol/J-JEPA/data/top/normalized/semi-processed/{tag}/{tag}_20_30.h5"
    if os.path.exists(file_path):
        print(f"File exists: {file_path}")
    else:
        print(f"File does not exist: {file_path}")
    file_name = f"processed_{tag}_20_30.h5"
    dataset_path = f"/j-jepa-vol/J-JEPA/data/top/normalized/{tag}/"
    os.makedirs(dataset_path, exist_ok=True)
    out_path = dataset_path+file_name

    train_dataset = JetDataset(file_path)

    
    # Initialize DataLoader
    batch_size = 1000
    total_data = len(train_dataset)
    part_size = total_data // 3
    
    # Create three parts of the dataset
    datasets = [Subset(train_dataset, range(i * part_size, (i + 1) * part_size)) for i in range(3)]
    if total_data % 3 != 0:  # Handle the remainder
        datasets[-1] = Subset(train_dataset, range(2 * part_size, total_data))
    
    # Function to process and save each part
    def process_and_save(part_data, file_path):
        train_loader = DataLoader(part_data, batch_size=batch_size, shuffle=True)
        with h5py.File(file_path, 'w') as hf:
            first_batch = next(iter(train_loader))
            num_subjets = first_batch[0].shape[1]
            num_ptcls_per_subjet = first_batch[0].shape[2]
            num_ptcl_ftrs = first_batch[0].shape[3]
            num_ptcls_per_jet = first_batch[1].shape[1]
            num_subjet_ftrs = first_batch[2].shape[2]
    
            hf.create_dataset('x', shape=(0, num_subjets, num_ptcls_per_subjet, num_ptcl_ftrs), maxshape=(None, num_subjets, num_ptcls_per_subjet, num_ptcl_ftrs), dtype='float32')
            hf.create_dataset('particle_features', shape=(0, num_ptcls_per_jet, num_ptcl_ftrs), maxshape=(None, num_ptcls_per_jet, num_ptcl_ftrs), dtype='float64')
            hf.create_dataset('subjets', shape=(0, num_subjets, num_subjet_ftrs), maxshape=(None, num_subjets, num_subjet_ftrs), dtype='float64')
            hf.create_dataset('particle_indices', shape=(0, num_subjets, num_ptcls_per_subjet), maxshape=(None, num_subjets, num_ptcls_per_subjet), dtype='int32')
            hf.create_dataset('subjet_mask', shape=(0, num_subjets), maxshape=(None, num_subjets), dtype='bool')
            hf.create_dataset('particle_mask', shape=(0, num_subjets, num_ptcls_per_subjet), maxshape=(None, num_subjets, num_ptcls_per_subjet), dtype='bool')
    
            num_batches_processed = 0
            for data in tqdm(train_loader):
                x, particle_features, subjets, particle_indices, subjet_mask, particle_mask = [d.detach().cpu() for d in data]
                num_new = x.shape[0]
                particle_indices = particle_indices.to(torch.int32)
                subjet_mask = subjet_mask.to(torch.int32)
                particle_mask = particle_mask.to(torch.int32)
                
                hf['x'].resize(num_batches_processed * batch_size + num_new, axis=0)
                hf['particle_features'].resize(num_batches_processed * batch_size + num_new, axis=0)
                hf['subjets'].resize(num_batches_processed * batch_size + num_new, axis=0)
                hf['particle_indices'].resize(num_batches_processed * batch_size + num_new, axis=0)
                hf['subjet_mask'].resize(num_batches_processed * batch_size + num_new, axis=0)
                hf['particle_mask'].resize(num_batches_processed * batch_size + num_new, axis=0)
                
                hf['x'][num_batches_processed * batch_size:num_batches_processed * batch_size + num_new] = x
                hf['particle_features'][num_batches_processed * batch_size:num_batches_processed * batch_size + num_new] = particle_features
                hf['subjets'][num_batches_processed * batch_size:num_batches_processed * batch_size + num_new] = subjets
                hf['particle_indices'][num_batches_processed * batch_size:num_batches_processed * batch_size + num_new] = particle_indices
                hf['subjet_mask'][num_batches_processed * batch_size:num_batches_processed * batch_size + num_new] = subjet_mask
                hf['particle_mask'][num_batches_processed * batch_size:num_batches_processed * batch_size + num_new] = particle_mask
                
                num_batches_processed += 1
    
    # Save each part to a separate file
    for i, dataset in enumerate(datasets):
        file_path = f"{out_path}_p{i+1}.hdf5"
        process_and_save(dataset, file_path)


    
    # Usage example
    dataset = JEPADataset(dataset_path)
    print(f"successfully loaded finally processed {tag} dataset")



