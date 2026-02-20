import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import argparse
from pathlib import Path
from src.models.train_model import main, parse_args
from src.options import Options
import sys
import h5py
import numpy as np
import torch.multiprocessing as mp


def get_data_path(path):
    train_data_path = path / "processed_train_20_30_torch_new.h5"
    return str(train_data_path)


def test_train_model():

    # if you want to change the data path, you can do so here --- change data path
    tmp_path = Path("/mnt/d/physic/output_without_eps/")
    checkpoint_path = Path("/mnt/d/physic/output_without_eps/best_model.pth")

    
    tmp_path.mkdir(exist_ok=True)

    config_path = tmp_path / "mock_config.json"
    options = Options()

    options.num_subjets = 20
    options.num_particles = 30
    options.num_part_ftr = 4

    options.batch_size = 32
    options.num_epochs = 300
    options.num_workers = 0

    options.learning_rate = 1e-4
    options.base_momentum = 0.99
    options.max_grad_norm = 1.0

    options.emb_dim = 1024
    options.encoder_depth = 24
    options.pred_depth = 12
    options.mlp_ratio = 1.0

    options.num_heads = 16
    options.dropout = 0.1
    options.max_grad_norm = 0.1
    options.use_amp = False
    options.num_steps_per_epoch = options.num_jets // options.batch_size
    options.save(config_path)

 
    data_path = get_data_path(tmp_path)

    sys.argv = [
        "train_model.py",
        "--config",
        str(config_path),
        "--data_path",
        data_path,
        "--output_dir",
        str(tmp_path),
        "--load_checkpoint",
        str(checkpoint_path)
    ]

    args = parse_args()
    world_size = torch.cuda.device_count()
    print(f"World size: {world_size}")

    if world_size == 0:
        print("No CUDA devices available. Running on CPU.")
        main(0, 1, args)
    else:
        mp.spawn(main, args=(world_size, args), nprocs=world_size, join=True)

    print(f"Checking for files in {tmp_path}")
    print(f"Files in directory: {os.listdir(tmp_path)}")

    checkpoint_files = list(tmp_path.glob("checkpoint_epoch_*.pth"))
    log_files = list(tmp_path.glob("train_rank_*.log"))

    if len(checkpoint_files) == 0:
        print("No checkpoint files found. Checking log files...")
        if len(log_files) > 0:
            print(f"Log files found: {log_files}")
            with open(log_files[0], "r") as f:
                print("Last few lines of the log file:")
                print("\n".join(f.readlines()[-10:]))
        else:
            print("No log files found either.")
    else:
        print(f"Checkpoint files found: {checkpoint_files}")

    assert len(checkpoint_files) > 0, f"No checkpoint files found in {tmp_path}"
    assert len(log_files) > 0, f"No log files found in {tmp_path}"

    print("Test completed successfully!")


if __name__ == "__main__":
    test_train_model()
