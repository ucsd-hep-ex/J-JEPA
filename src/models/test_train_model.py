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


def create_mock_data(options, tmp_path):
    num_jets = options.num_jets
    num_subjets = options.num_subjets
    num_particles = options.num_particles
    num_features = options.num_part_ftr

    train_file_path = tmp_path / "train_mock_dataset.h5"
    val_file_path = tmp_path / "val_mock_dataset.h5"
    with h5py.File(train_file_path, "w") as f:
        f.create_dataset(
            "x",
            data=np.random.randn(num_jets, num_subjets, num_particles, num_features),
        )
        f.create_dataset(
            "particle_features",
            data=np.random.randn(num_jets, num_particles, num_features),
        )
        f.create_dataset("subjets", data=np.random.randn(num_jets, num_subjets, 5))
        f.create_dataset(
            "particle_indices",
            data=np.random.randint(
                0, num_particles, (num_jets, num_subjets, num_particles)
            ),
        )
        f.create_dataset("subjet_mask", data=np.ones((num_jets, num_subjets)))
        f.create_dataset("particle_mask", data=np.ones((num_jets, num_particles)))
    num_jets //= 10
    with h5py.File(val_file_path, "w") as f:
        f.create_dataset(
            "x",
            data=np.random.randn(num_jets, num_subjets, num_particles, num_features),
        )
        f.create_dataset(
            "particle_features",
            data=np.random.randn(num_jets, num_particles, num_features),
        )
        f.create_dataset("subjets", data=np.random.randn(num_jets, num_subjets, 5))
        f.create_dataset(
            "particle_indices",
            data=np.random.randint(
                0, num_particles, (num_jets, num_subjets, num_particles)
            ),
        )
        f.create_dataset("subjet_mask", data=np.ones((num_jets, num_subjets)))
        f.create_dataset("particle_mask", data=np.ones((num_jets, num_particles)))
    return str(train_file_path)


def test_train_model():
    tmp_path = Path("./tmp_test_output")
    tmp_path.mkdir(exist_ok=True)

    config_path = tmp_path / "mock_config.json"
    options = Options()
    options.num_jets = 1000
    options.num_subjets = 20
    options.num_particles = 30
    options.num_part_ftr = 4
    options.batch_size = 32
    options.num_epochs = 2
    options.num_workers = 2
    options.lr = 1e-4
    options.base_momentum = 0.99
    options.max_grad_norm = 1.0
    options.use_amp = False
    options.log_freq = 10
    options.start_epochs = 0
    options.emb_dim = 128
    options.num_layers = 4
    options.num_heads = 4
    options.dropout = 0.1
    options.num_steps_per_epoch = options.num_jets // options.batch_size
    options.save(config_path)

    data_path = create_mock_data(options, tmp_path)

    sys.argv = [
        "train_model.py",
        "--config",
        str(config_path),
        "--data_path",
        data_path,
        "--output_dir",
        str(tmp_path),
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
