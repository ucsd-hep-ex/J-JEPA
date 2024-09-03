import json
from argparse import Namespace
from typing import Union, Dict, Any


class Options(Namespace):
    def __init__(self, event_info_file: str = "", training_file: str = "", validation_file: str = "", testing_file: str = ""):
        super(Options, self).__init__()

        # =========================================================================================
        # Dataset Structure
        # =========================================================================================
        # Top level of the .h5 dataset is jet
        # number of subjets per jet
        self.num_subjets: int = 20

        # number of particles per jet
        self.num_particles: int = 30

        # number of particle features per particle
        self.num_part_ftr: int = 4

        # =========================================================================================
        # Network Architecture
        # =========================================================================================

        # initial embeddings
        # initial embedding layer size
        self.initial_embedding_dim: int = 256

        # whether to add skip connection to the first embedding layer
        self.initial_embedding_skip_connections: bool = False

        # later embedding layers
        # embedding dimension size
        self.emb_dim: int = 512

        # whether to add skip connections to the later embedding layers
        self.embedding_skip_connections: bool = True

        # linear block type:
        self.linear_block_type: str = 'basic'

        # Maximum Number of double-sized embedding layers to add between the features and the encoder.
        # The size of the embedding dimension will be capped at the hidden_dim,
        # So setting this option to a very large integer will just keep embedding up to the hidden_dim.
        self.num_embedding_layers: int = 10

        # Activation function for all transformer layers, 'relu' or 'gelu'.
        self.activation: str = 'gelu'

        # Whether or not to add skip connections to internal linear layers.
        # All layers support skip connections, this can turn them off.
        self.skip_connections: bool = True

        # Dropout added to all layers.
        self.dropout: float = 0.0

        # Whether or not to apply a normalization layer during linear / embedding layers.
        #
        # Options are:
        # -------------------------------------------------
        # None
        # BatchNorm
        # LayerNorm
        # MaskedBatchNorm
        # -------------------------------------------------
        self.normalization: str = "LayerNorm"

        # =========================================================================================
        # JJEPA specific parameters
        # =========================================================================================
        # Number of attention heads in the multi-head attention layers
        self.num_heads: int = 8

        # Number of transformer layers in the encoder
        self.num_layers: int = 6

        # Number of subjets to use as context
        self.num_context_subjets: int = 10

        # Depth of the predictor network
        self.pred_depth: int = 3

        # Ratio for the MLP in transformer layers
        self.mlp_ratio: float = 4.0

        # =========================================================================================
        # Optimizer Parameters
        # =========================================================================================

        # Training batch size.
        self.batch_size: int = 4096

        # The optimizer to use for training the network.
        # This must be a valid class in torch.optim or nvidia apex with 'apex' prefix.
        self.optimizer: str = "AdamW"

        # Optimizer learning rate.
        self.learning_rate: float = 0.001

        # Weight decay for optimizer
        self.weight_decay: float = 0.01

        # Start of tranining epochs
        self.start_epochs: int = 0

        # Number of training epochs
        self.num_epochs: int = 100

        # Number of warm-up epochs
        self.warmup_epochs: int = 10

        # Whether to use automatic mixed precision
        self.use_amp: bool = True

        # EMA parameters [start_value, end_value]
        self.ema: list = [0.996, 0.999]

        # Checkpoint saving frequency (in epochs)
        self.checkpoint_freq: int = 10

        # =========================================================================================
        # Scheduler Parameters
        # =========================================================================================

        # Type of learning rate scheduler
        self.scheduler: str = "cosine"

        # Minimum learning rate for scheduler
        self.min_lr: float = 1e-6

        # Starting learning rate for warm-up
        self.warmup_start_lr: float = 1e-8

        # =========================================================================================
        # Dataset Parameters
        # =========================================================================================

        # Number of jets to use in training
        self.num_jets: int = 100000

        # Number of worker processes for data loading
        self.num_workers: int = 4

        # =========================================================================================
        # File Paths
        # =========================================================================================

        self.event_info_file: str = event_info_file
        self.training_file: str = training_file
        self.validation_file: str = validation_file
        self.testing_file: str = testing_file

    def display(self):
        try:
            from rich import get_console
            from rich.table import Table

            default_options = self.__class__().__dict__
            console = get_console()

            table = Table(title="Configuration", header_style="bold magenta")
            table.add_column("Parameter", justify="left")
            table.add_column("Value", justify="left")

            for key, value in sorted(self.__dict__.items()):
                table.add_row(key, str(value), style="red" if value != default_options.get(key) else None)

            console.print(table)

        except ImportError:
            print("=" * 70)
            print("Options")
            print("-" * 70)
            for key, val in sorted(self.__dict__.items()):
                print(f"{key:32}: {val}")
            print("=" * 70)

    def update_options(self, new_options: Dict[str, Any], update_datasets: bool = True):
        integer_options = {key for key, val in self.__dict__.items() if isinstance(val, int)}
        float_options = {key for key, val in self.__dict__.items() if isinstance(val, float)}
        boolean_options = {key for key, val in self.__dict__.items() if isinstance(val, bool)}
        
        for key, value in new_options.items():
            if not update_datasets and key in {"event_info_file", "training_file", "validation_file", "testing_file"}:
                continue

            if key in integer_options:
                setattr(self, key, int(value))
            elif key in float_options:
                setattr(self, key, float(value))
            elif key in boolean_options:
                setattr(self, key, bool(value))
            else:
                setattr(self, key, value)

    def update(self, filepath: str):
        with open(filepath, 'r') as json_file:
            self.update_options(json.load(json_file))

    @classmethod
    def load(cls, filepath: str):
        options = cls()
        with open(filepath, 'r') as json_file:
            options.update_options(json.load(json_file))
        return options

    def save(self, filepath: str):
        with open(filepath, 'w') as json_file:
            json.dump(self.__dict__, json_file, indent=4, sort_keys=True)
