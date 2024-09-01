import json
from argparse import Namespace


class Options(Namespace):
    def __init__(self, event_info_file: str = "", training_file: str = "", validation_file: str = "", testing_file: str = ""):
        super(Options, self).__init__()

        # =========================================================================================
        # Dataset Structure
        # =========================================================================================
        # Top level of the .h5 dataset is jet
        # number of subjets per jet
        self.num_subjets = 20

        # number of particles per jet
        self.num_particles = 30

        # number of particle features per particle
        self.num_part_ftr = 4

        # =========================================================================================
        # Network Architecture
        # =========================================================================================

        # initial embeddings
        # initial embedding layer size
        self.initial_embedding_dim: int = 256

        # whether to add skip connection to the first embedding layer
        self.initial_embedding_skip_connections = False

        # later embedding layers
        # embedding dimension size
        self.emb_dim: int = 512

        # whether to add skip connections to the later emebdding layers
        self.embedding_skip_connections = True

        # linear block type:
        self.linear_block_type = 'basic'

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
        # Optimizer Parameters
        # =========================================================================================

        # Training batch size.
        self.batch_size: int = 4096

        # The optimizer to use for trianing the network.
        # This must be a valid class in torch.optim or nvidia apex with 'apex' prefix.
        self.optimizer: str = "AdamW"

        # Optimizer learning rate.
        self.learning_rate: float = 0.001

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
                table.add_row(key, str(value), style="red" if value != default_options[key] else None)

            console.print(table)

        except ImportError:
            print("=" * 70)
            print("Options")
            print("-" * 70)
            for key, val in sorted(self.__dict__.items()):
                print(f"{key:32}: {val}")
            print("=" * 70)

    def update_options(self, new_options, update_datasets: bool = True):
        integer_options = {key for key, val in self.__dict__.items() if isinstance(val, int)}
        boolean_options = {key for key, val in self.__dict__.items() if isinstance(val, bool)}
        for key, value in new_options.items():
            if not update_datasets and key in {"event_info_file", "training_file", "validation_file", "testing_file"}:
                continue

            if key in integer_options:
                setattr(self, key, int(value))
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
