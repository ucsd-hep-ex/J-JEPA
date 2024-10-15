import json
from argparse import Namespace
from typing import Union, Dict, Any


class Options(Namespace):
    def __init__(
        self,
        event_info_file: str = "",
        training_file: str = "",
        validation_file: str = "",
        testing_file: str = "",
    ):
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

        # Whether to use the particle transformer encoder
        self.use_parT: bool = False

        # Use predictor
        self.use_predictor: bool = True

        # embedding layers type
        self.embedding_layers_type = "EmbeddingStack"
        self.predictor_embedding_layers_type = "EmbeddingStack"

        # pos embedding type
        self.pos_emb_type = "space"

        # initial embeddings
        # initial embedding layer size
        self.initial_embedding_dim: int = 256

        # whether to add skip connection to the first embedding layer
        self.initial_embedding_skip_connections: bool = False

        # later embedding layers
        # embedding dimension size
        self.emb_dim: int = 1024
        self.predictor_emb_dim: int = 512

        # whether to add skip connections to the later embedding layers
        self.embedding_skip_connections: bool = True

        # linear block type:
        self.linear_block_type: str = "basic"

        # Maximum Number of double-sized embedding layers to add between the features and the encoder.
        # The size of the embedding dimension will be capped at the hidden_dim,
        # So setting this option to a very large integer will just keep embedding up to the hidden_dim.
        self.num_embedding_layers: int = 10

        # Activation function for all transformer layers, 'relu' or 'gelu'.
        """
        ACTIVATION_LAYERS = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "LeakyRelU": nn.LeakyReLU,
        "SELU": nn.SELU,
        }
        default is gelu
        """
        self.activation: str = "gelu"

        # Whether or not to add skip connections to internal linear layers.
        # All layers support skip connections, this can turn them off.
        self.skip_connections: bool = True

        # Dropout added to all layers.
        self.dropout: float = 0.0

        # Attention dropout added to attention layers.
        self.attn_drop: float = 0.0

        # Drop path rate applied to linear layers.
        self.drop_path: float = 0.0

        # Drop path rate applied to projector in attention layers.
        self.proj_drop: float = 0.0

        # qkv_bias added to attention layers.
        self.qkv_bias: bool = True

        # qk_scale applied to attention layers.
        self.qk_scale: float = None

        self.attn_dim: int = self.emb_dim

        # Number of features in the hidden layers in MLP.
        self.hidden_features: int = 512

        # Number of input features in MLP.
        self.in_features: int = self.emb_dim

        # Number of output features in MLP.
        self.out_features: int = self.emb_dim

        # drop rate for the MLP
        self.drop_mlp: float = 0.0

        # Whether or not to apply a normalization layer during linear / embedding layers.
        #
        # Options are:
        # -------------------------------------------------
        # None
        # BatchNorm
        # LayerNorm
        # MaskedBatchNorm
        """
        NORM_LAYERS = {
            "None": None,
            "BatchNorm": nn.BatchNorm1d,
            "LayerNorm": nn.LayerNorm,
            "MaskedBatchNorm": None,
        }
        default is None
        """
        # -------------------------------------------------
        self.normalization: str = "LayerNorm"

        # =========================================================================================
        # ParTEncoder specific parameters
        # =========================================================================================

        # projector MLP params, None -> no projector after attention layers.
        # Format: [(out_dim, drop_rate) for layer in range(num_layers)]
        self.fc_params: list = None

        # number of target particles in a jet
        self.N_trgt: int = 30

        # parameters for class attention blocks (used for aggregating ptcl features into jet features)
        self.cls_block_params: dict = {
            "dropout": 0,
            "attn_dropout": 0,
            "activation_dropout": 0,
        }

        # parameters for attention blocks
        self.block_params: list = None

        # number of class attention blocks (used for aggregating ptcl features into jet features)
        self.num_cls_layers: int = 0

        # number of input dimensions for pair embedding
        self.pair_input_dim: int = 4

        # embedding dimensions for pair embedding blocks
        self.pair_embed_dims = [64, 64, 64]

        # embedding dimensions for the transformer layers
        self.embed_dims = [128, 512, 128]

        # embedding dimensions for the predictor
        self.predictor_embed_dims = [64, 64, 64]

        # input dim for particles (default 4: deta, dphi, pt_log, e_log)
        self.input_dim: int = 4

        # =========================================================================================
        # JJEPA specific parameters
        # =========================================================================================
        # Number of attention heads in the multi-head attention layers
        self.num_heads: int = 8

        # Number of transformer layers in the encoder
        self.num_layers: int = 6

        # Number of subjets to use as context
        self.num_context_subjets: int = 10

        # Depth of the encoder network
        self.encoder_depth: int = 3

        # Depth of the predictor network
        self.pred_depth: int = 3

        # Ratio for the MLP in transformer layers
        self.mlp_ratio: float = 4.0

        # Initial std for trunc_normal_
        self.init_std: float = 0.02

        # Dimension of subjet representations
        # Note that this is a dynamic variable
        # depending on where you create the attention block
        self.repr_dim: int = -1

        # Whether to use positional embeddings in the encoder
        self.encoder_pos_emb: bool = True

        # =========================================================================================
        # Optimizer Parameters
        # =========================================================================================

        # Training batch size.
        self.batch_size: int = 256

        # The optimizer to use for training the network.
        # This must be a valid class in torch.optim or nvidia apex with 'apex' prefix.
        self.optimizer: str = "AdamW"

        # adjust eps to improve numerical stability
        # reference https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
        self.eps: float = 1e-7

        # Optimizer learning rate.
        self.learning_rate: float = 0.0001

        # Weight decay for optimizer
        self.weight_decay: float = 0.01

        # Start of tranining epochs
        self.start_epochs: int = 0

        # Number of training epochs
        self.num_epochs: int = 100

        # Number of warm-up epochs
        self.warmup_epochs: int = 10

        # Whether to use automatic mixed precision
        self.use_amp: bool = False

        # EMA parameters [start_value, end_value]
        self.ema: list = [0.996, 0.999]

        # Checkpoint saving frequency (in epochs)
        self.checkpoint_freq: int = 10

        # weight for the covariance loss
        self.cov_loss_weight: float = 0.0

        # weight for the variance loss
        self.var_loss_weight: float = 0.0

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
        self.num_jets: int = None

        # Number of worker processes for data loading
        self.num_workers: int = 0

        # =========================================================================================
        # File Paths
        # =========================================================================================

        self.event_info_file: str = event_info_file
        self.training_file: str = training_file
        self.validation_file: str = validation_file
        self.testing_file: str = testing_file

        # =========================================================================================
        # Logging Parameters
        # =========================================================================================

        # Whether to display logging information
        self.display_logging: bool = True

        # =========================================================================================
        # random Parameters
        # =========================================================================================

        # learing rate
        self.lr: float = 1e-4

        # base momentum
        self.base_momentum: float = 0.99

        # max grad norm
        self.max_grad_norm: float = 0.1

        # number of steps per epoch
        self.num_steps_per_epoch: int = None

        # number of steps per epoch
        self.log_freq: int = 1000

        # debug mode
        self.debug: bool = False

        # number of val jets
        self.num_val_jets: int = None

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
                table.add_row(
                    key,
                    str(value),
                    style="red" if value != default_options.get(key) else None,
                )

            console.print(table)

        except ImportError:
            print("=" * 70)
            print("Options")
            print("-" * 70)
            for key, val in sorted(self.__dict__.items()):
                print(f"{key:32}: {val}")
            print("=" * 70)

    def update_options(self, new_options: Dict[str, Any], update_datasets: bool = True):
        integer_options = {
            key for key, val in self.__dict__.items() if isinstance(val, int)
        }
        float_options = {
            key for key, val in self.__dict__.items() if isinstance(val, float)
        }
        boolean_options = {
            key for key, val in self.__dict__.items() if isinstance(val, bool)
        }

        for key, value in new_options.items():
            if not update_datasets and key in {
                "event_info_file",
                "training_file",
                "validation_file",
                "testing_file",
            }:
                continue

            if key in boolean_options:
                setattr(self, key, bool(value))
            elif key in float_options:
                setattr(self, key, float(value))
            elif key in integer_options:
                setattr(self, key, int(value))
            else:
                setattr(self, key, value)

    def update(self, filepath: str):
        with open(filepath, "r") as json_file:
            self.update_options(json.load(json_file))
        self.embed_dims[-1] = self.emb_dim

    @classmethod
    def load(cls, filepath: str):
        options = cls()
        with open(filepath, "r") as json_file:
            options.update_options(json.load(json_file))
            options.embed_dims[-1] = options.emb_dim
        return options

    def save(self, filepath: str):
        with open(filepath, "w") as json_file:
            json.dump(self.__dict__, json_file, indent=4, sort_keys=True)
