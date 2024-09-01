from torch import Tensor, nn

from src.layers.linear_block.activations import create_activation, create_dropout, create_residual_connection
from src.layers.linear_block.normalizations import create_normalization
from src.options import Options


class BasicBlock(nn.Module):
    def __init__(self, options: Options, input_dim: int, output_dim: int,
                 skip_connection: bool = False):
        super(BasicBlock, self).__init__()

        self.output_dim: int = output_dim
        self.skip_connection: bool = skip_connection

        # Basic matrix multiplication layer as the base.
        self.linear = nn.Linear(input_dim, output_dim)

        # Select non-linearity.
        self.activation = create_activation(options.activation, output_dim)

        # Create normalization layer for keeping values in good ranges.
        self.normalization = create_normalization(options.normalization, output_dim)

        # Optional dropout for regularization.
        self.dropout = create_dropout(options.dropout)

        # Possibly need a linear layer to create residual connection.
        self.residual = create_residual_connection(skip_connection, input_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        """ Simple robust linear layer with non-linearity, normalization, and dropout.

        Parameters
        ----------
        x: [*, input_dim]
            Input data.

        Returns
        -------
        y: [*, output_dim]
            Output data.
        """
        y = self.linear(x)
        y = self.activation(y)

        # ----------------------------------------------------------------------------
        # Optionally add a skip-connection to the network to add residual information.
        # ----------------------------------------------------------------------------
        if self.skip_connection:
            y = y + self.residual(x)

        y = self.normalization(y)
        return self.dropout(y)
