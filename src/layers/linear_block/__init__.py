from src.layers.linear_block.basic_block import BasicBlock
from src.options import Options


def create_linear_block(
        options: Options,
        input_dim: int,
        output_dim: int,
        skip_connection: bool = False
):
    linear_block_type = options.linear_block_type.lower().replace("_", "").replace(" ", "")

    if linear_block_type == "resnet":
        raise Exception("Sorry, not implemented yet")
    elif linear_block_type == 'gated':
        raise Exception("Sorry, not implemented yet")
    elif linear_block_type == 'gru':
        raise Exception("Sorry, not implemented yet")
    else:
        return BasicBlock(options, input_dim, output_dim, skip_connection)
