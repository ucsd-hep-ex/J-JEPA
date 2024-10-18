from src.layers.embedding_stack import(
    EmbeddingStack,
    LinearEmbeddingStack,
    PredictorEmbeddingStack,
    PredictorLinearEmbeddingStack,
    PlainAttentionEmbeddingStack
)
from src.options import Options


def create_embedding_layers(
        options: Options,
        input_dim: int,
):
    embedding_layers_type = options.embedding_layers_type.lower().replace("_", "").replace(" ", "")

    if embedding_layers_type == "embeddingstack":
        return EmbeddingStack(options, input_dim)
    elif embedding_layers_type == "linearembeddingstack":
        return LinearEmbeddingStack(options, input_dim)
    elif embedding_layers_type == "plainattentionembeddingstack":
        return PlainAttentionEmbeddingStack(options, input_dim)
    else:
        raise Exception(f"Sorry, embedding layer type: {embedding_layers_type} not implemented yet")

def create_predictor_embedding_layers(
        options: Options,
        input_dim: int,
):
    embedding_layers_type = options.predictor_embedding_layers_type.lower().replace("_", "").replace(" ", "")

    if embedding_layers_type == "embeddingstack":
        return PredictorEmbeddingStack(options, input_dim)
    elif embedding_layers_type == "linearembeddingstack":
        return PredictorLinearEmbeddingStack(options, input_dim)
