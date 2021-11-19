import torch.nn as nn

from .sinusoidal_positional_embedding import SinusoidalPositionalEmbedding


def PositionalEmbedding(
    num_embeddings: int,
    embedding_dim: int,
    padding_idx: int,
):

    m = SinusoidalPositionalEmbedding(
        embedding_dim,
        padding_idx,
        left_pad=False,
        init_size=num_embeddings + padding_idx + 1,
    )
    return m
