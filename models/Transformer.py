import math
from typing import Any, Dict, List, Optional, Tuple
from torch import Tensor
import torch
import torch.nn as nn
from torch import Tensor
import logging

from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
)

from modules.layer_norm import LayerNorm
from modules.sinusoidal_positional_embedding import SinusoidalPositionalEmbedding
from modules.positional_embedding import PositionalEmbedding
from modules.transformer_layer import TransformerEncoderLayer, TransformerDecoderLayer


logger = logging.getLogger()


class TransformerEncoder(FairseqEncoder):  # 尝试换成nn.Module应该可行

    def __init__(self, args, dictionary):
        super().__init__(dictionary)
        self.padding_idx = dictionary.pad()
        self.embed_dim = args.encoder_embed_dim
        self.hidden_dim = args.encoder_ffn_embed_dim
        self.num_layers = args.encoder_layers
        self.embed_scale = math.sqrt(self.embed_dim)
        self.dropout_module = nn.Dropout(args.dropout)
        self.embeddings = Embedding(len(dictionary), self.embed_dim, self.padding_idx)
        self.max_source_positions = args.max_source_positions

        if args.token_positional_embeddings:
            self.embed_positions = PositionalEmbedding(
                    args.max_source_positions,
                    self.embed_dim,
                    self.padding_idx,
                )
        else:
            None

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(args)
            for _ in range(self.num_layers)])

        if args.normalize_before:
            self.layer_norm = LayerNorm(self.embed_dim)
        else:
            self.layer_norm = None

    def forward(self, src_tokens, **unused):

        # embed tokens and positions
        x = self.embed_scale * self.embeddings(src_tokens)
        x += self.embed_positions(src_tokens)
        x = self.dropout_module(x)
        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)

        if self.layer_norm is not None:  # Norm
            outputs = self.layer_norm(x)

        return tuple(outputs, encoder_padding_mask)

    def reorder_encoder_out(self, encoder_out, new_order):

        new_encoder_outs = encoder_out[0].index_select(1, new_order)
        new_encoder_padding_mask = encoder_out[1].index_select(0, new_order)

        return tuple(
            new_encoder_out,  # B X T X C
            new_encoder_padding_mask,  # B x C
        )

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self.max_source_positions

class TransformerDecoder(FairseqIncrementalDecoder):

    def __init__(self, args, dictionary, no_encoder_attn=False):
        self.args = args
        super().__init__(dictionary)
        self.padding_idx = dictionary.pad()
        self._future_mask = torch.empty(0)

        self.dropout_module = nn.Dropout(args.dropout)
        self.share_input_output_embed = args.share_decoder_input_output_embed

        self.embed_dim = args.decoder_embed_dim
        self.hiddens_dim = args.decoder_ffn_embed_dim
        self.num_layers = args.decoder_layers

        self.max_target_positions = args.max_target_positions
        self.embeddings = Embedding(len(dictionary), self.embed_dim, self.padding_idx)
        self.embed_scale = math.sqrt(self.embed_dim)

        if args.token_positional_embeddings:
            self.embed_positions = PositionalEmbedding(
                    args.max_target_positions,
                    self.embed_dim,
                    self.padding_idx,
            )
        else:
            None

        self.layers = nn.ModuleList([
            TransformerDecoderLayer(args)
            for _ in range(self.num_layers)])
        self.layer_norm = LayerNorm(self.embed_dim)

        if self.embed_dim != self.hiddens_dim:
            self.project_out_dim = Linear(self.hiddens_dim, self.embed_dim, bias=False)
        else:
            self.project_out_dim = None

        if self.share_input_output_embed:
            logger.info("Sharing input embeddings and projection matrix in the decoder")
            self.output_projection = Linear(
                self.embeddings.weight.shape[1],
                self.embeddings.weight.shape[0],
                bias=False,
            )
            self.output_projection.weight = self.embeddings.weight
        else:
            self.output_projection = Linear(
                self.output_embed_dim, len(dictionary), bias=False
            )
            nn.init.normal_(
                self.output_projection.weight, mean=0, std=self.output_embed_dim ** -0.5
            )

    def forward(self, prev_output_tokens, encoder_out, incremental_state = None, **unused):
        encoder_outs, encoder_padding_mask = encoder_out
        embed_tokens = self.embeddings
        # embed positions
        positions = self.embed_positions(prev_output_tokens, incremental_state)
        # embed tokens and positions
        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[-1:, :]# only keep last time step
            if positions is not None:
                positions = positions[:, -1:]

        x = self.embed_scale * embed_tokens(prev_output_tokens)
        x += positions
        x = self.dropout_module(x)
        # decoder layers -------------------encoder-decoder-attention
        for layer in self.layers:
            x, attn = layer(
                x,
                encoder_outs,
                encoder_padding_mask,
                incremental_state=incremental_state,
            )

        x = self.layer_norm(x)
        x = self.output_projection(x)
        return x


    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return self.max_target_positions


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m

def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m




