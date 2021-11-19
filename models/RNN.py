import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq_utils import get_incremental_state, set_incremental_state
import math
from fairseq.models import (
    FairseqEncoder,
    FairseqIncrementalDecoder,
    FairseqEncoderDecoderModel,
)


class RNNEncoder(FairseqEncoder):   # 换成nn.Module应该可行,colab上试试看
    def __init__(self, args, dictionary):
        super().__init__(dictionary)
        self.padding_idx = dictionary.pad()
        self.embed_dim = args.encoder_embed_dim
        self.embeddings = Embedding(len(dictionary), self.embed_dim, self.padding_idx)

        self.hidden_dim = args.encoder_ffn_embed_dim
        self.nums_layers = args.encoder_layers
        self.embed_scale = math.sqrt(self.embed_dim)
        self.dropout = nn.Dropout(args.dropout)
        self.rnn = nn.GRU(
                        self.embed_dim,
                        self.hidden_dim,
                        self.nums_layers,
                        dropout = args.dropout,
                        batch_first = False,
                        bidirectional = True
        )

    def combine_birdir(self, hidden_outs, batch_size):
        hidden_out = hidden_outs.view(self.nums_layers, 2, batch_size, -1).transpose(1,2).contiguous
        return hidden_out.view(self.nums_layers, batch_size, -1)

    def forward(self, src_tokens, **unused):
        batch_size, seq_len = src_tokens.size()

        x = self.embed_scale * self.embeddings(src_tokens)
        x = self.dropout(x)

        h0 = x.new_zeros(2 * self.nums_layers, batch_size, self.hidden_dim)
        x, final_hiddens = self.rnn(x, h0)

        output = self.dropout(x)
        final_hiddens = self.combine_birdir(final_hiddens, batch_size)
        encoder_padding_mask = src_tokens.eq(self.padding_idx).t()

        return tuple((output, final_hiddens, encoder_padding_mask))

    def reorder_encoder_out(self, encoder_out, new_order):
        return tuple((encoder_out[0].index_select(1, new_order),
                      encoder_out[1].index_select(1, new_order),
                      encoder_out[2].index_select(0, new_order)))

class AttentionLayer(nn.Module):
    def __init__(self, input_embed_dim, source_embed_dim, output_embed_dim, bias = False):
        super(AttentionLayer, self).__init__()
        self.input_project = Linear(input_embed_dim, source_embed_dim, bias = bias)
        self.output_project = Linear(input_embed_dim + source_embed_dim, output_embed_dim, bias = bias)

    def forward(self, input, encoder_outputs, encoder_padding_mask):

        x = self.input_project(input)
        atten_socres = torch.bmm(x, encoder_outputs.transpose(1, 2))
        # pad src_tokens unuseful information
        if encoder_padding_mask is not None:
            encoder_padding_mask = encoder_padding_mask.unsqueeze(1)
            atten_socres = (atten_socres.float()
                            .masked_fill_(encoder_padding_mask, float('-inf'))
                            .type_as(atten_socres)
                            )

        atten_socres = F.softmax(atten_socres, dim = -1)
        x = torch.bmm(atten_socres, encoder_outputs)
        x = torch.cat((x, input), dim = -1)
        x = torch.tanh(self.output_project(x))

        return x, atten_socres

class RNNDecoder(FairseqIncrementalDecoder):
    def __init__(self, args, dictionary):
        super().__init__(dictionary)
        self.padding_idx = dictionary.pad()
        self.embed_dim = args.decoder_embed_dim
        self.embeddings = Embedding(len(dictionary), self.embed_dim, self.padding_idx)
        self.hidden_dim = args.decoder_ffn_embed_dim
        self.nums_layers = args.decoder_layers

        self.embed_scale = math.sqrt(self.embed_dim)
        self.dropout = nn.Dropout(args.dropout)
        self.rnn = nn.GRU(
                self.embed_dim,
                self.hidden_dim,
                self.nums_layers,
                dropout = args.dropout,
                batch_first = False,
                bidirectional = False     # 这里是单向的
        )
        self.attention = AttentionLayer(self.embed_dim, self.hidden_dim, self.embed_dim, bias = False)

        self.dropout = nn.Dropout(args.dropout)
        if self.hidden_dim != self.embed_dim:
            self.project_out_dim = Linear(self.hidden_dim, self.embed_dim)
        else:
            self.project_out_dim = None
        if args.share_decoder_input_output_embed:
            self.output_project = Linear(self.embeddings.weight.shape[1],
                                            self.embeddings.weight.shape[0],
                                            bias = False)
            self.output_project.weight = self.embeddings.weight
        else:
            self.output_project = Linear(self.output_embed_dim, len(dictionary), bias = False)
            nn.init.normal_(self.output_project.weight, mean = 0, std = self.output_embed_dim ** -0.5)

    def forward(self, prev_output_tokens, encoder_outs, incremental_state = None, **unused):

        encoder_out, encoder_hidden, encoder_padding_mask = encoder_outs
        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            cache_state = get_incremental_state(incremental_state, "cache_state")
            prev_hidden = cache_state["prev_hidden"]
        else:
            prev_hidden = encoder_hidden

        batch_size, seq_len = prev_output_tokens.size()

        x = self.embed_scale * self.embeddings(prev_output_tokens)
        x = self.dropout(x)
        x, atten_socres = self.attention(x, encoder_out, encoder_padding_mask)
        x, final_hiddens = self.rnn(x, prev_hidden)
        cache_state = {"prev_hidden": final_hiddens}
        set_incremental_state(incremental_state, "cached_state", cache_state)
        x = self.dropout(x)
        if self.project_out_dim != None:
            x = self.project_out_dim(x)
        x = self.output_project(x)

        return x, None

def Linear(in_features, out_features, bias=False):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    return m

def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m




