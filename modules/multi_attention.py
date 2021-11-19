import torch
import math
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from fairseq_utils import (
    get_incremental_state,
    set_incremental_state,
    _get_full_incremental_state_key,)

class MultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = embed_dim
        self.vdim = embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.kv_same_dim = self.kdim == self.vdim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim
        self.scaling = self.head_dim**-0.5
        self._mask = None

        self.k_proj = nn.Linear(self.kdim, self.embed_dim, bias=bias)

        self.v_proj = nn.Linear(self.vdim, self.embed_dim, bias=bias)

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)

        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)


        self.reset_parameters()

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)


    def forward(self, query, key, value, mask_future_timesteps=False,
                key_padding_mask=None, incremental_state=None,
                need_weights=True, static_kv=False):
        """Input shape: Time x Batch x Channel

        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Future timesteps can be masked with the
        `mask_future_timesteps` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """
        # 代表是做encoder-decoder-attention
        qkv_same = self.qkv_same_dim
        # 代表是做self-attention
        kv_same = self.kv_same_dim

        if incremental_state is not None: # 只有在预测的时候才起作用，进行缓存
            saved_state = get_incremental_state(
                self,
                incremental_state,
                'attn_state',
            )
            if 'prev_key' in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert kv_same
                    key = value = None
        else:
            saved_state = None

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        if qkv_same:
            # self-attention
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        elif kv_same:
            # encoder-decoder attention
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(key)
        else:
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        q *= self.scaling

        if saved_state is not None:
            if 'prev_key' in saved_state:
                k = torch.cat((saved_state['prev_key'], k), dim=0)
            if 'prev_value' in saved_state:
                v = torch.cat((saved_state['prev_value'], v), dim=0)
            saved_state['prev_key'] = k
            saved_state['prev_value'] = v
            set_incremental_state(
                self,
                incremental_state,
                'attn_state',
                saved_state,
            )

        src_len = k.size(0)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        # only apply masking at training time (when incremental state is None)
        if mask_future_timesteps and incremental_state is None:
            assert query.size() == key.size(), \
                'mask_future_timesteps only applies to self-attention'
            attn_weights += self.buffered_mask(attn_weights.data).detach().unsqueeze(0)
        if key_padding_mask is not None:
            # don't attend to padding symbols
            if key_padding_mask.data.max() > 0:
                attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    -1e18,
                )
                attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        if need_weights:
            # average attention weights over heads
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.sum(dim=1) / self.num_heads
        else:
            attn_weights = None

        return attn, attn_weights

    def buffered_mask(self, tensor):
        dim = tensor.size(-1)
        if self._mask is None:
            self._mask = torch.triu(tensor.new(dim, dim).fill_(-1e18), 1)
        if self._mask.size(0) < dim:
            self._mask = torch.triu(self._mask.resize_(dim, dim).fill_(-1e18), 1)
        self._mask = self._mask.to(tensor)
        return self._mask[:dim, :dim]

    def reorder_incremental_state(self, incremental_state, new_order):
        saved_state = get_incremental_state(self, incremental_state, 'attn_state')
        if saved_state is not None:
            for k in saved_state.keys():
                saved_state[k] = saved_state[k].index_select(1, new_order)
            utils.set_incremental_state(self, incremental_state, 'attn_state', saved_state)
