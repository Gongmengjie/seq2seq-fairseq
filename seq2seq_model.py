from models.RNN import RNNEncoder,RNNDecoder
from models.Transformer import TransformerEncoder,TransformerDecoder
import logging
import torch.nn as nn
from fairseq.models import FairseqEncoderDecoderModel
logger = logging.getLogger()

class Seq2Seq(FairseqEncoderDecoderModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(encoder, decoder)
        self.encoder = encoder   # 该不该加上这两行
        self.decoder = decoder

    def forward(self, src_tokens, src_lengths, prev_output_tokens):
        """
        Run the forward pass for an encoder-decoder model.
        """
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths)
        logits, extra = self.decoder(prev_output_tokens, encoder_out=encoder_out, src_lengths=src_lengths)
        return logits, extra


def build_model(args, task, model_name):
    """ 按照参数配置模型 """
    # 词表
    src_dict, tgt_dict = task.source_dictionary, task.target_dictionary  # 项目数据集的词表

    if model_name =='RNN':
        encoder = RNNEncoder(args, src_dict)
        decoder = RNNDecoder(args, tgt_dict)
    elif model_name == 'Transformer':
        encoder = TransformerEncoder(args, src_dict)
        decoder = TransformerDecoder(args, tgt_dict)
    # 序列到序列模型

    model = Seq2Seq(args, encoder, decoder)

    # 序列到序列模型的初始化很重要 需要特別處理
    def init_params(module):
        from modules.multi_attention import MultiheadAttention
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        if isinstance(module, MultiheadAttention):
            module.q_proj.weight.data.normal_(mean=0.0, std=0.02)
            module.k_proj.weight.data.normal_(mean=0.0, std=0.02)
            module.v_proj.weight.data.normal_(mean=0.0, std=0.02)

        if isinstance(module, nn.RNNBase):
            for name, param in module.named_parameters():
                if "weight" in name or "bias" in name:
                    param.data.uniform_(-0.1, 0.1)


    model.apply(init_params)
    return model



