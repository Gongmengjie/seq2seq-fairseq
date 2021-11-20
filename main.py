import logging
import time
import torch
import numpy as np
import argparse
import random
from importlib import import_module

from seq2seq_model import build_model
from train_eval import train, validate, generate_prediction
from data.load import task, load_data_iterator
from modules.optimizer import optimizer
from modules.criterion import criterion
from modules.set_seed import set_seed
from modules.load_checkpoint import load_checkpoint

def get_parser(parser):

    parser.add_argument('--model', type=str, required=True, help='choose a model: RNN or Transformer')
    parser.add_argument('--data_dir', type=str, default='./TranslationData/raw_data', help='raw data dir')
    parser.add_argument('--binpath', type=str, default='./TranslationData/data_bin', help='bin data dir')
    parser.add_argument('--src_lang', type=str, default='en')  # 不能--src-lang命名
    parser.add_argument('--tgt_lang', type=str, default='zh')
    parser.add_argument('--save_dir', type=str, default='./results/save_dict')

    parser.add_argument('--post_process', type=str, default='sentencepiece', help='word segmentation method')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--max_tokens', type=int, default=4000)
    parser.add_argument('--accu_steps', type=int, default=1)
    parser.add_argument('--lr_factor', type=int, default=2)
    parser.add_argument('--lr_warmup', type=int, default=4000)
    parser.add_argument('--clip_norm', type=float, default=1.0)
    parser.add_argument('--max_epoch', type=int, default=30)
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--beam', type=int, default=5)
    parser.add_argument('--max_len_a', type=float, default=1.2)
    parser.add_argument('--max_len_b', type=int, default=10)
    parser.add_argument('--label_smoothing_rate', type=float, default=0.1)

    parser.add_argument('--encoder_embed_dim', type=int, default=256)
    parser.add_argument('--encoder_ffn_embed_dim', type=int, default=512)
    parser.add_argument('--encoder_layers', type=int, default=1)
    parser.add_argument('--decoder_embed_dim', type=int, default=256)
    parser.add_argument('--decoder_ffn_embed_dim', type=int, default=1024)
    parser.add_argument('--decoder_layers', type=int, default=1)
    parser.add_argument('--share_decoder_input_output_embed', type=bool, default=True)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--encoder_attention_heads', type=int, default=4)
    parser.add_argument('--normalize_before', type=bool, default=True)
    parser.add_argument('--encoder_layer_drop', type=float, default=0.3)
    parser.add_argument('--decoder_attention_heads', type=int, default=4)
    parser.add_argument('--activation_fn', type=str, default='relu')
    parser.add_argument('--max_source_positions', type=int, default=1024)
    parser.add_argument('--max_target_positions', type=int, default=1024)
    parser.add_argument('--activation_dropout', type=float, default=0.3)
    parser.add_argument('--token_positional_embeddings', type=bool, default=True)
    parser.add_argument('--attention_dropout', type=float, default=0)
    parser.add_argument("--relu_dropout", type=float, default=0, help="relu_dropout")

    return parser

parser = argparse.ArgumentParser(description='NMT en-zh')
args = get_parser(parser).parse_args()

if __name__ == '__main__':

    logger = logging.getLogger()
    model_name = args.model
    set_seed(73)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 准备数据
    task = task(args)
    epoch_itr = load_data_iterator(
            task,
            "train",
            args.start_epoch,
            args.max_tokens,
            args.num_workers,
        )
    # 构建模型
    model = build_model(args, task, model_name=model_name)
    model = model.to(device=device)
    logger.info(model)
    # 损失、优化函数
    criterion = criterion(task, args)
    optimizer = optimizer(args, model)
    # 训练
    train(
        epoch_itr = epoch_itr,
        model = model, task = task,
        criterion = criterion,
        optimizer = optimizer,
        args = args
    )
    # 利用下载好的模型进行测试
    # checkpoint_last.pt : 最后一次模型档案
    # checkpoint_best.pt : 最高的模型档案
    load_checkpoint(args, model, name="checkpoint_best.pt")
    validate(model, task, criterion, args, log_to_wandb=False)
    generate_prediction(args, model, task)










