import sys
import os
import json
import re
import logging
from pathlib import Path
import random
from argparse import Namespace
import sentencepiece as spm

class DataProcess(object):
    def __init__(self, args):
        super(DataProcess, self).__init__()
        self.file_prefix = args.data_dir
        self.src_lang = args.src_lang
        self.tgt_lang = args.tgt_lang
        self.data_prefix = Path(self.file_prefix, 'train_dev.raw')
        self.test_prefix = Path(self.file_prefix, 'test.raw')

    def cut(self):
        dataset_name = ['/translation2019zh_train.json', '/translation2019zh_valid.json']
        for i in range(len(dataset_name)):
            file_path = data_dir + dataset_name[i]
            fp = open(file_path, encoding='utf-8')
            if i == 0:
                src_fp = open(f'{self.data_prefix}.{self.src_lang}', 'w', encoding='utf-8')
                tgt_fp = open(f'{self.data_prefix}.{self.tgt_lang}', 'w', encoding='utf-8')
            else:
                src_fp = open(f'{self.test_prefix}.{self.src_lang}', 'w', encoding='utf-8')
                tgt_fp = open(f'{self.test_prefix}.{self.tgt_lang}', 'w', encoding='utf-8')
            count = 0
            for line in fp.readlines():
                line.strip()
                if count % 10 == 1:
                    new_dict = json.loads(line)
                    if i == 0:
                        src_fp.write(new_dict['english'] + '\n')
                        tgt_fp.write(new_dict['chinese'] + '\n')
                    if i == 1:
                        src_fp.write(new_dict['english'] + '\n')
                        tgt_fp.write('。' + '\n')
                count += 1
        src_fp.close()
        tgt_fp.close()

    def strQ2B(self, ustring):
        """把字符串全形转半形"""
        # 參考來源:https://ithelp.ithome.com.tw/articles/10233122
        ss = []
        for s in ustring:

            rstring = ""
            for uchar in s:
                inside_code = ord(uchar)
                if inside_code == 12288:  # 中文空格字符转化为英文状态下空格
                    inside_code = 32
                elif (inside_code >= 65281 and inside_code <= 65374):  # 全形字元（除空格）根據關係轉化
                    inside_code -= 65248
                rstring += chr(inside_code)
            ss.append(rstring)
        return ''.join(ss)


    def clean_s(self, s, lang):
        if lang == 'en':
            s = re.sub(r"\([^()]*\)", "", s)    # remove ([text]) 将这些"\([^()]*\)"字符删去，即用""代替
            s = s.replace('-', '')         # remove '-'
            # s = re.sub(r'-', '', s)

            s = re.sub('([.,;!?()\"])' , r' \1 ', s) # 保留标点
        elif lang == 'zh':
            s = self.strQ2B(s) # Q2B
            s = re.sub(r"\([^()]*\)", "", s) # remove ([text])
            s = s.replace(' ', '')
            s = s.replace('——', '')
            s = s.replace('“', '"')
            s = s.replace('”', '"')
            s = s.replace('_', '')
            s = re.sub('([。,;!?()\"~「」])' , r' \1 ', s) # 保留标点
        # 将每一行的元素变为list，strip()删除的字符,按照split()中的符号进行每行元素分割为list的元素
        s = ' '.join(s.strip().split())
        return s

    def len_s(self, s, lang):
        if lang == 'zh':
            return len(s)
        return len(s.split())


    def clean_corpus(self, data_prefix, l1, l2, ratio=9, max_len=100, min_len=1):
      # 清除过长或果断的句子，l1、l2表示语言1和语言2
        if not Path(f'{self.data_prefix}.{self.src_lang}').exists():
            self.cut()

        if Path(f'{prefix}.clean.{l1}').exists() and Path(f'{prefix}.clean.{l2}').exists():
            print(f'{prefix}.clean.{l1} & {l2} exists. skipping clean.')
            return

        l1_in_f = open(f'{prefix}.{l1}', 'r', encoding='utf-8')
        l2_in_f = open(f'{prefix}.{l2}', 'r', encoding='utf-8')
        l1_out_f = open(f'{prefix}.clean.{l1}', 'w', encoding='utf-8')
        l2_out_f = open(f'{prefix}.clean.{l2}', 'w', encoding='utf-8')
        for s1 in l1_in_f:
            s1 = s1.strip()
            s2 = l2_in_f.readline().strip()
            s1 = self.clean_s(s1, l1)
            s2 = self.clean_s(s2, l2)
            s1_len = self.len_s(s1, l1)
            s2_len = self.len_s(s2, l2)
            if min_len > 0:  # remove short sentence
                if s1_len < min_len or s2_len < min_len:
                    continue
            if max_len > 0:  # remove long sentence
                if s1_len > max_len or s2_len > max_len:
                    continue
            if ratio > 0:
                # remove by ratio of length，删除长度比过大的不好翻译的句子
                if s1_len / s2_len > ratio or s2_len / s1_len > ratio:
                    continue
            print(s1, file=l1_out_f)
            print(s2, file=l2_out_f)

    def split(self, prefix, l1, l2, train_ratio = 0.99, valid_ratio = 0.01):
        if Path(prefix, f'train.clean.{self.src_lang}').exists():
            print('train/valid splits exists.')

        if not Path(f'{prefix}.clean.{l1}').exists() and Path(f'{prefix}.clean.{l2}').exists():
            self.clean_corpus(self.data_prefix,  self.src_lang, self.tgt_lang)
            self.clean_corpus(self.test_prefix, self.src_lang, self.tgt_lang, ratio=-1, max_len= -1, min_len= -1)
        else:
            line_num = sum(1 for line in open(f'{self.data_prefix}.clean.{self.src_lang}', encoding='utf-8'))
            labels = list(range(line_num))
            random.shuffle(labels)  # 打乱标签的顺序
            for lang in [self.src_lang, self.tgt_lang]:
                train_f = open(Path(prefix, f'train.clean.{lang}'), 'w', encoding='utf-8')
                valid_f = open(Path(prefix, f'valid.clean.{lang}'), 'w', encoding='utf-8')
                count = 0
                for line in open(f'{self.data_prefix}.clean.{lang}', 'r', encoding='utf-8'):
                    if labels[count] / line_num < train_ratio:
                        train_f.write(line)
                    else:
                        valid_f.write(line)
                    count += 1
                train_f.close()
                valid_f.close()


class SpmmBinary(object):

    def __init__(self, args):
        super(SpmmBinary, self).__init__()
        self.src_lang = args.src_lang
        self.tgt_lang = args.tgt_lang

    def Spm(self, prefix, vocab_size = 8000):

        if Path(prefix, f'spm{vocab_size}.model').exists():
            print(f'{prefix}/spm{vocab_size}.model exists, skipping spm_train.')
        else:
            spm.SentencePieceTrainer.train(
                input=','.join([f'{prefix}/train.clean.{self.src_lang}',
                                f'{prefix}/valid.clean.{self.src_lang}',
                                f'{prefix}/train.clean.{self.tgt_lang}',
                                f'{prefix}/valid.clean.{self.tgt_lang}']), # 自己的语料库
                model_prefix=Path(prefix,'spm8000'),
                vocab_size=vocab_size,
                character_coverage=1,
                model_type='unigram', # 'bpe'也可，unigram代表按照一元组进行分词把句子从头到尾分成一个一个的汉字，bpe是按字节对编码分词
                           # 'char'：字符型分词，'word'：使用这种模式，使用的语料首先要经过预分词
                input_sentence_size=1e6,
                shuffle_input_sentence=True,
                normalization_rule_name='nmt_nfkc_cf',
            )

        spm_model = spm.SentencePieceProcessor(model_file=str(Path(prefix, f'spm{vocab_size}.model')))
        # 用这个子词切分模型，对语料库进行词语切分
        in_tag = {
            'train': 'train.clean',
            'valid': 'valid.clean',
            'test': 'test.raw.clean',
        }
        for split in ['train', 'valid', 'test']:
            for lang in [self.src_lang, self.tgt_lang]:
                out_path = Path(prefix,f'{split}.{lang}')
                if out_path.exists():
                    print(f"{out_path} exists. skipping spm_encode.")
                else:
                    out_f = open(Path(prefix, f'{split}.{lang}'), 'w', encoding='utf-8')
                    in_f = open(Path(prefix,f'{in_tag[split]}.{lang}'), 'r', encoding='utf-8')
                    for line in in_f:
                        line = line.strip()
                        tok = spm_model.encode(line, out_type=str)    # 这里好像直接编码了
                        print(' '.join(tok), file=out_f)












