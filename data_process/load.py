import logging
from fairseq.tasks.translation import TranslationTask, TranslationConfig
from fairseq import utils

from fairseq.tasks.fairseq_task import FairseqDataset
"""
--如果导入的fairseq-0.10.2中没有这个类，需要去git中找main版本,
--把translation的文件复制到本地translation,
--并在TranslationTask的构造函数__init__()下添加self.cfg = cfg
"""

def task(args):
    task_cfg = TranslationConfig(
        data=args.binpath,
        source_lang=args.src_lang,
        target_lang=args.tgt_lang,
        train_subset="train",
        required_seq_len_multiple=8,
        dataset_impl="mmap",
        upsample_primary=1,
    )
            # 自己定义一个日志
    proj_name = "TranslationData.seq2seq"
    logger = logging.getLogger(proj_name)
    logger.info("loading data for epoch 1")

    task = TranslationTask.setup_task(task_cfg)
    task.load_dataset(split="train", epoch=1)
    task.load_dataset(split="valid", epoch=1)

    return task

def load_data_iterator(task, split, epoch=1, max_tokens=4000, num_workers=1, cached=True):  # 每个batch含有4000个词

    batch_iterator = task.get_batch_iterator(
        dataset=task.dataset(split),
        max_tokens=max_tokens,
        max_sentences=None,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            max_tokens,
        ),
        ignore_invalid_inputs=True,
        seed=73,
        num_workers=num_workers,
        epoch=epoch,
        disable_iterator_cache=not cached,
            # Set this to False to speed up. However, if set to False, changing max_tokens beyond
            # first call of this method has no effect.
    )
    return batch_iterator