from fairseq.data import iterators
from torch.cuda.amp import GradScaler, autocast
from fairseq import utils
from data.load import load_data_iterator
from pathlib import Path
import shutil
import sacrebleu
import torch
import torch.nn as nn
import tqdm.auto as tqdm


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 1. 训练

def train(epoch_itr, model, task, criterion, optimizer, args):
    for i in range(args.max_epoch):
        train_one_epoch(epoch_itr, model, criterion, optimizer, args.accu_steps)
        stats = validate_and_save(model, task, criterion, optimizer, epoch=epoch_itr.epoch, args=args)
        logger.info("end of epoch {}".format(epoch_itr.epoch))
        epoch_itr = load_data_iterator(
                            task=task,
                            split="train",
                            epoch=epoch_itr.next_epoch_idx,
                            max_tokens=config.max_tokens,
                            num_workers=config.num_workers,
        )

def train_one_epoch(epoch_itr, model, criterion, optimizer, accum_steps=1):
    itr = epoch_itr.next_epoch_itr(shuffle=True)
    itr = iterators.GroupedIterator(itr, accum_steps)  # 梯度累积: 每 accum_steps 个 sample 更新一次

    stats = {"loss": []}
    scaler = GradScaler()  # 混合精度训练 automatic mixed precision (amp)

    model.train()
    progress = tqdm.tqdm(itr, desc=f"train epoch {epoch_itr.epoch}", leave=False)
    for samples in progress:
        model.zero_grad()
        accum_loss = 0
        sample_size = 0
        # 梯度累积: 每 accum_steps 个 sample 更新一次
        for i, sample in enumerate(samples):
            if i == 1:
                # 第一步后清空CUDA缓存可以减少OOM的机会
                torch.cuda.empty_cache()

            sample = utils.move_to_cuda(sample, device=device)
            target = sample["target"]
            sample_size_i = sample["ntokens"]
            sample_size += sample_size_i
            # 混合精度训练
            with autocast():
                # 数据送到模型
                # net_output = model(**sample["net_input"])
                net_output = model.forward(**sample["net_input"])
                lprobs = F.log_softmax(net_output[0], -1)

                loss = criterion(lprobs.view(-1, lprobs.size(-1)), target.view(-1))
                accum_loss += loss.item()
                scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        optimizer.multiply_grads(1 / (sample_size or 1.0))  # (sample_size or 1.0) 处理零梯度的情况
        gnorm = nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)  # 梯度裁剪 防止梯度爆炸

        scaler.step(optimizer)
        scaler.update()

        loss_print = accum_loss / sample_size
        stats["loss"].append(loss_print)
        progress.set_postfix(loss=loss_print)

    loss_print = np.mean(stats["loss"])
    logger.info(f"training loss: {loss_print:.4f}")
    return stats

# 2. 验证

def validate(model, task, criterion, args, log_to_wandb=True):
    logger.info('begin validation')
    itr = load_data_iterator(task, "valid", 1, args.max_tokens, args.num_workers).next_epoch_itr(shuffle=False)

    stats = {"loss": [], "bleu": 0, "srcs": [], "hyps": [], "refs": []}
    srcs = []
    hyps = []
    refs = []

    model.eval()
    progress = tqdm.tqdm(itr, desc=f"validation", leave=False)
    with torch.no_grad():
        for i, sample in enumerate(progress):
            # validation loss
            sample = utils.move_to_cuda(sample, device=device)
            net_output = model.forward(**sample["net_input"])
            lprobs = F.log_softmax(net_output[0], -1)

            target = sample["target"]
            sample_size = sample["ntokens"]

            loss = criterion(lprobs.view(-1, lprobs.size(-1)), target.view(-1)) / sample_size
            progress.set_postfix(valid_loss=loss.item())
            stats["loss"].append(loss)
            # 进行预测
            s, h, r = inference_step(task, sample, model, args)
            srcs.extend(s)
            hyps.extend(h)
            refs.extend(r)

    tok =  task.cfg.target_lang
    # stats["loss"] = np.mean(stats["loss"])
    stats["loss"] = torch.stack(stats["loss"]).mean().item()
    stats["bleu"] = sacrebleu.corpus_bleu(hyps, [refs], tokenize=tok)  # 计算BLEU score
    stats["srcs"] = srcs
    stats["hyps"] = hyps
    stats["refs"] = refs
    # 是否显示验证集上的损失和belu
    if args.use_wandb and log_to_wandb:
        wandb.log({
            "valid/loss": stats["loss"],
            "valid/bleu": stats["bleu"].score,
        }, commit=False)

    show_id = np.random.randint(len(hyps))
    logger.info("example source: " + srcs[show_id])
    logger.info("example hypothesis: " + hyps[show_id])
    logger.info("example reference: " + refs[show_id])

    logger.info(f"validation loss:\t{stats['loss']:.4f}")
    logger.info(stats["bleu"].format())
    return stats

def validate_and_save(model, task, criterion, optimizer, epoch, args):
    stats = validate(model, task, criterion, args)
    bleu = stats['bleu']
    loss = stats['loss']

    # 存储模型
    save_dir = Path(args.save_dir).absolute()
    save_dir.mkdir(parents=True, exist_ok=True)

    check = {
        "model": model.state_dict(),
        "stats": {"bleu": bleu.score, "loss": loss},
        "optim": {"step": optimizer._step}
    }
    # 保存最后一次结果
    if epoch == args.max_epoch:
        torch.save(check, Path(save_dir, f"checkpoint_last.pt"))
        logger.info(f"saved epoch checkpoint: {save_dir}/checkpoint_last.pt")

        with open(save_dir / f"samples——last.{args.src-lang}-{args.tgt-lang}.txt", "w") as f:
            for s, h in zip(stats["srcs"], stats["hyps"]):
                f.write(f"{s}\t{h}\n")
    # 获得的做好的结果
    if getattr(validate_and_save, "best_bleu", 0) < bleu.score:
        validate_and_save.best_bleu = bleu.score
        torch.save(check, save_dir / f"checkpoint_best.pt")

    return stats

# 3. 测试

def inference_step(task, sample, model, args):

    sequence_generator = task.build_generator([models], args)
    gen_out = sequence_generator.generate([model], sample)
    srcs = []
    hyps = []
    refs = []
    for i in range(len(gen_out)):
        # 收集输入信息
        srcs.append(decode(
            utils.strip_pad(sample["net_input"]["src_tokens"][i], task.source_dictionary.pad()),
            task.source_dictionary,
        ))
        # 收集输出信息
        hyps.append(decode(
            gen_out[i][0]["tokens"], # 0 代表取出 beam 內分数第一的结果
            task.target_dictionary,
        ))
        # 收集目标信息
        refs.append(decode(
            utils.strip_pad(sample["target"][i], task.target_dictionary.pad()),
            task.target_dictionary,
        ))
    return srcs, hyps, refs


def generate_prediction(args, model, task, split="test", outfile="./prediction.txt"):
    task.load_dataset(split=split, epoch=1)
    itr = load_data_iterator(task, split, 1, args.max_tokens, args.num_workers).next_epoch_itr(shuffle=False)

    idxs = []
    hyps = []

    model.eval()
    progress = tqdm.tqdm(itr, desc=f"prediction")
    with torch.no_grad():
        for i, sample in enumerate(progress):

            sample = utils.move_to_cuda(sample, device=device)
            s, h, r = inference_step(sample, model)

            hyps.extend(h)
            idxs.extend(list(sample['id']))

    hyps = [x for _, x in sorted(zip(idxs, hyps))]
    with open(outfile, "w") as f:
        for h in hyps:
            f.write(h + "\n")



# 给定模型和输入序列，用 beam search 生成翻译结果
def decode(toks, dictionary):
    # 解码出人能看懂的句子输出
    s = dictionary.string(
        toks.int().cpu(),
        args.post_process,
    )
    return s if s else "<unk>"