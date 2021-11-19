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
def train_one_epoch(epoch_itr, model, task, criterion, optimizer, accum_steps=1):
    itr = epoch_itr.next_epoch_itr(shuffle=True)
    itr = iterators.GroupedIterator(itr, accum_steps)  # 梯度累積: 每 accum_steps 個 sample 更新一次

    stats = {"loss": []}
    scaler = GradScaler()  # 混和精度訓練 automatic mixed precision (amp)

    model.train()
    progress = tqdm.tqdm(itr, desc=f"train epoch {epoch_itr.epoch}", leave=False)
    for samples in progress:
        model.zero_grad()
        accum_loss = 0
        sample_size = 0
        # 梯度累積: 每 accum_steps 個 sample 更新一次
        for i, sample in enumerate(samples):
            if i == 1:
                # emptying the CUDA cache after the first step can reduce the chance of OOM
                torch.cuda.empty_cache()

            sample = utils.move_to_cuda(sample, device=device)
            target = sample["target"]
            sample_size_i = sample["ntokens"]
            sample_size += sample_size_i

            # 混和精度訓練
            with autocast():
                # 数据送到模型
                # net_output = model(**sample["net_input"])
                net_output = model.forward(**sample["net_input"])
                lprobs = F.log_softmax(net_output[0], -1)
                loss = criterion(lprobs.view(-1, lprobs.size(-1)), target.view(-1))

                # logging
                accum_loss += loss.item()
                # back-prop
                scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        optimizer.multiply_grads(1 / (sample_size or 1.0))  # (sample_size or 1.0) handles the case of a zero gradient
        gnorm = nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)  # 梯度裁剪 防止梯度爆炸

        scaler.step(optimizer)
        scaler.update()

        # logging
        loss_print = accum_loss / sample_size
        stats["loss"].append(loss_print)
        progress.set_postfix(loss=loss_print)

    loss_print = np.mean(stats["loss"])
    logger.info(f"training loss: {loss_print:.4f}")
    return stats

# 2. 验证和测试
# fairseq 的 beam search generator
# 給定模型和輸入序列，用 beam search 生成翻译結果



def decode(toks, dictionary):
    # 從 Tensor 轉成人看得懂的句子
    s = dictionary.string(
        toks.int().cpu(),
        args.post_process,
    )
    return s if s else "<unk>"
# 测试，产生翻译结果，可用于验证集上产生对比结果
def inference_step(task, sample, model, args):

    sequence_generator = task.build_generator([models], args)
    gen_out = sequence_generator.generate([model], sample)
    srcs = []
    hyps = []
    refs = []
    for i in range(len(gen_out)):
        # 對於每個 sample, 收集輸入，輸出和參考答案，稍後計算 BLEU
        srcs.append(decode(
            utils.strip_pad(sample["net_input"]["src_tokens"][i], task.source_dictionary.pad()),
            task.source_dictionary,
        ))  # utils可自己写
        hyps.append(decode(
            gen_out[i][0]["tokens"], # 0 代表取出 beam 內分數第一的輸出結果
            task.target_dictionary,
        ))
        refs.append(decode(
            utils.strip_pad(sample["target"][i], task.target_dictionary.pad()),
            task.target_dictionary,
        ))
    return srcs, hyps, refs


def validate(model, task, criterion, args, log_to_wandb=True):
    logger.info('begin validation')
    itr = load_data_iterator(task, "valid", 1, args.max_tokens, args.num_workers).next_epoch_itr(shuffle=False)
    # 在验证集上显示更多的信息
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

            # 進行推論
            s, h, r = inference_step(task, sample, model, args)
            srcs.extend(s)
            hyps.extend(h)
            refs.extend(r)

    tok =  task.cfg.target_lang
    # stats["loss"] = np.mean(stats["loss"])
    stats["loss"] = torch.stack(stats["loss"]).mean().item()
    stats["bleu"] = sacrebleu.corpus_bleu(hyps, [refs], tokenize=tok)  # 計算BLEU score
    stats["srcs"] = srcs
    stats["hyps"] = hyps
    stats["refs"] = refs

    if args.use_wandb and log_to_wandb:
        wandb.log({
            "valid/loss": stats["loss"],
            "valid/bleu": stats["bleu"].score,
        }, commit=False)

    show_id = np.random.randint(len(hyps))
    logger.info("example source: " + srcs[show_id])
    logger.info("example hypothesis: " + hyps[show_id])
    logger.info("example reference: " + refs[show_id])

    # show bleu results
    logger.info(f"validation loss:\t{stats['loss']:.4f}")
    logger.info(stats["bleu"].format())
    return stats

# 3. 储存即载入模型参数

def validate_and_save(model, task, criterion, optimizer, epoch, args, save):
    stats = validate(model, task, criterion, args)
    bleu = stats['bleu']
    loss = stats['loss']
    if save:
        # save epoch checkpoints
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


def train(epoch_itr, model, task, criterion, optimizer, args):
    for i in range(args.max_epoch):
        train_one_epoch(epoch_itr, model, task, criterion, optimizer, args.accu_steps)
        stats = validate_and_save(model, task, criterion, optimizer, epoch=epoch_itr.epoch, args=args, save=True)
        logger.info("end of epoch {}".format(epoch_itr.epoch))
        epoch_itr = load_data_iterator(
                            task=task,
                            split="train",
                            epoch=epoch_itr.next_epoch_idx,
                            max_tokens=config.max_tokens,
                            num_workers=config.num_workers,
        )

def load_checkpoint(model, optimizer=None, name=None):
    name = name if name else "checkpoint_last.pt"
    checkpath = Path(args.save_dir) / name
    if checkpath.exists():
        check = torch.load(checkpath)
        model.load_state_dict(check["model"])
        stats = check["stats"]
        step = "unknown"
        if optimizer != None:
            optimizer._step = step = check["optim"]["step"]
        logger.info(f"loaded checkpoint {checkpath}: step={step} loss={stats['loss']} bleu={stats['bleu']}")
    else:
        logger.info(f"no checkpoints found at {checkpath}!")


