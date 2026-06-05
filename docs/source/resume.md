Resume Training
===============

训练途中一定会有失败，继续训练必不可少。
继续训练的 cmd line 基本上复用了 megatron-lm 的设计。并不能说 megatron-lm 的设计很完美，而是因为没有必要改动。

Failures are inevitable during training, and resuming training is essential.
The command line for resuming training essentially reuses the design of Megatron-LM.
It's not that Megatron-LM's design is perfect, but rather that there's no need to make changes.

## save checkpoint

你**不能**打开这两个开关，否则这些状态在训练中会被丢弃。
这两个开关打开后放弃 optim 的存储。

you must not enable these two flags; otherwise, these states will be discarded during training.
Enabling these two flags will result in abandoning the storage of the optimizer (optim).

```bash
--no-save-optim \
--no-save-rng \
```

## load checkpoint (start)

从 dirA load checkpoint，通过 `--finetune` 放弃掉其中的（sft）训练进度，开始下一轮 post train。
`--no-load-optim` 和 `--no-load-rng` 会放弃掉 optim 和 rng，一般 post train 会放弃掉。
checkpoint 会存到 dirB。

Load the checkpoint from `dirA`, and use --finetune to discard the (SFT) training progress, starting the next round of post-training.
--no-load-optim and --no-load-rng will discard the optimizer (optim) and RNG states, which are typically discarded during post-training.
The checkpoint will be saved to `dirB`.

```bash
    --load dirA \
    --save dirB \
    --finetune \
    --no-load-optim \
    --no-load-rng \
```

## load checkpoint (resume)

去掉这几个 flag，因为你需要 load trainer 状态和 optim 。

Remove these flags because you need to load the trainer state and optimizer (optim).

```
    --finetune \
    --no-load-optim \
    --no-load-rng \
```

然后 load 目录改成 dirB，因为你要 load 你存下来的进度。

Then change the load directory to dirB, as you need to load the progress you previously saved.

```bash
    --load dirB \
    --save dirB \
```

## One-click solution

上面 3 个步骤改来改去还挺烦人的，我们做了一个小 trick 一键搞定。

The three steps above can be quite tedious to modify back and forth, so we came up with a little trick to get it all done in one go.

```bash
    --load dirA \
    --save dirB \
    --auto-set-finetune-arg \
```

不要忘记**去掉** `--no-save-optim` 和 `--no-save-rng`。

Don't forget to remove --no-save-optim and --no-save-rng.

code:

```python
if args.auto_set_finetune_arg:
    txt = os.path.join(args.save, 'latest_checkpointed_iteration.txt')
    if (not os.path.exists(txt) or open(txt, 'r').read(7).strip() == 'release'):
        args.finetune = True
        args.no_load_optim = True
        args.no_load_rng = True
    else:
        args.finetune = False
        args.no_load_optim = False
        args.no_load_rng = False
        args.load = args.save
```
