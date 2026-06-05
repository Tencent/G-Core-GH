# coding=utf-8
"""
自定义 Loss 的 Actor 模型 - DeepSeek 3.2 无偏 KL 估计实现

DeepSeek 3.2 论文中提出了无偏的 KL 散度估计方法：
原始 KL 估计: E_{π_old}[log(π_old/π_ref)] 
无偏 KL 估计: E_{π_old}[(π_θ/π_old) * log(π_θ/π_ref)]

通过乘上重要性采样系数 (π_θ/π_old)，将基于旧策略采样的 KL 估计
修正为对当前策略的无偏估计。

参考: https://arxiv.org/abs/2501.12948
"""
from typing import Any, Dict, List
from typing_extensions import override
from functools import partial
from packaging.version import Version

import torch

from megatron.core import mpu, package_info
from megatron.training import get_args
from megatron.training.utils import unwrap_model

from gpatch.core.models.gpt.gpt_ppo_actor_model import GptPpoActorModel
from gpatch.core.aligner_helper import (
    from_parallel_logits_to_logprobs,
    masked_mean,
    average_losses_across_data_parallel_group,
)
from gpatch.core.ppo_helper import vocab_parallel_entropy, calculate_kl_loss


class GptPpoCustomLossActorModel(GptPpoActorModel):
    """
    DeepSeek 3.2 无偏 KL 估计的 Actor 模型
    """
    @override
    def get_actor_grpo_forward_output_and_loss_func(self, seqlen: int):
        """
        重写此方法来实现 DeepSeek 3.2 的无偏 KL 估计
        """
        def fwd_output_and_loss_func(seqlen, data_iterator, model):
            # ========== 数据准备部分 (保持不变) ==========
            batches: List[Dict[str, Any]] = next(data_iterator)

            batch, fwd_kwargs = self.prepare_data_for_grpo_loss(batches, seqlen)
            for key in ["mask", "advantages", "prev_log_probs", "ref_log_probs", "target"]:
                assert key in batch

            # 前向传播
            if not self.ppo_pack_seq:
                parallel_logits = model(**fwd_kwargs)
            else:
                parallel_logits = model(**fwd_kwargs)

            if isinstance(parallel_logits, tuple):
                parallel_logits = parallel_logits[0]
            assert isinstance(parallel_logits, torch.Tensor)

            # ========== 自定义 Loss 函数 ==========
            def loss_func(parallel_logits):
                parallel_logits = parallel_logits.float()

                # 获取必要的数据
                mask = batch["mask"]
                advantages = batch["advantages"]
                prev_log_probs = batch["prev_log_probs"]
                ref_log_probs = batch["ref_log_probs"]
                target = batch["target"]

                assert advantages.dtype == torch.float32
                assert prev_log_probs.dtype == torch.float32

                parallel_logits_clone = parallel_logits.clone()

                # 计算当前 log probs
                curr_log_probs = from_parallel_logits_to_logprobs(
                    vocab_parallel_logits=parallel_logits,
                    target=target,
                    ignore_cp=self.ppo_pack_seq,
                )

                # 计算熵
                scaled_entropy = vocab_parallel_entropy(
                    parallel_logits_clone, mask, ignore_cp=self.ppo_pack_seq
                )

                # ################################################################
                # ########### 【开始】DeepSeek 3.2 无偏 KL 估计实现 ###########
                # ################################################################

                # 1. 计算重要性采样系数 (importance sampling ratio)
                #    ratio = π_θ(a|s) / π_θ_old(a|s) = exp(log π_θ - log π_θ_old)
                log_ratio = curr_log_probs - prev_log_probs
                ratios = log_ratio.exp()

                # 2. 带 clip 的 ratio (用于 actor loss)
                ratios_clamped = ratios.clamp(1.0 - self.ratio_eps, 1.0 + self.ratio_eps)

                # 3. 计算 Actor Loss (PPO clip loss)
                loss1 = -advantages * ratios
                loss2 = -advantages * ratios_clamped
                actor_loss = torch.maximum(loss1, loss2)
                actor_loss = masked_mean(actor_loss, mask)

                # 4. 计算原始 KL loss (有偏估计)
                #    原始: E_{π_old}[log(π_θ/π_ref)] = E_{π_old}[log π_θ - log π_ref]
                #    使用 low variance 形式: (π_ref/π_θ) - log(π_ref/π_θ) - 1
                raw_kl = calculate_kl_loss(
                    cur_log_probs=curr_log_probs,
                    ref_log_probs=ref_log_probs,
                    use_absolute_kl=False,
                    use_low_var_kl=True,
                )

                # 5. 【核心改动】DeepSeek 3.2 无偏 KL 估计
                #    无偏估计: E_{π_old}[(π_θ/π_old) * KL(π_θ || π_ref)]
                #    通过乘上重要性采样系数 ratio，将有偏估计修正为无偏估计
                #
                #    原理: 我们用 π_old 采样，但想估计 E_{π_θ}[f(x)]
                #    E_{π_θ}[f(x)] = E_{π_old}[(π_θ/π_old) * f(x)]
                #
                #    注意: 这里对 ratio 做 detach，避免 KL loss 的梯度通过 ratio 传播
                #    只让 KL loss 的梯度通过 raw_kl 传播
                unbiased_kl = raw_kl * ratios.detach()

                # 6. 计算最终的 KL loss
                kl_loss = masked_mean(unbiased_kl, mask)

                # 7. 同时记录原始有偏 KL loss 用于对比
                biased_kl_loss = masked_mean(raw_kl, mask)

                # 8. 最终 loss = actor_loss + kl_loss - entropy_bonus
                loss = actor_loss + kl_loss * self.config.grpo_kl_loss_beta - scaled_entropy * self.entropy_bonus

                # ################################################################
                # ########### 【结束】DeepSeek 3.2 无偏 KL 估计实现 ###########
                # ################################################################

                # 跨 DP 组平均
                reduced_losses = average_losses_across_data_parallel_group(
                    [loss, actor_loss, kl_loss, biased_kl_loss]
                )

                # 处理版本兼容性
                if Version(package_info.__version__) < Version("0.12.1"):
                    bwd_loss = loss * self.config.context_parallel_size
                else:
                    bwd_loss = loss.clone()

                # 返回 loss 和 metrics
                return (
                    bwd_loss,
                    {
                        "loss": reduced_losses[0],
                        "actor_loss": reduced_losses[1],
                        # ################################################################
                        # ########### 【新增 metrics】对比有偏和无偏 KL ###########
                        # ################################################################
                        "kl_loss_unbiased": reduced_losses[2],  # DeepSeek 3.2 无偏 KL
                        "kl_loss_biased": reduced_losses[3],  # 原始有偏 KL (用于对比)
                        "ppo_ratio": masked_mean(ratios.detach(), mask),
                        "ppo_ratio_clamped": masked_mean(ratios_clamped.detach(), mask),
                        "scaled_entropy": scaled_entropy.detach(),
                    },
                )

            return parallel_logits, loss_func

        return partial(fwd_output_and_loss_func, seqlen)
