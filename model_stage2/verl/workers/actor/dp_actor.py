# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Implement Actor
"""

import os
from collections import defaultdict
from typing import Any, Optional

import torch
import torch.nn.functional as F
import torch.distributed as dist
from einops import rearrange
from ray.experimental.tqdm_ray import tqdm
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from ...protocol import DataProto, batch_collate
from ...trainer.core_algos import average_loss, compute_kl, compute_policy_loss
from ...utils import torch_functional as VF
from ...utils.py_functional import append_to_dict
from ...utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from ...utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs
from ...utils.torch_functional import masked_mean
from .base import BasePPOActor
from .config import ActorConfig


try:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
except ImportError:
    pass


__all__ = ["DataParallelPPOActor"]


class DataParallelPPOActor(BasePPOActor):
    def __init__(
        self,
        config: ActorConfig,
        actor_module: nn.Module,
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """
        When optimizer is None, it is Reference Policy
        """
        super().__init__(config)
        self.rank = int(os.getenv("RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        if config.use_torch_compile:
            self.log_probs_from_logits = torch.compile(VF.log_probs_from_logits, dynamic=True)
        else:
            self.log_probs_from_logits = VF.log_probs_from_logits
            
        self.config = config
        # Only ActorConfig has training-specific fields; RefConfig does not
        if isinstance(config, ActorConfig):
            self.ppo_epochs = config.ppo_epochs
            self.ppo_mini_batch_size = config.global_batch_size // config.micro_batch_size_per_device_for_update
            self.max_grad_norm = config.max_grad_norm
            self.clip_range = config.clip_range
            self.vf_coef = config.vf_coef
            self.ent_coef = config.ent_coef
            self.use_critic = config.use_critic
            self.rl_steps_completed = 0
            self.total_rl_steps = 0
            self.rl_training_start_time = None
            self.rl_training_end_time = None
            self.rl_training_active = False
        # 为bbox预测头创建单独的Adam优化器
        # self.bbox_optimizer = None
    # def update_parameters(self, 
    #                      policy_module: FSDP, 
    #                      critic_module: FSDP, 
    #                      ref_module: FSDP, 
    #                      rollout_data: dict,
    #                      is_sft_step: bool = None):
    #     """
    #     更新参数，支持交替SFT和RL步骤
    #     """
    #     # 如果未指定步骤类型，则根据计数器决定
    #     if is_sft_step is None:
    #         self.step_count += 1
    #         is_sft_step = (self.step_count % 2) == 1  # 奇数步为SFT，偶数步为RL
        
    #     if is_sft_step:
    #         return self._update_sft_step(policy_module, rollout_data)
    #     else:
    #         return self._update_rl_step(policy_module, critic_module, ref_module, rollout_data)

    # def _update_sft_step(self, policy_module: FSDP, rollout_data: dict):
    #     """
    #     执行SFT步骤更新
    #     """
    #     obs = rollout_data['obs']
    #     attention_mask = rollout_data['attention_mask']
    #     sft_targets = rollout_data.get('sft_targets', None)
        
    #     if sft_targets is None:
    #         # 如果没有SFT目标，跳过这一步或使用默认值
    #         print("Warning: No SFT targets provided, skipping SFT step")
    #         return {}
        
    #     # 计算SFT损失
    #     policy_output = policy_module(obs, attention_mask=attention_mask)
    #     policy_logits = policy_output.logits
        
    #     # 计算SFT损失
    #     sft_loss = self._compute_sft_loss(policy_logits, sft_targets, attention_mask)
        
    #     # 反向传播
    #     sft_loss.backward()
        
    #     # 梯度裁剪
    #     if self.max_grad_norm is not None:
    #         torch.nn.utils.clip_grad_norm_(policy_module.parameters(), self.max_grad_norm)
        
    #     # 更新参数
    #     self.optimizer.step()
    #     self.optimizer.zero_grad()
        
    #     return {'sft_loss': sft_loss.item(), 'step_type': 'SFT'}

    # def _update_rl_step(self, 
    #                    policy_module: FSDP, 
    #                    critic_module: FSDP, 
    #                    ref_module: FSDP, 
    #                    rollout_data: dict):
    #     """
    #     执行RL步骤更新（原来的PPO更新逻辑）
    #     """
    #     # 提取rollout数据
    #     obs = rollout_data['obs']
    #     action = rollout_data['action']
    #     old_logprob = rollout_data['logprob']
    #     reward = rollout_data['reward']
    #     advantage = rollout_data['advantage']
    #     return_ = rollout_data['return']
    #     old_value = rollout_data['value']
    #     attention_mask = rollout_data['attention_mask']
        
    #     with torch.no_grad():
    #         ref_output = ref_module(obs, attention_mask=attention_mask)
    #         ref_logits = ref_output.logits
    #         ref_logprob = self.compute_logprob(ref_logits, action)
        
    #     # 存储所有损失
    #     policy_losses = []
    #     value_losses = []
        
    #     for _ in range(self.ppo_epochs):
    #         batch_indices = torch.randperm(obs.size(0))
    #         mini_batch_size = obs.size(0) // self.ppo_mini_batch_size
            
    #         for i in range(0, obs.size(0), mini_batch_size):
    #             mini_batch_idx = batch_indices[i:i+mini_batch_size]
                
    #             # 获取当前批次数据
    #             mini_obs = obs[mini_batch_idx]
    #             mini_action = action[mini_batch_idx]
    #             mini_old_logprob = old_logprob[mini_batch_idx]
    #             mini_advantage = advantage[mini_batch_idx]
    #             mini_return = return_[mini_batch_idx]
    #             mini_old_value = old_value[mini_batch_idx]
    #             mini_mask = attention_mask[mini_batch_idx]
                
    #             # 前向传播
    #             policy_output = policy_module(mini_obs, attention_mask=mini_mask)
    #             policy_logits = policy_output.logits
                
    #             # 计算当前动作的log概率
    #             policy_logprob = self.compute_logprob(policy_logits, mini_action)
                
    #             # 计算价值
    #             if self.use_critic:
    #                 value_output = critic_module(mini_obs, attention_mask=mini_mask)
    #                 value = value_output.logits.squeeze(-1)
    #             else:
    #                 value = mini_old_value
                
    #             # 计算PPO损失
    #             ratio = torch.exp(policy_logprob - mini_old_logprob)
    #             pg_loss = self._compute_policy_gradient_loss(ratio, mini_advantage, mini_mask)
                
    #             # 价值损失
    #             vf_loss = self._compute_value_loss(value, mini_old_value, mini_return, mini_mask)
                
    #             # 总损失
    #             total_loss = pg_loss + self.vf_coef * vf_loss
                
    #             # 反向传播
    #             total_loss.backward()
                
    #             # 梯度裁剪
    #             if self.max_grad_norm is not None:
    #                 torch.nn.utils.clip_grad_norm_(policy_module.parameters(), self.max_grad_norm)
                
    #             # 更新参数
    #             self.optimizer.step()
    #             self.optimizer.zero_grad()
                
    #             # 记录损失
    #             policy_losses.append(pg_loss.item())
    #             value_losses.append(vf_loss.item())
        
    #     # 返回统计信息
    #     return {
    #         'policy_loss': sum(policy_losses) / len(policy_losses) if policy_losses else 0,
    #         'value_loss': sum(value_losses) / len(value_losses) if value_losses else 0,
    #         'step_type': 'RL'
    #     }

    # def _compute_sft_loss(self, logits, targets, mask):
    #     """
    #     计算监督学习损失
    #     """
    #     # 将logits和targets调整为合适形状
    #     shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
    #     shift_labels = targets[..., 1:].contiguous().view(-1)
        
    #     # 计算交叉熵损失，忽略pad token
    #     loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100, reduction='none')
        
    #     # 应用mask并计算平均损失
    #     loss = loss.view(targets.size(0), -1)
    #     mask = mask[:, 1:]  # 对齐维度
    #     loss = (loss * mask).sum() / mask.sum()
        
    #     return loss

    # def _compute_policy_gradient_loss(self, ratio, advantage, mask):
    #     """
    #     计算策略梯度损失（PPO的clip loss）
    #     """
    #     pg_loss1 = advantage * ratio
    #     pg_loss2 = advantage * torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
    #     pg_loss = -torch.min(pg_loss1, pg_loss2)
    #     pg_loss = masked_mean(pg_loss, mask)
    #     return pg_loss

    # def _compute_value_loss(self, values, old_values, returns, mask):
    #     """
    #     计算价值函数损失
    #     """
    #     # 计算价值预测误差
    #     value_pred_clipped = old_values + torch.clamp(
    #         values - old_values,
    #         -self.clip_range,
    #         self.clip_range
    #     )
    #     value_loss1 = (values - returns) ** 2
    #     value_loss2 = (value_pred_clipped - returns) ** 2
    #     value_loss = torch.max(value_loss1, value_loss2)
    #     value_loss = 0.5 * masked_mean(value_loss, mask)
    #     return value_loss

    #   以上是改的sft的



    def _forward_micro_batch(self, micro_batch: dict[str, torch.Tensor], temperature: float) -> torch.Tensor:
        """
        Returns:
            log_probs: # (bs, response_len)
        """
        input_ids = micro_batch["input_ids"]
        batch_size, seqlen = input_ids.shape
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        responses = micro_batch["responses"]
        response_length = responses.size(-1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

        multi_modal_inputs = defaultdict(list)
        if "multi_modal_inputs" in micro_batch:
            multi_modal_inputs = batch_collate(micro_batch["multi_modal_inputs"])
            multi_modal_inputs = {key: torch.cat(value, dim=0) for key, value in multi_modal_inputs.items()}
        else:
            multi_modal_inputs = {}

        if self.config.padding_free:
            input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # (total_nnz, 1)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

            # unpad the position_ids to align the rotary
            if position_ids.dim() == 3:
                position_ids_rmpad = (
                    index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                    .transpose(0, 1)
                    .unsqueeze(1)
                )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
            else:
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)

            # for compute the log_prob
            input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

            # pad and slice the inputs if sp > 1
            if self.config.ulysses_size > 1:
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=self.config.ulysses_size
                )
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad_rolled, None, self.config.ulysses_size
                )

            input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

            # only pass input_ids and position_ids to enable flash_attn_varlen
            output = self.actor_module(
                input_ids=input_ids_rmpad,
                attention_mask=None,
                position_ids=position_ids_rmpad,
                **multi_modal_inputs,
                use_cache=False,
            )  # prevent model thinks we are generating
            logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
            logits_rmpad.div_(temperature)
            # ((total_nnz / sp) + pad)
            log_probs = self.log_probs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

            # gather log_prob if sp > 1
            if self.config.ulysses_size > 1:
                # gather and unpad for the ulysses sp
                log_probs = gather_outputs_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)

            # pad back to (bsz, seqlen)
            full_log_probs = pad_input(
                hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
            )
            log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
        else:
            output = self.actor_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **multi_modal_inputs,
                use_cache=False,
            )
            logits: torch.Tensor = output.logits
            logits.div_(temperature)
            logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
            log_probs = self.log_probs_from_logits(logits, responses)  # (bsz, response_length)

        return log_probs

    def _optimizer_step(self) -> torch.Tensor:
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(self.config.max_grad_norm)
        else:
            grad_norm = nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.max_grad_norm)

        if not torch.isfinite(grad_norm):
            print("Gradient norm is not finite. Skip update.")
        else:
            self.actor_optimizer.step()

        self.actor_optimizer.zero_grad()
        return grad_norm

    @torch.no_grad()
    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        self.actor_module.eval()

        temperature = data.meta_info["temperature"]
        select_keys = ["input_ids", "attention_mask", "position_ids", "responses"]
        non_tensor_select_keys = ["multi_modal_inputs"]

        data = data.select(select_keys, non_tensor_select_keys)
        if self.config.dynamic_batching:
            max_token_len = self.config.micro_batch_size_per_device_for_experience * data.batch["input_ids"].size(-1)
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(self.config.micro_batch_size_per_device_for_experience)

        log_probs_lst = []
        if self.rank == 0:
            micro_batches = tqdm(micro_batches, desc="Compute log probs", position=1)

        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            log_probs = self._forward_micro_batch(model_inputs, temperature=temperature)
            log_probs_lst.append(log_probs)

        log_probs = torch.concat(log_probs_lst, dim=0)

        if self.config.dynamic_batching:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)

        return log_probs

    def update_policy(self, data: DataProto) -> dict[str, Any]: # 原代码
    
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid slient error
        select_keys = ["input_ids", "attention_mask", "position_ids", "responses", "response_mask"]
        select_keys.extend(["old_log_probs", "ref_log_probs", "advantages"])
        non_tensor_select_keys = ["multi_modal_inputs"]

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.select(select_keys, non_tensor_select_keys).split(self.config.global_batch_size_per_device)

        metrics = defaultdict(list)
        for _ in range(self.config.ppo_epochs):
            if self.rank == 0:
                mini_batches = tqdm(mini_batches, desc="Train mini-batches", position=1)

            for mini_batch in mini_batches:
                total_response_tokens = torch.sum(mini_batch.batch["response_mask"])
                dist.all_reduce(total_response_tokens, op=dist.ReduceOp.SUM)

                if self.config.dynamic_batching:
                    max_input_len = mini_batch.batch["input_ids"].size(-1)
                    max_token_len = self.config.micro_batch_size_per_device_for_update * max_input_len
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    micro_batches = mini_batch.split(self.config.micro_batch_size_per_device_for_update)

                if self.rank == 0:
                    micro_batches = tqdm(micro_batches, desc="Update policy", position=2)

                for micro_batch in micro_batches:
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_probs = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    # all return: (bsz, response_length)
                    log_probs = self._forward_micro_batch(model_inputs, temperature=temperature)

                    pg_loss, pg_metrics = compute_policy_loss(
                        old_log_probs=old_log_probs,
                        log_probs=log_probs,
                        advantages=advantages,
                        response_mask=response_mask,
                        clip_ratio_low=self.config.clip_ratio_low,
                        clip_ratio_high=self.config.clip_ratio_high,
                        clip_ratio_dual=self.config.clip_ratio_dual,
                        tau_positive=self.config.tau_positive,
                        tau_negative=self.config.tau_negative,
                        loss_type=self.config.loss_type,
                        loss_avg_mode=self.config.loss_avg_mode,
                    )
                    if self.config.use_kl_loss and "ref_log_probs" in model_inputs:
                        ref_log_probs = model_inputs["ref_log_probs"]
                        # compute kl loss
                        kld = compute_kl(
                            log_probs=log_probs,
                            ref_log_probs=ref_log_probs,
                            kl_penalty=self.config.kl_penalty,
                        )
                        kl_loss = average_loss(kld, response_mask, mode=self.config.loss_avg_mode)
                        loss = pg_loss + kl_loss * self.config.kl_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_coef
                    else:
                        loss = pg_loss

                    loss = loss * torch.sum(response_mask) * self.world_size / total_response_tokens
                    loss.backward()

                    batch_metrics = {f"actor/{k}": v for k, v in pg_metrics.items()}
                    batch_metrics["actor/pg_loss"] = pg_loss.detach().item()
                    append_to_dict(metrics, batch_metrics)

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        return metrics

    # 以下是改的sft的代码
    # def update_policy(self, data: DataProto) -> dict[str, Any]:
    #     """
    #     Update policy using RL (original logic)
    #     """
    #     return self._update_rl_step_from_dataproto(data)

    # def update_policy_sft(self, data: DataProto) -> dict[str, Any]:
    #     """
    #     Update policy using SFT for bbox regression
    #     """
    #     return self._update_sft_step_from_dataproto(data)

    # def update_policy_rl(self, data: DataProto) -> dict[str, Any]:
        """
        Update policy using RL (same as original)
        """
        return self._update_rl_step_from_dataproto(data)
    # 以上是改的sft的代码

    # def _update_rl_step_from_dataproto(self, data: DataProto) -> dict[str, Any]:
    #     """
    #     执行基于DataProto的RL步骤更新（原有的update_policy逻辑）
    #     """
    #     import time
    #     if not self.rl_training_active:
    #         self.rl_training_start_time = time.time()
    #         self.rl_training_active = True
    #         print(f"RL Training started at step {self.step_count}")

    #     self.actor_module.train()

    #     temperature = data.meta_info["temperature"]  # temperature必须在data.meta_info中以避免静默错误
    #     select_keys = ["input_ids", "attention_mask", "position_ids", "responses", "response_mask"]
    #     select_keys.extend(["old_log_probs", "ref_log_probs", "advantages", "returns", "values"])
    #     non_tensor_select_keys = ["multi_modal_inputs"]

    #     # 检查是否有bbox标签用于多任务学习
    #     print(f" ￥￥￥Data has bbox labels: {data.non_tensor_batch}")
    #     has_bbox_labels = 'bbox' in data.non_tensor_batch
    #     # Split to make minibatch iterator for updating the actor
    #     # See PPO paper for details. https://arxiv.org/abs/1707.06347
    #     mini_batches = data.select(select_keys, non_tensor_select_keys).split(self.config.global_batch_size_per_device)

    #     metrics = defaultdict(list)
    #     total_mini_batches = len(mini_batches) * self.config.ppo_epochs
    #     processed_mini_batches = 0
    #     print(f"Starting RL training: {self.config.ppo_epochs} epochs, {total_mini_batches} total mini-batches")
    #     for epoch in range(self.config.ppo_epochs):
    #         print(f"Starting epoch {epoch + 1}/{self.config.ppo_epochs}")

    #         if self.rank == 0:
    #             mini_batches = tqdm(mini_batches, desc="Train mini-batches", position=1)

    #         for mini_batch in mini_batches:
    #             total_response_tokens = torch.sum(mini_batch.batch["response_mask"])
    #             dist.all_reduce(total_response_tokens, op=dist.ReduceOp.SUM)

    #             if self.config.dynamic_batching:
    #                 max_input_len = mini_batch.batch["input_ids"].size(-1)
    #                 max_token_len = self.config.micro_batch_size_per_device_for_update * max_input_len
    #                 micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
    #             else:
    #                 micro_batches = mini_batch.split(self.config.micro_batch_size_per_device_for_update)

    #             if self.rank == 0:
    #                 micro_batches = tqdm(micro_batches, desc="Update policy", position=2)

    #             for micro_batch in micro_batches:
    #                 model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
    #                 response_mask = model_inputs["response_mask"]
    #                 old_log_probs = model_inputs["old_log_probs"]
    #                 advantages = model_inputs["advantages"]
    #                 returns = model_inputs["returns"]
    #                 # old_values = model_inputs["values"]

    #                 # all return: (bsz, response_length)
    #                 log_probs = self._forward_micro_batch(model_inputs, temperature=temperature)

    #                 pg_loss, pg_metrics = compute_policy_loss(
    #                     old_log_probs=old_log_probs,
    #                     log_probs=log_probs,
    #                     advantages=advantages,
    #                     response_mask=response_mask,
    #                     clip_ratio_low=self.config.clip_ratio_low,
    #                     clip_ratio_high=self.config.clip_ratio_high,
    #                     clip_ratio_dual=self.config.clip_ratio_dual,
    #                     tau_positive=self.config.tau_positive,
    #                     tau_negative=self.config.tau_negative,
    #                     loss_type=self.config.loss_type,
    #                     loss_avg_mode=self.config.loss_avg_mode,
    #                 )

    #                 # 计算bbox损失（多任务学习）
    #                 bbox_loss = None
    #                 if has_bbox_labels and 'bbox' in micro_batch.non_tensor_batch:
    #                     # 获取输入数据
    #                     obs = model_inputs["input_ids"]
    #                     attention_mask = model_inputs["attention_mask"]
    #                     bbox_targets = micro_batch.non_tensor_batch['bbox']
                        
    #                     # LLM前向传播获取logits
    #                     llm_output = self.actor_module(obs, attention_mask=attention_mask)
    #                     policy_logits = llm_output.logits
                        
    #                     # 转换bbox目标并计算损失
    #                     bbox_tensor = self._convert_bbox_to_tensor(bbox_targets)
    #                     bbox_tensor = bbox_tensor.to(policy_logits.device).to(policy_logits.dtype)
    #                     bbox_loss = self._compute_bbox_regression_loss(policy_logits, bbox_tensor, attention_mask)

                    
    #                 if self.config.use_kl_loss and "ref_log_probs" in model_inputs:
    #                     ref_log_probs = model_inputs["ref_log_probs"]
    #                     # compute kl loss
    #                     kld = compute_kl(
    #                         log_probs=log_probs,
    #                         ref_log_probs=ref_log_probs,
    #                         kl_penalty=self.config.kl_penalty,
    #                     )
    #                     kl_loss = average_loss(kld, response_mask, mode=self.config.loss_avg_mode)
    #                     loss = pg_loss + kl_loss * self.config.kl_coef
    #                     metrics["actor/kl_loss"] = kl_loss.detach().item()
    #                     metrics["actor/kl_coef"] = self.config.kl_coef
    #                 else:
    #                     loss = pg_loss

    #                 # 添加bbox损失到总损失中（如果存在）
    #                 if bbox_loss is not None:
    #                     # bbox_loss_weight可以设为较小的值，比如0.1或0.01
    #                     loss = loss + bbox_loss * 0.1
    #                 loss = loss * torch.sum(response_mask) * self.world_size / total_response_tokens
    #                 loss.backward()

    #                 batch_metrics = {f"actor/{k}": v for k, v in pg_metrics.items()}
    #                 batch_metrics["actor/pg_loss"] = pg_loss.detach().item()
    #                 append_to_dict(metrics, batch_metrics)

    #                 processed_mini_batches += 1
    #                 if processed_mini_batches % max(1, total_mini_batches // 10) == 0:
    #                     progress_percent = (processed_mini_batches / total_mini_batches) * 100
    #                     print(f"RL Training Progress: {progress_percent:.1f}% ({processed_mini_batches}/{total_mini_batches})")

    #             # grad_norm = self._optimizer_step()  # 原代码
    #             grad_norm = self._optimizer_step_with_bbox()
    #             append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})
        
    #     self.rl_steps_completed += 1
        
    #     # 计算平均指标
    #     avg_metrics = {}
    #     for key, values in metrics.items():
    #         if values:  # 如果列表非空
    #             avg_metrics[key] = sum(values) / len(values)
    #         else:
    #             avg_metrics[key] = 0
        
    #     avg_metrics["rl_steps_completed"] = self.rl_steps_completed
    #     avg_metrics["rl_training_status"] = "completed"
        
    #     # 标记RL训练结束
    #     self.rl_training_end_time = time.time()
    #     training_duration = self.rl_training_end_time - self.rl_training_start_time
    #     avg_metrics["rl_training_duration"] = training_duration
    #     self.rl_training_active = False
        
    #     print(f"RL Training completed in {training_duration:.2f} seconds")
    #     print(f"Completed RL step #{self.rl_steps_completed}")

    #     return avg_metrics
    

    # def _optimizer_step_with_bbox(self) -> torch.Tensor:
    #     """同时更新主模型和bbox预测头的优化步骤"""
    #     # 更新主模型
    #     if isinstance(self.actor_module, FSDP):
    #         grad_norm = self.actor_module.clip_grad_norm_(self.config.max_grad_norm)
    #     else:
    #         grad_norm = nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.max_grad_norm)

    #     if not torch.isfinite(grad_norm):
    #         print("Gradient norm is not finite. Skip update.")
    #     else:
    #         self.actor_optimizer.step()

    #     self.actor_optimizer.zero_grad()
        
    #     # 同时更新bbox预测头（如果存在）
    #     if hasattr(self, 'bbox_optimizer') and self.bbox_optimizer is not None:
    #         self.bbox_optimizer.step()
    #         self.bbox_optimizer.zero_grad()
            
    #     return grad_norm


    # def _update_sft_step_from_dataproto(self, data: DataProto) -> dict[str, Any]:
    #     """
    #     执行基于DataProto的SFT步骤更新，专门针对bbox坐标预测任务
    #     """
    #     print("DEBUG: Starting _update_sft_step_from_dataproto in DataParallelPPOActor")  # 添加调试输出
        
    #     # 查看actor_optimizer的信息
    #     print(f"Actor optimizer: {self.actor_optimizer}")
    #     if self.actor_optimizer is not None:
    #         print(f"Actor optimizer type: {type(self.actor_optimizer)}")
    #         if hasattr(self.actor_optimizer, 'param_groups'):
    #             print(f"Actor optimizer param groups: {[group['lr'] for group in self.actor_optimizer.param_groups]}")
    #             # 检查并修复学习率
    #             for i, param_group in enumerate(self.actor_optimizer.param_groups):
    #                 # if param_group['lr'] == 0.0:
    #                 # 设置一个合适的学习率用于SFT训练
    #                 original_lr = param_group['lr']
    #                 param_group['lr'] = 1e-4  # 设置一个SFT学习率
    #                 print(f"Adjusted learning rate from {original_lr} to {param_group['lr']} for SFT training")
    #     else:
    #         print("Actor optimizer is None")
        
    #     # 获取输入数据
    #     obs = data.batch['input_ids']  # (batch_size, seq_len)
    #     attention_mask = data.batch['attention_mask']  # (batch_size, seq_len)
        
    #     # 获取bbox坐标作为监督信号
    #     if 'bbox' in data.non_tensor_batch:
    #         bbox_targets = data.non_tensor_batch['bbox']  # (batch_size,) 数组，每个元素是[[x1, y1, x2, y2]]
    #     else:
    #         print("Warning: No bbox targets provided, skipping SFT step")
    #         return {}
        
    #     self.actor_module.train()
        
    #     # 前向传播获取模型输出
    #     policy_output = self.actor_module(obs, attention_mask=attention_mask)
    #     policy_logits = policy_output.logits  # (batch_size, seq_len, vocab_size)
        
    #     # 处理bbox targets，将其转换为数值tensor
    #     print("处理之前的bbox_tensor:", bbox_targets)
    #     bbox_tensor = self._convert_bbox_to_tensor(bbox_targets)
    #     bbox_tensor = bbox_tensor.to(policy_logits.device).to(policy_logits.dtype)
    #     print("处理之后的bbox_tensor:", bbox_tensor)

        
    #     print(f"DEBUG: Before computing bbox regression loss")  # 添加调试输出
    #     # print("logiys：",policy_logits)
    #     # print("logits.shape:", policy_logits.shape)
    #     # print("attention_mask", attention_mask)
    #     # 计算bbox回归损失
    #     bbox_loss = self._compute_bbox_regression_loss(policy_logits, bbox_tensor, attention_mask)
    #     print(f"DEBUG: After computing bbox regression loss, loss value: {bbox_loss.item()}")  # 添加调试输出

    #     # 反向传播
    #     print(f"DEBUG: Starting backward pass")  # 添加调试输出
    #     bbox_loss.backward()
    #     print(f"DEBUG: Completed backward pass")  # 添加调试输出
        
    #     # 梯度裁剪
    #     if self.max_grad_norm is not None:
    #         torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), self.max_grad_norm)
        
    #     # 更新参数# 更新参数
    #     print(f"DEBUG: Starting optimizer step")  # 添加调试输出
    #     if hasattr(self, 'bbox_optimizer') and self.bbox_optimizer is not None:
    #         self.bbox_optimizer.step()
    #         self.bbox_optimizer.zero_grad()
    #     else:
    #         # 如果bbox_optimizer不存在，回退到actor_optimizer（但这种情况不应该发生）
    #         print("Warning: bbox_optimizer not found, using actor_optimizer")
    #         self.actor_optimizer.step()
    #         self.actor_optimizer.zero_grad()
    #     print(f"DEBUG: Completed optimizer step")  # 添加调试输出

    #     result = {
    #         'bbox_regression_loss': bbox_loss.item(), 
    #         'step_type': 'SFT_BBOX'
    #     }
    #     print(f"DEBUG: Completed _update_sft_step_from_dataproto, result: {result}")  # 添加调试输出
    #     return result

    # def _convert_bbox_to_tensor(self, bbox_targets):
    #     """
    #     将bbox目标转换为tensor格式
    #     输入格式: array([[['146.5123967', '126.2222222', '210.9355372', '188.1637427']], ...], dtype=object)
    #     """
    #     batch_size = len(bbox_targets)
    #     bbox_tensor = torch.zeros((batch_size, 4), dtype=torch.float32)  # (batch_size, 4) -> [x1, y1, x2, y2]
        
    #     for i, bbox_item in enumerate(bbox_targets):
    #         try:
    #             if isinstance(bbox_item, (list, tuple)) and len(bbox_item) > 0:
    #                 # 处理格式如 [['146.5123967', '126.2222222', '210.9355372', '188.1637427']]
    #                 if isinstance(bbox_item[0], (list, tuple)) and len(bbox_item[0]) == 4:
    #                     bbox_coords = [float(coord) for coord in bbox_item[0]]
    #                 elif len(bbox_item) == 4:
    #                     # 直接是 [x1, y1, x2, y2] 格式
    #                     bbox_coords = [float(coord) for coord in bbox_item]
    #                 else:
    #                     # 其他情况，使用默认值
    #                     bbox_coords = [0.0, 0.0, 0.0, 0.0]
    #             elif hasattr(bbox_item, '__len__') and len(bbox_item) > 0:
    #                 # numpy数组或其他可迭代对象
    #                 if hasattr(bbox_item[0], '__len__') and len(bbox_item[0]) == 4:
    #                     bbox_coords = [float(coord) for coord in bbox_item[0]]
    #                 else:
    #                     bbox_coords = [0.0, 0.0, 0.0, 0.0]
    #             else:
    #                 # 默认值
    #                 bbox_coords = [0.0, 0.0, 0.0, 0.0]
                
    #             bbox_tensor[i] = torch.tensor(bbox_coords, dtype=torch.float32)
    #         except Exception as e:
    #             print(f"Error converting bbox item {i}: {e}, using default values")
    #             bbox_tensor[i] = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        
    #     return bbox_tensor

    # def _compute_bbox_regression_loss(self, logits, bbox_targets, attention_mask):
    #     """
    #     计算bbox坐标回归损失，让模型回归x1, y1, x2, y2坐标值
    #     """
    #     batch_size, seq_len, vocab_size = logits.shape
    #     device = logits.device
        
    #     # 使用attention mask找到实际的非padding token位置
    #     # 找到最后一个有效的token位置，而不是简单的最后一个位置
    #     valid_lengths = attention_mask.sum(dim=1)  # (batch_size,) 实际序列长度
    #     last_valid_indices = valid_lengths - 1  # 最后一个有效token的索引
        
    #     # 提取最后一个有效token的logits
    #     batch_indices = torch.arange(batch_size, device=device)
    #     last_token_logits = logits[batch_indices, last_valid_indices, :]  # (batch_size, vocab_size)
        
    #     # 创建bbox预测头（如果没有的话）
    #     if not hasattr(self, 'bbox_predictor'):
    #         # 创建一个更合适的MLP来预测bbox坐标
    #         self.bbox_predictor = nn.Sequential(
    #             nn.Linear(vocab_size, 1024),  # 增加宽度
    #             nn.ReLU(),
    #             nn.Dropout(0.1),
    #             nn.Linear(1024, 512),
    #             nn.ReLU(),
    #             nn.Dropout(0.1),
    #             nn.Linear(512, 256),
    #             nn.ReLU(),
    #             nn.Linear(256, 128),
    #             nn.ReLU(),
    #             nn.Linear(128, 4)  # 输出4个坐标值 [x1, y1, x2, y2]
    #         ).to(device).to(logits.dtype)
        
    #         # 为bbox预测头创建独立的Adam优化器
    #         self.bbox_optimizer = torch.optim.Adam(
    #             self.bbox_predictor.parameters(), 
    #             lr=1e-4,  # 可以根据需要调整学习率
    #             weight_decay=1e-4
    #         )
    #     max_coord_value = 256.0  # 假设的最大坐标值
    #     # 使用最后token的logits预测bbox坐标
    #     raw_predictions = self.bbox_predictor(last_token_logits)  # (batch_size, 4)
    #     predicted_bbox = self.decode_bbox_predictions(raw_predictions, use_anchors=False)  # (batch_size, 4)
    #     # 确保bbox_targets与predicted_bbox具有相同的数据类型
    #     bbox_targets = bbox_targets.to(predicted_bbox.dtype)
        
    #     normalized_bbox_targets = bbox_targets / max_coord_value
    #     # 计算MSE损失
    #     print("predicted_bbox: ", predicted_bbox, "normalized_bbox_targets: ", normalized_bbox_targets)
    #     mse_loss = F.mse_loss(predicted_bbox, normalized_bbox_targets)
        
    #     # 也可以加入Smooth L1损失，对异常值更鲁棒
    #     smooth_l1_loss = F.smooth_l1_loss(predicted_bbox, normalized_bbox_targets)
        
    #     # 组合损失
    #     total_loss = 0.8 * mse_loss + 0.2 * smooth_l1_loss  # 可以根据需要调整权重

    #     # 计算IoU损失以辅助训练
    #     predicted_bbox_256 = predicted_bbox * max_coord_value  # 还原到原始尺度
    #     iou_loss = self._compute_iou_loss(predicted_bbox_256, bbox_targets)
    #     total_loss = total_loss + 0.5 * iou_loss  # 添加IoU损失项

    #     # 将损失放大回原始尺度便于观察
    #     scaled_loss = total_loss * (max_coord_value ** 2)
        
        
    #     # 打印损失值以便监控训练过程
    #     print(f"@@@ BBOX Regression Loss - MSE: {mse_loss.item():.6f}, SmoothL1: {smooth_l1_loss.item():.6f}, Total: {total_loss.item():.6f}")
    #     print(f"Predicted bbox (raw): {predicted_bbox_256.detach().cpu().to(torch.float32).numpy()}, Target bbox: {bbox_targets.detach().cpu().to(torch.float32).numpy()}")
        
    #     return scaled_loss  # 返回接近原始尺度的损失值，便于观察训练进展

    # def decode_bbox_predictions(self, raw_predictions, use_anchors=False):
    #     """
    #     将原始网络输出解码为归一化的边界框坐标
        
    #     Args:
    #         raw_predictions: [batch, 4] - 网络原始输出 [tx, ty, tw, th]
    #         image_shape: (H, W) - 图像尺寸
    #         use_anchors: 是否使用锚点
    #     """
    #     if not use_anchors:
    #         # 简化版本：假设整个图像是一个网格
    #         grid_size = 1  # 单网格
            
    #         # 中心点偏移 [0,1]
    #         cx_offset = torch.sigmoid(raw_predictions[:, 0])
    #         cy_offset = torch.sigmoid(raw_predictions[:, 1])
            
    #         # 宽高缩放（正数）
    #         bw = torch.exp(raw_predictions[:, 2])
    #         bh = torch.exp(raw_predictions[:, 3])
            
    #         # 由于是单网格，中心点就是偏移量本身
    #         cx = cx_offset
    #         cy = cy_offset
            
    #         # 确保宽高不会太大（可选）
    #         bw = torch.clamp(bw, max=2.0)  # 最大2倍图像宽
    #         bh = torch.clamp(bh, max=2.0)  # 最大2倍图像高
            
    #         # 转换为 [x1, y1, x2, y2] 格式（如果需要）
    #         x1 = cx - bw / 2
    #         y1 = cy - bh / 2  
    #         x2 = cx + bw / 2
    #         y2 = cy + bh / 2
            
    #         # 确保在[0,1]范围内
    #         predicted_bbox = torch.stack([
    #             torch.clamp(x1, 0, 1),
    #             torch.clamp(y1, 0, 1), 
    #             torch.clamp(x2, 0, 1),
    #             torch.clamp(y2, 0, 1)
    #         ], dim=1)
            
    #     else:
    #         # 使用锚点的完整YOLO实现（需要定义锚点）
    #         pass
            
    #     return predicted_bbox

    # def _compute_iou_loss(self, pred_bbox, target_bbox):
        """
        计算IoU损失，用于辅助边界框回归
        """
        # 确保bbox格式为 [x1, y1, x2, y2]
        pred_x1, pred_y1, pred_x2, pred_y2 = pred_bbox[:, 0], pred_bbox[:, 1], pred_bbox[:, 2], pred_bbox[:, 3]
        target_x1, target_y1, target_x2, target_y2 = target_bbox[:, 0], target_bbox[:, 1], target_bbox[:, 2], target_bbox[:, 3]

        # 计算交集
        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)

        inter_width = torch.clamp(inter_x2 - inter_x1, min=0)
        inter_height = torch.clamp(inter_y2 - inter_y1, min=0)
        intersection = inter_width * inter_height

        # 计算并集
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
        union = pred_area + target_area - intersection

        # 计算IoU
        iou = intersection / (union + 1e-8)
        
        # IoU损失 = 1 - IoU
        iou_loss = 1 - iou.mean()
        
        return iou_loss