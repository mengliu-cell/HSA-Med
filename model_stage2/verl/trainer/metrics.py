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

from typing import Any

import numpy as np
import torch

from ..protocol import DataProto

# 原代码
def reduce_metrics(metrics: dict[str, list[Any]]) -> dict[str, Any]:
    return {key: np.mean(value) for key, value in metrics.items()}

# def reduce_metrics(metrics: dict[str, list[Any]]) -> dict[str, Any]:
#     """
#     Reduce metrics by computing the mean for each metric across all entries.
#     Only compute mean for numeric values, skip non-numeric values.
#     """
#     reduced = {}
#     for key, value in metrics.items():
#         if not value:  # Empty list
#             reduced[key] = 0.0
#             continue
            
#         # Check if the values are numeric before applying mean
#         first_val = value[0]
#         if isinstance(first_val, (int, float)) or (hasattr(first_val, 'dtype') and np.issubdtype(first_val.dtype, np.number)):
#             reduced[key] = float(np.mean(value))
#         elif isinstance(first_val, (list, np.ndarray)) and len(value) > 0:
#             # Handle array-like values
#             try:
#                 # Try to convert first element if it's numeric
#                 flat_first = np.asarray(first_val).flat[0] if hasattr(first_val, 'flat') else first_val[0] if isinstance(first_val, list) else first_val
#                 if isinstance(flat_first, (int, float, np.number)):
#                     reduced[key] = float(np.mean(value))
#                 else:
#                     # Skip non-numeric array-like values
#                     continue
#             except (TypeError, IndexError):
#                 # Skip values that can't be processed
#                 continue
#         else:
#             # Skip non-numeric values (strings, objects, etc.)
#             continue
    
#     return reduced

def compute_length_metrics(batch: DataProto) -> dict[str, Any]:
    max_response_length = batch.batch["responses"].size(-1)
    max_prompt_length = batch.batch["attention_mask"].size(-1) - max_response_length

    prompt_length = batch.batch["attention_mask"][:, :-max_response_length].sum(-1).float()
    response_length = batch.batch["attention_mask"][:, -max_response_length:].sum(-1).float()

    return {
        # response length
        "response_length/mean": torch.mean(response_length).detach().item(),
        "response_length/max": torch.max(response_length).detach().item(),
        "response_length/min": torch.min(response_length).detach().item(),
        "response_length/clip_ratio": torch.eq(response_length, max_response_length).float().mean().detach().item(),
        # prompt length
        "prompt_length/mean": torch.mean(prompt_length).detach().item(),
        "prompt_length/max": torch.max(prompt_length).detach().item(),
        "prompt_length/min": torch.min(prompt_length).detach().item(),
        "prompt_length/clip_ratio": torch.eq(prompt_length, max_prompt_length).float().mean().detach().item(),
    }


def compute_data_metrics(batch: DataProto, use_critic: bool = False) -> dict[str, Any]:
    sequence_score = batch.batch["token_level_scores"].sum(-1)
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)

    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]

    max_response_length = batch.batch["responses"].size(-1)
    response_mask = batch.batch["attention_mask"][:, -max_response_length:].bool()

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch["values"]
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    return {
        # score
        "critic/score/mean": torch.mean(sequence_score).detach().item(),
        "critic/score/max": torch.max(sequence_score).detach().item(),
        "critic/score/min": torch.min(sequence_score).detach().item(),
        # reward
        "critic/rewards/mean": torch.mean(sequence_reward).detach().item(),
        "critic/rewards/max": torch.max(sequence_reward).detach().item(),
        "critic/rewards/min": torch.min(sequence_reward).detach().item(),
        # adv
        "critic/advantages/mean": torch.mean(valid_adv).detach().item(),
        "critic/advantages/max": torch.max(valid_adv).detach().item(),
        "critic/advantages/min": torch.min(valid_adv).detach().item(),
        # returns
        "critic/returns/mean": torch.mean(valid_returns).detach().item(),
        "critic/returns/max": torch.max(valid_returns).detach().item(),
        "critic/returns/min": torch.min(valid_returns).detach().item(),
        **(
            {
                # values
                "critic/values/mean": torch.mean(valid_values).detach().item(),
                "critic/values/max": torch.max(valid_values).detach().item(),
                "critic/values/min": torch.min(valid_values).detach().item(),
                # vf explained var
                "critic/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
            }
            if use_critic
            else {}
        ),
        **compute_length_metrics(batch),
    }


def compute_timing_metrics(batch: DataProto, timing_raw: dict[str, float]) -> dict[str, Any]:
    # Check if response_mask exists in the batch, otherwise derive it from attention_mask
    if "response_mask" in batch.batch.keys():
        num_response_tokens = torch.sum(batch.batch["response_mask"]).item()
    else:
        # Derive response_mask from attention_mask if not present
        # Assuming the response portion is the latter part after prompt
        max_response_length = batch.batch["input_ids"].size(-1) - batch.batch["attention_mask"].sum(-1).min().item()
        if max_response_length > 0:
            response_mask = batch.batch["attention_mask"][:, -max_response_length:].bool()
            num_response_tokens = torch.sum(response_mask).item()
        else:
            # If there's no response part, use the whole attention_mask
            num_response_tokens = torch.sum(batch.batch["attention_mask"]).item()

     # Check if global_token_num exists in meta_info, otherwise calculate from batch
    if "global_token_num" in batch.meta_info:
        num_overall_tokens = sum(batch.meta_info["global_token_num"])
    else:
        # Calculate total tokens from batch dimensions as fallback
        # Use the total number of tokens in input_ids as approximation
        if "input_ids" in batch.batch.keys():
            num_overall_tokens = batch.batch["input_ids"].numel()
        else:
            # If no input_ids, use the batch size multiplied by a reasonable sequence length estimate
            batch_size = len(batch)
            seq_length = batch.batch[list(batch.batch.keys())[0]].size(-1) if len(batch.batch) > 0 else 1
            num_overall_tokens = batch_size * seq_length
    # num_response_tokens = torch.sum(batch.batch["response_mask"]).item()
    # num_overall_tokens = sum(batch.meta_info["global_token_num"])
    num_tokens_of_section = {
        **dict.fromkeys(["gen", "reward"], num_response_tokens),
        **dict.fromkeys(["ref", "old", "values", "adv", "update_critic", "update_actor"], num_overall_tokens),
    }
    return {
        **{f"timing_s/{name}": value for name, value in timing_raw.items()},
        **{
            f"timing_per_token_ms/{name}": timing_raw[name] * 1000 / num_tokens_of_section[name]
            for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys())
        },
    }


def compute_throughout_metrics(batch: DataProto, timing_raw: dict[str, float], num_gpus: int) -> dict[str, Any]:
    # total_num_tokens = sum(batch.meta_info["global_token_num"])
    # Check if global_token_num exists in meta_info, otherwise calculate from batch
    if "global_token_num" in batch.meta_info:
        total_num_tokens = sum(batch.meta_info["global_token_num"])
    else:
        # Calculate total tokens from batch dimensions as fallback
        # Use the total number of tokens in input_ids as approximation
        if "input_ids" in batch.batch.keys():
            total_num_tokens = batch.batch["input_ids"].numel()
        else:
            # If no input_ids, use the batch size multiplied by a reasonable sequence length estimate
            batch_size = len(batch) if len(batch) > 0 else 1
            seq_length = batch.batch[list(batch.batch.keys())[0]].size(-1) if len(batch.batch) > 0 else 1
            total_num_tokens = batch_size * seq_length

    # time = timing_raw["step"]
    # Check if step exists in timing_raw, otherwise use default value
    if "step" in timing_raw:
        time = timing_raw["step"]
    else:
        # Use a default time value when step is not provided
        time = 1.0  # 1 second as default for throughput calculation
    return {
        "perf/total_num_tokens": total_num_tokens,
        "perf/time_per_step": time,
        "perf/throughput": total_num_tokens / (time * num_gpus),
    }
