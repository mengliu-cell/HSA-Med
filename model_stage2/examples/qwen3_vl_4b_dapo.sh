#!/bin/bash

set -x

# 模型路径（已更新为你的实际路径）
# MODEL_PATH="/home/ubuntu/pretrained_model/qwen3_vl_4b"
MODEL_PATH="/pretrained_model/qwen3_vl_4b" # docker容器路径

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=hiyouga/geometry3k@train \
    data.val_files=hiyouga/geometry3k@test \
    data.format_prompt=./examples/format_prompt/math.jinja \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.rollout_batch_size=256 \
    data.mini_rollout_batch_size=128 \
    worker.actor.ulysses_size=4 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.model.lora.rank=64 \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.actor.optim.weight_decay=0.1 \
    worker.actor.optim.lr=5e-6 \
    worker.actor.optim.lr_warmup_steps=10 \
    worker.actor.global_batch_size=32 \
    worker.actor.clip_ratio_low=0.2 \
    worker.actor.clip_ratio_high=0.28 \
    worker.actor.clip_ratio_dual=10.0 \
    worker.rollout.n=16 \
    worker.rollout.temperature=1.0 \
    worker.rollout.top_p=0.7 \
    worker.rollout.max_num_batched_tokens=16384 \
    worker.rollout.val_override_config='{"n":1,"temperature":0.7,"top_p":0.95}' \
    worker.rollout.gpu_memory_utilization=0.7 \
    worker.rollout.tensor_parallel_size=2 \
    worker.reward.reward_function=./examples/reward_function/math.py:compute_score \
    algorithm.disable_kl=True \
    algorithm.online_filtering=True \
    algorithm.filter_key=accuracy \
    algorithm.filter_low=0.01 \
    algorithm.filter_high=0.99 \
    trainer.total_epochs=10 \
    trainer.max_try_make_batch=10 \
    trainer.experiment_name=qwen3_vl_4b_geo3k_dapo \
    trainer.n_gpus_per_node=4 \
    trainer.logger='["file"]'