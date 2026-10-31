#!/bin/bash

set -x

# 创建必要的目录
mkdir -p /home/ubuntu/datasets
mkdir -p /home/ubuntu/ray_temp

# 检查是否有运行中的容器
if docker ps | grep -q hiyouga/verl; then
    echo "Container already running, connecting..."
    docker exec -it $(docker ps | grep hiyouga/verl | awk '{print $1}') bash
else
    echo "Starting new training container..."
    docker run -it --ipc=host --gpus=all \
        -v /home/ubuntu/PycharmProjects/EasyR1:/app \
        -v /home/ubuntu/下载/qwen3_vl_4b:/models/qwen3_vl_4b \
        -v /home/ubuntu/datasets:/datasets \
        -v /home/ubuntu/ray_temp:/ray_temp \
        -e HF_ENDPOINT=https://hf-mirror.com \
        -e RAY_ROOT_DIR=/ray_temp \
        -w /app \
        hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0 \
        bash -c "
            cd /app &&
            python3 -m verl.trainer.main \
                config=examples/config.yaml \
                data.train_files=hiyouga/geometry3k@train \
                data.val_files=hiyouga/geometry3k@test \
                worker.actor.model.model_path=/models/qwen3_vl_4b \
                worker.actor.model.lora.rank=64 \
                worker.actor.optim.lr=1e-5 \
                trainer.experiment_name=qwen3_vl_4b_geo_grpo_lora \
                trainer.n_gpus_per_node=2 \
                trainer.logger='[\"file\"]'
        "
fi
