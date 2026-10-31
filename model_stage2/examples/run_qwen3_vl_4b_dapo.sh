#!/bin/bash

set -x

echo "========================================="
echo "Qwen3-VL-4B DAPO Training - One Click Start"
echo "========================================="
echo ""

# ==================== 配置区域 ====================

# 模型路径（宿主机）
MODEL_HOST_PATH="/home/ubuntu/pretrained_model/qwen3_vl_4b"

# 数据集保存路径
DATASET_HOST_PATH="/home/ubuntu/datasets"

# Ray 临时文件路径
RAY_TEMP_PATH="/home/ubuntu/ray_temp"

# Docker 镜像名称
DOCKER_IMAGE="hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0"

# ================================================

# 检查模型是否存在
echo "[1/6] Checking model files..."
if [ ! -d "$MODEL_HOST_PATH" ]; then
    echo "❌ Error: Model not found at $MODEL_HOST_PATH"
    exit 1
fi

# 检查关键文件
REQUIRED_FILES=("config.json" "tokenizer.json" "tokenizer_config.json")
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$MODEL_HOST_PATH/$file" ]; then
        echo "❌ Error: Missing required file: $file"
        exit 1
    fi
done
echo "✓ Model files verified"

# 创建必要的目录
echo "[2/6] Creating directories..."
mkdir -p "$DATASET_HOST_PATH"
mkdir -p "$RAY_TEMP_PATH"
echo "✓ Directories created"

# 停止旧的容器
echo "[3/6] Stopping existing containers..."
if docker ps -a | grep -q "$DOCKER_IMAGE"; then
    RUNNING_CONTAINERS=$(docker ps -q --filter ancestor=$DOCKER_IMAGE)
    if [ ! -z "$RUNNING_CONTAINERS" ]; then
        echo "Stopping running containers..."
        docker stop $RUNNING_CONTAINERS
        sleep 3
    fi
    
    REMOVED_CONTAINERS=$(docker ps -aq --filter ancestor=$DOCKER_IMAGE)
    if [ ! -z "$REMOVED_CONTAINERS" ]; then
        echo "Removing old containers..."
        docker rm $REMOVED_CONTAINERS
    fi
fi
echo "✓ Old containers cleaned"

# 启动新的 Docker容器并运行训练
echo "[4/6] Starting training container..."
echo "This may take a while, please wait..."
echo ""

docker run -it --rm --ipc=host --gpus=all \
    -v /home/ubuntu/PycharmProjects/EasyR1:/app \
    -v $MODEL_HOST_PATH:/pretrained_model/qwen3_vl_4b:ro \
    -v $DATASET_HOST_PATH:/datasets \
    -v $RAY_TEMP_PATH:/ray_temp \
    -e HF_ENDPOINT=https://hf-mirror.com \
    -e RAY_ROOT_DIR=/ray_temp \
    -w /app \
    $DOCKER_IMAGE \
    bash -c "
        set -x
        echo ''
        echo '========================================='
        echo 'Container Started Successfully'
        echo '========================================='
        echo ''
        
        # 验证模型
        echo '[Container] Verifying model files...'
        if [ ! -d '/pretrained_model/qwen3_vl_4b' ]; then
            echo 'ERROR: Model not mounted correctly!'
            exit 1
        fi
        
        echo 'Model files:'
        ls -lh /pretrained_model/qwen3_vl_4b/ | head -15
        echo ''
        
        # 设置环境变量
        export HF_ENDPOINT=https://hf-mirror.com
        
        # 开始训练
        echo '========================================='
        echo 'Starting Qwen3-VL-4B DAPO Training'
        echo '========================================='
        echo ''
        
        python3 -m verl.trainer.main \
            config=examples/config.yaml \

            # 修改数据集路径
            data.train_files=hiyouga/geometry3k@train \
            data.val_files=hiyouga/geometry3k@test \
            # 修改数据键名（根据你的数据集格式）
            # data.prompt_key=text      # 医学报告的文本描述或问题
            # data.answer_key=label     # 诊断结果或答案
            # data.image_key=image_path # 影像图片路径

            # data.format_prompt=./examples/format_prompt/math.jinja \
            data.format_prompt=./examples/format_prompt/ms_cxr.jinja \ # prompt格式模板

            # data.max_prompt_length=2048 \
            # data.max_response_length=2048 \
            data.max_prompt_length=1024 \
            data.max_response_length=512 \

            data.rollout_batch_size=256 \
            data.mini_rollout_batch_size=128 \
            worker.actor.ulysses_size=4 \
            worker.actor.model.model_path=/pretrained_model/qwen3_vl_4b \
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
            worker.rollout.val_override_config='{\"n\":1,\"temperature\":0.7,\"top_p\":0.95}' \
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
            trainer.logger='[\"file\"]'
    "

# 获取退出状态
EXIT_CODE=$?

echo ""
echo "========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Training completed successfully!"
else
    echo "⚠ Training exited with code: $EXIT_CODE"
fi
echo "========================================="
echo ""
echo "Checkpoints saved to: $RAY_TEMP_PATH/checkpoints/easy_r1/qwen3_vl_4b_geo3k_dapo/"
echo "Logs saved to: $RAY_TEMP_PATH/checkpoints/easy_r1/qwen3_vl_4b_geo3k_dapo/*.log"
echo ""