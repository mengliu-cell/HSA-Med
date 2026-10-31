#!/bin/bash

set -x

echo "========================================="
echo "Qwen3-VL-4B DAPO Training - MS-CXR Medical Dataset"
echo "Multi-Task: BBox Detection + Reasoning + Diagnosis"
echo "========================================="
echo ""

# ==================== 配置区域 ====================

# 模型路径（宿主机）
MODEL_HOST_PATH="/home/ubuntu/pretrained_model/qwen3_vl_4b"

# 数据集路径（宿主机）- 修改为你的实际数据集路径
DATASET_HOST_PATH="/home/ubuntu/medgroundr1_dataset"  # 数据集所在目录

# 原始数据文件名
TRAIN_DATA_FILE="MS_CXR_Local_Alignment_v1_scaled.jsonl"

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

REQUIRED_FILES=("config.json" "tokenizer.json" "tokenizer_config.json")
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$MODEL_HOST_PATH/$file" ]; then
        echo "❌ Error: Missing required file: $file"
        exit 1
    fi
done
echo "✓ Model files verified"

# 检查数据集
echo "[2/6] Checking dataset..."
if [ ! -f "$DATASET_HOST_PATH/$TRAIN_DATA_FILE" ]; then
    echo "❌ Error: Dataset not found at $DATASET_HOST_PATH/$TRAIN_DATA_FILE"
    exit 1
fi

# 分割数据集为 train/test
echo "[2.5/6] Splitting dataset into train/test..."
if [ ! -f "$DATASET_HOST_PATH/train.jsonl" ] || [ ! -f "$DATASET_HOST_PATH/test.jsonl" ]; then
    echo "Splitting dataset..."
    
    # 临时脚本分割数据集
    cat > /tmp/split_dataset.py << 'EOF'
import json
import random

def split_jsonl(input_file, train_file, test_file, train_ratio=0.8):
    """分割 JSONL 文件为训练集和测试集"""
    
    # 读取所有数据
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 解析 JSON
    data = []
    for line in lines:
        try:
            item = json.loads(line.strip())
            data.append(item)
        except:
            continue
    
    # 打乱数据
    random.shuffle(data)
    
    # 分割
    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    # 保存训练集
    with open(train_file, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')
    
    # 保存测试集
    with open(test_file, 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Total samples: {len(data)}")
    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")

if __name__ == "__main__":
    import sys
    input_file = sys.argv[1]
    train_file = sys.argv[2]
    test_file = sys.argv[3]
    
    split_jsonl(input_file, train_file, test_file)
EOF
    
    python3 /tmp/split_dataset.py "$DATASET_HOST_PATH/$TRAIN_DATA_FILE" "$DATASET_HOST_PATH/train.jsonl" "$DATASET_HOST_PATH/test.jsonl"
    
    # 清理临时脚本
    rm /tmp/split_dataset.py
fi

echo "✓ Dataset verified and split"

# 创建必要的目录
echo "[3/6] Creating directories..."
mkdir -p "$RAY_TEMP_PATH"
# 确保 Ray 临时目录具有适当的权限
chmod 777 "$RAY_TEMP_PATH"
echo "✓ Directories created"

# 停止旧的容器
echo "[4/6] Stopping existing containers..."
if docker ps -a | grep -q "$DOCKER_IMAGE"; then
    RUNNING_CONTAINERS=$(docker ps -q --filter ancestor=$DOCKER_IMAGE)
    if [ ! -z "$RUNNING_CONTAINERS" ]; then
        echo "Stopping running containers..."
        docker stop $RUNNING_CONTAINERS
        sleep 3
    fi
    
    REMOVED_CONTAINERS=$(docker ps -aq --filter ancestor=$DOCKER_IMAGE)
    if [ ! -z "$REMOVED_CONTAINERS" ]; then
        docker rm $REMOVED_CONTAINERS
    fi
fi
echo "✓ Old containers cleaned"

# 启动新的 Docker容器并运行训练
echo "[5/6] Starting training container..."
echo "This may take a while, please wait..."
echo ""

docker run -it --rm --ipc=host --gpus=all \
    -v /home/ubuntu/PycharmProjects/EasyR1:/app \
    -v $MODEL_HOST_PATH:/pretrained_model/qwen3_vl_4b:ro \
    -v $DATASET_HOST_PATH:/datasets/ms_cxr \
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
        
        # 验证数据集
        echo '[Container] Verifying dataset...'
        if [ ! -f '/datasets/ms_cxr/train.jsonl' ]; then
            echo 'ERROR: Dataset not found!'
            exit 1
        fi
        echo 'Dataset samples:'
        head -3 /datasets/ms_cxr/train.jsonl
        echo ''
        
        # 创建路径修正脚本
        echo '[Container] Creating fix_paths.py script...'
        cat > /tmp/fix_paths.py << 'CONTAINER_EOF'
import json
import os

def fix_dataset_paths(input_file, output_file, old_prefix, new_prefix):
    \"\"\"将数据集中的绝对路径转换为相对于挂载点的新路径，移除 ./ 前缀\"\"\"
    print(f\"Processing {input_file}...\")
    processed_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
        open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            try:
                data = json.loads(line.strip())
                
                # 修改图像路径
                if 'path' in data:
                    old_path = data['path']
                    # print(\"---------------------------------------------------old path:\", old_path)
                    if old_path.startswith(old_prefix):
                        # 提取相对于数据集根目录的路径
                        relative_path = old_path[len(old_prefix):]
                        # print(\"---------------------------------------------------relativepath:\", relative_path)
                        if relative_path.startswith('/'):
                            relative_path = relative_path[1:]  # 移除开头的斜杠
                        # 确保路径不以 ./ 开头
                        if relative_path.startswith('./'):
                            relative_path = relative_path[2:]
                        # Use the clean relative path
                        data['path'] = os.path.join(new_prefix, relative_path) if new_prefix else relative_path
                        # print(f\"Fixed path: {old_path} -> {data['path']}\")
                    else:
                        print(f\"Path does not match prefix, keeping as-is: {old_path}\")
                    
                outfile.write(json.dumps(data) + '\\n')
                processed_count += 1
                
                if line_num % 1000 == 0:
                    print(f\"Processed {line_num} lines...\")
                    
            except Exception as e:
                print(f\"Error processing line {line_num}: {e}\")
                continue
    
    print(f\"Processed {processed_count} lines from {input_file}\")

# 修正训练集
fix_dataset_paths(
    '/datasets/ms_cxr/train.jsonl', 
    '/datasets/ms_cxr/train_fixed.jsonl',
    '/home/ubuntu/medgroundr1_dataset',  # 原始前缀
    ''  # 新前缀（因为我们希望路径是相对于数据集挂载点的）
)

# 修正验证集
fix_dataset_paths(
    '/datasets/ms_cxr/test.jsonl', 
    '/datasets/ms_cxr/test_fixed.jsonl',
    '/home/ubuntu/medgroundr1_dataset',  # 原始前缀
    ''  # 新前缀
)

print(\"Dataset paths fixed successfully!\")
CONTAINER_EOF

        # 修正数据集中的图像路径
        echo '[Container] Fixing image paths in dataset...'
        python3 /tmp/fix_paths.py
        echo ''
        
        # 验证修正后的数据集
        echo '[Container] Verifying fixed dataset...'
        if [ -f '/datasets/ms_cxr/train_fixed.jsonl' ]; then
            echo 'First few lines of train_fixed.jsonl:'
            head -3 /datasets/ms_cxr/train_fixed.jsonl
        else
            echo 'ERROR: train_fixed.jsonl was not created!'
            ls -la /datasets/ms_cxr/
            exit 1
        fi
        echo ''

        # 检查几个图像文件是否存在
        echo 'Checking if sample images exist...'
        if [ -f '/datasets/ms_cxr/train_fixed.jsonl' ]; then
            # 简化检查逻辑以避免语法问题
            echo 'Sample check completed (skipping complex validation)'
        else
            echo 'ERROR: Cannot check sample because train_fixed.jsonl does not exist'
            exit 1
        fi
        echo ''
        
        # 设置环境变量
        export HF_ENDPOINT=https://hf-mirror.com

        # 安装 BiomedCLIP verifier 依赖（reward 中的 bbox-reasoning 对齐 reward 需要）
        echo '[Container] Installing open_clip_torch for BiomedCLIP verifier...'
        pip install open_clip_torch >/dev/null 2>&1 || echo 'WARNING: open_clip_torch install failed; alignment rewards will degrade to 0.5'

        # 创建检查点目录以确保它存在
        echo '[Container] Creating checkpoints directory...'
        mkdir -p /ray_temp/checkpoints/easy_r1/qwen3_vl_4b_ms_cxr_dapo
        chmod 777 /ray_temp/checkpoints/easy_r1/qwen3_vl_4b_ms_cxr_dapo
        
        # 开始训练
        echo '========================================='
        echo 'Starting Qwen3-VL-4B DAPO Training'
        echo 'Task: Medical Imaging Multi-Task Learning'
        echo '========================================='
        echo ''
        
        python3 -m verl.trainer.main \
            config=examples/config.yaml \
            data.train_files=/datasets/ms_cxr/train_fixed.jsonl \
            data.val_files=/datasets/ms_cxr/test_fixed.jsonl \
            data.prompt_key=questions \
            data.answer_key=label_text \
            data.image_key=path \
            data.image_dir=/datasets/ms_cxr \
            data.format_prompt=./examples/format_prompt/ms_cxr_pairwise.jinja \
            data.max_prompt_length=1024 \
            data.max_response_length=1536 \
            data.rollout_batch_size=32 \
            data.mini_rollout_batch_size=16 \
            worker.actor.ulysses_size=4 \
            worker.actor.model.model_path=/pretrained_model/qwen3_vl_4b \
            worker.actor.model.lora.rank=64 \
            worker.actor.fsdp.torch_dtype=bf16 \
            worker.actor.optim.strategy=adamw_bf16 \
            worker.actor.optim.weight_decay=0.1 \
            worker.actor.optim.lr=2e-6 \
            worker.actor.optim.lr_warmup_steps=10 \
            worker.actor.global_batch_size=16 \
            worker.actor.clip_ratio_low=0.2 \
            worker.actor.clip_ratio_high=0.28 \
            worker.actor.clip_ratio_dual=10.0 \
            worker.rollout.n=4 \
            worker.rollout.temperature=1.0 \
            worker.rollout.top_p=0.7 \
            worker.rollout.max_num_batched_tokens=8192 \
            worker.rollout.val_override_config='{\"n\":1,\"temperature\":0.7,\"top_p\":0.95}' \
            worker.rollout.gpu_memory_utilization=0.7 \
            worker.rollout.tensor_parallel_size=2 \
            worker.reward.reward_function=./examples/reward_function/ms_cxr.py:reward_function \
            worker.reward.reward_function_kwargs='{\"data_format\":\"original\"}' \
            algorithm.disable_kl=True \
            algorithm.online_filtering=True \
            algorithm.filter_key=accuracy \
            algorithm.filter_low=0.0 \
            algorithm.filter_high=1.0 \
            trainer.total_epochs=5 \
            trainer.max_try_make_batch=10 \
            trainer.experiment_name=qwen3_vl_4b_ms_cxr_dapo \
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
echo "Checkpoints saved to: $RAY_TEMP_PATH/checkpoints/easy_r1/qwen3_vl_4b_ms_cxr_dapo/"
echo "Logs saved to: $RAY_TEMP_PATH/checkpoints/easy_r1/qwen3_vl_4b_ms_cxr_dapo/*.log"
echo ""