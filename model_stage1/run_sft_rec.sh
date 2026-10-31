#!/bin/bash
# Supervised Fine-Tuning for medical grounding (llamafactory data format)
# Stage I: frozen Qwen2.5-VL + trainable APA/DSE + LoRA
# Objective: L_gnd + lambda_apa * L_APA; bbox annotations are original pixels.
# Run from: open-r1-multimodal/

export CUDA_VISIBLE_DEVICES=1,2,3
export DEBUG_MODE="true"

RUN_NAME="Qwen2.5-VL-3B-StageI-Liver"
export LOG_PATH="./debug_log_${RUN_NAME}.txt"

torchrun \
    --nproc_per_node=3 \
    src/open_r1/sft_rec.py \
    --deepspeed local_scripts/zero3.json \
    --output_dir output/${RUN_NAME} \
    --model_name_or_path /home/ubuntu/pretrained_model/qwen25_vl_3b \
    --dataset_name data_config/sft_rec_liver.yaml \
    --use_adsa true \
    --lambda_apa 0.5 \
    --bbox_coordinate_format pixel \
    --use_peft \
    --lora_r 16 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type cosine \
    --torch_dtype bfloat16 \
    --attn_implementation flash_attention_2 \
    --gradient_checkpointing true \
    --max_pixels 12845056 \
    --min_pixels 3136 \
    --report_to wandb \
    --run_name ${RUN_NAME} \
    --logging_steps 5 \
    --save_steps 100 \
    --save_only_model false \
    --dataloader_num_workers 4

# To evaluate a held-out liver dataset after training, add:
# --eval_data_path /path/to/liver_test.json --eval_batch_size 1 --eval_max_new_tokens 256
