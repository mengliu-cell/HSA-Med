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

import json

import ray
from omegaconf import OmegaConf

from ..single_controller.ray import RayWorkerGroup
from ..utils.tokenizer import get_processor, get_tokenizer
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import AutoRewardManager
from .config import PPOConfig
from .data_loader import create_dataloader
from .ray_trainer import RayPPOTrainer, ResourcePoolManager, Role


# please make sure main_task is not scheduled on head
@ray.remote(num_cpus=1) 
class Runner:
    """A runner for RL training."""

    def run(self, config: PPOConfig):
        # print config
        print(json.dumps(config.to_dict(), indent=2))  # total epochs=5, max_steps=5

        # instantiate tokenizer # vlidation里面的reward是需要进行的decoder的
        tokenizer = get_tokenizer(
            config.worker.actor.model.model_path,
            override_chat_template=config.data.override_chat_template,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )
        processor = get_processor(
            config.worker.actor.model.model_path,
            override_chat_template=config.data.override_chat_template,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )

        # define worker classes
        ray_worker_group_cls = RayWorkerGroup
        role_worker_mapping = {
            Role.ActorRolloutRef: ray.remote(FSDPWorker),
            Role.Critic: ray.remote(FSDPWorker),  # 核查是不是存在critic
        }
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRolloutRef: global_pool_id,
            Role.Critic: global_pool_id,
        }
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
        # 以下三行是原代码
        RemoteRewardManager = ray.remote(AutoRewardManager).options(num_cpus=config.worker.reward.num_cpus)
        reward_fn = RemoteRewardManager.remote(config.worker.reward, tokenizer)
        val_reward_fn = RemoteRewardManager.remote(config.worker.reward, tokenizer)
        # 以下两行是我为了debug方便直接实例化reward函数，进去看reward的调用的代码
        # reward_fn = AutoRewardManager(config.worker.reward, tokenizer)  # 直接实例化
        # val_reward_fn = AutoRewardManager(config.worker.reward, tokenizer)  # 直接实例化


        train_dataloader, val_dataloader = create_dataloader(config.data, tokenizer, processor)

        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )
        trainer.init_workers()
        trainer.fit()
        # trainer._validate()  # 最后再单独测试一次，输出最终的验证结果


def main():
    cli_args = OmegaConf.from_cli()  # 解析命令行参数
    default_config = OmegaConf.structured(PPOConfig())

    if hasattr(cli_args, "config"):
        config_path = cli_args.pop("config", None)
        file_config = OmegaConf.load(config_path)
        default_config = OmegaConf.merge(default_config, file_config)

    ppo_config = OmegaConf.merge(default_config, cli_args)
    ppo_config: PPOConfig = OmegaConf.to_object(ppo_config)
    ppo_config.deep_post_init()

    if not ray.is_initialized():
        runtime_env = {
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "WARN",
                "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                "VLLM_ALLREDUCE_USE_SYMM_MEM": "0",
            }
        }
        ray.init(runtime_env=runtime_env, num_gpus=1)
        

    runner = Runner.remote()  # 实例化一个runner
    ray.get(runner.run.remote(ppo_config)) # 启动强化训练流程
    # 以下两行是我为了debug方便直接实例化reward函数，进去看reward的调用的代码
    # runner = Runner()  # 实例化一个local runner
    # runner.run(ppo_config)

    if ppo_config.trainer.ray_timeline is not None:
        # use `export RAY_PROFILING=1` to record the ray timeline
        ray.timeline(filename=ppo_config.trainer.ray_timeline)


if __name__ == "__main__":
    main()





# 进入Dev container后的运行命令行：
# data.train_files=/datasets/ms_cxr/train_fixed.jsonl \                        # ms_cxr数据集
# data.val_files=/datasets/ms_cxr/test_fixed.jsonl \                           # ms_cxr数据集
# data.train_files=/datasets/ms_cxr/chestxray8_output_train_fixed.jsonl \      # chestxray8数据集
# data.val_files=/datasets/ms_cxr/chestxray8_output_test_fixed.jsonl \         # chestxray8数据集
'''
ms_cxr数据集：
    CUDA_VISIBLE_DEVICES=1 
    python -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=/datasets/ms_cxr/train_fixed_copy_with_consensus.jsonl \
    data.val_files=/datasets/ms_cxr/test_fixed_copy_with_consensus.jsonl \
    data.prompt_key=questions \
    data.answer_key=label_text \
    data.image_key=path \
    data.image_dir=/datasets/ms_cxr \
    data.format_prompt=./examples/format_prompt/ms_cxr_think_on_bbox5.jinja \
    data.max_prompt_length=1024 \
    data.max_response_length=1356 \
    data.min_pixels=65536 \
    worker.rollout.n=2 \
    data.rollout_batch_size=1 \
    data.mini_rollout_batch_size=1 \
    worker.rollout.max_num_batched_tokens=8192 \
    worker.rollout.gpu_memory_utilization=0.6 \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.temperature=1.0 \
    worker.rollout.top_p=0.7 \
    worker.rollout.val_override_config='{"n":1,"temperature":0.35,"top_p":0.95}' \
    trainer.n_gpus_per_node=1 \
    worker.actor.ulysses_size=1 \
    worker.actor.model.model_path=/pretrained_model/qwen25_vl_3b_sft \
    worker.actor.model.lora.rank=8 \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.actor.optim.weight_decay=0.1 \
    worker.actor.optim.lr=2e-6 \
    worker.actor.optim.lr_warmup_steps=10 \
    worker.actor.global_batch_size=1 \
    worker.actor.clip_ratio_low=0.2 \
    worker.actor.clip_ratio_high=0.28 \
    worker.actor.clip_ratio_dual=10.0 \
    worker.reward.reward_function=./examples/reward_function/ms_cxr.py:reward_function \
    worker.reward.reward_function_kwargs='{"data_format":"original"}' \
    algorithm.disable_kl=True \
    algorithm.online_filtering=True \
    algorithm.filter_key=overall \
    algorithm.filter_low=0 \
    algorithm.filter_high=1.0 \
    trainer.max_steps=500 \
    trainer.val_freq=2000 \
    trainer.save_freq=200 \
    trainer.max_try_make_batch=10 \
    trainer.experiment_name=qwen25_vl_3b_ms_cxr_dapo \
    trainer.val_before_train=False '''

'''
ms_cxr数据集--grid 32：
    CUDA_VISIBLE_DEVICES=1 
    python -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=/datasets/ms_cxr/train_fixed.jsonl \
    data.val_files=/datasets/ms_cxr/test_fixed.jsonl \
    data.prompt_key=questions \
    data.answer_key=label_text \
    data.image_key=path \
    data.image_dir=/datasets/ms_cxr \
    data.format_prompt=./examples/format_prompt/ms_cxr_think_on_bbox2.jinja \
    data.max_prompt_length=1024 \
    data.max_response_length=1356 \
    data.min_pixels=65536 \
    worker.rollout.n=2 \
    data.rollout_batch_size=1 \
    data.mini_rollout_batch_size=1 \
    worker.rollout.max_num_batched_tokens=8192 \
    worker.rollout.gpu_memory_utilization=0.6 \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.temperature=1.0 \
    worker.rollout.top_p=0.7 \
    worker.rollout.val_override_config='{"n":1,"temperature":0.35,"top_p":0.95}' \
    trainer.n_gpus_per_node=1 \
    worker.actor.ulysses_size=1 \
    worker.actor.model.model_path=/pretrained_model/qwen3_vl_2b_ins \
    worker.actor.model.lora.rank=32 \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.actor.optim.weight_decay=0.1 \
    worker.actor.optim.lr=2e-6 \
    worker.actor.optim.lr_warmup_steps=10 \
    worker.actor.global_batch_size=1 \
    worker.actor.clip_ratio_low=0.2 \
    worker.actor.clip_ratio_high=0.28 \
    worker.actor.clip_ratio_dual=10.0 \
    worker.reward.reward_function=./examples/reward_function/ms_cxr.py:reward_function \
    worker.reward.reward_function_kwargs='{"data_format":"original"}' \
    algorithm.disable_kl=True \
    algorithm.online_filtering=True \
    algorithm.filter_key=overall \
    algorithm.filter_low=0 \
    algorithm.filter_high=1.0 \
    trainer.total_epochs=5 \
    trainer.max_steps=500 \
    trainer.max_try_make_batch=10 \
    trainer.experiment_name=qwen3_vl_2b_ms_cxr_dapo'''



'''
chestxray8数据集: 
    CUDA_VISIBLE_DEVICES=2
    python -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=/datasets/ms_cxr/chestxray8_output_train_fixed.jsonl \
    data.val_files=/datasets/ms_cxr/chestxray8_output_test_fixed.jsonl \
    data.prompt_key=questions \
    data.answer_key=label_text \
    data.image_key=path \
    data.image_dir=/datasets/ms_cxr \
    data.format_prompt=./examples/format_prompt/chestxray8.jinja \
    data.max_prompt_length=1024 \
    data.max_response_length=1356 \
    data.min_pixels=65536 \
    worker.rollout.n=2 \
    data.rollout_batch_size=1 \
    data.mini_rollout_batch_size=1 \
    worker.rollout.max_num_batched_tokens=8192 \
    worker.rollout.gpu_memory_utilization=0.6 \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.temperature=1.0 \
    worker.rollout.top_p=0.7 \
    worker.rollout.val_override_config='{"n":1,"temperature":0.35,"top_p":0.95}' \
    trainer.n_gpus_per_node=1 \
    worker.actor.ulysses_size=1 \
    worker.actor.model.model_path=/pretrained_model/qwen25_vl_3b_sft \
    worker.actor.model.lora.rank=8 \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.actor.optim.weight_decay=0.1 \
    worker.actor.optim.lr=5e-5 \
    worker.actor.optim.lr_warmup_steps=10 \
    worker.actor.global_batch_size=1 \
    worker.actor.clip_ratio_low=0.2 \
    worker.actor.clip_ratio_high=0.28 \
    worker.actor.clip_ratio_dual=10.0 \
    worker.reward.reward_function=./examples/reward_function/ms_cxr.py:reward_function \
    worker.reward.reward_function_kwargs='{"data_format":"original"}' \
    algorithm.disable_kl=True \
    algorithm.online_filtering=True \
    algorithm.filter_key=overall \
    algorithm.filter_low=0 \
    algorithm.filter_high=1.0 \
    trainer.total_epochs=5 \
    trainer.max_steps=500 \
    trainer.max_try_make_batch=10 \
    trainer.experiment_name=qwen3_vl_2b_chestxray8_dapo'''


'''
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
    worker.actor.model.model_path=/pretrained_model/qwen25_vl_3b \      
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
    worker.rollout.gpu_memory_utilization=0.7 \
    worker.rollout.tensor_parallel_size=2 \
    worker.reward.reward_function=./examples/reward_function/ms_cxr.py:reward_function \
    worker.reward.reward_function_kwargs='{"data_format":"original"}' \
    algorithm.disable_kl=True \
    algorithm.online_filtering=True \
    algorithm.filter_key=accuracy \
    algorithm.filter_low=0.0 \
    algorithm.filter_high=1.0 \
    trainer.total_epochs=5 \
    trainer.max_try_make_batch=10 \
    trainer.experiment_name=qwen3_vl_4b_ms_cxr_dapo \
    trainer.n_gpus_per_node=4 \
    trainer.logger='["file"]' \
    worker.actor.micro_batch_size_per_device_for_experience=1
'''