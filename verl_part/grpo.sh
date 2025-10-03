#!/bin/bash

#SBATCH --time=2880
#SBATCH --mem=128gb
#SBATCH -o verlkd.%j.out
#SBATCH --job-name=vekd
#SBATCH --gpus=h100-96:2
source ~/miniconda3/etc/profile.d/conda.sh
conda activate verlkd
hostname
set -xeuo pipefail
unset ROCR_VISIBLE_DEVICES
export WANDB_API_KEY='b12499bd7d87c048103392ad82a80ea103be062c'
# export VLLM_USE_V1=1

timestamp=$(date "+%Y-%m-%d_%H-%M")
project_name='GRPO-1.5'
subexp='GRPO-noKL-'
kl_eff=0
exp_name=${subexp}${kl_eff}${timestamp}
custom_reward_function="${HOME}/verl/reward/reward_zoo.py"
custom_reward_function_name="compute_score"
train_files="data/mod-dapo-math-17k.parquet"
test_files="data/aime-2024-1.parquet"
model_path="/home/i/i0002066/model/DSDQ1-5"
teacher_ref_path="/home/i/i0002066/model/dqw-7b"

HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=64 \
    data.max_prompt_length=1024 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    actor_rollout_ref.model.path=${model_path} \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=${kl_eff} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.max_num_batched_tokens=9216 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path=${custom_reward_function} \
    custom_reward_function.name=${custom_reward_function_name} \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${exp_name} \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.log_val_generations=4 \
    trainer.val_before_train=False \
    trainer.save_freq=110 \
    trainer.test_freq=30 \
    trainer.total_epochs=1 $@
