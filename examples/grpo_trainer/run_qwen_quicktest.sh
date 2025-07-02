#!/bin/bash
# Logic-RL 快速验证版本 - 修复版2
set -x

echo "🚀 开始Logic-RL快速验证训练..."

# 内存优化环境变量
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export VLLM_ATTENTION_BACKEND=XFORMERS

# 清理GPU缓存
python3 -c "import torch; torch.cuda.empty_cache()"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=data/kk/instruct/3ppl/train.parquet \
    data.val_files=data/kk/instruct/3ppl/test.parquet \
    data.train_batch_size=2 \
    data.val_batch_size=4 \
    data.max_prompt_length=256 \
    data.max_response_length=512 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.01 \
    trainer.critic_warmup=0 \
    trainer.logger=[console] \
    trainer.project_name=Logic-RL-FastValidation \
    trainer.experiment_name=qwen2.5_7b_fast \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=25 \
    trainer.total_epochs=1 \
    +trainer.max_steps_per_epoch=200 \
    +trainer.gradient_accumulation_steps=2 \
    +trainer.early_stopping_patience=3 \
    +trainer.checkpoint_every_n_minutes=10

echo "✅ 快速验证训练完成！检查输出目录获取checkpoint。"