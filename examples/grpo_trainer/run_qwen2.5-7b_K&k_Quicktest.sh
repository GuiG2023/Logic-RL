#!/bin/bash
# Logic-RL 快速验证脚本 - 部分数据训练
# 目标: 快速验证是否有明显提升 (30-60分钟完成)

set -x

echo "🚀 开始Logic-RL快速验证训练..."
echo "📊 预期: baseline ~20% → RL训练后 ~40-60%"

export VLLM_ATTENTION_BACKEND=XFORMERS

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=data/kk/instruct/3ppl/train.parquet \
    data.val_files=data/kk/instruct/3ppl/test.parquet \
    data.train_batch_size=32 \
    data.val_batch_size=50 \
    data.max_prompt_length=512 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='Logic-RL-Quick-Test' \
    trainer.experiment_name='qwen2.5_7b_quick_validation' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=1 \
    trainer.test_freq=1 \
    trainer.total_epochs=3 $@

echo ""
echo "✅ 快速训练完成！"
echo "📁 模型保存在: outputs/Logic-RL-Quick-Test/"
echo ""
echo "🔍 下一步验证命令:"
echo "python eval_kk/main_eval_instruct.py \\"
echo "    --model outputs/Logic-RL-Quick-Test/qwen2.5_7b_quick_validation/checkpoint-3 \\"
echo "    --data_dir ./data/kk/instruct/3ppl/ \\"
echo "    --split test \\"
echo "    --limit 50 \\"
echo "    --cot \\"
echo "    --max_token 2048 \\"
echo "    --eval_nppl 3"