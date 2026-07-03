#!/usr/bin/env bash
# NVFP4 QAT W4A4 FFN-ONLY (Weight + Activation FP4, skip attention) for Qwen3-30B-A3B-Base
set -euxo pipefail
current_dir="$(dirname "$(readlink -f "$0")")"

project_name=${project_name:-'DAPO-NVFP4-QAT'}
exp_name=${exp_name:-'DAPO-Qwen3-30B-A3B-W4A4-FFN-ONLY'}

adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

# Rollout Correction parameters (for quantized rollout)
rollout_is=token               # token-level importance sampling
rollout_is_threshold=2.0
rollout_rs=null                # response-level sampling (null = disabled)
rollout_rs_threshold=null
# rollout_rs_threshold_lower and rollout_token_veto_threshold removed in new verl version

clip_ratio_low=0.2
clip_ratio_high=0.28

max_prompt_length=${max_prompt_length:-$((1024))}
max_response_length=${max_response_length:-$((1024 * 20))}
enable_overlong_buffer=True
overlong_buffer_len=512
overlong_penalty_factor=1.0

loss_agg_mode=${loss_agg_mode:-"token-mean"}
policy_loss_mode=${policy_loss_mode:-"vanilla"}

enable_filter_groups=${enable_filter_groups:-True}
filter_groups_metric=acc
max_num_gen_batches=${max_num_gen_batches:-10}
train_prompt_bsz=${train_prompt_bsz:-32}
gen_prompt_bsz=${gen_prompt_bsz:-$((train_prompt_bsz * 2))}
n_resp_per_prompt=${n_resp_per_prompt:-16}
train_prompt_mini_bsz=${train_prompt_mini_bsz:-32}


# Gap-guard / local segment align strategy knobs. Defaults are off.
pg_mask_enable=${pg_mask_enable:-False}
pg_mask_delta_min=${pg_mask_delta_min:-null}
pg_mask_delta_max=${pg_mask_delta_max:-null}
pg_mask_bad_weight=${pg_mask_bad_weight:-0.0}
seq_norm_adaptive_enable=${seq_norm_adaptive_enable:-False}
seq_norm_adaptive_quantile=${seq_norm_adaptive_quantile:-0.95}
seq_norm_adaptive_ema_beta=${seq_norm_adaptive_ema_beta:-0.90}
seq_norm_adaptive_threshold_1=${seq_norm_adaptive_threshold_1:-4096.0}
seq_norm_adaptive_threshold_2=${seq_norm_adaptive_threshold_2:-8192.0}
seq_norm_adaptive_threshold_3=${seq_norm_adaptive_threshold_3:-12288.0}
seq_norm_adaptive_denom_1=${seq_norm_adaptive_denom_1:-8192.0}
seq_norm_adaptive_denom_2=${seq_norm_adaptive_denom_2:-12288.0}
seq_norm_adaptive_denom_3=${seq_norm_adaptive_denom_3:-16384.0}
seq_norm_adaptive_denom_4=${seq_norm_adaptive_denom_4:-20480.0}
seq_norm_adaptive_tail_denom=${seq_norm_adaptive_tail_denom:-20480.0}
seq_norm_adaptive_tail_power=${seq_norm_adaptive_tail_power:-0.0}
adv_length_norm_enable=${adv_length_norm_enable:-False}
adv_length_norm_stage=${adv_length_norm_stage:-loss}
adv_length_norm_mode=${adv_length_norm_mode:-neg_only}
adv_length_norm_ref_len=${adv_length_norm_ref_len:-4096}
adv_length_norm_quantile=${adv_length_norm_quantile:-0.95}
adv_length_norm_alpha=${adv_length_norm_alpha:-0.25}
adv_length_norm_min_scale=${adv_length_norm_min_scale:-0.70}
adv_length_norm_max_scale=${adv_length_norm_max_scale:-1.0}
seg_gate_enable=${seg_gate_enable:-False}
seg_gate_size=${seg_gate_size:-64}
seg_gate_neg_delta_threshold=${seg_gate_neg_delta_threshold:--0.5}
seg_gate_neg_adv_max=${seg_gate_neg_adv_max:-0.0}
seg_gate_neg_weight=${seg_gate_neg_weight:-0.3}
seg_gate_severe_delta_threshold=${seg_gate_severe_delta_threshold:--1.5}
seg_gate_bad_delta_threshold=${seg_gate_bad_delta_threshold:--6.0}
seg_gate_bad_fraction_threshold=${seg_gate_bad_fraction_threshold:-0.02}
seg_gate_severe_weight=${seg_gate_severe_weight:-0.1}
seg_gate_pos_enable=${seg_gate_pos_enable:-False}
local_segment_align_enable=${local_segment_align_enable:-False}
local_segment_align_coef=${local_segment_align_coef:-0.0}
local_segment_align_target=${local_segment_align_target:--0.03}
local_segment_align_size=${local_segment_align_size:-128}
local_segment_align_loss_cap=${local_segment_align_loss_cap:-null}
local_segment_align_length_gate_enable=${local_segment_align_length_gate_enable:-False}
local_segment_align_length_gate_quantile=${local_segment_align_length_gate_quantile:-0.95}
local_segment_align_length_gate_min=${local_segment_align_length_gate_min:-8192.0}
local_segment_align_length_gate_mean_min=${local_segment_align_length_gate_mean_min:-7000.0}
local_segment_align_length_gate_clip_ratio_min=${local_segment_align_length_gate_clip_ratio_min:-0.02}
local_segment_align_kl_gate_enable=${local_segment_align_kl_gate_enable:-False}
local_segment_align_kl_gate_start=${local_segment_align_kl_gate_start:-0.01}
local_segment_align_kl_gate_full=${local_segment_align_kl_gate_full:-0.02}
local_segment_align_kl_gate_min_factor=${local_segment_align_kl_gate_min_factor:-0.0}
local_segment_align_kl_gate_max_factor=${local_segment_align_kl_gate_max_factor:-1.0}
local_segment_align_adaptive_tail_enable=${local_segment_align_adaptive_tail_enable:-False}
local_segment_align_adaptive_tail_offset=${local_segment_align_adaptive_tail_offset:-4096.0}
local_segment_align_adaptive_tail_min_start=${local_segment_align_adaptive_tail_min_start:-8192.0}
local_segment_align_adaptive_tail_max_start=${local_segment_align_adaptive_tail_max_start:-14336.0}
local_segment_align_adaptive_tail_width=${local_segment_align_adaptive_tail_width:-4096.0}
local_segment_align_adaptive_tail_mass_min=${local_segment_align_adaptive_tail_mass_min:-0.02}
local_segment_align_adaptive_tail_clip_gate_enable=${local_segment_align_adaptive_tail_clip_gate_enable:-False}
local_segment_align_adaptive_tail_clip_ratio_min=${local_segment_align_adaptive_tail_clip_ratio_min:-0.02}
local_segment_align_adaptive_tail_uniform_weight=${local_segment_align_adaptive_tail_uniform_weight:-False}
resume_mode=${resume_mode:-auto}
val_before_train=${val_before_train:-True}
test_freq=${test_freq:-5}
save_freq=${save_freq:-5}
max_actor_ckpt_to_keep=${max_actor_ckpt_to_keep:-3}
resume_from_path=${resume_from_path:-null}
total_training_steps=${total_training_steps:-null}
total_epochs=${total_epochs:-1}
data_shuffle=${data_shuffle:-True}
filter_overlong_prompts=${filter_overlong_prompts:-True}
filter_overlong_prompts_workers=${filter_overlong_prompts_workers:-1}
train_max_samples=${train_max_samples:--1}
enable_rollout_routing_replay=${enable_rollout_routing_replay:-False}
enable_prefix_caching=${enable_prefix_caching:-True}
rollout_enforce_eager=${rollout_enforce_eager:-False}
rollout_cudagraph_mode=${rollout_cudagraph_mode:-FULL_AND_PIECEWISE}
router_replay_mode=${router_replay_mode:-disabled}
fsdp_use_torch_compile=${fsdp_use_torch_compile:-True}
checkpoint_load_contents=${checkpoint_load_contents:-"['model','optimizer','extra']"}
wandb_run_id=${wandb_run_id:-null}
wandb_resume=${wandb_resume:-null}

if [[ "${router_replay_mode}" == "R3" ]]; then
    [[ "${enable_rollout_routing_replay}" == "True" ]] || {
        echo "R3 requires enable_rollout_routing_replay=True" >&2
        exit 2
    }
    [[ "${enable_prefix_caching}" == "False" ]] || {
        echo "FSDP R3 requires enable_prefix_caching=False until cached prompt routes are replayable" >&2
        exit 2
    }
    [[ "${fsdp_use_torch_compile}" == "False" ]] || {
        echo "FSDP R3 requires fsdp_use_torch_compile=False for hook/recompute correctness" >&2
        exit 2
    }
    # Keep production capture separate from VERL_ROUTE_DIAG.  The latter also
    # enables expensive natural-vs-forced counterfactual forward passes.
    export VERL_ROLLOUT_ROUTE_CAPTURE=1
fi

# Ray
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
NNODES=${NNODES:-1}

# Paths
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-30B-A3B-Base"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/dapo-math-17k.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/aime-2024.parquet"}

# QAT Configuration
qat_enable=True
qat_mode=${qat_mode:-w4a4}    # w4a4 for weight + activation FP4; override to w4a16 to A/B update_weights
qat_config_path="${qat_config_path:-"${WORKING_DIR}/recipe/qat/config/nvfp4_w4a4.json"}"

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=1.0

# Performance
sp_size=1
use_dynamic_bsz=True
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
offload=${offload:-True}
gen_tp=${gen_tp:-1}

export VERL_LOGGING_LEVEL=DEBUG
export VERL_PPO_LOGGING_LEVEL=DEBUG
export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_CONFIGURE_LOGGING=1
export VLLM_USE_V1=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_DIST_TIMEOUT=4000

python3 -m recipe.dapo.main_dapo \
    --config-path "${WORKING_DIR}/recipe/qat/config" \
    --config-name dapo_qat_trainer \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.return_raw_chat=True \
    data.shuffle=${data_shuffle} \
    data.train_max_samples=${train_max_samples} \
    data.filter_overlong_prompts=${filter_overlong_prompts} \
    data.filter_overlong_prompts_workers=${filter_overlong_prompts_workers} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.train_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    algorithm.rollout_correction.rollout_is=${rollout_is} \
    algorithm.rollout_correction.rollout_is_threshold=${rollout_is_threshold} \
    algorithm.rollout_correction.rollout_rs=${rollout_rs} \
    algorithm.rollout_correction.rollout_rs_threshold=${rollout_rs_threshold} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.90 \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.use_orig_params=${use_orig_params:-False} \
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=${fsdp_use_torch_compile} \
    actor_rollout_ref.actor.fsdp_config.router_replay.mode=${router_replay_mode} \
    actor_rollout_ref.actor.router_replay.mode=${router_replay_mode} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.policy_loss.loss_mode=${policy_loss_mode} \
    actor_rollout_ref.actor.pg_mask_enable=${pg_mask_enable} \
    actor_rollout_ref.actor.pg_mask_delta_min=${pg_mask_delta_min} \
    actor_rollout_ref.actor.pg_mask_delta_max=${pg_mask_delta_max} \
    actor_rollout_ref.actor.pg_mask_bad_weight=${pg_mask_bad_weight} \
    actor_rollout_ref.actor.seq_norm_adaptive_enable=${seq_norm_adaptive_enable} \
    actor_rollout_ref.actor.seq_norm_adaptive_quantile=${seq_norm_adaptive_quantile} \
    actor_rollout_ref.actor.seq_norm_adaptive_ema_beta=${seq_norm_adaptive_ema_beta} \
    actor_rollout_ref.actor.seq_norm_adaptive_threshold_1=${seq_norm_adaptive_threshold_1} \
    actor_rollout_ref.actor.seq_norm_adaptive_threshold_2=${seq_norm_adaptive_threshold_2} \
    actor_rollout_ref.actor.seq_norm_adaptive_threshold_3=${seq_norm_adaptive_threshold_3} \
    actor_rollout_ref.actor.seq_norm_adaptive_denom_1=${seq_norm_adaptive_denom_1} \
    actor_rollout_ref.actor.seq_norm_adaptive_denom_2=${seq_norm_adaptive_denom_2} \
    actor_rollout_ref.actor.seq_norm_adaptive_denom_3=${seq_norm_adaptive_denom_3} \
    actor_rollout_ref.actor.seq_norm_adaptive_denom_4=${seq_norm_adaptive_denom_4} \
    actor_rollout_ref.actor.seq_norm_adaptive_tail_denom=${seq_norm_adaptive_tail_denom} \
    actor_rollout_ref.actor.seq_norm_adaptive_tail_power=${seq_norm_adaptive_tail_power} \
    actor_rollout_ref.actor.adv_length_norm_enable=${adv_length_norm_enable} \
    actor_rollout_ref.actor.adv_length_norm_stage=${adv_length_norm_stage} \
    actor_rollout_ref.actor.adv_length_norm_mode=${adv_length_norm_mode} \
    actor_rollout_ref.actor.adv_length_norm_ref_len=${adv_length_norm_ref_len} \
    actor_rollout_ref.actor.adv_length_norm_quantile=${adv_length_norm_quantile} \
    actor_rollout_ref.actor.adv_length_norm_alpha=${adv_length_norm_alpha} \
    actor_rollout_ref.actor.adv_length_norm_min_scale=${adv_length_norm_min_scale} \
    actor_rollout_ref.actor.adv_length_norm_max_scale=${adv_length_norm_max_scale} \
    actor_rollout_ref.actor.seg_gate_enable=${seg_gate_enable} \
    actor_rollout_ref.actor.seg_gate_size=${seg_gate_size} \
    actor_rollout_ref.actor.seg_gate_neg_delta_threshold=${seg_gate_neg_delta_threshold} \
    actor_rollout_ref.actor.seg_gate_neg_adv_max=${seg_gate_neg_adv_max} \
    actor_rollout_ref.actor.seg_gate_neg_weight=${seg_gate_neg_weight} \
    actor_rollout_ref.actor.seg_gate_severe_delta_threshold=${seg_gate_severe_delta_threshold} \
    actor_rollout_ref.actor.seg_gate_bad_delta_threshold=${seg_gate_bad_delta_threshold} \
    actor_rollout_ref.actor.seg_gate_bad_fraction_threshold=${seg_gate_bad_fraction_threshold} \
    actor_rollout_ref.actor.seg_gate_severe_weight=${seg_gate_severe_weight} \
    actor_rollout_ref.actor.seg_gate_pos_enable=${seg_gate_pos_enable} \
    actor_rollout_ref.actor.local_segment_align_enable=${local_segment_align_enable} \
    actor_rollout_ref.actor.local_segment_align_coef=${local_segment_align_coef} \
    actor_rollout_ref.actor.local_segment_align_target=${local_segment_align_target} \
    actor_rollout_ref.actor.local_segment_align_size=${local_segment_align_size} \
    actor_rollout_ref.actor.local_segment_align_loss_cap=${local_segment_align_loss_cap} \
    actor_rollout_ref.actor.local_segment_align_length_gate_enable=${local_segment_align_length_gate_enable} \
    actor_rollout_ref.actor.local_segment_align_length_gate_quantile=${local_segment_align_length_gate_quantile} \
    actor_rollout_ref.actor.local_segment_align_length_gate_min=${local_segment_align_length_gate_min} \
    actor_rollout_ref.actor.local_segment_align_length_gate_mean_min=${local_segment_align_length_gate_mean_min} \
    actor_rollout_ref.actor.local_segment_align_length_gate_clip_ratio_min=${local_segment_align_length_gate_clip_ratio_min} \
    actor_rollout_ref.actor.local_segment_align_kl_gate_enable=${local_segment_align_kl_gate_enable} \
    actor_rollout_ref.actor.local_segment_align_kl_gate_start=${local_segment_align_kl_gate_start} \
    actor_rollout_ref.actor.local_segment_align_kl_gate_full=${local_segment_align_kl_gate_full} \
    actor_rollout_ref.actor.local_segment_align_kl_gate_min_factor=${local_segment_align_kl_gate_min_factor} \
    actor_rollout_ref.actor.local_segment_align_kl_gate_max_factor=${local_segment_align_kl_gate_max_factor} \
    actor_rollout_ref.actor.local_segment_align_adaptive_tail_enable=${local_segment_align_adaptive_tail_enable} \
    actor_rollout_ref.actor.local_segment_align_adaptive_tail_offset=${local_segment_align_adaptive_tail_offset} \
    actor_rollout_ref.actor.local_segment_align_adaptive_tail_min_start=${local_segment_align_adaptive_tail_min_start} \
    actor_rollout_ref.actor.local_segment_align_adaptive_tail_max_start=${local_segment_align_adaptive_tail_max_start} \
    actor_rollout_ref.actor.local_segment_align_adaptive_tail_width=${local_segment_align_adaptive_tail_width} \
    actor_rollout_ref.actor.local_segment_align_adaptive_tail_mass_min=${local_segment_align_adaptive_tail_mass_min} \
    actor_rollout_ref.actor.local_segment_align_adaptive_tail_clip_gate_enable=${local_segment_align_adaptive_tail_clip_gate_enable} \
    actor_rollout_ref.actor.local_segment_align_adaptive_tail_clip_ratio_min=${local_segment_align_adaptive_tail_clip_ratio_min} \
    actor_rollout_ref.actor.local_segment_align_adaptive_tail_uniform_weight=${local_segment_align_adaptive_tail_uniform_weight} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.qat.enable=${qat_enable} \
    actor_rollout_ref.actor.fsdp_config.qat.mode=${qat_mode} \
    actor_rollout_ref.actor.fsdp_config.qat.quantization_config_path="${qat_config_path}" \
    'actor_rollout_ref.actor.fsdp_config.qat.ignore_patterns=["lm_head", "embed_tokens", "re:.*mlp.gate$", "re:.*self_attn.*"]' \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$(( 1024 * 32 )) \
    actor_rollout_ref.rollout.max_num_seqs=256 \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=${top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.enable_rollout_routing_replay=${enable_rollout_routing_replay} \
    actor_rollout_ref.rollout.enable_prefix_caching=${enable_prefix_caching} \
    actor_rollout_ref.rollout.enforce_eager=${rollout_enforce_eager} \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode=${rollout_cudagraph_mode} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    reward.reward_manager.name=dapo \
    reward.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    reward.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    reward.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.wandb_run_id=${wandb_run_id} \
    trainer.wandb_resume=${wandb_resume} \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=${val_before_train} \
    trainer.test_freq=${test_freq} \
    trainer.save_freq=${save_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.total_training_steps=${total_training_steps} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=${resume_mode} \
    trainer.resume_from_path=${resume_from_path} \
    "actor_rollout_ref.actor.checkpoint.load_contents=${checkpoint_load_contents}" \
    trainer.max_actor_ckpt_to_keep=${max_actor_ckpt_to_keep} \
    trainer.use_legacy_worker_impl=disable
