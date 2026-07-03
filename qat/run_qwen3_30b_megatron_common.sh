#!/usr/bin/env bash
# Shared Qwen3-30B-A3B Megatron recipe for BF16 and NVFP4 QAT runs.
set -xeuo pipefail

project_name=${project_name:-"DAPO-NVFP4-QAT"}
exp_name=${exp_name:?"set exp_name in the wrapper or submission env"}

adv_estimator=${adv_estimator:-grpo}
use_kl_in_reward=${use_kl_in_reward:-False}
kl_coef=${kl_coef:-0.0}
use_kl_loss=${use_kl_loss:-False}
kl_loss_coef=${kl_loss_coef:-0.0}
clip_ratio_low=${clip_ratio_low:-0.2}
clip_ratio_high=${clip_ratio_high:-0.28}

max_prompt_length=${max_prompt_length:-$((1024))}
max_response_length=${max_response_length:-$((1024 * 20))}
enable_overlong_buffer=${enable_overlong_buffer:-True}
overlong_buffer_len=${overlong_buffer_len:-512}
overlong_penalty_factor=${overlong_penalty_factor:-1.0}

loss_agg_mode=${loss_agg_mode:-"token-mean"}
policy_loss_mode=${policy_loss_mode:-vanilla}
entropy_coeff=${entropy_coeff:-0}

train_prompt_bsz=${train_prompt_bsz:-32}
n_resp_per_prompt=${n_resp_per_prompt:-16}
train_prompt_mini_bsz=${train_prompt_mini_bsz:-${train_prompt_bsz}}
gen_prompt_bsz=${gen_prompt_bsz:-$((train_prompt_bsz * 2))}
total_training_steps=${total_training_steps:-}
filter_overlong_prompts=${filter_overlong_prompts:-True}

WORKING_DIR=${WORKING_DIR:-"${PWD}"}
NNODES=${NNODES:-1}
n_gpus_per_node=${n_gpus_per_node:-4}
total_gpus=$((NNODES * n_gpus_per_node))

RAY_DATA_HOME=${RAY_DATA_HOME:-"${WORKING_DIR}"}
MODEL_PATH=${MODEL_PATH:?"set MODEL_PATH to Qwen3-30B-A3B-Base"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/dapo-math-17k.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/aime-2024.parquet"}

temperature=${temperature:-1.0}
top_p=${top_p:-1.0}
top_k=${top_k:--1}
val_top_p=${val_top_p:-1.0}
enable_filter_groups=${enable_filter_groups:-True}
filter_groups_metric=${filter_groups_metric:-acc}
max_num_gen_batches=${max_num_gen_batches:-10}

use_dynamic_bsz=${use_dynamic_bsz:-True}
actor_ppo_max_token_len=${actor_ppo_max_token_len:-$((max_prompt_length + max_response_length))}
infer_ppo_max_token_len=${infer_ppo_max_token_len:-$((max_prompt_length + max_response_length))}
offload=${offload:-True}

train_tp=${train_tp:-1}
train_pp=${train_pp:-1}
train_ep=${train_ep:-${total_gpus}}
train_etp=${train_etp:-1}
if [ -z "${train_cp+x}" ]; then
    if [ "${total_gpus}" -ge 32 ]; then
        train_cp=8
    else
        train_cp=1
    fi
fi

gen_tp=${gen_tp:-1}
gen_dp=${gen_dp:-1}
gen_ep=${gen_ep:-1}
gpu_memory_utilization=${gpu_memory_utilization:-0.50}
max_num_batched_tokens=${max_num_batched_tokens:-$((1024 * 16))}
max_num_seqs=${max_num_seqs:-128}
vllm_moe_backend=${vllm_moe_backend:-null}

rollout_is=${rollout_is:-token}
rollout_is_threshold=${rollout_is_threshold:-2.0}
rollout_rs=${rollout_rs:-null}

pg_mask_enable=${pg_mask_enable:-False}
pg_mask_delta_min=${pg_mask_delta_min:-null}
pg_mask_delta_max=${pg_mask_delta_max:-null}
pg_mask_bad_weight=${pg_mask_bad_weight:-0.0}

vllm_align_enable=${vllm_align_enable:-False}
vllm_align_coef=${vllm_align_coef:-0.0}
vllm_align_loss_type=${vllm_align_loss_type:-k3}
vllm_align_delta_min=${vllm_align_delta_min:-null}
vllm_align_delta_max=${vllm_align_delta_max:-null}
vllm_align_abs_delta_min=${vllm_align_abs_delta_min:-null}
vllm_align_abs_delta_max=${vllm_align_abs_delta_max:-null}
vllm_align_adv_min=${vllm_align_adv_min:-null}
vllm_align_adv_max=${vllm_align_adv_max:-null}
vllm_align_segment_pos_enable=${vllm_align_segment_pos_enable:-False}
vllm_align_huber_beta=${vllm_align_huber_beta:-1.0}
vllm_align_normalize_by_window=${vllm_align_normalize_by_window:-False}
vllm_align_min_window_fraction=${vllm_align_min_window_fraction:-0.0}
vllm_align_loss_cap=${vllm_align_loss_cap:-null}
vllm_align_hard_enable=${vllm_align_hard_enable:-False}
vllm_align_hard_coef=${vllm_align_hard_coef:-0.0}
vllm_align_hard_loss_type=${vllm_align_hard_loss_type:-huber}
vllm_align_hard_delta_min=${vllm_align_hard_delta_min:-null}
vllm_align_hard_delta_max=${vllm_align_hard_delta_max:-null}
vllm_align_hard_abs_delta_min=${vllm_align_hard_abs_delta_min:-null}
vllm_align_hard_abs_delta_max=${vllm_align_hard_abs_delta_max:-null}
vllm_align_hard_huber_beta=${vllm_align_hard_huber_beta:-1.0}
vllm_align_hard_normalize_by_window=${vllm_align_hard_normalize_by_window:-False}
vllm_align_hard_loss_cap=${vllm_align_hard_loss_cap:-null}

delta_mean_guard_enable=${delta_mean_guard_enable:-False}
delta_mean_guard_coef=${delta_mean_guard_coef:-0.0}
delta_mean_guard_target=${delta_mean_guard_target:--0.01}
delta_mean_guard_mode=${delta_mean_guard_mode:-token}
delta_mean_guard_delta_min=${delta_mean_guard_delta_min:-null}
delta_mean_guard_delta_max=${delta_mean_guard_delta_max:-null}
delta_mean_guard_adv_min=${delta_mean_guard_adv_min:-null}
delta_mean_guard_adv_max=${delta_mean_guard_adv_max:-null}
delta_mean_guard_loss_cap=${delta_mean_guard_loss_cap:-null}

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
local_segment_align_activation_gate_enable=${local_segment_align_activation_gate_enable:-False}
local_segment_align_activation_clip_min=${local_segment_align_activation_clip_min:-0.005}
local_segment_align_activation_clip_period=${local_segment_align_activation_clip_period:-4}
local_segment_align_activation_raw_ratio=${local_segment_align_activation_raw_ratio:-1.3}
local_segment_align_activation_raw_min=${local_segment_align_activation_raw_min:-5e-5}
local_segment_align_activation_raw_period=${local_segment_align_activation_raw_period:-3}
local_segment_align_activation_ema_span=${local_segment_align_activation_ema_span:-20.0}
local_segment_align_activation_baseline_span=${local_segment_align_activation_baseline_span:-100.0}

adv_length_norm_enable=${adv_length_norm_enable:-False}
adv_length_norm_mode=${adv_length_norm_mode:-neg_only}
adv_length_norm_ref_len=${adv_length_norm_ref_len:-4096}
adv_length_norm_alpha=${adv_length_norm_alpha:-0.5}
adv_length_norm_min_scale=${adv_length_norm_min_scale:-0.5}
adv_length_norm_max_scale=${adv_length_norm_max_scale:-1.0}

seg_gate_enable=${seg_gate_enable:-False}
seg_gate_size=${seg_gate_size:-128}
seg_gate_neg_delta_threshold=${seg_gate_neg_delta_threshold:--0.5}
seg_gate_neg_adv_max=${seg_gate_neg_adv_max:-0.0}
seg_gate_neg_weight=${seg_gate_neg_weight:-0.3}
seg_gate_severe_delta_threshold=${seg_gate_severe_delta_threshold:--1.5}
seg_gate_bad_delta_threshold=${seg_gate_bad_delta_threshold:--6.0}
seg_gate_bad_fraction_threshold=${seg_gate_bad_fraction_threshold:-0.02}
seg_gate_severe_weight=${seg_gate_severe_weight:-0.1}
seg_gate_pos_enable=${seg_gate_pos_enable:-False}
seg_gate_pos_delta_threshold=${seg_gate_pos_delta_threshold:--0.5}
seg_gate_pos_adv_min=${seg_gate_pos_adv_min:-0.0}

qat_enable=${qat_enable:-False}
qat_mode=${qat_mode:-w4a16}
qat_config_path=${qat_config_path:-}
qat_ignore_patterns=${qat_ignore_patterns:-'["lm_head","*mlp.gate","*self_attn*"]'}

# W4A4/W4A8 periodic activation-amax re-calibration (0 = off). Refreshes the frozen actor
# input_quantizer amax every N policy updates so the FP4 activation scale tracks drift.
recalib_every=${recalib_every:-0}

moe_router_dtype=${moe_router_dtype:-fp32}
moe_enable_deepep=${moe_enable_deepep:-True}
moe_token_dispatcher_type=${moe_token_dispatcher_type:-flex}
moe_permute_fusion=${moe_permute_fusion:-True}

export VERL_LOGGING_LEVEL=${VERL_LOGGING_LEVEL:-INFO}
export VLLM_CONFIGURE_LOGGING=${VLLM_CONFIGURE_LOGGING:-1}
export VLLM_USE_V1=${VLLM_USE_V1:-1}
export TORCH_NCCL_AVOID_RECORD_STREAMS=${TORCH_NCCL_AVOID_RECORD_STREAMS:-1}
export VERL_QAT_MODE=${qat_mode}
if [ "${qat_mode}" = "w4a16" ]; then
    export VLLM_NVFP4_GEMM_BACKEND=marlin
fi

DATA=(
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.prompt_key=prompt
    data.truncation='left'
    data.return_raw_chat=True
    data.filter_overlong_prompts=${filter_overlong_prompts}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.gen_batch_size=${gen_prompt_bsz}
    data.train_batch_size=${train_prompt_bsz}
)

ALGORITHM=(
    algorithm.adv_estimator=${adv_estimator}
    algorithm.use_kl_in_reward=${use_kl_in_reward}
    algorithm.kl_ctrl.kl_coef=${kl_coef}
    algorithm.filter_groups.enable=${enable_filter_groups}
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches}
    algorithm.filter_groups.metric=${filter_groups_metric}
    algorithm.rollout_correction.rollout_is=${rollout_is}
    algorithm.rollout_correction.rollout_is_threshold=${rollout_is_threshold}
    algorithm.rollout_correction.rollout_rs=${rollout_rs}
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=True
)

ACTOR=(
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss}
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low}
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high}
    actor_rollout_ref.actor.clip_ratio_c=10.0
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len}
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.lr_warmup_steps=0
    actor_rollout_ref.actor.optim.weight_decay=0.1
    actor_rollout_ref.actor.optim.clip_grad=1.0
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz}
    actor_rollout_ref.actor.megatron.param_offload=${offload}
    actor_rollout_ref.actor.megatron.optimizer_offload=${offload}
    actor_rollout_ref.actor.megatron.grad_offload=${offload}
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${train_tp}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${train_pp}
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${train_ep}
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${train_etp}
    actor_rollout_ref.actor.megatron.context_parallel_size=${train_cp}
    actor_rollout_ref.actor.megatron.sequence_parallel=False
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.vanilla_mbridge=False
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff}
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode}
    actor_rollout_ref.actor.policy_loss.loss_mode=${policy_loss_mode}
    actor_rollout_ref.actor.pg_mask_enable=${pg_mask_enable}
    actor_rollout_ref.actor.pg_mask_delta_min=${pg_mask_delta_min}
    actor_rollout_ref.actor.pg_mask_delta_max=${pg_mask_delta_max}
    actor_rollout_ref.actor.pg_mask_bad_weight=${pg_mask_bad_weight}
    actor_rollout_ref.actor.vllm_align_enable=${vllm_align_enable}
    actor_rollout_ref.actor.vllm_align_coef=${vllm_align_coef}
    actor_rollout_ref.actor.vllm_align_loss_type=${vllm_align_loss_type}
    actor_rollout_ref.actor.vllm_align_delta_min=${vllm_align_delta_min}
    actor_rollout_ref.actor.vllm_align_delta_max=${vllm_align_delta_max}
    actor_rollout_ref.actor.vllm_align_abs_delta_min=${vllm_align_abs_delta_min}
    actor_rollout_ref.actor.vllm_align_abs_delta_max=${vllm_align_abs_delta_max}
    actor_rollout_ref.actor.vllm_align_adv_min=${vllm_align_adv_min}
    actor_rollout_ref.actor.vllm_align_adv_max=${vllm_align_adv_max}
    actor_rollout_ref.actor.vllm_align_segment_pos_enable=${vllm_align_segment_pos_enable}
    actor_rollout_ref.actor.vllm_align_huber_beta=${vllm_align_huber_beta}
    actor_rollout_ref.actor.vllm_align_normalize_by_window=${vllm_align_normalize_by_window}
    actor_rollout_ref.actor.vllm_align_min_window_fraction=${vllm_align_min_window_fraction}
    actor_rollout_ref.actor.vllm_align_loss_cap=${vllm_align_loss_cap}
    actor_rollout_ref.actor.vllm_align_hard_enable=${vllm_align_hard_enable}
    actor_rollout_ref.actor.vllm_align_hard_coef=${vllm_align_hard_coef}
    actor_rollout_ref.actor.vllm_align_hard_loss_type=${vllm_align_hard_loss_type}
    actor_rollout_ref.actor.vllm_align_hard_delta_min=${vllm_align_hard_delta_min}
    actor_rollout_ref.actor.vllm_align_hard_delta_max=${vllm_align_hard_delta_max}
    actor_rollout_ref.actor.vllm_align_hard_abs_delta_min=${vllm_align_hard_abs_delta_min}
    actor_rollout_ref.actor.vllm_align_hard_abs_delta_max=${vllm_align_hard_abs_delta_max}
    actor_rollout_ref.actor.vllm_align_hard_huber_beta=${vllm_align_hard_huber_beta}
    actor_rollout_ref.actor.vllm_align_hard_normalize_by_window=${vllm_align_hard_normalize_by_window}
    actor_rollout_ref.actor.vllm_align_hard_loss_cap=${vllm_align_hard_loss_cap}
    actor_rollout_ref.actor.delta_mean_guard_enable=${delta_mean_guard_enable}
    actor_rollout_ref.actor.delta_mean_guard_coef=${delta_mean_guard_coef}
    actor_rollout_ref.actor.delta_mean_guard_target=${delta_mean_guard_target}
    actor_rollout_ref.actor.delta_mean_guard_mode=${delta_mean_guard_mode}
    actor_rollout_ref.actor.delta_mean_guard_delta_min=${delta_mean_guard_delta_min}
    actor_rollout_ref.actor.delta_mean_guard_delta_max=${delta_mean_guard_delta_max}
    actor_rollout_ref.actor.delta_mean_guard_adv_min=${delta_mean_guard_adv_min}
    actor_rollout_ref.actor.delta_mean_guard_adv_max=${delta_mean_guard_adv_max}
    actor_rollout_ref.actor.delta_mean_guard_loss_cap=${delta_mean_guard_loss_cap}
    actor_rollout_ref.actor.local_segment_align_enable=${local_segment_align_enable}
    actor_rollout_ref.actor.local_segment_align_coef=${local_segment_align_coef}
    actor_rollout_ref.actor.local_segment_align_target=${local_segment_align_target}
    actor_rollout_ref.actor.local_segment_align_size=${local_segment_align_size}
    actor_rollout_ref.actor.local_segment_align_loss_cap=${local_segment_align_loss_cap}
    actor_rollout_ref.actor.local_segment_align_length_gate_enable=${local_segment_align_length_gate_enable}
    actor_rollout_ref.actor.local_segment_align_length_gate_quantile=${local_segment_align_length_gate_quantile}
    actor_rollout_ref.actor.local_segment_align_length_gate_min=${local_segment_align_length_gate_min}
    actor_rollout_ref.actor.local_segment_align_length_gate_mean_min=${local_segment_align_length_gate_mean_min}
    actor_rollout_ref.actor.local_segment_align_length_gate_clip_ratio_min=${local_segment_align_length_gate_clip_ratio_min}
    actor_rollout_ref.actor.local_segment_align_kl_gate_enable=${local_segment_align_kl_gate_enable}
    actor_rollout_ref.actor.local_segment_align_kl_gate_start=${local_segment_align_kl_gate_start}
    actor_rollout_ref.actor.local_segment_align_kl_gate_full=${local_segment_align_kl_gate_full}
    actor_rollout_ref.actor.local_segment_align_kl_gate_min_factor=${local_segment_align_kl_gate_min_factor}
    actor_rollout_ref.actor.local_segment_align_kl_gate_max_factor=${local_segment_align_kl_gate_max_factor}
    actor_rollout_ref.actor.local_segment_align_adaptive_tail_enable=${local_segment_align_adaptive_tail_enable}
    actor_rollout_ref.actor.local_segment_align_adaptive_tail_offset=${local_segment_align_adaptive_tail_offset}
    actor_rollout_ref.actor.local_segment_align_adaptive_tail_min_start=${local_segment_align_adaptive_tail_min_start}
    actor_rollout_ref.actor.local_segment_align_adaptive_tail_max_start=${local_segment_align_adaptive_tail_max_start}
    actor_rollout_ref.actor.local_segment_align_adaptive_tail_width=${local_segment_align_adaptive_tail_width}
    actor_rollout_ref.actor.local_segment_align_adaptive_tail_mass_min=${local_segment_align_adaptive_tail_mass_min}
    actor_rollout_ref.actor.local_segment_align_adaptive_tail_clip_gate_enable=${local_segment_align_adaptive_tail_clip_gate_enable}
    actor_rollout_ref.actor.local_segment_align_adaptive_tail_clip_ratio_min=${local_segment_align_adaptive_tail_clip_ratio_min}
    actor_rollout_ref.actor.local_segment_align_adaptive_tail_uniform_weight=${local_segment_align_adaptive_tail_uniform_weight}
    actor_rollout_ref.actor.local_segment_align_activation_gate_enable=${local_segment_align_activation_gate_enable}
    actor_rollout_ref.actor.local_segment_align_activation_clip_min=${local_segment_align_activation_clip_min}
    actor_rollout_ref.actor.local_segment_align_activation_clip_period=${local_segment_align_activation_clip_period}
    actor_rollout_ref.actor.local_segment_align_activation_raw_ratio=${local_segment_align_activation_raw_ratio}
    actor_rollout_ref.actor.local_segment_align_activation_raw_min=${local_segment_align_activation_raw_min}
    actor_rollout_ref.actor.local_segment_align_activation_raw_period=${local_segment_align_activation_raw_period}
    actor_rollout_ref.actor.local_segment_align_activation_ema_span=${local_segment_align_activation_ema_span}
    actor_rollout_ref.actor.local_segment_align_activation_baseline_span=${local_segment_align_activation_baseline_span}
    actor_rollout_ref.actor.adv_length_norm_enable=${adv_length_norm_enable}
    actor_rollout_ref.actor.adv_length_norm_mode=${adv_length_norm_mode}
    actor_rollout_ref.actor.adv_length_norm_ref_len=${adv_length_norm_ref_len}
    actor_rollout_ref.actor.adv_length_norm_alpha=${adv_length_norm_alpha}
    actor_rollout_ref.actor.adv_length_norm_min_scale=${adv_length_norm_min_scale}
    actor_rollout_ref.actor.adv_length_norm_max_scale=${adv_length_norm_max_scale}
    actor_rollout_ref.actor.seg_gate_enable=${seg_gate_enable}
    actor_rollout_ref.actor.seg_gate_size=${seg_gate_size}
    actor_rollout_ref.actor.seg_gate_neg_delta_threshold=${seg_gate_neg_delta_threshold}
    actor_rollout_ref.actor.seg_gate_neg_adv_max=${seg_gate_neg_adv_max}
    actor_rollout_ref.actor.seg_gate_neg_weight=${seg_gate_neg_weight}
    actor_rollout_ref.actor.seg_gate_severe_delta_threshold=${seg_gate_severe_delta_threshold}
    actor_rollout_ref.actor.seg_gate_bad_delta_threshold=${seg_gate_bad_delta_threshold}
    actor_rollout_ref.actor.seg_gate_bad_fraction_threshold=${seg_gate_bad_fraction_threshold}
    actor_rollout_ref.actor.seg_gate_severe_weight=${seg_gate_severe_weight}
    actor_rollout_ref.actor.seg_gate_pos_enable=${seg_gate_pos_enable}
    actor_rollout_ref.actor.seg_gate_pos_delta_threshold=${seg_gate_pos_delta_threshold}
    actor_rollout_ref.actor.seg_gate_pos_adv_min=${seg_gate_pos_adv_min}
    ++actor_rollout_ref.actor.megatron.qat.recalib_every=${recalib_every}
)

QAT=(
    actor_rollout_ref.actor.megatron.qat.enable=${qat_enable}
    ++actor_rollout_ref.rollout.qat.enable=${qat_enable}
)
if [ "${qat_enable}" = "True" ]; then
    QAT+=(
        actor_rollout_ref.actor.megatron.qat.mode=${qat_mode}
        actor_rollout_ref.actor.megatron.qat.quantization_config_path="${qat_config_path}"
        "actor_rollout_ref.actor.megatron.qat.ignore_patterns=${qat_ignore_patterns}"
        ++actor_rollout_ref.rollout.qat.mode=${qat_mode}
        ++actor_rollout_ref.rollout.qat.quantization_config_path="${qat_config_path}"
        "++actor_rollout_ref.rollout.qat.ignore_patterns=${qat_ignore_patterns}"
    )
    if [ "${qat_mode}" = "w4a4" ] || [ "${qat_mode}" = "w4a8" ]; then
        QAT+=(++actor_rollout_ref.actor.megatron.qat.calib_data_path="${TRAIN_FILE}")
    fi
fi

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.enforce_eager=${enforce_eager:-True}
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization}
    actor_rollout_ref.rollout.max_model_len=$((max_prompt_length + max_response_length))
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp}
    actor_rollout_ref.rollout.data_parallel_size=${gen_dp}
    actor_rollout_ref.rollout.expert_parallel_size=${gen_ep}
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens}
    actor_rollout_ref.rollout.max_num_seqs=${max_num_seqs}
    actor_rollout_ref.rollout.temperature=${temperature}
    actor_rollout_ref.rollout.top_p=${top_p}
    actor_rollout_ref.rollout.top_k=${top_k}
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p}
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k}
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.n=${n_resp_per_prompt}
)
if [ "${vllm_moe_backend}" != "null" ]; then
    ROLLOUT+=(+actor_rollout_ref.rollout.engine_kwargs.vllm.moe_backend=${vllm_moe_backend})
fi

PERF_OPT=(
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=${moe_router_dtype}
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_enable_deepep=${moe_enable_deepep}
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=${moe_token_dispatcher_type}
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=${moe_permute_fusion}
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_arbitrary_attention_mask=False
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True
)

REWARD=(
    reward.reward_manager.name=dapo
    reward.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer}
    reward.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len}
    reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor}
    reward.reward_kwargs.max_resp_len=${max_response_length}
)

TRAINER=(
    trainer.logger='["console","wandb"]'
    trainer.project_name="${project_name}"
    trainer.experiment_name="${exp_name}"
    trainer.n_gpus_per_node=${n_gpus_per_node}
    trainer.nnodes="${NNODES}"
    trainer.val_before_train=${val_before_train:-False}
    trainer.test_freq=${test_freq:-10}
    trainer.save_freq=${save_freq:-5}
    trainer.total_epochs=1
    trainer.default_local_dir="${CKPTS_DIR}"
    trainer.resume_mode=${resume_mode:-auto}
    ${resume_from_path:+trainer.resume_from_path="${resume_from_path}"}
    trainer.max_actor_ckpt_to_keep=${max_actor_ckpt_to_keep:-2}
    trainer.use_legacy_worker_impl=disable
    ${total_training_steps:+trainer.total_training_steps=${total_training_steps}}
)

FORWARD_ONLY_SETS=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${train_tp}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${train_pp}
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${train_ep}
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${train_etp}
    actor_rollout_ref.ref.megatron.context_parallel_size=${train_cp}
    actor_rollout_ref.ref.megatron.sequence_parallel=False
)

EXTRA_HYDRA_ARGS=()
if [ -n "${extra_hydra_args:-}" ]; then
    # Space-separated Hydra overrides for one-off smoke tests.
    read -r -a EXTRA_HYDRA_ARGS <<< "${extra_hydra_args}"
fi

python3 -m recipe.dapo.main_dapo \
    --config-path "${WORKING_DIR}/recipe/qat/config" \
    --config-name dapo_qat_megatron_trainer \
    "${DATA[@]}" \
    "${ALGORITHM[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${QAT[@]}" \
    "${ROLLOUT[@]}" \
    "${PERF_OPT[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "${FORWARD_ONLY_SETS[@]}" \
    "${EXTRA_HYDRA_ARGS[@]}"
