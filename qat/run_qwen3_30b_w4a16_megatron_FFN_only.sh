#!/usr/bin/env bash
# Qwen3-30B-A3B Megatron W4A16 NVFP4 QAT, FFN/MoE-expert only.
set -euo pipefail
current_dir="$(dirname "$(readlink -f "$0")")"

project_name=${project_name:-"DAPO-NVFP4-QAT"}
exp_name=${exp_name:-"gb200_30B_w4a16_megatron_FFN_$(date +%m%d)"}

qat_enable=True
qat_mode=w4a16
qat_config_path=${qat_config_path:-"${PWD}/recipe/qat/config/nvfp4_w4a16_megatron.json"}
qat_ignore_patterns=${qat_ignore_patterns:-'["lm_head","*mlp.gate","*self_attn*"]'}
export VLLM_NVFP4_GEMM_BACKEND=marlin

# Match the working bf16/w4a4 actors + the original w4a16 baseline: Megatron-native
# all-to-all. The v020 common default later switched to flex+DeepEP, which crashes the
# actor MoE all-to-all at 8 nodes (deep_ep.cpp illegal memory access / ActorDiedError).
moe_enable_deepep=${moe_enable_deepep:-False}
moe_token_dispatcher_type=${moe_token_dispatcher_type:-alltoall}

source "${current_dir}/run_qwen3_30b_megatron_common.sh"
