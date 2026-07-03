#!/usr/bin/env bash
# Qwen3-30B-A3B Megatron BF16 baseline.
set -euo pipefail
current_dir="$(dirname "$(readlink -f "$0")")"

project_name=${project_name:-"DAPO-NVFP4-QAT"}
exp_name=${exp_name:-"gb200_30B_bf16_megatron_$(date +%m%d)"}

qat_enable=False

# --- bf16 30B fix (2026-06-05) ---------------------------------------------
# v020 common defaults (moe_token_dispatcher_type=flex + moe_enable_deepep=True)
# crash in DeepEP (deep_ep.cpp:155 'illegal memory access') at the step-0
# compute_log_prob MoE all-to-all for the bf16 actor. bf16 30B weights are ~2x
# w4a16's and squeeze DeepEP's NVSHMEM symmetric heap under colocate. Fall back to
# the Megatron-native all-to-all that every pre-v020 run used (incl. the working
# 30B smoke 2182587). Scoped to bf16 only; w4a16 stays on flex+DeepEP (healthy).
moe_enable_deepep=${moe_enable_deepep:-False}
moe_token_dispatcher_type=${moe_token_dispatcher_type:-alltoall}

source "${current_dir}/run_qwen3_30b_megatron_common.sh"
