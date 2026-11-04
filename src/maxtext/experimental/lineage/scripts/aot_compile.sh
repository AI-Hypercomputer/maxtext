#!/bin/bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# AOT-compiles the Lineage train step without TPUs (needs jax + libtpu,
# tokamax<0.1, typeguard 2.x and tensorflow for the HLO proto), dumping the
# optimized HLO to $OUT/xla. Pass the model dimensions, topology and
# parallelism as MaxText key=value overrides, e.g. for a GQA model:
#
#   OUT=/tmp/aot bash src/maxtext/experimental/lineage/scripts/aot_compile.sh \
#     model_name=deepseek3-671b-lineage override_model_config=True \
#     compile_topology=<topology> compile_topology_num_slices=<n> \
#     "ici_parallelism=[...]" "dcn_parallelism=[...]" \
#     base_emb_dim=... base_num_query_heads=... base_num_kv_heads=... \
#     q_lora_rank=0 kv_lora_rank=0 qk_nope_head_dim=0 \
#     lineage_sharding_variant=cp4_gqa_a2a lineage_quantization=fp8_fsdp_ag
set -euo pipefail
OUT="${OUT:-/tmp/lineage_aot}"
mkdir -p "${OUT}/xla"
export XLA_FLAGS="--xla_dump_to=${OUT}/xla --xla_dump_hlo_module_re=jit_train_step ${EXTRA_XLA_FLAGS:-}"
export PYTHONPATH="src${PYTHONPATH:+:${PYTHONPATH}}"
python3 -m maxtext.experimental.lineage.dsv3_aot_compile src/maxtext/configs/base.yml use_lineage=True log_config=False "$@"
