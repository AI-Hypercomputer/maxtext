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

"""Qwen3.8-Flash-Next (180B-A6B) 64-chip TPU7x (`tpu7x:4x4x4`) training recipes.

Parameterizes `(ici_fsdp_parallelism=16, ici_expert_parallelism=8)` across 64 chips
(128 TensorCores) for the three hardware-verified training recipes (`"8k"`, `"8k_2m"`,
and `"64k"`):

  * `"8k"` (`max_target_length=8192, per_device_batch_size=0.25`, 256K global tokens,
    `sharding_preset="ep_as_cp"`, `compound_fsdp_ep=True`): sets `index_select_budget=8192`
    so `s <= index_select_budget` routes 8K attention directly through dense Tokamax
    Splash Attention (`qsa_attention_kernel="flash"`), bypassing the QSA indexer at 8K
    (note: the HuggingFace model-card default is `index_select_budget=2048`; pass
    `index_select_budget=2048` explicitly to evaluate 2K sparse block selection at 8K).
  * `"8k_2m"` (`max_target_length=8192, per_device_batch_size=2.0`, 2M global tokens,
    `sharding_preset="default"`, `compound_fsdp_ep=True`): keeps tokens batch-sharded
    across `FSDP x EP = 128` (`16,384` local tokens/TC) with zero context-parallel
    communication (`qsa_attention_kernel="flash"`, `index_select_budget=8192`,
    `use_tokamax_linear_ce=True`).
  * `"64k"` (`max_target_length=65536, per_device_batch_size=0.25`, 2M global tokens,
    `sharding_preset="ep_as_cp"`, `compound_fsdp_ep=True`): dispatches
    `qsa_attention_kernel="pallas"` (`qsa_pallas_bkv=2048, qsa_pallas_bkv_dkv=2048`,
    `index_select_budget=2048`) with Zig-Zag paired CP load balancing
    (`sa_ignore_segment_ids=True, packing=False`), `use_tokamax_linear_ce=True`, and
    `_REMAT_TENSORS_ON_DEVICE_64K_TC_QTR`.
"""

import dataclasses
from typing import Any

from benchmarks.qwen3_8_flash_next_tpu7x import xla_flags

_REMAT_TENSORS_ON_DEVICE_8K = (
    "moe_routing",
    "moe_disp_ep",
    "moe_gate_up",
    "moe_routed_output",
    "qkv_proj",
    "query_proj",
    "key_proj",
    "value_proj",
    "context",
    "out_proj",
    "mlpwi_0",
    "mlpwi_1",
    "mlpwo",
)

# Used on 2M-token `moe_ep_allgather_tc=True` recipes (`8k_2m` and `64k`), where
# `moe_routed_output` tags only the `[b_loc, s_loc, D]` token output (`+3.75 GiB`
# across 48 layers) and eliminates `down_proj` GMM + `unsort` + EP `reduce-scatter` from remat.
_REMAT_TENSORS_ON_DEVICE_2M_TC = (
    "moe_routing",
    "moe_routed_output",
    "query_proj",
    "key_proj",
    "value_proj",
    "context",
    "out_proj",
    "mlpwi_0",
    "mlpwi_1",
    "mlpwo",
)

# Selective-layer HBM variants saving FP8 `sorted_qval` on 12 (`moe_disp_ep_qtr`) or 24
# (`moe_disp_ep_half`) of 48 MoE layers. Default `64k` recipe uses `_REMAT_TENSORS_ON_DEVICE_64K_TC_QTR`
# (`80.92 GB <= 82.12 GB` `LatencyHidingScheduler` limit on TPU7x).
_REMAT_TENSORS_ON_DEVICE_64K_TC_QTR = _REMAT_TENSORS_ON_DEVICE_2M_TC + ("moe_disp_ep_qtr",)
_REMAT_TENSORS_ON_DEVICE_64K_TC_HALF = _REMAT_TENSORS_ON_DEVICE_2M_TC + ("moe_disp_ep_half",)

_DEFAULT_REMAT_TENSORS_ON_DEVICE = _REMAT_TENSORS_ON_DEVICE_8K


@dataclasses.dataclass(frozen=True)
class Qwen38Tpu7xRecipe:
  """Tuned MaxText overrides and XLA flags for Qwen3.8-Flash-Next on TPU7x."""

  name: str
  topology: str
  maxtext_flags: dict[str, Any]
  xla_flags: str = xla_flags.BASE_FLAGS


def _base_flags(
    *,
    max_target_length: int,
    per_device_batch_size: float,
    fsdp: int,
    ep: int,
    sharding_preset: str,
    use_tokamax_linear_ce: bool,
    moe_bwd_wgrad_overlap: bool = False,
    moe_ep_allgather_tc: bool = True,
    moe_tc_ragged_flatten_block_size: int = 512,
    moe_ep_combine_mode: str = "auto",
    moe_row_scale_placement: str = "tc",
    moe_tc_3d_gmm: bool = False,
    moe_fused_bwd_quant: bool = True,
    use_hc_pallas: bool = True,
    use_gdn_conv_pallas: bool = True,
    qsa_attention_kernel: str = "flash",
    qsa_indexer_approx_topk: bool = False,
    qsa_pallas_bkv: int = 0,
    qsa_pallas_bkv_dkv: int = 0,
    tensors_on_device: tuple[str, ...] = _REMAT_TENSORS_ON_DEVICE_8K,
    tensors_to_offload: tuple[str, ...] = (),
    index_select_budget: int = 2048,
) -> dict[str, Any]:
  """Builds the shared MaxText override dictionary for a 64-chip TPU7x recipe."""
  flags_dict: dict[str, Any] = {
      "model_name": "qwen3.8-flash-next",
      "override_model_config": True,
      "use_m3_model": True,
      "scan_layers": True,
      "dtype": "bfloat16",
      "weight_dtype": "bfloat16",
      "mu_dtype": "float32",
      "grad_dtype": "float32",
      "per_device_batch_size": per_device_batch_size,
      "max_target_length": max_target_length,
      "packing": False,
      "ici_data_parallelism": 1,
      "ici_fsdp_parallelism": fsdp,
      "ici_tensor_parallelism": 1,
      "ici_expert_parallelism": ep,
      "sharding_preset": sharding_preset,
      "custom_mesh": "tpu7x_torus_yx_zcore",
      "compound_fsdp_ep": True,
      "sharding_tolerance": 0.05,
      "quantization": "fp8_full",
      "use_qwix_quantization": True,
      "quantize_router_proj": False,
      "moe_fp8": True,
      "moe_ep_allgather_tc": moe_ep_allgather_tc,
      "moe_tc_ragged_flatten_block_size": moe_tc_ragged_flatten_block_size,
      "ragged_buffer_factor": 1.25,
      "moe_ep_combine_mode": moe_ep_combine_mode,
      "moe_row_scale_placement": moe_row_scale_placement,
      "moe_tc_3d_gmm": moe_tc_3d_gmm,
      "moe_fused_bwd_quant": moe_fused_bwd_quant,
      "use_hc_pallas": use_hc_pallas,
      "use_gdn_conv_pallas": use_gdn_conv_pallas,
      "qsa_attention_kernel": qsa_attention_kernel,
      "qsa_pallas_bkv": qsa_pallas_bkv,
      "qsa_pallas_bkv_dkv": qsa_pallas_bkv_dkv,
      "qsa_indexer_approx_topk": qsa_indexer_approx_topk,
      "sa_ignore_segment_ids": True,
      "index_select_budget": index_select_budget,
      "capacity_factor": -1.0,
      "gdn_kernel_backend": "tokamax",
      "use_tokamax_gmm": True,
      "use_tokamax_linear_ce": use_tokamax_linear_ce,
      "moe_bwd_wgrad_overlap": moe_bwd_wgrad_overlap,
      "remat_policy": "custom",
      "tensors_on_device": list(tensors_on_device),
  }
  if tensors_to_offload:
    flags_dict["tensors_to_offload"] = list(tensors_to_offload)
  return flags_dict


QWEN3_8_64C_8K = Qwen38Tpu7xRecipe(
    name="qwen3_8_flash_next_64c_8k",
    topology="4x4x4",
    maxtext_flags=_base_flags(
        max_target_length=8192,
        per_device_batch_size=0.25,
        fsdp=16,
        ep=8,
        sharding_preset="ep_as_cp",
        use_tokamax_linear_ce=False,
        qsa_attention_kernel="flash",
        tensors_on_device=_REMAT_TENSORS_ON_DEVICE_8K,
        index_select_budget=8192,
    ),
)

QWEN3_8_64C_8K_2M = Qwen38Tpu7xRecipe(
    name="qwen3_8_flash_next_64c_8k_2m",
    topology="4x4x4",
    maxtext_flags=_base_flags(
        max_target_length=8192,
        per_device_batch_size=2.0,
        fsdp=16,
        ep=8,
        sharding_preset="default",
        use_tokamax_linear_ce=True,
        qsa_attention_kernel="flash",
        tensors_on_device=_REMAT_TENSORS_ON_DEVICE_2M_TC,
        index_select_budget=8192,
    ),
    xla_flags=xla_flags.HIGH_TOKEN_FLAGS,
)

QWEN3_8_64C_64K = Qwen38Tpu7xRecipe(
    name="qwen3_8_flash_next_64c_64k",
    topology="4x4x4",
    maxtext_flags=_base_flags(
        max_target_length=65536,
        per_device_batch_size=0.25,
        fsdp=16,
        ep=8,
        sharding_preset="ep_as_cp",
        use_tokamax_linear_ce=True,
        qsa_attention_kernel="pallas",
        qsa_pallas_bkv=2048,
        qsa_pallas_bkv_dkv=2048,
        tensors_on_device=_REMAT_TENSORS_ON_DEVICE_64K_TC_QTR,
        index_select_budget=2048,
    ),
    xla_flags=xla_flags.HIGH_TOKEN_FLAGS,
)

RECIPES: dict[str, Qwen38Tpu7xRecipe] = {
    "8k": QWEN3_8_64C_8K,
    "8k_2m": QWEN3_8_64C_8K_2M,
    "64k": QWEN3_8_64C_64K,
}
