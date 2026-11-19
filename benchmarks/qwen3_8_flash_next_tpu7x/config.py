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

Parameterizes `(ici_fsdp_parallelism, ici_expert_parallelism)` with `EP == CP`
(`expert_context_parallel=True`) and `local_batch_size = global_batch / FSDP = 2`
across 8K, 64K, 128K, and 256K sequence lengths on 64 chips (128 TensorCores).

Notes:
  * `use_tokamax_linear_ce` defaults to `False` at 8K (where measured A/B showed
    `+15.2 ms / +1.5 GiB` vs plain CE at 2,048 local tokens) and `True` at
    `>= 64K` to bound vocab-logit HBM.
  * `moe_bwd_wgrad_overlap` defaults to `False` at 8K (measured 64-chip TPU7x
    A/B showed `-1.79 GiB` HBM at `+11.9 ms` step-time cost) and 64K
    (extrapolated from 8K, pending a direct 64K HBM/step-time A/B; flip to
    `True` if 64K hits HBM pressure), and `True` at 128K/256K where activation
    HBM is tighter.
  * `tensors_on_device` retains `moe_disp_ep`, `moe_routed_output`, and `qkv_proj`
    at 8K (`2,048` local tokens/TC, `~34.0 GiB` HBM) and drops those three bulky
    tags at `>= 64K` (`16,384` local tokens/TC, saving `~102 GiB/TC` so 64K–256K
    fit comfortably within `95 GiB` HBM per TensorCore).
  * `mu_dtype` remains `float32` by default; `bfloat16` cuts AdamW moment state
    by 25% (`m+v`: 8B -> 6B) and should be enabled only after convergence sign-off.
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

_REMAT_TENSORS_ON_DEVICE_LONG_CONTEXT = (
    "moe_routing",
    "moe_gate_up",
    "query_proj",
    "key_proj",
    "value_proj",
    "context",
    "out_proj",
    "mlpwi_0",
    "mlpwi_1",
    "mlpwo",
)

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
    use_tokamax_linear_ce: bool,
    moe_bwd_wgrad_overlap: bool,
    tensors_on_device: tuple[str, ...] = _REMAT_TENSORS_ON_DEVICE_LONG_CONTEXT,
    index_select_budget: int = 2048,
) -> dict[str, Any]:
  """Builds the shared MaxText override dictionary for a 64-chip TPU7x recipe."""
  return {
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
      "ici_data_parallelism": 1,
      "ici_fsdp_parallelism": fsdp,
      "ici_tensor_parallelism": 1,
      "ici_expert_parallelism": ep,
      "expert_context_parallel": True,
      "sharding_tolerance": 0.05,
      "qsa_attention_kernel": "flash",
      "index_select_budget": index_select_budget,
      "capacity_factor": 1.25,
      "gdn_kernel_backend": "tokamax",
      "use_tokamax_gmm": True,
      "use_tokamax_linear_ce": use_tokamax_linear_ce,
      "moe_bwd_wgrad_overlap": moe_bwd_wgrad_overlap,
      "remat_policy": "custom",
      "tensors_on_device": list(tensors_on_device),
  }


QWEN3_8_64C_8K = Qwen38Tpu7xRecipe(
    name="qwen3_8_flash_next_64c_8k",
    topology="4x4x4",
    maxtext_flags=_base_flags(
        max_target_length=8192,
        per_device_batch_size=0.25,
        fsdp=16,
        ep=8,
        use_tokamax_linear_ce=False,
        moe_bwd_wgrad_overlap=False,
        tensors_on_device=_REMAT_TENSORS_ON_DEVICE_8K,
    ),
)

QWEN3_8_64C_64K = Qwen38Tpu7xRecipe(
    name="qwen3_8_flash_next_64c_64k",
    topology="4x4x4",
    maxtext_flags=_base_flags(
        max_target_length=65536,
        per_device_batch_size=0.25,
        fsdp=16,
        ep=8,
        use_tokamax_linear_ce=True,
        moe_bwd_wgrad_overlap=False,
    ),
)

QWEN3_8_64C_128K = Qwen38Tpu7xRecipe(
    name="qwen3_8_flash_next_64c_128k",
    topology="4x4x4",
    maxtext_flags=_base_flags(
        max_target_length=131072,
        per_device_batch_size=0.125,
        fsdp=8,
        ep=16,
        use_tokamax_linear_ce=True,
        moe_bwd_wgrad_overlap=True,
    ),
)

QWEN3_8_64C_256K = Qwen38Tpu7xRecipe(
    name="qwen3_8_flash_next_64c_256k",
    topology="4x4x4",
    maxtext_flags=_base_flags(
        max_target_length=262144,
        per_device_batch_size=0.0625,
        fsdp=4,
        ep=32,
        use_tokamax_linear_ce=True,
        moe_bwd_wgrad_overlap=True,
    ),
)

RECIPES: dict[str, Qwen38Tpu7xRecipe] = {
    "8k": QWEN3_8_64C_8K,
    "64k": QWEN3_8_64C_64K,
    "128k": QWEN3_8_64C_128K,
    "256k": QWEN3_8_64C_256K,
}
