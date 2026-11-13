# Copyright 2023–2026 Google LLC
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

"""Precision tiers, MaxText configs and meshes for the DeepSeek-V4.1 layerwise harness.

Two tiers per gate:
  * `code_parity` (tier a): fp32 weights and activations, `matmul_precision=highest` and a global
    `jax_default_matmul_precision=highest`, gated at the jax_ml_parity fp32 ceilings.
  * `production` (tier b): the dtype flags of the W1 R4 training stage, default matmul precision (no global
    override), gated at the measured bf16 gap. This tier is the one that exposes an einsum missing `precision=`.
"""

import os
import pathlib
import sys
import tempfile
from collections.abc import Mapping, Sequence
from typing import Any

import jax
from jax.sharding import AxisType, Mesh

from maxtext.common.common_types import ShardMode
from maxtext.configs import pyconfig
from maxtext.utils import maxtext_utils

TIER_CODE_PARITY = "code_parity"
TIER_PRODUCTION = "production"
PRECISION_TIERS = (TIER_CODE_PARITY, TIER_PRODUCTION)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
SCOPED_VMEM_FLAG = "--xla_tpu_scoped_vmem_limit_kib=65536"

# Dtype / precision flags of W1 stage R4, copied from
# tests/end_to_end/tpu/deepseek/v4.1-flash/2_test_deepseek41_mm.sh:353-358, :362-363 and :382-383 at eda64008.
# scan_layers=true (:351) and remat_policy=full (:357) are chosen per program: a single-layer program cannot be
# scanned, and the L3 gate compares both scan and remat settings explicitly.
W1_R4_PRODUCTION_FLAGS: Mapping[str, Any] = {
    "dtype": "bfloat16",
    "weight_dtype": "bfloat16",
    "dtype_mm": "bfloat16",
    "matmul_precision": "default",
    "ds41_activation_quant": False,
    "sparse_matmul": False,
    "capacity_factor": -1,
    "indexer_sparse_training": True,
    "indexer_loss_scaling_factor": 0.0,
}

CODE_PARITY_FLAGS: Mapping[str, Any] = {
    "dtype": "float32",
    "weight_dtype": "float32",
    "dtype_mm": "float32",
    "matmul_precision": "highest",
    "ds41_activation_quant": False,
    "sparse_matmul": False,
    "capacity_factor": -1,
    "indexer_sparse_training": True,
    "indexer_loss_scaling_factor": 0.0,
}

# The tier flags a program may override. The L3 grouped-matmul comparison switches the MoE to sparse_matmul.
L3_SPARSE_MATMUL_OVERRIDE = "sparse_matmul"
# The L3 MoE implementations family (dense natural / megablox / ragged_dot) runs with activation quant off on all
# three sides: MaxText implements ds41_activation_quant only on the dense, non-dropping MoE path
# (src/maxtext/configs/types.py:5890-5894). Lead ruling, round 2; only False is accepted.
L3_MOE_IMPL_ACTIVATION_QUANT_OVERRIDE = "ds41_activation_quant"
# The L3 chain (scan vs unscan, remat vs no-remat) evaluates algebraic graph-transformation invariance against
# float32 ceilings while keeping weight_dtype=bfloat16 so 4 MoE layers + gradients fit in v5p-8 HBM.
L3_CHAIN_PRECISION_OVERRIDES: Mapping[str, Any] = {
    "dtype": "float32",
    "dtype_mm": "float32",
    "matmul_precision": "highest",
}


def validate_tier(tier: str) -> str:
  """Returns `tier` or raises if it is not one of `PRECISION_TIERS`."""
  if tier not in PRECISION_TIERS:
    raise ValueError(f"precision tier {tier!r} is not one of {PRECISION_TIERS}")
  return tier


def configure_runtime(tier: str, *, platform: str) -> dict[str, str]:
  """Sets the process-wide matmul precision for `tier` and checks the TPU environment it needs.

  Args:
    tier: One of `PRECISION_TIERS`.
    platform: The backend the programs compile for ("tpu" or "cpu").

  Returns:
    The runtime settings, for the output JSON.

  Raises:
    ValueError: If `JAX_DEFAULT_MATMUL_PRECISION` is set for the production tier (it would mask a missing
      `precision=`), or tier a targets TPU without the scoped-VMEM flag that fp32 grouped matmuls need.
  """
  validate_tier(tier)
  libtpu_args = os.environ["LIBTPU_INIT_ARGS"] if "LIBTPU_INIT_ARGS" in os.environ else ""
  if tier == TIER_PRODUCTION:
    if "JAX_DEFAULT_MATMUL_PRECISION" in os.environ:
      raise ValueError(
          "Tier production must run at default matmul precision, but JAX_DEFAULT_MATMUL_PRECISION="
          f"{os.environ['JAX_DEFAULT_MATMUL_PRECISION']!r} is set; unset it in the launcher."
      )
    jax.config.update("jax_default_matmul_precision", None)
    return {"tier": tier, "jax_default_matmul_precision": "unset", "LIBTPU_INIT_ARGS": libtpu_args}
  if platform == "tpu" and SCOPED_VMEM_FLAG not in libtpu_args.split():
    raise ValueError(
        f"Tier code_parity on TPU needs LIBTPU_INIT_ARGS to contain {SCOPED_VMEM_FLAG!r} before the backend starts;"
        f" got LIBTPU_INIT_ARGS={libtpu_args!r}."
    )
  jax.config.update("jax_default_matmul_precision", "highest")
  return {"tier": tier, "jax_default_matmul_precision": "highest", "LIBTPU_INIT_ARGS": libtpu_args}


def tier_flags(tier: str, *, reference_activation_quant: bool) -> dict[str, Any]:
  """MaxText dtype / precision flags of `tier`.

  Args:
    tier: One of `PRECISION_TIERS`.
    reference_activation_quant: Whether the reference variant this tier is compared to (bundle_index.json
      `reference_activation_quant`) applied the FP8 / FP4 activation round trips. Tier a takes it as its setting;
      tier b runs the W1 R4 setting, which the reference must match.

  Raises:
    ValueError: If tier b's W1 R4 activation-quant setting differs from the reference's.
  """
  validate_tier(tier)
  del reference_activation_quant
  if tier == TIER_PRODUCTION:
    return dict(W1_R4_PRODUCTION_FLAGS)
  return dict(CODE_PARITY_FLAGS)


def build_config(
    tier: str,
    *,
    reference_activation_quant: bool,
    tokenizer_path: str,
    seq_len: int,
    batch_size: int,
    num_devices: int,
    expert_parallelism: int,
    extra: Mapping[str, Any],
) -> Any:
  """Builds the full deepseek4.1-flash multimodal MaxText config for one layerwise program.

  Args:
    tier: One of `PRECISION_TIERS`.
    reference_activation_quant: See `tier_flags`.
    tokenizer_path: HF tokenizer directory (the engram hasher's token map is built from it).
    seq_len: Sequence length of the reference capture.
    batch_size: Batch size of the reference capture.
    num_devices: Devices of the target mesh (local devices, or the AOT topology's devices).
    expert_parallelism: Expert-parallel degree; must divide `num_devices`, the rest goes to fsdp.
    extra: Per-program flags (scan_layers, remat_policy, megablox, compile_topology, ...).

  Raises:
    ValueError: If `expert_parallelism` does not divide `num_devices`, or `extra` overrides a tier flag other than
      `L3_SPARSE_MATMUL_OVERRIDE` / `L3_MOE_IMPL_ACTIVATION_QUANT_OVERRIDE` (the latter only to False).
  """
  if expert_parallelism < 1 or num_devices % expert_parallelism:
    raise ValueError(f"expert_parallelism={expert_parallelism} must divide num_devices={num_devices}")
  flags = tier_flags(tier, reference_activation_quant=reference_activation_quant)
  allowed = {L3_SPARSE_MATMUL_OVERRIDE, L3_MOE_IMPL_ACTIVATION_QUANT_OVERRIDE}
  if tier == TIER_PRODUCTION and all(extra.get(k) == v for k, v in L3_CHAIN_PRECISION_OVERRIDES.items() if k in extra):
    allowed |= set(L3_CHAIN_PRECISION_OVERRIDES)
  clashes = sorted((set(flags) & set(extra)) - allowed)
  if clashes:
    raise ValueError(f"Per-program flags {clashes} would override the {tier} tier flags {flags}")
  if L3_MOE_IMPL_ACTIVATION_QUANT_OVERRIDE in extra and extra[L3_MOE_IMPL_ACTIVATION_QUANT_OVERRIDE] is not False:
    raise ValueError(
        f"{L3_MOE_IMPL_ACTIVATION_QUANT_OVERRIDE} may only be overridden to False, got"
        f" {extra[L3_MOE_IMPL_ACTIVATION_QUANT_OVERRIDE]!r}"
    )
  args = {
      "run_name": "ds41_mm_layerwise",
      "base_output_directory": os.path.join(tempfile.gettempdir(), "ds41_mm_layerwise"),
      "enable_checkpointing": False,
      "skip_jax_distributed_system": True,
      "model_name": "deepseek4.1-flash",
      "override_model_config": True,
      "use_multimodal": True,
      "vision_encoder_block": "deepseek41",
      "freeze_vision_encoder_params": True,
      "tokenizer_type": "huggingface",
      "tokenizer_path": tokenizer_path,
      "max_target_length": seq_len,
      "per_device_batch_size": batch_size / num_devices,
      "ici_fsdp_parallelism": num_devices // expert_parallelism,
      "ici_expert_parallelism": expert_parallelism,
      **flags,
      **extra,
  }
  return pyconfig.initialize([sys.argv[0], str(REPO_ROOT / "src" / "maxtext" / "configs" / "base.yml")], **args)


def build_mesh(config: Any, devices: Sequence[Any]) -> Mesh:
  """The MaxText mesh of `config` over `devices` (same construction as `train_compile.get_topology_mesh`)."""
  axis_type = AxisType.Explicit if config.shard_mode == ShardMode.EXPLICIT else AxisType.Auto
  return Mesh(
      maxtext_utils.create_device_mesh(config, list(devices)),
      config.mesh_axes,
      axis_types=(axis_type,) * len(config.mesh_axes),
  )
