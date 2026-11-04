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

"""Fast standalone AOT compiler for the Lineage DeepSeek-V3 train step.

Usage (from the repo root, with `src` on PYTHONPATH):
  python -m maxtext.experimental.lineage.dsv3_aot_compile \
      src/maxtext/configs/base.yml model_name=deepseek3-671b-lineage \
      compile_topology=tpu7x:4x8x8 ...

Replaces `maxtext.trainers.pre_train.train_compile` for Lineage DSv3 AOT
compilation verification while preserving identical HLO,
parameter/optimizer shardings, and memory analysis output:
  1. Disables runtime `jaxtyping` + `typeguard` checks during lowering.
  2. Loads MaxText YAML + CLI configs directly with PyYAML without importing
     PyTorch, vLLM, Tunix, Orbax, HuggingFace Transformers, or OmegaConf.
  3. Constructs abstract sharded parameters and optimizer state in a single
     `nnx.eval_shape` pass instead of 3 redundant evaluation passes.
"""

import ast
import collections
from collections.abc import Mapping, Sequence
import dataclasses
import functools
import math
import os
import re
from typing import Any

# Disable runtime jaxtyping + typeguard shape/dtype checking during JAX tracing.
# Must be set before importing jaxtyping or any Lineage module.
os.environ.setdefault("JAXTYPING_DISABLE", "1")

# pylint: disable=g-import-not-at-top
from absl import app
from absl import logging
from flax import nnx
from flax import traverse_util
from flax.core import spmd
import jax
from jax.experimental import topologies
import jax.numpy as jnp
import numpy as np
import optax
import yaml

from maxtext.experimental.lineage import dsv3_config
from maxtext.experimental.lineage import dsv3_loss
from maxtext.experimental.lineage import dsv3_model
from maxtext.experimental.lineage import dsv3_sharding
from maxtext.experimental.lineage import dsv3_types

# pylint: enable=g-import-not-at-top

P = jax.sharding.PartitionSpec
NamedSharding = jax.sharding.NamedSharding

_RAGGED_BUFFER_FACTOR = 1.2
# MaxText's config directory: src/maxtext/configs, relative to this file.
_CONFIGS_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "configs"))
_DEFAULT_BASE_YML = os.path.join(_CONFIGS_DIR, "base.yml")
_MODELS_YML_DIR = os.path.join(_CONFIGS_DIR, "models")


class AOTConfig:
  """Read-only configuration wrapper compatible with MaxText pyconfig."""

  def __init__(self, config_values: Mapping[str, Any]):
    """Initializes the instance."""
    object.__setattr__(self, "_data", dict(config_values))

  def __getattr__(self, name: str) -> Any:
    config_values = object.__getattribute__(self, "_data")
    if name in config_values:
      return config_values[name]
    raise AttributeError(f"AOTConfig has no attribute {name!r}")

  def __setattr__(self, name: str, value: Any) -> None:
    raise AttributeError("AOTConfig is read-only")

  def get_keys(self) -> dict[str, Any]:
    """Returns the underlying configuration dictionary."""
    return object.__getattribute__(self, "_data")


def _resolve_config_path(raw_path: str) -> str:
  """Resolves a MaxText YAML config path relative to CWD or the repo."""
  repo_root = os.path.dirname(os.path.dirname(_CONFIGS_DIR))
  candidates = [
      raw_path,
      os.path.join(repo_root, raw_path),
      os.path.join(_CONFIGS_DIR, raw_path),
      os.path.join(_CONFIGS_DIR, os.path.basename(raw_path)),
  ]
  for candidate in candidates:
    if os.path.isfile(candidate):
      return candidate
  raise FileNotFoundError(f"Could not find config file: {raw_path!r}")


def _load_yaml_with_base(path: str) -> dict[str, Any]:
  """Loads a YAML file and recursively merges `base_config` if present."""
  with open(path, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f) or {}
  base_name = data.get("base_config")
  if not base_name:
    return data
  base_candidates = [
      os.path.join(os.path.dirname(path), base_name),
      os.path.join(os.path.dirname(os.path.dirname(path)), base_name),
  ]
  base_path = None
  for candidate in base_candidates:
    if os.path.isfile(candidate):
      base_path = candidate
      break
  if base_path is None:
    base_path = _resolve_config_path(base_name)
  parent = _load_yaml_with_base(base_path)
  return {**parent, **data}


def _apply_logical_axis_rules(
    base_rules: list[Any],
    new_rules: list[Any],
    cfg_fragment: Mapping[str, Any],
) -> list[Any]:
  """Merges `logical_axis_rules` following MaxText `pyconfig._apply_rules`."""
  if cfg_fragment.get("override_logical_axis_rules"):
    return list(new_rules)
  if not new_rules:
    return list(base_rules)
  overridden_keys = {rule[0] for rule in new_rules}
  merged = [rule for rule in base_rules if rule[0] not in overridden_keys]
  merged.extend(new_rules)
  return merged


def _coerce_to_list(val: Any) -> list[Any]:
  """Coerces string or sequence config values into a list."""
  if val is None:
    return []
  if isinstance(val, str):
    stripped = val.strip()
    if stripped.startswith(("[", "(")):
      return list(ast.literal_eval(stripped))
    return [val] if val else []
  if isinstance(val, (list, tuple)):
    return list(val)
  return [val]


def _lists_to_tuples(val: Any) -> Any:
  """Recursively converts nested lists into tuples."""
  if isinstance(val, list):
    return tuple(_lists_to_tuples(item) for item in val)
  if isinstance(val, tuple):
    return tuple(_lists_to_tuples(item) for item in val)
  return val


def _get_individual_scales(scale: int) -> tuple[int, int, int, int]:
  """Splits `global_parameter_scale` into dimension scales (`types.py`)."""
  if scale <= 0:
    raise ValueError(f"global_parameter_scale must be > 0, got {scale}")
  log_2_scale = int(math.log2(scale))
  if 2**log_2_scale != scale:
    raise ValueError(f"global_parameter_scale must be a power of two, got {scale}")
  base_scale, rem = divmod(log_2_scale, 3)
  num_head_scale = base_scale + int(rem > 0)
  mlp_dim_scale = num_head_scale
  emb_scale = base_scale + int(rem > 1)
  layer_scale = base_scale
  return emb_scale, num_head_scale, mlp_dim_scale, layer_scale


def _resolve_parallelism(
    configured: Any,
    axis_map: Mapping[str, Any],
    mesh_axes: Sequence[str],
) -> list[int]:
  """Resolves `ici_parallelism` / `dcn_parallelism` along `mesh_axes`."""
  if configured and len(configured) == len(mesh_axes) and any(axis not in axis_map for axis in mesh_axes):
    return [int(v) for v in configured]
  return [int(axis_map.get(axis, 1)) for axis in mesh_axes]


def _compute_batch_sizes(
    per_device_bs: float,
    num_devices: int,
    grad_accum_steps: int,
    expansion_factor: float,
) -> tuple[int, int, int]:
  """Computes global load/train and micro batch sizes (`types.py`)."""
  factor = expansion_factor if expansion_factor > 0 else 1.0
  if per_device_bs < 1.0:
    micro_batch_to_load = int(num_devices * factor)
  else:
    micro_batch_to_load = int(num_devices * per_device_bs * factor)
  micro_batch_to_train = int(num_devices * per_device_bs)
  return (
      int(micro_batch_to_load * grad_accum_steps),
      int(micro_batch_to_train * grad_accum_steps),
      micro_batch_to_train,
  )


def _parse_cli_value(val_str: str) -> Any:
  """Parses a CLI `key=value` string value using YAML semantics."""
  if not val_str:
    return ""
  val = yaml.safe_load(val_str)
  if isinstance(val, str) and val.lower() == "none":
    return None
  return val


def load_aot_config(argv: Sequence[str]) -> AOTConfig:
  """Loads and resolves MaxText YAML + CLI config for Lineage AOT compilation.

  Args:
    argv: Command-line arguments (`[prog, base.yml, key=val, ...]`).

  Returns:
    Read-only `AOTConfig` with derived model, mesh, and batch fields populated.
  """
  if len(argv) >= 2 and argv[1].lower().endswith((".yml", ".yaml")):
    config_path = _resolve_config_path(argv[1])
    raw_cli_args = argv[2:]
  else:
    config_path = _resolve_config_path(_DEFAULT_BASE_YML)
    raw_cli_args = argv[1:]

  base_cfg = _load_yaml_with_base(config_path)

  cli_cfg: dict[str, Any] = {}
  for arg in raw_cli_args:
    if "=" not in arg:
      raise ValueError(f"Expected key=value CLI argument, got {arg!r}")
    key, val_str = arg.split("=", 1)
    cli_cfg[key] = _parse_cli_value(val_str)

  model_name = str(cli_cfg.get("model_name", base_cfg.get("model_name", "default")))
  lookup_model_name = model_name.removesuffix("-Instruct")
  model_cfg: dict[str, Any] = {}
  if lookup_model_name != "default":
    local_model_yml = os.path.join(os.path.dirname(config_path), "models", f"{lookup_model_name}.yml")
    if os.path.isfile(local_model_yml):
      model_yml_path = local_model_yml
    else:
      model_yml_path = _resolve_config_path(os.path.join(_MODELS_YML_DIR, f"{lookup_model_name}.yml"))
    model_cfg = _load_yaml_with_base(model_yml_path)

  override_model_config = bool(
      cli_cfg.get(
          "override_model_config",
          base_cfg.get("override_model_config", False),
      )
  )
  if override_model_config:
    model_cfg = {k: v for k, v in model_cfg.items() if k not in cli_cfg}
  else:
    for k, v in model_cfg.items():
      if k in cli_cfg and k != "logical_axis_rules" and cli_cfg[k] != v:
        raise ValueError(
            f"CLI key {k!r}={cli_cfg[k]!r} conflicts with model config {v!r}" " without override_model_config=True."
        )

  rules = list(base_cfg.get("logical_axis_rules", []))
  rules = _apply_logical_axis_rules(rules, model_cfg.get("logical_axis_rules", []), model_cfg)
  rules = _apply_logical_axis_rules(rules, cli_cfg.get("logical_axis_rules", []), cli_cfg)

  merged: dict[str, Any] = {**base_cfg, **model_cfg, **cli_cfg}
  merged["logical_axis_rules"] = rules

  # Apply M_<KEY> environment variable overrides.
  lower_to_key = {k.lower(): k for k in merged}
  for env_k, env_v in os.environ.items():
    if env_k.startswith("M_"):
      target_k = lower_to_key.get(env_k[2:].lower())
      if target_k is not None:
        merged[target_k] = _parse_cli_value(env_v)

  # Normalize "none" strings to None (except lineage_quantization).
  for k, v in list(merged.items()):
    if isinstance(v, str) and v.lower() == "none":
      merged[k] = None
  raw_quant = merged.get("lineage_quantization")
  merged["lineage_quantization"] = "none" if raw_quant is None else str(raw_quant).lower()

  for list_key in (
      "trainable_parameters_mask",
      "adamw_mask",
      "train_data_columns",
      "eval_data_columns",
  ):
    if list_key in merged:
      merged[list_key] = _coerce_to_list(merged[list_key])

  if int(merged.get("learning_rate_schedule_steps", -1)) == -1:
    merged["learning_rate_schedule_steps"] = int(merged.get("steps", -1))
  if int(merged.get("steps", -1)) == -1:
    merged["steps"] = int(merged.get("learning_rate_schedule_steps", -1))

  if float(merged.get("eval_per_device_batch_size", 0.0) or 0.0) == 0.0:
    merged["eval_per_device_batch_size"] = float(merged["per_device_batch_size"])

  if merged.get("attn_logits_soft_cap") == 0.0:
    merged["attn_logits_soft_cap"] = None
  if merged.get("final_logits_soft_cap") == 0.0:
    merged["final_logits_soft_cap"] = None

  # Set XLA_FLAGS for HLO dump if requested and not already set.
  if not os.environ.get("XLA_FLAGS") and not merged.get("dump_hlo_xla_flags") and merged.get("dump_hlo_local_dir"):
    xla_flags = f"--xla_dump_to={merged['dump_hlo_local_dir']}" " --xla_dump_large_constants"
    if merged.get("dump_hlo_local_module_name"):
      xla_flags += f" --xla_dump_hlo_module_re={merged['dump_hlo_local_module_name']}"
    merged["dump_hlo_xla_flags"] = xla_flags
    if merged.get("dump_hlo"):
      os.environ["XLA_FLAGS"] = xla_flags

  emb_scale, num_head_scale, mlp_dim_scale, layer_scale = _get_individual_scales(int(merged["global_parameter_scale"]))
  merged["emb_dim"] = (2**emb_scale) * int(merged["base_emb_dim"])
  merged["num_query_heads"] = (2**num_head_scale) * int(merged["base_num_query_heads"])
  merged["num_kv_heads"] = (2**num_head_scale) * int(merged["base_num_kv_heads"])
  merged["mlp_dim"] = (2**mlp_dim_scale) * int(merged["base_mlp_dim"])
  merged["moe_mlp_dim"] = (2**mlp_dim_scale) * int(merged["base_moe_mlp_dim"])
  merged["num_decoder_layers"] = (2**layer_scale) * int(merged["base_num_decoder_layers"])
  if int(merged.get("head_dim", 0) or 0) <= 0:
    merged["head_dim"] = merged["emb_dim"] // merged["num_query_heads"]

  compile_topology = str(merged.get("compile_topology", "") or "")
  compile_num_slices = int(merged.get("compile_topology_num_slices", -1))
  merged["using_compiled_topology"] = bool(compile_topology)
  if compile_topology:
    raw_devices = topologies.get_topology_desc(
        platform="tpu",
        topology_name=compile_topology,
        num_slices=compile_num_slices,
    ).devices
    num_target_devices = len(raw_devices)
  else:
    num_target_devices = jax.device_count()
  merged["num_target_devices"] = num_target_devices

  raw_num_slices = int(merged.get("num_slices", -1))
  if raw_num_slices == -1:
    merged["num_slices"] = compile_num_slices if compile_num_slices > 0 else 1
  else:
    merged["num_slices"] = raw_num_slices

  grad_accum = int(merged.get("gradient_accumulation_steps", 1))
  expansion_factor = float(merged.get("expansion_factor_real_data", 0) or 0)
  (
      merged["global_batch_size_to_load"],
      merged["global_batch_size_to_train_on"],
      merged["micro_batch_size_to_train_on"],
  ) = _compute_batch_sizes(
      float(merged["per_device_batch_size"]),
      num_target_devices,
      grad_accum,
      expansion_factor,
  )
  (
      merged["global_batch_size_to_load_eval"],
      merged["global_batch_size_to_eval_on"],
      merged["micro_batch_size_to_eval_on"],
  ) = _compute_batch_sizes(
      float(merged["eval_per_device_batch_size"]),
      num_target_devices,
      1,
      expansion_factor,
  )

  ici_map = {
      "diloco": merged.get("ici_diloco_parallelism", 1),
      "data": merged["ici_data_parallelism"],
      "stage": merged["ici_pipeline_parallelism"],
      "fsdp": merged["ici_fsdp_parallelism"],
      "fsdp_transpose": merged["ici_fsdp_transpose_parallelism"],
      "sequence": merged["ici_sequence_parallelism"],
      "context": merged["ici_context_parallelism"],
      "context_usp_ulysses": merged.get("ici_context_usp_ulysses_parallelism", 1),
      "context_autoregressive": merged["ici_context_autoregressive_parallelism"],
      "tensor": merged["ici_tensor_parallelism"],
      "tensor_sequence": merged["ici_tensor_sequence_parallelism"],
      "model": merged["ici_tensor_parallelism"],
      "expert": merged["ici_expert_parallelism"],
      "autoregressive": merged["ici_autoregressive_parallelism"],
      "attn_dp": 1,
      "attn_dp_expert": 1,
      "dcp": 1,
      "pcp": 1,
  }
  dcn_map = {
      "diloco": merged.get("dcn_diloco_parallelism", 1),
      "data": merged["dcn_data_parallelism"],
      "stage": merged["dcn_pipeline_parallelism"],
      "fsdp": merged["dcn_fsdp_parallelism"],
      "fsdp_transpose": merged["dcn_fsdp_transpose_parallelism"],
      "sequence": merged["dcn_sequence_parallelism"],
      "context": merged["dcn_context_parallelism"],
      "context_usp_ulysses": merged.get("dcn_context_usp_ulysses_parallelism", 1),
      "context_autoregressive": merged["dcn_context_autoregressive_parallelism"],
      "tensor": merged["dcn_tensor_parallelism"],
      "tensor_sequence": merged["dcn_tensor_sequence_parallelism"],
      "model": merged["dcn_tensor_parallelism"],
      "expert": merged["dcn_expert_parallelism"],
      "autoregressive": merged["dcn_autoregressive_parallelism"],
      "attn_dp": 1,
      "attn_dp_expert": 1,
      "dcp": 1,
      "pcp": 1,
  }
  mesh_axes = [str(a) for a in merged.get("mesh_axes", [])]
  merged["mesh_axes"] = mesh_axes
  merged["ici_parallelism"] = _resolve_parallelism(merged.get("ici_parallelism", []), ici_map, mesh_axes)
  merged["dcn_parallelism"] = _resolve_parallelism(merged.get("dcn_parallelism", []), dcn_map, mesh_axes)

  merged["dtype"] = jnp.dtype(merged["dtype"])
  merged["grad_dtype"] = jnp.dtype(merged["grad_dtype"])
  merged["weight_dtype"] = jnp.dtype(merged["weight_dtype"])
  merged["mu_dtype"] = merged["weight_dtype"] if not merged.get("mu_dtype") else jnp.dtype(merged["mu_dtype"])

  for tuple_key in (
      "logical_axis_rules",
      "logical_axis_rules_for_eval",
      "data_sharding",
  ):
    if tuple_key in merged and merged[tuple_key] is not None:
      merged[tuple_key] = _lists_to_tuples(merged[tuple_key])

  if merged.get("log_config", True):
    for k, v in sorted(merged.items()):
      if k != "hf_access_token":
        logging.info("Config param %s: %s", k, v)

  return AOTConfig(merged)


def fill_unspecified_mesh_axes(
    parallelism_vals: Sequence[int],
    target_product: int,
    parallelism_type: str,
) -> list[int]:
  """Evaluates unspecified (-1) DCN/ICI parallelism values."""
  vals = list(parallelism_vals)
  if -1 in vals:
    if vals.count(-1) != 1:
      raise ValueError(
          f"Found unspecified values (-1) for more than one {parallelism_type}"
          " parallelism axis. At most one axis can be unspecified."
      )
    determined_val = target_product / np.prod(vals) * -1
    if determined_val < 1 or not determined_val.is_integer():
      raise ValueError(f"Unable to determine unspecified {parallelism_type} parallelism" " value")
    vals[vals.index(-1)] = int(determined_val)
  target_type = "slices" if parallelism_type == "DCN" else "devices per slice"
  if np.prod(vals) != target_product:
    raise ValueError(
        f"Number of {target_type} {target_product} does not match the product"
        f" of the {parallelism_type} parallelism {np.prod(vals)}"
    )
  return vals


def create_topology_mesh(config: AOTConfig) -> jax.sharding.Mesh:
  """Creates the compile topology `jax.sharding.Mesh` for `config`."""
  raw_devices = topologies.get_topology_desc(
      platform="tpu",
      topology_name=config.compile_topology,
      num_slices=int(config.compile_topology_num_slices),
  ).devices
  jax.config.update(
      "jax_remove_size_one_mesh_axis_from_type",
      bool(config.remove_size_one_mesh_axis_from_type),
  )
  num_slices = int(config.num_slices)
  num_devices_per_slice = len(raw_devices) // num_slices
  ici_parallelism = fill_unspecified_mesh_axes(config.ici_parallelism, num_devices_per_slice, "ICI")
  dcn_parallelism = fill_unspecified_mesh_axes(config.dcn_parallelism, num_slices, "DCN") if num_slices > 1 else None
  devices_array = dsv3_sharding.create_device_mesh(
      ici_parallelism,
      raw_devices,
      dcn_parallelism=dcn_parallelism,
      allow_split_physical_axes=bool(config.allow_split_physical_axes),
      log_fn=logging.info,
  )
  axis_type = jax.sharding.AxisType.Explicit if str(config.shard_mode).lower() == "explicit" else jax.sharding.AxisType.Auto
  return jax.sharding.Mesh(
      devices_array,
      config.mesh_axes,
      axis_types=(axis_type,) * len(config.mesh_axes),
  )


def validate_supported(cfg: Any) -> None:
  """Raises `ValueError` if `cfg` uses options unsupported by Lineage DSv3."""
  problems: list[str] = []

  def require(ok: bool, message: str) -> None:
    if not ok:
      problems.append(message)

  get = functools.partial(getattr, cfg)
  require(get("mtp_num_layers", 0) >= 0, "mtp_num_layers must be >= 0")
  require(get("shared_experts", 1) == 1, "shared_experts must be 1")
  require(
      not get("quantization", None),
      "quantization (AQT/Qwix) is unsupported; use lineage_quantization",
  )
  require(
      get("gradient_accumulation_steps", 1) == 1,
      "gradient_accumulation_steps must be 1",
  )
  require(get("num_vocab_tiling", 1) <= 1, "num_vocab_tiling must be 1")
  require(
      not get("moe_dropless_fallback", None),
      "moe_dropless_fallback is unsupported",
  )
  require(
      not get("log_required_ragged_buffer_factor", False),
      "log_required_ragged_buffer_factor is unsupported",
  )
  require(
      not get("moe_log_max_load_ratio", False),
      "moe_log_max_load_ratio is unsupported",
  )
  require(not get("use_indexer", False), "use_indexer is unsupported")
  require(
      get("training_objective", "causal_lm") == "causal_lm",
      "training_objective must be causal_lm",
  )
  require(not get("use_multimodal", False), "use_multimodal is unsupported")
  require(
      not (get("enable_dropout", False) and get("dropout_rate", 0.0) > 0.0),
      "dropout is unsupported",
  )
  require(not get("te_moe_block", False), "te_moe_block is unsupported")
  require(not get("use_qk_clip", False), "use_qk_clip is unsupported")
  require(not get("enable_diloco", False), "enable_diloco is unsupported")
  require(
      not getattr(get("lora", None), "enable_lora", False),
      "lora is unsupported",
  )
  require(
      get("final_logits_soft_cap", None) is None,
      "final_logits_soft_cap is unsupported",
  )
  require(
      not (get("logits_via_embedding", False) and not get("normalize_embedding_logits", True)),
      "logits_via_embedding requires normalize_embedding_logits",
  )
  require(
      str(get("rope_type", "yarn")).lower().endswith("yarn"),
      "rope_type must be yarn",
  )
  require(get("rope_interleave", True), "rope_interleave must be true")
  require(
      get("routed_score_func", "sigmoid") == "sigmoid",
      "routed_score_func must be sigmoid",
  )
  require(get("scan_layers", True), "scan_layers must be true")
  require(
      not get("parameter_memory_host_offload", False),
      "parameter_memory_host_offload is unsupported",
  )
  require(
      not get("optimizer_memory_host_offload", False),
      "optimizer_memory_host_offload is unsupported",
  )
  require(
      not get("shard_optimizer_over_data", False),
      "shard_optimizer_over_data is unsupported",
  )
  require(
      not get("skip_step_on_spikes", False),
      "skip_step_on_spikes is unsupported",
  )
  require(
      not get("adamw_mask", None),
      "adamw_mask is unsupported",
  )
  require(
      not get("trainable_parameters_mask", None),
      "trainable_parameters_mask is unsupported",
  )
  if problems:
    raise ValueError("use_lineage=True does not support this config: " + "; ".join(problems) + ".")


def dsv3_config_from_maxtext(cfg: Any, mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh) -> dsv3_config.DSv3Config:
  """Builds the Lineage `DSv3Config` corresponding to a MaxText config."""
  num_decoder_layers = int(cfg.num_decoder_layers)
  num_dense_layers = int(cfg.first_num_dense_layers)
  router_dtype = jnp.float32 if getattr(cfg, "float32_gate_logits", False) else None
  router_bias_dtype = router_dtype or getattr(cfg, "dtype", jnp.bfloat16)
  model = dsv3_config.DSv3ModelConfig(
      vocab_size=int(cfg.vocab_size),
      emb_dim=int(cfg.emb_dim),
      num_dense_layers=num_dense_layers,
      num_sparse_layers=num_decoder_layers - num_dense_layers,
      num_mtp_layers=int(getattr(cfg, "mtp_num_layers", 0)),
      num_query_heads=int(cfg.num_query_heads),
      num_kv_heads=int(cfg.num_kv_heads),
      q_lora_rank=int(cfg.q_lora_rank),
      kv_lora_rank=int(cfg.kv_lora_rank),
      qk_head_dim=int(cfg.qk_nope_head_dim),
      rope_head_dim=int(cfg.qk_rope_head_dim),
      v_head_dim=int(cfg.v_head_dim),
      mscale=float(cfg.mscale),
      mlp_dim=int(cfg.mlp_dim),
      moe_mlp_dim=int(cfg.moe_mlp_dim),
      num_experts=int(cfg.num_experts),
      num_experts_per_tok=int(cfg.num_experts_per_tok),
      n_routing_groups=int(cfg.n_routing_groups),
      topk_routing_group=int(cfg.topk_routing_group),
      routed_scaling_factor=float(cfg.routed_scaling_factor),
      rope_theta=int(cfg.rope_max_timescale),
      max_position_embeddings=int(cfg.max_position_embeddings),
      original_max_position_embeddings=int(cfg.original_max_position_embeddings),
      rope_factor=int(cfg.rope_factor),
      beta_fast=int(cfg.beta_fast),
      beta_slow=int(cfg.beta_slow),
      norm_epsilon=float(cfg.normalization_layer_epsilon),
      tied_head=bool(getattr(cfg, "logits_via_embedding", False)),
      iota_embed=bool(getattr(cfg, "use_iota_embed", False)),
      dtype=jnp.dtype(cfg.dtype),
      weight_dtype=jnp.dtype(cfg.weight_dtype),
      router_dtype=None if router_dtype is None else jnp.dtype(router_dtype),
      router_bias_dtype=jnp.dtype(router_bias_dtype),
      head_dot_in_fp32=bool(getattr(cfg, "logits_dot_in_fp32", False)),
      cast_logits_to_fp32=bool(getattr(cfg, "cast_logits_to_fp32", True)),
  )
  sharding = dsv3_sharding.sharding_config(mesh, variant=getattr(cfg, "lineage_sharding_variant", None) or None)
  if cfg.capacity_factor <= 0:
    raise ValueError(f"capacity_factor must be > 0, got: {cfg.capacity_factor}")
  quant = dsv3_config.quant_config_from_name(cfg.lineage_quantization)
  if quant is not None and dsv3_types.is_gqa_dims(model.q_lora_rank, model.kv_lora_rank):
    # The GQA attention path (q_lora_rank = kv_lora_rank = 0) runs its
    # projections in bf16.
    quant = dataclasses.replace(quant, mla=None)
  expert_permutation = getattr(cfg, "lineage_expert_permutation", None)
  kernels = dsv3_config.DSv3KernelConfig(
      max_target_length=int(cfg.max_target_length),
      sa_block_q=int(cfg.sa_block_q),
      sa_block_kv=int(cfg.sa_block_kv),
      sa_block_kv_compute=int(cfg.sa_block_kv_compute),
      sa_block_q_dkv=int(cfg.sa_block_q_dkv),
      sa_block_kv_dkv=int(cfg.sa_block_kv_dkv),
      sa_block_kv_dkv_compute=int(cfg.sa_block_kv_dkv_compute),
      sa_q_layout=cfg.sa_q_layout,
      sa_k_layout=cfg.sa_k_layout,
      sa_v_layout=cfg.sa_v_layout,
      capacity_factor=float(cfg.capacity_factor),
      ragged_buffer_factor=_RAGGED_BUFFER_FACTOR,
      quant=quant,
      max_async_overlap_transform=bool(getattr(cfg, "lineage_max_async_overlap_transform", True)),
      expert_permutation=("none" if expert_permutation is None else str(expert_permutation)),
  )
  routed_bias_update_rate = float(cfg.routed_bias_update_rate) if getattr(cfg, "routed_bias", False) else 0.0
  megatron_seq_aux_loss = bool(
      getattr(cfg, "moe_use_megatron_seq_aux_loss", False)
      and getattr(cfg, "routed_score_func", "") == "sigmoid"
      and not getattr(cfg, "te_moe_block", False)
  )
  training = dsv3_config.DSv3TrainingConfig(
      load_balance_loss_weight=float(getattr(cfg, "load_balance_loss_weight", 0.0) or 0.0),
      megatron_seq_aux_loss=megatron_seq_aux_loss,
      routed_bias_update_rate=routed_bias_update_rate,
      z_loss_multiplier=float(getattr(cfg, "z_loss_multiplier", 0.0) or 0.0),
      mtp_loss_scaling_factor=float(getattr(cfg, "mtp_loss_scaling_factor", 0.1)),
      mtp_eval_target_module=int(getattr(cfg, "mtp_eval_target_module", 0)),
      mtp_reuse_input_embedding=bool(getattr(cfg, "mtp_reuse_input_embedding", False)),
  )
  lineage_cfg = dsv3_config.DSv3Config(model=model, sharding=sharding, kernels=kernels, training=training)
  lineage_cfg.validate()
  return lineage_cfg


class LineageRouterBiasVar(nnx.Variable):
  """Routed gate bias of a Lineage MoE stack."""


class LineageExpertPermutationVar(nnx.BatchStat):
  """Per-layer int32 expert permutation of a Lineage MoE stack."""


def _path_key(entry: Any) -> str:
  """Converts one `jax.tree_util` key-path entry into a string dict key."""
  if isinstance(entry, jax.tree_util.GetAttrKey):
    return entry.name
  if isinstance(entry, jax.tree_util.SequenceKey):
    return str(entry.idx)
  if isinstance(entry, jax.tree_util.DictKey):
    return str(entry.key)
  raise TypeError(f"Unsupported key path entry: {entry!r}")


def _param_tree(
    params: dsv3_model.DSv3Params[Any],
    shardings: dsv3_model.DSv3Params[Any],
) -> tuple[dict[str, Any], tuple[tuple[str, ...], ...]]:
  """Wraps `params` leaves in `nnx.Param`s nested as dicts by field name."""
  leaves_with_path, _ = jax.tree_util.tree_flatten_with_path(params)
  sharding_leaves = jax.tree.leaves(shardings)
  if len(sharding_leaves) != len(leaves_with_path):
    raise ValueError(f"Got {len(sharding_leaves)} parameter shardings for" f" {len(leaves_with_path)} parameters.")
  flat = {}
  for (path, leaf), sharding in zip(leaves_with_path, sharding_leaves):
    key = tuple(_path_key(entry) for entry in path)
    flat[key] = nnx.Param(leaf, out_sharding=sharding, eager_sharding=False)
  return traverse_util.unflatten_dict(flat), tuple(flat)


def _state_var(
    var_type: type[nnx.Variable],
    value: Any,
    sharding: Any,
) -> nnx.Variable:
  return var_type(value, out_sharding=sharding, eager_sharding=False)


class LineageAOTModel(nnx.Module):
  """Lineage NNX model wrapper for AOT compilation."""

  def __init__(self, config: AOTConfig, mesh: jax.sharding.Mesh):
    """Initializes the instance."""
    validate_supported(config)
    self.config = config
    self.mesh = mesh
    self.model_mode = "train"
    cfg = dsv3_config_from_maxtext(config, mesh)
    param_shardings, state_shardings = dsv3_model.dsv3_param_shardings(cfg, mesh)
    with jax.sharding.use_abstract_mesh(mesh.abstract_mesh):
      params, state = dsv3_model.init_dsv3_params(jax.random.key(0), cfg, mesh)
    params_dict, self._param_paths = _param_tree(params, param_shardings)
    self.params = nnx.data(params_dict)
    assert state.router_bias is not None
    assert state_shardings.router_bias is not None
    self.router_bias = _state_var(LineageRouterBiasVar, state.router_bias, state_shardings.router_bias)
    self.mtp_router_bias = nnx.data(
        [
            _state_var(LineageRouterBiasVar, bias, sharding)
            for bias, sharding in zip(state.mtp_router_bias, state_shardings.mtp_router_bias)
        ]
    )
    if state.expert_permutations is not None:
      assert state_shardings.expert_permutations is not None
      self.expert_permutations = _state_var(
          LineageExpertPermutationVar,
          state.expert_permutations,
          state_shardings.expert_permutations,
      )
      self.mtp_expert_permutations = nnx.data(
          [
              _state_var(LineageExpertPermutationVar, permutation, sharding)
              for permutation, sharding in zip(
                  state.mtp_expert_permutations,
                  state_shardings.mtp_expert_permutations,
              )
          ]
      )
    else:
      self.expert_permutations = None
      self.mtp_expert_permutations = nnx.data([])

  def lineage_config(self) -> dsv3_config.DSv3Config:
    """Returns the Lineage `DSv3Config` of this model."""
    return dsv3_config_from_maxtext(self.config, self.mesh)

  def lineage_params(self) -> dsv3_model.DSv3Params[Any]:
    """Reconstructs `DSv3Params` from the NNX `params` dictionary."""
    param_shardings, _ = dsv3_model.dsv3_param_shardings(self.lineage_config(), self.mesh)
    treedef = jax.tree_util.tree_structure(param_shardings)
    flat = traverse_util.flatten_dict(self.params)
    leaves = [flat[path].get_value() for path in self._param_paths]
    return jax.tree_util.tree_unflatten(treedef, leaves)

  def lineage_state(self) -> dsv3_model.DSv3MutableState[Any]:
    """Reconstructs `DSv3MutableState` from the non-Param NNX variables."""
    return dsv3_model.DSv3MutableState(
        router_bias=self.router_bias.get_value(),
        mtp_router_bias=tuple(bias.get_value() for bias in self.mtp_router_bias),
        expert_permutations=(None if self.expert_permutations is None else self.expert_permutations.get_value()),
        mtp_expert_permutations=tuple(p.get_value() for p in self.mtp_expert_permutations),
    )

  def set_lineage_state(self, state: dsv3_model.DSv3MutableState[Any]) -> None:
    """Updates the non-Param NNX variables from `state`."""
    assert state.router_bias is not None
    self.router_bias.set_value(state.router_bias)
    for var, bias in zip(self.mtp_router_bias, state.mtp_router_bias, strict=True):
      var.set_value(bias)
    if self.expert_permutations is not None:
      assert state.expert_permutations is not None
      self.expert_permutations.set_value(state.expert_permutations)
      for var, permutation in zip(
          self.mtp_expert_permutations,
          state.mtp_expert_permutations,
          strict=True,
      ):
        var.set_value(permutation)


class TrainStateNNX(nnx.Module):
  """Unified container for the NNX model and optimizer."""

  def __init__(self, model: nnx.Module, optimizer: nnx.Optimizer):
    """Initializes the instance."""
    self.model = model
    self.optimizer = optimizer

  def apply_gradients(self, grads: Any, **kwargs: Any) -> None:
    """Applies parameter updates via the NNX optimizer."""
    self.optimizer.update(self.model, grads, **kwargs)


def create_learning_rate_schedule(config: AOTConfig) -> optax.Schedule:
  """Creates the MaxText warmup + cosine / WSD learning rate schedule."""

  def make_cos_schedule(init_lr: float, final_lr: float, len_steps: int) -> optax.Schedule:
    def schedule(step: jax.typing.ArrayLike) -> jax.typing.ArrayLike:
      pct = step / (len_steps - 1) if len_steps > 1 else 1.0
      a = 0.5 * (jnp.cos(jnp.pi * pct) + 1)
      return init_lr * a + final_lr * (1 - a)

    return schedule

  lr = float(config.learning_rate)
  final_lr = lr * float(config.learning_rate_final_fraction)
  schedule_steps = int(config.learning_rate_schedule_steps)
  warmup_steps = int(schedule_steps * float(config.warmup_steps_fraction))
  constant_zero_steps = int(config.steps) - schedule_steps

  pieces = []
  boundaries = []

  if warmup_steps > 0:
    pieces.append(optax.linear_schedule(init_value=0.0, end_value=lr, transition_steps=warmup_steps))
    boundaries.append(warmup_steps)

  if str(config.lr_schedule_type).lower() == "cosine":
    cos_steps = schedule_steps - warmup_steps
    if cos_steps > 0:
      pieces.append(make_cos_schedule(lr, final_lr, cos_steps))
      boundaries.append(warmup_steps + cos_steps)
  else:
    decay_steps = int(schedule_steps * float(config.wsd_decay_steps_fraction))
    stable_steps = schedule_steps - warmup_steps - decay_steps
    if stable_steps > 0:
      pieces.append(optax.constant_schedule(lr))
      boundaries.append(warmup_steps + stable_steps)
    if decay_steps > 0:
      if str(config.wsd_decay_style).lower() == "linear":
        pieces.append(
            optax.linear_schedule(
                init_value=lr,
                end_value=final_lr,
                transition_steps=decay_steps - 1,
            )
        )
      else:
        pieces.append(make_cos_schedule(lr, final_lr, decay_steps))
      boundaries.append(warmup_steps + stable_steps + decay_steps)

  if constant_zero_steps > 0:
    pieces.append(optax.constant_schedule(0.0))
    boundaries.append(schedule_steps)

  return optax.join_schedules(pieces, boundaries)


def get_optimizer(config: AOTConfig, learning_rate_schedule: optax.Schedule) -> optax.GradientTransformation:
  """Creates the Optax optimizer specified by `config`."""
  if config.opt_type == "adamw":
    return optax.adamw(
        learning_rate_schedule,
        b1=float(config.adam_b1),
        b2=float(config.adam_b2),
        eps=float(config.adam_eps),
        eps_root=float(config.adam_eps_root),
        weight_decay=float(config.adam_weight_decay),
        mu_dtype=config.mu_dtype,
    )
  if config.opt_type == "sgd":
    return optax.sgd(learning_rate_schedule)
  raise ValueError(f"Unsupported opt_type for Lineage AOT: {config.opt_type}")


class _UnassignedAxis:
  """Sentinel for a logical dimension not yet assigned to a mesh axis."""


_UNASSIGNED_AXIS = _UnassignedAxis()


def _mesh_axes_free(new_assignment: Any, existing_assignments: Any) -> bool:
  new = set(jax.tree_util.tree_leaves(new_assignment))
  existing = set(jax.tree_util.tree_leaves(existing_assignments))
  new.discard(P.UNCONSTRAINED)
  new.discard(None)
  return not existing.intersection(new)


def logical_to_spec(
    logical_names: Sequence[Any] | None,
    rules: Sequence[Any],
) -> P | None:
  """Maps logical dimension names to a `PartitionSpec` via `rules`."""
  if logical_names is None:
    return None
  counts = collections.Counter(logical_names)
  dups = tuple(k for k, v in counts.items() if v > 1 and isinstance(k, str))
  if dups:
    raise ValueError(f"Unsupported: Dimensions {dups} occur more than once in array names.")
  result = [(_UNASSIGNED_AXIS if isinstance(name, str) else name) for name in logical_names]
  for rule_logical_name, rule_mesh_axes in rules:
    if rule_logical_name in logical_names:
      pos = logical_names.index(rule_logical_name)
      if _mesh_axes_free(rule_mesh_axes, result) and result[pos] is _UNASSIGNED_AXIS:
        result[pos] = rule_mesh_axes
  return P(*(None if x is _UNASSIGNED_AXIS else x for x in result))


def remove_size_one_mesh_axis(spec: P | None, mesh: jax.sharding.Mesh) -> P | None:
  """Removes mesh axes from `spec` where the mesh axis size is 1."""
  if spec is None:
    return None
  new_spec = []
  for s in spec:
    if s is None or s == P.UNCONSTRAINED:
      new_spec.append(s)
    elif isinstance(s, tuple):
      new_spec.append(tuple(i for i in s if mesh.shape.get(i, 1) != 1))
    else:
      new_spec.append(None if mesh.shape.get(s, 1) == 1 else s)
  return P(*new_spec, unreduced=spec.unreduced, reduced=spec.reduced)


def get_input_data_sharding(config: AOTConfig, mesh: jax.sharding.Mesh) -> NamedSharding:
  """Resolves the input batch `NamedSharding` for `config` and `mesh`."""
  spec = remove_size_one_mesh_axis(
      logical_to_spec(config.input_data_sharding_logical_axes, config.logical_axis_rules),
      mesh,
  )
  return NamedSharding(mesh, P() if spec is None else spec)


def get_shaped_batch(config: AOTConfig, batch_sharding: NamedSharding) -> dict[str, jax.ShapeDtypeStruct]:
  """Creates the abstract input batch dictionary for `train_step`."""
  # The train step consumes the training batch directly, so a fractional
  # per_device_batch_size must not load the padded `num_devices` rows.
  batch_shape = (
      int(config.global_batch_size_to_train_on),
      int(config.max_target_length),
  )
  sds = jax.ShapeDtypeStruct(batch_shape, jnp.int32, sharding=batch_sharding)
  return {
      "inputs": sds,
      "inputs_position": sds,
      "inputs_segmentation": sds,
      "targets": sds,
      "targets_position": sds,
      "targets_segmentation": sds,
  }


def _construct_named_sharding_for_var(v: nnx.Variable, mesh: jax.sharding.Mesh) -> nnx.Variable:
  """Extracts or constructs the `NamedSharding` for an `nnx.Variable`."""
  val = v.get_value()
  if not hasattr(val, "shape"):
    leaves = jax.tree_util.tree_leaves(val)
    if leaves:
      first_leaf = leaves[0]
      if hasattr(first_leaf, "shape"):
        leaf_var = _construct_named_sharding_for_var(v.replace(value=first_leaf), mesh)
        leaf_sharding = leaf_var.get_value()
        if not isinstance(leaf_sharding, NamedSharding):
          leaf_sharding = NamedSharding(mesh, P())
        return v.replace(jax.tree.map(lambda _: leaf_sharding, val))
      replicated = NamedSharding(mesh, P())
      return v.replace(jax.tree.map(lambda _: replicated, val))
    return v
  metadata = v.get_metadata()
  out_sharding = metadata.get("out_sharding") or metadata.get("sharding_names") or metadata.get("sharding")
  if isinstance(out_sharding, NamedSharding):
    return v.replace(out_sharding)
  if isinstance(out_sharding, P):
    return v.replace(NamedSharding(mesh, out_sharding))
  return v.replace(NamedSharding(mesh, P()))


def get_shaped_inputs_single_pass(mesh: jax.sharding.Mesh, config: AOTConfig) -> tuple[
    tuple[nnx.State, dict[str, jax.ShapeDtypeStruct]],
    dict[str, Any],
    nnx.State,
    nnx.GraphDef[TrainStateNNX],
]:
  """Builds abstract train state, shardings, and GraphDef in a single pass."""
  learning_rate_schedule = create_learning_rate_schedule(config)
  tx = get_optimizer(config, learning_rate_schedule)

  def init_state_fn() -> TrainStateNNX:
    model = LineageAOTModel(config, mesh)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    return TrainStateNNX(model, optimizer)

  with spmd.logical_axis_rules(config.logical_axis_rules):
    abs_train_state = nnx.eval_shape(init_state_fn)
    graphdef, abs_var_state = nnx.split(abs_train_state)
    named_sharding_state = jax.tree.map(
        lambda v: _construct_named_sharding_for_var(v, mesh),
        abs_var_state,
        is_leaf=lambda x: isinstance(x, nnx.Variable),
    )

    def _to_abstract_var(a_var: nnx.Variable, s_var: nnx.Variable) -> nnx.Variable:
      a_val = a_var.get_value()
      s_val = s_var.get_value()

      def _make_leaf(leaf_a: Any, leaf_s: Any) -> jax.ShapeDtypeStruct:
        if hasattr(leaf_s, "spec") and len(leaf_s.spec) > len(leaf_a.shape):
          leaf_s = NamedSharding(leaf_s.mesh, P(*leaf_s.spec[: len(leaf_a.shape)]))
        return jax.ShapeDtypeStruct(leaf_a.shape, leaf_a.dtype, sharding=leaf_s)

      if hasattr(a_val, "shape") and hasattr(a_val, "dtype"):
        new_val = _make_leaf(a_val, s_val)
      else:
        s_tree = jax.tree.map(lambda _: s_val, a_val) if isinstance(s_val, (jax.sharding.Sharding, P)) else s_val
        new_val = jax.tree.map(
            _make_leaf,
            a_val,
            s_tree,
            is_leaf=lambda x: hasattr(x, "shape") and hasattr(x, "dtype"),
        )
      return a_var.replace(value=new_val)

    abstract_sharded_state = jax.tree.map(
        _to_abstract_var,
        abs_var_state,
        named_sharding_state,
        is_leaf=lambda x: isinstance(x, nnx.Variable),
    )

  state_mesh_shardings = jax.tree.map(
      lambda x: x.sharding,
      abstract_sharded_state,
      is_leaf=lambda x: isinstance(x, jax.ShapeDtypeStruct),
  )

  data_sharding = get_input_data_sharding(config, mesh)
  shaped_batch = get_shaped_batch(config, data_sharding)
  return (
      (abstract_sharded_state, shaped_batch),
      {},
      state_mesh_shardings,
      graphdef,
  )


def _stream_chunked_train_data(cfg: Any) -> bool:
  """Returns whether `cfg` trains on stream-chunked `c4_mlperf`."""
  if getattr(cfg, "dataset_type", None) != "c4_mlperf":
    return False
  use_stream_chunking = getattr(cfg, "use_stream_chunking", None)
  if use_stream_chunking is None:
    use_stream_chunking = not getattr(cfg, "tokenize_train_data", True)
  return bool(use_stream_chunking)


def loss_and_aux(
    model: LineageAOTModel,
    config: AOTConfig,
    batch_inputs: Mapping[str, jax.Array],
    *,
    is_train: bool,
) -> tuple[jax.Array, dict[str, Any]]:
  """Computes the Lineage DSv3 training loss and auxiliary metrics dict."""
  cfg = model.lineage_config()
  drop_segment_ids = is_train and _stream_chunked_train_data(config)
  batch = dsv3_model.DSv3Batch(
      inputs=batch_inputs["inputs"],
      targets=batch_inputs["targets"],
      positions=batch_inputs["inputs_position"],
      segment_ids=(None if drop_segment_ids else batch_inputs["inputs_segmentation"]),
      target_mask=batch_inputs["targets_segmentation"],
  )
  with jax.sharding.use_abstract_mesh(model.mesh.abstract_mesh):
    loss, aux = dsv3_model.dsv3_loss_and_aux(
        model.lineage_params(),
        model.lineage_state(),
        batch,
        cfg,
        mesh=model.mesh,
        train=is_train,
    )
  if is_train:
    model.set_lineage_state(aux.new_state)

  new_state = aux.new_state
  diag_bias_values = []
  diag_bias_updates = []
  if is_train and aux.router_bias_updates is not None:
    diag_bias_values.append(new_state.router_bias)
    diag_bias_updates.append(aux.router_bias_updates)
    diag_bias_values.extend(new_state.mtp_router_bias)
    diag_bias_updates.extend(aux.mtp_router_bias_updates)
  return loss, {
      "intermediate_outputs": {},
      "xent_sum": aux.xent_sum,
      "z_loss": aux.z_loss_sum / (aux.total_weights + dsv3_loss.EPS),
      "total_weights": aux.total_weights,
      "moe_lb_loss": aux.moe_lb_loss,
      "indexer_loss": 0.0,
      "moe_bias_updates": None,
      "mtp_moe_bias_updates": None,
      "mtp_loss": aux.mtp_loss,
      "batch_stats": None,
      "has_moe_overflow": jnp.bool_(False),
      "diag_bias_values": diag_bias_values,
      "diag_bias_updates": diag_bias_updates,
  }


def l2norm_pytree(x: Any) -> jax.Array:
  """Computes the L2 norm of all leaves in a pytree."""
  return jnp.sqrt(jax.tree_util.tree_reduce(lambda acc, y: acc + jnp.sum(jnp.square(y)), x, initializer=0.0))


def train_step(
    model_graphdef_container: nnx.GraphDef[TrainStateNNX],
    config: AOTConfig,
    state: nnx.State,
    batch_inputs: Mapping[str, jax.Array],
    dropout_rng: Any = None,
) -> tuple[nnx.State, dict[str, Any]]:
  """Executes a single Lineage DSv3 training step for AOT compilation."""
  del dropout_rng
  train_state = nnx.merge(model_graphdef_container, state)

  owg_type = nnx.variablelib.variable_type_from_name("_overwrite_with_gradient", allow_register=True)
  custom_param_filter = nnx.Any(owg_type)
  nnx.pop(train_state.model, nnx.Intermediate)
  model_graphdef, curr_params, custom_params, rest = nnx.split(train_state.model, nnx.Param, custom_param_filter, ...)

  def diff_wrapper(
      curr_p: Any,
      custom_p: Any,
      rest_s: Any,
      cfg: AOTConfig,
      batch_data: Mapping[str, jax.Array],
  ) -> tuple[jax.Array, tuple[dict[str, Any], nnx.State]]:
    local_model: Any = nnx.merge(model_graphdef, curr_p, custom_p, rest_s, copy=True)
    loss_val, aux_dict = loss_and_aux(local_model, cfg, batch_data, is_train=True)
    non_param_rest = nnx.state(
        local_model,
        nnx.Not(nnx.Any(nnx.Param, custom_param_filter, nnx.Intermediate)),
    )
    return loss_val, (aux_dict, non_param_rest)

  grad_func = jax.value_and_grad(diff_wrapper, argnums=(0, 1), has_aux=True)
  (loss, (aux, non_param_rest)), (raw_grads, custom_grads) = grad_func(
      curr_params, custom_params, rest, config, batch_inputs
  )
  nnx.update(train_state.model, nnx.State.merge(custom_grads, non_param_rest))

  raw_grads = jax.tree_util.tree_map(
      lambda x: x.astype(config.grad_dtype) if x.dtype == jnp.float32 else x,
      raw_grads,
  )

  if float(config.gradient_clipping_threshold) > 0:
    grads, _ = optax.clip_by_global_norm(float(config.gradient_clipping_threshold)).update(
        raw_grads, optax.EmptyState(), None
    )
  else:
    grads = raw_grads

  train_state.apply_gradients(grads)

  lm_loss = aux["xent_sum"] / (aux["total_weights"] + dsv3_loss.EPS)
  scalar_metrics: dict[str, Any] = {
      "learning/loss": loss,
      "learning/lm_loss": lm_loss,
      "learning/perplexity": jnp.exp(lm_loss),
      "learning/z_loss": aux.get("z_loss", 0.0),
      "learning/moe_lb_loss": aux["moe_lb_loss"],
      "learning/indexer_loss": aux.get("indexer_loss", 0.0),
      "learning/mtp_loss": aux.get("mtp_loss", 0.0),
      "learning/total_weights": aux["total_weights"],
      "learning/grad_norm": l2norm_pytree(grads),
      "learning/raw_grad_norm": l2norm_pytree(raw_grads),
      "learning/param_norm": l2norm_pytree(nnx.state(train_state.model, nnx.Param)),
  }
  metrics = {"scalar": scalar_metrics, "scalars": {}}
  out_state = nnx.state(train_state, nnx.Not(nnx.Intermediate))
  return out_state, metrics


def parse_libtpu_flags_to_dict(flags_str: str) -> dict[str, str]:
  """Parses a string of `--key=value` XLA flags into a compiler options dict."""
  if not flags_str or not flags_str.strip():
    return {}
  cleaned_str = flags_str.replace("\\", " ")
  tokens = cleaned_str.split()
  options_dict: dict[str, str] = {}
  token_pattern = re.compile(r"^--([a-zA-Z0-9_]+)=(.+)$")
  for token in tokens:
    match = token_pattern.match(token)
    if not match:
      raise ValueError(f"Invalid flag format detected: '{token}'." " Expected format: '--key=value'")
    key, value = match.groups()
    if key in options_dict:
      raise ValueError(f"Duplicate flag detected: '--{key}'")
    options_dict[key] = value
  return options_dict


def _serialize_compiled(compiled: jax.stages.Compiled) -> bytes:
  """Serializes the underlying PJRT executable to bytes."""
  executable = getattr(compiled, "_executable", None)
  unloaded_executable = getattr(executable, "_unloaded_executable", None)
  if unloaded_executable is None:
    raise ValueError("Compilation does not support serialization")
  if hasattr(unloaded_executable, "client") and hasattr(unloaded_executable.client, "serialize_executable"):
    return unloaded_executable.client.serialize_executable(unloaded_executable)
  if hasattr(unloaded_executable, "serialize"):
    return unloaded_executable.serialize()
  raise ValueError("Underlying executable does not support serialization")


def main(argv: Sequence[str]) -> None:
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["LIBTPU_INIT_ARGS"] = os.environ.get("LIBTPU_INIT_ARGS", "") + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"
  print("Starting train_compile.py...", flush=True)

  config = load_aot_config(argv)
  if not config.use_lineage:
    raise ValueError("dsv3_aot_compile requires use_lineage=True.")
  validate_supported(config)

  topology_mesh = create_topology_mesh(config)
  print(f"System Information: Jax Version: {jax.__version__}", flush=True)

  (
      shaped_train_args,
      shaped_train_kwargs,
      state_mesh_shardings,
      model_graphdef,
  ) = get_shaped_inputs_single_pass(topology_mesh, config)
  data_sharding = get_input_data_sharding(config, topology_mesh)

  func_to_compile = functools.partial(
      train_step,
      model_graphdef,
      config,
  )
  setattr(func_to_compile, "__name__", "train_step")
  in_shard = (state_mesh_shardings, data_sharding)
  out_shard = (state_mesh_shardings, None)

  print("Jitting and compiling train step...", flush=True)
  with (
      jax.set_mesh(topology_mesh),
      spmd.logical_axis_rules(config.logical_axis_rules),
  ):
    jitted = jax.jit(
        func_to_compile,
        in_shardings=in_shard,
        out_shardings=out_shard,
        static_argnums=(),
        donate_argnums=0,
    )
    lowered = jitted.lower(*shaped_train_args, **shaped_train_kwargs)
  compiler_options = parse_libtpu_flags_to_dict(str(getattr(config, "compile_xla_flags", "") or ""))
  compiled = lowered.compile(compiler_options=compiler_options)
  print("Jitting and compilation complete!", flush=True)

  if config.compiled_trainstep_file:
    print("Saving compiled object...")
    serialized = _serialize_compiled(compiled)
    with open(config.compiled_trainstep_file, "wb") as f:
      f.write(serialized)
    print("Successfully saved compiled object as" f" {config.compiled_trainstep_file}")
  print("Finished train_compile.py successfully!", flush=True)
  print(f"Cost analysis: {compiled.cost_analysis()}")
  print(f"Memory analysis: {compiled.memory_analysis()}")


if __name__ == "__main__":
  app.run(main)
