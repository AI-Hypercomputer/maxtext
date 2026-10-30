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

"""Transitional `from_pyconfig` bridge translating `MaxTextConfig` / `HyperParameters` to m3 configs."""

import logging
from typing import Any

from maxtext.configs.types import (
    AOT,
    DType,
    DatasetType,
    Goodput,
    LearningRateScheduleType,
    ManagedMLDiagnostics,
    MatmulPrecision,
    OptimizerType,
    OrbaxStorage,
    Profiling,
    QuantizationType,
    RematPolicy,
    Tensorboard,
)
from maxtext.m3.configs.training_config import (
    DataPipelineConfig,
    MeshConfig,
    TrainingConfig,
    TransformConfig,
)
from maxtext.m3.models.registry import ModelConfig

logger = logging.getLogger(__name__)


class ConfigMigrationError(ValueError):
  """Raised when a user explicitly sets a `pyconfig` knob that has no `m3` equivalent or alters training."""


# Explicit user overrides that map cleanly onto `(ModelConfig, TrainingConfig)`.
_TRANSLATED_PYCONFIG_KEYS: frozenset[str] = frozenset(
    {
        # Run & model identity
        "model_name",
        "use_m3_model",
        "override_model_config",
        "run_name",
        "base_output_directory",
        "init_weights_seed",
        # Model architecture overrides (when `override_model_config=True` or matching preset)
        "vocab_size",
        "emb_dim",
        "base_emb_dim",
        "mlp_dim",
        "base_mlp_dim",
        "num_decoder_layers",
        "base_num_decoder_layers",
        "num_query_heads",
        "base_num_query_heads",
        "num_kv_heads",
        "base_num_kv_heads",
        "head_dim",
        "normalization_layer_epsilon",
        "rope_max_timescale",
        "logits_via_embedding",
        "use_qk_norm",
        "float32_logits",
        "logits_dot_in_fp32",
        "cast_logits_to_fp32",
        "normalize_embedding_logits",
        "final_logits_soft_cap",
        # Training horizon, batching, precision, loss
        "steps",
        "max_target_length",
        "per_device_batch_size",
        "gradient_accumulation_steps",
        "dtype",
        "weight_dtype",
        "grad_dtype",
        "matmul_precision",
        "num_vocab_tiling",
        "z_loss_multiplier",
        "routed_bias_update_rate",
        # Optimizer & schedule
        "opt_type",
        "learning_rate",
        "lr_schedule_type",
        "learning_rate_schedule_steps",
        "warmup_steps_fraction",
        "cosine_learning_rate_final_fraction",
        "adam_b1",
        "adam_b2",
        "adam_eps",
        "adam_weight_decay",
        "gradient_clipping_threshold",
        "freeze_vision_encoder_params",
        # Cadence & checkpoints
        "log_period",
        "eval_interval",
        "eval_steps",
        "enable_checkpointing",
        "checkpoint_period",
        "load_parameters_path",
        "load_full_state_path",
        # Mesh sizing (`ici_*` / `dcn_*` / `shard_mode`)
        "shard_mode",
        "ici_data_parallelism",
        "ici_fsdp_parallelism",
        "ici_context_parallelism",
        "ici_tensor_parallelism",
        "ici_expert_parallelism",
        "dcn_data_parallelism",
        "dcn_fsdp_parallelism",
        "dcn_context_parallelism",
        "dcn_tensor_parallelism",
        "dcn_expert_parallelism",
        # Transforms
        "remat_policy",
        "scan_layers",
        "param_scan_axis",
        "quantization",
        "optimizer_memory_host_offload",
        "parameter_memory_host_offload",
        # Data pipeline
        "dataset_type",
        "dataset_path",
        "dataset_name",
        "grain_train_files",
        "grain_eval_files",
        "grain_worker_count",
        "tokenizer_path",
        "hf_access_token",
        "packing",
        "grain_packing_type",
        "data_shuffle_seed",
        # Shared leaf models
        *AOT.model_fields.keys(),
        *Goodput.model_fields.keys(),
        *ManagedMLDiagnostics.model_fields.keys(),
        *OrbaxStorage.model_fields.keys(),
        *Profiling.model_fields.keys(),
        *Tensorboard.model_fields.keys(),
    }
)

# Meta / environment flags that are harmless no-ops when passed via `base.yml` CLI scripts.
_IGNORED_PYCONFIG_KEYS: frozenset[str] = frozenset(
    {
        "base_config",
        "log_config",
        "skip_jax_distributed_system",
        "jax_cache_dir",
        "hardware",
        "max_prefill_predict_length",
        "async_checkpointing",
        "save_checkpoint_on_completion",
        "save_checkpoint_on_start",
        "enable_single_controller",
        "tokenizer_type",
        "use_multimodal",
        "decoder_block",
        "mlp_activations",
        "enable_dropout",
    }
)


def _unwrap_pydantic_config(cfg: Any) -> tuple[Any, set[str]]:
  """Extracts the underlying `MaxTextConfig` and explicit `model_fields_set` from `cfg`."""
  pyd_cfg = getattr(cfg, "_pydantic_config", cfg)
  explicit = set(getattr(pyd_cfg, "model_fields_set", set()))
  return pyd_cfg, explicit


def _extract_mesh_dict(pyd_cfg: Any, prefix: str, default_auto: dict[str, int]) -> dict[str, int]:
  """Extracts non-unit 5-axis parallelism overrides from `pyd_cfg` for `prefix` ('ici' or 'dcn')."""
  mapping: dict[str, int] = {}
  axis_fields = (
      ("data", f"{prefix}_data_parallelism"),
      ("fsdp", f"{prefix}_fsdp_parallelism"),
      ("context", f"{prefix}_context_parallelism"),
      ("tensor", f"{prefix}_tensor_parallelism"),
      ("expert", f"{prefix}_expert_parallelism"),
  )
  for axis_name, attr in axis_fields:
    val = int(getattr(pyd_cfg, attr, 1))
    if val != 1:
      mapping[axis_name] = val
  return mapping if mapping else dict(default_auto)


def _extract_leaf_model(cls: type[Any], pyd_cfg: Any) -> Any:
  """Constructs a shared leaf Pydantic sub-config `cls` from matching fields on `pyd_cfg`."""
  kwargs = {k: getattr(pyd_cfg, k) for k in cls.model_fields if hasattr(pyd_cfg, k)}
  return cls(**kwargs)


def format_cli_migration_hint(preset_name: str, explicit_keys: set[str], pyd_cfg: Any) -> str:
  """Builds a concise 1-line native `m3` CLI command equivalent to the `pyconfig` invocation."""
  parts = [f"python3 -m maxtext.m3.train --preset {preset_name}"]
  hint_map = (
      ("steps", "steps"),
      ("per_device_batch_size", "per_device_batch_size"),
      ("max_target_length", "max_target_length"),
      ("learning_rate", "learning_rate"),
      ("scan_layers", "scan_layers"),
      ("base_output_directory", "base_output_directory"),
      ("run_name", "run_name"),
  )
  for py_key, m3_flag in hint_map:
    if py_key in explicit_keys and hasattr(pyd_cfg, py_key):
      val = getattr(pyd_cfg, py_key)
      if isinstance(val, bool):
        parts.append(f"--{m3_flag}={str(val).lower()}")
      else:
        parts.append(f"--{m3_flag}={val}")
  return " ".join(parts)


def from_pyconfig(cfg: Any) -> tuple[ModelConfig, TrainingConfig]:
  """Translates a `MaxTextConfig` or `HyperParameters` instance into `(ModelConfig, TrainingConfig)`.

  Inspects `model_fields_set` (explicit CLI / kwarg / env overrides) and classifies every user-set
  key into three deterministic categories:
    1. Translated: mapped onto `ModelConfig` or `TrainingConfig`.
    2. Ignored: benign environment / meta flags logged at INFO/DEBUG without failing.
    3. Rejected: unsupported behavioral overrides (e.g., `ici_pipeline_parallelism=2`,
       `moe_load_balance_loss_weight=0.01`, `use_dpo=True`) raise `ConfigMigrationError` immediately
       rather than silently changing training semantics.
  """
  pyd_cfg, explicit_keys = _unwrap_pydantic_config(cfg)

  rejected = sorted(k for k in explicit_keys if k not in _TRANSLATED_PYCONFIG_KEYS and k not in _IGNORED_PYCONFIG_KEYS)
  if rejected:
    details = ", ".join(f"{k}={getattr(pyd_cfg, k, None)!r}" for k in rejected)
    raise ConfigMigrationError(
        f"Explicit pyconfig override(s) [{details}] are not supported by `m3` and cannot be silently "
        "ignored. Remove unsupported flags or pass supported `TrainingConfig` options via "
        "`python3 -m maxtext.m3.train --preset <model>`."
    )

  model_name = str(getattr(pyd_cfg, "model_name", "qwen3-0.6b"))
  try:
    ModelConfig.from_preset(model_name)
  except KeyError as exc:
    raise ConfigMigrationError(str(exc)) from exc

  arch_overrides: dict[str, Any] = {
      "param_dtype": DType(str(getattr(getattr(pyd_cfg, "weight_dtype", "float32"), "value", pyd_cfg.weight_dtype))),
      "compute_dtype": DType(str(getattr(getattr(pyd_cfg, "dtype", "bfloat16"), "value", pyd_cfg.dtype))),
  }
  arch_field_pairs = (
      ("vocab_size", ("vocab_size",)),
      ("hidden_size", ("emb_dim", "base_emb_dim")),
      ("intermediate_size", ("mlp_dim", "base_mlp_dim")),
      ("num_hidden_layers", ("num_decoder_layers", "base_num_decoder_layers")),
      ("num_attention_heads", ("num_query_heads", "base_num_query_heads")),
      ("num_key_value_heads", ("num_kv_heads", "base_num_kv_heads")),
      ("head_dim", ("head_dim",)),
      ("rms_norm_eps", ("normalization_layer_epsilon",)),
      ("rope_theta", ("rope_max_timescale",)),
      ("tie_word_embeddings", ("logits_via_embedding",)),
      ("use_qk_norm", ("use_qk_norm",)),
      ("float32_logits", ("float32_logits",)),
      ("logits_dot_in_fp32", ("logits_dot_in_fp32",)),
      ("cast_logits_to_fp32", ("cast_logits_to_fp32",)),
      ("normalize_embedding_logits", ("normalize_embedding_logits",)),
      ("final_logits_soft_cap", ("final_logits_soft_cap",)),
  )
  for m3_key, py_candidates in arch_field_pairs:
    if any(cand in explicit_keys for cand in py_candidates):
      val = getattr(pyd_cfg, py_candidates[0], None)
      if val is not None:
        arch_overrides[m3_key] = val

  model_cfg = ModelConfig.from_preset(model_name, **arch_overrides)

  shard_mode_raw = str(getattr(getattr(pyd_cfg, "shard_mode", "auto"), "value", getattr(pyd_cfg, "shard_mode", "auto")))
  shard_mode: Any = "explicit" if "explicit" in shard_mode_raw.lower() else "auto"
  mesh_cfg = MeshConfig(
      ici=_extract_mesh_dict(pyd_cfg, "ici", {"fsdp": -1}),
      dcn=_extract_mesh_dict(pyd_cfg, "dcn", {"data": -1}),
      shard_mode=shard_mode,
  )

  raw_remat = getattr(pyd_cfg, "remat_policy", RematPolicy.FULL)
  remat_val_str = str(getattr(raw_remat, "value", raw_remat)).strip().lower()
  remat_policy: RematPolicy | str = "none" if remat_val_str in ("", "none") else RematPolicy(remat_val_str)

  raw_quant = getattr(pyd_cfg, "quantization", QuantizationType.NONE)
  quant_val = (
      QuantizationType.NONE if raw_quant in (None, "") else QuantizationType(getattr(raw_quant, "value", raw_quant))
  )

  transform_cfg = TransformConfig(
      remat_policy=remat_policy,  # type: ignore[arg-type]
      scan_layers=bool(getattr(pyd_cfg, "scan_layers", True)),
      param_scan_axis=int(getattr(pyd_cfg, "param_scan_axis", 1)),
      quantization=quant_val,
      optimizer_memory_host_offload=bool(getattr(pyd_cfg, "optimizer_memory_host_offload", False)),
      parameter_memory_host_offload=bool(getattr(pyd_cfg, "parameter_memory_host_offload", False)),
  )

  packing_type_raw = str(getattr(pyd_cfg, "grain_packing_type", "first_fit"))
  if packing_type_raw not in ("first_fit", "best_fit"):
    if "grain_packing_type" in explicit_keys:
      raise ConfigMigrationError(
          f"grain_packing_type={packing_type_raw!r} is not supported in m3; use 'first_fit' or 'best_fit'."
      )
    packing_type_raw = "first_fit"

  data_cfg = DataPipelineConfig(
      dataset_type=DatasetType(getattr(getattr(pyd_cfg, "dataset_type", "synthetic"), "value", pyd_cfg.dataset_type)),
      dataset_path=str(getattr(pyd_cfg, "dataset_path", "") or ""),
      dataset_name=str(getattr(pyd_cfg, "dataset_name", "") or ""),
      grain_train_files=str(getattr(pyd_cfg, "grain_train_files", "") or ""),
      grain_eval_files=str(getattr(pyd_cfg, "grain_eval_files", "") or ""),
      grain_worker_count=max(0, int(getattr(pyd_cfg, "grain_worker_count", 1))),
      tokenizer_path=str(getattr(pyd_cfg, "tokenizer_path", "") or ""),
      hf_access_token=getattr(pyd_cfg, "hf_access_token", None),
      packing=bool(getattr(pyd_cfg, "packing", True)),
      grain_packing_type=packing_type_raw,  # type: ignore[arg-type]
      data_shuffle_seed=int(getattr(pyd_cfg, "data_shuffle_seed", 42)),
  )

  load_ckpt = str(getattr(pyd_cfg, "load_full_state_path", "") or getattr(pyd_cfg, "load_parameters_path", "") or "")
  steps_val = int(getattr(pyd_cfg, "steps", 100))
  if steps_val <= 0:
    steps_val = 1

  train_cfg = TrainingConfig(
      run_name=str(getattr(pyd_cfg, "run_name", "") or "m3_run"),
      base_output_directory=str(getattr(pyd_cfg, "base_output_directory", "") or ""),
      seed=int(getattr(pyd_cfg, "init_weights_seed", 0)),
      steps=steps_val,
      max_target_length=int(getattr(pyd_cfg, "max_target_length", 2048)),
      per_device_batch_size=float(getattr(pyd_cfg, "per_device_batch_size", 1.0)),
      gradient_accumulation_steps=max(1, int(getattr(pyd_cfg, "gradient_accumulation_steps", 1))),
      param_dtype=arch_overrides["param_dtype"],
      compute_dtype=arch_overrides["compute_dtype"],
      grad_dtype=DType(str(getattr(getattr(pyd_cfg, "grad_dtype", "float32"), "value", pyd_cfg.grad_dtype))),
      matmul_precision=MatmulPrecision(
          str(getattr(getattr(pyd_cfg, "matmul_precision", "default"), "value", pyd_cfg.matmul_precision)).lower()
      ),
      num_vocab_tiles=max(1, int(getattr(pyd_cfg, "num_vocab_tiling", 1))),
      z_loss_multiplier=max(0.0, float(getattr(pyd_cfg, "z_loss_multiplier", 0.0))),
      router_bias_update_rate=max(0.0, float(getattr(pyd_cfg, "routed_bias_update_rate", 0.0))),
      optimizer_type=OptimizerType(str(getattr(getattr(pyd_cfg, "opt_type", "adamw"), "value", pyd_cfg.opt_type))),
      learning_rate=float(getattr(pyd_cfg, "learning_rate", 3e-4)),
      lr_schedule_type=LearningRateScheduleType(
          str(getattr(getattr(pyd_cfg, "lr_schedule_type", "cosine"), "value", pyd_cfg.lr_schedule_type))
      ),
      learning_rate_schedule_steps=int(getattr(pyd_cfg, "learning_rate_schedule_steps", -1)),
      warmup_steps_fraction=float(getattr(pyd_cfg, "warmup_steps_fraction", 0.1)),
      cosine_learning_rate_final_fraction=float(getattr(pyd_cfg, "cosine_learning_rate_final_fraction", 0.1)),
      adam_b1=float(getattr(pyd_cfg, "adam_b1", 0.9)),
      adam_b2=float(getattr(pyd_cfg, "adam_b2", 0.95)),
      adam_eps=float(getattr(pyd_cfg, "adam_eps", 1e-8)),
      adam_weight_decay=float(getattr(pyd_cfg, "adam_weight_decay", 0.1)),
      gradient_clipping_threshold=max(0.0, float(getattr(pyd_cfg, "gradient_clipping_threshold", 1.0))),
      freeze_vision_encoder=bool(getattr(pyd_cfg, "freeze_vision_encoder_params", False)),
      log_period=max(1, int(getattr(pyd_cfg, "log_period", 10))),
      eval_interval=max(0, int(getattr(pyd_cfg, "eval_interval", 0))),
      eval_steps=max(1, int(getattr(pyd_cfg, "eval_steps", 1))),
      enable_checkpointing=bool(getattr(pyd_cfg, "enable_checkpointing", False)),
      checkpoint_period=max(1, int(getattr(pyd_cfg, "checkpoint_period", 100))),
      load_checkpoint_path=load_ckpt,
      mesh=mesh_cfg,
      transforms=transform_cfg,
      data=data_cfg,
      aot=_extract_leaf_model(AOT, pyd_cfg),
      goodput=_extract_leaf_model(Goodput, pyd_cfg),
      diagnostics=_extract_leaf_model(ManagedMLDiagnostics, pyd_cfg),
      orbax_storage=_extract_leaf_model(OrbaxStorage, pyd_cfg),
      profiling=_extract_leaf_model(Profiling, pyd_cfg),
      tensorboard=_extract_leaf_model(Tensorboard, pyd_cfg),
  )

  migration_hint = format_cli_migration_hint(model_name, explicit_keys, pyd_cfg)
  logger.info("m3 pyconfig bridge active; native m3 CLI equivalent: %s", migration_hint)
  return model_cfg, train_cfg
