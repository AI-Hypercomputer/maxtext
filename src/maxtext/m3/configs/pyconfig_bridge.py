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
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from maxtext.configs.types import (
    AOT,
    DType,
    DatasetType,
    LearningRateScheduleType,
    MatmulPrecision,
    MaxTextConfig,
    OrbaxStorage,
    Profiling,
    QuantizationType,
    RematPolicy,
)
from maxtext.m3.configs.training_config import (
    CheckpointConfig,
    DataPipelineConfig,
    MeshConfig,
    MetricsConfig,
    OptimizerConfig,
    TrainingConfig,
    TransformConfig,
    _NESTED_SUBCONFIG_FIELDS,
)
from maxtext.m3.models.model_config import ModelConfig
from maxtext.m3.models.registry import normalize_model_name

logger = logging.getLogger(__name__)


class ConfigMigrationError(ValueError):
  """Raised when a user passes a `pyconfig` knob (via CLI, env, or YAML) with no m3 v1 equivalent."""


_MESH_AXES: tuple[str, ...] = ("data", "fsdp", "context", "tensor", "expert")

# Architecture overrides: `ModelConfig` field -> candidate `MaxTextConfig` keys.
_ARCH_FIELD_MAP: dict[str, tuple[str, ...]] = {
    "vocab_size": ("vocab_size",),
    "hidden_size": ("emb_dim", "base_emb_dim"),
    "intermediate_size": ("mlp_dim", "base_mlp_dim"),
    "moe_intermediate_size": ("moe_mlp_dim", "base_moe_mlp_dim"),
    "num_hidden_layers": ("num_decoder_layers", "base_num_decoder_layers"),
    "num_attention_heads": ("num_query_heads", "base_num_query_heads"),
    "num_key_value_heads": ("num_kv_heads", "base_num_kv_heads"),
    "head_dim": ("head_dim",),
    "n_routed_experts": ("num_experts",),
    "num_experts_per_tok": ("num_experts_per_tok",),
    "n_shared_experts": ("shared_experts",),
    "router_aux_loss_coef": ("load_balance_loss_weight",),
    "rms_norm_eps": ("normalization_layer_epsilon",),
    "rope_theta": ("rope_max_timescale",),
    "tie_word_embeddings": ("logits_via_embedding",),
    "use_qk_norm": ("use_qk_norm",),
    "float32_logits": ("float32_logits",),
    "logits_dot_in_fp32": ("logits_dot_in_fp32",),
    "cast_logits_to_fp32": ("cast_logits_to_fp32",),
    "normalize_embedding_logits": ("normalize_embedding_logits",),
    "final_logits_soft_cap": ("final_logits_soft_cap",),
}

# Explicit field renames (`m3_field -> pyconfig_field`). Same-name fields are discovered from Pydantic
# `model_fields` intersected with `MaxTextConfig.model_fields` so `_TRANSLATED_PYCONFIG_KEYS` cannot drift.
_OPT_RENAMES: dict[str, str] = {
    "optimizer_type": "opt_type",
    "cosine_learning_rate_final_fraction": "learning_rate_final_fraction",
    "freeze_vision_encoder": "freeze_vision_encoder_params",
}
_CKPT_RENAMES: dict[str, str] = {"todelete_full_path": "checkpoint_todelete_full_path"}
_TOP_RENAMES: dict[str, str] = {
    "output_dir": "base_output_directory",
    "seed": "init_weights_seed",
    "max_seq_length": "max_target_length",
    "num_vocab_tiles": "num_vocab_tiling",
    "router_bias_update_rate": "routed_bias_update_rate",
}

_MAXTEXT_FIELDS: frozenset[str] = frozenset(MaxTextConfig.model_fields)
_SUBCONFIG_CLASSES: tuple[type[BaseModel], ...] = (
    TransformConfig,
    OptimizerConfig,
    CheckpointConfig,
    MetricsConfig,
    DataPipelineConfig,
    AOT,
    OrbaxStorage,
    Profiling,
)

_TRANSLATED_PYCONFIG_KEYS: frozenset[str] = frozenset(
    {
        "model_name",
        "use_m3_model",
        "override_model_config",
        "use_qwix_quantization",
        "dtype",
        "weight_dtype",
        "shard_mode",
        "shard_optimizer_over_data",
        *(f"{p}_{axis}_parallelism" for p in ("ici", "dcn") for axis in _MESH_AXES),
        *(k for py_keys in _ARCH_FIELD_MAP.values() for k in py_keys),
        *_OPT_RENAMES.values(),
        *_CKPT_RENAMES.values(),
        *_TOP_RENAMES.values(),
        *(f for f in TrainingConfig.model_fields if f in _MAXTEXT_FIELDS and f not in _NESTED_SUBCONFIG_FIELDS),
        *(f for cls in _SUBCONFIG_CLASSES for f in cls.model_fields if f in _MAXTEXT_FIELDS),
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
        "save_checkpoint_on_completion",
        "enable_single_controller",
        "tokenizer_type",
        "use_multimodal",
        "decoder_block",
        "mlp_activations",
        "enable_dropout",
    }
)

# Non-goal flags audited by live value on `pyd_cfg` (flag -> inactive default value),
# catching custom `.yml` overrides even when `pyconfig.py` omits them from `model_fields_set`.
_UNSUPPORTED_FLAGS: dict[str, bool | int | str | None] = {
    # Goodput, telemetry, & MLPerf
    "enable_goodput_recording": False,
    "monitor_goodput": False,
    "enable_pathways_goodput": False,
    "enable_checkpoint_cloud_logger": False,
    "enable_wandb": False,
    "use_vertex_tensorboard": False,
    "managed_mldiagnostics": False,
    "enable_mllog": False,
    # Advanced checkpoint resilience & Pathways colocated checkpointing
    "enable_emergency_checkpoint": False,
    "enable_multi_tier_checkpointing": False,
    "enable_continuous_checkpointing": False,
    "colocated_python_checkpointing": False,
    "checkpoint_todelete_subdir": None,
    # Stability / post-training / offload / elastic / rampup / DiLoCo / MoE fallback
    "use_qk_clip": False,
    "elastic_enabled": False,
    "grain_use_elastic_iterator": False,
    "enable_rampup_batch_size": False,
    "enable_diloco": False,
    "parameter_memory_host_offload": False,
    "use_dpo": False,
    "moe_dropless_fallback": None,
    # Custom meshes & extended parallelism axes unsupported in v1
    "custom_mesh": "",
    "custom_mesh_and_rule": "",
    "custom_mesh_and_rule_for_eval": "",
    "ici_pipeline_parallelism": 1,
    "dcn_pipeline_parallelism": 1,
    "ici_context_usp_ulysses_parallelism": 1,
    "dcn_context_usp_ulysses_parallelism": 1,
    "ici_fsdp_transpose_parallelism": 1,
    "dcn_fsdp_transpose_parallelism": 1,
}

_NULLABLE_STR_TARGETS: frozenset[str] = frozenset({"metrics_file", "todelete_full_path"})
_NON_NULL_STR_TARGETS: frozenset[str] = frozenset(
    {
        "output_dir",
        "load_full_state_path",
        "load_parameters_path",
        "dataset_path",
        "dataset_name",
        "grain_train_files",
        "grain_eval_files",
        "tokenizer_path",
    }
)


def _enum_str(val: Any) -> str:
  return str(getattr(val, "value", val))


def _str_tuple(val: Any) -> tuple[str, ...]:
  if not val:
    return ()
  return (val,) if isinstance(val, str) else tuple(val)


def _check_non_goal_values(pyd_cfg: Any, *, mode: Literal["train", "aot"]) -> list[str]:
  """Inspects live config values against inactive defaults to catch custom `.yml` overrides."""
  violations: list[str] = []
  for flag, inactive in _UNSUPPORTED_FLAGS.items():
    val = getattr(pyd_cfg, flag, inactive)
    if isinstance(inactive, bool):
      if bool(val):
        violations.append(f"{flag}={val!r}")
    elif isinstance(inactive, int):
      if val is not None and (isinstance(val, bool) or val != inactive):
        violations.append(f"{flag}={val!r}")
    elif val is not None and _enum_str(val).strip() not in ("", "None"):
      violations.append(f"{flag}={_enum_str(val).strip()!r}")

  opt_val = _enum_str(getattr(pyd_cfg, "opt_type", "adamw")).strip().lower()
  if opt_val not in ("adamw", "adam_pax", "sgd"):
    violations.append(f"opt_type={opt_val!r}")

  eval_rules = getattr(pyd_cfg, "logical_axis_rules_for_eval", None)
  train_rules = getattr(pyd_cfg, "logical_axis_rules", None)
  if eval_rules and train_rules is not None and eval_rules != train_rules:
    violations.append("logical_axis_rules_for_eval=<custom>")

  lora_cfg = getattr(pyd_cfg, "lora", None)
  if lora_cfg is not None and bool(getattr(lora_cfg, "enable_lora", False)):
    violations.append("lora.enable_lora=True")

  compiled_file = str(getattr(pyd_cfg, "compiled_trainstep_file", "") or "")
  if mode == "train" and compiled_file:
    violations.append(f"compiled_trainstep_file={compiled_file!r} (allowed only in mode='aot')")

  ckpt_layout = str(getattr(pyd_cfg, "source_checkpoint_layout", "orbax") or "orbax")
  if ckpt_layout not in ("orbax", "safetensors_dynamic"):
    violations.append(f"source_checkpoint_layout={ckpt_layout!r}")

  raw_quant = _enum_str(getattr(pyd_cfg, "quantization", "") or "").strip()
  if raw_quant and not getattr(pyd_cfg, "use_qwix_quantization", True):
    violations.append(f"quantization={raw_quant!r} requires use_qwix_quantization=True")

  return violations


def _extract_model_kwargs(
    cls: type[BaseModel],
    pyd_cfg: Any,
    renames: dict[str, str] | None = None,
    skip: frozenset[str] = frozenset(),
) -> dict[str, Any]:
  """Extracts matching fields for `cls` from `pyd_cfg` using `cls.model_fields` + `renames`."""
  rename_map = renames or {}
  out: dict[str, Any] = {}
  for m3_key in cls.model_fields:
    if m3_key in skip:
      continue
    py_key = rename_map.get(m3_key, m3_key)
    if py_key not in _MAXTEXT_FIELDS or not hasattr(pyd_cfg, py_key):
      continue
    val = getattr(pyd_cfg, py_key)
    if m3_key in _NULLABLE_STR_TARGETS:
      out[m3_key] = None if val in (None, "", "None") else str(val)
    elif m3_key in _NON_NULL_STR_TARGETS:
      out[m3_key] = "" if val is None else str(val)
    else:
      out[m3_key] = val
  return out


def _extract_mesh_dict(pyd_cfg: Any, prefix: str, default_auto: dict[str, int]) -> dict[str, int]:
  """Extracts non-unit 5-axis parallelism overrides from `pyd_cfg` for `prefix` ('ici' or 'dcn')."""
  mapping = {axis: int(val) for axis in _MESH_AXES if (val := getattr(pyd_cfg, f"{prefix}_{axis}_parallelism", 1)) != 1}
  return mapping or dict(default_auto)


def from_pyconfig(
    cfg: Any,
    *,
    mode: Literal["train", "aot"] = "train",
) -> tuple[ModelConfig, TrainingConfig]:
  """Translates a `MaxTextConfig` or `HyperParameters` instance into `(ModelConfig, TrainingConfig)`.

  Performs two deterministic fail-fast checks before translating:
    1. Live non-goal value audit: inspects active values on `cfg` against inactive defaults so
       unsupported features set in custom `.yml` files (omitted from `model_fields_set`) still fail fast.
    2. Explicit override audit (`model_fields_set`): classifies user-passed CLI / kwarg / env
       keys into translated, ignored meta/env no-ops, or rejected (`ConfigMigrationError`).

  Args:
    cfg: Parsed `MaxTextConfig` or `HyperParameters` instance.
    mode: `'train'` (rejects runtime loading of `compiled_trainstep_file`) or `'aot'` (allows
      saving serialized AOT executables via `compiled_trainstep_file`).
  """
  if mode not in ("train", "aot"):
    raise ValueError(f"`mode` must be 'train' or 'aot'; got {mode!r}.")

  pyd_cfg = getattr(cfg, "_pydantic_config", cfg)
  explicit_keys = set(getattr(pyd_cfg, "model_fields_set", ()))

  violations = _check_non_goal_values(pyd_cfg, mode=mode)
  for key in sorted(k for k in explicit_keys if k not in _TRANSLATED_PYCONFIG_KEYS and k not in _IGNORED_PYCONFIG_KEYS):
    if not any(v.startswith(f"{key}=") for v in violations):
      violations.append(f"{key}={getattr(pyd_cfg, key, None)!r}")
  if violations:
    raise ConfigMigrationError(
        f"Unsupported pyconfig setting(s) [{', '.join(violations)}] have no m3 v1 equivalent and cannot be silently "
        "ignored. Remove unsupported flags or configure directly via `--model_name=<model>` with `ArgumentParser`."
    )

  model_name = normalize_model_name(str(getattr(pyd_cfg, "model_name", "qwen3-0.6b")))
  try:
    base_model_cfg = ModelConfig.from_model_name(model_name)
  except KeyError as exc:
    raise ConfigMigrationError(str(exc)) from exc

  model_fields = set(type(base_model_cfg).model_fields)
  param_dtype = DType(_enum_str(getattr(pyd_cfg, "weight_dtype", "float32")))
  compute_dtype = DType(_enum_str(getattr(pyd_cfg, "dtype", "bfloat16")))
  arch_overrides: dict[str, Any] = {"param_dtype": param_dtype, "compute_dtype": compute_dtype}
  for m3_key, py_candidates in _ARCH_FIELD_MAP.items():
    if any(cand in explicit_keys for cand in py_candidates):
      val = getattr(pyd_cfg, py_candidates[0], None)
      if m3_key not in model_fields:
        raise ConfigMigrationError(
            f"Explicit architecture override {py_candidates[0]}={val!r} is not valid for "
            f"model {model_name!r} ({type(base_model_cfg).__name__})."
        )
      if val is not None:
        arch_overrides[m3_key] = val

  try:
    model_cfg = ModelConfig.from_model_name(model_name, **arch_overrides)
  except (ValidationError, ValueError) as exc:
    raise ConfigMigrationError(f"Invalid architecture override for {model_name!r}: {exc}") from exc

  # Default to explicit sharding for m3 unless the user explicitly requested `shard_mode=auto`.
  shard_mode: Literal["explicit", "auto"] = "explicit"
  if "shard_mode" in explicit_keys:
    shard_mode = "explicit" if "explicit" in _enum_str(getattr(pyd_cfg, "shard_mode", "explicit")).lower() else "auto"

  tx_kwargs = _extract_model_kwargs(TransformConfig, pyd_cfg)
  if "remat_policy" in tx_kwargs:
    raw_remat = tx_kwargs["remat_policy"]
    remat_str = _enum_str(raw_remat).strip().lower() if raw_remat is not None else "none"
    tx_kwargs["remat_policy"] = "none" if remat_str in ("", "none") else RematPolicy(remat_str)
  if "quantization" in tx_kwargs:
    raw_quant = tx_kwargs["quantization"]
    tx_kwargs["quantization"] = (
        QuantizationType.NONE if raw_quant in (None, "") else QuantizationType(_enum_str(raw_quant))
    )

  opt_kwargs = _extract_model_kwargs(OptimizerConfig, pyd_cfg, renames=_OPT_RENAMES)
  if "optimizer_type" in opt_kwargs:
    opt_kwargs["optimizer_type"] = _enum_str(opt_kwargs["optimizer_type"]).strip().lower()
  if "lr_schedule_type" in opt_kwargs:
    lr_str = _enum_str(opt_kwargs["lr_schedule_type"]).strip().lower()
    opt_kwargs["lr_schedule_type"] = lr_str if lr_str in ("constant", "linear") else LearningRateScheduleType(lr_str)
  if "mu_dtype" in opt_kwargs:
    mu_raw = opt_kwargs["mu_dtype"]
    mu_str = _enum_str(mu_raw).strip() if mu_raw not in (None, "") else ""
    opt_kwargs["mu_dtype"] = DType(mu_str) if mu_str else DType.FLOAT32
  for mask_key in ("adamw_mask", "trainable_parameters_mask"):
    if mask_key in opt_kwargs:
      opt_kwargs[mask_key] = _str_tuple(opt_kwargs[mask_key])

  ckpt_kwargs = _extract_model_kwargs(CheckpointConfig, pyd_cfg, renames=_CKPT_RENAMES)
  for int_opt in ("max_num_checkpoints_to_keep", "async_checkpointing_timeout_secs"):
    if ckpt_kwargs.get(int_opt) in ("", "None"):
      ckpt_kwargs[int_opt] = None

  data_kwargs = _extract_model_kwargs(DataPipelineConfig, pyd_cfg)
  if "dataset_type" in data_kwargs:
    data_kwargs["dataset_type"] = DatasetType(_enum_str(data_kwargs["dataset_type"]))
  packing_type = str(data_kwargs.get("grain_packing_type", "first_fit"))
  if packing_type not in ("first_fit", "best_fit"):
    if "grain_packing_type" in explicit_keys:
      raise ConfigMigrationError(
          f"grain_packing_type={packing_type!r} is not supported in m3; use 'first_fit' or 'best_fit'."
      )
    data_kwargs["grain_packing_type"] = "first_fit"

  top_kwargs = _extract_model_kwargs(
      TrainingConfig, pyd_cfg, renames=_TOP_RENAMES, skip=frozenset(_NESTED_SUBCONFIG_FIELDS)
  )
  if not top_kwargs.get("run_name"):
    top_kwargs["run_name"] = "m3_run"
  # Translate `base.yml`'s `-1` disabled-eval sentinels without clamping arbitrary negative values.
  if top_kwargs.get("eval_interval") == -1:
    top_kwargs["eval_interval"] = 0
  if top_kwargs.get("eval_steps") == -1:
    top_kwargs["eval_steps"] = 1
  if "grad_dtype" in top_kwargs:
    top_kwargs["grad_dtype"] = DType(_enum_str(top_kwargs["grad_dtype"]))
  if "matmul_precision" in top_kwargs:
    top_kwargs["matmul_precision"] = MatmulPrecision(_enum_str(top_kwargs["matmul_precision"]).lower())

  try:
    train_cfg = TrainingConfig(
        **top_kwargs,
        param_dtype=param_dtype,
        compute_dtype=compute_dtype,
        mesh=MeshConfig(
            ici=_extract_mesh_dict(pyd_cfg, "ici", {"fsdp": -1}),
            dcn=_extract_mesh_dict(pyd_cfg, "dcn", {"data": -1}),
            shard_mode=shard_mode,
            shard_optimizer_over_data=bool(getattr(pyd_cfg, "shard_optimizer_over_data", False)),
        ),
        transforms=TransformConfig(**tx_kwargs),
        optimizer=OptimizerConfig(**opt_kwargs),
        checkpoint=CheckpointConfig(**ckpt_kwargs, checkpoint_format="train_state"),
        metrics=MetricsConfig(**_extract_model_kwargs(MetricsConfig, pyd_cfg)),
        data=DataPipelineConfig(**data_kwargs),
        aot=AOT(**_extract_model_kwargs(AOT, pyd_cfg)),
        orbax_storage=OrbaxStorage(**_extract_model_kwargs(OrbaxStorage, pyd_cfg)),
        profiling=Profiling(**_extract_model_kwargs(Profiling, pyd_cfg)),
    )
  except (ValidationError, ValueError) as exc:
    raise ConfigMigrationError(f"Invalid training config for {model_name!r}: {exc}") from exc

  return model_cfg, train_cfg
