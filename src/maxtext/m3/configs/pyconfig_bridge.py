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

import enum
from typing import Any, Literal

from maxtext.configs.types import MaxTextConfig
from maxtext.m3.configs.training_config import MESH_AXES, TrainingConfig, validate_training_config
from maxtext.m3.models.model_config import PYCONFIG_NONE_VALUES, ModelConfig
from maxtext.m3.models.registry import MODEL_GARDEN_REGISTRY, normalize_model_name
from pydantic import BaseModel, ValidationError


class ConfigMigrationError(ValueError):
  """Raised when a `pyconfig` setting (via CLI, env, or YAML) cannot be expressed as `ModelConfig` / `TrainingConfig`.

  Covers settings with no counterpart in those configs and values they reject.
  """


# `pyconfig` derives these from their `base_*` keys (scaled by `global_parameter_scale`), so overrides read the derived
# value and either key counts as set.
_DERIVED_KEYS: dict[str, str] = {
    "base_emb_dim": "emb_dim",
    "base_mlp_dim": "mlp_dim",
    "base_moe_mlp_dim": "moe_mlp_dim",
    "base_num_decoder_layers": "num_decoder_layers",
    "base_num_query_heads": "num_query_heads",
    "base_num_kv_heads": "num_kv_heads",
}
_BASE_KEYS: dict[str, str] = {derived: base for base, derived in _DERIVED_KEYS.items()}
# Architecture keys any registered family maps (`ModelConfig.PYCONFIG_KEYS`), plus their derived names.
_ARCH_KEYS: frozenset[str] = frozenset(
    {key for factory in MODEL_GARDEN_REGISTRY.values() for key in type(factory()).PYCONFIG_KEYS} | set(_BASE_KEYS)
)

# `TrainingConfig` fields use `base.yml` names, so same-name fields are discovered from Pydantic `model_fields`
# intersected with `MaxTextConfig.model_fields`, and `_TRANSLATED_PYCONFIG_KEYS` cannot drift.
_MAXTEXT_FIELDS: frozenset[str] = frozenset(MaxTextConfig.model_fields)
# `TrainingConfig` sub-configs (`mesh`, `optimizer`, ...) by field name.
_SUBCONFIGS: dict[str, type[BaseModel]] = {
    name: info.annotation
    for name, info in TrainingConfig.model_fields.items()
    if isinstance(info.annotation, type) and issubclass(info.annotation, BaseModel)
}
# Where `MaxTextConfig` and `TrainingConfig` values differ: `base.yml`'s -1 means None for the step fields; `None`
# becomes the m3 default on non-optional fields; and empty strings mean None on optional string fields.
_MINUS_ONE_IS_NONE: frozenset[str] = frozenset({"steps", "eval_interval", "eval_steps", "learning_rate_schedule_steps"})
_NONE_REPLACEMENTS: dict[str, Any] = {
    "metrics_file": "",
    "tokenizer_path": "",
    "remat_policy": "none",
    "quantization": "",
}
_EMPTY_IS_NONE: frozenset[str] = frozenset({"checkpoint_todelete_full_path", "hf_access_token"})

_TRANSLATED_PYCONFIG_KEYS: frozenset[str] = frozenset(
    {
        "model_name",
        "use_m3_model",
        "override_model_config",
        "use_qwix_quantization",
        "dtype",
        "weight_dtype",
        *(f"{p}_{axis}_parallelism" for p in ("ici", "dcn") for axis in MESH_AXES),
        *_ARCH_KEYS,
        *(f for f in TrainingConfig.model_fields if f in _MAXTEXT_FIELDS),
        *(f for cls in _SUBCONFIGS.values() for f in cls.model_fields if f in _MAXTEXT_FIELDS),
    }
)

# Goodput flags default to `True` in `base.yml` outside test runners, so only reject when
# explicitly enabled by the caller in `model_fields_set`.
_EXPLICIT_GOODPUT_FLAGS: frozenset[str] = frozenset({"enable_goodput_recording", "monitor_goodput"})

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
        "decoder_block",
        "vision_encoder_block",
        "mlp_activations",
        "enable_dropout",
        # Checked in `_check_non_goal_values` when context parallelism is on.
        "context_parallel_load_balance",
        "context_parallel_strategy",
    }
)

# Non-goal flags audited by live value on `pyd_cfg` (flag -> inactive default value),
# catching custom `.yml` overrides even when `pyconfig.py` omits them from `model_fields_set`.
_UNSUPPORTED_FLAGS: dict[str, bool | int | str | None] = {
    # Telemetry & MLPerf
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
    # Mesh rules and parallelism axes m3 does not build
    "custom_mesh_and_rule": "",
    "custom_mesh_and_rule_for_eval": "",
    "ici_pipeline_parallelism": 1,
    "dcn_pipeline_parallelism": 1,
    "ici_context_usp_ulysses_parallelism": 1,
    "dcn_context_usp_ulysses_parallelism": 1,
}


def _enum_str(val: Any) -> str:
  return str(getattr(val, "value", val))


def _check_non_goal_values(
    pyd_cfg: Any,
    explicit_keys: set[str],
    *,
    mode: Literal["train", "aot"],
) -> list[str]:
  """Inspects live config values against inactive defaults to catch unsupported settings."""
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

  for gp_flag in sorted(_EXPLICIT_GOODPUT_FLAGS):
    if gp_flag in explicit_keys and bool(getattr(pyd_cfg, gp_flag, False)):
      violations.append(f"{gp_flag}={getattr(pyd_cfg, gp_flag)!r}")

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

  raw_quant = _enum_str(getattr(pyd_cfg, "quantization", "") or "").strip()
  if raw_quant and not getattr(pyd_cfg, "use_qwix_quantization", True):
    violations.append(f"quantization={raw_quant!r} requires use_qwix_quantization=True")

  # m3 context parallelism is all-gather without load balancing (`m3/core/sharding.py`).
  if any(getattr(pyd_cfg, f"{net}_context_parallelism", 1) not in (1, None) for net in ("ici", "dcn")):
    cp_strategy = _enum_str(getattr(pyd_cfg, "context_parallel_strategy", "all_gather") or "all_gather").lower()
    if cp_strategy != "all_gather":
      violations.append(f"context_parallel_strategy={cp_strategy!r}")
    if bool(getattr(pyd_cfg, "context_parallel_load_balance", False)):
      violations.append("context_parallel_load_balance=True")

  return violations


def _extract_model_kwargs(cls: type[BaseModel], pyd_cfg: Any) -> dict[str, Any]:
  """Reads the fields of `cls` that `pyd_cfg` also has (same names as `base.yml`); enums become their values."""
  out: dict[str, Any] = {}
  for key in cls.model_fields:
    if key not in _MAXTEXT_FIELDS or not hasattr(pyd_cfg, key):
      continue
    val = getattr(pyd_cfg, key)
    if isinstance(val, enum.Enum):
      val = val.value
    if key in _MINUS_ONE_IS_NONE and val == -1:
      val = None
    elif key in _EMPTY_IS_NONE and val in ("", "None"):
      val = None
    elif val is None and key in _NONE_REPLACEMENTS:
      val = _NONE_REPLACEMENTS[key]
    out[key] = val
  return out


def _extract_mesh_dict(pyd_cfg: Any, prefix: str) -> dict[str, int]:
  """Reads `<prefix>_<axis>_parallelism` for each `MESH_AXES` axis present on `pyd_cfg` (`prefix` is 'ici' or 'dcn')."""
  return {
      axis: int(val) for axis in MESH_AXES if (val := getattr(pyd_cfg, f"{prefix}_{axis}_parallelism", None)) is not None
  }


def _build_model_config(model_name: str, pyd_cfg: Any, explicit_keys: set[str]) -> ModelConfig:
  """Resolves `model_name` via `MODEL_GARDEN_REGISTRY` and applies explicit architecture overrides."""
  try:
    base_model_cfg = ModelConfig.from_model_name(model_name)
  except KeyError as exc:
    raise ConfigMigrationError(str(exc)) from exc

  model_cls = type(base_model_cfg)
  overrides: dict[str, Any] = {
      "param_dtype": _enum_str(getattr(pyd_cfg, "weight_dtype", "float32")),
      "compute_dtype": _enum_str(getattr(pyd_cfg, "dtype", "bfloat16")),
  }
  derived: dict[str, Any] = {}  # Derived values are checked against the built config instead of set.
  for key in sorted({_BASE_KEYS.get(k, k) for k in explicit_keys & _ARCH_KEYS} - {"model_name"}):
    value = getattr(pyd_cfg, _DERIVED_KEYS.get(key, key), None)
    if value is None:
      value = getattr(pyd_cfg, key, None)
    field = model_cls.PYCONFIG_KEYS.get(key)
    if field is None:
      raise ConfigMigrationError(
          f"Explicit architecture override {key}={value!r} is not valid for model {model_name!r} ({model_cls.__name__})."
      )
    if value is None:
      continue
    value = getattr(value, "value", value)
    if key in PYCONFIG_NONE_VALUES and value == PYCONFIG_NONE_VALUES[key]:
      value = None
    if field not in model_cls.model_fields:  # Derived property, e.g. GraniteMoE `head_dim`.
      derived[field] = value
    elif field in overrides and overrides[field] != value:
      raise ConfigMigrationError(
          f"Conflicting overrides for {model_cls.__name__}.{field}: {overrides[field]!r} vs {key}={value!r}."
      )
    else:
      overrides[field] = value

  try:
    model_cfg = ModelConfig.from_model_name(model_name, **overrides)
  except (ValidationError, ValueError) as exc:
    raise ConfigMigrationError(f"Invalid architecture override for {model_name!r}: {exc}") from exc

  for field, value in derived.items():
    if getattr(model_cfg, field) != value:
      raise ConfigMigrationError(
          f"Explicit {field}={value!r} conflicts with the value {model_cls.__name__} derives "
          f"({getattr(model_cfg, field)!r})."
      )
  return model_cfg


def from_pyconfig(
    cfg: Any,
    *,
    mode: Literal["train", "aot"] = "train",
) -> tuple[ModelConfig, TrainingConfig]:
  """Translates a `MaxTextConfig` or `HyperParameters` instance into `(ModelConfig, TrainingConfig)`.

  Performs two deterministic fail-fast checks before translating:
    1. Live non-goal value audit: inspects active values on `cfg` against inactive defaults so
       unsupported features set in custom `.yml` files (omitted from `model_fields_set`) fail fast.
    2. Explicit override audit (`model_fields_set`): classifies caller-passed CLI / kwarg / env
       keys into translated, ignored no-ops, or rejected (`ConfigMigrationError`).

  Args:
    cfg: Parsed `MaxTextConfig` or `HyperParameters` instance.
    mode: `'train'` (rejects `compiled_trainstep_file`) or `'aot'` (allows saving compiled AOT artifacts).
  """
  if mode not in ("train", "aot"):
    raise ValueError(f"`mode` must be 'train' or 'aot'; got {mode!r}.")

  pyd_cfg = getattr(cfg, "_pydantic_config", cfg)
  explicit_keys = set(getattr(pyd_cfg, "model_fields_set", ()))

  violations = _check_non_goal_values(pyd_cfg, explicit_keys, mode=mode)
  known_keys = _TRANSLATED_PYCONFIG_KEYS | _IGNORED_PYCONFIG_KEYS | _EXPLICIT_GOODPUT_FLAGS
  for key in sorted(k for k in explicit_keys if k not in known_keys):
    if not any(v.startswith(f"{key}=") for v in violations):
      violations.append(f"{key}={getattr(pyd_cfg, key, None)!r}")
  if violations:
    raise ConfigMigrationError(
        f"Unsupported pyconfig setting(s) [{', '.join(violations)}] have no `ModelConfig` / `TrainingConfig` "
        "equivalent and cannot be silently ignored. Remove the unsupported flags, or build `ModelConfig` / "
        "`TrainingConfig` directly."
    )

  model_name = normalize_model_name(str(getattr(pyd_cfg, "model_name", "default")))
  model_cfg = _build_model_config(model_name, pyd_cfg, explicit_keys)

  try:
    subconfigs = {name: _extract_model_kwargs(cls, pyd_cfg) for name, cls in _SUBCONFIGS.items()}
    subconfigs["mesh"].update(ici=_extract_mesh_dict(pyd_cfg, "ici"), dcn=_extract_mesh_dict(pyd_cfg, "dcn"))
    train_cfg = TrainingConfig(**_extract_model_kwargs(TrainingConfig, pyd_cfg), **subconfigs)
    validate_training_config(model_cfg, train_cfg)
  except (ValidationError, ValueError) as exc:
    raise ConfigMigrationError(f"Invalid training config for {model_name!r}: {exc}") from exc

  return model_cfg, train_cfg
