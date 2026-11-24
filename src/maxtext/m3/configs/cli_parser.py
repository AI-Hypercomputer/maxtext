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

"""Zero-YAML CLI parser for `ModelConfig` and `TrainingConfig`."""

from collections.abc import Sequence
import json
import math
import os
import sys
import types
from typing import Any, Union, get_args, get_origin

from maxtext.m3.configs.training_config import TrainingConfig, validate_training_config
from maxtext.m3.models.model_config import ModelConfig
from pydantic import BaseModel, SecretStr

_SHARED_DTYPE_FIELDS = frozenset({"param_dtype", "compute_dtype"})


def _subconfig_models() -> dict[str, type[BaseModel]]:
  """Returns `{field_name: subconfig_cls}` for every nested `BaseModel` field on `TrainingConfig`."""
  return {
      name: info.annotation
      for name, info in TrainingConfig.model_fields.items()
      if isinstance(info.annotation, type) and issubclass(info.annotation, BaseModel)
  }


def _annotation_members(annotation: Any) -> tuple[Any, ...]:
  """Flattens `Union` / `X | Y` annotations into their member types."""
  origin = get_origin(annotation)
  if origin in (Union, types.UnionType):
    return get_args(annotation)
  return (annotation,)


def _allows_none(annotation: Any) -> bool:
  """Returns True when `annotation` allows `None`."""
  return types.NoneType in _annotation_members(annotation)


def _is_bool_field(annotation: Any) -> bool:
  """Returns True when `annotation` accepts a boolean shorthand flag."""
  return bool in _annotation_members(annotation)


def _is_string_only_field(annotation: Any) -> bool:
  """Returns True when `annotation` is `str` / `SecretStr` (optionally `| None`)."""
  non_none = [m for m in _annotation_members(annotation) if m is not types.NoneType]
  return bool(non_none) and all(m in (str, SecretStr) for m in non_none)


def _strip_matching_quotes(value: str) -> str:
  """Strips one pair of matching surrounding quotes if present."""
  if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
    return value[1:-1]
  return value


def _parse_cli_value(raw_val: str, flag_name: str, annotation: Any) -> Any:
  """Parses a raw CLI string according to the target Pydantic field annotation."""
  stripped = raw_val.strip()
  lowered = stripped.lower()

  if _is_string_only_field(annotation):
    if _allows_none(annotation) and lowered in ("none", "null"):
      return None
    return _strip_matching_quotes(stripped)

  if _is_bool_field(annotation):
    if lowered == "true":
      return True
    if lowered == "false":
      return False
  if _allows_none(annotation) and lowered in ("none", "null"):
    return None

  unquoted = _strip_matching_quotes(stripped)
  if (unquoted.startswith("{") and unquoted.endswith("}")) or (unquoted.startswith("[") and unquoted.endswith("]")):
    try:
      return json.loads(unquoted)
    except json.JSONDecodeError as exc:
      raise ValueError(f"Invalid JSON value for `--{flag_name}`: {raw_val!r} ({exc.msg}).") from exc

  try:
    return int(unquoted)
  except ValueError:
    pass
  try:
    parsed_float = float(unquoted)
    if math.isfinite(parsed_float):
      return parsed_float
  except ValueError:
    pass
  return unquoted


def _resolve_value(raw_val: str | None, flag_name: str, annotation: Any) -> Any:
  """Resolves bare boolean shorthand or parses a raw CLI string value."""
  if raw_val is None:
    if _is_bool_field(annotation):
      return True
    raise ValueError(f"Flag `--{flag_name}` requires a value.")
  return _parse_cli_value(raw_val, flag_name, annotation)


def _looks_like_program_token(token: str) -> bool:
  """Returns True when `token` matches an `argv[0]` program path."""
  if token.lower().endswith((".yml", ".yaml", ".json")):
    return False
  return token == "" or token.endswith(".py") or os.sep in token


def _tokenize_argv(argv: Sequence[str] | None) -> list[tuple[str, str | None]]:
  """Parses `--flag=value`, `--flag value`, and bare `--flag` tokens into `(key, raw_value)` pairs."""
  tokens = list(sys.argv[1:] if argv is None else argv)
  if argv is not None and tokens and not tokens[0].startswith("-") and _looks_like_program_token(tokens[0]):
    tokens = tokens[1:]

  pairs: list[tuple[str, str | None]] = []
  idx = 0
  while idx < len(tokens):
    tok = tokens[idx].strip()
    if not tok:
      idx += 1
      continue
    if not tok.startswith("--"):
      raise ValueError(f"Expected `--flag` argument in ArgumentParser, got {tok!r}.")
    body = tok[2:]
    if "=" in body:
      raw_key, val_str = body.split("=", 1)
      key = raw_key.strip().replace("-", "_")
      if not key:
        raise ValueError(f"Empty flag name in argument {tok!r}.")
      pairs.append((key, val_str))
      idx += 1
    else:
      key = body.strip().replace("-", "_")
      if not key:
        raise ValueError(f"Empty flag name in argument {tok!r}.")
      if idx + 1 < len(tokens) and not tokens[idx + 1].startswith("--"):
        pairs.append((key, tokens[idx + 1]))
        idx += 2
      else:
        pairs.append((key, None))
        idx += 1
  return pairs


def _suggest_scoped_flags(
    key: str,
    model_fields: set[str],
    subconfigs: dict[str, type[BaseModel]],
) -> list[str]:
  """Returns valid dotted `--<scope>.<key>` suggestions for an unscoped leaf flag."""
  suggestions: list[str] = []
  if key in model_fields:
    suggestions.append(f"--model.{key}")
  for sub_name, sub_cls in sorted(subconfigs.items()):
    if key in sub_cls.model_fields:
      suggestions.append(f"--{sub_name}.{key}")
  return suggestions


class ArgumentParser:
  """Zero-YAML CLI parser producing validated `(ModelConfig, TrainingConfig)` pairs.

  Supported forms:
    * `--model_name=qwen3-0.6b` (or `--model_name qwen3-0.6b`; required)
    * `--steps=500`, `--max_target_length=8192` (top-level `TrainingConfig` fields)
    * `--compute_dtype=float32`, `--param_dtype=bfloat16` (shared dtype flags set both configs)
    * `--model.num_hidden_layers=4` (scoped `ModelConfig` overrides)
    * `--transforms.scan_layers=false`, `--mesh.ici='{"fsdp": -1, "tensor": 2}'` (nested sub-config fields)
    * Boolean shorthand `--transforms.scan_layers` (valid on `bool` fields; equivalent to `=true`)
  """

  def parse_args(self, argv: Sequence[str] | None = None) -> tuple[ModelConfig, TrainingConfig]:
    """Parses CLI tokens into validated `(ModelConfig, TrainingConfig)` instances."""
    raw_pairs = _tokenize_argv(argv)
    model_name: str | None = None
    remaining: list[tuple[str, str | None]] = []

    for key, raw_val in raw_pairs:
      if key == "model_name":
        if raw_val is None or not _strip_matching_quotes(raw_val.strip()):
          raise ValueError("Flag `--model_name` requires a non-empty value.")
        model_name = _strip_matching_quotes(raw_val.strip())
      else:
        remaining.append((key, raw_val))

    if model_name is None:
      raise ValueError("Missing required `--model_name <name>` argument (e.g. `--model_name=qwen3-0.6b`).")

    base_model_cfg = ModelConfig.from_model_name(model_name)
    model_cls = type(base_model_cfg)
    model_fields: dict[str, Any] = dict(model_cls.model_fields)
    train_fields: dict[str, Any] = dict(TrainingConfig.model_fields)
    subconfigs = _subconfig_models()

    model_overrides: dict[str, Any] = {}
    train_overrides: dict[str, Any] = {}

    for key, raw_val in remaining:
      if "." in key:
        parts = key.split(".")
        if len(parts) != 2 or not parts[0] or not parts[1]:
          raise ValueError(f"Unknown dotted CLI argument `--{key}`.")
        root, leaf = parts
        if root == "model":
          if leaf not in model_fields:
            raise ValueError(
                f"Unknown field {leaf!r} on `model` ({model_cls.__name__}); valid fields: {sorted(model_fields)}."
            )
          ann = model_fields[leaf].annotation
          model_overrides[leaf] = _resolve_value(raw_val, key, ann)
        elif root in subconfigs:
          sub_cls = subconfigs[root]
          if leaf not in sub_cls.model_fields:
            raise ValueError(
                f"Unknown field {leaf!r} on `{root}` ({sub_cls.__name__}); "
                f"valid fields: {sorted(sub_cls.model_fields)}."
            )
          ann = sub_cls.model_fields[leaf].annotation
          train_overrides[root] = {**train_overrides.get(root, {}), leaf: _resolve_value(raw_val, key, ann)}
        else:
          raise ValueError(f"Unknown dotted CLI argument `--{key}`.")
      elif key in train_fields:
        ann = train_fields[key].annotation
        parsed = _resolve_value(raw_val, key, ann)
        if key in subconfigs:
          if not isinstance(parsed, dict):
            raise ValueError(f"Flag `--{key}` expects a JSON object for {subconfigs[key].__name__}.")
          train_overrides[key] = {**train_overrides.get(key, {}), **parsed}
        else:
          train_overrides[key] = parsed
          if key in _SHARED_DTYPE_FIELDS:
            model_overrides[key] = parsed
      else:
        hints = _suggest_scoped_flags(key, set(model_fields), subconfigs)
        hint_msg = f" Did you mean {', '.join(repr(h) for h in hints)}?" if hints else ""
        raise ValueError(f"Unrecognized CLI flag `--{key}` for model {model_name!r} and `TrainingConfig`.{hint_msg}")

    final_model_cfg = ModelConfig.from_model_name(model_name, **model_overrides)
    final_train_cfg = TrainingConfig(**train_overrides)
    validate_training_config(final_model_cfg, final_train_cfg)
    return final_model_cfg, final_train_cfg
