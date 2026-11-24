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

"""CLI parser for `ModelConfig` and `TrainingConfig`.

Supported forms:
  * `--model_name=qwen3-0.6b` or `--model_name qwen3-0.6b` (required, given once)
  * `--steps=500` (top-level `TrainingConfig` fields)
  * `--model.num_hidden_layers=4` (`ModelConfig` overrides)
  * `--transforms.scan_layers=false`, `--mesh.ici.tensor=2` (nested fields, any depth)
  * `--transforms='{"scan_layers": false}'`, `--model='{"num_hidden_layers": 2}'` (JSON object for a whole config)
  * `--transforms.named_scopes` (bare boolean shorthand for `=true`)

Hyphens in flag names become underscores. Values are converted using each field's Pydantic type; `none` / `null`
sets an optional field to `None`, and list, tuple, dict, and sub-config fields take JSON.
"""

from collections.abc import Iterator, Sequence
import json
import sys
import types
from typing import Any, Union, get_args, get_origin

from maxtext.m3.configs.training_config import TrainingConfig, validate_training_config
from maxtext.m3.models.model_config import ModelConfig
from pydantic import BaseModel, SecretStr, TypeAdapter


def _union_members(annotation: Any) -> tuple[Any, ...]:
  """Flattens `Union` / `X | Y` annotations into their member types."""
  return get_args(annotation) if get_origin(annotation) in (Union, types.UnionType) else (annotation,)


def _submodel(annotation: Any) -> type[BaseModel] | None:
  """Returns the `BaseModel` subclass a field holds (including `X | None`), else None."""
  subs = [m for m in _union_members(annotation) if isinstance(m, type) and issubclass(m, BaseModel)]
  return subs[0] if len(subs) == 1 else None


def _parse_value(raw: str | None, flag: str, annotation: Any) -> Any:
  """Converts a raw CLI value to the type of the target field."""
  members = _union_members(annotation)
  if raw is None:
    if bool in members:
      return True
    raise ValueError(f"Flag `--{flag}` requires a value.")
  value = raw.strip()
  if types.NoneType in members and value.lower() in ("none", "null"):
    return None
  sub = _submodel(annotation)
  if sub is not None and not value.startswith("{"):
    raise ValueError(f"Flag `--{flag}` expects a JSON object for {sub.__name__}.")
  is_string = all(m in (str, SecretStr, types.NoneType) for m in members)
  if value[:1] in ("[", "{") and not is_string:
    try:
      return json.loads(value)
    except json.JSONDecodeError as exc:
      raise ValueError(f"Invalid JSON value for `--{flag}`: {raw!r} ({exc.msg}).") from exc
  return TypeAdapter(annotation).validate_strings(value)


def _tokenize(argv: Sequence[str] | None) -> list[tuple[str, str | None]]:
  """Splits `--flag=value`, `--flag value`, and bare `--flag` tokens into `(name, raw_value)` pairs."""
  pairs: list[tuple[str, str | None]] = []
  for tok in sys.argv[1:] if argv is None else argv:
    tok = tok.strip()
    if tok.startswith("--"):
      name, sep, value = tok[2:].partition("=")
      name = name.strip().replace("-", "_")
      if not name:
        raise ValueError(f"Empty flag name in argument {tok!r}.")
      pairs.append((name, value if sep else None))
    elif pairs and pairs[-1][1] is None:  # Value for the preceding bare `--flag`.
      pairs[-1] = (pairs[-1][0], tok)
    elif tok:
      raise ValueError(f"Expected `--flag` argument, got {tok!r}.")
  return pairs


def _deep_merge(dst: dict[str, Any], src: dict[str, Any]) -> None:
  """Merges `src` into `dst` in place, recursing into nested dicts."""
  for key, value in src.items():
    if isinstance(value, dict) and isinstance(dst.get(key), dict):
      _deep_merge(dst[key], value)
    else:
      dst[key] = value


def _field_annotation(cls: type[BaseModel], name: str, scope: str) -> Any:
  if name not in cls.model_fields:
    raise ValueError(f"Unknown field {name!r} on `{scope}` ({cls.__name__}); valid fields: {sorted(cls.model_fields)}.")
  return cls.model_fields[name].annotation


def _set_override(
    overrides: dict[str, Any], cls: type[BaseModel], scope: str, path: Sequence[str], raw: str | None, flag: str
) -> None:
  """Parses `raw` for the field at `path` under `cls` and stores it in the nested `overrides` dict."""
  node = overrides
  for part in path[:-1]:
    sub = _submodel(_field_annotation(cls, part, scope))
    scope = f"{scope}.{part}" if scope else part
    if sub is None:
      raise ValueError(f"`{scope}` is not a sub-config, so `--{flag}` is invalid.")
    node = node.setdefault(part, {})
    cls = sub
  if not path:  # The whole config as a JSON object.
    _deep_merge(node, _parse_value(raw, flag, cls))
    return
  leaf = path[-1]
  value = _parse_value(raw, flag, _field_annotation(cls, leaf, scope))
  if isinstance(value, dict) and isinstance(node.get(leaf), dict):
    _deep_merge(node[leaf], value)
  else:
    node[leaf] = value


def _leaf_paths(cls: type[BaseModel], prefix: str) -> Iterator[str]:
  """Yields the dotted path of every non-config field under `cls`."""
  for name, info in cls.model_fields.items():
    path = f"{prefix}.{name}" if prefix else name
    sub = _submodel(info.annotation)
    if sub is None:
      yield path
    else:
      yield from _leaf_paths(sub, path)


def _unknown_flag_error(name: str, model_cls: type[ModelConfig]) -> ValueError:
  """Builds the error for an unrecognized flag, suggesting full flags that end in `name` (e.g. `ici.tensor`)."""
  paths = (*_leaf_paths(model_cls, "model"), *_leaf_paths(TrainingConfig, ""))
  hints = [f"`--{path}`" for path in paths if path.endswith(f".{name}")]
  hint_msg = f" Did you mean {', '.join(hints)}?" if hints else ""
  return ValueError(f"Unrecognized CLI flag `--{name}`.{hint_msg}")


def parse_args(argv: Sequence[str] | None = None) -> tuple[ModelConfig, TrainingConfig]:
  """Parses CLI flags into validated `(ModelConfig, TrainingConfig)` instances.

  Args:
    argv: Flags without the program name. Defaults to `sys.argv[1:]`.

  Returns:
    The `ModelConfig` preset for `--model_name` with `--model.*` overrides, and the `TrainingConfig`.
  """
  pairs = _tokenize(argv)
  model_names = [raw for name, raw in pairs if name == "model_name"]
  flags = [(name, raw) for name, raw in pairs if name != "model_name"]
  if len(model_names) != 1:
    raise ValueError("Pass `--model_name` exactly once (e.g. `--model_name=qwen3-0.6b`).")
  model_name = (model_names[0] or "").strip()
  if not model_name:
    raise ValueError("Flag `--model_name` requires a non-empty value.")

  # Build the preset to get its family class, whose field types decide how `--model.*` values are parsed.
  model_cls = type(ModelConfig.from_model_name(model_name))
  model_overrides: dict[str, Any] = {}
  train_overrides: dict[str, Any] = {}
  for name, raw in flags:
    path = name.split(".")
    if path[0] == "model":
      _set_override(model_overrides, model_cls, "model", path[1:], raw, name)
    elif path[0] in TrainingConfig.model_fields:
      _set_override(train_overrides, TrainingConfig, "", path, raw, name)
    else:
      raise _unknown_flag_error(name, model_cls)
  if "model_name" in model_overrides:
    raise ValueError("Set the model with `--model_name`, not `--model.model_name`.")

  # Rebuild through the registry so overrides are re-validated and computed fields are recomputed.
  model_cfg = ModelConfig.from_model_name(model_name, **model_overrides)
  train_cfg = TrainingConfig(**train_overrides)
  validate_training_config(model_cfg, train_cfg)
  return model_cfg, train_cfg
