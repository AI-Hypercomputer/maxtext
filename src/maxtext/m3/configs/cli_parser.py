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

"""Pure-Python zero-YAML CLI parser introspecting `ModelConfig` and `TrainingConfig` schemas."""

from collections.abc import Sequence
import json
import os
import sys
from typing import Any

from pydantic import BaseModel

from maxtext.m3.configs.training_config import TrainingConfig, _NESTED_SUBCONFIG_FIELDS
from maxtext.m3.models.model_config import ModelConfig


def _parse_cli_scalar(raw_val: str) -> Any:
  """Parses a CLI value string into bool / int / float / dict / list / str (JSON for containers)."""
  stripped = raw_val.strip()
  lowered = stripped.lower()
  if lowered == "true":
    return True
  if lowered == "false":
    return False
  if lowered in ("none", "null"):
    return None
  if (stripped.startswith("{") and stripped.endswith("}")) or (stripped.startswith("[") and stripped.endswith("]")):
    return json.loads(stripped)
  try:
    return int(stripped)
  except ValueError:
    pass
  try:
    return float(stripped)
  except ValueError:
    pass
  if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in ('"', "'"):
    return stripped[1:-1]
  return stripped


def _collect_leaf_field_map() -> dict[str, str]:
  """Maps unambiguous nested leaf field names to their owning sub-config attribute on `TrainingConfig`."""
  counts: dict[str, list[str]] = {}
  top_fields = set(TrainingConfig.model_fields)
  for sub_name, sub_cls in _NESTED_SUBCONFIG_FIELDS.items():
    for field_name in sub_cls.model_fields:
      counts.setdefault(field_name, []).append(sub_name)
  return {
      field_name: owners[0] for field_name, owners in counts.items() if len(owners) == 1 and field_name not in top_fields
  }


def _looks_like_program_token(token: str) -> bool:
  """Returns True when `token` matches an `app.run`-style leading `argv[0]` program path."""
  if token.lower().endswith((".yml", ".yaml", ".json")):
    return False
  return token == "" or token.endswith(".py") or os.sep in token


class ArgumentParser:
  """Zero-YAML CLI parser producing `(ModelConfig, TrainingConfig)` from `--model_name` and typed flags.

  Supported forms:
    * `--model_name qwen3-0.6b` (or `--model_name=qwen3-0.6b`)
    * `--config_json path/to/config.json` (loads JSON base dictionary before applying CLI flags)
    * `--save_config_json path/to/out.json` (writes resolved JSON provenance)
    * `--steps 500` or `--max_seq_length=8192` (top-level `TrainingConfig` or `ModelConfig` fields)
    * `--transforms.scan_layers=false` or `--mesh.ici='{"fsdp": -1, "tensor": 2}'` (dotted nested fields)
    * `--scan_layers=false` or `--learning_rate=1e-3` (unambiguous leaf fields routed automatically)
    * Boolean shorthand `--enable_checkpointing` (equivalent to `--enable_checkpointing=true`)
  """

  def parse_args(self, argv: Sequence[str] | None = None) -> tuple[ModelConfig, TrainingConfig]:
    """Parses CLI tokens into validated `(ModelConfig, TrainingConfig)` instances."""
    tokens = list(sys.argv[1:] if argv is None else argv)
    if argv is not None and tokens and not tokens[0].startswith("-") and _looks_like_program_token(tokens[0]):
      tokens = tokens[1:]

    raw_pairs: list[tuple[str, Any]] = []
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
        key, val_str = body.split("=", 1)
        raw_pairs.append((key.strip().replace("-", "_"), _parse_cli_scalar(val_str)))
        idx += 1
      elif idx + 1 < len(tokens) and not tokens[idx + 1].startswith("--"):
        raw_pairs.append((body.strip().replace("-", "_"), _parse_cli_scalar(tokens[idx + 1])))
        idx += 2
      else:
        raw_pairs.append((body.strip().replace("-", "_"), True))
        idx += 1

    model_name: str | None = None
    save_json_path: str | None = None
    json_payload: dict[str, Any] = {}
    remaining: list[tuple[str, Any]] = []

    for key, val in raw_pairs:
      if key == "model_name":
        model_name = str(val)
      elif key == "config_json":
        with open(str(val), "r", encoding="utf-8") as f:
          loaded = json.load(f)
        if not isinstance(loaded, dict):
          raise ValueError(f"--config_json file {val!r} must contain a JSON object.")
        json_payload.update(loaded)
      elif key == "save_config_json":
        save_json_path = str(val)
      else:
        remaining.append((key, val))

    saved_model_name = json_payload.pop("model_name", None)
    if model_name is None and saved_model_name is not None:
      model_name = str(saved_model_name)
    if model_name is None:
      raise ValueError("Missing required `--model_name <name>` argument (e.g. `--model_name=qwen3-0.6b`).")

    base_model_cfg = ModelConfig.from_model_name(model_name)
    model_fields = set(type(base_model_cfg).model_fields)
    train_fields = set(TrainingConfig.model_fields)
    leaf_owners = _collect_leaf_field_map()

    model_overrides: dict[str, Any] = dict(json_payload.pop("model", {}))
    # Drop `model_name` from loaded provenance JSON so it never collides with `from_model_name(model_name, **overrides)`.
    model_overrides.pop("model_name", None)
    train_overrides: dict[str, Any] = dict(json_payload)

    for key, val in remaining:
      if "." in key:
        parts = key.split(".")
        if len(parts) != 2:
          raise ValueError(f"Unknown dotted CLI argument `--{key}`.")
        root, leaf = parts
        if root == "model":
          model_overrides[leaf] = val
        elif root in _NESTED_SUBCONFIG_FIELDS:
          sub_cls: type[BaseModel] = _NESTED_SUBCONFIG_FIELDS[root]
          if leaf not in sub_cls.model_fields:
            raise ValueError(
                f"Unknown field {leaf!r} on `{root}` ({sub_cls.__name__}); valid fields: {sorted(sub_cls.model_fields)}."
            )
          train_overrides[root] = {**train_overrides.get(root, {}), leaf: val}
        else:
          raise ValueError(f"Unknown dotted CLI argument `--{key}`.")
      elif key in train_fields or key in model_fields:
        # Shared top-level fields (`param_dtype`, `compute_dtype`) apply to both configs unless scoped via `--model.*`.
        if key in train_fields:
          train_overrides[key] = val
        if key in model_fields:
          model_overrides[key] = val
      elif key in leaf_owners:
        owner = leaf_owners[key]
        train_overrides[owner] = {**train_overrides.get(owner, {}), key: val}
      else:
        raise ValueError(f"Unrecognized CLI flag `--{key}` for model {model_name!r} and `TrainingConfig`.")

    final_model_cfg = ModelConfig.from_model_name(model_name, **model_overrides)
    final_train_cfg = TrainingConfig(**train_overrides)
    if save_json_path:
      parent_dir = os.path.dirname(os.path.abspath(save_json_path))
      if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
      provenance = {
          "model_name": model_name,
          "model": final_model_cfg.model_dump(mode="json"),
          **json.loads(final_train_cfg.to_json()),
      }
      with open(save_json_path, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2, sort_keys=True)
    return final_model_cfg, final_train_cfg
