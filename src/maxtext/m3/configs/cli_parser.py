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

"""Pure-Python CLI parser introspecting `ModelConfig` and `TrainingConfig` Pydantic schemas."""

from collections.abc import Sequence
import json
import os
import sys
from typing import Any

from pydantic import BaseModel
import yaml

from maxtext.m3.configs.training_config import TrainingConfig, _NESTED_SUBCONFIG_FIELDS
from maxtext.m3.models.registry import ModelConfig


def _parse_cli_scalar(raw_val: str) -> Any:
  """Parses a CLI value string into bool / int / float / dict / list / str."""
  stripped = raw_val.strip()
  lowered = stripped.lower()
  if lowered == "true":
    return True
  if lowered == "false":
    return False
  if lowered in ("none", "null"):
    return None
  if (stripped.startswith("{") and stripped.endswith("}")) or (stripped.startswith("[") and stripped.endswith("]")):
    try:
      return json.loads(stripped)
    except json.JSONDecodeError:
      return yaml.safe_load(stripped)
  try:
    return int(stripped)
  except ValueError:
    pass
  try:
    return float(stripped)
  except ValueError:
    pass
  if (stripped.startswith('"') and stripped.endswith('"')) or (stripped.startswith("'") and stripped.endswith("'")):
    return stripped[1:-1]
  return stripped


def _collect_leaf_field_map() -> dict[str, str]:
  """Maps unambiguous leaf field names to their owning nested sub-config attribute on `TrainingConfig`."""
  counts: dict[str, list[str]] = {}
  top_fields = set(TrainingConfig.model_fields.keys())
  for sub_name, sub_cls in _NESTED_SUBCONFIG_FIELDS.items():
    for field_name in sub_cls.model_fields:
      counts.setdefault(field_name, []).append(sub_name)
  return {
      field_name: owners[0] for field_name, owners in counts.items() if len(owners) == 1 and field_name not in top_fields
  }


class M3ArgumentParser:
  """Zero-YAML CLI parser producing `(ModelConfig, TrainingConfig)` from `--preset` and typed flags.

  Supported forms:
    * `--preset qwen3-0.6b` (or `--preset=qwen3-0.6b`)
    * `--config_json path/to/config.json` (loads JSON base dictionary before applying CLI flags)
    * `--steps 500` or `--steps=500` (top-level `TrainingConfig` or `ModelConfig` fields)
    * `--transforms.scan_layers=false` or `--mesh.ici='{"fsdp": -1, "tensor": 2}'` (dotted nested fields)
    * `--scan_layers=false` (unambiguous leaf field routed automatically to `transforms.scan_layers`)
    * Boolean shorthand `--enable_checkpointing` (equivalent to `--enable_checkpointing=true`)
  """

  def parse_args(
      self,
      argv: Sequence[str] | None = None,
  ) -> tuple[ModelConfig, TrainingConfig]:
    """Parses CLI tokens into validated `(ModelConfig, TrainingConfig)` instances."""
    tokens = list(sys.argv[1:] if argv is None else argv)
    # Strip leading program path if `app.run` style `argv` (`[prog, ...]`) was passed.
    if tokens and not tokens[0].startswith("-"):
      tokens = tokens[1:]

    raw_pairs: list[tuple[str, Any]] = []
    idx = 0
    while idx < len(tokens):
      tok = tokens[idx].strip()
      if not tok:
        idx += 1
        continue
      if not tok.startswith("--"):
        raise ValueError(f"Expected `--flag` argument in M3ArgumentParser, got {tok!r}.")
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

    preset_name: str | None = None
    save_json_path: str | None = None
    json_payload: dict[str, Any] = {}
    remaining: list[tuple[str, Any]] = []

    for key, val in raw_pairs:
      if key == "preset":
        preset_name = str(val)
      elif key == "config_json":
        with open(str(val), "r", encoding="utf-8") as f:
          loaded = json.load(f)
        if not isinstance(loaded, dict):
          raise ValueError(f"--config_json file {val!r} must contain a JSON object.")
        json_payload.update(loaded)
      elif key in ("save_config_json", "dump_config_json"):
        save_json_path = str(val)
      else:
        remaining.append((key, val))

    if preset_name is None and "preset" in json_payload:
      preset_name = str(json_payload.pop("preset"))
    if preset_name is None:
      raise ValueError("Missing required `--preset <name>` argument (e.g. `--preset qwen3-0.6b`).")

    base_model_cfg = ModelConfig.from_preset(preset_name)
    model_fields = set(type(base_model_cfg).model_fields.keys())
    train_fields = set(TrainingConfig.model_fields.keys())
    leaf_owners = _collect_leaf_field_map()

    model_overrides: dict[str, Any] = dict(json_payload.pop("model", {}))
    train_overrides: dict[str, Any] = dict(json_payload)

    def _assign_dotted(target: dict[str, Any], dotted_key: str, value: Any) -> None:
      parts = dotted_key.split(".")
      root = parts[0]
      if root == "model" and len(parts) == 2:
        model_overrides[parts[1]] = value
        return
      if root not in _NESTED_SUBCONFIG_FIELDS or len(parts) != 2:
        raise ValueError(f"Unknown dotted CLI argument `--{dotted_key}`.")
      sub_cls: type[BaseModel] = _NESTED_SUBCONFIG_FIELDS[root]
      if parts[1] not in sub_cls.model_fields:
        raise ValueError(
            f"Unknown field {parts[1]!r} on `{root}` ({sub_cls.__name__}); "
            f"valid fields: {sorted(sub_cls.model_fields)}."
        )
      nested_dict = dict(target.get(root, {}))
      nested_dict[parts[1]] = value
      target[root] = nested_dict

    for key, val in remaining:
      if "." in key:
        _assign_dotted(train_overrides, key, val)
      elif key in train_fields:
        train_overrides[key] = val
      elif key in leaf_owners:
        owner = leaf_owners[key]
        nested_dict = dict(train_overrides.get(owner, {}))
        nested_dict[key] = val
        train_overrides[owner] = nested_dict
      elif key in model_fields:
        model_overrides[key] = val
      else:
        raise ValueError(f"Unrecognized CLI flag `--{key}` for preset {preset_name!r} and `TrainingConfig`.")

    final_model_cfg = ModelConfig.from_preset(preset_name, **model_overrides)
    final_train_cfg = TrainingConfig(**train_overrides)
    if save_json_path:
      parent_dir = os.path.dirname(os.path.abspath(save_json_path))
      if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
      provenance = {
          "preset": preset_name,
          "model": final_model_cfg.model_dump(mode="json"),
          **json.loads(final_train_cfg.to_json()),
      }
      with open(save_json_path, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2, sort_keys=True)
    return final_model_cfg, final_train_cfg
