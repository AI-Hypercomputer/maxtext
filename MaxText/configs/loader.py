"""
Config loader that manually builds pydantic classes from dictionaries and CLI overrides.
"""

import os
from typing import Any, Dict, Optional

import yaml

from MaxText.configs.types import (
    MaxTextConfig,
    CoreConfig,
    ModelConfig,
    CheckpointConfig,
    OptimizerConfig,
    DatasetConfig,
    TokenizerConfig,
    ParallelismConfig,
    InferenceConfig,
)


def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
  """Recursively merge two dicts, with `override` taking priority."""
  merged = dict(base)
  for k, v in override.items():
    if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
      merged[k] = _merge_dicts(merged[k], v)
    else:
      merged[k] = v
  return merged


def load_yaml(path: str) -> Dict[str, Any]:
  with open(path, "rt", encoding="utf8") as f:
    return yaml.safe_load(f) or {}


def load_config(
    config_path: str, overrides: Optional[Dict[str, Any]] = None, base_dir: Optional[str] = None
) -> MaxTextConfig:
  """
  Load config YAML file, recursively apply `base_config`, merge overrides,
  construct and return a validated MaxTextConfig pydantic object.
  """

  base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
  if not os.path.isabs(config_path):
    config_path = os.path.join(base_dir, config_path)

  # Load the config YAML
  config_data = load_yaml(config_path)

  # Load and merge base config recursively
  if "base_config" in config_data and config_data["base_config"]:
    base_config_path = config_data.pop("base_config")
    base_config_data = load_config(base_config_path, base_dir=base_dir).dict()
    config_data = _merge_dicts(base_config_data, config_data)

  # Apply manual overrides if any
  if overrides:
    config_data = _merge_dicts(config_data, overrides)

  # Extract sub-config dicts from config_data
  core_data = {k: v for k, v in config_data.items() if k in CoreConfig.__fields__}
  # For other submodels stored flat in root config (e.g. model fields might be at root)
  # We must extract by their declared fields

  # model config fields are keys in ModelConfig.__fields__:
  model_keys = set(ModelConfig.__fields__)
  model_data = {k: v for k, v in config_data.items() if k in model_keys}

  checkpoint_keys = set(CheckpointConfig.__fields__)
  checkpoint_data = {k: v for k, v in config_data.items() if k in checkpoint_keys}

  optimizer_keys = set(OptimizerConfig.__fields__)
  optimizer_data = {k: v for k, v in config_data.items() if k in optimizer_keys}

  dataset_keys = set(DatasetConfig.__fields__)
  dataset_data = {k: v for k, v in config_data.items() if k in dataset_keys}

  tokenizer_keys = set(TokenizerConfig.__fields__)
  tokenizer_data = {k: v for k, v in config_data.items() if k in tokenizer_keys}

  parallelism_keys = set(ParallelismConfig.__fields__)
  parallelism_data = {k: v for k, v in config_data.items() if k in parallelism_keys}

  inference_keys = set(InferenceConfig.__fields__)
  inference_data = {k: v for k, v in config_data.items() if k in inference_keys}

  # Construct model subobjects
  core = CoreConfig(**core_data)
  model = ModelConfig(**model_data)
  checkpoint = CheckpointConfig(**checkpoint_data)
  optimizer = OptimizerConfig(**optimizer_data)
  dataset = DatasetConfig(**dataset_data)
  tokenizer = TokenizerConfig(**tokenizer_data)
  parallelism = ParallelismConfig(**parallelism_data)
  inference = InferenceConfig(**inference_data)

  # Compose and construct final MaxTextConfig instance
  final_config = MaxTextConfig(
      **core.dict(),
      model=model,
      checkpoint=checkpoint,
      optimizer=optimizer,
      dataset=dataset,
      tokenizer=tokenizer,
      parallelism=parallelism,
      inference=inference,
  )

  return final_config


initialize = load_config

__all__ = ["load_config", "initialize"]
