# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unified Modern MaxText Models (M3) configuration and initialization."""

import os
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Union
import jax
import jax.numpy as jnp
from maxtext.utils import max_logging
from pydantic import BaseModel, ConfigDict, Field
import yaml


def _lists_to_tuples(l: Any) -> Any:
  """Recursively converts lists to tuples for Flax Linen SPMD rules."""
  if isinstance(l, (list, tuple)):
    return tuple(_lists_to_tuples(x) for x in l)
  return l


def get_global_batch_size(per_device_batch_size: float) -> int:
  """Calculates global batch size across available JAX devices."""
  num_devices = len(jax.devices())
  if per_device_batch_size < 1:
    return num_devices
  return int(num_devices * per_device_batch_size)


def maybe_initialize_jax_distributed_system(raw_keys: dict[str, Any]) -> None:
  """Initializes the JAX distributed cluster if needed."""
  if raw_keys.get("skip_jax_distributed_system", False):
    max_logging.log(
        "Skipping jax distributed system due to"
        " skip_jax_distributed_system=True flag."
    )
    return
  try:
    from jax._src.xla_bridge import backends_are_initialized

    if backends_are_initialized():
      max_logging.log(
          "XLA backends already initialized; skipping"
          " jax.distributed.initialize()."
      )
      return
  except Exception:
    pass
  if jax.distributed.is_initialized():
    max_logging.log("Jax distributed system is already initialized.")
    return
  if raw_keys.get("enable_single_controller", False):
    max_logging.log(
        "Skipping jax distributed system since its not needed for single"
        " controller."
    )
    return
  if raw_keys.get("inference_benchmark_test", False) or raw_keys.get(
      "compile_topology", ""
  ):
    max_logging.log("Skipping jax distributed system initialization.")
    return

  try:
    timeout = raw_keys.get("jax_distributed_initialization_timeout", 300)
    jax.distributed.initialize(initialization_timeout=timeout)
    max_logging.log("Jax distributed system initialized!")
  except Exception as e:
    max_logging.log(
        f"Warning: jax.distributed.initialize() skipped or failed: {e}"
    )


# ----------------------------------------------------------------------------
# Core Component Configs
# ----------------------------------------------------------------------------


class DeviceConfig(BaseModel):
  """Hardware, mesh topology, and SPMD partitioning configurations."""

  model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

  hardware: str = "tpu"
  skip_jax_distributed_system: bool = False
  jax_cache_dir: str = ""
  mesh_axes: list[str] = Field(
      default_factory=lambda: ["data", "fsdp", "context", "tensor"]
  )
  logical_axis_rules: tuple = ()
  vae_logical_axis_rules: tuple = ()
  data_sharding: tuple = ()
  dcn_data_parallelism: int = 1
  dcn_fsdp_parallelism: int = 1
  dcn_context_parallelism: int = -1
  dcn_tensor_parallelism: int = 1
  ici_data_parallelism: int = 1
  ici_fsdp_parallelism: int = 1
  ici_context_parallelism: int = -1
  ici_tensor_parallelism: int = 1
  allow_split_physical_axes: bool = False
  compile_topology_num_slices: int = -1
  num_slices: int = 1
  quantization_local_shard_count: int = 1


class CheckpointConfig(BaseModel):
  """Checkpointing, directories, Orbax, and weights loading configurations."""

  model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

  run_name: str = ""
  base_output_directory: str = ""
  checkpoint_directory: str = ""
  output_dir: str = ""
  checkpoint_every: int = -1
  checkpoint_dir: str = ""
  tensorboard_dir: str = ""
  metrics_dir: str = ""
  save_final_checkpoint: bool = False
  save_config_to_gcs: bool = False
  enable_single_replica_ckpt_restoring: bool = False
  pretrained_orbax_dir: str = ""
  converted_weights_dir: str = ""
  aot_cache_dir: str = ""
  unet_checkpoint: str = ""
  revision: str = ""
  lightning_from_pt: bool = True
  lightning_repo: str = ""
  lightning_ckpt: str = ""


class OptimizerConfig(BaseModel):
  """Optimization parameters and learning rate schedules."""

  model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

  adam_b1: float = 0.9
  adam_b2: float = 0.999
  adam_eps: float = 1e-8
  adam_weight_decay: float = 0.0
  opt_enable_grad_clipping: bool = False
  max_grad_value: float = 1.0
  opt_enable_grad_global_norm_clipping: bool = False
  max_grad_norm: float = 1.0
  warmup_steps_fraction: float = 0.1
  learning_rate_schedule_steps: int = -1
  save_optimizer: bool = False


class TrainerConfig(BaseModel):
  """Training loop, evaluation, and logging configurations."""

  model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

  learning_rate: float = 1e-5
  scale_lr: bool = False
  max_train_samples: int = -1
  max_train_steps: int = 1500
  num_train_epochs: int = 1
  seed: int = 0
  per_device_batch_size: float = 1.0
  global_batch_size: int = 0
  disable_training_weights: bool = False
  train_text_encoder: bool = False
  text_encoder_learning_rate: float = 4.25e-6
  snr_gamma: float = -1.0
  remat_policy: str = "NONE"
  names_which_can_be_saved: list[str] = Field(default_factory=list)
  names_which_can_be_offloaded: list[str] = Field(default_factory=list)
  enable_tile_search: bool = False
  tile_search_mode: str = "smart"
  tile_search_iters: int = 10
  tile_search_out: str = ""
  optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
  eval_every: int = -1
  eval_data_dir: str = ""
  enable_generate_video_for_eval: bool = False
  eval_max_number_of_samples_in_bucket: int = 60
  enable_eval_timesteps: bool = False
  timesteps_list: list[int] = Field(
      default_factory=lambda: [125, 250, 375, 500, 625, 750, 875]
  )
  num_eval_samples: int = 420
  metrics_file: str = ""
  write_metrics: bool = True
  timing_metrics_file: str = ""
  write_timing_metrics: bool = True
  gcs_metrics: bool = False
  log_period: int = 100
  enable_mllog: bool = False
  enable_ssim: bool = False
  enable_ml_diagnostics: bool = False


class DataConfig(BaseModel):
  """Dataset and preprocessing configurations."""

  model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

  dataset_name: str = ""
  dataset_path: str = ""
  train_data_dir: str = ""
  eval_data_dir: str = ""
  per_device_batch_size: float = 1.0
  max_sequence_length: int = 512


class TransformerConfig(BaseModel):
  """Shared transformer architecture configurations between LLMs and Diffusion Models."""

  model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

  # Architecture dimensions
  dim: Optional[int] = None
  base_emb_dim: Optional[int] = None
  num_layers: Optional[int] = None
  base_num_decoder_layers: Optional[int] = None
  num_heads: Optional[int] = None
  base_num_query_heads: Optional[int] = None
  num_kv_heads: Optional[int] = None
  base_num_kv_heads: Optional[int] = None
  head_dim: Optional[int] = None
  mlp_dim: Optional[int] = None
  base_mlp_dim: Optional[int] = None

  # Precision and execution
  weights_dtype: Any = "bfloat16"
  activations_dtype: Any = "bfloat16"
  attention: str = "flash"
  precision: str = "DEFAULT"
  scan_layers: bool = False
  dropout: float = 0.0
  split_head_dim: bool = True
  norm_num_groups: int = 32
  use_base2_exp: bool = True


# ----------------------------------------------------------------------------
# Top-Level Base Config & HyperParameters
# ----------------------------------------------------------------------------


class BaseConfig(BaseModel):
  """Base top-level configuration class coordinating sub-configs and runtime setups."""

  model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

  device: DeviceConfig = Field(default_factory=DeviceConfig)
  checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
  trainer: TrainerConfig = Field(default_factory=TrainerConfig)
  data: DataConfig = Field(default_factory=DataConfig)
  transformer: Optional[TransformerConfig] = None

  def _get_subconfigs(self) -> list[BaseModel]:
    subs = []
    for attr in (
        "device",
        "checkpoint",
        "trainer",
        "data",
        "transformer",
        "autoencoder",
        "scheduler",
        "text_encoder",
        "pipeline",
    ):
      sub = getattr(self, attr, None)
      if isinstance(sub, BaseModel):
        subs.append(sub)
    return subs

  def __getattr__(self, name: str) -> Any:
    # Check extra fields on self
    if self.__pydantic_extra__ and name in self.__pydantic_extra__:
      return self.__pydantic_extra__[name]
    # Check sub-configs
    for sub in self._get_subconfigs():
      if hasattr(sub, name):
        return getattr(sub, name)
      if sub.__pydantic_extra__ and name in sub.__pydantic_extra__:
        return sub.__pydantic_extra__[name]
    raise AttributeError(
        f"'{type(self).__name__}' object has no attribute '{name}'"
    )

  def __getitem__(self, key: str) -> Any:
    try:
      return getattr(self, key)
    except AttributeError:
      raise KeyError(key)

  def __setitem__(self, key: str, value: Any) -> None:
    setattr(self, key, value)

  def __setattr__(self, name: str, value: Any) -> None:
    if name in self.__class__.model_fields:
      super().__setattr__(name, value)
      return
    # Check if a sub-config already holds this attribute
    for sub in self._get_subconfigs():
      if hasattr(sub, name) or (
          sub.__pydantic_extra__ and name in sub.__pydantic_extra__
      ):
        setattr(sub, name, value)
        return
    if self.__pydantic_extra__ is None:
      object.__setattr__(self, "__pydantic_extra__", {})
    self.__pydantic_extra__[name] = value

  def get(self, key: str, default: Any = None) -> Any:
    try:
      return getattr(self, key)
    except AttributeError:
      return default

  def __contains__(self, key: str) -> bool:
    try:
      _ = getattr(self, key)
      return True
    except AttributeError:
      return False

  def keys(self):
    return self.get_keys().keys()

  def values(self):
    return self.get_keys().values()

  def items(self):
    return self.get_keys().items()

  def __iter__(self):
    return iter(self.get_keys())

  def get_keys(self) -> dict[str, Any]:
    """Returns a combined dictionary of all keys across sub-configs and extra fields."""
    d = {}
    for sub in self._get_subconfigs():
      d.update(sub.model_dump())
      if sub.__pydantic_extra__:
        d.update(sub.__pydantic_extra__)
    d.update(self.model_dump())
    if self.__pydantic_extra__:
      d.update(self.__pydantic_extra__)
    return d

  @classmethod
  def from_dict(cls, raw: dict[str, Any]) -> "BaseConfig":
    device = DeviceConfig.model_validate(raw)
    checkpoint = CheckpointConfig.model_validate(raw)
    trainer = TrainerConfig.model_validate(raw)
    data = DataConfig.model_validate(raw)
    transformer = TransformerConfig.model_validate(raw)
    return cls(
        device=device,
        checkpoint=checkpoint,
        trainer=trainer,
        data=data,
        transformer=transformer,
        **raw,
    )


class HyperParameters(BaseConfig):
  """Alias/subclass for backward compatibility with MaxText legacy config interfaces."""

  pass


# ----------------------------------------------------------------------------
# YAML Loading, Overrides, and Initialization
# ----------------------------------------------------------------------------


def _strip_yaml_ext(name: str) -> str:
  name = os.path.basename(str(name))
  for ext in (".yaml", ".yml"):
    if name.endswith(ext):
      return name[: -len(ext)]
  return name


def _clean_default_entry(entry: Any) -> tuple[str, Optional[str]]:
  """Parses a modular defaults list entry into (group_or_file, item_name)."""
  if isinstance(entry, str):
    s = entry.strip()
    if s.startswith("/"):
      s = s[1:]
    if "@" in s:
      s = s.split("@")[0]
    return s, None
  elif isinstance(entry, dict):
    for k, v in entry.items():
      k = k.strip()
      if k.startswith("/"):
        k = k[1:]
      if "@" in k:
        k = k.split("@")[0]
      return k, str(v).strip()
  return "", None


def _find_yaml_file(
    rel_path: str, search_paths: Sequence[str]
) -> Optional[str]:
  for sp in search_paths:
    for ext in ("", ".yml", ".yaml"):
      candidate = os.path.join(sp, rel_path + ext)
      if os.path.isfile(candidate):
        return candidate
  return None


def _resolve_interpolations(
    d: dict[str, Any], root: dict[str, Any] | None = None
) -> dict[str, Any]:
  """Recursively resolves simple ${var} or ${group.var} interpolations within dictionary values."""
  if root is None:
    root = d
  pattern = re.compile(r"\$\{([a-zA-Z0-9_\.]+)\}")

  def _lookup_path(path: str) -> Any:
    parts = path.split(".")
    curr = root
    for p in parts:
      if isinstance(curr, dict) and p in curr:
        curr = curr[p]
      else:
        return None
    return curr

  for k, v in list(d.items()):
    if isinstance(v, str) and "${" in v:
      m = pattern.fullmatch(v.strip())
      if m:
        resolved = _lookup_path(m.group(1))
        if resolved is not None:
          d[k] = resolved
      else:
        d[k] = pattern.sub(
            lambda m: str(_lookup_path(m.group(1)) or m.group(0)), v
        )
    elif isinstance(v, dict):
      _resolve_interpolations(v, root)
  return d


def load_yaml_config(
    file_path: str,
    search_paths: Sequence[str] | None = None,
) -> dict[str, Any]:
  """Loads a model configuration YAML and recursively merges modular defaults."""
  file_path = os.path.abspath(os.path.expanduser(file_path))
  if not os.path.isfile(file_path):
    raise FileNotFoundError(f"Configuration file not found: {file_path}")

  base_dir = os.path.dirname(file_path)
  if search_paths is None:
    search_paths = [
        base_dir,
        os.path.abspath(os.path.join(base_dir, "..")),
        os.path.abspath(os.path.join(base_dir, "..", "..")),
        os.path.abspath(os.path.join(base_dir, "models")),
        os.path.abspath(os.path.join(base_dir, "..", "models")),
    ]

  with open(file_path, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f) or {}

  merged: dict[str, Any] = {}
  defaults = data.get("defaults", [])
  for d in defaults:
    if d == "_self_":
      continue
    prefix, name = _clean_default_entry(d)
    if not prefix:
      continue
    target = f"{prefix}/{name}" if name else prefix
    found = _find_yaml_file(target, search_paths)
    if not found and prefix:
      found = _find_yaml_file(prefix, search_paths)
    if found:
      sub = load_yaml_config(found, search_paths)
      merged.update(sub)
    else:
      max_logging.log(
          f"Warning: could not resolve default config entry: {target}"
      )

  for k, v in data.items():
    if k not in ("defaults", "hydra"):
      merged[k] = v

  return merged


def _parse_val(v: str) -> Any:
  """Parses a string CLI token into a strongly typed Python value."""
  if (v.startswith('"') and v.endswith('"')) or (
      v.startswith("'") and v.endswith("'")
  ):
    inner = v[1:-1]
    if inner.startswith("{") and inner.endswith("}"):
      try:
        inner_with_space = re.sub(r":(?=[^\s])", ": ", inner)
        return yaml.safe_load(inner_with_space)
      except Exception:
        pass
    return inner
  if v.startswith("{") and v.endswith("}"):
    try:
      v_with_space = re.sub(r":(?=[^\s])", ": ", v)
      return yaml.safe_load(v_with_space)
    except Exception:
      pass
  try:
    return yaml.safe_load(v)
  except Exception:
    return v


def parse_cli_args(
    argv: Sequence[str] | None = None,
) -> tuple[str, dict[str, Any]]:
  """Extracts config_name and key=value overrides from argv."""
  if argv is None:
    argv = sys.argv

  if not argv or len(argv) <= 1:
    return "base", {}

  config_name = "base"
  overrides = {}

  for token in argv[1:]:
    token = token.strip()
    if not token:
      continue
    if token.startswith(
        ("--config-name=", "config_name=", "--model-name=", "model_name=")
    ):
      val = token.split("=", 1)[1].strip("\"'")
      config_name = _strip_yaml_ext(val)
    elif "=" in token and not token.startswith("-"):
      k, v = token.split("=", 1)
      overrides[k.strip()] = _parse_val(v.strip())
    elif not token.startswith("-") and "=" not in token:
      config_name = _strip_yaml_ext(token)

  return config_name, overrides


def parse_cli_overrides(argv: Sequence[str] | None = None) -> dict[str, Any]:
  if argv is None:
    argv = sys.argv
  _, overrides = parse_cli_args(argv)
  return overrides


def get_config_name(argv: Sequence[str] | None = None) -> str:
  if argv is None:
    argv = sys.argv
  name, _ = parse_cli_args(argv)
  return name


def _find_model_yaml(config_name: str, config_dir: Optional[str] = None) -> str:
  """Locates the YAML configuration file for the given config name."""
  for path_candidate in (
      config_name,
      config_name.replace("src/maxtext/models/", "src/maxtext/m3/models/"),
  ):
    if os.path.isfile(path_candidate):
      return os.path.abspath(path_candidate)

  clean_name = _strip_yaml_ext(config_name)
  candidate_dirs = []
  if config_dir:
    candidate_dirs.append(config_dir)

  module_dir = os.path.dirname(os.path.abspath(__file__))
  candidate_dirs.extend([
      os.path.join(module_dir, "..", "models", "wan"),
      os.path.join(module_dir, "..", "models", "wan", "configs"),
      os.path.join(module_dir, "..", "models"),
      os.path.join(module_dir, "..", "..", "models", "wan"),
      os.path.join(module_dir, "..", "..", "models", "wan", "configs"),
      os.path.join(module_dir, "..", "..", "configs", "models"),
      os.path.join(module_dir, "models"),
      module_dir,
  ])

  for cd in candidate_dirs:
    for ext in (".yml", ".yaml"):
      p = os.path.join(cd, clean_name + ext)
      if os.path.isfile(p):
        return p
      p_nested = os.path.join(cd, clean_name, clean_name + ext)
      if os.path.isfile(p_nested):
        return p_nested

  if os.path.isfile(config_name):
    return os.path.abspath(config_name)

  raise FileNotFoundError(
      f"Could not find configuration YAML for '{config_name}' in search dirs:"
      f" {candidate_dirs}"
  )


def process_config(
    raw: Union[dict[str, Any], BaseConfig],
    cli_overrides: Optional[dict[str, Any]] = None,
    **kwargs,
) -> BaseConfig:
  """Post-processes raw configuration dictionary into a strongly typed Config object."""
  if isinstance(raw, BaseConfig):
    raw_dict = raw.get_keys()
  else:
    raw_dict = dict(raw)

  if cli_overrides:
    raw_dict.update(cli_overrides)
  if kwargs:
    raw_dict.update(kwargs)

  # 1. Resolve interpolations
  _resolve_interpolations(raw_dict)

  # 2. Convert JAX dtypes
  for dtype_field in (
      "weights_dtype",
      "activations_dtype",
      "vae_weights_dtype",
      "vae_dtype",
      "scheduler_dtype",
      "text_encoder_dtype",
  ):
    if dtype_field in raw_dict and isinstance(raw_dict[dtype_field], str):
      try:
        raw_dict[dtype_field] = jnp.dtype(raw_dict[dtype_field])
      except Exception:
        pass

  # 3. Post-process run_name and directory paths
  run_name = raw_dict.get("run_name", "")
  if not run_name:
    run_name = os.environ.get("JOBSET_NAME", "")
  if not run_name:
    import datetime

    run_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
  raw_dict["run_name"] = run_name

  base_output_directory = (
      raw_dict.get("base_output_directory", "")
      or raw_dict.get("checkpoint_directory", "")
      or raw_dict.get("output_dir", "")
  )
  if base_output_directory:
    raw_dict["base_output_directory"] = base_output_directory
    if not raw_dict.get("tensorboard_dir"):
      raw_dict["tensorboard_dir"] = os.path.join(
          base_output_directory, run_name, "tensorboard", ""
      )
    if not raw_dict.get("checkpoint_dir"):
      raw_dict["checkpoint_dir"] = os.path.join(
          base_output_directory, run_name, "checkpoints", ""
      )
    if not raw_dict.get("metrics_dir"):
      raw_dict["metrics_dir"] = os.path.join(
          base_output_directory, run_name, "metrics", ""
      )
  else:
    if not raw_dict.get("tensorboard_dir"):
      raw_dict["tensorboard_dir"] = ""
    if not raw_dict.get("checkpoint_dir"):
      raw_dict["checkpoint_dir"] = ""
    if not raw_dict.get("metrics_dir"):
      raw_dict["metrics_dir"] = ""

  # 4. Post-process logical axis rules & sharding to true tuples (required by Flax SPMD)
  for rule_field in (
      "logical_axis_rules",
      "vae_logical_axis_rules",
      "data_sharding",
  ):
    if rule_field in raw_dict and raw_dict[rule_field] is not None:
      raw_dict[rule_field] = _lists_to_tuples(raw_dict[rule_field])

  # 5. Post-process batch sizes
  per_device_batch_size = raw_dict.get("per_device_batch_size", 1.0)
  raw_dict["total_train_batch_size"] = get_global_batch_size(
      per_device_batch_size
  )
  num_devices = len(jax.devices())
  raw_dict["global_batch_size_to_load"] = (
      num_devices
      if per_device_batch_size < 1
      else int(num_devices * per_device_batch_size)
  )
  raw_dict["global_batch_size_to_train_on"] = int(
      num_devices * per_device_batch_size
  )

  # 6. Hardware & JAX distributed initialization
  if not kwargs.get("unittest", False):
    maybe_initialize_jax_distributed_system(raw_dict)

  # 7. JAX cache directory
  if raw_dict.get("jax_cache_dir"):
    jax.config.update("jax_compilation_cache_dir", raw_dict["jax_cache_dir"])

  # 8. Instantiate target Config model (DiffusionConfig for diffusion architectures, BaseConfig otherwise)
  is_diffusion = any(
      k in raw_dict
      for k in (
          "pretrained_model_name_or_path",
          "model_type",
          "vae_dtype",
          "vae_decode_chunk",
          "flow_shift",
          "num_frames",
          "num_inference_steps",
      )
  )

  if is_diffusion:
    from maxtext.m3.configs.diffusers_config import DiffusionConfig

    cfg = DiffusionConfig.from_dict(raw_dict)
  else:
    cfg = BaseConfig.from_dict(raw_dict)

  for k in sorted(raw_dict.keys()):
    max_logging.log(f"Config param {k}: {raw_dict[k]}")

  return cfg


def initialize(
    argv: Sequence[str] | None = None,
    config_name: str | None = None,
    config_dir: str | None = None,
    **kwargs,
) -> BaseConfig:
  """Entrypoint to load config from CLI arguments and apply overrides."""
  if argv is None:
    argv = sys.argv

  target_config_name, cli_overrides = parse_cli_args(argv)
  if config_name is not None:
    target_config_name = config_name

  yaml_path = _find_model_yaml(target_config_name, config_dir=config_dir)
  raw_dict = load_yaml_config(yaml_path)

  return process_config(raw_dict, cli_overrides=cli_overrides, **kwargs)
