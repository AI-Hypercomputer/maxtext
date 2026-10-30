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

"""Universal Hugging Face Diffusers checkpoint converter and manager.

Converts a multi-component diffusers checkpoint (e.g. `transformer/`, `vae/`, `text_encoder/` sub-folders of
safetensors) into one MaxText Orbax checkpoint, using the per-model mapping class registered in
`param_mapping.DIFFUSERS_PARAM_MAPPING` (keyed by `model_name`). Each component is stored as two Orbax
items, `<component>_state` (the NNX param tree) and `<component>_config` (the model config), so inference
only restores the checkpoint and never reads safetensors.

Usage:
  python src/maxtext/checkpoint_conversion/to_maxtext_diffusers.py <model.yml> \
      base_output_directory=<dir> scan_layers=false hf_access_token=<token>
"""

import concurrent.futures
import importlib
import inspect
import json
import os
import time
from typing import Any, Optional, Sequence, Tuple

from etils import epath
from flax import nnx
from flax.traverse_util import flatten_dict
from huggingface_hub import snapshot_download
import jax
import jax.numpy as jnp
from maxtext.checkpoint_conversion.utils.param_mapping import DIFFUSERS_PARAM_MAPPING
from maxtext.checkpoint_conversion.utils.utils import apply_hook_fns, print_peak_memory, print_ram_usage, validate_and_filter_param_map_keys
from maxtext.m3.configs import config as config_lib
from maxtext.utils import max_logging
import ml_dtypes
import numpy as np
import orbax.checkpoint as ocp
from safetensors import safe_open

try:
  import torch
except ImportError:
  torch = None

# ==============================================================================
# 1. Orbax checkpoint layout, save and restore
# ==============================================================================

# Diffusers sub-folders that can be stored; each becomes `<name>_state` + `<name>_config` items.
_COMPONENT_ITEMS = ("transformer", "transformer_2", "vae", "text_encoder")
# Weight-free diffusers sub-folders stored as a `<name>_config` item only: {name: config file}.
_CONFIG_ONLY_COMPONENTS = {"scheduler": "scheduler_config.json"}
# Item names written by earlier versions of this converter (restore-only).
_LEGACY_ITEMS = (
    "model_state",
    "model_config",
    "wan_state",
    "wan_config",
    "low_noise_transformer_state",
    "high_noise_transformer_state",
)
CHECKPOINT_ITEM_NAMES = (
    tuple(f"{c}_{s}" for c in _COMPONENT_ITEMS for s in ("state", "config"))
    + tuple(f"{c}_config" for c in _CONFIG_ONLY_COMPONENTS)
    + _LEGACY_ITEMS
)


def get_cpu_mesh_and_sharding() -> Tuple[jax.sharding.Mesh, jax.sharding.NamedSharding]:
  """Constructs a local CPU mesh for checkpoint serialization/deserialization."""
  devices = jax.devices("cpu")
  devices_array = np.array(devices).reshape((len(devices), 1))
  mesh = jax.sharding.Mesh(devices_array, ("data", "model"))
  replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
  return mesh, replicated_sharding


def add_sharding_to_struct(leaf_struct: Any, sharding: jax.sharding.Sharding) -> Any:
  """Manually constructs jax.ShapeDtypeStruct with a specific sharding."""
  if hasattr(leaf_struct, "shape") and hasattr(leaf_struct, "dtype"):
    return jax.ShapeDtypeStruct(shape=leaf_struct.shape, dtype=leaf_struct.dtype, sharding=sharding)
  return leaf_struct


def _get_item_from_composite(composite_or_dict: Any, key: str) -> Any:
  """Safely retrieves an item from an Orbax Composite metadata object or dict without raising KeyError."""
  if composite_or_dict is None:
    return None
  # Orbax Composite metadata only exposes its items via the private `_items`.
  # pylint: disable-next=protected-access
  if hasattr(composite_or_dict, "_items") and isinstance(composite_or_dict._items, dict):
    return composite_or_dict._items.get(key)  # pylint: disable=protected-access
  if hasattr(composite_or_dict, "keys"):
    try:
      if key in composite_or_dict.keys():
        return composite_or_dict[key]
    except Exception:  # pylint: disable=broad-except
      pass
  if isinstance(composite_or_dict, dict):
    return composite_or_dict.get(key)
  try:
    return getattr(composite_or_dict, key)
  except (AttributeError, KeyError):
    return None


def create_diffusers_checkpoint_manager(
    checkpoint_dir: str,
    save_interval_steps: int = 1,
    use_async: bool = True,
    item_names: Optional[Sequence[str]] = None,
) -> ocp.CheckpointManager | None:
  """Creates an Orbax CheckpointManager for diffusion model checkpoints."""
  if not checkpoint_dir:
    return None
  item_names = tuple(item_names or CHECKPOINT_ITEM_NAMES)
  item_handlers = {
      name: (ocp.JsonCheckpointHandler() if name.endswith("_config") else ocp.StandardCheckpointHandler())
      for name in item_names
  }
  return ocp.CheckpointManager(
      epath.Path(checkpoint_dir),
      item_names=item_names,
      options=ocp.CheckpointManagerOptions(
          create=True,
          save_interval_steps=save_interval_steps,
          enable_async_checkpointing=use_async,
      ),
      item_handlers=item_handlers,
  )


def save_diffusers_checkpoint(checkpoint_dir: str, states: dict, configs: dict, step: int = 0):
  """Saves `{component: param_tree}` and `{component: config}` as one Orbax checkpoint step.

  `configs` may contain config-only components (e.g. `scheduler`) that have no entry in `states`.
  """
  checkpoint_manager = create_diffusers_checkpoint_manager(checkpoint_dir, save_interval_steps=1, use_async=False)
  if checkpoint_manager is None:
    raise ValueError(f"Could not create checkpoint manager for {checkpoint_dir}")
  items = {f"{component}_state": ocp.args.StandardSave(state) for component, state in states.items()}
  items.update({f"{component}_config": ocp.args.JsonSave(config) for component, config in configs.items()})
  checkpoint_manager.save(step, args=ocp.args.Composite(**items))
  checkpoint_manager.wait_until_finished()
  max_logging.log(f"Saved Orbax checkpoint ({', '.join(configs)}) to {checkpoint_dir} at step {step}")


def restore_diffusers_checkpoint(checkpoint_dir: str, step: Optional[int] = None):
  """Restores every item present in a diffusion checkpoint (states replicated on a CPU mesh)."""
  checkpoint_manager = create_diffusers_checkpoint_manager(checkpoint_dir, save_interval_steps=1, use_async=False)
  if checkpoint_manager is None:
    return None
  if step is None:
    step = checkpoint_manager.latest_step()
  if step is None:
    return None

  mesh, replicated_sharding = get_cpu_mesh_and_sharding()
  metadatas = checkpoint_manager.item_metadata(step)
  # Orbax Composite metadata only exposes its items via the private `_items`.
  # pylint: disable-next=protected-access
  if hasattr(metadatas, "_items") and isinstance(metadatas._items, dict):
    available = set(metadatas._items.keys())  # pylint: disable=protected-access
  else:
    available = set(metadatas.keys())
  restore_items = {}
  for name in CHECKPOINT_ITEM_NAMES:
    if name not in available:
      continue
    if name.endswith("_config"):  # JSON items report no metadata.
      restore_items[name] = ocp.args.JsonRestore()
    else:
      meta = _get_item_from_composite(metadatas, name)
      target_shardings = jax.tree_util.tree_map(lambda x: replicated_sharding, meta)
      with mesh:
        abstract_state = jax.tree_util.tree_map(add_sharding_to_struct, meta, target_shardings)
      restore_items[name] = ocp.args.StandardRestore(abstract_state)
  return checkpoint_manager.restore(step=step, args=ocp.args.Composite(**restore_items))


# ==============================================================================
# 2. Hugging Face Diffusers download and conversion
# ==============================================================================

# Non-converted pipeline components loaded at inference time by their own libraries (transformers, ...).
_AUXILIARY_COMPONENTS = ("tokenizer", "scheduler", "image_encoder", "image_processor")


def download_diffusers_components(
    repo_id: str,
    components: Sequence[str],
    token: Optional[str] = None,
    cache_dir: Optional[str] = None,
    revision: Optional[str] = None,
) -> str:
  """Downloads the converted components plus the auxiliary pipeline components; returns the local repo dir."""
  if os.path.isdir(repo_id):
    max_logging.log(f"Model path '{repo_id}' is a local directory. Skipping HF download.")
    return repo_id
  allow_patterns = [f"{c}*/*" for c in (*components, *_AUXILIARY_COMPONENTS)] + ["model_index.json"]
  max_logging.log(f"Downloading/verifying Diffusers components for '{repo_id}' (patterns: {allow_patterns})...")
  kwargs = {
      "repo_id": repo_id,
      "token": token,
      "cache_dir": cache_dir,
      "revision": revision,
      "allow_patterns": allow_patterns,
  }
  try:
    local_dir = snapshot_download(**kwargs)
  except Exception as e:  # pylint: disable=broad-except
    max_logging.log(f"Online download failed ({e}), attempting local cache fallback...")
    local_dir = snapshot_download(**kwargs, local_files_only=True)
  max_logging.log(f"Diffusers components available at: {local_dir}")
  print_ram_usage("Post-Component Download")
  return local_dir


class _SafetensorsReader:
  """Reads single tensors (as numpy) from a diffusers component folder, sharded or not.

  Diffusers models store `diffusion_pytorch_model*.safetensors`, transformers models (e.g. the text encoder)
  `model*.safetensors`.
  """

  _PREFIXES = ("diffusion_pytorch_model", "model")

  def __init__(self, component_dir: str):
    if torch is None:
      raise ImportError("Reading diffusers safetensors needs torch (bfloat16 tensors load via framework='pt').")
    for prefix in self._PREFIXES:
      index_path = os.path.join(component_dir, f"{prefix}.safetensors.index.json")
      path = os.path.join(component_dir, f"{prefix}.safetensors")
      if os.path.isfile(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
          weight_map = json.load(f)["weight_map"]
        self.key_to_file = {k: os.path.join(component_dir, v) for k, v in weight_map.items()}
        return
      if os.path.isfile(path):
        with safe_open(path, framework="pt") as f:
          self.key_to_file = {k: path for k in f.keys()}
        return
    raise FileNotFoundError(f"No {' or '.join(self._PREFIXES)} safetensors found in {component_dir}.")

  def get(self, key: str) -> np.ndarray:
    if key not in self.key_to_file:
      raise KeyError(f"Hugging Face key '{key}' not found in checkpoint.")
    with safe_open(self.key_to_file[key], framework="pt") as f:
      tensor = f.get_tensor(key)
    if tensor.dtype == torch.bfloat16:  # No native numpy bfloat16: reinterpret the bits.
      return tensor.view(torch.uint16).numpy().view(ml_dtypes.bfloat16)
    return tensor.numpy()


def _import_class(path: str):
  module_name, class_name = path.rsplit(".", 1)
  return getattr(importlib.import_module(module_name), class_name)


def model_init_config(model_cls, hf_config: dict) -> dict:
  """Keeps only the HF config entries that are `model_cls.__init__` arguments.

  Drops diffusers metadata (`_class_name`, `_diffusers_version`, ...) and HF-only keys, so the stored
  config can be passed straight to `model_cls(**config)`.
  """
  params = set(inspect.signature(model_cls.__init__).parameters) - {"self", "rngs"}
  return {k: v for k, v in hf_config.items() if k in params}


def _model_kwargs(model_cls, scan_layers: bool) -> dict:
  """Constructor kwargs that are not part of the HF config."""
  params = inspect.signature(model_cls.__init__).parameters
  kwargs = {}
  if "mesh" in params:
    kwargs["mesh"] = None
  if "scan_layers" in params:
    kwargs["scan_layers"] = scan_layers
  return kwargs


def _abstract_param_shapes(model_cls, model_config: dict, scan_layers: bool) -> dict:
  """`{maxtext_key: shape}` of the model's NNX params, without allocating weights."""
  kwargs = {**model_config, **_model_kwargs(model_cls, scan_layers)}
  model = nnx.eval_shape(lambda: model_cls(rngs=nnx.Rngs(0), **kwargs))
  params = nnx.state(model, nnx.Param).to_pure_dict()
  return {"-".join(map(str, path)): tuple(leaf.shape) for path, leaf in flatten_dict(params).items()}


def convert_diffusers_component(
    repo_dir: str,
    component: str,
    mapping_cls,
    scan_layers: bool,
    weights_dtype,
    max_workers: int = 16,
) -> Tuple[dict, dict]:
  """Converts one diffusers component folder into a nested MaxText param tree.

  Returns:
    (state, model_config): the param tree (string keys, numpy leaves) and the model config to store.
  """
  start = time.perf_counter()
  component_dir = os.path.join(repo_dir, component)
  model_cls = _import_class(mapping_cls.COMPONENTS[component])
  with open(os.path.join(component_dir, "config.json"), "r", encoding="utf-8") as f:
    model_config = model_init_config(model_cls, json.load(f))
  target_shapes = _abstract_param_shapes(model_cls, model_config, scan_layers)

  param_map = mapping_cls.param_mapping(component, model_config, scan_layers)
  hook_fns = mapping_cls.hook_fns(component, model_config, scan_layers)
  maxtext_keys = validate_and_filter_param_map_keys(param_map.keys(), set(target_shapes))
  reader = _SafetensorsReader(component_dir)

  def convert(maxtext_key):
    hf_keys, shape = param_map[maxtext_key], target_shapes[maxtext_key]
    hook = hook_fns.get(maxtext_key)
    dtype = mapping_cls.param_dtype(component, maxtext_key, weights_dtype)

    def load(hf_key, target_shape):
      weight = apply_hook_fns(reader.get(hf_key), target_shape, hook)
      # Always copy (C order): converted params must not alias the safetensors buffers.
      return np.array(weight, dtype=dtype or weight.dtype, order="C", copy=True)

    if isinstance(hf_keys, list):  # Scanned layers, stacked along axis 0.
      value = np.stack([load(hf_key, shape[1:]) for hf_key in hf_keys], axis=0)
    else:
      value = load(hf_keys, shape)
    if value.shape != shape:
      raise ValueError(f"Shape mismatch for {maxtext_key}: expected {shape}, got {value.shape}")
    return maxtext_key, value

  with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    flat_state = dict(executor.map(convert, maxtext_keys))

  state = {}
  for maxtext_key, value in flat_state.items():
    *parents, leaf = maxtext_key.split("-")
    node = state
    for part in parents:
      node = node.setdefault(part, {})
    node[leaf] = value
  if "scan_layers" in inspect.signature(model_cls.__init__).parameters:
    model_config["scan_layers"] = scan_layers
  num_bytes = sum(v.nbytes for v in flat_state.values())
  max_logging.log(
      f"Converted {component}: {len(flat_state)} params, {num_bytes / 2**30:.2f} GiB in"
      f" {time.perf_counter() - start:.1f}s"
  )
  return state, model_config


def _hf_cache_dir() -> str:
  return os.environ.get("CACHE_DIR", os.path.join(os.environ.get("HF_HOME", "/dev/shm/hf_cache"), "maxdiffusion"))


def convert_diffusers_states(config) -> dict:
  """Downloads and converts every component of `config.pretrained_model_name_or_path` in memory.

  Returns a dict laid out like a restored checkpoint: `{"<component>_state": ..., "<component>_config": ...}`.
  """
  model_name = getattr(config, "model_name", "")
  if model_name not in DIFFUSERS_PARAM_MAPPING:
    raise ValueError(
        f"No diffusers parameter mapping registered for model_name '{model_name}'."
        f" Registered: {sorted(DIFFUSERS_PARAM_MAPPING)} (param_mapping.DIFFUSERS_PARAM_MAPPING)."
    )
  mapping_cls = DIFFUSERS_PARAM_MAPPING[model_name]
  token = getattr(config, "hf_access_token", None) or os.environ.get("HF_TOKEN")
  repo_dir = download_diffusers_components(
      config.pretrained_model_name_or_path,
      components=list(mapping_cls.COMPONENTS),
      token=token,
      cache_dir=_hf_cache_dir(),
  )
  weights_dtype = np.dtype(jnp.dtype(getattr(config, "weights_dtype", "bfloat16")))
  scan_layers = bool(getattr(config, "scan_layers", False))
  outputs = {}
  for component in mapping_cls.COMPONENTS:
    state, model_config = convert_diffusers_component(repo_dir, component, mapping_cls, scan_layers, weights_dtype)
    outputs[f"{component}_state"], outputs[f"{component}_config"] = state, model_config
  for component, config_file in _CONFIG_ONLY_COMPONENTS.items():
    config_path = os.path.join(repo_dir, component, config_file)
    if os.path.isfile(config_path):
      with open(config_path, "r", encoding="utf-8") as f:
        outputs[f"{component}_config"] = json.load(f)
  return outputs


def convert_diffusers_checkpoint(config, output_dir: str) -> str:
  """Converts `config.pretrained_model_name_or_path` to an Orbax checkpoint at `output_dir` (skips if present)."""
  checkpoint_manager = create_diffusers_checkpoint_manager(output_dir, use_async=False)
  if checkpoint_manager is not None and checkpoint_manager.latest_step() is not None:
    max_logging.log(f"Orbax checkpoint already exists at {output_dir}, skipping conversion.")
    return output_dir

  max_logging.log(f"Converting {config.pretrained_model_name_or_path} to an Orbax checkpoint at {output_dir}...")
  start = time.perf_counter()
  outputs = convert_diffusers_states(config)
  save_diffusers_checkpoint(
      output_dir,
      states={name[: -len("_state")]: v for name, v in outputs.items() if name.endswith("_state")},
      configs={name[: -len("_config")]: v for name, v in outputs.items() if name.endswith("_config")},
  )
  max_logging.log(f"Checkpoint conversion completed in {time.perf_counter() - start:.1f}s. Saved to {output_dir}")
  print_peak_memory()
  return output_dir


# ==============================================================================
# 3. CLI Entrypoint
# ==============================================================================


def main(argv: Sequence[str] | None = None) -> None:
  config = config_lib.initialize(argv)
  output_dir = (
      getattr(config, "base_output_directory", None)
      or getattr(config, "checkpoint_directory", None)
      or getattr(config, "checkpoint_dir", None)
      or getattr(config, "output_dir", None)
  )
  if not output_dir:
    raise ValueError("base_output_directory or checkpoint_directory must be provided.")
  if not getattr(config, "pretrained_model_name_or_path", None):
    raise ValueError("pretrained_model_name_or_path must be specified in config or via CLI.")
  checkpoint_path = convert_diffusers_checkpoint(config, output_dir)
  max_logging.log(f"Diffusers conversion complete! Checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
  main()
