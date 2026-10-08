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

"""Universal Hugging Face Diffusers checkpoint converter and manager."""

import gc
import importlib
import importlib.util
import json
import os
import re
import time
from typing import Any, Optional, Sequence, Tuple

from etils import epath
from flax import nnx
from huggingface_hub import snapshot_download
import jax
from maxtext.checkpoint_conversion.utils.utils import _get_local_directory, print_peak_memory, print_ram_usage, upload_folder_to_gcs
from maxtext.m3.configs import config as config_lib
from maxtext.utils import max_logging
import numpy as np
import orbax.checkpoint as ocp

# ==============================================================================
# 1. Mesh, Sharding & Universal Orbax Checkpoint Management
# ==============================================================================


def get_cpu_mesh_and_sharding() -> (
    Tuple[jax.sharding.Mesh, jax.sharding.NamedSharding]
):
  """Constructs a local CPU mesh for checkpoint serialization/deserialization."""
  devices = jax.devices("cpu")
  devices_array = np.array(devices).reshape((len(devices), 1))
  mesh = jax.sharding.Mesh(devices_array, ("data", "model"))
  replicated_sharding = jax.sharding.NamedSharding(
      mesh, jax.sharding.PartitionSpec()
  )
  return mesh, replicated_sharding


def add_sharding_to_struct(
    leaf_struct: Any, sharding: jax.sharding.Sharding
) -> Any:
  """Manually constructs jax.ShapeDtypeStruct with a specific sharding."""
  if hasattr(leaf_struct, "shape") and hasattr(leaf_struct, "dtype"):
    return jax.ShapeDtypeStruct(
        shape=leaf_struct.shape, dtype=leaf_struct.dtype, sharding=sharding
    )
  return leaf_struct


def _get_item_from_composite(composite_or_dict: Any, key: str) -> Any:
  """Safely retrieves an item from an Orbax Composite metadata object or dict without raising KeyError."""
  if composite_or_dict is None:
    return None
  if hasattr(composite_or_dict, "_items") and isinstance(
      composite_or_dict._items, dict
  ):
    return composite_or_dict._items.get(key)
  if hasattr(composite_or_dict, "keys"):
    try:
      if key in composite_or_dict.keys():
        return composite_or_dict[key]
    except Exception:
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
  """Creates a universal Orbax CheckpointManager for diffusion models."""
  if not checkpoint_dir:
    return None
  p = epath.Path(checkpoint_dir)
  if item_names is None:
    item_names = (
        "model_state",
        "model_config",
        "wan_state",
        "wan_config",
        "low_noise_transformer_state",
        "high_noise_transformer_state",
    )
  item_handlers = {
      name: (
          ocp.JsonCheckpointHandler()
          if "config" in name
          else ocp.StandardCheckpointHandler()
      )
      for name in item_names
  }
  return ocp.CheckpointManager(
      p,
      item_names=item_names,
      options=ocp.CheckpointManagerOptions(
          create=True,
          save_interval_steps=save_interval_steps,
          enable_async_checkpointing=use_async,
      ),
      item_handlers=item_handlers,
  )


def save_diffusers_checkpoint(
    checkpoint_dir: str,
    pipeline_or_model: Any,
    step: int = 0,
):
  """Saves a diffusion model or pipeline transformer state to Orbax format."""
  checkpoint_manager = create_diffusers_checkpoint_manager(
      checkpoint_dir, save_interval_steps=1, use_async=False
  )
  if checkpoint_manager is None:
    raise ValueError(
        f"Could not create checkpoint manager for {checkpoint_dir}"
    )

  if (
      hasattr(pipeline_or_model, "transformer")
      and pipeline_or_model.transformer is not None
  ):
    model = pipeline_or_model.transformer
  elif (
      hasattr(pipeline_or_model, "unet") and pipeline_or_model.unet is not None
  ):
    model = pipeline_or_model.unet
  else:
    model = pipeline_or_model

  if model is None:
    raise ValueError(
        "Pipeline or model has no transformer/unet attribute to save."
    )

  _, state, _ = nnx.split(model, nnx.Param, ...)

  config_dict = {}
  if hasattr(model, "config"):
    for k, v in dict(model.config).items():
      if isinstance(v, (int, float, str, bool, list, tuple, dict)) or v is None:
        try:
          json.dumps(v)
          config_dict[k] = v
        except (TypeError, OverflowError):
          pass
  elif hasattr(model, "to_json_string"):
    try:
      config_dict = json.loads(model.to_json_string())
    except Exception:
      config_dict = {}

  if hasattr(model, "scan_layers"):
    config_dict["scan_layers"] = model.scan_layers

  pure_dict = state.to_pure_dict()
  # Save both generic keys and legacy keys for backwards-compatibility
  items = {
      "model_config": ocp.args.JsonSave(config_dict),
      "model_state": ocp.args.StandardSave(pure_dict),
      "wan_config": ocp.args.JsonSave(config_dict),
      "wan_state": ocp.args.StandardSave(pure_dict),
  }
  checkpoint_manager.save(step, args=ocp.args.Composite(**items))
  checkpoint_manager.wait_until_finished()
  max_logging.log(f"Saved Orbax checkpoint to {checkpoint_dir} at step {step}")


def restore_diffusers_checkpoint(
    checkpoint_dir: str, step: Optional[int] = None
):
  """Restores a diffusion model checkpoint from Orbax format."""
  checkpoint_manager = create_diffusers_checkpoint_manager(
      checkpoint_dir, save_interval_steps=1, use_async=False
  )
  if checkpoint_manager is None:
    return None
  if step is None:
    step = checkpoint_manager.latest_step()
  if step is None:
    return None

  mesh, replicated_sharding = get_cpu_mesh_and_sharding()
  metadatas = checkpoint_manager.item_metadata(step)
  restore_items = {}

  available_keys = set()
  if hasattr(metadatas, "_items") and isinstance(metadatas._items, dict):
    available_keys = set(metadatas._items.keys())
  elif hasattr(metadatas, "keys"):
    try:
      available_keys = set(metadatas.keys())
    except Exception:
      pass
  elif isinstance(metadatas, dict):
    available_keys = set(metadatas.keys())

  for config_key in ("model_config", "wan_config"):
    if config_key in available_keys:
      restore_items[config_key] = ocp.args.JsonRestore()

  for state_key in (
      "model_state",
      "wan_state",
      "low_noise_transformer_state",
      "high_noise_transformer_state",
  ):
    if state_key in available_keys:
      state_meta = _get_item_from_composite(metadatas, state_key)
      if state_meta is not None:
        target_shardings = jax.tree_util.tree_map(
            lambda x: replicated_sharding, state_meta
        )
        with mesh:
          abstract_state = jax.tree_util.tree_map(
              add_sharding_to_struct, state_meta, target_shardings
          )
        restore_items[state_key] = ocp.args.StandardRestore(abstract_state)

  return checkpoint_manager.restore(
      step=step, args=ocp.args.Composite(**restore_items)
  )


# ==============================================================================
# 2. Universal Checkpointer Classes
# ==============================================================================


class DiffusersCheckpointer:
  """Universal Checkpointer for Hugging Face diffusion models."""

  pipeline_class = None

  def __init__(self, config):
    self.config = config
    checkpoint_dir = getattr(config, "checkpoint_dir", "")
    self.checkpoint_manager = (
        create_diffusers_checkpoint_manager(
            checkpoint_dir,
            save_interval_steps=1,
            use_async=True,
        )
        if checkpoint_dir
        else None
    )

  @classmethod
  def load_pretrained_pipeline_or_diffusers(
      cls,
      config,
      pipeline_cls,
      pretrained_state_sources=(),
      pretrained_config_transformer_attr="",
  ):
    """Loads a diffusion pipeline from pretrained Orbax checkpoint or HF Diffusers."""
    checkpoint_dir = (
        getattr(config, "pretrained_orbax_dir", None)
        or getattr(config, "checkpoint_directory", None)
        or getattr(config, "checkpoint_dir", None)
        or getattr(config, "base_output_directory", None)
    )

    if checkpoint_dir:
      manager = create_diffusers_checkpoint_manager(
          checkpoint_dir, use_async=False
      )
      if manager is not None and manager.latest_step() is not None:
        max_logging.log(
            "Pretrained Orbax checkpoint found at"
            f" {checkpoint_dir}/step_{manager.latest_step()}, loading from"
            " Orbax."
        )
        try:
          return pipeline_cls.from_checkpoint(
              config, checkpoint_dir=checkpoint_dir
          )
        except Exception as e:
          max_logging.log(
              f"Failed loading from Orbax checkpoint ({e}), falling back to HF"
              " diffusers."
          )

    max_logging.log(
        f"Loading pipeline for {getattr(config, 'model_name', 'model')} from HF"
        " diffusers."
    )
    return pipeline_cls.from_pretrained(config)

  @classmethod
  def maybe_save_initial_checkpoint(cls, config, pipeline):
    """Saves initial checkpoint (step 0) if one does not already exist."""
    checkpoint_dir = getattr(config, "checkpoint_dir", "")
    if not checkpoint_dir:
      return
    manager = create_diffusers_checkpoint_manager(
        checkpoint_dir, use_async=False
    )
    if manager is not None and manager.latest_step() is not None:
      return

    max_logging.log(
        "No checkpoint found in checkpoint directory"
        f" ({checkpoint_dir}), saving model as initial checkpoint."
    )
    save_diffusers_checkpoint(checkpoint_dir, pipeline, step=0)

  def restore(self, step: Optional[int] = None):
    """Restores the latest or specified training step checkpoint."""
    if self.checkpoint_manager is None:
      return None
    if step is None:
      step = self.checkpoint_manager.latest_step()
    if step is None:
      return None

    mesh, replicated_sharding = get_cpu_mesh_and_sharding()
    metadatas = self.checkpoint_manager.item_metadata(step)
    restore_items = {}

    available_keys = set()
    if hasattr(metadatas, "_items") and isinstance(metadatas._items, dict):
      available_keys = set(metadatas._items.keys())
    elif hasattr(metadatas, "keys"):
      try:
        available_keys = set(metadatas.keys())
      except Exception:
        pass
    elif isinstance(metadatas, dict):
      available_keys = set(metadatas.keys())

    for config_key in ("model_config", "wan_config"):
      if config_key in available_keys:
        restore_items[config_key] = ocp.args.JsonRestore()

    for state_key in (
        "model_state",
        "wan_state",
        "low_noise_transformer_state",
        "high_noise_transformer_state",
    ):
      if state_key in available_keys:
        state_meta = _get_item_from_composite(metadatas, state_key)
        if state_meta is not None:
          target_shardings = jax.tree_util.tree_map(
              lambda x: replicated_sharding, state_meta
          )
          with mesh:
            abstract_state = jax.tree_util.tree_map(
                add_sharding_to_struct, state_meta, target_shardings
            )
          restore_items[state_key] = ocp.args.StandardRestore(abstract_state)

    return self.checkpoint_manager.restore(
        step=step, args=ocp.args.Composite(**restore_items)
    )

  def save_checkpoint(
      self,
      train_step: int,
      train_states: Any,
      config_dict: Optional[dict] = None,
  ):
    """Saves training step state to Orbax."""
    if self.checkpoint_manager is None:
      return

    if config_dict is None:
      config_dict = {}

    items = {
        "model_config": ocp.args.JsonSave(config_dict),
        "model_state": ocp.args.StandardSave(train_states),
        "wan_config": ocp.args.JsonSave(config_dict),
        "wan_state": ocp.args.StandardSave(train_states),
    }
    self.checkpoint_manager.save(train_step, args=ocp.args.Composite(**items))


# ==============================================================================
# 3. Model-Agnostic Resolution (following checkpoint_handoff.md)
# ==============================================================================


def _module_exists(mod_name: str) -> bool:
  """Safely checks if a module exists without importing or throwing ModuleNotFoundError."""
  try:
    return importlib.util.find_spec(mod_name) is not None
  except Exception:
    return False


def get_model_type(
    config: Optional[Any],
    model_name_or_path: str,
    local_repo_dir: str = "",
) -> str:
  """Model-agnostic architecture resolver for diffusion pipelines."""
  # 1. Config precedence: explicit model_family or model_type
  if config is not None:
    family = getattr(config, "model_family", None)
    if family:
      return str(family).lower().strip()

    m_type = getattr(config, "model_type", None)
    if m_type and str(m_type).lower() not in ("t2v", "i2v", "t2i", "i2i"):
      return str(m_type).lower().strip()

  # 2. Inspect Diffusers model_index.json if cached locally
  index_path = (
      os.path.join(local_repo_dir, "model_index.json") if local_repo_dir else ""
  )
  if index_path and os.path.isfile(index_path):
    try:
      with open(index_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
      class_name = meta.get("_class_name", "").lower()
      for suffix in (
          "pipeline2_1",
          "pipeline",
          "transformer3dmodel",
          "transformer2dmodel",
      ):
        class_name = class_name.replace(suffix, "")
      if class_name:
        return class_name
    except Exception:
      pass

  # 3. Dynamic discovery against available pipelines or converters
  raw_names = [
      str(getattr(config, "model_name", "")) if config else "",
      model_name_or_path,
  ]
  candidates = []
  for raw in raw_names:
    if not raw:
      continue
    tokens = re.split(r"[^a-zA-Z0-9]+", raw.lower())
    for t in tokens:
      if not t:
        continue
      candidates.append(t)
      alpha_only = re.sub(r"\d+$", "", t)
      if alpha_only and alpha_only != t:
        candidates.append(alpha_only)

  for candidate in candidates:
    if _module_exists(f"maxtext.inference.pipelines.{candidate}_pipeline"):
      return candidate
    if _module_exists(f"maxtext.checkpoint_conversion.convert_{candidate}"):
      return candidate

  # Fallback to repo folder prefix
  first_token = re.split(
      r"[^a-zA-Z0-9]+", model_name_or_path.split("/")[-1].lower()
  )[0]
  return re.sub(r"\d+$", "", first_token) or first_token


def get_pipeline_class(
    config: Optional[Any] = None, model_name: Optional[str] = None
):
  """Resolves the pipeline class for any supported Hugging Face diffusion model."""
  pipeline_name = getattr(config, "pipeline_name", "") or getattr(
      config, "pipeline_class", ""
  )
  if pipeline_name:
    for mod_name in (
        f"maxtext.inference.pipelines.{pipeline_name}",
        f"maxtext.inference.pipelines.{pipeline_name}_pipeline",
    ):
      try:
        mod = importlib.import_module(mod_name)
        for attr in dir(mod):
          if (
              attr.lower() == f"{pipeline_name.lower()}pipeline"
              or attr.endswith("Pipeline")
              or attr.endswith("Pipeline2_1")
          ):
            return getattr(mod, attr)
      except ModuleNotFoundError:
        pass

  name = (
      str(getattr(config, "model_name", "")) if config else (model_name or "")
  )
  model_type = get_model_type(config, model_name or name)

  try:
    module = importlib.import_module(
        f"maxtext.inference.pipelines.{model_type}_pipeline"
    )
    # 1. Match version-specific pipeline classes first (e.g. WanPipeline2_1 for wan2.1)
    version_digits = "".join(re.findall(r"\d+", name))
    if version_digits:
      for attr in dir(module):
        if (
            attr.lower() == f"{model_type}pipeline{version_digits}"
            or attr.lower()
            == f"{model_type}pipeline{version_digits[0]}_{version_digits[1:]}"
            or attr.endswith(f"Pipeline{version_digits}")
            or attr.endswith(
                f"Pipeline{version_digits[0]}_{version_digits[1:]}"
            )
        ):
          cls_candidate = getattr(module, attr)
          if isinstance(cls_candidate, type) and not getattr(
              cls_candidate, "__abstractmethods__", None
          ):
            return cls_candidate

    # 2. Check any concrete versioned pipeline (e.g. Pipeline2_1, Pipeline2_2)
    for attr in sorted(dir(module), key=len, reverse=True):
      if re.search(r"Pipeline\d", attr):
        cls_candidate = getattr(module, attr)
        if isinstance(cls_candidate, type) and not getattr(
            cls_candidate, "__abstractmethods__", None
        ):
          return cls_candidate

    # 3. Check generic unversioned Pipeline class
    for attr in sorted(dir(module), key=len, reverse=True):
      if attr.endswith("Pipeline"):
        cls_candidate = getattr(module, attr)
        if isinstance(cls_candidate, type) and not getattr(
            cls_candidate, "__abstractmethods__", None
        ):
          return cls_candidate

    for attr in dir(module):
      if attr.endswith("Pipeline"):
        return getattr(module, attr)
  except ModuleNotFoundError:
    pass

  raise ValueError(
      "Could not determine diffusion pipeline class for model"
      f" '{model_name or name}' (type '{model_type}'). Please configure"
      " 'pipeline_name' or ensure a matching pipeline exists in"
      " maxtext.inference.pipelines."
  )


# ==============================================================================
# 4. Universal Diffusers Downloading & Conversion
# ==============================================================================


def download_diffusers_components(
    repo_id: str,
    token: Optional[str] = None,
    cache_dir: Optional[str] = None,
    revision: Optional[str] = None,
    components: Optional[Sequence[str]] = None,
) -> str:
  """Downloads general diffusers pipeline components (VAE, text encoder, tokenizer, scheduler)."""
  if os.path.isdir(repo_id):
    max_logging.log(
        f"Model path '{repo_id}' is a local directory. Skipping HF download."
    )
    return repo_id

  if components is None:
    allow_patterns = [
        "vae/*",
        "text_encoder*/*",
        "tokenizer*/*",
        "scheduler/*",
        "model_index.json",
        "*.json",
    ]
  else:
    allow_patterns = [f"{c}*/*" for c in components] + [
        "model_index.json",
        "*.json",
    ]

  max_logging.log(
      f"Downloading/verifying general Diffusers components for '{repo_id}' "
      f"(patterns: {allow_patterns})..."
  )
  try:
    local_dir = snapshot_download(
        repo_id=repo_id,
        token=token,
        cache_dir=cache_dir,
        revision=revision,
        allow_patterns=allow_patterns,
    )
    max_logging.log(
        f"General Diffusers components verified in cache at: {local_dir}"
    )
    print_ram_usage("Post-Component Download")
    return local_dir
  except Exception as e:
    max_logging.log(
        f"Online download failed ({e}), attempting local cache fallback..."
    )
    local_dir = snapshot_download(
        repo_id=repo_id,
        token=token,
        cache_dir=cache_dir,
        revision=revision,
        allow_patterns=allow_patterns,
        local_files_only=True,
    )
    max_logging.log(f"Found cached Diffusers components at: {local_dir}")
    print_ram_usage("Post-Component Cache Fallback")
    return local_dir


def convert_diffusers_model(
    model_name: str,
    config: Optional[Any] = None,
    output_dir: Optional[str] = None,
    **kwargs,
) -> str:
  """Universal converter for Hugging Face Diffusers models to MaxText Orbax format.

  Supports any diffusion model family (Wan, Flux, SD3, CogVideoX, etc.) by
  instantiating its pipeline from Diffusers and serializing the model state to
  Orbax.
  """
  if output_dir is None:
    if config is not None:
      output_dir = (
          getattr(config, "base_output_directory", "")
          or getattr(config, "checkpoint_directory", "")
          or getattr(config, "pretrained_orbax_dir", "")
          or getattr(config, "checkpoint_dir", "")
      )
    if not output_dir:
      cache_dir = os.environ.get(
          "CACHE_DIR",
          os.path.join(
              os.environ.get("HF_HOME", "/dev/shm/hf_cache"), "maxdiffusion"
          ),
      )
      output_dir = os.path.join(
          cache_dir, "orbax", model_name.replace("/", "_")
      )

  checkpoint_manager = create_diffusers_checkpoint_manager(
      output_dir, use_async=False
  )
  if (
      checkpoint_manager is not None
      and checkpoint_manager.latest_step() is not None
  ):
    max_logging.log(
        f"Orbax checkpoint for {model_name} already exists at {output_dir}"
    )
    return output_dir

  max_logging.log(
      f"Converting diffusers weights for {model_name} to Orbax checkpoint at"
      f" {output_dir}..."
  )
  print_ram_usage("Pre-Model Instantiation")
  start_time = time.perf_counter()

  if config is None:
    config = config_lib.initialize(config_name=model_name)

  pipeline_cls = get_pipeline_class(config, model_name=model_name)
  pipeline = pipeline_cls.from_pretrained(
      config,
      load_vae=False,
      load_text_encoder=False,
      load_scheduler=False,
      load_transformer=True,
  )
  print_ram_usage("Post-Model Instantiation")
  save_diffusers_checkpoint(output_dir, pipeline)
  del pipeline
  gc.collect()
  elapsed = time.perf_counter() - start_time
  max_logging.log(
      f"Checkpoint conversion completed in {elapsed:.1f}s. Saved to"
      f" {output_dir}"
  )
  print_peak_memory()
  return output_dir


def convert_diffusers_checkpoint(
    model_name: str,
    config: Optional[Any] = None,
    output_dir: Optional[str] = None,
    download_components: bool = True,
    **kwargs,
) -> str:
  """Universal Diffusers checkpoint conversion entrypoint function.

  1. Downloads and caches general Diffusers pipeline components (VAE, text
  encoder,
     tokenizer, scheduler, model_index.json).
  2. Converts the backbone model (transformer or UNet) to MaxText Orbax format.

  Args:
    model_name: Hugging Face model repository ID or local path.
    config: Optional MaxText configuration.
    output_dir: Target directory for the converted Orbax checkpoint.
    download_components: Whether to download and verify auxiliary pipeline
      components.
    **kwargs: Additional parameters passed to the conversion function.

  Returns:
    Path to the converted Orbax checkpoint.
  """
  if download_components:
    token = (
        getattr(config, "hf_access_token", None)
        or getattr(config, "hf_token", None)
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HF_AUTH_TOKEN")
    )
    cache_dir = os.environ.get(
        "CACHE_DIR",
        os.path.join(
            os.environ.get("HF_HOME", "/dev/shm/hf_cache"), "maxdiffusion"
        ),
    )
    try:
      download_diffusers_components(
          repo_id=model_name,
          token=token,
          cache_dir=cache_dir,
      )
    except Exception as e:
      max_logging.log(
          f"Notice: Auxiliary diffusers component download encountered: {e}."
          " Proceeding to model conversion."
      )

  return convert_diffusers_model(
      model_name=model_name,
      config=config,
      output_dir=output_dir,
      **kwargs,
  )


# Backward-compatibility aliases
WanCheckpointer = DiffusersCheckpointer
WanCheckpointer2_1 = DiffusersCheckpointer
create_wan_checkpoint_manager = create_diffusers_checkpoint_manager
save_wan_checkpoint = save_diffusers_checkpoint
restore_wan_checkpoint = restore_diffusers_checkpoint
convert_wan_checkpoint = convert_diffusers_model
convert_checkpoint = convert_diffusers_checkpoint
checkpoint_conversion = convert_diffusers_checkpoint


# ==============================================================================
# 5. CLI Entrypoint
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
    raise ValueError(
        "base_output_directory or checkpoint_directory must be provided."
    )

  model_name = getattr(
      config, "pretrained_model_name_or_path", None
  ) or getattr(config, "model_name", None)
  if not model_name:
    raise ValueError(
        "pretrained_model_name_or_path must be specified in config or via CLI."
    )

  checkpoint_path = convert_diffusers_checkpoint(
      model_name=model_name,
      config=config,
      output_dir=output_dir,
  )
  max_logging.log(
      f"Diffusers conversion complete! Checkpoint saved to: {checkpoint_path}"
  )


if __name__ == "__main__":
  main()
