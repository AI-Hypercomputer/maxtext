# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wan diffusion model checkpoint conversion and management for MaxText.

Provides model-specific utilities for:
1. Converting Wan Hugging Face safetensors checkpoints to Orbax format.
2. Managing Wan Orbax checkpoints (save, restore, sharded layouts).
3. Wan checkpointer classes (WanCheckpointer, WanCheckpointer2_1).
"""

import gc
import json
import os
import time
from typing import Any, Optional, Sequence, Tuple

from absl import app
from etils import epath
from flax import nnx
import jax
import numpy as np
import orbax.checkpoint as ocp

from maxtext.utils import max_logging


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


def create_wan_checkpoint_manager(
    checkpoint_dir: str,
    save_interval_steps: int = 1,
    use_async: bool = True,
) -> ocp.CheckpointManager | None:
  """Creates an Orbax CheckpointManager for WAN models."""
  if not checkpoint_dir:
    return None
  p = epath.Path(checkpoint_dir)
  item_names = ("low_noise_transformer_state", "high_noise_transformer_state", "wan_state", "wan_config")
  item_handlers = {
      "wan_config": ocp.JsonCheckpointHandler(),
      "wan_state": ocp.StandardCheckpointHandler(),
      "low_noise_transformer_state": ocp.StandardCheckpointHandler(),
      "high_noise_transformer_state": ocp.StandardCheckpointHandler(),
  }
  return ocp.CheckpointManager(
      p,
      item_names=item_names,
      options=ocp.CheckpointManagerOptions(
          create=True, save_interval_steps=save_interval_steps, enable_async_checkpointing=use_async
      ),
      item_handlers=item_handlers,
  )


def save_wan_checkpoint(
    checkpoint_dir: str,
    pipeline_or_transformer,
    step: int = 0,
):
  """Saves a Wan model or pipeline transformer state to Orbax format."""
  checkpoint_manager = create_wan_checkpoint_manager(checkpoint_dir, save_interval_steps=1, use_async=False)
  if checkpoint_manager is None:
    raise ValueError(f"Could not create checkpoint manager for {checkpoint_dir}")

  if hasattr(pipeline_or_transformer, "transformer"):
    transformer = pipeline_or_transformer.transformer
  else:
    transformer = pipeline_or_transformer

  if transformer is None:
    raise ValueError("Pipeline or model has no transformer attribute to save.")

  _, state, _ = nnx.split(transformer, nnx.Param, ...)

  config_dict = {}
  if hasattr(transformer, "config"):
    for k, v in dict(transformer.config).items():
      if isinstance(v, (int, float, str, bool, list, tuple, dict)) or v is None:
        try:
          json.dumps(v)
          config_dict[k] = v
        except (TypeError, OverflowError):
          pass
  elif hasattr(transformer, "to_json_string"):
    try:
      config_dict = json.loads(transformer.to_json_string())
    except Exception:
      config_dict = {}

  if hasattr(transformer, "scan_layers"):
    config_dict["scan_layers"] = transformer.scan_layers

  items = {
      "wan_config": ocp.args.JsonSave(config_dict),
      "wan_state": ocp.args.StandardSave(state.to_pure_dict()),
  }
  checkpoint_manager.save(step, args=ocp.args.Composite(**items))
  checkpoint_manager.wait_until_finished()
  max_logging.log(f"Saved Wan Orbax checkpoint to {checkpoint_dir} at step {step}")


def restore_wan_checkpoint(checkpoint_dir: str, step: Optional[int] = None):
  """Restores a Wan model checkpoint from Orbax format."""
  checkpoint_manager = create_wan_checkpoint_manager(checkpoint_dir, save_interval_steps=1, use_async=False)
  if checkpoint_manager is None:
    return None
  if step is None:
    step = checkpoint_manager.latest_step()
  if step is None:
    return None

  mesh, replicated_sharding = get_cpu_mesh_and_sharding()
  metadatas = checkpoint_manager.item_metadata(step)
  restore_items = {"wan_config": ocp.args.JsonRestore()}
  for item_name in ("wan_state", "low_noise_transformer_state", "high_noise_transformer_state"):
    try:
      state_meta = getattr(metadatas, item_name)
    except (KeyError, AttributeError):
      state_meta = None
    if state_meta is None and isinstance(metadatas, dict):
      state_meta = metadatas.get(item_name)
    if state_meta is not None:
      target_shardings = jax.tree_util.tree_map(lambda x: replicated_sharding, state_meta)
      with mesh:
        abstract_state = jax.tree_util.tree_map(add_sharding_to_struct, state_meta, target_shardings)
      restore_items[item_name] = ocp.args.StandardRestore(abstract_state)
  return checkpoint_manager.restore(step=step, args=ocp.args.Composite(**restore_items))


class WanCheckpointer:
  """Base Checkpointer for Wan diffusion models."""

  pipeline_class = None

  def __init__(self, config):
    self.config = config
    checkpoint_dir = getattr(config, "checkpoint_dir", "")
    self.checkpoint_manager = (
        create_wan_checkpoint_manager(
            checkpoint_dir,
            save_interval_steps=1,
            use_async=True,
        )
        if checkpoint_dir
        else None
    )

  @classmethod
  def load_pretrained_pipeline_or_diffusers(
      cls, config, pipeline_cls, pretrained_state_sources, pretrained_config_transformer_attr
  ):
    pretrained_dir = getattr(config, "pretrained_orbax_dir", "")
    if pretrained_dir:
      restored_checkpoint = cls._restore_pretrained_checkpoint(
          pretrained_dir, tuple(state_item_name for state_item_name, _ in pretrained_state_sources)
      )
      if restored_checkpoint is not None:
        max_logging.log(f"Loading WAN pipeline from pretrained orbax checkpoint at {pretrained_dir}")
        return pipeline_cls.from_checkpoint(config, restored_checkpoint=restored_checkpoint)

    max_logging.log("No checkpoint found, loading default pipeline.")
    pipeline = pipeline_cls.from_pretrained(config)
    if pretrained_dir:
      cls._save_pretrained_checkpoint(
          pretrained_dir, pipeline, pretrained_state_sources, pretrained_config_transformer_attr
      )
    return pipeline

  @classmethod
  def _restore_pretrained_checkpoint(cls, pretrained_dir: str, state_item_names: Tuple[str, ...]):
    return restore_wan_checkpoint(pretrained_dir)

  @classmethod
  def _save_pretrained_checkpoint(
      cls, pretrained_dir: str, pipeline, pretrained_state_sources, pretrained_config_transformer_attr
  ):
    try:
      save_wan_checkpoint(pretrained_dir, pipeline)
    except Exception as e:
      max_logging.log(f"Failed to save pretrained orbax checkpoint to {pretrained_dir}: {e}")

  def load_diffusers_checkpoint(self, **kwargs):
    return self.pipeline_class.from_pretrained(self.config, **kwargs)

  def load_checkpoint(self, step=None, **kwargs):
    restored_checkpoint, step = self.load_wan_configs_from_orbax(step)
    opt_state = None
    if restored_checkpoint:
      pipeline = self.pipeline_class.from_checkpoint(self.config, restored_checkpoint=restored_checkpoint, **kwargs)
      opt_state = self._extract_opt_state(restored_checkpoint)
    else:
      pipeline = self.load_diffusers_checkpoint(**kwargs)
    return pipeline, opt_state, step

  def load_wan_configs_from_orbax(self, step: Optional[int] = None):
    raise NotImplementedError

  def _extract_opt_state(self, restored_checkpoint):
    return None


class WanCheckpointer2_1(WanCheckpointer):
  """Checkpointer for Wan 2.1."""

  def __init__(self, config):
    super().__init__(config)
    from maxtext.inference.pipelines.wan_pipeline import WanPipeline2_1
    self.pipeline_class = WanPipeline2_1

  def load_wan_configs_from_orbax(self, step: Optional[int] = None) -> Tuple[Optional[dict], Optional[int]]:
    if self.checkpoint_manager is None:
      return None, None
    if step is None:
      step = self.checkpoint_manager.latest_step()
      if step is None:
        return None, None
    restored_checkpoint = restore_wan_checkpoint(self.checkpoint_manager.directory, step=step)
    return restored_checkpoint, step

  def _extract_opt_state(self, restored_checkpoint):
    if hasattr(restored_checkpoint, "wan_state") and "opt_state" in restored_checkpoint.wan_state:
      return restored_checkpoint.wan_state["opt_state"]
    return None

  def save_checkpoint(self, train_step, pipeline, train_states: dict):
    if self.checkpoint_manager is None:
      return
    config_dict = {}
    if hasattr(pipeline.transformer, "config"):
      for k, v in dict(pipeline.transformer.config).items():
        if isinstance(v, (int, float, str, bool, list, tuple, dict)) or v is None:
          try:
            json.dumps(v)
            config_dict[k] = v
          except (TypeError, OverflowError):
            pass
    elif hasattr(pipeline.transformer, "to_json_string"):
      try:
        config_dict = json.loads(pipeline.transformer.to_json_string())
      except Exception:
        config_dict = {}
    items = {
        "wan_config": ocp.args.JsonSave(config_dict),
        "wan_state": ocp.args.StandardSave(train_states),
    }
    self.checkpoint_manager.save(train_step, args=ocp.args.Composite(**items))


def convert_checkpoint(
    model_name: str,
    config: Optional[Any] = None,
    output_dir: Optional[str] = None,
    **kwargs,
) -> str:
  """Converts Wan Hugging Face safetensors to MaxText Orbax format.

  Args:
    model_name: Hugging Face model repository ID.
    config: Optional MaxText configuration.
    output_dir: Target directory for the Orbax checkpoint.
    **kwargs: Additional model conversion arguments.

  Returns:
    The path to the Orbax checkpoint directory.
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
          os.path.join(os.environ.get("HF_HOME", "/dev/shm/hf_cache"), "maxdiffusion")
      )
      output_dir = os.path.join(cache_dir, "orbax", model_name.replace("/", "_"))

  # Check if checkpoint already exists and has a valid step
  checkpoint_manager = create_wan_checkpoint_manager(output_dir, use_async=False)
  if checkpoint_manager is not None and checkpoint_manager.latest_step() is not None:
    max_logging.log(f"Wan Orbax checkpoint for {model_name} already exists at {output_dir}")
    return output_dir

  max_logging.log(f"Converting Wan safetensors for {model_name} to Orbax checkpoint at {output_dir}...")
  start_time = time.perf_counter()

  if config is None:
    from maxtext.configs import pyconfig
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "models", "wan", "configs", "wan2.1-t2v-14b.yml"
    )
    if os.path.exists(config_path):
      config = pyconfig.initialize([config_path], pretrained_model_name_or_path=model_name)
    else:
      config = pyconfig.initialize([], pretrained_model_name_or_path=model_name)

  from maxtext.inference.pipelines.wan_pipeline import WanPipeline2_1
  # Download from HF (or read cache) and initialize transformer only
  pipeline = WanPipeline2_1.from_pretrained(
      config,
      load_vae=False,
      load_text_encoder=False,
      load_scheduler=False,
      load_transformer=True,
  )
  save_wan_checkpoint(output_dir, pipeline)
  del pipeline
  gc.collect()
  elapsed = time.perf_counter() - start_time
  max_logging.log(f"Wan checkpoint conversion completed in {elapsed:.1f}s. Saved to {output_dir}")
  return output_dir


# Alias for backward compatibility
checkpoint_conversion = convert_checkpoint


def main(argv: Sequence[str]) -> None:
  from maxtext.configs import pyconfig
  pyconfig.initialize(argv)
  config = pyconfig.config

  output_dir = (
      getattr(config, "base_output_directory", None)
      or getattr(config, "checkpoint_directory", None)
      or getattr(config, "output_dir", None)
      or getattr(config, "checkpoint_dir", None)
  )
  if not output_dir:
    raise ValueError("base_output_directory or checkpoint_directory must be provided.")

  model_name = getattr(config, "pretrained_model_name_or_path", None)
  if not model_name:
    raise ValueError("pretrained_model_name_or_path must be specified in config or via CLI.")

  checkpoint_path = convert_checkpoint(
      model_name=model_name,
      config=config,
      output_dir=output_dir,
  )
  max_logging.log(f"Wan conversion complete! Checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
  app.run(main)
