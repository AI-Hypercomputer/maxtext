# Copyright 2026 Google LLC
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

"""Utility to stitch vision and LLM checkpoints into a single unified model.

This script initializes a target multimodal model and restores subtrees from separate checkpoints:
- restore vision encoder (excluding projector) from a vision checkpoint,
- restore the decoder and token embedders from a different LLM checkpoint,
- merge them and save as a single unified MaxText checkpoint.


Example usage:
JAX_PLATFORMS=cpu python -m maxtext.experimental.omni_poc.utils.stitch_checkpoint \
    src/maxtext/experimental/omni_poc/maxtext-omni-gemma3-qwen3.yml \
    --vision_load_path=gs://YOUR_BUCKET_NAME/checkpoints/gemma3-4b_converted/0/items \
    --llm_load_path=gs://YOUR_BUCKET_NAME/checkpoints/qwen3-4b_converted/0/items \
    --stitched_output_path=gs://YOUR_BUCKET_NAME/checkpoints/omni-gemma3-qwen3-4b/0/items
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import maxtext
# Eagerly initialize core MaxText C++ and model dependencies before train.py
_ = (maxtext.Mesh, maxtext.pyconfig, maxtext.models, maxtext.model_creation_utils)

from typing import Any, Dict

from absl import app, flags
from etils import epath
from flax import nnx
import jax
import jax.numpy as jnp
import omegaconf
from orbax import checkpoint as ocp

from maxtext.common import checkpoint_context
from maxtext.common import checkpointing
from maxtext.configs import pyconfig as pyconfig_mod
from maxtext.trainers.pre_train.train import initialize
from maxtext.utils import max_logging
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils
from maxtext.utils import maxtext_utils_nnx
from maxtext.utils import model_creation_utils

FLAGS = flags.FLAGS
flags.DEFINE_string("vision_load_path", "", "Path to the vision model checkpoint.")
flags.DEFINE_string("llm_load_path", "", "Path to the LLM checkpoint.")
flags.DEFINE_string("stitched_output_path", "", "Path to save the stitched checkpoint.")


def _unwrap_var(v):
  """Unwraps flax nnx Variable instances (e.g. nnx.Param) to raw JAX arrays."""
  return v.get_value() if hasattr(v, "get_value") else v


def _restore_subtrees_from_path(
    ckpt_path: str, subtrees_abstract: Dict[str, Any], ckptr: ocp.Checkpointer
) -> Dict[str, Any]:
  """Restores subtrees from a checkpoint.

  Determines if the source checkpoint uses double params wrapper, single params wrapper,
  base wrapper, or a flat structure, and restores the matching subtree layout.
  """
  metadata = ckptr.metadata(epath.Path(ckpt_path))
  tree = metadata.item_metadata.tree
  has_params_params = "params" in tree and isinstance(tree.get("params"), dict) and "params" in tree["params"]
  has_params = "params" in tree and not has_params_params
  has_base = "base" in tree

  if has_params_params:
    item = {"params": {"params": subtrees_abstract}}
    restore_args = {"params": {"params": ocp.checkpoint_utils.construct_restore_args(subtrees_abstract)}}
  elif has_params:
    item = {"params": subtrees_abstract}
    restore_args = {"params": ocp.checkpoint_utils.construct_restore_args(subtrees_abstract)}
  elif has_base:
    item = {"base": subtrees_abstract}
    restore_args = {"base": ocp.checkpoint_utils.construct_restore_args(subtrees_abstract)}
  else:
    item = subtrees_abstract
    restore_args = ocp.checkpoint_utils.construct_restore_args(subtrees_abstract)

  restored = ckptr.restore(
      epath.Path(ckpt_path),
      item=item,
      transforms={},
      restore_args=restore_args,
  )
  if has_params_params:
    return restored["params"]["params"]
  elif has_params:
    return restored["params"]
  elif has_base:
    return restored["base"]
  else:
    return restored


def _materialize_weights(tree: Any, key: jax.Array) -> Any:
  """Recursively materializes any abstract ShapeDtypeStruct leaves with initial weights.

  Initializes normalization scales to 1.0 (ones) and biases to 0.0 (zeros),
  while initializing linear weight matrices to random normal (stddev=0.02).
  """
  def _init_leaf(path, leaf):
    if isinstance(leaf, jax.ShapeDtypeStruct):
      path_str = jax.tree_util.keystr(path, simple=True, separator="/").lower()
      if "scale" in path_str or ("norm" in path_str and len(leaf.shape) == 1):
        return jnp.ones(leaf.shape, dtype=leaf.dtype)
      elif "bias" in path_str and len(leaf.shape) == 1:
        return jnp.zeros(leaf.shape, dtype=leaf.dtype)
      else:
        sub_key = jax.random.fold_in(key, hash(path_str) % (2**31 - 1))
        return jax.random.normal(sub_key, leaf.shape, dtype=leaf.dtype) * 0.02
    return leaf

  return jax.tree_util.tree_map_with_path(_init_leaf, tree)


def _assemble(k: str, v: Any, stitched_subtrees: Dict[str, Any], rng_key: jax.Array) -> Any:
  """Merges restored parameter subtrees with fresh target model initial values.

  If a sub-module name is present in stitched_subtrees, we restore it.
  If it's missing (e.g. the new vision projector), we materialize fresh random weights.
  """
  if k in stitched_subtrees:  # e.g., k is vision_encoder, decoder, or token_embedder
    restored_val = stitched_subtrees[k]
    if isinstance(restored_val, dict) and isinstance(v, dict):
      # Merge sub-modules inside this namespace
      merged_module = {}
      for sub_module_name, fresh_weights in v.items():
        if sub_module_name in restored_val:
          # If the sub-module exists in the checkpoint, load it
          merged_module[sub_module_name] = restored_val[sub_module_name]
        else:
          # If the sub-module is missing from the checkpoint (e.g. the new projector), materialize fresh random weights
          max_logging.log(f"Materializing fresh random initialization for new sub-module: '{k}.{sub_module_name}'")
          rng_key, sub_key = jax.random.split(rng_key)
          merged_module[sub_module_name] = _materialize_weights(fresh_weights, sub_key)
      return merged_module
    # k is pointing to a single tensor
    return restored_val

  # k is a new layer (not in stitched_subtrees), materialize fresh random init
  max_logging.log(f"Materializing fresh random normal initialization for new layer: '{k}'")
  return _materialize_weights(v, rng_key)


def stitch_and_save_checkpoints(
    config: Any,
    vision_checkpoint_path: str,
    llm_checkpoint_path: str,
    output_checkpoint_path: str,
):
  """Stitches vision model weights and LLM weights into one MaxText checkpoint.

  Args:
    config: The MaxText target model configuration.
    vision_checkpoint_path: Path to the vision model checkpoint.
    llm_checkpoint_path: Path to the LLM checkpoint.
    output_checkpoint_path: Path to save the stitched checkpoint.
  """
  max_logging.log("=" * 60)
  max_logging.log("Starting Omni Multi-Directory Checkpoint Stitching...")

  vision_model_name = getattr(config, "vision_encoder_block", None)
  llm_model_name = getattr(config, "decoder_block", None)
  vision_projector_type = getattr(config, "vision_projector_type", None)
  assert vision_model_name, "vision_encoder_block must be configured for vision component."
  assert llm_model_name, "decoder_block must be configured for LLM component."
  assert vision_projector_type, "vision_projector_type must be configured for vision component."

  max_logging.log(f"  Vision (Model {vision_model_name}) Path: {vision_checkpoint_path}")
  max_logging.log(f"  LLM (Model {llm_model_name}) Path:    {llm_checkpoint_path}")
  max_logging.log(f"  Projector Type:        {vision_projector_type}")
  max_logging.log(f"  Output Stitched Path:  {output_checkpoint_path}")
  max_logging.log("=" * 60)

  mesh = maxtext_utils.get_mesh_from_config(config)
  init_rng = jax.random.PRNGKey(config.init_weights_seed)

  # 1. Generate target model abstract structure without materializing arrays
  max_logging.log("Tracing target omni model shape using abstract model creation (zero memory allocation)...")
  if config.pure_nnx:
    _, abstract_model = model_creation_utils.create_nnx_abstract_model(
        config, mesh=mesh, rng_key=init_rng
    )
    init_params = nnx.state(abstract_model, nnx.Param)
  else:
    with jax.set_mesh(mesh):
      model = model_creation_utils.from_config(config, jax.devices())
      abstract_vars = maxtext_utils.get_abstract_param(model, config)
      init_params = abstract_vars["params"]

  # Convert to pure pytree for easier processing
  is_nnx = isinstance(init_params, nnx.State)
  params_dict = init_params.to_pure_dict() if is_nnx else init_params
  inner_params = params_dict.get("params", params_dict)

  inner_params = jax.tree.map(_unwrap_var, inner_params, is_leaf=lambda n: isinstance(n, nnx.Variable))

  # Use a conservative concurrent I/O budget for single-host offline stitching:
  # The scanned 32B MLP weight tensors across 64 layers are ~33.6 GB.
  # The budget must be >= 64 GB, safely below the host VM's 336 GB RAM.
  io_concurrent_gb = max(64, min(getattr(config, "checkpoint_storage_concurrent_gb", 64), 96))
  max_logging.log(f"Using checkpoint I/O concurrent budget: {io_concurrent_gb} GB")
  ckptr = ocp.Checkpointer(
      ocp.PyTreeCheckpointHandler(
          restore_concurrent_gb=io_concurrent_gb,
          save_concurrent_gb=io_concurrent_gb,
          use_ocdbt=config.checkpoint_storage_use_ocdbt,
          use_zarr3=config.checkpoint_storage_use_zarr3,
      )
  )

  stitched_subtrees = {}

  # 2. Restore Vision Encoder subtree from Model A
  if "vision_encoder" in inner_params and vision_checkpoint_path:
    max_logging.log(f"Restoring 'vision_encoder' from {vision_checkpoint_path}...")
    # Filter out projector/embedder keys from the abstract state so they are not loaded from disk
    vision_encoder_abstract = {
        k: v
        for k, v in inner_params["vision_encoder"].items()
        if not ("projector" in k.lower() or "embedder" in k.lower())
    }
    vision_abstract = {"vision_encoder": vision_encoder_abstract}
    vision_restored = _restore_subtrees_from_path(vision_checkpoint_path, vision_abstract, ckptr)
    stitched_subtrees["vision_encoder"] = vision_restored["vision_encoder"]

  # 3. Restore LLM Decoder subtrees from Model B
  llm_keys = [k for k in ["decoder", "token_embedder"] if k in inner_params]
  if llm_keys and llm_checkpoint_path:
    max_logging.log(f"Restoring LLM subtrees ({llm_keys}) from {llm_checkpoint_path}...")
    llm_abstract = {k: inner_params[k] for k in llm_keys}
    llm_restored = _restore_subtrees_from_path(llm_checkpoint_path, llm_abstract, ckptr)
    for k in llm_keys:
      stitched_subtrees[k] = llm_restored[k]

  # 4. Assemble: Vision (Model A) + LLM (Model B) + Random Init Projector
  max_logging.log("Assembling stitched subtrees and materializing projector...")
  rng_key = jax.random.PRNGKey(config.init_weights_seed)
  stitched_inner = {k: _assemble(k, v, stitched_subtrees, rng_key) for k, v in inner_params.items()}
  final_params = {"params": stitched_inner}

  # 5. Save unified parameter tree to output_checkpoint_path
  save_dir = checkpointing._normalize_checkpoint_root(output_checkpoint_path)
  max_logging.log(f"Saving stitched checkpoint to step root: {save_dir} (target items: {output_checkpoint_path})")
  checkpointing.save_params_to_path(
      save_dir,
      final_params,
      use_ocdbt=config.checkpoint_storage_use_ocdbt,
      use_zarr3=config.checkpoint_storage_use_zarr3,
  )

  # Ensure both step root (e.g. 0/) and item root (e.g. 0/items/) have commit and metadata markers
  for marker in ["_CHECKPOINT_METADATA", "commit_success.txt"]:
    f_root = epath.Path(save_dir) / marker
    f_items = epath.Path(output_checkpoint_path) / marker
    try:
      if f_root.exists() and not f_items.exists():
        f_items.write_bytes(f_root.read_bytes())
      elif f_items.exists() and not f_root.exists():
        f_root.write_bytes(f_items.read_bytes())
    except Exception as e:
      max_logging.log(f"Notice: could not mirror {marker}: {e}")

  total_params = max_utils.calculate_num_params_from_pytree(final_params)
  max_logging.log(f"Total Stitched Model Parameters: {total_params:,} (~{total_params/1e9:.3f}B)")
  max_logging.log("Checkpoint stitching complete!")


def _load_custom_yaml_overrides(yaml_path: str, omni_keys: set[str]):
  """Loads a custom YAML config and splits it into omni-specific keys and MaxText overrides."""
  custom_cfg = omegaconf.OmegaConf.to_container(omegaconf.OmegaConf.load(yaml_path), resolve=True)

  omni_yaml_args = {}
  maxtext_overrides = {}
  for key, value in custom_cfg.items():
    if key in omni_keys:
      omni_yaml_args[key] = value
    else:
      maxtext_overrides[key] = value

  return omni_yaml_args, maxtext_overrides


def main(argv):
  omni_keys = {
      "vision_load_path",
      "llm_load_path",
      "stitched_output_path",
      "vision_model_name",
      "llm_model_name",
      "base_config",
      "model_name",
  }
  omni_kwargs = {}
  cleaned_argv = []
  for arg in argv:
    cleaned_arg = arg.lstrip("-")
    if "=" in cleaned_arg and cleaned_arg.split("=", 1)[0] in omni_keys:
      k, v = cleaned_arg.split("=", 1)
      omni_kwargs[k] = v
    else:
      cleaned_argv.append(arg)

  # To populate all system-wide defaults, MaxText requires base.yml as argv[1].
  # To apply our custom overrides on top of these defaults, we convert the custom config
  # overrides into cleaned_argv for initialization.
  if len(cleaned_argv) >= 2 and cleaned_argv[1].endswith(".yml") and not cleaned_argv[1].endswith("base.yml"):
    custom_yaml_path = cleaned_argv[1]

    # Load and split custom settings
    yaml_omni_args, yaml_overrides = _load_custom_yaml_overrides(custom_yaml_path, omni_keys)

    # Merge settings
    for k, v in yaml_omni_args.items():
      omni_kwargs.setdefault(k, v)

    # Convert YAML overrides to CLI-style arguments for standard initialize()
    for k, v in yaml_overrides.items():
      if isinstance(v, str):
        cleaned_argv.append(f"{k}='{v}'")
      else:
        cleaned_argv.append(f"{k}={v}")
    cleaned_argv.append("override_model_config=True")

    cleaned_argv[1] = os.path.join(pyconfig_mod.MAXTEXT_CONFIGS_DIR, "base.yml")

  if not any(arg.startswith("skip_jax_distributed_system=") for arg in cleaned_argv):
    cleaned_argv.append("skip_jax_distributed_system=True")

  # Initialize MaxText config using standard train.initialize
  config, _ = initialize(cleaned_argv)
  object.__setattr__(config, "model_name", "maxtext-omni-gemma3-qwen3")
  # Extract paths from command-line arguments or FLAGS
  vision_path = FLAGS.vision_load_path or omni_kwargs.get("vision_load_path")
  llm_path = FLAGS.llm_load_path or omni_kwargs.get("llm_load_path")
  output_path = FLAGS.stitched_output_path or omni_kwargs.get("stitched_output_path")
  assert (
      vision_path and llm_path and output_path
  ), "Must specify vision_load_path, llm_load_path, and stitched_output_path"

  stitch_and_save_checkpoints(config, vision_path, llm_path, output_path)


if __name__ == "__main__":
  app.run(main)
