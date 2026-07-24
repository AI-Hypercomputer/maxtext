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
python -m maxtext.experimental.omni_poc.utils.stitch_checkpoint \
    --vision_load_path=gs://YOUR_BUCKET_NAME/checkpoints/gemma3-4b_converted/0/items \
    --llm_load_path=gs://YOUR_BUCKET_NAME/checkpoints/qwen3-4b_converted/0/items \
    --stitched_output_path=gs://YOUR_BUCKET_NAME/checkpoints/omni-gemma3-qwen3-4b/0/items
"""

import os
from typing import Any, Dict

from absl import app
from etils import epath
from flax import nnx
import jax
import omegaconf
from orbax import checkpoint as ocp

from maxtext.common import checkpointing
from maxtext.configs import pyconfig as pyconfig_mod
from maxtext.trainers.pre_train.train import initialize
from maxtext.utils import max_logging
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils
from maxtext.utils import maxtext_utils_nnx
from maxtext.utils import model_creation_utils


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


def _assemble(k: str, v: Any, stitched_subtrees: Dict[str, Any]) -> Any:
  """Merges restored parameter subtrees with fresh target model initial values.

  If a sub-module name is present in stitched_subtrees, we restore it.
  If it's missing (e.g. the new vision projector), we keep the fresh random initialization.
  """
  if k in stitched_subtrees:  # e.g., k is vision_encoder, decoder, or token_embedder
    restored_val = stitched_subtrees[k]
    if isinstance(restored_val, dict) and isinstance(v, dict):
      # Merge sub-modules inside this namespace
      merged_module = {}
      for sub_module_name, fresh_weights in v.items():
        if sub_module_name in restored_val:
          # If the sub-module (e.g. Gemma3VisionEncoderLayer_0) exists in the checkpoint, load it
          merged_module[sub_module_name] = restored_val[sub_module_name]
        else:
          # If the sub-module is missing from the checkpoint (e.g. the new projector), keep its fresh random weights
          merged_module[sub_module_name] = fresh_weights
      return merged_module
    # k is pointing to a single tensor
    return restored_val

  # k is a new layer (not in stitched_subtrees), keep fresh random init
  max_logging.log(f"Keeping fresh random normal initialization for new layer: '{k}'")
  return v


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

  vision_model_name = getattr(config, "model_name", None)
  llm_model_name = getattr(config, "decoder_block", None)
  assert vision_model_name, "model_name must be configured for vision component."
  assert llm_model_name, "decoder_block must be configured for LLM component."

  max_logging.log(f"  Vision (Model {vision_model_name}) Path: {vision_checkpoint_path}")
  max_logging.log(f"  LLM (Model {llm_model_name}) Path:    {llm_checkpoint_path}")
  max_logging.log(f"  Output Stitched Path:  {output_checkpoint_path}")
  max_logging.log("=" * 60)

  mesh = maxtext_utils.get_mesh_from_config(config)
  init_rng = jax.random.PRNGKey(config.init_weights_seed)

  # 1. Generate full target model with initial random weights
  max_logging.log("Generating target omni model from config with initial random weights...")
  with jax.set_mesh(mesh):
    if config.pure_nnx:
      rngs = maxtext_utils_nnx.create_nnx_rngs(config, rng_key=init_rng)
      model = model_creation_utils.from_config(config, mesh=mesh, rngs=rngs)
      init_params = nnx.state(model, nnx.Param)
    else:
      model = model_creation_utils.from_config(config, jax.devices())
      _, _, init_params = maxtext_utils.init_initial_state(model, None, config, is_training=False, init_rng=init_rng)

  # Convert to pure pytree for easier processing
  is_nnx = isinstance(init_params, nnx.State)
  params_dict = init_params.to_pure_dict() if is_nnx else init_params
  inner_params = params_dict.get("params", params_dict)

  inner_params = jax.tree.map(_unwrap_var, inner_params, is_leaf=lambda n: isinstance(n, nnx.Variable))

  # Wrap pytree into a Checkpointer object for partial checkpoint restoration
  ckptr = ocp.Checkpointer(
      ocp.PyTreeCheckpointHandler(
          restore_concurrent_gb=config.checkpoint_storage_concurrent_gb,
          save_concurrent_gb=config.checkpoint_storage_concurrent_gb,
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
  stitched_inner = {k: _assemble(k, v, stitched_subtrees) for k, v in inner_params.items()}
  final_params = (
      {"params": stitched_inner}
      if "params" in params_dict and isinstance(params_dict["params"], dict)
      else stitched_inner
  )

  # 5. Save unified parameter tree to output_checkpoint_path
  max_logging.log(f"Saving stitched checkpoint to: {output_checkpoint_path}")
  checkpointing.save_params_to_path(
      output_checkpoint_path,
      final_params,
      use_ocdbt=config.checkpoint_storage_use_ocdbt,
      use_zarr3=config.checkpoint_storage_use_zarr3,
  )
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
  # Extract omni stitching arguments directly from argv before passing to pyconfig.initialize
  omni_keys = {"vision_load_path", "llm_load_path", "stitched_output_path", "vision_model_name", "llm_model_name"}
  omni_kwargs = {}
  cleaned_argv = []
  for arg in argv:
    if "=" in arg and arg.split("=", 1)[0] in omni_keys:
      k, v = arg.split("=", 1)
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

  # Initialize MaxText config using standard train.initialize
  config, _ = initialize(cleaned_argv)
  # Extract paths from command-line arguments
  vision_path = omni_kwargs.get("vision_load_path")
  llm_path = omni_kwargs.get("llm_load_path")
  output_path = omni_kwargs.get("stitched_output_path")
  assert (
      vision_path and llm_path and output_path
  ), "Must specify vision_load_path, llm_load_path, and stitched_output_path"

  stitch_and_save_checkpoints(config, vision_path, llm_path, output_path)


if __name__ == "__main__":
  app.run(main)
