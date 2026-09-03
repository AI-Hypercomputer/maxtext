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

"""
This is NOT an unit test.

Sanity check utility to verify parameter freezing and training after SFT.

1. Loads the original stitched checkpoint.
2. Loads the SFT-trained checkpoint (auto-discovered or specified).
3. Compares the parameter PyTrees to verify that:
   - Custom MLP vision projector weights were updated (trained).
   - Vision encoder and LLM backbone weights are 100% identical (frozen).

Example usage:
  python3 -m maxtext.experimental.omni_poc.tests.compare_sft_checkpoint_test \
      src/maxtext/experimental/omni_poc/configs/sft-maxtext-omni-gemma3-qwen3.yml \
      load_parameters_path=gs://YOUR_BUCKET/omni_checkpoints/omni_stitched_gemma3-4b_qwen3-4b/0/items \
      base_output_directory=gs://YOUR_BUCKET/sft_multimodal_omni_output \
      run_name=sft_omni_chartqa
"""

import os
import sys
from typing import Any, Optional, Sequence

from absl import app, flags
from etils import epath
import jax
import jax.numpy as jnp

from maxtext.common import checkpointing
from maxtext.configs import pyconfig
from maxtext.utils import max_utils, maxtext_utils, model_creation_utils
from maxtext.utils.globals import MAXTEXT_PKG_DIR

FLAGS = flags.FLAGS


def _define_flag(fn, name, default, help_str):
  """Defines an ABSL flag if it hasn't already been registered."""
  if name not in FLAGS:
    fn(name, default, help_str)


_define_flag(
    flags.DEFINE_string,
    "config_path",
    os.path.join(MAXTEXT_PKG_DIR, "experimental", "omni_poc", "configs", "sft-maxtext-omni-gemma3-qwen3.yml"),
    "Path to the config YAML file.",
)
_define_flag(
    flags.DEFINE_string,
    "stitched_checkpoint_path",
    "",
    "Path to original stitched checkpoint directory (defaults to load_parameters_path from config).",
)
_define_flag(
    flags.DEFINE_string,
    "sft_checkpoint_path",
    "",
    "Path to SFT trained checkpoint directory (defaults to latest checkpoint in base_output_directory/run_name).",
)


def _format_path(key_path) -> str:
  """Formats a JAX PyTree key path into a readable slash-separated string."""
  parts = []
  for k in key_path:
    if hasattr(k, "key"):
      parts.append(str(k.key))
    elif hasattr(k, "name"):
      parts.append(str(k.name))
    elif hasattr(k, "idx"):
      parts.append(str(k.idx))
    else:
      parts.append(str(k))
  return "/".join(parts)


def resolve_sft_checkpoint_path(config: Any, explicit_path: str = "") -> str:
  """Finds the latest SFT checkpoint step directory if not explicitly provided."""
  if explicit_path:
    return explicit_path
  ckpt_dir = epath.Path(config.base_output_directory) / config.run_name / "checkpoints"
  if ckpt_dir.exists():
    step_dirs = [d.name for d in ckpt_dir.iterdir() if d.name.isdigit() and d.is_dir()]
    if step_dirs:
      latest_step = max(step_dirs, key=int)
      return str(ckpt_dir / latest_step / "items")
  raise ValueError(f"No valid SFT checkpoint steps found in {ckpt_dir}.")


def compare_checkpoints(
    config: Any,
    stitched_checkpoint_path: Optional[str] = None,
    sft_checkpoint_path: Optional[str] = None,
) -> bool:
  """Compares stitched checkpoint against SFT checkpoint and verifies freezing.

  Args:
    config: The MaxText configuration object.
    stitched_checkpoint_path: Path to original pre-SFT stitched checkpoint.
    sft_checkpoint_path: Path to post-SFT checkpoint (or None for auto-discovery).

  Returns:
    True if only trainable projector MLP parameters changed and all other weights
    are 100% identical. False otherwise.
  """
  stitched_path = stitched_checkpoint_path or getattr(config, "load_parameters_path", "")
  if not stitched_path:
    raise ValueError("stitched_checkpoint_path or load_parameters_path must be provided.")

  sft_path = resolve_sft_checkpoint_path(config, sft_checkpoint_path or "")

  print("=" * 80)
  print(f"  Stitched Original Checkpoint: {stitched_path}")
  print(f"  SFT Trained Checkpoint:       {sft_path}")
  print("=" * 80)

  mesh = maxtext_utils.get_mesh_from_config(config)
  with jax.set_mesh(mesh):
    model = model_creation_utils.from_config(config, mesh=mesh)
    abstract_vars = maxtext_utils.get_abstract_param(model, config)
    target_params_abstract = max_utils.unbox_logicallypartioned(abstract_vars["params"])

    # Load original stitched checkpoint
    stitched_loaded = checkpointing.load_params_from_path(
        stitched_path,
        {"params": target_params_abstract},
        config.checkpoint_storage_concurrent_gb,
        use_ocdbt=config.checkpoint_storage_use_ocdbt,
        use_zarr3=config.checkpoint_storage_use_zarr3,
    )
    stitched_params = stitched_loaded.get("params", stitched_loaded)

    # Load SFT trained checkpoint
    sft_loaded = checkpointing.load_params_from_path(
        sft_path,
        {"params": target_params_abstract},
        config.checkpoint_storage_concurrent_gb,
        use_ocdbt=config.checkpoint_storage_use_ocdbt,
        use_zarr3=config.checkpoint_storage_use_zarr3,
    )
    sft_params = sft_loaded.get("params", sft_loaded)

  stitched_leaves = {_format_path(k): v for k, v in jax.tree_util.tree_leaves_with_path(stitched_params)}
  sft_leaves = {_format_path(k): v for k, v in jax.tree_util.tree_leaves_with_path(sft_params)}

  # Validate trainable parameter mask (ensure this test runs only on masked training runs)
  trainable_masks = getattr(config, "trainable_parameters_mask", None)
  if not trainable_masks:
    print(
        "Error: 'trainable_parameters_mask' is empty or not provided. "
        "This test specifically verifies custom MLP projector training with frozen encoder and decoder. "
        "Exiting."
    )
    return False
  elif isinstance(trainable_masks, str):
    trainable_masks = [trainable_masks]

  def _is_trainable(path: str) -> bool:
    """Returns True if the tensor path matches any trainable mask pattern."""
    return any(mask in path for mask in trainable_masks)

  stats = {
      "Vision Encoder": {"total": 0, "identical": 0, "different": 0, "expected": "FROZEN"},
      "Projector MLP": {"total": 0, "identical": 0, "different": 0, "expected": "TRAINED"},
      "LLM Backbone": {"total": 0, "identical": 0, "different": 0, "expected": "FROZEN"},
  }
  projector_diffs = []
  unexpected_diffs = []

  # Go through all parameter tensors and compare weights before vs after SFT
  sft_keys = set(sft_leaves.keys())
  stitched_keys = set(stitched_leaves.keys())
  if sft_keys != stitched_keys:
    missing_in_sft = stitched_keys - sft_keys
    missing_in_stitched = sft_keys - stitched_keys
    if missing_in_sft:
      print(f"Error: Parameters in stitched but missing in SFT checkpoint: {missing_in_sft}")
    if missing_in_stitched:
      print(f"Error: Parameters in SFT but missing in stitched checkpoint: {missing_in_stitched}")
    return False
  for path, sft_arr in sft_leaves.items():
    stitched_arr = stitched_leaves[path]

    if _is_trainable(path):
      cat = "Projector MLP"
    elif path.startswith("vision_encoder"):
      cat = "Vision Encoder"
    else:
      cat = "LLM Backbone"

    # Align array sharding in JAX so both arrays share the same TPU sharding
    if hasattr(sft_arr, "sharding") and hasattr(stitched_arr, "sharding") and sft_arr.sharding != stitched_arr.sharding:
      stitched_arr = jax.device_put(stitched_arr, sft_arr.sharding)

    # Compute maximum absolute difference between checkpoints using JAX
    diff = jnp.max(jnp.abs(sft_arr.astype(jnp.float32) - stitched_arr.astype(jnp.float32)))
    max_abs_diff = float(diff)
    is_identical = max_abs_diff == 0.0

    stats[cat]["total"] += 1
    if is_identical:
      stats[cat]["identical"] += 1
    else:
      stats[cat]["different"] += 1
      if cat == "Projector MLP":
        projector_diffs.append((path, max_abs_diff))
      else:
        unexpected_diffs.append((path, max_abs_diff))

  # Test summary
  print("\n" + "-" * 60)
  print("Checkpoint Freezing & Training Verification:")
  print("-" * 60)

  all_passed = True
  for cat, data in stats.items():
    if data["expected"] == "FROZEN":
      passed = data["different"] == 0 and data["total"] > 0
      detail = f"{data['identical']}/{data['total']} tensors frozen"
    else:
      passed = data["different"] > 0 and data["total"] > 0
      max_diff_val = max((d for _, d in projector_diffs), default=0.0)
      detail = f"{data['different']}/{data['total']} tensors updated (max diff: {max_diff_val:.6f})"

    if not passed:
      all_passed = False
    tag = "PASS" if passed else "FAIL"
    print(f"  [{tag}] {cat:<15}: {detail}")

  print("-" * 60)
  if unexpected_diffs:
    print("Unexpected changes in frozen weights:")
    for path, d in unexpected_diffs:
      print(f"    - {path}: diff={d:.6f}")
    print("-" * 60)

  return all_passed


def main(argv: Sequence[str]) -> None:
  argv = list(argv)

  # Find YAML config path from CLI arguments or default flag
  config_path = None
  for a in argv[1:]:
    if a.endswith(".yml") or a.endswith(".yaml"):
      config_path = a
      break

  if not config_path:
    config_path = FLAGS.config_path
    argv.insert(1, config_path)

  # Set required flags for checkpoint inspection
  if not any(a.startswith("override_model_config=") for a in argv):
    argv.append("override_model_config=True")
  if not any(a.startswith("skip_jax_distributed_system=") for a in argv):
    argv.append("skip_jax_distributed_system=True")
  if not any(a.startswith("ici_fsdp_parallelism=") for a in argv):
    argv.append("ici_fsdp_parallelism=1")
  if not any(a.startswith("ici_tensor_parallelism=") for a in argv):
    argv.append("ici_tensor_parallelism=-1")

  config = pyconfig.initialize(
      argv,
      override_model_config=True,
      skip_jax_distributed_system=True,
      log_config=False,
  )

  passed = compare_checkpoints(
      config=config,
      stitched_checkpoint_path=FLAGS.stitched_checkpoint_path or getattr(config, "load_parameters_path", ""),
      sft_checkpoint_path=FLAGS.sft_checkpoint_path,
  )

  # Exit with non-zero code if frozen weights changed or projector wasn't updated
  if not passed:
    sys.exit(1)


if __name__ == "__main__":
  app.run(main)
