# Copyright 2023-2025 Google LLC
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

"""Minimal NNX training script for testing NNX models.

Usage:
  python3 -m MaxText.simple_nnx_train MaxText/configs/base.yml \\
    run_name=my-run \\
    base_output_directory=gs://bucket/path \\
    model_name=gemma2-2b \\
    steps=20

  # Resume from checkpoint:
  python3 -m MaxText.simple_nnx_train MaxText/configs/base.yml \\
    run_name=my-run \\
    load_full_state_path=gs://bucket/path/checkpoints/0/items \\
    steps=100
"""

import os
import sys
import traceback

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import statelib
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import optax
import orbax.checkpoint as ocp
from etils import epath

# Add MaxText to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from MaxText import pyconfig
from MaxText import max_utils
from MaxText import maxtext_utils
from MaxText import model_creation_utils
from MaxText import optimizers
from MaxText.common_types import MODEL_MODE_TRAIN


def create_synthetic_batch(config, mesh, step=0):
  """Creates synthetic batch with random tokens.

  Args:
    config: Configuration object.
    mesh: JAX mesh for sharding.
    step: Training step number for varying data.

  Returns:
    Batch dict with sharded inputs, targets, positions, and segmentations.
  """
  batch_size = config.global_batch_size_to_train_on
  seq_len = config.max_target_length
  vocab_size = config.vocab_size

  rng = jax.random.PRNGKey(42 + step)
  tokens = jax.random.randint(rng, (batch_size, seq_len + 1), 0, vocab_size, dtype=jnp.int32)

  positions = jnp.broadcast_to(
      jnp.arange(seq_len + 1, dtype=jnp.int32).reshape(1, -1),
      (batch_size, seq_len + 1),
  )

  segmentation = jnp.ones((batch_size, seq_len), dtype=jnp.int32)

  batch = {
      "inputs": tokens[:, :-1],
      "inputs_position": positions[:, :-1],
      "inputs_segmentation": segmentation,
      "targets": tokens[:, 1:],
      "targets_position": positions[:, 1:],
      "targets_segmentation": segmentation,
  }

  data_pspec = P(*config.data_sharding)
  data_sharding = NamedSharding(mesh, data_pspec)

  batch = jax.tree.map(lambda x: jax.device_put(x, data_sharding), batch)
  return batch


def loss_fn(model, batch, config):
  """Computes masked cross-entropy loss."""
  logits = model(
      decoder_input_tokens=batch["inputs"],
      decoder_positions=batch["inputs_position"],
      decoder_segment_ids=batch["inputs_segmentation"],
      enable_dropout=config.enable_dropout,
      decoder_target_tokens=batch["targets"],
      decoder_target_mask=batch["targets_segmentation"],
  )

  one_hot_targets = jax.nn.one_hot(batch["targets"], config.vocab_size)
  xent, _ = max_utils.cross_entropy_with_logits(logits, one_hot_targets)

  xent = xent * (batch["targets_segmentation"] != 0)
  total_loss = jnp.sum(xent)
  total_weights = jnp.sum(batch["targets_segmentation"] != 0)
  loss = total_loss / (total_weights + 1e-8)

  return loss


def train_step(model, tx, opt_state, batch, config):
  """Performs one training step with gradient update."""

  def _loss_fn(model):
    return loss_fn(model, batch, config)

  loss, grads = nnx.value_and_grad(_loss_fn)(model)
  params = nnx.state(model, nnx.Param)

  # Convert NNX State/Variables to plain dicts for optax.
  def state_to_dict(tree):
    """Converts NNX State/Variables to plain dicts/arrays."""
    if isinstance(tree, statelib.State):
      result = {}
      for key, value in tree.items():
        result[key] = state_to_dict(value)
      return result
    elif isinstance(tree, nnx.Variable):
      return tree.value
    elif isinstance(tree, dict):
      return {k: state_to_dict(v) for k, v in tree.items()}
    elif isinstance(tree, (list, tuple)):
      return type(tree)(state_to_dict(item) for item in tree)
    else:
      return tree

  grads_values = state_to_dict(grads)
  params_values = state_to_dict(params)

  updates, new_opt_state = tx.update(grads_values, opt_state, params_values)
  new_params_values = optax.apply_updates(params_values, updates)

  # Reconstruct NNX State from plain dict.
  def dict_to_state(template, values_dict):
    """Converts plain dict back to NNX State with Variable wrappers."""
    if isinstance(template, statelib.State):
      result = {}
      for key, template_val in template.items():
        if key in values_dict:
          result[key] = dict_to_state(template_val, values_dict[key])
        else:
          result[key] = template_val
      return statelib.State(result)
    elif isinstance(template, nnx.Variable):
      return type(template)(values_dict)
    elif isinstance(template, dict):
      return {k: dict_to_state(template[k], values_dict[k]) for k in template}
    elif isinstance(template, (list, tuple)):
      return type(template)(dict_to_state(t, v) for t, v in zip(template, values_dict))
    else:
      return values_dict

  new_params = dict_to_state(params, new_params_values)
  nnx.update(model, new_params)

  return loss, new_opt_state


def create_checkpoint_manager(checkpoint_dir: str, config):
  """Creates Orbax CheckpointManager for saving checkpoints."""
  p = epath.Path(checkpoint_dir)
  p.mkdir(exist_ok=True, parents=True)

  manager = ocp.CheckpointManager(
      p,
      item_names=("items",),
      item_handlers={
          "items": ocp.PyTreeCheckpointHandler(
              use_ocdbt=config.checkpoint_storage_use_ocdbt,
              use_zarr3=config.checkpoint_storage_use_zarr3,
          )
      },
      options=ocp.CheckpointManagerOptions(
          create=True,
          enable_async_checkpointing=False,
      ),
  )
  return manager


def save_checkpoint(checkpoint_manager, model, opt_state, step: int, config):
  """Saves full training state (params, opt_state, step) in standard NNX format.

  NNX checkpoint format:
    - params: {'decoder': {'layer': {'kernel': {'value': array}}}}
    - opt_state: {'mu': {'decoder': {..., {'value': array}}}, 'nu': {...}}
    - step: scalar array

  This format is compatible with MaxText's checkpoint conversion utilities.
  """
  print(f"Saving checkpoint at step {step}...")

  params = nnx.state(model, nnx.Param)

  def state_to_nnx_format(tree):
    """Converts NNX State to standard NNX checkpoint format with {'value': ...} wrappers."""
    if isinstance(tree, statelib.State):
      result = {}
      for key, value in tree.items():
        result[key] = state_to_nnx_format(value)
      return result
    elif isinstance(tree, nnx.Variable):
      return {"value": tree.value}
    elif isinstance(tree, dict):
      return {k: state_to_nnx_format(v) for k, v in tree.items()}
    elif isinstance(tree, (list, tuple)):
      return type(tree)(state_to_nnx_format(item) for item in tree)
    else:
      return tree

  def wrap_arrays_with_value(tree):
    """Wraps arrays in {'value': ...} for opt_state."""
    if isinstance(tree, dict):
      return {k: wrap_arrays_with_value(v) for k, v in tree.items()}
    elif isinstance(tree, (list, tuple)):
      return type(tree)(wrap_arrays_with_value(item) for item in tree)
    elif hasattr(tree, "shape"):
      return {"value": tree}
    else:
      return tree

  params_nnx = state_to_nnx_format(params)
  opt_state_nnx = wrap_arrays_with_value(opt_state)

  state_dict = {
      "params": params_nnx,
      "opt_state": opt_state_nnx,
      "step": jnp.array(step, dtype=jnp.int32),
  }

  checkpoint_args = ocp.args.Composite(items=ocp.args.PyTreeSave(item=state_dict))

  checkpoint_manager.save(step, args=checkpoint_args)
  print(f"Checkpoint saved at step {step}")


def main(argv: list[str]):
  print("=" * 60)
  print("Simple NNX Training Script")
  print("=" * 60)

  default_overrides = [
      "enable_checkpointing=False",
      "enable_dropout=False",
      "scan_layers=True",
      "async_checkpointing=False",
      "ici_fsdp_parallelism=-1",
      "ici_tensor_parallelism=1",
      "remat_policy=full",
      "use_iota_embed=True",
      "per_device_batch_size=1",
      "max_target_length=128",
  ]

  config_args = argv[:2] + default_overrides + argv[2:]

  print("\nInitializing config...")
  config = pyconfig.initialize(config_args)

  print("\nConfig loaded:")
  print(f"  - Run name: {config.run_name}")
  print(f"  - Model: {config.model_name}")
  print(f"  - Decoder block: {config.decoder_block}")
  print(f"  - Batch size: {config.global_batch_size_to_train_on}")
  print(f"  - Sequence length: {config.max_target_length}")
  print(f"  - Vocab size: {config.vocab_size}")
  print(f"  - Num decoder layers: {config.num_decoder_layers}")
  print(f"  - Steps: {config.steps}")
  print(f"  - Output directory: {config.base_output_directory}")

  # Create mesh
  print("\nCreating device mesh...")
  devices_array = maxtext_utils.create_device_mesh(config)
  mesh = Mesh(devices_array, config.mesh_axes)
  print(f"  - Mesh shape: {mesh.shape}")
  print(f"  - Mesh axes: {mesh.axis_names}")

  print("\nCreating NNX model with proper sharding...")
  rng_key = jax.random.PRNGKey(config.init_weights_seed)

  model, _ = model_creation_utils.create_nnx_model(
      config,
      mesh=mesh,
      model_mode=MODEL_MODE_TRAIN,
      rng_key=rng_key,
  )

  print(f"  - Model type: {type(model).__name__}")

  checkpoint_path = config.load_parameters_path or config.load_full_state_path
  if checkpoint_path:
    checkpoint_type = "parameters" if config.load_parameters_path else "full state"
    print(f"\n✅ Loaded {checkpoint_type} from: {checkpoint_path}")
  else:
    print(f"\n  - Initialized with random weights (seed: {config.init_weights_seed})")

  print("\nCreating optimizer...")
  learning_rate_schedule = maxtext_utils.create_learning_rate_schedule(config)
  tx = optimizers.get_optimizer(config, learning_rate_schedule)

  params = nnx.state(model, nnx.Param)

  def state_to_dict_helper(tree):
    """Converts NNX State/Variables to plain dicts/arrays."""
    if isinstance(tree, statelib.State):
      result = {}
      for key, value in tree.items():
        result[key] = state_to_dict_helper(value)
      return result
    elif isinstance(tree, nnx.Variable):
      return tree.value
    elif isinstance(tree, dict):
      return {k: state_to_dict_helper(v) for k, v in tree.items()}
    elif isinstance(tree, (list, tuple)):
      return type(tree)(state_to_dict_helper(item) for item in tree)
    else:
      return tree

  opt_state_loaded = False
  checkpoint_path = config.load_parameters_path or config.load_full_state_path
  if checkpoint_path:
    try:
      print("  - Attempting to load optimizer state from checkpoint...")
      ckptr = ocp.PyTreeCheckpointer()
      restored = ckptr.restore(epath.Path(checkpoint_path))
      if "opt_state" in restored:

        def unwrap_value(tree):
          """
          Unwraps {'value': x} dicts from NNX checkpoint format.
          Args:
            tree: Nested structure from checkpoint.
          Returns:
            Unwrapped nested structure.
          Example:
            {"mu": {"layer1": {"kernel": {"value": array([...])}}}}
            becomes
            {"mu": {"layer1": {"kernel": array([...])}}}
          """
          if isinstance(tree, dict) and set(tree.keys()) == {"value"}:
            return tree["value"]
          elif isinstance(tree, dict):
            return {k: unwrap_value(v) for k, v in tree.items()}
          elif isinstance(tree, (list, tuple)):
            return type(tree)(unwrap_value(item) for item in tree)
          else:
            return tree

        unwrapped_opt_state = unwrap_value(restored["opt_state"])

        # Reconstruct optax NamedTuples from checkpoint dicts.
        params_values = state_to_dict_helper(params)
        dummy_opt_state = tx.init(params_values)

        def reconstruct_state(template, loaded):
          """
          Reconstructs optax state objects from checkpoint dicts.
          Args:
            template: Template object (optax NamedTuple or structure).
            loaded: Loaded dict structure from checkpoint.
          Returns:
            Reconstructed optax state object.
          Example:
            template: OptState(count=Array(...), mu=..., nu=...)
            loaded: {'count': array([...]), 'mu': {...}, 'nu': {...}}
            becomes:
            OptState(count=array([...]), mu=..., nu=...)
          """
          if hasattr(template, "_fields"):
            if isinstance(loaded, dict):
              return type(template)(**{k: reconstruct_state(getattr(template, k), loaded[k]) for k in template._fields})
            else:
              return loaded
          elif isinstance(template, (list, tuple)):
            return type(template)(reconstruct_state(t, loaded_item) for t, loaded_item in zip(template, loaded))
          else:
            return loaded

        opt_state = reconstruct_state(dummy_opt_state, unwrapped_opt_state)
        opt_state_loaded = True

        if isinstance(opt_state, (list, tuple)) and len(opt_state) > 0:
          first_elem = opt_state[0]
          if hasattr(first_elem, "count"):
            count_val = first_elem.count
            print(f"  - Optimizer count: {count_val}")
          if hasattr(first_elem, "mu"):
            mu_leaves = jax.tree.leaves(first_elem.mu)
            if mu_leaves:
              sample_val = float(jnp.mean(jnp.abs(mu_leaves[0])))
              num_nonzero = jnp.sum(jnp.abs(mu_leaves[0]) > 1e-10)
              print(f"  - ✅ Loaded optimizer state (mu mean: {sample_val:.6e}, nonzero: {num_nonzero})")
          else:
            print("  - ✅ Loaded optimizer state from checkpoint")
        else:
          print("  - ✅ Loaded optimizer state from checkpoint")
      else:
        print("  - ⚠️  No optimizer state in checkpoint, initializing fresh")
    except (KeyError, ValueError, TypeError, OSError) as e:
      print(f"  - ⚠️  Could not load optimizer state: {e}")
      traceback.print_exc()

  if not opt_state_loaded:
    params_values = state_to_dict_helper(params)
    opt_state = tx.init(params_values)
    print("  - Initialized fresh optimizer state")
    if isinstance(opt_state, (list, tuple)) and len(opt_state) > 0:
      first_elem = opt_state[0]
      if hasattr(first_elem, "count"):
        print(f"    - Fresh count: {first_elem.count}")
      if hasattr(first_elem, "mu"):
        mu_leaves = jax.tree.leaves(first_elem.mu)
        if mu_leaves:
          sample_val = float(jnp.mean(jnp.abs(mu_leaves[0])))
          print(f"    - Fresh mu mean: {sample_val:.6e} (should be ~0)")

  print(f"  - Learning rate schedule: {type(learning_rate_schedule).__name__}")
  print(f"  - Initial learning rate: {learning_rate_schedule(0)}")
  print(f"  - Num param arrays: {len(jax.tree.leaves(params))}")

  print("\nJIT compiling train step...")

  @nnx.jit
  def jit_train_step(model, opt_state, batch):
    return train_step(model, tx, opt_state, batch, config)

  base_output_dir = config.base_output_directory
  if not (os.path.isabs(base_output_dir) or base_output_dir.startswith("gs://")):
    base_output_dir = os.path.abspath(base_output_dir)
  checkpoint_dir = os.path.join(base_output_dir, config.run_name, "checkpoints")
  print(f"\nCheckpoint directory: {checkpoint_dir}")
  checkpoint_manager = create_checkpoint_manager(checkpoint_dir, config)

  start_step = 0
  checkpoint_path = config.load_parameters_path or config.load_full_state_path
  if checkpoint_path:
    try:
      ckptr = ocp.PyTreeCheckpointer()
      restored = ckptr.restore(epath.Path(checkpoint_path))
      if "step" in restored:
        start_step = int(restored["step"])
        print(f"  - Resuming from step {start_step}")
    except (KeyError, ValueError, TypeError, OSError) as e:
      print(f"  - Could not load step from checkpoint: {e}")
      print("  - Starting from step 0")

  num_steps = config.steps
  print("\n" + "=" * 60)
  print(f"Starting training from step {start_step + 1} to {num_steps}...")
  print("=" * 60)

  if start_step == 0:
    save_checkpoint(checkpoint_manager, model, opt_state, 0, config)

  with mesh:
    for step_idx in range(start_step, num_steps):
      batch = create_synthetic_batch(config, mesh, step=step_idx)

      loss, opt_state = jit_train_step(model, opt_state, batch)
      jax.block_until_ready(loss)
      loss_val = float(loss)
      print(f"Step {step_idx + 1}/{num_steps}: loss = {loss_val:.4f}")

  print("\nTraining complete!")

  # Save final checkpoint
  print("\n" + "=" * 60)
  print("Saving final checkpoint...")
  print("=" * 60)
  save_checkpoint(checkpoint_manager, model, opt_state, num_steps, config)

  # Wait for checkpoint manager to finish
  checkpoint_manager.wait_until_finished()

  print("\n" + "=" * 60)
  print("Done!")
  print("=" * 60)
  print(f"\nCheckpoints saved to: {checkpoint_dir}")


if __name__ == "__main__":
  main(sys.argv)
