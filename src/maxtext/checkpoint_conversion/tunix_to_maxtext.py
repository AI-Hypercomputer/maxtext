"""Rewrites post-training checkpoints written in Tunix's layout into MaxText's.

Post-training used to write

  <run>/checkpoints/<step>/model_params/       the whole nnx.state(model)
  <run>/checkpoints/<step>/optimizer_state/    opt_state and step

which pre-training cannot read. It now writes

  <run>/checkpoints/<step>/items/              params, opt_state, step, nnx_aux

Trainers still restore the old layout, so a run resumed from one converts itself on its next
save. This script is for the checkpoints nobody is going to resume: it rewrites them in place of
a training run, so pre-training can pick them up.

The split between `params` and `nnx_aux` is the reason this needs a model rather than being a
rename. Both live mixed together under model_params -- weights beside the rng counters that drive
dropout -- and only the model says which is which. Weights are never materialised: an abstract
model supplies the classification and the arrays are streamed through.

Usage:
  python -m maxtext.checkpoint_conversion.tunix_to_maxtext src/maxtext/configs/base.yml \
    --source gs://bucket/run/checkpoints \
    model_name=llama3.1-8b scan_layers=True \
    base_output_directory=gs://bucket/run_converted \
    hardware=cpu skip_jax_distributed_system=True

RL keeps an actor and a reference model, so its steps sit under checkpoints/actor/<step>. Both
shapes are found automatically.
"""

import argparse
import sys

import absl
from etils import epath
import jax
import numpy as np
from flax import nnx
import orbax.checkpoint as ocp

from maxtext.common import train_state_nnx
from maxtext.configs import pyconfig
from maxtext.utils import max_logging, model_creation_utils

absl.logging.set_verbosity(absl.logging.INFO)  # for max_logging.log

_ITEM_NAME = "items"
_OLD_MODEL_ITEM = "model_params"
_OLD_OPTIMIZER_ITEM = "optimizer_state"


def find_old_checkpoints(root):
  """Returns (step, step_directory) for every step still in the Tunix layout.

  RL nests its steps one level deeper, under actor/, so both shapes are searched. A step that
  already carries an items/ directory has been converted and is skipped.
  """
  root = epath.Path(root)
  if not root.exists():
    raise FileNotFoundError(f"No checkpoint directory at {root}")

  found = []
  for parent in (root, *(c for c in root.iterdir() if c.is_dir() and not c.name.isdigit())):
    for step_dir in parent.iterdir():
      if not step_dir.is_dir() or not step_dir.name.isdigit():
        continue
      if (step_dir / _ITEM_NAME).exists():
        max_logging.log(f"Skipping {step_dir}: already in MaxText's layout")
        continue
      if (step_dir / _OLD_MODEL_ITEM).exists():
        found.append((int(step_dir.name), step_dir))
  return sorted(found)


def model_layout(config):
  """Returns (aux_paths, all_paths) for the model the checkpoints were trained with.

  Read off an abstract model, so nothing is allocated for it. `all_paths` is what the checkpoint
  is checked against: a model that does not match the checkpoint would otherwise split it by the
  wrong rule and write something that looks fine and is not.
  """
  _, abstract_model = model_creation_utils.create_nnx_abstract_model(config)
  state = nnx.state(abstract_model)
  linen_state, aux_state, _ = train_state_nnx.split_for_checkpoint(state)
  # split_for_checkpoint nests under "model" when it is handed a train state and does not when it
  # is handed the model's own state, which is what an abstract model gives.
  aux = aux_state.to_pure_dict()
  weights = linen_state.to_pure_dict()
  aux_tree = aux.get("model", aux)
  weight_tree = weights.get("model", weights)
  # Shapes are compared for weights only. An rng key is a typed key on the model side and the pair
  # of uint32 it serialises to on disk, which is a difference in representation, not in the model.
  return _leaf_paths(aux_tree), _leaf_paths(aux_tree) | set(_leaf_shapes(weight_tree)), _leaf_shapes(weight_tree)


def _leaf_shapes(tree, prefix=()):
  """Returns {path: shape} for every leaf, with None where a leaf carries no shape."""
  if not isinstance(tree, dict):
    return {prefix: tuple(tree.shape) if hasattr(tree, "shape") else None}
  shapes = {}
  for key, value in tree.items():
    shapes.update(_leaf_shapes(value, prefix + (str(key),)))
  return shapes


def check_model_matches(model_tree, expected, weight_shapes, step_dir):
  """Fails if the checkpoint does not hold exactly the model's weights, at the model's shapes.

  A wrong model_name is the easy mistake to make here, and it does not announce itself: the split
  would route the leaves it does not recognise to nnx_aux. Shapes are checked too, because a
  config that differs only in size -- a vocab that grew, a width that changed -- has all the right
  names and produces a checkpoint that fails much later, while being read.
  """
  found = _leaf_shapes(model_tree)
  missing, extra = expected - set(found), set(found) - expected
  mismatched = [
      p
      for p in set(weight_shapes) & set(found)
      if weight_shapes[p] is not None and found[p] is not None and weight_shapes[p] != found[p]
  ]
  if not missing and not extra and not mismatched:
    return
  detail = []
  if missing:
    detail.append(f"{len(missing)} in the model but not the checkpoint, e.g. {'.'.join(sorted(missing)[0])}")
  if extra:
    detail.append(f"{len(extra)} in the checkpoint but not the model, e.g. {'.'.join(sorted(extra)[0])}")
  if mismatched:
    path = sorted(mismatched)[0]
    detail.append(
        f"{len(mismatched)} at a different shape, e.g. {'.'.join(path)} is {found[path]}, "
        f"model wants {weight_shapes[path]}"
    )
  raise ValueError(
      f"{step_dir} does not match the model this was pointed at ({'; '.join(detail)}). "
      "Pass the model_name, scan_layers and dimensions the checkpoint was trained with."
  )


def _leaf_paths(tree, prefix=()):
  """Returns the path of every leaf in `tree`, as tuples of keys."""
  if not isinstance(tree, dict):
    return {prefix}
  paths = set()
  for key, value in tree.items():
    paths |= _leaf_paths(value, prefix + (str(key),))
  return paths


def _partition(tree, wanted, prefix=()):
  """Splits `tree` into (matching, remaining) by whether a leaf's path is in `wanted`."""
  if not isinstance(tree, dict):
    return (tree, None) if prefix in wanted else (None, tree)
  matching, remaining = {}, {}
  for key, value in tree.items():
    hit, miss = _partition(value, wanted, prefix + (str(key),))
    if hit is not None:
      matching[key] = hit
    if miss is not None:
      remaining[key] = miss
  return (matching or None), (remaining or None)


def _unwrap_variables(tree):
  """Turns the `{'value': array}` a serialised nnx.Variable becomes back into the array."""
  if isinstance(tree, dict):
    if set(tree) == {"value"}:
      return tree["value"]
    return {k: _unwrap_variables(v) for k, v in tree.items()}
  return tree


def _restore_as_numpy(path):
  """Reads a checkpoint item into numpy arrays.

  Restoring into jax arrays would want a sharding per leaf, and this only moves bytes from one
  layout to another, so it never needs the values on a device.
  """
  checkpointer = ocp.PyTreeCheckpointer()
  tree = checkpointer.metadata(path).item_metadata
  restore_args = jax.tree.map(lambda _: ocp.RestoreArgs(restore_type=np.ndarray), tree)
  return checkpointer.restore(path, args=ocp.args.PyTreeRestore(item=tree, restore_args=restore_args))


def convert_step(step_dir, aux, expected, weight_shapes):
  """Reads one step in the Tunix layout and returns the items dict MaxText's layout wants."""
  model_tree = _unwrap_variables(_restore_as_numpy(step_dir / _OLD_MODEL_ITEM))
  optimizer_tree = _unwrap_variables(_restore_as_numpy(step_dir / _OLD_OPTIMIZER_ITEM))
  check_model_matches(model_tree, expected, weight_shapes, step_dir)

  aux, params = _partition(model_tree, aux)

  items = {"params": train_state_nnx.to_linen_checkpoint_dict({"model": params or {}})["params"]}
  if aux:
    items["nnx_aux"] = {"model": aux}

  opt_state = optimizer_tree.get("opt_state")
  if opt_state is not None:
    # The trainers that schedule a learning rate wrap the optimizer, and pre-training reads mu and
    # nu from inside that shell, one level up from where the wrapper leaves them.
    inner = _drop_inject_hyperparams(opt_state)
    if inner is not opt_state:
      inner = train_state_nnx.opt_state_to_linen(inner)
    items["opt_state"] = inner
  if "step" in optimizer_tree:
    items["step"] = optimizer_tree["step"]
  else:
    items["step"] = int(step_dir.name)
  return items


def _drop_inject_hyperparams(opt_state):
  """Strips the optax.inject_hyperparams shell, matching what the trainers write today."""
  if isinstance(opt_state, dict) and {"count", "hyperparams", "hyperparams_states", "inner_state"}.issubset(opt_state):
    return opt_state["inner_state"]
  return opt_state


def write_step(items, out_dir):
  """Writes one converted step, under the items/ name the new layout uses."""
  out_dir = epath.Path(out_dir)
  ocp.PyTreeCheckpointer().save(out_dir / _ITEM_NAME, items)
  max_logging.log(f"Wrote {out_dir / _ITEM_NAME}")


def describe(items):
  """Returns a one-line summary of what a converted step holds."""
  counts = {}
  for name, subtree in items.items():
    counts[name] = len(_leaf_paths(subtree)) if isinstance(subtree, dict) else int(subtree)
  return ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))


def main(argv, source, dry_run=False):
  config = pyconfig.initialize(argv)
  destination = epath.Path(config.base_output_directory) / "checkpoints"

  steps = find_old_checkpoints(source)
  if not steps:
    max_logging.log(f"Nothing to convert under {source}")
    return

  max_logging.log(f"Converting {len(steps)} step(s) from {source}")
  aux, expected, weight_shapes = model_layout(config)

  for step, step_dir in steps:
    items = convert_step(step_dir, aux, expected, weight_shapes)
    max_logging.log(f"step {step}: {describe(items)}")
    if dry_run:
      continue
    # Keep the actor/ level RL writes, so a converted run reads back the way it was written.
    write_step(items, destination / step_dir.relative_to(epath.Path(source)))

  max_logging.log(f"Converted {len(steps)} step(s) into {destination}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--source", required=True, help="The run's checkpoints/ directory, in the Tunix layout.")
  parser.add_argument("--dry_run", action="store_true", help="Report what each step would hold without writing.")
  local_args, config_args = parser.parse_known_args()

  jax.config.update("jax_platforms", "cpu")
  main([sys.argv[0]] + config_args, local_args.source, dry_run=local_args.dry_run)
