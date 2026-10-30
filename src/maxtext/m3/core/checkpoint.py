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

"""Orbax checkpoint helpers and unscanned container layer-name translation for m3.

m3 restores both scanned and unscanned checkpoint layouts automatically:

* Scanned checkpoints: `decoder/layers/<param>` stacked at `param_scan_axis` (default `1`),
  matching existing scanned MaxText checkpoints directly with zero axis transpositions.
* Unscanned container checkpoints: `decoder/layers/layers/<i>` (`nnx.Sequential`), restored as-is.
* Unscanned flat-attribute checkpoints: `decoder/layers_<i>`, translated in memory via
  `match_checkpoint_layer_names` and `from_flat_layer_names` without copying array leaves.
"""

from collections.abc import Mapping
import json
import os
from typing import Any

from flax import nnx
import jax
from jax.sharding import NamedSharding
from orbax import checkpoint as ocp


def _is_index(key: Any) -> bool:
  return isinstance(key, int) or (isinstance(key, str) and key.isdecimal())


def _is_flat_layer_name(name: Any) -> bool:
  return isinstance(name, str) and name.startswith("layers_") and name[7:].isdecimal()


def _indexed_layers(subtree: Any) -> Mapping[Any, Any] | None:
  """Returns the `{index: layer}` mapping of an unscanned container, else `None`."""
  if isinstance(subtree, Mapping) and set(subtree) == {"layers"}:
    subtree = subtree["layers"]  # `nnx.Sequential` stores its children under `.layers`.
  if isinstance(subtree, Mapping) and subtree and all(_is_index(index) for index in subtree):
    return subtree
  return None


def _translate(tree: Any, to_flat: bool, nested: bool = True) -> Any:
  """Renames decoder layer keys recursively while reusing array leaves directly."""
  if isinstance(tree, nnx.State):
    raw_map = tree.raw_mapping if hasattr(tree, "raw_mapping") else dict(tree)
    return nnx.State(_translate(raw_map, to_flat=to_flat, nested=nested))
  if isinstance(tree, Mapping):
    result: dict[Any, Any] = {}
    for key, value in tree.items():
      if key == "decoder" and isinstance(value, Mapping):
        decoder: dict[Any, Any] = {}
        layers: dict[int, Any] = {}
        for name, subtree in value.items():
          indexed = _indexed_layers(subtree) if to_flat and name == "layers" else None
          if indexed is not None:
            decoder.update({f"layers_{index}": _translate(layer, to_flat, nested) for index, layer in indexed.items()})
          elif not to_flat and _is_flat_layer_name(name):
            layers[int(name[7:])] = _translate(subtree, to_flat, nested)
          else:
            decoder[name] = _translate(subtree, to_flat, nested)
        if layers:
          decoder["layers"] = {"layers": layers} if nested else layers
        result[key] = decoder
      else:
        result[key] = _translate(value, to_flat, nested)
    return result
  if isinstance(tree, list):
    return [_translate(value, to_flat, nested) for value in tree]
  if isinstance(tree, tuple):
    values = [_translate(value, to_flat, nested) for value in tree]
    return type(tree)(*values) if hasattr(tree, "_fields") else tuple(values)
  return tree


def to_flat_layer_names(tree: Any) -> Any:
  """Maps `decoder/layers/[layers/]<index>` to `decoder/layers_<index>` without copying array leaves."""
  return _translate(tree, to_flat=True)


def match_checkpoint_layer_names(target: Any, checkpoint_tree: Any) -> Any:
  """Renames restore `target` to `decoder/layers_<i>` if `checkpoint_tree` uses flat layer names."""
  decoder = checkpoint_tree.get("decoder", None) if hasattr(checkpoint_tree, "get") else None
  if hasattr(decoder, "keys") and any(_is_flat_layer_name(name) for name in decoder.keys()):
    return to_flat_layer_names(target)
  return target


def from_flat_layer_names(tree: Any, model_state: Any = None) -> Any:
  """Restores `nnx.Sequential` (`decoder/layers/layers/<i>`) or `nnx.List` container names from `layers_<i>`."""
  nested = True
  if model_state is not None:
    raw = model_state.raw_mapping if isinstance(model_state, nnx.State) else model_state
    model = raw.get("model", raw) if isinstance(raw, Mapping) else {}
    decoder = model.get("decoder", {}) if isinstance(model, Mapping) else {}
    layers = decoder.get("layers", {}) if isinstance(decoder, Mapping) else {}
    indexed = _indexed_layers(layers)
    if indexed is None:
      return tree
    nested = indexed is not layers
  return _translate(tree, to_flat=False, nested=nested)


def _stringify_int_keys(tree: Any) -> Any:
  """Converts integer dict keys (e.g. from `nnx.Sequential.layers`) to strings for Orbax PyTree serialization."""
  if isinstance(tree, Mapping):
    return {str(k) if isinstance(k, int) else k: _stringify_int_keys(v) for k, v in tree.items()}
  return tree


def _align_keys_to_template(restored: Any, template: Any) -> Any:
  """Restores integer dict keys matching `template` after Orbax deserialization."""
  if isinstance(template, Mapping) and isinstance(restored, Mapping):
    aligned: dict[Any, Any] = {}
    for k, tmpl_val in template.items():
      if k in restored:
        aligned[k] = _align_keys_to_template(restored[k], tmpl_val)
      elif isinstance(k, int) and str(k) in restored:
        aligned[k] = _align_keys_to_template(restored[str(k)], tmpl_val)
    return aligned
  return restored


def _state_to_pure_arrays(state: nnx.State) -> dict[str, Any]:
  pure = nnx.to_pure_dict(state)
  return _stringify_int_keys(pure)


def _build_restore_args(pure_target: Any) -> Any:
  def _arg(leaf: Any) -> Any:
    sharding = getattr(leaf, "sharding", None)
    if isinstance(sharding, NamedSharding):
      return ocp.ArrayRestoreArgs(sharding=sharding, dtype=leaf.dtype)
    return ocp.RestoreArgs(dtype=getattr(leaf, "dtype", None))

  return jax.tree.map(_arg, pure_target)


def save_m3_checkpoint(
    checkpoint_dir: str,
    step: int,
    model: nnx.Module,
    optimizer: nnx.Optimizer | None = None,
    data_iterator: Any = None,
    *,
    flat_layer_names: bool = False,
) -> str:
  """Saves model, optional optimizer, step metadata, and optional data iterator state via Orbax.

  Args:
    checkpoint_dir: Root directory under which `<step>/` will be written.
    step: Global training step index.
    model: `nnx.Module` to checkpoint.
    optimizer: Optional `nnx.Optimizer` whose state is saved alongside `model`.
    data_iterator: Optional stateful iterator exposing `get_state() -> dict`.
    flat_layer_names: When `True`, writes unscanned layers under `decoder/layers_<i>`
      (useful for testing cross-layout restore compatibility).

  Returns:
    Path to the written `<checkpoint_dir>/<step>` directory.
  """
  step_dir = os.path.abspath(os.path.join(checkpoint_dir, str(step)))
  os.makedirs(checkpoint_dir, exist_ok=True)

  model_dict = _state_to_pure_arrays(nnx.state(model))
  if flat_layer_names:
    model_dict = to_flat_layer_names(model_dict)

  payload: dict[str, Any] = {"model": model_dict}
  if optimizer is not None:
    opt_dict = _state_to_pure_arrays(nnx.state(optimizer))
    if flat_layer_names:
      opt_dict = to_flat_layer_names(opt_dict)
    payload["optimizer"] = opt_dict

  checkpointer = ocp.PyTreeCheckpointer()
  checkpointer.save(os.path.join(step_dir, "pytree"), payload, force=True)

  meta: dict[str, Any] = {"step": int(step)}
  if data_iterator is not None and hasattr(data_iterator, "get_state"):
    meta["data_iterator_state"] = data_iterator.get_state()
  meta_path = os.path.join(step_dir, "m3_meta.json")
  with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, sort_keys=True)

  return step_dir


def latest_checkpoint_step(checkpoint_dir: str) -> int | None:
  """Returns the highest numeric checkpoint step under `checkpoint_dir`, or `None`."""
  if not checkpoint_dir or not os.path.isdir(checkpoint_dir):
    return None
  steps = [
      int(entry)
      for entry in os.listdir(checkpoint_dir)
      if entry.isdecimal() and os.path.isdir(os.path.join(checkpoint_dir, entry, "pytree"))
  ]
  return max(steps) if steps else None


def restore_m3_checkpoint(
    checkpoint_dir: str,
    model: nnx.Module,
    optimizer: nnx.Optimizer | None = None,
    data_iterator: Any = None,
    *,
    step: int | None = None,
) -> int:
  """Restores model, optional optimizer, and optional data iterator state from an Orbax checkpoint.

  Automatically translates unscanned `decoder/layers_<i>` checkpoints into `nnx.Sequential`
  `decoder/layers/layers/<i>` state without copying array leaves, and preserves sharding specs
  on the target `model` and `optimizer`.

  Args:
    checkpoint_dir: Root checkpoint directory or direct `<step>` directory.
    model: Target `nnx.Module` (constructed on the active mesh) to update in place.
    optimizer: Optional `nnx.Optimizer` to update in place.
    data_iterator: Optional stateful iterator exposing `set_state(dict)`.
    step: Explicit step number to restore; defaults to latest step in `checkpoint_dir`.

  Returns:
    The restored integer `step`.
  """
  if os.path.isdir(os.path.join(checkpoint_dir, "pytree")):
    step_dir = os.path.abspath(checkpoint_dir)
  else:
    if step is None:
      step = latest_checkpoint_step(checkpoint_dir)
    if step is None:
      raise FileNotFoundError(f"No m3 checkpoints found under {checkpoint_dir!r}.")
    step_dir = os.path.abspath(os.path.join(checkpoint_dir, str(step)))

  pytree_dir = os.path.join(step_dir, "pytree")
  checkpointer = ocp.PyTreeCheckpointer()
  ckpt_metadata = checkpointer.metadata(pytree_dir)
  item_meta = getattr(ckpt_metadata, "item_metadata", ckpt_metadata)
  ckpt_tree = item_meta.tree if hasattr(item_meta, "tree") else item_meta
  ckpt_model_tree = ckpt_tree.get("model", ckpt_tree) if isinstance(ckpt_tree, Mapping) else {}

  model_state = nnx.state(model)
  model_pure_template = nnx.to_pure_dict(model_state)
  model_target = _stringify_int_keys(model_pure_template)
  model_target = match_checkpoint_layer_names(model_target, ckpt_model_tree)

  target_payload: dict[str, Any] = {"model": model_target}
  opt_state = None
  opt_pure_template = None
  if optimizer is not None and isinstance(ckpt_tree, Mapping) and "optimizer" in ckpt_tree:
    opt_state = nnx.state(optimizer)
    opt_pure_template = nnx.to_pure_dict(opt_state)
    opt_target = _stringify_int_keys(opt_pure_template)
    if isinstance(ckpt_model_tree, Mapping):
      decoder = ckpt_model_tree.get("decoder", {})
      if hasattr(decoder, "keys") and any(_is_flat_layer_name(k) for k in decoder.keys()):
        opt_target = to_flat_layer_names(opt_target)
    target_payload["optimizer"] = opt_target

  restore_args = _build_restore_args(target_payload)
  restored = checkpointer.restore(pytree_dir, item=target_payload, restore_args=restore_args)

  restored_model = from_flat_layer_names(restored["model"], model_state=model_state)
  restored_model = _align_keys_to_template(restored_model, model_pure_template)
  nnx.replace_by_pure_dict(model_state, restored_model)
  nnx.update(model, model_state)

  if opt_state is not None and opt_pure_template is not None and "optimizer" in restored:
    restored_opt = from_flat_layer_names(restored["optimizer"], model_state=model_state)
    restored_opt = _align_keys_to_template(restored_opt, opt_pure_template)
    nnx.replace_by_pure_dict(opt_state, restored_opt)
    nnx.update(optimizer, opt_state)

  restored_step = int(step) if step is not None else 0
  meta_path = os.path.join(step_dir, "m3_meta.json")
  if os.path.isfile(meta_path):
    with open(meta_path, "r", encoding="utf-8") as f:
      meta = json.load(f)
    restored_step = int(meta.get("step", restored_step))
    if data_iterator is not None and "data_iterator_state" in meta and hasattr(data_iterator, "set_state"):
      data_iterator.set_state(meta["data_iterator_state"])

  return restored_step
