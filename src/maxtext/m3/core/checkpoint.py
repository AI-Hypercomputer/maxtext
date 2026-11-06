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

"""Translate m3 container layer names to and from the legacy checkpoint layout.

m3 loads both checkpoint layouts automatically; standard MaxText models are
untouched (callers gate on `config.use_m3_model`):

* Scanned: m3 and legacy paths are identical (`decoder/layers/<param>`), so
  nothing is renamed.
* Unscanned m3 / new checkpoints: `decoder/layers/layers/<i>` (the decoder's
  `nnx.Sequential` keeps its children in `.layers`), restored as-is.
* Unscanned legacy checkpoints: `decoder/layers_<i>`; the restore target is
  renamed to match (`match_checkpoint_layer_names`) and the restored tree is
  renamed back (`from_legacy_layer_names`).

Only decoder layer names change. Arrays and other leaves are reused. The older
`nnx.List` layout (`decoder/layers/<i>`) is also recognized.
"""

from collections.abc import Mapping


def _is_index(key):
  return isinstance(key, int) or (isinstance(key, str) and key.isdecimal())


def _indexed_layers(subtree):
  """Returns the `{index: layer}` mapping of an unscanned container, else `None`."""
  if isinstance(subtree, Mapping) and set(subtree) == {"layers"}:
    subtree = subtree["layers"]  # `nnx.Sequential` keeps its children in `.layers`.
  if isinstance(subtree, Mapping) and subtree and all(_is_index(index) for index in subtree):
    return subtree
  return None


def _translate(tree, to_legacy, nested=True):
  """Rename decoder layer keys recursively, preserving array leaves."""
  if isinstance(tree, Mapping):
    result = {}
    for key, value in tree.items():
      if key == "decoder" and isinstance(value, Mapping):
        decoder = {}
        layers = {}
        for name, subtree in value.items():
          indexed = _indexed_layers(subtree) if to_legacy and name == "layers" else None
          if indexed is not None:
            decoder.update({f"layers_{index}": _translate(layer, to_legacy) for index, layer in indexed.items()})
          elif not to_legacy and _is_legacy_layer_name(name):
            layers[int(name[7:])] = _translate(subtree, to_legacy, nested)
          else:
            decoder[name] = _translate(subtree, to_legacy, nested)
        if layers:
          decoder["layers"] = {"layers": layers} if nested else layers
        result[key] = decoder
      else:
        result[key] = _translate(value, to_legacy, nested)
    return result
  if isinstance(tree, list):
    return [_translate(value, to_legacy, nested) for value in tree]
  if isinstance(tree, tuple):
    values = [_translate(value, to_legacy, nested) for value in tree]
    return type(tree)(*values) if hasattr(tree, "_fields") else tuple(values)
  return tree


def to_legacy_layer_names(tree):
  """Map decoder/layers/[layers/]<index> to decoder/layers_<index> without copying leaves."""
  return _translate(tree, to_legacy=True)


def _is_legacy_layer_name(name):
  return isinstance(name, str) and name.startswith("layers_") and name[7:].isdecimal()


def match_checkpoint_layer_names(target, checkpoint_tree):
  """Returns the restore `target` renamed to the legacy layout if `checkpoint_tree` uses it.

  Args:
    target: Restore target built from the m3 model state.
    checkpoint_tree: The on-disk tree (Orbax metadata) holding `decoder`.
  """
  decoder = checkpoint_tree.get("decoder", None) if hasattr(checkpoint_tree, "get") else None
  if hasattr(decoder, "keys") and any(_is_legacy_layer_name(name) for name in decoder.keys()):
    return to_legacy_layer_names(target)
  return target


def from_legacy_layer_names(tree, model_state=None):
  """Restore container names, optionally only when the target uses indexed layers.

  A target can be a model state or a train state with a `model` entry. Scanned
  and historical non-container targets retain their existing checkpoint names.
  The target also selects the `nnx.List` or `nnx.Sequential` layout; without
  one, the `nnx.Sequential` layout is produced.
  """
  nested = True
  if model_state is not None:
    model = model_state.get("model", model_state)
    decoder = model.get("decoder", {})
    layers = decoder.get("layers", {})
    indexed = _indexed_layers(layers)
    if indexed is None:
      return tree
    nested = indexed is not layers
  return _translate(tree, to_legacy=False, nested=nested)
