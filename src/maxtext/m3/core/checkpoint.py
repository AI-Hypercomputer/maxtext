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

"""Translate container layer names to the historical checkpoint layout.

Only decoder layer names change. Arrays and other leaves are reused, including
optimizer moments and persistent auxiliary variables nested anywhere in a tree.
"""

from collections.abc import Mapping


def _is_index(key):
  return isinstance(key, int) or (isinstance(key, str) and key.isdecimal())


def _translate(tree, to_legacy):
  """Rename decoder layer keys recursively, preserving array leaves."""
  if isinstance(tree, Mapping):
    result = {}
    for key, value in tree.items():
      if key == "decoder" and isinstance(value, Mapping):
        decoder = {}
        layers = {}
        for name, subtree in value.items():
          if (
              to_legacy
              and name == "layers"
              and isinstance(subtree, Mapping)
              and subtree
              and all(_is_index(index) for index in subtree)
          ):
            decoder.update({f"layers_{index}": _translate(layer, to_legacy) for index, layer in subtree.items()})
          elif not to_legacy and isinstance(name, str) and name.startswith("layers_") and name[7:].isdecimal():
            layers[int(name[7:])] = _translate(subtree, to_legacy)
          else:
            decoder[name] = _translate(subtree, to_legacy)
        if layers:
          decoder["layers"] = layers
        result[key] = decoder
      else:
        result[key] = _translate(value, to_legacy)
    return result
  if isinstance(tree, list):
    return [_translate(value, to_legacy) for value in tree]
  if isinstance(tree, tuple):
    values = [_translate(value, to_legacy) for value in tree]
    return type(tree)(*values) if hasattr(tree, "_fields") else tuple(values)
  return tree


def to_legacy_layer_names(tree):
  """Map decoder/layers/<index> to decoder/layers_<index> without copying leaves."""
  return _translate(tree, to_legacy=True)


def from_legacy_layer_names(tree, model_state=None):
  """Restore container names, optionally only when the target uses indexed layers.

  A target can be a model state or a train state with a `model` entry. Scanned
  and historical non-container targets retain their existing checkpoint names.
  """
  if model_state is not None:
    model = model_state.get("model", model_state)
    decoder = model.get("decoder", {})
    layers = decoder.get("layers", {})
    if not isinstance(layers, Mapping) or not layers or not all(_is_index(key) for key in layers):
      return tree
  return _translate(tree, to_legacy=False)
