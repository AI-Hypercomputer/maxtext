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

"""Rematerialization, applied after construction.

Remat is not a wrapper module. Wrapping each layer in a `Remat(nnx.Module)`
would add a `module` segment to every parameter path, and parameter paths feed
checkpoint restore, HF weight mappings, optimizer masks and weight sync --
several of which fail silently on a path they do not recognise. Instead, every
`Layers` container in the model gets a `Remat` setting and checkpoints each
layer call itself (see `maxtext.m3.core.layers`). The parameter tree is
byte-identical with and without remat; `tests/unit/m3/remat_test.py` pins
that.

Policy names are legacy's, so a `remat_policy=` flag means the same thing on
both stacks. They refer to the `checkpoint_name` tags model files place on
activations (`query_proj`, `mlpwi`, ...), which are the only remat-related
thing a model file contains.
"""

from collections.abc import Callable

from flax import nnx
import jax

from maxtext.m3.core.layers import Layers, Remat

_PROJECTIONS = ("query_proj", "value_proj", "key_proj", "kv_proj", "qkv_proj")
_MINIMAL = _PROJECTIONS + ("out_proj", "mlpwi_0", "mlpwi_1", "mlpwi", "mlpwo")


def _save(*names: str) -> Callable[..., bool]:
  return jax.checkpoint_policies.save_only_these_names(*names)


def _offload(names: tuple[str, ...]) -> Callable[..., bool]:
  return jax.checkpoint_policies.save_and_offload_only_these_names(
      names_which_can_be_saved=[],
      names_which_can_be_offloaded=list(names),
      offload_src="device",
      offload_dst="pinned_host",
  )


def resolve_policy(config) -> Remat | None:
  """Maps `config.remat_policy` to a `Remat`, or `None` when remat is off.

  Covers the legacy names that need nothing beyond the tags m3 models carry.
  `minimal_with_context` saves the attention kernel's `context` output, which
  m3's attention does not tag yet, so it is accepted as plain `minimal`.
  """
  match config.remat_policy:
    case "none" | "" | None:
      return None
    case "full":
      policy = None
    case "minimal" | "minimal_with_context" | "minimal_flash":
      policy = _save(*_MINIMAL)
    case "save_qkv_proj":
      policy = _save(*_PROJECTIONS)
    case "save_dot_except_mlp":
      policy = _save(*_PROJECTIONS, "out_proj")
    case "save_dot_except_mlpwi":
      policy = _save(*_PROJECTIONS, "out_proj", "mlpwo")
    case "save_out_proj":
      policy = _save("out_proj")
    case "qkv_proj_offloaded":
      policy = _offload(("query_proj", "value_proj", "key_proj", "kv_proj"))
    case "minimal_offloaded":
      policy = _offload(_MINIMAL)
    case "custom":
      policy = jax.checkpoint_policies.save_and_offload_only_these_names(
          names_which_can_be_saved=list(config.tensors_on_device),
          names_which_can_be_offloaded=list(config.tensors_to_offload),
          offload_src="device",
          offload_dst="pinned_host",
      )
    case other:
      raise ValueError(f"Unknown remat_policy {other!r} for m3.")
  return Remat(policy=policy)


def apply_remat(model: nnx.Module, remat: Remat | None) -> nnx.Module:
  """Sets `remat` on every `Layers` container reachable from `model`, in place."""
  for _, node in nnx.iter_graph(model):
    if isinstance(node, Layers):
      node.remat = remat
  return model


def maybe_apply_remat(model: nnx.Module, config) -> nnx.Module:
  """Applies `config.remat_policy` to `model`. A no-op for `remat_policy=none`."""
  return apply_remat(model, resolve_policy(config))
