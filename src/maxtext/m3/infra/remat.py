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
byte-identical with and without remat; `RematTest` in `tests/unit/m3/layers_test.py` pins
that.

Policy names are legacy's, so a `remat_policy=` flag means the same thing on
both stacks. They refer to the `checkpoint_name` tags model files place on
activations (`query_proj`, `mlpwi`, ...), which are the only remat-related
thing a model file contains. The names are a convention, not an interface
JAX checks: a policy whose names a model never tags saves nothing and raises
nothing. `check_tags` makes that an error at setup.
"""

from collections.abc import Iterator
from typing import Any

from flax import nnx
import jax
import jax.extend as jex

from maxtext.m3.core.layers import Layers, Remat

_PROJECTIONS = ("query_proj", "value_proj", "key_proj", "kv_proj", "qkv_proj")
_MINIMAL = _PROJECTIONS + ("out_proj", "mlpwi_0", "mlpwi_1", "mlpwi", "mlpwo")


def _save(*names: str) -> Remat:
  return Remat(policy=jax.checkpoint_policies.save_only_these_names(*names), names=names)


def _offload(names: tuple[str, ...]) -> Remat:
  return Remat(
      policy=jax.checkpoint_policies.save_and_offload_only_these_names(
          names_which_can_be_saved=[],
          names_which_can_be_offloaded=list(names),
          offload_src="device",
          offload_dst="pinned_host",
      ),
      names=names,
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
      return Remat(policy=None, names=None)
    case "minimal" | "minimal_with_context" | "minimal_flash":
      return _save(*_MINIMAL)
    case "save_qkv_proj":
      return _save(*_PROJECTIONS)
    case "save_dot_except_mlp":
      return _save(*_PROJECTIONS, "out_proj")
    case "save_dot_except_mlpwi":
      return _save(*_PROJECTIONS, "out_proj", "mlpwo")
    case "save_out_proj":
      return _save("out_proj")
    case "qkv_proj_offloaded":
      return _offload(("query_proj", "value_proj", "key_proj", "kv_proj"))
    case "minimal_offloaded":
      return _offload(_MINIMAL)
    case "custom":
      on_device, to_offload = tuple(config.tensors_on_device), tuple(config.tensors_to_offload)
      return Remat(
          policy=jax.checkpoint_policies.save_and_offload_only_these_names(
              names_which_can_be_saved=list(on_device),
              names_which_can_be_offloaded=list(to_offload),
              offload_src="device",
              offload_dst="pinned_host",
          ),
          names=on_device + to_offload,
      )
    case other:
      raise ValueError(f"Unknown remat_policy {other!r} for m3.")


def apply_remat(model: nnx.Module, remat: Remat | None) -> nnx.Module:
  """Sets `remat` on every `Layers` container reachable from `model`, in place."""
  for _, node in nnx.iter_graph(model):
    if isinstance(node, Layers):
      node.remat = remat
  return model


def maybe_apply_remat(model: nnx.Module, config) -> nnx.Module:
  """Applies `config.remat_policy` to `model`. A no-op for `remat_policy=none`."""
  return apply_remat(model, resolve_policy(config))


def _jaxprs_in(value: Any) -> Iterator[jex.core.Jaxpr]:
  """The jaxprs held by one equation parameter: `scan`, `checkpoint`, `pjit`, `shard_map`, ..."""
  if isinstance(value, jex.core.ClosedJaxpr):
    yield value.jaxpr
  elif isinstance(value, jex.core.Jaxpr):
    yield value
  elif isinstance(value, (list, tuple)):
    for item in value:
      yield from _jaxprs_in(item)


def _names_in(jaxpr: jex.core.Jaxpr) -> Iterator[str]:
  for eqn in jaxpr.eqns:
    if eqn.primitive.name == "name":
      yield eqn.params["name"]
    for value in eqn.params.values():
      for sub in _jaxprs_in(value):
        yield from _names_in(sub)


def checkpoint_names(model: nnx.Module, *args: Any, **kwargs: Any) -> frozenset[str]:
  """The `checkpoint_name` tags one forward pass `model(*args, **kwargs)` emits.

  Traces the call (no compilation, no execution) and collects the tags,
  including those inside `jax.checkpoint` and `jax.lax.scan` bodies, so it
  sees the same tags whether or not remat and scan have been applied.
  """
  graphdef, state = nnx.split(model)

  def forward(state, *args):
    return nnx.merge(graphdef, state)(*args, **kwargs)

  return frozenset(_names_in(jax.make_jaxpr(forward)(state, *args).jaxpr))


def check_tags(model: nnx.Module, config, *example_args: Any, **example_kwargs: Any) -> frozenset[str]:
  """Raises if `config.remat_policy` names only tags `model` never emits.

  Such a policy is indistinguishable from `full` when it runs: everything is
  recomputed, the step is slower, and nothing reports it. One forward pass
  over the example inputs is traced to compare. Call it after scan, so the
  trace covers one layer body rather than every layer. Returns the tags
  found, which is also what a new model's author wants to see.
  """
  remat = resolve_policy(config)
  if remat is None or remat.names is None:
    return frozenset()
  found = checkpoint_names(model, *example_args, **example_kwargs)
  if not found & set(remat.names):
    raise ValueError(
        f"remat_policy={config.remat_policy!r} saves or offloads {sorted(remat.names)}, but the model tags "
        f"{sorted(found)}; it would save nothing. Tag the activations with those names or choose another policy."
    )
  return found
