# Copyright 2023–2026 Google LLC
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

"""Workarounds that let a `reduced` PartitionSpec tag survive the pipeline's stage vmap (JAX 0.12.1).

`pipeline_unreduced_weight_grads` tags the BSW-gathered stage weights `reduced`, so each microbatch's weight gradient
is an `unreduced` partial sum and the cross-device reduction happens once per repeat, at the transpose of the BSW
gather. Three spots in JAX drop or reject the tag on the way:

1. `jax.vmap` reads `spec[i]` of every mapped input to find the mesh axis it maps over; `PartitionSpec.__getitem__`
   raises once the spec carries reduced/unreduced axes.
2. `shard_map`'s batching rule rebuilds its in_specs with `pxla.batch_spec`, which also indexes the spec and returns
   a PartitionSpec without the tags.
3. Under a mesh with Auto axes, lowering marks every dimension of a sharding open. Shardy only attaches a dot's
   `unreduced` axes to the dot when the constraint after it is fully closed; otherwise it reduces the dot output in
   place. The pipeline's stage mesh has exactly one Auto axis, 'stage', and every unreduced value the stage function
   produces is a per-stage weight cotangent with the stage dimension leading (vmap maps the weights over axis 0), so
   those shardings are emitted closed, with 'stage' on dimension 0.

`apply()` installs all three. Each one only changes behavior for specs that carry reduced/unreduced axes.
"""

# This module patches JAX internals by design.
# pylint: disable=protected-access

import dataclasses

import jax
from jax._src import api as _jax_api
from jax._src import mesh as _mesh_lib
from jax._src import named_sharding as _named_sharding
from jax._src.interpreters import pxla as _pxla
from jax._src.util import tuple_insert as _tuple_insert
from jax.sharding import PartitionSpec as P

from packaging.version import Version

# Oldest JAX the tagged pipeline was verified on. JAX 0.11.1 asserts in einsum on a `reduced` operand.
MIN_JAX_VERSION = "0.12.1.dev20261020"

_APPLIED = False


def jax_supports_reduced_stage_weights() -> bool:
  return Version(jax.__version__) >= Version(MIN_JAX_VERSION)


def _mapped_axis_spec(args_flat, in_axes_flat):
  """`jax._src.api._mapped_axis_spec`, reading `.partitions` so a tagged spec does not raise."""
  out_spec = None
  non_none_count = 0
  for arg, i in zip(args_flat, in_axes_flat):
    if i is None:
      continue
    try:
      spec = _jax_api.shaped_abstractify(arg).sharding.spec.partitions[i]
    except (IndexError, TypeError, AttributeError):
      spec = None
    if non_none_count != 0 and out_spec != spec:
      raise ValueError(
          "Mapped away dimension of inputs passed to vmap should be sharded"
          f" the same. Got inconsistent axis specs: {out_spec} vs {spec}"
      )
    out_spec = spec
    non_none_count += 1
  if out_spec is not None and not isinstance(out_spec, tuple):
    out_spec = (out_spec,)
  return out_spec


def _batch_spec(spec, dim, val):
  """`pxla.batch_spec`, reading `.partitions` and keeping the reduced/unreduced tags."""
  partitions = tuple(spec.partitions) if isinstance(spec, P) else tuple(spec)
  if dim > len(partitions):
    partitions += (None,) * (dim - len(partitions))
  partitions = _tuple_insert(partitions, dim, val)
  return spec.update(partitions=partitions) if isinstance(spec, P) else P(*partitions)


_original_named_sharding_to_sdy_sharding = _named_sharding.named_sharding_to_sdy_sharding


def _named_sharding_to_sdy_sharding(self, num_dimensions, modify_wrt_axis_types):
  """Emits an unreduced sharding on the stage mesh fully closed, with 'stage' on dimension 0."""
  out = _original_named_sharding_to_sdy_sharding(self, num_dimensions, modify_wrt_axis_types)
  if not (modify_wrt_axis_types and self.spec.unreduced and num_dimensions):
    return out
  auto_axes = [name for name, axis_type in self.mesh._name_to_type.items() if axis_type == _mesh_lib.AxisType.Auto]
  if auto_axes != ["stage"] or out.dim_shardings[0].axes:
    return out
  dims = (_named_sharding.SdyDim(axes=("stage",), is_open=False),) + tuple(
      d.replace(is_open=False) for d in out.dim_shardings[1:]
  )
  return dataclasses.replace(out, dim_shardings=dims)


def apply():
  """Installs the workarounds (idempotent)."""
  global _APPLIED
  if _APPLIED:
    return
  _jax_api._mapped_axis_spec = _mapped_axis_spec
  _pxla.batch_spec = _batch_spec
  _named_sharding.named_sharding_to_sdy_sharding = _named_sharding_to_sdy_sharding
  _APPLIED = True
