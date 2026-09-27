# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runtime utilities and CPU interpretation helpers for GDN backward pass."""

from typing import Any, Optional

import jax
import jax.numpy as jnp

from .. import compute_gdn as local_compute_gdn


def target_platform(*arrays: Any) -> str:
  """Returns the platform ('tpu', 'cpu', 'gpu') the computation is being compiled for.

  Like `pltpu.get_tpu_info()`, this prefers the active abstract mesh (`jax.set_mesh`, `shard_map`,
  `jax.sharding.use_abstract_mesh`) and then the inputs' mesh over `jax.default_backend()`, so CPU-host AOT
  compilation for a TPU mesh selects the Pallas TPU kernel instead of the CPU fallback.
  """
  meshes = [jax.sharding.get_abstract_mesh()]
  meshes += [jax.typeof(x).sharding.mesh for x in arrays if x is not None]
  for mesh in meshes:
    if mesh.abstract_device is not None:
      return mesh.abstract_device.platform
  return jax.default_backend()


def pallas_unsupported_reason(*, head_k_dim: int, head_v_dim: int, chunk_size: int) -> Optional[str]:
  """Returns why the Pallas TPU GDN kernel cannot run these shapes, or None if it can."""
  if head_k_dim % 128 != 0:
    return f"head_k_dim={head_k_dim} is not a multiple of 128"
  if head_v_dim % 128 != 0:
    return f"head_v_dim={head_v_dim} is not a multiple of 128"
  if chunk_size != 64:
    return f"chunk_size={chunk_size} != 64"
  return None


def ensure_cpu_interpret_registered() -> None:
  """Ensures Pallas CPU interpretation registers TPU hardware info without top-level import side-effects."""
  # pylint: disable=protected-access,broad-exception-caught,import-outside-toplevel
  try:
    from jax._src.pallas.mosaic import tpu_info as _ti  # noqa: E402

    if "cpu" not in _ti.registry:
      _ti.registry["cpu"] = lambda: _ti.get_tpu_info_for_chip(_ti.ChipVersion.TPU_V6E, 1)
    try:
      _ti.get_tpu_info.cache_clear()
    except Exception:
      pass
  except Exception:  # pragma: no cover
    pass
  # pylint: enable=protected-access,broad-exception-caught,import-outside-toplevel


@jax.custom_vjp
def invert_triangular_matrix(t: jax.Array) -> jax.Array:
  """Computes inverse of unit lower-triangular matrix using Tokamax block forward substitution."""
  return local_compute_gdn.invert_triangular_matrix(t, block_size=16, precision=jax.lax.Precision.HIGHEST)


def _invert_triangular_matrix_fwd(t: jax.Array):
  t_inv = local_compute_gdn.invert_triangular_matrix(t, block_size=16, precision=jax.lax.Precision.HIGHEST)
  return t_inv, t_inv


def _invert_triangular_matrix_bwd(res, g):
  """Backward VJP pass for unit lower-triangular matrix inversion."""
  t_inv = res
  high_prec = jax.lax.Precision.HIGHEST
  grad_t = jnp.tril(
      -jnp.matmul(
          jnp.matmul(t_inv.mT, g, precision=high_prec),
          t_inv.mT,
          precision=high_prec,
      ),
      k=-1,
  )
  return (grad_t,)


invert_triangular_matrix.defvjp(
    _invert_triangular_matrix_fwd,
    _invert_triangular_matrix_bwd,
)
