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

import functools
import logging
from typing import Optional

import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp

from .. import compute_gdn as local_compute_gdn

_logger = logging.getLogger(__name__)


def pallas_unsupported_reason(
    *, head_k_dim: int, head_v_dim: int, chunk_size: int, seq_len: Optional[int] = None
) -> Optional[str]:
  """Returns why the Pallas TPU GDN kernel cannot run these shapes, or None if it can."""
  if head_k_dim % 128 != 0:
    return f"head_k_dim={head_k_dim} is not a multiple of 128"
  if head_v_dim % 128 != 0:
    return f"head_v_dim={head_v_dim} is not a multiple of 128"
  if chunk_size != 64:
    return f"chunk_size={chunk_size} != 64"
  if seq_len is not None and seq_len % chunk_size != 0:
    return f"local seq_len={seq_len} is not a multiple of chunk_size={chunk_size}"
  return None


@functools.lru_cache(maxsize=None)
def warn_gdn_pallas_fallback_once(where: str, reason: str) -> None:
  """Logs (once per `where`/`reason` pair) that the GDN Pallas TPU kernel was skipped."""
  _logger.warning("GDN %s: not using the Pallas TPU kernel (%s).", where, reason)


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

  try:
    from jax._src.pallas.mosaic import pipeline as _pl_pipeline  # noqa: E402

    if not getattr(_pl_pipeline, "_is_cpu_safe_cbs_patched", False):
      _orig_cbs = getattr(
          _pl_pipeline,
          "_original_create_bounded_slice",
          _pl_pipeline._create_bounded_slice,
      )
      _pl_pipeline._original_create_bounded_slice = _orig_cbs

      def _cpu_safe_create_bounded_slice(
          slice_start,
          slice_size,
          block_size,
          dim_size,
          *args,
          tiling=None,
          **kwargs,
      ):
        if isinstance(slice_size, int) and (tiling is None or slice_size % tiling == 0):
          return pl.ds(slice_start, slice_size)
        return _orig_cbs(
            slice_start,
            slice_size,
            block_size,
            dim_size,
            tiling,
            *args,
            **kwargs,
        )

      _pl_pipeline._create_bounded_slice = _cpu_safe_create_bounded_slice
      _pl_pipeline._is_cpu_safe_cbs_patched = True
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
