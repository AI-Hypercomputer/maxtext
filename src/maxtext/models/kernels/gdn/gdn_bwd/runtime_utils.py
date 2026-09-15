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

import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp

try:
  from maxtext.models.kernels.gdn import compute_gdn as local_compute_gdn
except (ImportError, ModuleNotFoundError):
  try:
    from maxtext.src.maxtext.models.kernels.gdn import compute_gdn as local_compute_gdn
  except (ImportError, ModuleNotFoundError):
    from .. import compute_gdn as local_compute_gdn


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
  return local_compute_gdn.invert_triangular_matrix(t, block_size=16)


def _invert_triangular_matrix_fwd(t: jax.Array):
  t_inv = invert_triangular_matrix(t)
  return t_inv, t_inv


def _invert_triangular_matrix_bwd(res, g):
  t_inv = res
  grad_t = jnp.tril(-(t_inv.mT @ g @ t_inv.mT), k=-1)
  return (grad_t,)


invert_triangular_matrix.defvjp(
    _invert_triangular_matrix_fwd,
    _invert_triangular_matrix_bwd,
)
