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

"""Backwards-compatible Megablox GMM facade delegating to `kernels.gmm`."""

# pylint: disable=too-many-positional-arguments

from typing import List, Literal, Tuple

import jax
import jax.numpy as jnp
from maxtext.kernels.gmm import backends as gmm_backends
from maxtext.kernels.gmm import quantize as gmm_quantize
from maxtext.kernels.megablox import (  # pylint: disable=unused-import
    pallas_mosaic_tpu_v2_gmm_kernel as gmm_v2,
    pallas_mosaic_tpu_v2_tgmm_kernel as tgmm_v2,
)
import qwix
import qwix.pallas as qpl

DLHS_RAGGED_DOT_DIM_NUMS = gmm_backends.DLHS_RAGGED_DOT_DIM_NUMS
DRHS_RAGGED_DOT_DIM_NUMS = gmm_backends.DRHS_RAGGED_DOT_DIM_NUMS
TcUnsortCfg = gmm_quantize.TcUnsortCfg

# Kept as aliases for backwards compatibility with internal unit tests.
_fwd_prepare_lhs_scale = gmm_quantize.prepare_lhs_scale
_check_tc_unsort_bwd_supported = gmm_quantize.check_tc_unsort_bwd_supported
_lhs_rhs_share_fixed_scale = gmm_quantize.lhs_rhs_share_fixed_scale


def _dlhs_run_tokamax_v2(
    dlhs_dout: jnp.ndarray | qpl.QArray,
    rhs: jnp.ndarray,
    group_sizes: jnp.ndarray,
    group_offset: jnp.ndarray | None,
    lhs_dtype: jax.typing.DTypeLike,
    tiling: tuple,
    use_gmm_v2_heuristic_tiling: bool,
    partial_sum: jnp.ndarray | None = None,
    lhs_scale: jnp.ndarray | None = None,
    fuse_dlhs_scale: bool = False,
    use_dlhs_transpose_rhs: bool = False,
    out_is_3d: bool = False,
    fuse_3d_dlhs_ps: bool = False,
) -> tuple[jax.Array, bool]:
  """Backwards-compatible helper forwarding to `TokamaxV2Backend.dlhs`."""
  v2_backend = gmm_backends.TokamaxV2Backend(
      heuristic_tiling=use_gmm_v2_heuristic_tiling,
      dlhs_transpose_rhs=use_dlhs_transpose_rhs,
      fuse_dlhs_scale=fuse_dlhs_scale,
      fuse_3d_dlhs_ps=fuse_3d_dlhs_ps,
  )
  return v2_backend.dlhs(
      dlhs_dout,
      rhs,
      group_sizes,
      preferred_element_type=lhs_dtype,
      tiling=tiling,
      group_offset=group_offset,
      partial_sum=partial_sum,
      lhs_scale=lhs_scale,
      out_is_3d=out_is_3d,
  )


def gmm(
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    group_sizes: jnp.ndarray,
    preferred_element_type: jnp.dtype = jnp.float32,  # pyrefly: ignore[bad-function-definition]
    tiling: tuple[int, int, int, int, int, int, int, int, int] = (
        128,
        128,
        128,
        128,
        128,
        128,
        128,
        128,
        128,
    ),
    group_offset: jnp.ndarray | None = None,
    existing_out: jnp.ndarray | None = None,
    *,
    interpret: bool | None = None,
    lhs_quantize_dtype: Literal[jnp.int4, jnp.int8] | None = None,  # pyrefly: ignore[invalid-literal]
    rhs_quantize_dtype: Literal[jnp.int4, jnp.int8] | None = None,  # pyrefly: ignore[invalid-literal]
    use_qwix_quantization: bool = False,
    use_tokamax_backend: bool = False,
    weight_gather_axes: List[Tuple[str, int]] | None = None,
    lhs_vma_axes: tuple = tuple(),
    rhs_vma_axes: tuple = tuple(),
    # TODO(amandaliang): get rid of the qwix_rule in favor of Qwix's interception feature
    qwix_rule: qwix.QtRule | None = None,
    use_manual_quantization: bool = False,  # used in batchsplit
    use_gmm_v2: bool = False,
    use_gmm_v2_heuristic_tiling: bool = False,
    partial_sum: jnp.ndarray | None = None,
    return_rhs: bool = False,
    return_lhs: bool = False,
    fuse_dlhs_scale: bool = False,
    use_dlhs_transpose_rhs: bool = False,
    flat_fsdp_shard: tuple[int, int, int] | None = None,
    out_is_3d: bool = False,
    fuse_3d_dlhs_ps: bool = False,
    tc_routing: object = None,
    tc_unsort_cfg: TcUnsortCfg | None = None,
):
  """Backwards-compatible grouped matrix multiplication operation.

  Selects the backend from legacy `(use_tokamax_backend, use_gmm_v2)` flags and
  forwards execution through `kernels.gmm.quantize.quantized_gmm`.
  New code should use `kernels.gmm.select_gmm_backend` + `kernels.gmm.grouped_matmul`.
  """
  if use_tokamax_backend and use_gmm_v2:
    selected_backend: gmm_backends.GmmBackend = gmm_backends.TokamaxV2Backend(
        heuristic_tiling=use_gmm_v2_heuristic_tiling,
        dlhs_transpose_rhs=use_dlhs_transpose_rhs,
        fuse_dlhs_scale=fuse_dlhs_scale,
        fuse_3d_dlhs_ps=fuse_3d_dlhs_ps,
    )
  elif use_tokamax_backend:
    selected_backend = gmm_backends.TokamaxV1Backend()
  else:
    selected_backend = gmm_backends.MegabloxBackend()

  return gmm_quantize.quantized_gmm(
      lhs,
      rhs,
      group_sizes,
      backend=selected_backend,
      preferred_element_type=preferred_element_type,
      tiling=tiling,
      group_offset=group_offset,
      existing_out=existing_out,
      interpret=interpret,
      lhs_quantize_dtype=lhs_quantize_dtype,
      rhs_quantize_dtype=rhs_quantize_dtype,
      use_qwix_quantization=use_qwix_quantization,
      weight_gather_axes=weight_gather_axes,
      lhs_vma_axes=lhs_vma_axes,
      rhs_vma_axes=rhs_vma_axes,
      qwix_rule=qwix_rule,
      use_manual_quantization=use_manual_quantization,
      partial_sum=partial_sum,
      return_rhs=return_rhs,
      return_lhs=return_lhs,
      flat_fsdp_shard=flat_fsdp_shard,
      out_is_3d=out_is_3d,
      tc_routing=tc_routing,
      tc_unsort_cfg=tc_unsort_cfg,
  )
