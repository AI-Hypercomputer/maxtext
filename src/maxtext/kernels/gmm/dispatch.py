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

"""Unified grouped matrix multiplication dispatch decoupling backend from quantization."""

# pylint: disable=too-many-positional-arguments

from typing import List, Literal, Tuple

import jax
import jax.numpy as jnp
from maxtext.kernels.gmm import backends as gmm_backends
from maxtext.kernels.gmm import quantize as gmm_quantize
import qwix
import qwix.pallas as qpl


def grouped_matmul(
    lhs: jnp.ndarray | qpl.QArray,
    rhs: jnp.ndarray | qpl.QArray,
    group_sizes: jnp.ndarray,
    *,
    backend: gmm_backends.GmmBackend,
    preferred_element_type: jax.typing.DTypeLike = jnp.float32,
    tiling: tuple = (128, 128, 128, 128, 128, 128, 128, 128, 128),
    group_offset: jnp.ndarray | None = None,
    existing_out: jnp.ndarray | None = None,
    interpret: bool | None = None,
    lhs_quantize_dtype: Literal[jnp.int4, jnp.int8] | None = None,  # pyrefly: ignore[invalid-literal]
    rhs_quantize_dtype: Literal[jnp.int4, jnp.int8] | None = None,  # pyrefly: ignore[invalid-literal]
    use_qwix_quantization: bool = False,
    weight_gather_axes: List[Tuple[str, int]] | None = None,
    lhs_vma_axes: tuple = (),
    rhs_vma_axes: tuple = (),
    qwix_rule: qwix.QtRule | None = None,
    use_manual_quantization: bool = False,
    partial_sum: jnp.ndarray | None = None,
    return_rhs: bool = False,
    return_lhs: bool = False,
    flat_fsdp_shard: tuple[int, int, int] | None = None,
    out_is_3d: bool = False,
    tc_routing: object = None,
    tc_unsort_cfg: gmm_quantize.TcUnsortCfg | None = None,
    bias: jnp.ndarray | None = None,
    use_rpa_group_sizes: bool = False,
    is_quantized_config: bool | None = None,
):
  """Executes grouped matrix multiplication, decoupling kernel backend from quantization.

  Direct native-autodiff execution (`backend.call`) is used when:
    - The backend supports native autodiff (`not backend.requires_custom_vjp`), AND
    - No custom-VJP-only MoE options are active (`weight_gather_axes`, `existing_out`,
      `partial_sum` on non-native backends, `return_lhs`, `return_rhs`, `flat_fsdp_shard`,
      `out_is_3d`, `tc_unsort_cfg`, VMA axes), AND
    - Either the call is unquantized, or quantization is handled natively by XLA/Qwix
      primitive interception (`backend.supports_native_qwix_interception`, e.g.
      `jax.lax.ragged_dot` inside a `qwix.QtProvider` without explicit `qwix_rule`,
      `lhs_quantize_dtype`/`rhs_quantize_dtype`, `use_manual_quantization`, or pre-quantized
      `QArray` operands).

  All quantized custom-kernel paths (Megablox, Tokamax v1 quantized, Tokamax v2, or explicit
  quantization on JaxRaggedDot) route through `quantized_gmm(..., backend=backend)`.
  """
  if bias is not None and not backend.supports_bias:
    raise NotImplementedError(f"Per-expert bias is not supported by backend {backend.name!r}.")

  if isinstance(backend, gmm_backends.TeGroupedGemmBackend):
    if return_lhs or return_rhs or out_is_3d or tc_unsort_cfg is not None:
      raise NotImplementedError(
          "use_te_grouped_gemm does not support return_lhs, return_rhs, out_is_3d or a fused TC unsort "
          f"(got {return_lhs=}, {return_rhs=}, {out_is_3d=}, tc_unsort={tc_unsort_cfg is not None})."
      )
    if (
        isinstance(lhs, qpl.QArray)
        or isinstance(rhs, qpl.QArray)
        or qwix_rule is not None
        or lhs_quantize_dtype is not None
        or rhs_quantize_dtype is not None
        or use_manual_quantization
        or bool(weight_gather_axes)
        or existing_out is not None
        or flat_fsdp_shard is not None
    ):
      raise NotImplementedError("TeGroupedGemmBackend does not support QArray/Qwix quantization, QAG, or existing_out.")
    return backend.call(
        lhs,
        rhs,
        group_sizes,
        preferred_element_type=preferred_element_type,
        tiling=tiling,
        group_offset=group_offset,
        bias=bias,
        partial_sum=partial_sum,
    )

  resolved_rule = gmm_quantize.resolve_quantization_rule(
      use_qwix_quantization=use_qwix_quantization,
      qwix_rule=qwix_rule,
      lhs_quantize_dtype=lhs_quantize_dtype,
      rhs_quantize_dtype=rhs_quantize_dtype,
  )
  has_prequantized = isinstance(lhs, qpl.QArray) or isinstance(rhs, qpl.QArray)
  has_explicit_quant = (
      qwix_rule is not None
      or lhs_quantize_dtype is not None
      or rhs_quantize_dtype is not None
      or use_manual_quantization
      or has_prequantized
  )
  has_custom_vjp_features = (
      bool(weight_gather_axes)
      or existing_out is not None
      or partial_sum is not None
      or return_rhs
      or return_lhs
      or flat_fsdp_shard is not None
      or out_is_3d
      or tc_unsort_cfg is not None
      or bool(lhs_vma_axes)
      or bool(rhs_vma_axes)
  )

  if not backend.requires_custom_vjp and not has_custom_vjp_features:
    if backend.supports_native_qwix_interception:
      if not has_explicit_quant:
        use_full_k_tiling = bool(
            use_qwix_quantization
            and (is_quantized_config if is_quantized_config is not None else resolved_rule is not None)
        )
        return backend.call(
            lhs,
            rhs,
            group_sizes,
            preferred_element_type=preferred_element_type,
            tiling=tiling,
            group_offset=group_offset,
            use_full_k_tiling=use_full_k_tiling,
        )
    elif resolved_rule is None and not has_explicit_quant:
      return backend.call(
          lhs,
          rhs,
          group_sizes,
          preferred_element_type=preferred_element_type,
          tiling=tiling,
          group_offset=group_offset,
          use_rpa_group_sizes=use_rpa_group_sizes,
      )

  return gmm_quantize.quantized_gmm(
      lhs,
      rhs,
      group_sizes,
      backend=backend,
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
