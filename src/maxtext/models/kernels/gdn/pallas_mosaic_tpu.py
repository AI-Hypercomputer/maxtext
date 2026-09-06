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
# ==============================================================================
"""Pallas Mosaic TPU kernel implementation for Causal Conv1D Gated Delta Rule."""

# pylint: disable=missing-function-docstring,unused-import

import dataclasses
from typing import Any, Optional, override

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops import op
from tokamax._src.ops.causal_conv1d_gated_delta_rule import base
from tokamax._src.ops.causal_conv1d_gated_delta_rule import config
from tokamax._src.ops.causal_conv1d_gated_delta_rule import wrapper

GDNConfig = config.GDNConfig


@dataclasses.dataclass(frozen=True, kw_only=True)
class PallasMosaicTpuCausalConv1dGatedDeltaRule(base.CausalConv1dGatedDeltaRule[GDNConfig]):
  """Wrapper for the tokamax Op API for Pallas Mosaic TPU kernel."""

  # pylint: disable=redefined-outer-name
  def _fwd(
      self,
      qkv: jax.Array,
      b: jax.Array,
      a: jax.Array,
      conv_state: jax.Array,
      recurrent_state: jax.Array,
      conv_weight: jax.Array,
      conv_bias: Optional[jax.Array],
      a_log: jax.Array,
      dt_bias: jax.Array,
      query_start_loc: jax.Array,
      state_indices: jax.Array,
      distribution: jax.Array,
      seq_lens: jax.Array,
      *,
      n_kq: int,
      n_v: int,
      d_k: int,
      d_v: int,
      kernel_size: int,
      zero_initialize_out: bool = True,
      compute_precision: jnp.dtype = jnp.float32.dtype,
      decode_tile_size: int = 4,
      mixed_tile_size: int = 64,
      # TODO: Calculate tile size based on input dimensions.
      config: GDNConfig | None = None,
      return_residuals: bool = False,
  ) -> tuple[tuple[tuple[jax.Array, jax.Array], jax.Array], None]:
    del return_residuals, config
    return (
        wrapper.fused_conv1d_gdn(
            qkv=qkv,
            b=b,
            a=a,
            conv_state=conv_state,
            recurrent_state=recurrent_state,
            conv_weight=conv_weight,
            conv_bias=conv_bias,
            a_log=a_log,
            dt_bias=dt_bias,
            query_start_loc=query_start_loc,
            state_indices=state_indices,
            distribution=distribution,
            seq_lens=seq_lens,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=kernel_size,
            zero_initialize_out=zero_initialize_out,
            compute_precision=compute_precision,
            decode_tile_size=decode_tile_size,
            mixed_tile_size=mixed_tile_size,
        ),
        None,
    )

  @override
  def supported_on(self, device: jax.Device) -> bool:
    try:
      return device.platform == "tpu" and pltpu.get_tpu_info().generation >= 6
    except Exception:  # pylint: disable=broad-exception-caught
      return False
