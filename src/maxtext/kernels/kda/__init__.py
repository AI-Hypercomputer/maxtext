# Copyright 2026 Ant Group. All Rights Reserved.
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

"""KDA (Kimi Delta Attention) kernels.

Entry point that delegates to tokamax ``kimi_delta_attention`` with
native ``[B, T]`` segment_ids (head-first layout internally).

Supports CP (context parallelism) via ``context_parallel_metadata``.
"""

from __future__ import annotations

import jax.numpy as jnp
from maxtext.kernels.kda.tokamax import tokamax_chunk_kda


def chunk_kda(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    g: jnp.ndarray,
    beta: jnp.ndarray,
    scale: float | None = None,
    initial_state: jnp.ndarray | None = None,
    output_final_state: bool = False,
    a_log: jnp.ndarray | None = None,
    delta_time_bias: jnp.ndarray | None = None,
    use_gate_in_kernel: bool = False,
    use_qk_l2norm: bool = False,
    lower_bound: float | None = None,
    segment_ids: jnp.ndarray | None = None,
    max_num_segments: int | None = None,
    context_parallel_metadata: object | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
  """KDA entry point via tokamax backend.

  Tokamax natively accepts ``[B, T]`` segment_ids so no B*T flatten
  or per-batch offset computation is needed.

  Args:
      q: [B, T, H, K] queries.
      k: [B, T, H, K] keys.
      v: [B, T, H, V] values.
      g: [B, T, H, K] gate values.
      beta: [B, T, H] delta-rule mixing coefficient.
      scale: attention scale (default ``K ** -0.5``).
      initial_state: must be None (not yet supported).
      output_final_state: must be False (not yet supported).
      a_log: [H] per-head log decay-rate parameter. Required when
          ``use_gate_in_kernel=True``.
      delta_time_bias: [H*K] optional per-head, per-key-channel gate bias.
      use_gate_in_kernel: whether ``g`` is activated (``a_log`` /
          ``delta_time_bias``) inside the kernel.
      use_qk_l2norm: whether to L2-normalize q/k in-kernel.
      lower_bound: optional sigmoid-gate lower bound in ``[-5, 0)``. When
          None, the standard ``softplus`` gate path is used.
      segment_ids: [B, T] 1-based segment IDs for varlen mode (0=padding).
      max_num_segments: static upper bound on varlen segments. Required when
          ``segment_ids`` is provided without ``initial_state``.
      context_parallel_metadata: optional ``ContextParallelMetadata`` for CP.
          When set, the kernel derives cross-rank metadata from
          ``segment_ids`` and coordinates recurrent state across CP ranks.

  Returns:
      (o, final_state) where o is [B, T, H, V] and final_state is None.
  """
  if initial_state is not None:
    raise NotImplementedError("initial_state is not supported")
  if output_final_state:
    raise NotImplementedError("output_final_state is not supported")

  return tokamax_chunk_kda(
      q=q,
      k=k,
      v=v,
      g=g,
      beta=beta,
      scale=scale,
      a_log=a_log,
      delta_time_bias=delta_time_bias,
      use_gate_in_kernel=use_gate_in_kernel,
      use_qk_l2norm=use_qk_l2norm,
      lower_bound=lower_bound,
      segment_ids=segment_ids,
      max_num_segments=max_num_segments,
      context_parallel_metadata=context_parallel_metadata,
  )
