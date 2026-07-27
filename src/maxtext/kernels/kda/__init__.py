"""KDA (Kimi Delta Attention) kernels.

Entry point that delegates to tokamax ``kimi_delta_attention`` with
native ``[B, T]`` segment_ids (head-first layout internally).

Supports AG-CP (all-gather context parallelism) via ``cp_context``.
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
    chunk_size: int = 64,
    A_log: jnp.ndarray | None = None,
    dt_bias: jnp.ndarray | None = None,
    use_gate_in_kernel: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    segment_ids: jnp.ndarray | None = None,
    disable_recompute: bool = False,
    N_max: int | None = None,
    kda_backend: str = "tokamax",
    cp_context: object | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
  """KDA entry point via tokamax backend.

  Tokamax natively accepts ``[B, T]`` segment_ids so no B*T flatten
  or per-batch offset computation is needed.

  Args:
      q: [B, T, H, K] queries.
      k: [B, T, H, K] keys.
      v: [B, T, H, V] values.
      g: [B, T, H, K] gate values.
      beta: [B, T, H] delta rule mixing coefficient.
      scale: attention scale (default 1/sqrt(K)).
      initial_state: must be None.
      output_final_state: must be False.
      chunk_size: chunk size (64).
      A_log: [H] learnable decay in log space.
      dt_bias: [H*K] dt bias.
      use_gate_in_kernel: apply gate inside kernel.
      use_qk_l2norm_in_kernel: apply L2 norm to q/k in kernel.
      safe_gate: numerically safe gate mode.
      lower_bound: gate value lower bound.
      segment_ids: [B, T] segment IDs for varlen mode (2D, 1-based, 0=padding).
      N_max: max segments per sample.
      kda_backend: ``"tokamax"``.
      cp_context: Optional ``CPContext`` for AG-CP.  When set, the
          kernel derives cross-rank metadata from ``segment_ids``
          and coordinates recurrent state across CP ranks.

  Returns:
      (o, final_state) where o is [B, T, H, V] and final_state is None.
  """
  assert initial_state is None, "initial_state not supported"
  assert not output_final_state, "output_final_state not supported"

  return tokamax_chunk_kda(
      q=q, k=k, v=v, g=g, beta=beta,
      scale=scale,
      chunk_size=chunk_size,
      A_log=A_log,
      dt_bias=dt_bias,
      use_gate_in_kernel=use_gate_in_kernel,
      use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
      safe_gate=safe_gate,
      lower_bound=lower_bound,
      segment_ids=segment_ids,
      disable_recompute=disable_recompute,
      N_max=N_max,
      cp_context=cp_context,
  )