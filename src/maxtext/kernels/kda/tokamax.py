"""Tokamax KDA backend for maxtext.

Wraps ``tokamax.kimi_delta_attention`` with a maxtext-compatible interface
(batch-first ``[B, T, H, K]`` layout).  Tokamax natively supports
``[B, T]`` segment_ids, so no
B*T flatten / per-batch offset is needed.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _to_tokamax(q, k, v, g, beta):
  """[B, T, H, K] -> [H, B, T, K];  [B, T, H] -> [H, B, T]."""
  return (
      jnp.transpose(q, (2, 0, 1, 3)),
      jnp.transpose(k, (2, 0, 1, 3)),
      jnp.transpose(v, (2, 0, 1, 3)),
      jnp.transpose(g, (2, 0, 1, 3)),
      jnp.transpose(beta, (2, 0, 1)),
  )


def _to_maxtext(o_h):
  """[H, B, T, V] -> [B, T, H, V]."""
  return jnp.transpose(o_h, (1, 2, 0, 3))


def tokamax_chunk_kda(
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
    cp_context: object | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
  """KDA via tokamax, batch-first interface matching ``kernels.kda.chunk_kda``.

  Tokamax accepts ``[B, T]`` segment_ids natively so no B*T flatten
  or per-batch offset computation is needed.

  Args:
      q: [B, T, H, K]
      k: [B, T, H, K]
      v: [B, T, H, V]
      g: [B, T, H, K]
      beta: [B, T, H]
      segment_ids: [B, T] 1-based, 0=padding.  Passed directly to tokamax.
      cp_context: Optional ``tokamax...CPContext`` for context parallelism.
      (other args match the backend interface)

  Returns:
      (o, None) where o is [B, T, H, V].
  """
  from tokamax._src.ops.experimental.kda.api import kimi_delta_attention

  assert initial_state is None, "initial_state not supported with tokamax backend"
  assert not output_final_state, "output_final_state not supported with tokamax backend"

  q_h, k_h, v_h, g_h, beta_h = _to_tokamax(q, k, v, g, beta)

  o_h, _final_state = kimi_delta_attention(
      q=q_h,
      k=k_h,
      v=v_h,
      g=g_h,
      beta=beta_h,
      A_log=A_log,
      dt_bias=dt_bias,
      scale=scale,
      segment_ids=segment_ids,
      chunk_size=chunk_size,
      use_gate_in_kernel=use_gate_in_kernel,
      use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
      safe_gate=safe_gate,
      lower_bound=lower_bound,
      disable_recompute=disable_recompute,
      N_max=N_max,
      implementation="pallas_tpu",
      cp_context=cp_context,
  )

  o = _to_maxtext(o_h)
  return o, None
