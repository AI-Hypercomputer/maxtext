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

"""CPU PyTorch reference for Kimi Delta Attention (KDA), used as a test oracle.

The Kimi-K3 reference modeling code runs KDA through the Triton kernels of
flash-linear-attention (`fla.ops.kda.chunk_kda` / `fused_recurrent_kda`), which do not run
on CPU. This module re-implements, for the flags the reference passes, the math those
kernels compute, so the HF model can be evaluated on CPU and compared with MaxText.

Specification: fla commit 9f38d24980c46d46bd38614e743cdacd21906578, specifically
`fla/ops/kda/gate.py` (`naive_kda_gate`, `naive_kda_lowerbound_gate` and the kernel's
`USE_LOWER_BOUND` branch), `fla/ops/kda/naive.py` (`naive_recurrent_kda`) and
`fla/modules/l2norm.py`. The implementation here is independent; it was checked against
those pure-torch fla functions at that commit (max abs diff <= 1e-6 in fp32 over random
inputs, with and without a lower bound and an initial state).

Decay gate (log space, per channel), computed in fp32:
    with lower_bound:    g = lower_bound * sigmoid(exp(A_log) * (g_raw + dt_bias))   in [lb, 0)
    without:             g = -exp(A_log) * softplus(g_raw + dt_bias)
With a lower bound fla does NOT clamp the softplus gate; it switches activation.

Recurrence (fp32 state S of shape [K, V] per batch/head; q pre-scaled by `scale`):
    S_t = exp(g_t)[:, None] * S_{t-1}
    S_t = S_t + beta_t * k_t[:, None] * (v_t - k_t @ S_t)[None, :]
    o_t = q_t @ S_t
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

SPEC_FLA_COMMIT = "9f38d24980c46d46bd38614e743cdacd21906578"


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
  """fla-style L2 normalization over the last dim in fp32: x / sqrt(sum(x^2) + eps)."""
  x = x.float()
  return x * torch.rsqrt(x.pow(2).sum(dim=-1, keepdim=True) + eps)


def kda_gate(
    g: torch.Tensor,
    A_log: Optional[torch.Tensor],  # pylint: disable=invalid-name
    dt_bias: Optional[torch.Tensor],
    lower_bound: Optional[float],
) -> torch.Tensor:
  """In-kernel KDA decay gate. `g` is `[..., H, K]`, A_log `[H]`, dt_bias `[H*K]`; returns fp32."""
  num_heads, head_dim = g.shape[-2:]
  x = g.float()
  if dt_bias is not None:
    x = x + dt_bias.float().reshape(num_heads, head_dim)
  if lower_bound is not None:
    if A_log is not None:
      x = x * A_log.float().exp().reshape(num_heads, 1)
    return lower_bound * torch.sigmoid(x)
  rate = A_log.float().exp().reshape(num_heads, 1) if A_log is not None else 1.0
  return -rate * F.softplus(x)  # pylint: disable=not-callable


def recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
  """Token-by-token KDA recurrence in fp32.

  q, k: `[B, T, H, K]`; v: `[B, T, H, V]`; g (log decay): `[B, T, H, K]`; beta: `[B, T, H]`.
  State layout `[B, H, K, V]`. Returns `o` in v's dtype and the fp32 final state (or None).
  """
  out_dtype = v.dtype
  head_dim = q.shape[-1]
  scale = head_dim**-0.5 if scale is None else scale
  q, k, v, g, beta = (t.float() for t in (q, k, v, g, beta))
  q = q * scale
  bsz, seq_len, num_heads, _ = q.shape
  state = torch.zeros(bsz, num_heads, head_dim, v.shape[-1], dtype=torch.float32)
  if initial_state is not None:
    state = state + initial_state.float()
  outs = []
  for t in range(seq_len):
    state = state * g[:, t].exp().unsqueeze(-1)
    k_t = k[:, t]  # [B, H, K]
    pred = torch.einsum("bhk,bhkv->bhv", k_t, state)
    delta = beta[:, t].unsqueeze(-1) * (v[:, t] - pred)  # [B, H, V]
    state = state + torch.einsum("bhk,bhv->bhkv", k_t, delta)
    outs.append(torch.einsum("bhk,bhkv->bhv", q[:, t], state))
  o = torch.stack(outs, dim=1).to(out_dtype)
  return o, (state if output_final_state else None)


def chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: Optional[torch.Tensor] = None,  # pylint: disable=invalid-name
    dt_bias: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    safe_gate: bool = False,
    lower_bound: Optional[float] = None,
    transpose_state_layout: bool = False,
    cu_seqlens: Any = None,
    **unused_kwargs,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
  """CPU stand-in for `fla.ops.kda.chunk_kda` / `fused_recurrent_kda` (forward only).

  Same keyword interface and defaults as the fla entry points. The state is `[B, H, K, V]`,
  or `[B, H, V, K]` with `transpose_state_layout=True` (as in fla). `safe_gate` only selects
  a kernel tiling in fla (the activation is chosen by `lower_bound` alone), so it is ignored.
  """
  del safe_gate, unused_kwargs
  if cu_seqlens is not None:
    raise NotImplementedError("variable-length (cu_seqlens) KDA is not supported by this oracle")
  if use_qk_l2norm_in_kernel:
    q, k = l2norm(q), l2norm(k)
  if use_gate_in_kernel:
    g = kda_gate(g, A_log, dt_bias, lower_bound)
  if use_beta_sigmoid_in_kernel:
    beta = torch.sigmoid(beta.float())
  if initial_state is not None and transpose_state_layout:
    initial_state = initial_state.transpose(-1, -2)
  o, state = recurrent_kda(
      q, k, v, g, beta, scale=scale, initial_state=initial_state, output_final_state=output_final_state
  )
  if state is not None and transpose_state_layout:
    state = state.transpose(-1, -2)
  return o, state
