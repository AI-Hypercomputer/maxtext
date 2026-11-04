# Copyright 2026 Google LLC
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

"""PyTorch clean-room reference modules and E8 parity helpers for Muse Spark."""

import collections.abc
import contextlib
from typing import Any
from unittest import mock

import jax
import jax.numpy as jnp
from maxtext.configs import pyconfig
import numpy as np
import torch


class _RefNorm(torch.nn.Module):
  """Clean-room PyTorch normalization base."""

  zero_init = False

  def __init__(self, dim: int, eps: float = 1e-5, with_scale: bool = True) -> None:
    super().__init__()
    self.eps = eps
    self.weight = torch.nn.Parameter((torch.zeros if self.zero_init else torch.ones)(dim)) if with_scale else None


class ReferenceMuseSparkRMSNorm(_RefNorm):
  """Clean-room PyTorch reference for MuseSparkRMSNorm."""

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    c = torch.promote_types(x.dtype, torch.float32)
    y = x.to(c) * torch.pow(x.to(c).pow(2).mean(-1, keepdim=True) + self.eps, -0.5)
    return (y * self.weight.to(c) if self.weight is not None else y).to(x.dtype)


class ReferenceCenteredRMSNorm(_RefNorm):
  """Clean-room PyTorch reference for CenteredRMSNorm."""

  zero_init = True

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    c = torch.promote_types(x.dtype, torch.float32)
    y = x.to(c) * torch.rsqrt(x.to(c).pow(2).mean(-1, keepdim=True) + self.eps)
    return (y * (1.0 + self.weight).to(c) if self.weight is not None else y).to(x.dtype)


class ReferenceInputScaledRMSNorm(_RefNorm):
  """Clean-room PyTorch reference for InputScaledRMSNorm."""

  zero_init = True

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    c = torch.promote_types(x.dtype, torch.float32)
    s = (x.to(c) * (1.0 + self.weight).to(c)).to(x.dtype) if self.weight is not None else x
    return (s.to(c) * torch.rsqrt(s.to(c).pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype)


class ReferenceMuseSparkResidualGate(torch.nn.Module):
  """Clean-room PyTorch reference for MuseSparkResidualGate."""

  def __init__(self, dim: int, temperature: float = 0.3, eps: float = 1e-3) -> None:
    super().__init__()
    self.gate, self.temperature, self.eps = (torch.nn.Parameter(torch.zeros(dim)), temperature, eps)

  def forward(self, residual: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
    c = torch.promote_types(residual.dtype, torch.float32)
    w = self.gate.to(c) / self.temperature
    beta = torch.sigmoid(w)
    alpha = torch.sqrt(torch.clamp(torch.sigmoid(-w) * (1.0 + beta), min=self.eps))
    return (alpha * residual.to(c) + beta * hidden.to(c)).to(residual.dtype)


class ReferenceMuseSparkTextMLP(torch.nn.Module):
  """Clean-room PyTorch reference for MuseSparkTextMLP."""

  def __init__(self, hidden_size: int, intermediate_size: int) -> None:
    super().__init__()
    self.gate_proj, self.up_proj = (torch.nn.Linear(hidden_size, intermediate_size, bias=False) for _ in (0, 1))
    self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=False)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class ReferenceMuseSparkDecoder(torch.nn.Module):
  """Clean-room PyTorch reference for CL 1 unscanned Muse Spark decoder."""

  def __init__(self) -> None:
    super().__init__()
    v_s, d, md, hd, eps, p_eps, temp = 256, 64, 128, 16, 1e-5, 1e-8, 0.5
    self.num_query_heads, self.num_kv_heads, self.output_multiplier, self.soft_cap = (4, 2, 0.1767766952966369, 20.0)
    self.shared_emb = torch.nn.Parameter(torch.zeros(v_s, d))
    self.embed_norm = ReferenceMuseSparkRMSNorm(d, eps=eps, with_scale=False)

    def _layer() -> torch.nn.Module:
      qh, kh = self.num_query_heads, self.num_kv_heads
      m = torch.nn.Module()
      m.input_norm, m.pre_ffn_norm = (ReferenceCenteredRMSNorm(d, eps=eps) for _ in (0, 1))
      m.post_attn_norm = ReferenceMuseSparkRMSNorm(d, eps=p_eps, with_scale=False)
      m.post_ffn_norm = ReferenceInputScaledRMSNorm(d, eps=p_eps)
      m.q_proj, m.out_proj = torch.nn.Parameter(torch.zeros(d, qh, hd)), torch.nn.Parameter(torch.zeros(qh, hd, d))
      m.k_proj, m.v_proj = (torch.nn.Parameter(torch.zeros(d, kh, hd)) for _ in (0, 1))
      m.post_attn_gate, m.post_ffn_gate = (ReferenceMuseSparkResidualGate(d, temperature=temp, eps=1e-3) for _ in (0, 1))
      m.wi_0, m.wi_1 = [torch.nn.Parameter(torch.zeros(d, md)) for _ in (0, 1)]
      m.wo = torch.nn.Parameter(torch.zeros(md, d))
      return m

    self.layers = torch.nn.ModuleList([_layer(), _layer()])
    self.decoder_norm = ReferenceMuseSparkRMSNorm(d, eps=eps, with_scale=True)
    self.logits_dense_w = torch.nn.Parameter(torch.zeros(d, v_s))

  def embed_tokens(self, toks: torch.Tensor) -> torch.Tensor:
    c = torch.where((toks == 101) | (toks == 102) | (toks == 103), 0, toks)
    return self.embed_norm(torch.nn.functional.embedding(c, self.shared_emb))

  def forward(self, toks: torch.Tensor) -> torch.Tensor:
    """Forward pass of clean-room reference Muse Spark decoder stack."""
    c = torch.promote_types(self.shared_emb.dtype, torch.float32)
    x = self.embed_tokens(toks)
    rep, dev, s = (self.num_query_heads // self.num_kv_heads, toks.device, x.shape[1])
    causal_mask = torch.triu(torch.full((s, s), -1e9, dtype=c, device=dev), diagonal=1)
    frac = 2 * torch.arange(0, 8, device=dev, dtype=c) / 16
    ang = torch.arange(s, device=dev, dtype=c)[None, None, :, None] / (10000.0**frac)
    cos, sin = torch.cos(ang).repeat(1, 1, 1, 2), torch.sin(ang).repeat(1, 1, 1, 2)

    def _rope(t: torch.Tensor) -> torch.Tensor:
      t1, t2 = torch.chunk(t, 2, dim=-1)
      return t * cos + torch.cat((-t2, t1), dim=-1) * sin

    for idx, lyr in enumerate(self.layers):
      res, hn = x, lyr.input_norm(x).to(c)
      q = torch.einsum("bsd,dqh->bqsh", hn, lyr.q_proj.to(c))
      k, v = (torch.einsum("bsd,dkh->bksh", hn, p.to(c)) for p in (lyr.k_proj, lyr.v_proj))
      if (len(self.layers) - 1 - idx) % 4 != 0:
        q, k = _rope(q), _rope(k)
      k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
      attn = torch.matmul(torch.softmax(torch.matmul(q, k.transpose(-2, -1)) + causal_mask, dim=-1), v)
      x = lyr.post_attn_gate(res, lyr.post_attn_norm(torch.einsum("bqsh,qhd->bsd", attn, lyr.out_proj.to(c)).to(x.dtype)))
      fres, fn = x, lyr.pre_ffn_norm(x).to(c)
      ffn_act = torch.nn.functional.silu(fn @ lyr.wi_0.to(c)) * (fn @ lyr.wi_1.to(c))
      x = lyr.post_ffn_gate(fres, lyr.post_ffn_norm((ffn_act @ lyr.wo.to(c)).to(x.dtype)))

    logits = (self.decoder_norm(x).to(c) @ self.logits_dense_w.to(c)).to(x.dtype) * self.output_multiplier
    return (torch.tanh(logits / self.soft_cap) * self.soft_cap if self.soft_cap else logits).to(c)


def get_layer_param_pairs(j_lyr: Any, r_lyr: Any, is_dense: bool) -> list[tuple[Any, Any, str]]:
  """Returns (jax_param, ref_param, name) tuples for every parameter in a decoder layer."""
  f, a = (j_lyr.mlp if is_dense else j_lyr.moe_block), j_lyr.self_attention
  p = [(j_lyr.input_layernorm.scale, r_lyr.input_norm.weight, "input_norm")]
  p += [(a.query.kernel, r_lyr.q_proj, "q_proj"), (a.key.kernel, r_lyr.k_proj, "k_proj")]
  p += [(a.value.kernel, r_lyr.v_proj, "v_proj"), (a.out.kernel, r_lyr.out_proj, "out_proj")]
  p += [(j_lyr.post_attn_gate.gate, r_lyr.post_attn_gate.gate, "post_attn_gate")]
  p += [(j_lyr.pre_ffn_norm.scale, r_lyr.pre_ffn_norm.weight, "pre_ffn_norm")]
  p += [(f.wi_0.kernel, r_lyr.wi_0, "wi0"), (f.wi_1.kernel, r_lyr.wi_1, "wi1"), (f.wo.kernel, r_lyr.wo, "wo")]
  p += [(j_lyr.post_ffn_norm.scale, r_lyr.post_ffn_norm.weight, "post_ffn_norm")]
  p += [(j_lyr.post_ffn_gate.gate, r_lyr.post_ffn_gate.gate, "post_ffn_gate")]
  return p


@contextlib.contextmanager
def fp64_mode() -> collections.abc.Iterator[None]:
  """Test-scoped fp64 execution without adding a production YAML dtype knob."""
  prev = jax.config.read("jax_enable_x64")
  jax.config.update("jax_enable_x64", True)
  orig = pyconfig.HyperParameters.__init__

  def patched(self: Any, pydantic_config: Any) -> None:
    orig(self, pydantic_config)
    d64 = {k: jnp.dtype("float64") for k in ("dtype", "weight_dtype", "grad_dtype", "mu_dtype", "dtype_mm")}
    object.__getattribute__(self, "_flat_config").update(d64)

  with mock.patch.object(pyconfig.HyperParameters, "__init__", patched):
    try:
      yield
    finally:
      jax.config.update("jax_enable_x64", prev)


def _ensure(cond: bool, msg: str) -> None:
  if not cond:
    raise AssertionError(msg)


def check_e8_parity(jax_val: Any, ref_val: Any, name: str, is_fp64: bool = False) -> None:
  """Checks elementwise and relative Frobenius parity according to E8 protocol."""
  j_arr, r_arr = np.asarray(jax_val, dtype=np.float64), np.asarray(ref_val, dtype=np.float64)
  _ensure(j_arr.shape == r_arr.shape, f"Shape mismatch on {name}: {j_arr.shape} != {r_arr.shape}")
  _ensure(bool(np.all(np.isfinite(j_arr)) and np.all(np.isfinite(r_arr))), f"Non-finite values found in {name}")
  ref_norm = float(np.linalg.norm(r_arr))
  _ensure(ref_norm >= 1e-7, f"Zero ref norm ({ref_norm:.4e} < 1e-7) on {name}")
  if not is_fp64:
    atol = 1e-5 * max(1.0, float(np.max(np.abs(r_arr))))
    np.testing.assert_allclose(j_arr, r_arr, rtol=1e-5, atol=atol, err_msg=f"E8 fp32 {name}")
  lim, denom = (1e-10, ref_norm) if is_fp64 else (1e-5, max(1.0, ref_norm))
  _ensure((rel := float(np.linalg.norm(j_arr - r_arr)) / denom) <= lim, f"E8 Frobenius on {name}: {rel=:.4e} > {lim}")
