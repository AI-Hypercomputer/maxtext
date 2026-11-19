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

"""Parity tests: MaxText Kimi Delta Attention (KDA) vs a PyTorch fla-spec reference.

The oracle is `tests/utils/kimi_k3_kda_reference.py` (an independent CPU implementation of
fla's `chunk_kda` semantics). HF `modeling_kimi_linear.py` itself is not imported: its KDA path
calls fla's Triton kernels, which do not run on CPU. Skipped when torch is not installed; the
torch-free float64 NumPy oracle and mapping round-trip tests in `kimi_k3_kda_test.py` run in CI.
"""

import math
from typing import Any, Optional
import unittest

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

torch = pytest.importorskip("torch")

# pylint: disable=wrong-import-position
from torch import nn
import torch.nn.functional as F

from maxtext.common.common_types import MODEL_MODE_AUTOREGRESSIVE, MODEL_MODE_TRAIN
from maxtext.layers.attention_kda import (
    KimiDeltaAttention,
    ShortConvolution,
    jax_chunk_kimi_delta_rule,
    kda_gate,
)
from maxtext.layers.normalizations import l2norm
from tests.utils import kimi_k3_kda_reference
from tests.utils.kimi_k3_conversion_utils import convert_pytorch_module_to_maxtext_params, load_params_into_nnx

# pylint: enable=wrong-import-position


# ==============================================================================
# PyTorch Reference Components for KDA
# ==============================================================================
class ShortConvolution_PT(nn.Module):
  """PyTorch Reference for Causal 1D Depthwise Convolution."""

  def __init__(self, hidden_size: int, kernel_size: int = 4, activation: str = "silu"):
    super().__init__()
    self.hidden_size = hidden_size
    self.kernel_size = kernel_size
    self.activation = activation
    self.weight = nn.Parameter(torch.empty(hidden_size, 1, kernel_size))
    nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

  def forward(self, x, cache=None, output_final_state=False):
    """Applies 1D short depthwise convolution along sequence length (fp32 accumulate, as in fla)."""
    _, t, _ = x.shape
    x_t = x.transpose(1, 2).float()  # [B, C, T]
    if cache is not None:
      x_t = torch.cat([cache.float(), x_t], dim=2)
    else:
      x_t = F.pad(x_t, (self.kernel_size - 1, 0))
    out = F.conv1d(x_t, self.weight.float(), groups=self.hidden_size)  # pylint: disable=not-callable
    out = out[..., :t].transpose(1, 2)
    if self.activation == "silu":
      out = F.silu(out)
    final_state = x_t[..., -(self.kernel_size - 1) :].to(x.dtype) if output_final_state else None
    return out.to(x.dtype), final_state


class FusedRMSNormGated_PT(nn.Module):
  """PyTorch Reference for fla's FusedRMSNormGated (sigmoid): fp32 math, one cast to the input dtype."""

  def __init__(self, hidden_size: int, eps: float = 1e-5):
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(hidden_size))

  def forward(self, x, g):
    xf = x.float()
    variance = xf.pow(2).mean(-1, keepdim=True)
    return (xf * torch.rsqrt(variance + self.eps) * self.weight.float() * torch.sigmoid(g.float())).to(x.dtype)


def chunk_kda_PT(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = True,
    use_gate_in_kernel: bool = True,
    use_beta_sigmoid_in_kernel: bool = True,
    lower_bound: Optional[float] = -5.0,
) -> tuple[Any, Any]:
  """fla-spec CPU oracle; the state is kept in MaxText's `[B, H, K, V]` layout."""
  return kimi_k3_kda_reference.chunk_kda(
      q,
      k,
      v,
      g,
      beta,
      A_log=A_log,
      dt_bias=dt_bias,
      initial_state=initial_state,
      output_final_state=output_final_state,
      use_qk_l2norm_in_kernel=True,
      use_gate_in_kernel=use_gate_in_kernel,
      use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
      lower_bound=lower_bound,
      transpose_state_layout=False,
  )


class KimiDeltaAttention_PT(nn.Module):
  """PyTorch Reference for the KimiDeltaAttention module (HF `KimiDeltaAttention` layout)."""

  def __init__(self, hidden_size, num_heads, head_dim, conv_size=4, rms_norm_eps=1e-5, gate_lower_bound=-5.0):
    super().__init__()
    self.num_heads = num_heads
    self.head_dim = head_dim
    self.gate_lower_bound = gate_lower_bound
    projection_size = head_dim * num_heads

    self.q_proj = nn.Linear(hidden_size, projection_size, bias=False)
    self.k_proj = nn.Linear(hidden_size, projection_size, bias=False)
    self.v_proj = nn.Linear(hidden_size, projection_size, bias=False)

    self.q_conv1d = ShortConvolution_PT(hidden_size=projection_size, kernel_size=conv_size)
    self.k_conv1d = ShortConvolution_PT(hidden_size=projection_size, kernel_size=conv_size)
    self.v_conv1d = ShortConvolution_PT(hidden_size=projection_size, kernel_size=conv_size)

    # Released layout: the `num_heads` log-decays zero-padded to `head_dim` entries.
    a_log = torch.log(torch.empty(num_heads, dtype=torch.float32).uniform_(1, 16))
    self.A_log = nn.Parameter(F.pad(a_log, (0, max(head_dim - num_heads, 0))))
    self.f_a_proj = nn.Linear(hidden_size, head_dim, bias=False)
    self.f_b_proj = nn.Linear(head_dim, projection_size, bias=False)
    self.dt_bias = nn.Parameter(torch.empty(projection_size, dtype=torch.float32).uniform_(-1, 1))
    self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)
    self.g_proj = nn.Linear(hidden_size, projection_size, bias=False)
    self.o_norm = FusedRMSNormGated_PT(head_dim, eps=rms_norm_eps)
    self.o_proj = nn.Linear(projection_size, hidden_size, bias=False)

  def forward(self, hidden_states, conv_state=None, recurrent_state=None, output_final_state=False):
    """Forward pass of PyTorch reference KimiDeltaAttention_PT."""
    b, t, _ = hidden_states.shape
    conv_q, conv_k, conv_v = conv_state if conv_state is not None else (None, None, None)

    q, next_conv_q = self.q_conv1d(self.q_proj(hidden_states), cache=conv_q, output_final_state=output_final_state)
    k, next_conv_k = self.k_conv1d(self.k_proj(hidden_states), cache=conv_k, output_final_state=output_final_state)
    v, next_conv_v = self.v_conv1d(self.v_proj(hidden_states), cache=conv_v, output_final_state=output_final_state)
    next_conv_state = (next_conv_q, next_conv_k, next_conv_v) if output_final_state else None

    heads = (b, t, self.num_heads, self.head_dim)
    g = self.f_b_proj(self.f_a_proj(hidden_states)).reshape(heads)
    beta = self.b_proj(hidden_states).float()

    o, next_rec_state = chunk_kda_PT(
        q=q.reshape(heads),
        k=k.reshape(heads),
        v=v.reshape(heads),
        g=g,
        beta=beta,
        A_log=self.A_log[: self.num_heads],
        dt_bias=self.dt_bias,
        initial_state=recurrent_state,
        output_final_state=output_final_state,
        lower_bound=self.gate_lower_bound,
    )

    o = self.o_norm(o, self.g_proj(hidden_states).reshape(heads))
    out = self.o_proj(o.reshape(b, t, -1))
    return out, next_conv_state, next_rec_state


# ==============================================================================
# Checkpoint conversion (exact `self_attn.` key match through the kimi-k3 param mapping)
# ==============================================================================
_PREFIX = "params-decoder-layers_0-self_attention"


def load_reference_into_kda(pt_module: nn.Module, kda_module: KimiDeltaAttention, hf_config: dict[str, Any]):
  """Converts `pt_module` through PARAM_MAPPING/HOOK_FNS (with MaxText target shapes) and loads it."""
  target_shapes = {f"{_PREFIX}-A_log": tuple(kda_module.A_log.get_value().shape)}
  converted = convert_pytorch_module_to_maxtext_params(
      pt_module.state_dict(), hf_config, prefix_filter=_PREFIX, target_shapes=target_shapes
  )
  load_params_into_nnx(kda_module, converted, _PREFIX, strict=True)


class KimiK3KDAVsReferenceTest(unittest.TestCase):
  """Kimi-K3 KDA layer and core vs the PyTorch fla-spec reference."""

  def setUp(self):
    super().setUp()
    self.enterContext(jax.default_matmul_precision("highest"))
    self.batch_size = 2
    self.seq_len = 16
    self.hidden_size = 256
    self.num_heads = 4
    self.head_dim = 64
    self.conv_size = 4
    self.rms_norm_eps = 1e-5
    self.gate_lower_bound = -5.0
    self.hf_config = {
        "hidden_size": self.hidden_size,
        "num_hidden_layers": 4,
        "num_attention_heads": self.num_heads,
        "num_key_value_heads": self.num_heads,
        "full_attn_layers": [3],
        "rms_norm_eps": self.rms_norm_eps,
        "first_k_dense_replace": 4,  # all-dense toy: only the KDA keys are exercised
    }

  def _make_pair(self, chunk_size: int = 8):
    """Builds a torch reference layer and a MaxText layer loaded with the same (converted) weights."""
    torch.manual_seed(42)
    pt_module = KimiDeltaAttention_PT(
        hidden_size=self.hidden_size,
        num_heads=self.num_heads,
        head_dim=self.head_dim,
        conv_size=self.conv_size,
        rms_norm_eps=self.rms_norm_eps,
        gate_lower_bound=self.gate_lower_bound,
    )
    jax_module = KimiDeltaAttention(
        hidden_size=self.hidden_size,
        num_heads=self.num_heads,
        head_dim=self.head_dim,
        conv_size=self.conv_size,
        chunk_size=chunk_size,
        rms_norm_eps=self.rms_norm_eps,
        gate_lower_bound=self.gate_lower_bound,
        rngs=nnx.Rngs(params=0),
    )
    load_reference_into_kda(pt_module, jax_module, self.hf_config)
    return pt_module, jax_module

  def test_kda_gate_matches_reference(self):
    """MaxText `kda_gate` equals the fla-spec torch gate, with and without a lower bound."""
    rng = np.random.default_rng(0)
    h, d = self.num_heads, self.head_dim
    g_raw = rng.normal(scale=3.0, size=(2, 5, h, d)).astype(np.float32)
    a_log = np.log(rng.uniform(1.0, 16.0, size=(h,))).astype(np.float32)
    dt_bias = rng.uniform(-1.0, 1.0, size=(h * d,)).astype(np.float32)
    t = torch.from_numpy
    for lower_bound in (-5.0, None):
      with self.subTest(lower_bound=lower_bound):
        expected = kimi_k3_kda_reference.kda_gate(t(g_raw), t(a_log), t(dt_bias), lower_bound)
        got = kda_gate(jnp.asarray(g_raw), jnp.asarray(a_log), jnp.asarray(dt_bias), lower_bound)
        self.assertEqual(got.dtype, jnp.float32)
        np.testing.assert_allclose(np.asarray(got), expected.numpy(), rtol=1e-6, atol=1e-6)

  def test_short_convolution_parity(self):
    """ShortConvolution output and returned cache match the torch reference."""
    torch.manual_seed(42)
    b, t, c, k = self.batch_size, self.seq_len, self.hidden_size, self.conv_size
    x_np = np.random.default_rng(0).standard_normal((b, t, c)).astype(np.float32)

    conv_pt = ShortConvolution_PT(hidden_size=c, kernel_size=k, activation="silu")
    with torch.no_grad():
      y_pt, state_pt = conv_pt(torch.tensor(x_np), output_final_state=True)

    conv_jax = ShortConvolution(hidden_size=c, kernel_size=k, activation="silu")
    conv_jax.kernel[...] = jnp.array(conv_pt.weight.detach().numpy().transpose(2, 1, 0))  # pylint: disable=not-callable
    y_jax, state_jax = conv_jax(jnp.array(x_np), output_final_state=True)

    np.testing.assert_allclose(np.array(y_jax), y_pt.numpy(), rtol=1e-5, atol=1e-5)
    # PyTorch cache: [B, C, K-1], JAX cache: [B, K-1, C]
    np.testing.assert_allclose(np.array(state_jax).transpose(0, 2, 1), state_pt.numpy(), rtol=1e-5, atol=1e-5)

  def test_chunked_core_matches_reference(self):
    """`jax_chunk_kimi_delta_rule` (with an initial state) matches the oracle's output and final state."""
    b, t, h, d = self.batch_size, self.seq_len, self.num_heads, self.head_dim
    rng = np.random.default_rng(42)
    q, k, v = (rng.standard_normal((b, t, h, d)).astype(np.float32) for _ in range(3))
    g = -rng.uniform(0.1, 1.0, size=(b, t, h, d)).astype(np.float32)
    beta = rng.uniform(0.1, 0.9, size=(b, t, h)).astype(np.float32)
    h0 = rng.standard_normal((b, h, d, d)).astype(np.float32)

    with torch.no_grad():
      out_pt, final_h_pt = chunk_kda_PT(
          *(torch.tensor(x) for x in (q, k, v, g, beta)),
          initial_state=torch.tensor(h0),
          use_gate_in_kernel=False,
          use_beta_sigmoid_in_kernel=False,
      )

    q_norm = l2norm(jnp.array(q), dim=-1, eps=1e-6) / math.sqrt(d)
    k_norm = l2norm(jnp.array(k), dim=-1, eps=1e-6)
    out_jax, final_h_jax = jax_chunk_kimi_delta_rule(
        q_norm, k_norm, jnp.array(v), jnp.array(g), jnp.array(beta), chunk_size=8, initial_state=jnp.array(h0)
    )

    np.testing.assert_allclose(np.array(out_jax), out_pt.numpy(), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(np.array(final_h_jax), final_h_pt.numpy(), rtol=1e-4, atol=1e-4)

  def test_layer_forward_parity(self):
    """Full-layer forward (weights converted through the kimi-k3 param mapping) matches the reference."""
    pt_module, jax_module = self._make_pair(chunk_size=8)
    x_np = np.random.default_rng(42).standard_normal((self.batch_size, self.seq_len, self.hidden_size))
    x_np = x_np.astype(np.float32)
    with torch.no_grad():
      y_torch, _, _ = pt_module(torch.tensor(x_np))
    y_jax, _, _ = jax_module(jnp.array(x_np), model_mode=MODEL_MODE_TRAIN)
    np.testing.assert_allclose(np.array(y_jax), y_torch.numpy(), rtol=1e-4, atol=1e-4)

  def test_layer_autoregressive_decode_parity(self):
    """Token-by-token decode with threaded conv/recurrent state matches the reference."""
    pt_module, jax_module = self._make_pair(chunk_size=8)
    rng = np.random.default_rng(7)
    pt_conv_state = pt_rec_state = jax_conv_state = jax_rec_state = None
    for _ in range(4):
      x_step = rng.standard_normal((self.batch_size, 1, self.hidden_size)).astype(np.float32)
      with torch.no_grad():
        y_pt, pt_conv_state, pt_rec_state = pt_module(
            torch.tensor(x_step), conv_state=pt_conv_state, recurrent_state=pt_rec_state, output_final_state=True
        )
      y_jax, jax_conv_state, jax_rec_state = jax_module(
          jnp.array(x_step),
          conv_state=jax_conv_state,
          recurrent_state=jax_rec_state,
          model_mode=MODEL_MODE_AUTOREGRESSIVE,
          output_final_state=True,
      )
      np.testing.assert_allclose(np.array(y_jax), y_pt.numpy(), rtol=1e-4, atol=1e-4)

  def test_layer_vjp_parity(self):
    """Input and parameter gradients of a random-cotangent VJP match torch autograd through the reference."""
    pt_module, jax_module = self._make_pair(chunk_size=8)
    rng = np.random.default_rng(11)
    x_np = rng.standard_normal((self.batch_size, self.seq_len, self.hidden_size)).astype(np.float32)
    cot_np = rng.standard_normal((self.batch_size, self.seq_len, self.hidden_size)).astype(np.float32)

    x_pt = torch.tensor(x_np, requires_grad=True)
    y_pt, _, _ = pt_module(x_pt)
    (y_pt * torch.tensor(cot_np)).sum().backward()

    graphdef, params, rest = nnx.split(jax_module, nnx.Param, ...)

    def f(params, x):
      y, _, _ = nnx.merge(graphdef, params, rest)(x, model_mode=MODEL_MODE_TRAIN)
      return jnp.sum(y * jnp.asarray(cot_np))

    g_params, g_x = jax.grad(f, argnums=(0, 1))(params, jnp.asarray(x_np))
    tol = {"rtol": 1e-4, "atol": 1e-4}
    np.testing.assert_allclose(np.asarray(g_x), x_pt.grad.numpy(), **tol)
    h = self.num_heads
    pairs = {
        "q_proj": (g_params["q_proj"]["kernel"], pt_module.q_proj.weight.grad.T),
        "f_b_proj": (g_params["f_b_proj"]["kernel"], pt_module.f_b_proj.weight.grad.T),
        "b_proj": (g_params["b_proj"]["kernel"], pt_module.b_proj.weight.grad.T),
        "out": (g_params["out"]["kernel"], pt_module.o_proj.weight.grad.T),
        "k_conv1d": (g_params["k_conv1d"]["kernel"], pt_module.k_conv1d.weight.grad.permute(2, 1, 0)),
        "A_log": (g_params["A_log"], pt_module.A_log.grad[:h]),
        "dt_bias": (g_params["dt_bias"], pt_module.dt_bias.grad),
        "o_norm": (g_params["o_norm"]["scale"], pt_module.o_norm.weight.grad),
    }
    for name, (got, ref) in pairs.items():
      np.testing.assert_allclose(
          np.asarray(got.get_value() if hasattr(got, "get_value") else got), ref.numpy(), err_msg=name, **tol
      )

  def test_layer_bf16_parity(self):
    """bf16 layer (bf16 activations / projection weights, fp32 KDA params) vs the bf16 torch reference."""
    pt_module, _ = self._make_pair(chunk_size=8)
    jax_module = KimiDeltaAttention(
        hidden_size=self.hidden_size,
        num_heads=self.num_heads,
        head_dim=self.head_dim,
        conv_size=self.conv_size,
        chunk_size=8,
        rms_norm_eps=self.rms_norm_eps,
        gate_lower_bound=self.gate_lower_bound,
        dtype=jnp.bfloat16,
        weight_dtype=jnp.bfloat16,
        compute_dtype=jnp.bfloat16,
        rngs=nnx.Rngs(params=0),
    )
    # As released: projections in bf16, A_log / dt_bias / conv / o_norm in fp32.
    for mod in (
        pt_module.q_proj,
        pt_module.k_proj,
        pt_module.v_proj,
        pt_module.f_a_proj,
        pt_module.f_b_proj,
        pt_module.b_proj,
        pt_module.g_proj,
        pt_module.o_proj,
    ):
      mod.to(torch.bfloat16)
    load_reference_into_kda(pt_module, jax_module, self.hf_config)
    x_np = np.random.default_rng(3).standard_normal((self.batch_size, self.seq_len, self.hidden_size))
    x_np = x_np.astype(np.float32)
    with torch.no_grad():
      y_torch, _, _ = pt_module(torch.tensor(x_np).to(torch.bfloat16))
    y_jax, _, _ = jax_module(jnp.asarray(x_np, jnp.bfloat16), model_mode=MODEL_MODE_TRAIN)
    self.assertEqual(y_jax.dtype, jnp.bfloat16)
    y_ref = y_torch.float().numpy()
    rel = np.linalg.norm(np.asarray(y_jax, np.float32) - y_ref) / np.linalg.norm(y_ref)
    self.assertLess(rel, 2e-2)


if __name__ == "__main__":
  unittest.main()
