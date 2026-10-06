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

"""Unit tests for Group 1: Kimi Delta Attention (KDA) Linear Attention in MaxText."""

import math
from typing import Any, Optional
import unittest
from einops import rearrange
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from maxtext.checkpoint_conversion.utils.param_mapping import HOOK_FNS, PARAM_MAPPING
from maxtext.common.common_types import MODEL_MODE_AUTOREGRESSIVE, MODEL_MODE_TRAIN
from maxtext.layers.attention_kda import (
    KimiDeltaAttention,
    ShortConvolution,
    jax_chunk_kimi_delta_rule,
    kda_gate,
)
from maxtext.layers.normalizations import l2norm
from tests.utils import kimi_k3_kda_reference
from tests.utils.kimi_k3_conversion_utils import convert_pytorch_module_to_maxtext_params

_ORIG_MATMUL_PRECISION = None


def setUpModule():
  global _ORIG_MATMUL_PRECISION
  _ORIG_MATMUL_PRECISION = jax.config.jax_default_matmul_precision
  jax.config.update("jax_default_matmul_precision", "highest")
  jax.config.update("jax_platforms", "cpu")


def tearDownModule():
  if _ORIG_MATMUL_PRECISION is not None:
    jax.config.update("jax_default_matmul_precision", _ORIG_MATMUL_PRECISION)


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

  def forward(self, x, cache=None, output_final_state=False, cu_seqlens=None):
    """Applies 1D short depthwise convolution along sequence length."""
    _, t, _ = x.shape
    x_t = x.transpose(1, 2)  # [B, C, T]
    if cache is not None:
      x_t = torch.cat([cache, x_t], dim=2)
    else:
      x_t = F.pad(x_t, (self.kernel_size - 1, 0))
    out = F.conv1d(x_t, self.weight, groups=self.hidden_size)  # pylint: disable=not-callable
    out = out[..., :t].transpose(1, 2)
    if self.activation == "silu":
      out = F.silu(out)
    final_state = x_t[..., -(self.kernel_size - 1) :] if output_final_state else None
    return out, final_state


class FusedRMSNormGated_PT(nn.Module):
  """PyTorch Reference for Gated RMSNorm."""

  def __init__(self, hidden_size: int, eps: float = 1e-5, activation: str = "sigmoid"):
    super().__init__()
    self.hidden_size = hidden_size
    self.eps = eps
    self.activation = activation
    self.weight = nn.Parameter(torch.ones(hidden_size))

  def forward(self, x, g):
    variance = x.pow(2).mean(-1, keepdim=True)
    normed = x * torch.rsqrt(variance + self.eps) * self.weight
    if self.activation == "sigmoid":
      gated = normed * torch.sigmoid(g)
    else:
      gated = normed * g
    return gated


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
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
    use_beta_sigmoid_in_kernel: bool = True,
    safe_gate: bool = True,
    lower_bound: Optional[float] = -5.0,
    transpose_state_layout: bool = True,
    cu_seqlens: Any = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
  """PyTorch reference for KDA: the fla-spec CPU oracle (see `tests/utils/kimi_k3_kda_reference.py`).

  The recurrent state is kept in MaxText's `[B, H, K, V]` layout so it can be compared
  directly with `jax_chunk_kimi_delta_rule`; `transpose_state_layout` is accepted for
  signature compatibility with `fla` and ignored.
  """
  del transpose_state_layout
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
      use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
      use_gate_in_kernel=use_gate_in_kernel,
      use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
      safe_gate=safe_gate,
      lower_bound=lower_bound,
      transpose_state_layout=False,
      cu_seqlens=cu_seqlens,
  )


class KimiDeltaAttention_PT(nn.Module):
  """PyTorch Reference for KimiDeltaAttention module."""

  def __init__(
      self,
      hidden_size: int = 7168,
      num_heads: int = 96,
      head_dim: int = 128,
      conv_size: int = 4,
      rms_norm_eps: float = 1e-5,
      gate_lower_bound: float = -5.0,
  ):
    super().__init__()
    self.hidden_size = hidden_size
    self.num_heads = num_heads
    self.head_dim = head_dim
    self.head_k_dim = head_dim
    self.num_k_heads = num_heads
    self.conv_size = conv_size
    self.gate_lower_bound = gate_lower_bound
    projection_size = head_dim * num_heads

    self.q_proj = nn.Linear(hidden_size, projection_size, bias=False)
    self.k_proj = nn.Linear(hidden_size, projection_size, bias=False)
    self.v_proj = nn.Linear(hidden_size, projection_size, bias=False)

    self.q_conv1d = ShortConvolution_PT(hidden_size=projection_size, kernel_size=conv_size)
    self.k_conv1d = ShortConvolution_PT(hidden_size=projection_size, kernel_size=conv_size)
    self.v_conv1d = ShortConvolution_PT(hidden_size=projection_size, kernel_size=conv_size)

    self.A_log = nn.Parameter(torch.log(torch.empty(num_heads, dtype=torch.float32).uniform_(1, 16)))
    self.f_a_proj = nn.Linear(hidden_size, head_dim, bias=False)
    self.f_b_proj = nn.Linear(head_dim, projection_size, bias=False)
    self.dt_bias = nn.Parameter(torch.empty(projection_size, dtype=torch.float32).uniform_(-1, 1))
    self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)
    self.g_proj = nn.Linear(hidden_size, projection_size, bias=False)
    self.o_norm = FusedRMSNormGated_PT(head_dim, eps=rms_norm_eps, activation="sigmoid")
    self.o_proj = nn.Linear(projection_size, hidden_size, bias=False)

  def forward(self, hidden_states, conv_state=None, recurrent_state=None, output_final_state=False):
    """Forward pass of PyTorch reference KimiDeltaAttention_PT."""
    conv_q, conv_k, conv_v = (None, None, None)
    if conv_state is not None:
      conv_q, conv_k, conv_v = conv_state

    q_proj_states = self.q_proj(hidden_states)
    k_proj_states = self.k_proj(hidden_states)
    v_proj_states = self.v_proj(hidden_states)

    q, next_conv_q = self.q_conv1d(q_proj_states, cache=conv_q, output_final_state=output_final_state)
    k, next_conv_k = self.k_conv1d(k_proj_states, cache=conv_k, output_final_state=output_final_state)
    v, next_conv_v = self.v_conv1d(v_proj_states, cache=conv_v, output_final_state=output_final_state)

    next_conv_state = (next_conv_q, next_conv_k, next_conv_v) if output_final_state else None

    g = self.f_b_proj(self.f_a_proj(hidden_states))
    g = rearrange(g, "... (h d) -> ... h d", d=self.head_dim)
    beta = self.b_proj(hidden_states).float()

    q = rearrange(q, "... (h d) -> ... h d", d=self.head_k_dim)
    k = rearrange(k, "... (h d) -> ... h d", d=self.head_k_dim)
    v = rearrange(v, "... (h d) -> ... h d", d=self.head_dim)

    o, next_rec_state = chunk_kda_PT(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=self.A_log,
        dt_bias=self.dt_bias,
        initial_state=recurrent_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        safe_gate=True,
        lower_bound=self.gate_lower_bound,
    )

    g_out = self.g_proj(hidden_states)
    g_out = rearrange(g_out, "... (h d) -> ... h d", d=self.head_dim)
    o = self.o_norm(o, g_out)
    o = rearrange(o, "b t h d -> b t (h d)")
    out = self.o_proj(o)
    return out, next_conv_state, next_rec_state


# ==============================================================================
# Helper to Load Converted Parameters into MaxText KDA
# ==============================================================================
def load_converted_params_into_kda(
    kda_module: KimiDeltaAttention,
    converted_params: dict[str, np.ndarray],
    layer_idx: int = 0,
):
  """Loads converted PyTorch weights into MaxText KimiDeltaAttention module."""
  prefix = f"params-decoder-layers_{layer_idx}-self_attention"
  kda_module.q_proj.kernel[...] = jnp.array(converted_params[f"{prefix}-q_proj-kernel"])
  kda_module.q_conv1d.kernel[...] = jnp.array(converted_params[f"{prefix}-q_conv1d-kernel"])
  kda_module.k_proj.kernel[...] = jnp.array(converted_params[f"{prefix}-k_proj-kernel"])
  kda_module.k_conv1d.kernel[...] = jnp.array(converted_params[f"{prefix}-k_conv1d-kernel"])
  kda_module.v_proj.kernel[...] = jnp.array(converted_params[f"{prefix}-v_proj-kernel"])
  kda_module.v_conv1d.kernel[...] = jnp.array(converted_params[f"{prefix}-v_conv1d-kernel"])
  kda_module.f_a_proj.kernel[...] = jnp.array(converted_params[f"{prefix}-f_a_proj-kernel"])
  kda_module.f_b_proj.kernel[...] = jnp.array(converted_params[f"{prefix}-f_b_proj-kernel"])
  kda_module.A_log[...] = jnp.array(converted_params[f"{prefix}-A_log"])
  kda_module.dt_bias[...] = jnp.array(converted_params[f"{prefix}-dt_bias"])
  kda_module.b_proj.kernel[...] = jnp.array(converted_params[f"{prefix}-b_proj-kernel"])
  kda_module.g_proj.kernel[...] = jnp.array(converted_params[f"{prefix}-g_proj-kernel"])
  kda_module.o_norm.scale[...] = jnp.array(converted_params[f"{prefix}-o_norm-scale"])
  kda_module.out.kernel[...] = jnp.array(converted_params[f"{prefix}-out-kernel"])


# ==============================================================================
# Unit Test Cases for Group 1 KDA
# ==============================================================================
class TestKimiK3G1KDA(unittest.TestCase):
  """Unit tests validating Kimi-K3 Group 1 KDA Linear Attention."""

  def setUp(self):
    super().setUp()
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
    }

  def _convert_pt_to_maxtext(self, pt_state_dict: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    """Helper converting PyTorch state dict using official param_mapping."""
    mapping_fn = PARAM_MAPPING["kimi-k3"]
    hook_fn_factory = HOOK_FNS["kimi-k3"]

    param_map = mapping_fn(self.hf_config, None, scan_layers=False)
    hook_map = hook_fn_factory(self.hf_config, None, scan_layers=False, saving_to_hf=False)

    converted = {}
    for mt_key, hf_target in param_map.items():
      if not mt_key.startswith("params-decoder-layers_0-self_attention"):
        continue

      matching_key = None
      for k in pt_state_dict.keys():
        if hf_target.endswith(k) or ("." in hf_target and ".".join(hf_target.split(".")[-2:]) == k):
          matching_key = k
          break

      if matching_key in pt_state_dict:
        tensor_np = pt_state_dict[matching_key].detach().cpu().float().numpy()
        if mt_key in hook_map:
          tensor_np = hook_map[mt_key](tensor_np)
        converted[mt_key] = tensor_np

    return converted

  def test_kda_gate_matches_reference(self):
    """MaxText `kda_gate` equals the fla-spec torch gate, with and without a lower bound."""
    rng = np.random.default_rng(0)
    h, d = self.num_heads, self.head_dim
    g_raw = rng.normal(scale=3.0, size=(2, 5, h, d)).astype(np.float32)
    a_log = np.log(rng.uniform(1.0, 16.0, size=(h,))).astype(np.float32)
    dt_bias = rng.uniform(-1.0, 1.0, size=(h * d,)).astype(np.float32)
    t = torch.from_numpy
    cases = {lb: kimi_k3_kda_reference.kda_gate(t(g_raw), t(a_log), t(dt_bias), lb) for lb in (-5.0, None)}
    for lower_bound, expected in cases.items():
      with self.subTest(lower_bound=lower_bound):
        got = kda_gate(jnp.asarray(g_raw), jnp.asarray(a_log), jnp.asarray(dt_bias), lower_bound)
        self.assertEqual(got.dtype, jnp.float32)
        np.testing.assert_allclose(np.asarray(got), expected.numpy(), rtol=1e-6, atol=1e-6)

  def test_kda_gate_lower_bound_is_bounded_sigmoid_not_clamp(self):
    """Regression: with a lower bound the gate is `lb * sigmoid(...)`, not `max(softplus gate, lb)`.

    At `g_raw + dt_bias = 0` the bounded-sigmoid gate is exactly `lb / 2` for every A_log,
    whereas the (wrong) clamped softplus gate would be `-exp(A_log) * log(2)`.
    """
    h, d = 3, 32
    a_log = jnp.log(jnp.array([1.0, 4.0, 16.0], dtype=jnp.float32))
    g = kda_gate(jnp.zeros((1, 1, h, d)), a_log, jnp.zeros((h * d,)), -5.0)
    np.testing.assert_allclose(np.asarray(g), -2.5, atol=1e-7)
    g_wide = kda_gate(jnp.linspace(-50.0, 50.0, h * d).reshape(1, 1, h, d), a_log, jnp.zeros((h * d,)), -5.0)
    self.assertTrue(bool(jnp.all((g_wide >= -5.0) & (g_wide <= 0.0))))

  def test_short_convolution_parity(self):
    """Verifies ShortConvolution parity against PyTorch reference for sequence and cache."""
    torch.manual_seed(42)
    b, t, c, k = self.batch_size, self.seq_len, self.hidden_size, self.conv_size
    x_np = np.random.randn(b, t, c).astype(np.float32)

    conv_pt = ShortConvolution_PT(hidden_size=c, kernel_size=k, activation="silu")
    with torch.no_grad():
      y_pt, state_pt = conv_pt(torch.tensor(x_np), output_final_state=True)
      y_pt = y_pt.numpy()
      state_pt = state_pt.numpy()

    # MaxText ShortConvolution
    conv_jax = ShortConvolution(hidden_size=c, kernel_size=k, activation="silu")
    conv_jax.kernel[...] = jnp.array(conv_pt.weight.detach().numpy().transpose(2, 1, 0))  # pylint: disable=not-callable

    y_jax, state_jax = conv_jax(jnp.array(x_np), output_final_state=True)

    np.testing.assert_allclose(np.array(y_jax), y_pt, rtol=1e-5, atol=1e-5)
    # PyTorch cache: [B, C, K-1], JAX cache: [B, K-1, C]
    np.testing.assert_allclose(np.array(state_jax).transpose(0, 2, 1), state_pt, rtol=1e-5, atol=1e-5)

  def test_chunked_kda_vs_recurrent_parity(self):
    """Verifies chunked WY scan parity against recurrent step-by-step execution."""
    b, t, h, d = self.batch_size, self.seq_len, self.num_heads, self.head_dim
    np.random.seed(42)
    torch.manual_seed(42)

    q = np.random.randn(b, t, h, d).astype(np.float32)
    k = np.random.randn(b, t, h, d).astype(np.float32)
    v = np.random.randn(b, t, h, d).astype(np.float32)
    g = -np.random.uniform(0.1, 1.0, size=(b, t, h, d)).astype(np.float32)
    beta = np.random.uniform(0.1, 0.9, size=(b, t, h)).astype(np.float32)
    h0 = np.random.randn(b, h, d, d).astype(np.float32)

    # PyTorch recurrence reference with QK L2 norm
    with torch.no_grad():
      out_pt, final_h_pt = chunk_kda_PT(
          q=torch.tensor(q),
          k=torch.tensor(k),
          v=torch.tensor(v),
          g=torch.tensor(g),
          beta=torch.tensor(beta),
          initial_state=torch.tensor(h0),
          output_final_state=True,
          use_qk_l2norm_in_kernel=True,
          use_gate_in_kernel=False,
          use_beta_sigmoid_in_kernel=False,
          safe_gate=False,
      )
      out_pt = out_pt.numpy()
      final_h_pt = final_h_pt.numpy()

    # JAX chunked WY scan with QK L2 norm
    scale = 1.0 / math.sqrt(d)
    q_norm = l2norm(jnp.array(q), dim=-1, eps=1e-6) * scale
    k_norm = l2norm(jnp.array(k), dim=-1, eps=1e-6)

    out_jax, final_h_jax = jax_chunk_kimi_delta_rule(
        query=q_norm,
        key=k_norm,
        value=jnp.array(v),
        g=jnp.array(g),
        beta=jnp.array(beta),
        chunk_size=8,
        initial_state=jnp.array(h0),
    )

    np.testing.assert_allclose(np.array(out_jax), out_pt, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(np.array(final_h_jax), final_h_pt, rtol=1e-4, atol=1e-4)

  def test_group1_kda_forward_parity(self):
    """Full layer forward parity test between PyTorch reference and MaxText KDA."""
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Instantiate PyTorch Reference Module
    pt_module = KimiDeltaAttention_PT(
        hidden_size=self.hidden_size,
        num_heads=self.num_heads,
        head_dim=self.head_dim,
        conv_size=self.conv_size,
        rms_norm_eps=self.rms_norm_eps,
        gate_lower_bound=self.gate_lower_bound,
    )

    # 2. Run PyTorch forward pass
    x_np = np.random.randn(self.batch_size, self.seq_len, self.hidden_size).astype(np.float32)
    with torch.no_grad():
      y_torch, _, _ = pt_module(torch.tensor(x_np))
      y_torch = y_torch.numpy()

    # 3. Convert weights via param_mapping
    converted_params = self._convert_pt_to_maxtext(pt_module.state_dict())

    # 4. Instantiate MaxText KDA Module and load converted parameters
    jax_module = KimiDeltaAttention(
        hidden_size=self.hidden_size,
        num_heads=self.num_heads,
        head_dim=self.head_dim,
        conv_size=self.conv_size,
        chunk_size=8,
        rms_norm_eps=self.rms_norm_eps,
        gate_lower_bound=self.gate_lower_bound,
    )
    load_converted_params_into_kda(jax_module, converted_params, layer_idx=0)

    # 5. Run MaxText forward pass
    x_jax = jnp.array(x_np)
    y_jax, _, _ = jax_module(x_jax, model_mode=MODEL_MODE_TRAIN)

    # 6. Verify numerical parity
    np.testing.assert_allclose(np.array(y_jax), y_torch, rtol=1e-4, atol=1e-4)

  def test_kda_compute_dtype_knob(self):
    """`compute_dtype` / `kda_compute_dtype` selects the core precision (default fp32, like fla)."""
    kwargs = {
        "hidden_size": self.hidden_size,
        "num_heads": self.num_heads,
        "head_dim": self.head_dim,
        "conv_size": self.conv_size,
        "chunk_size": 8,
        "rms_norm_eps": self.rms_norm_eps,
        "gate_lower_bound": self.gate_lower_bound,
    }
    self.assertEqual(KimiDeltaAttention(**kwargs).compute_dtype, jnp.float32)

    class _Cfg:  # minimal config object; KimiDeltaAttention reads attributes with getattr
      emb_dim, num_query_heads, head_dim = self.hidden_size, self.num_heads, self.head_dim
      kda_compute_dtype = "bfloat16"

    self.assertEqual(KimiDeltaAttention(config=_Cfg()).compute_dtype, jnp.bfloat16)

    x = jnp.asarray(np.random.default_rng(0).normal(size=(self.batch_size, self.seq_len, self.hidden_size)), jnp.float32)
    m32 = KimiDeltaAttention(**kwargs, rngs=nnx.Rngs(params=0))
    m16 = KimiDeltaAttention(**kwargs, compute_dtype=jnp.bfloat16, rngs=nnx.Rngs(params=0))
    y32, _, _ = m32(x, model_mode=MODEL_MODE_TRAIN)
    y16, _, _ = m16(x, model_mode=MODEL_MODE_TRAIN)
    self.assertEqual(y16.dtype, y32.dtype)  # core output is cast back to the activation dtype
    rel = float(jnp.linalg.norm(y16 - y32) / jnp.linalg.norm(y32))
    self.assertLess(rel, 2e-2)
    self.assertGreater(rel, 0.0)  # the knob really changes the core precision

  def test_group1_kda_linear_attention_parity(self):
    """Verifies production-scale KDA linear attention forward pass parity against PyTorch reference."""
    torch.manual_seed(42)
    np.random.seed(42)

    hidden_size = 7168
    num_heads = 96
    head_dim = 128
    conv_size = 4
    chunk_size = 16
    batch_size = 2
    seq_len = 16
    rms_norm_eps = 1e-5
    gate_lower_bound = -5.0

    hf_config = {
        "hidden_size": hidden_size,
        "num_hidden_layers": 93,
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_heads,
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": head_dim,
        "qk_rope_head_dim": 64,
        "v_head_dim": head_dim,
        "num_experts": 896,
        "num_experts_per_tok": 16,
        "routed_expert_hidden_size": 3584,
        "moe_intermediate_size": 3072,
        "shared_intermediate_size": 6144,
        "rms_norm_eps": rms_norm_eps,
        "first_k_dense_replace": 1,
        "attn_res_block_size": 12,
        "full_attn_layers": [
            3,
            7,
            11,
            15,
            19,
            23,
            27,
            31,
            35,
            39,
            43,
            47,
            51,
            55,
            59,
            63,
            67,
            71,
            75,
            79,
            83,
            87,
            91,
            92,
        ],
    }

    # 1. Instantiate PyTorch Reference Module at production scale
    pt_module = KimiDeltaAttention_PT(
        hidden_size=hidden_size,
        num_heads=num_heads,
        head_dim=head_dim,
        conv_size=conv_size,
        rms_norm_eps=rms_norm_eps,
        gate_lower_bound=gate_lower_bound,
    )

    # 2. Run forward pass with synthetic PyTorch input -> y_torch
    x_np = np.random.randn(batch_size, seq_len, hidden_size).astype(np.float32)
    with torch.no_grad():
      y_torch, _, _ = pt_module(torch.tensor(x_np))
      y_torch = y_torch.numpy()

    # 3. Convert PyTorch module weights to MaxText parameter format
    converted_params = convert_pytorch_module_to_maxtext_params(
        pt_module.state_dict(),
        hf_config,
        None,
        prefix_filter="params-decoder-layers_0-self_attention",
    )

    # 4. Instantiate MaxText module with the converted weights
    jax_module = KimiDeltaAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        head_dim=head_dim,
        conv_size=conv_size,
        chunk_size=chunk_size,
        rms_norm_eps=rms_norm_eps,
        gate_lower_bound=gate_lower_bound,
    )
    load_converted_params_into_kda(jax_module, converted_params, layer_idx=0)

    # 5. Convert input tensor to JAX array -> x_jax
    x_jax = jnp.array(x_np)

    # 6. Run forward pass through MaxText module -> y_jax
    y_jax, _, _ = jax_module(x_jax, model_mode=MODEL_MODE_TRAIN)

    # 7. Verify numerical parity
    np.testing.assert_allclose(np.array(y_jax), y_torch, rtol=1e-4, atol=1e-4)

  def test_kda_autoregressive_decode_step_parity(self):
    """Verifies single-token autoregressive decoding parity against PyTorch reference."""
    torch.manual_seed(42)
    np.random.seed(42)

    pt_module = KimiDeltaAttention_PT(
        hidden_size=self.hidden_size,
        num_heads=self.num_heads,
        head_dim=self.head_dim,
        conv_size=self.conv_size,
        rms_norm_eps=self.rms_norm_eps,
        gate_lower_bound=self.gate_lower_bound,
    )
    converted_params = self._convert_pt_to_maxtext(pt_module.state_dict())

    jax_module = KimiDeltaAttention(
        hidden_size=self.hidden_size,
        num_heads=self.num_heads,
        head_dim=self.head_dim,
        conv_size=self.conv_size,
        chunk_size=8,
        rms_norm_eps=self.rms_norm_eps,
        gate_lower_bound=self.gate_lower_bound,
    )
    load_converted_params_into_kda(jax_module, converted_params, layer_idx=0)

    # Step through tokens one by one
    pt_conv_state = None
    pt_rec_state = None
    jax_conv_state = None
    jax_rec_state = None

    for _ in range(4):
      x_step = np.random.randn(self.batch_size, 1, self.hidden_size).astype(np.float32)
      with torch.no_grad():
        y_pt, pt_conv_state, pt_rec_state = pt_module(
            torch.tensor(x_step),
            conv_state=pt_conv_state,
            recurrent_state=pt_rec_state,
            output_final_state=True,
        )
        y_pt = y_pt.numpy()

      y_jax, jax_conv_state, jax_rec_state = jax_module(
          jnp.array(x_step),
          conv_state=jax_conv_state,
          recurrent_state=jax_rec_state,
          model_mode=MODEL_MODE_AUTOREGRESSIVE,
          output_final_state=True,
      )

      np.testing.assert_allclose(np.array(y_jax), y_pt, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
  unittest.main()
