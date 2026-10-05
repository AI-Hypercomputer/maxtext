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
fla's `chunk_kda` semantics). Skipped when torch is not installed.
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

  def __init__(self, hidden_size: int, eps: float = 1e-5):
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(hidden_size))

  def forward(self, x, g):
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + self.eps) * self.weight * torch.sigmoid(g)


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

    self.A_log = nn.Parameter(torch.log(torch.empty(num_heads, dtype=torch.float32).uniform_(1, 16)))
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
        A_log=self.A_log,
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
_PREFIX = "params-decoder-layers_0-self_attention-"


def convert_kda_state_dict(pt_state_dict: dict[str, Any], hf_config: dict[str, Any]) -> dict[str, np.ndarray]:
  """Maps a `KimiDeltaAttention_PT` state dict to MaxText layer-0 KDA params via PARAM_MAPPING/HOOK_FNS."""
  param_map = PARAM_MAPPING["kimi-k3"](hf_config, None, scan_layers=False)
  hook_map = HOOK_FNS["kimi-k3"](hf_config, None, scan_layers=False, saving_to_hf=False)
  converted = {}
  for mt_key, hf_target in param_map.items():
    if not mt_key.startswith(_PREFIX):
      continue
    pt_key = hf_target.split(".self_attn.", 1)[1]
    tensor_np = pt_state_dict[pt_key].detach().cpu().float().numpy()  # KeyError on any mapping drift
    converted[mt_key] = hook_map[mt_key](tensor_np) if mt_key in hook_map else tensor_np
  return converted


def load_converted_params_into_kda(kda_module: KimiDeltaAttention, converted_params: dict[str, np.ndarray]):
  """Loads converted PyTorch weights into a MaxText KimiDeltaAttention module."""
  for name in ("q_proj", "k_proj", "v_proj", "q_conv1d", "k_conv1d", "v_conv1d", "f_a_proj", "f_b_proj"):
    getattr(kda_module, name).kernel[...] = jnp.array(converted_params[f"{_PREFIX}{name}-kernel"])
  for name in ("b_proj", "g_proj", "out"):
    getattr(kda_module, name).kernel[...] = jnp.array(converted_params[f"{_PREFIX}{name}-kernel"])
  kda_module.A_log[...] = jnp.array(converted_params[f"{_PREFIX}A_log"])
  kda_module.dt_bias[...] = jnp.array(converted_params[f"{_PREFIX}dt_bias"])
  kda_module.o_norm.scale[...] = jnp.array(converted_params[f"{_PREFIX}o_norm-scale"])


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
    load_converted_params_into_kda(jax_module, convert_kda_state_dict(pt_module.state_dict(), self.hf_config))
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


if __name__ == "__main__":
  unittest.main()
