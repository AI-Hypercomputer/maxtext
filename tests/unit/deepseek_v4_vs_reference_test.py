# pylint: skip-file
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

"""Tests for DeepSeek-V4 Attention and Compressor parity."""

import sys
import unittest
from collections.abc import Callable
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from flax import nnx
from maxtext.configs import pyconfig
from maxtext.layers.moe import DeepSeekV4TopKRouter, DeepSeekV4HashRouter
from maxtext.layers import attention_compressed, mhc
from maxtext.models.deepseek_v4 import DeepSeekV4DecoderLayer, DeepSeekV4ScannableBlock, DeepSeekV4HyperHead
from maxtext.layers.embeddings import DeepSeekV4RotaryEmbedding, apply_rotary_pos_emb, Embed
from maxtext.layers.normalizations import DeepSeekV4RMSNorm, DeepSeekV4UnweightedRMSNorm
from maxtext.layers.linears import DeepSeekGroupedLinear
from maxtext.layers.nnx_decoders import NNXDecoder
import maxtext.common.common_types as ctypes
from tests.utils.test_helpers import get_test_config_path, get_decoupled_parallelism_overrides


# ==============================================================================
# 1. Mock / Stub classes to support the exact Hugging Face / Scratch model code
# ==============================================================================


class RopeParameters(dict):
  pass


class PreTrainedConfig:

  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      setattr(self, k, v)


class DeepseekV4Config(PreTrainedConfig):

  def __init__(self, **kwargs):
    # Default V4-Flash configuration values for testing
    self.vocab_size = 129280
    self.hidden_size = 4096
    self.moe_intermediate_size = 2048
    self.num_hidden_layers = 43
    self.num_attention_heads = 64
    self.num_key_value_heads = 1
    self.head_dim = 512
    self.q_lora_rank = 1024
    self.partial_rotary_factor = 64 / 512
    self.qk_rope_head_dim = 64
    self.max_position_embeddings = 1048576
    self.rope_theta = 10000.0
    self.compress_rope_theta = 160000.0
    self.compress_rates = {
        "compressed_sparse_attention": 4,
        "heavily_compressed_attention": 128,
    }
    self.compress_ratios = [128] * 43
    self.sliding_window = 128
    self.o_groups = 8
    self.o_lora_rank = 1024
    self.index_n_heads = 64
    self.index_head_dim = 128
    self.index_topk = 512
    self.rms_norm_eps = 1.0e-6
    self.attention_dropout = 0.0
    self._attn_implementation = "eager"
    self.matmul_precision = "default"
    self.layer_types = ["compressed_sparse_attention"] * 43
    self.mlp_layer_types = ["hash_moe"] * 43
    self.num_experts_per_tok = 6
    self.n_routed_experts = 256
    self.num_local_experts = 256
    self.n_shared_experts = 1
    self.scoring_func = "sqrtsoftplus"
    self.routed_scaling_factor = 1.5
    self.intermediate_size = 2048
    self.hidden_act = "silu"
    self.swiglu_limit = 10.0
    self.mlp_bias = False
    self.attention_bias = False
    self.hc_mult = 4
    self.hc_sinkhorn_iters = 20
    self.hc_eps = 1e-6

    # Setup default rope parameters
    dim = int(self.head_dim * self.partial_rotary_factor)
    self.rope_parameters = {
        "main": {
            "rope_type": "default",
            "rope_theta": self.rope_theta,
            "partial_rotary_factor": self.partial_rotary_factor,
        },
        "compress": {
            "rope_type": "default",
            "rope_theta": self.compress_rope_theta,
            "partial_rotary_factor": self.partial_rotary_factor,
        },
    }
    super().__init__(**kwargs)


class DynamicSlidingWindowLayer:

  def __init__(self, config: DeepseekV4Config):
    self.sliding_window = config.sliding_window
    self.keys = None
    self.values = None
    self.is_initialized = False
    self.cumulative_length = 0

  def lazy_initialization(self, key_states, value_states):
    self.keys = key_states
    self.values = value_states
    self.is_initialized = True


class Cache:

  def __init__(self):
    self.layers = []


class OutputRecorder:
  pass


class FlashAttentionKwargs(dict):
  pass


try:
  from typing import Unpack
except ImportError:
  from typing_extensions import Unpack


# Mock / stub decorator
def use_kernel_forward_from_hub(*args, **kwargs):
  def decorator(cls):
    return cls

  return decorator


# Stub implementation of ALL_ATTENTION_FUNCTIONS
class AllAttentionFunctionsStub:

  def get_interface(self, implementation_name, default_fn):
    return default_fn


ALL_ATTENTION_FUNCTIONS = AllAttentionFunctionsStub()

# Dummy registry to make the copied file happy
ROPE_INIT_FUNCTIONS = {}
dynamic_rope_update = lambda fn: fn
maybe_autocast = lambda device_type, enabled: torch.enable_grad()  # No-op context

use_experts_implementation = lambda cls: cls


class TransformersKwargs(dict):
  pass


ACT2FN = {
    "silu": F.silu,
    "sigmoid": torch.sigmoid,
    "sqrtsoftplus": lambda x: torch.sqrt(F.softplus(x)),
}

# ==============================================================================
# 2. EXACT COPY OF PYTORCH REFERENCE CLASSES (SOURCE OF TRUTH - READ ONLY)
# ==============================================================================


class DeepseekV4RMSNorm_PT(nn.Module):

  def __init__(self, hidden_size, eps: float = 1e-6) -> None:
    """
    DeepseekV4RMSNorm is equivalent to T5LayerNorm
    """
    super().__init__()
    self.weight = nn.Parameter(torch.ones(hidden_size))
    self.variance_epsilon = eps

  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    return self.weight * hidden_states.to(input_dtype)

  def extra_repr(self):
    return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class DeepseekV4UnweightedRMSNorm_PT(nn.Module):

  def __init__(self, eps: float = 1.0e-6):
    super().__init__()
    self.eps = eps

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps).to(x.dtype)


class DeepseekV4RotaryEmbedding_PT(nn.Module):
  """
  Multi-layer-type rotary embedding (Laguna pattern: partial rotary on top of
  Gemma3's per-layer-type buffers), specialised for V4's *interleaved* RoPE.
  Interleaved RoPE: one `θ_i` per pair (`rope_head_dim // 2` entries),
  DIFF no end-to-end duplication. Same shape as `inv_freq @ position_ids`.

  V4 deliberately decouples its architecture `layer_types`
  (`sliding_attention` / `compressed_sparse_attention` /
  `heavily_compressed_attention`) from its rope-type labels (`main` /
  `compress`) — the latter live as keys in `config.rope_parameters` and
  only differ in their `rope_theta` base. So this override replaces
  Laguna's `set(config.layer_types)` iteration with `rope_parameters.keys()`
  when building the per-type inv_freq buffers.
  """

  inv_freq: torch.Tensor  # fix linting for `register_buffer`

  def __init__(self, config: DeepseekV4Config):
    super().__init__()
    self.max_seq_len_cached = config.max_position_embeddings
    self.original_max_seq_len = config.max_position_embeddings
    self.config = config
    # Only the nested per-rope-type sub-dicts are real layer types — the top-level
    # `rope_type` key that ``convert_rope_params_to_dict`` may leave on
    # ``config.rope_parameters`` is a flat-shape leftover, not a layer.
    self.layer_types = [k for k, v in config.rope_parameters.items() if isinstance(v, dict)]
    self.rope_type = {}
    for layer_type in self.layer_types:
      rope_params = config.rope_parameters[layer_type]
      self.rope_type[layer_type] = rope_params["rope_type"]
      rope_init_fn = self.compute_default_rope_parameters
      if self.rope_type[layer_type] != "default":
        rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type[layer_type]]
      inv_freq, attention_scaling = rope_init_fn(config, layer_type=layer_type)
      self.register_buffer(f"{layer_type}_inv_freq", inv_freq, persistent=False)
      self.register_buffer(f"{layer_type}_original_inv_freq", inv_freq.clone(), persistent=False)
      setattr(self, f"{layer_type}_attention_scaling", attention_scaling)

  @staticmethod
  def compute_default_rope_parameters(
      config: DeepseekV4Config | None = None,
      device: Optional["torch.device"] = None,
      seq_len: int | None = None,
      layer_type: str | None = None,
  ) -> tuple["torch.Tensor", float]:
    """
    Computes the inverse frequencies according to the original RoPE implementation
    Args:
        config ([`~transformers.PreTrainedConfig`]):
            The model configuration.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.
        layer_type (`str`, *optional*):
            The current layer type if the model has different RoPE parameters per type.
            Should not be used unless `config.layer_types is not None`
    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
    """
    base = config.rope_parameters[layer_type]["rope_theta"]
    # key difference to gemma3: partial rope
    partial_rotary_factor = config.rope_parameters[layer_type].get("partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial_rotary_factor)

    attention_factor = 1.0  # Unused in this type of RoPE

    # Compute the inverse frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
    return inv_freq, attention_factor

  @torch.no_grad()
  @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
  def forward(self, x, position_ids, layer_type=None):
    # Key difference vs Laguna's forward: no `torch.cat([freqs, freqs], dim=-1)`
    # duplication. V4's interleaved RoPE pairs consecutive channels, so we only need
    # `rope_head_dim // 2` unique θ entries — the `apply_rotary_pos_emb` helper does
    # the `repeat_interleave(2)` next to the rotation math, where the link between
    # the doubled dim and `rotate_half` is local and obvious.
    inv_freq = getattr(self, f"{layer_type}_inv_freq")
    attention_scaling = getattr(self, f"{layer_type}_attention_scaling")
    inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
    position_ids_expanded = position_ids[:, None, :].float()
    device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
    with maybe_autocast(device_type=device_type, enabled=False):
      freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
      cos = freqs.cos() * attention_scaling
      sin = freqs.sin() * attention_scaling
    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class DeepseekV4HCACache(DynamicSlidingWindowLayer):
  r"""Cache layer for HCA blocks (paper §2.3.2). Holds the long-range compressor's
  buffer / running compressed entries / count on top of the sliding-window K=V
  branch. HCA uses *non-overlapping* windows, so there is *no* overlap state,
  and HCA has *no* indexer either.

  State is dict-keyed by entry name — HCA only uses `"compressor"`, but
  :class:`DeepseekV4CSACache` adds `"indexer"` to the same dicts so a single
  set of methods (`store_compression_weights` / `update_compressor_states`)
  serves both:

    * `compressed_kv[name]` — the running list of compressed KV entries
      emitted so far (one every `compress_rate` source tokens; the long-range
      KVs the attention concatenates onto its sliding-window keys / values).
    * `buffer_kv[name]` / `buffer_gate[name]` — source tokens that arrived
      between two full windows; once the buffer hits `compress_rate` tokens
      the compressor closes a window, emits one entry, and drains the buffer.
    * `entry_count[name]` — number of compressed entries emitted so far, so
      `entry_count[name] * compress_rate` is the absolute position of the
      *next* window's first source token. Tracked separately from
      `position_ids` so prefill -> decode -> prefill stays consistent.
  """

  layer_type = "heavily_compressed_attention"

  def __init__(self, config: "DeepseekV4Config"):
    super().__init__(config)
    self.compress_rate = config.compress_rates["heavily_compressed_attention"]
    self.buffer_kv: dict[str, torch.Tensor | None] = {"compressor": None}
    self.buffer_gate: dict[str, torch.Tensor | None] = {"compressor": None}
    self.compressed_kv: dict[str, torch.Tensor | None] = {"compressor": None}
    self.entry_count: dict[str, int] = {"compressor": 0}

  def update(self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs):
    """
    Shared sliding-window K=V update body. V4 uses shared-KV MQA, so `keys` and
    `values` point to the same storage on every layer.
    """
    if not self.is_initialized:
      self.lazy_initialization(key_states, value_states)
      self.values = self.keys
    self.cumulative_length += key_states.shape[-2]
    full = torch.cat([self.keys, key_states], dim=-2)
    self.keys = full[:, :, -self.sliding_window + 1 :, :]
    self.values = self.keys
    return full, full

  def store_compression_weights(
      self, name: str, kv: torch.Tensor, gate: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, int]:
    r"""
    Concatenate the new projected `(kv, gate)` (paper §2.3.2 eqs. 20–21:
    `C = H·W^{KV}`, `Z = H·W^Z`) for entry `name` with what's already in
    the buffer, peel off the longest window-aligned prefix (the chunk
    ready to compress), keep the leftover in the buffer for next call,
    and return `(chunk_kv, chunk_gate, first_window_position)`. The
    returned chunk is softmax-aggregated by the compressor with
    `position_bias` to emit one compressed entry per window of
    `compress_rate` tokens.
    """
    first_window_position = self.entry_count[name] * self.compress_rate
    buffered_kv, buffered_gate = self.buffer_kv[name], self.buffer_gate[name]
    if buffered_kv is not None and buffered_kv.shape[1]:
      kv = torch.cat([buffered_kv, kv], dim=1)
      gate = torch.cat([buffered_gate, gate], dim=1)
    # only return the longest prefix that's a multiple of compress_rate; the rest stays in the buffer for next time
    usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
    self.buffer_kv[name], self.buffer_gate[name] = kv[:, usable:], gate[:, usable:]
    return kv[:, :usable], gate[:, :usable], first_window_position

  def update_compressor_states(self, name: str, compressed: torch.Tensor) -> torch.Tensor:
    r"""
    Append freshly emitted compressed entries to `compressed_kv[name]`
    (`C^{Comp}`, paper §2.3.2 eq. 23), bump `entry_count[name]`, and
    return the running `compressed_kv[name]`.
    """
    if self.compressed_kv[name] is None:
      self.compressed_kv[name] = compressed
    elif compressed.shape[1] > 0:
      self.compressed_kv[name] = torch.cat([self.compressed_kv[name], compressed], dim=1)
    self.entry_count[name] += compressed.shape[1]
    return self.compressed_kv[name]


class DeepseekV4CSACache(DeepseekV4HCACache):
  r"""Cache layer for CSA blocks (paper §2.3.1). Extends :class:`DeepseekV4HCACache`
  by adding an `"indexer"` entry to the inherited `buffer_kv` / `buffer_gate` /
  `compressed_kv` / `entry_count` dicts, plus per-name *overlap* state for the
  two-series window scheme.

  What "overlap" means here: the CSA `kv_proj` / `gate_proj` produce `2 * head_dim`
  features per source token — two independent compressed series Ca and Cb stored
  in one tensor. Ca occupies `[..., :head_dim]`, Cb occupies `[..., head_dim:]`.
  Pooled entry `w` is the softmax-gated convex combination of window `w-1`'s Ca
  slice with window `w`'s Cb slice — effective width `2 * compress_rate_csa`,
  stride `compress_rate_csa` (paper §2.3.1).

  Because adjacent windows share state only through *the previous window's Ca
  slice*, the only thing we need to carry across a forward boundary is
  `chunk[:, -1, :, :head_dim]` (Ca) of the last full window — Cb is never read
  again. That's what `overlap_kv[name]` / `overlap_gate[name]` persist.
  """

  layer_type = "compressed_sparse_attention"

  def __init__(self, config: "DeepseekV4Config"):
    super().__init__(config)
    self.compress_rate = config.compress_rates["compressed_sparse_attention"]
    self.buffer_kv["indexer"] = None
    self.buffer_gate["indexer"] = None
    self.compressed_kv["indexer"] = None
    self.entry_count["indexer"] = 0
    self.overlap_kv: dict[str, torch.Tensor | None] = {"compressor": None, "indexer": None}
    self.overlap_gate: dict[str, torch.Tensor | None] = {"compressor": None, "indexer": None}

  def update_overlap_state(
      self, name: str, chunk_kv: torch.Tensor, chunk_gate: torch.Tensor, head_dim: int
  ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    r"""
    Read the `name` entry's prior window's Ca slice (saved on the previous
    forward call) and persist the *current* call's last-window Ca slice for
    the next call. Only the `:head_dim` slice (Ca) is ever consumed
    downstream — Cb has already been folded into the previous window's
    emitted compressed entry — so we store half what `chunk[:, -1]` holds.
    Returns `(prior_kv, prior_gate)` — both `None` on the very first call.
    """
    prior_kv, prior_gate = self.overlap_kv[name], self.overlap_gate[name]
    self.overlap_kv[name] = chunk_kv[:, -1, :, :head_dim].clone()
    self.overlap_gate[name] = chunk_gate[:, -1, :, :head_dim].clone()
    return prior_kv, prior_gate


class DeepseekV4GroupedLinear_PT(nn.Linear):
  """Block-diagonal grouped linear used by the grouped output projection
  The core attention's stacked output is `num_attention_heads* head_dim`-dim,
  which is *very* large (V4-Flash: 32768; V4-Pro: 65536). A direct
  `num_attention_heads*head_dim → hidden_size` projection would dominate the per-token cost.

  The paper sidesteps that by splitting the heads into `g` groups, projecting
  each `num_attention_heads * head_dim/g`-dim group independently to a `d_g`-dim intermediate output
  (with `d_g < num_attention_heads * head_dim/g`), and then mixing the resulting `g·d_g` vector to
  `hidden_size` through a single follow-up linear (`self_attn.o_b_proj`). This
  module owns the per-group block (`self_attn.o_a_proj`).

  For V4-Flash (num_attention_heads=64, head_dim=512, o_groups=8, o_lora_rank=1024,
  hidden_size=4096), g=8 groups of 4096-dim each are projected to 1024-dim, then
  mixed to 4096-dim; for V4-Pro (num_attention_heads=128, head_dim=512, o_groups=16,
  o_lora_rank=1024, hidden_size=7168), g=16 groups of 4096-dim each are projected
  to 1024-dim, then mixed to 7168-dim.
  """

  def __init__(self, in_features_per_group: int, out_features: int, n_groups: int, bias: bool = False):
    super().__init__(in_features_per_group, out_features, bias=bias)
    self.n_groups = n_groups

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    input_shape = x.shape[:-2]
    hidden_dim = x.shape[-1]
    w = self.weight.view(self.n_groups, -1, hidden_dim).transpose(1, 2)
    x = x.reshape(-1, self.n_groups, hidden_dim).transpose(0, 1)
    y = torch.bmm(x, w).transpose(0, 1)
    return y.reshape(*input_shape, self.n_groups, -1)


def rotate_half(x):
  """Rotates half the hidden dims of the input."""
  x1 = x[..., 0::2]
  x2 = x[..., 1::2]
  return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rotary_pos_emb_PT(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1
) -> torch.Tensor:
  """V4 interleaved RoPE applied to the *trailing* rope slice of `x`.

  `cos` / `sin` come in half-sized (one entry per interleaved pair, from
  `DeepseekV4RotaryEmbedding`); we expand them to the full rope dim with
  `repeat_interleave`, then rotate the last `2 * cos.shape[-1]` channels of `x`
  with the standard `x*cos + rotate_half(x)*sin` formula in fp32 and leave the
  leading nope channels untouched. V4-Flash lays each head out as `[nope | rope]`,
  matching the reference's `x[..., -rd:]` indexing.
  """
  cos = cos.repeat_interleave(2, dim=-1).unsqueeze(unsqueeze_dim)
  sin = sin.repeat_interleave(2, dim=-1).unsqueeze(unsqueeze_dim)
  rope_dim = cos.shape[-1]
  nope, rope = x[..., :-rope_dim], x[..., -rope_dim:]
  rotated = ((rope.float() * cos) + (rotate_half(rope).float() * sin)).to(x.dtype)
  return torch.cat([nope, rotated], dim=-1)


class DeepseekV4HCACompressor_PT(nn.Module):
  """
  Heavily Compressed Attention compressor (paper §2.3.2, eqs. 20–23). compresses
  every `compress_rate_hca` (m'=128) source tokens into a single compressed KV
  entry.

  Each closed window of m' tokens produces one compressed entry:
  `C^{Comp}_i = Σ_{j∈window} softmax(Z_j + B)_j ⊙ C_j`. RoPE on the trailing
  `rope_head_dim` slice is applied at the deterministic absolute position
  `i * compress_rate_hca + first_window_position` so cross-call concatenation
  stays causality-correct. Returns the running list of *all* compressed
  entries emitted so far (shape `[B, 1, T, head_dim]` with
  `T = entry_count["compressor"]`), so the attention can attend over the
  full long-range history.

  When `past_key_values is None` runs in stateless single-shot mode: compress
  every complete window from `hidden_states` and discard the remainder
  (instead of caching it).
  """

  rope_layer_type = "compress"

  def __init__(self, config: DeepseekV4Config):
    super().__init__()
    self.compress_rate = config.compress_rates["heavily_compressed_attention"]
    self.head_dim = config.head_dim
    self.kv_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
    self.gate_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
    self.position_bias = nn.Parameter(torch.empty(self.compress_rate, self.head_dim))
    self.kv_norm = DeepseekV4RMSNorm_PT(self.head_dim, eps=config.rms_norm_eps)
    self.rotary_emb = DeepseekV4RotaryEmbedding_PT(config)

  def forward(
      self,
      hidden_states: torch.Tensor,
      q_residual: torch.Tensor,
      position_ids: torch.Tensor,
      past_key_values: Cache | None,
      layer_idx: int,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    batch, _, _ = hidden_states.shape
    cache_layer: DeepseekV4HCACache = past_key_values.layers[layer_idx] if past_key_values is not None else None
    kv = self.kv_proj(hidden_states)
    gate = self.gate_proj(hidden_states)
    if cache_layer is None:
      usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
      chunk_kv, chunk_gate, first_window_position = kv[:, :usable], gate[:, :usable], 0
    else:
      chunk_kv, chunk_gate, first_window_position = cache_layer.store_compression_weights("compressor", kv, gate)

    if chunk_kv.shape[1] > 0:  # there were at least self.compress_rate tokens
      n_windows = chunk_kv.shape[1] // self.compress_rate
      chunk_kv = chunk_kv.view(batch, n_windows, self.compress_rate, -1)
      chunk_gate = chunk_gate.view(batch, n_windows, self.compress_rate, -1) + self.position_bias.to(chunk_gate.dtype)
      compressed = self.kv_norm((chunk_kv * chunk_gate.softmax(dim=2, dtype=torch.float32).to(chunk_kv.dtype)).sum(dim=2))
      positions = torch.arange(n_windows, device=compressed.device)
      positions = (positions * self.compress_rate + first_window_position).unsqueeze(0).expand(batch, -1)
      cos, sin = self.rotary_emb(compressed, position_ids=positions, layer_type=self.rope_layer_type)
      compressed = apply_rotary_pos_emb_PT(compressed.unsqueeze(1), cos, sin).squeeze(1)
    else:
      compressed = chunk_kv.new_zeros((batch, 0, self.head_dim))

    if cache_layer is not None:
      compressed = cache_layer.update_compressor_states("compressor", compressed)
    compressed_kv = compressed.unsqueeze(1)

    compressed_len = compressed_kv.shape[2]
    seq_len = position_ids.shape[1]
    if seq_len == 1 or compressed_len == 0:
      return compressed_kv, None

    # query `t` may only see cache entries at pos `w` t > w * compress_rate (ex: t=7, w=2 t does not attend to it).
    entry_indices = torch.arange(compressed_len, device=compressed_kv.device)
    causal_threshold = (position_ids + 1) // self.compress_rate  # [B, S]
    block_bias = compressed_kv.new_zeros((batch, 1, seq_len, compressed_len))
    block_bias = block_bias.masked_fill(
        entry_indices.view(1, 1, 1, -1) >= causal_threshold.unsqueeze(1).unsqueeze(-1),
        float("-inf"),
    )
    return compressed_kv, block_bias


class DeepseekV4Indexer_PT(nn.Module):
  r"""Lightning Indexer (paper §2.3.1, eqs. 13–17). Used by Compressed Sparse
  Attention (CSA) to pick the top-`k` compressed KV blocks per query, with
  `k = config.index_topk`. Each query then attends only to those `k` of the
  `seq_len / compress_rate_csa` compressed entries — reduction factor
  `(seq_len / compress_rate_csa) / index_topk` over full attention against
  the entire compressed sequence.

  The indexer runs its own scaled-down compressor at `index_head_dim` over
  the same windows as the outer CSA compressor, then scores queries against
  the compressed keys with `∑_h w_{t,h} · ReLU(q_{t,h} · K^IComp_s)` and
  keeps the top `index_topk` indices.

  The indexer has its own rotary because it applies RoPE to two sets of
  tensors:

    * *compressed keys* at deterministic positions
      `i * compress_rate + first_window_position`,
    * *queries* at the model's current `position_ids` (variable per forward).

  Both must use the same theta as the outer compressor
  (`compress_rope_theta`) so query/key inner products are
  translation-invariant — if they used different thetas, `q · k` would carry
  a residual position-dependent skew. We can't precompute cos/sin once at
  init because the query positions vary per call, so the indexer owns its
  own rotary and calls it twice per forward (once for compressed keys, once
  for queries) with `layer_type=self.rope_layer_type` (always `"compress"`).
  """

  rope_layer_type = "compress"

  def __init__(self, config: DeepseekV4Config):
    super().__init__()
    self.compress_rate = config.compress_rates["compressed_sparse_attention"]
    self.num_heads = config.index_n_heads
    self.head_dim = config.index_head_dim
    self.index_topk = config.index_topk
    self.softmax_scale = self.head_dim**-0.5
    self.weights_scaling = self.num_heads**-0.5
    self.kv_proj = nn.Linear(config.hidden_size, 2 * self.head_dim, bias=False)
    self.gate_proj = nn.Linear(config.hidden_size, 2 * self.head_dim, bias=False)
    self.position_bias = nn.Parameter(torch.empty(self.compress_rate, 2 * self.head_dim))
    self.kv_norm = DeepseekV4RMSNorm_PT(self.head_dim, eps=config.rms_norm_eps)
    self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False)
    self.weights_proj = nn.Linear(config.hidden_size, self.num_heads, bias=False)
    self.rotary_emb = DeepseekV4RotaryEmbedding_PT(config)

  def forward(
      self,
      hidden_states: torch.Tensor,
      q_residual: torch.Tensor,
      position_ids: torch.Tensor,
      past_key_values: Cache | None,
      layer_idx: int,
  ) -> torch.LongTensor:
    batch, seq_len, _ = hidden_states.shape
    cache_layer: DeepseekV4CSACache = past_key_values.layers[layer_idx] if past_key_values is not None else None
    kv = self.kv_proj(hidden_states)
    gate = self.gate_proj(hidden_states)

    if cache_layer is None:
      usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
      chunk_kv, chunk_gate, first_window_position = kv[:, :usable], gate[:, :usable], 0
    else:
      chunk_kv, chunk_gate, first_window_position = cache_layer.store_compression_weights("indexer", kv, gate)

    if chunk_kv.shape[1] > 0:
      n_windows = chunk_kv.shape[1] // self.compress_rate
      ratio = self.compress_rate
      chunk_kv = chunk_kv.view(batch, n_windows, ratio, -1)
      chunk_gate = chunk_gate.view(batch, n_windows, ratio, -1) + self.position_bias.to(chunk_gate.dtype)

      # Same Ca / Cb overlap layout as the outer CSA compressor, at index_head_dim.
      new_kv = chunk_kv.new_zeros((batch, n_windows, 2 * ratio, self.head_dim))
      new_gate = chunk_gate.new_full((batch, n_windows, 2 * ratio, self.head_dim), float("-inf"))
      new_kv[:, :, ratio:] = chunk_kv[..., self.head_dim :]
      new_gate[:, :, ratio:] = chunk_gate[..., self.head_dim :]
      if n_windows > 1:
        new_kv[:, 1:, :ratio] = chunk_kv[:, :-1, :, : self.head_dim]
        new_gate[:, 1:, :ratio] = chunk_gate[:, :-1, :, : self.head_dim]
      if cache_layer is not None:
        prior_kv, prior_gate = cache_layer.update_overlap_state("indexer", chunk_kv, chunk_gate, self.head_dim)
        if prior_kv is not None:
          new_kv[:, 0, :ratio] = prior_kv.to(new_kv.dtype)
          new_gate[:, 0, :ratio] = prior_gate.to(new_gate.dtype)

      compressed = self.kv_norm((new_kv * new_gate.softmax(dim=2, dtype=torch.float32).to(new_kv.dtype)).sum(dim=2))
      positions = torch.arange(n_windows, device=compressed.device)
      positions = positions * self.compress_rate + first_window_position
      positions = positions.unsqueeze(0).expand(batch, -1)
      cos, sin = self.rotary_emb(compressed, position_ids=positions, layer_type=self.rope_layer_type)
      compressed = apply_rotary_pos_emb_PT(compressed.unsqueeze(1), cos, sin).squeeze(1)
    else:
      compressed = chunk_kv.new_zeros((batch, 0, self.head_dim))

    compressed_kv = compressed if cache_layer is None else cache_layer.update_compressor_states("indexer", compressed)

    cos_q, sin_q = self.rotary_emb(hidden_states, position_ids=position_ids, layer_type=self.rope_layer_type)
    q = self.q_b_proj(q_residual).view(batch, seq_len, -1, self.head_dim).transpose(1, 2)
    q = apply_rotary_pos_emb_PT(q, cos_q, sin_q).transpose(1, 2)

    # ReLU(q·kᵀ) * weights, then top-k
    scores = torch.matmul(q.float(), compressed_kv.transpose(-1, -2).float().unsqueeze(1))  # [B, S, H, T]
    scores = F.relu(scores) * self.softmax_scale
    weights = self.weights_proj(hidden_states).float() * self.weights_scaling  # [B, S, H]
    index_scores = (scores * weights.unsqueeze(-1)).sum(dim=2)  # [B, S, T]
    compressed_len = compressed_kv.shape[1]
    top_k = min(self.index_topk, compressed_len)

    # not all queries can attend to the compressed entries. If a query's position
    # is small than the relative position of the key (say m=4, query 2 cannot attend
    # to compressed key at position 4, because it compressed info for states at position
    # 12 to 16. Thus we need to make sure that top_k does not land in that range.
    # Picks that still point past `causal_threshold` (early queries with too few ready
    # blocks) are replaced with a `-1` sentinel that the compressor treats as invalid.
    if compressed_len > 0:
      causal_threshold = (position_ids + 1) // self.compress_rate  # [B, S]
      entry_indices = torch.arange(compressed_len, device=index_scores.device)
      future_mask = entry_indices.view(1, 1, -1) >= causal_threshold.unsqueeze(-1)  # [B, S, T]
      index_scores = index_scores.masked_fill(future_mask, float("-inf"))
      top_k_indices = index_scores.topk(top_k, dim=-1).indices  # [B, S, k]
      invalid = top_k_indices >= causal_threshold.unsqueeze(-1)
      return torch.where(invalid, torch.full_like(top_k_indices, -1), top_k_indices)

    return index_scores.topk(top_k, dim=-1).indices


class DeepseekV4CSACompressor_PT(nn.Module):
  """Compressed Sparse Attention compressor (paper §2.3.1, eqs. 9–17). Compresses
  every `compress_rate_csa` (m=4) source tokens and runs a Lightning Indexer on
  top of the compressed KV that scores queries with
  `∑_h w_{t,h} · ReLU(q_{t,h} · K^{IComp}_s)` to gather the top `index_topk`
  entries per query before they reach core attention.

  `kv_proj` / `gate_proj` / `position_bias` project to `2 * head_dim`: each
  token contributes two independent compressed series Ca and Cb stored in
  one tensor. Ca = `[..., :head_dim]` (its contribution to the *next*
  window's compressed entry), Cb = `[..., head_dim:]` (its contribution to
  the *current* window's compressed entry). Compressed entry `w` is the
  softmax-gated convex combination of window `w-1`'s Ca slice with window
  `w`'s Cb slice over `2 * compress_rate_csa` slots — width
  `2 * compress_rate_csa`, stride `compress_rate_csa`. For `w = 0` we need
  the previous window's Ca slice from the *previous forward call*; the
  cache holds it in `overlap_kv` and hands it back here. On the very first
  call (or when there is no cache) that slot stays zero-kv / `-inf`-gate,
  which gives it softmax weight 0.
  """

  rope_layer_type = "compress"

  def __init__(self, config: DeepseekV4Config):
    super().__init__()
    self.compress_rate = config.compress_rates["compressed_sparse_attention"]
    self.head_dim = config.head_dim
    self.kv_proj = nn.Linear(config.hidden_size, 2 * self.head_dim, bias=False)
    self.gate_proj = nn.Linear(config.hidden_size, 2 * self.head_dim, bias=False)
    self.position_bias = nn.Parameter(torch.empty(self.compress_rate, 2 * self.head_dim))
    self.kv_norm = DeepseekV4RMSNorm_PT(self.head_dim, eps=config.rms_norm_eps)
    self.rotary_emb = DeepseekV4RotaryEmbedding_PT(config)
    self.indexer = DeepseekV4Indexer_PT(config)

  def forward(
      self,
      hidden_states: torch.Tensor,
      q_residual: torch.Tensor,
      position_ids: torch.Tensor,
      past_key_values: Cache | None,
      layer_idx: int,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    batch, seq_len, _ = hidden_states.shape
    cache_layer: DeepseekV4CSACache = past_key_values.layers[layer_idx] if past_key_values is not None else None
    kv = self.kv_proj(hidden_states)
    gate = self.gate_proj(hidden_states)

    if cache_layer is None:
      usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
      chunk_kv, chunk_gate, first_window_position = kv[:, :usable], gate[:, :usable], 0
    else:
      chunk_kv, chunk_gate, first_window_position = cache_layer.store_compression_weights("compressor", kv, gate)

    if chunk_kv.shape[1] > 0:
      n_windows = chunk_kv.shape[1] // self.compress_rate
      ratio = self.compress_rate
      chunk_kv = chunk_kv.view(batch, n_windows, ratio, -1)
      chunk_gate = chunk_gate.view(batch, n_windows, ratio, -1) + self.position_bias.to(chunk_gate.dtype)

      # Lay out the two series in [B, n_win, 2*ratio, head_dim]: Cb
      # (`[..., head_dim:]`) goes in the second half (current window),
      # Ca of the previous window (`[..., :head_dim]`) goes in the
      # first half. Window 0's first half stays zero-kv / -inf-gate
      # (softmax weight 0) on the very first forward call; on later
      # calls the cache fills it with the saved Ca slice.
      new_kv = chunk_kv.new_zeros((batch, n_windows, 2 * ratio, self.head_dim))
      new_gate = chunk_gate.new_full((batch, n_windows, 2 * ratio, self.head_dim), float("-inf"))
      new_kv[:, :, ratio:] = chunk_kv[..., self.head_dim :]
      new_gate[:, :, ratio:] = chunk_gate[..., self.head_dim :]
      if n_windows > 1:
        new_kv[:, 1:, :ratio] = chunk_kv[:, :-1, :, : self.head_dim]
        new_gate[:, 1:, :ratio] = chunk_gate[:, :-1, :, : self.head_dim]
      if cache_layer is not None:
        prior_kv, prior_gate = cache_layer.update_overlap_state("compressor", chunk_kv, chunk_gate, self.head_dim)
        if prior_kv is not None:
          new_kv[:, 0, :ratio] = prior_kv.to(new_kv.dtype)
          new_gate[:, 0, :ratio] = prior_gate.to(new_gate.dtype)

      # Softmax in fp32 for stability (logits in bf16/fp16 can collapse pairs that
      # only differ by a small amount, especially with large window widths).
      compressed = self.kv_norm((new_kv * new_gate.softmax(dim=2, dtype=torch.float32).to(new_kv.dtype)).sum(dim=2))
      positions = torch.arange(n_windows, device=compressed.device)
      positions = positions * self.compress_rate + first_window_position
      positions = positions.unsqueeze(0).expand(batch, -1)
      cos, sin = self.rotary_emb(compressed, position_ids=positions, layer_type=self.rope_layer_type)
      compressed = apply_rotary_pos_emb_PT(compressed.unsqueeze(1), cos, sin).squeeze(1)
    else:
      compressed = chunk_kv.new_zeros((batch, 0, self.head_dim))

    if cache_layer is not None:
      compressed = cache_layer.update_compressor_states("compressor", compressed)
    compressed_kv = compressed.unsqueeze(1)

    # Lightning Indexer: gather top-`index_topk` compressed entries per query.
    # in some cases, the output index can return top-k positions that should not be attended to.
    # Ex: for query at index 5, m=4, and `index_topk=1024`, 1024 index are return but only 2 should be
    # attended to. The indexer marks the rest with `-1`; we clamp before the gather and keep the `valid`
    # to drop them from the per-query block mask afterwards.
    top_k_indices = self.indexer(hidden_states, q_residual, position_ids, past_key_values, layer_idx)  # [B, S, k]
    top_k = top_k_indices.shape[-1]
    compressed_len = compressed_kv.shape[2]
    valid = top_k_indices >= 0  # [B, S, k]
    # Flatten (B, T) into one row axis and shift picks by `b * T`, then index_select once.
    # Same kernel as an embedding lookup — cheaper than `gather` over an expanded view.
    safe_indices = top_k_indices.clamp(min=0)
    batch_offsets = (torch.arange(batch, device=compressed_kv.device) * compressed_len).view(batch, 1, 1)
    flat_indices = (safe_indices + batch_offsets).view(-1)  # [B*S*k]
    flat_kv = compressed_kv.reshape(batch * compressed_len, self.head_dim)
    gathered = flat_kv.index_select(0, flat_indices).view(batch, 1, -1, self.head_dim)  # [B, 1, S*k, D]

    # Per-query block bias: query `t` may only see the cache entries that are <= `seq_len // m`
    # and in these, only the ones marked valid by the indexer. Everything else is `-inf`.
    # While the above negated the indexer, here we apply the "causal" masking.
    block_bias = gathered.new_full((batch, 1, seq_len, seq_len, top_k), float("-inf"))
    allowed = torch.where(valid, gathered.new_zeros(()), gathered.new_full((), float("-inf")))  # [B, S, k]
    query_indices = torch.arange(seq_len, device=gathered.device)
    block_bias[:, 0, query_indices, query_indices, :] = allowed  # diagonal: q_idx == block_idx
    block_bias = block_bias.view(batch, 1, seq_len, seq_len * top_k)
    return gathered, block_bias


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
  """
  This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
  num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
  """
  batch, num_key_value_heads, slen, head_dim = hidden_states.shape
  if n_rep == 1:
    return hidden_states
  hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
  return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float | int = 0.0,
    **kwargs,
):
  key_states = repeat_kv(key, module.num_key_value_groups)
  value_states = repeat_kv(value, module.num_key_value_groups)
  attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
  if attention_mask is not None:
    attn_weights = attn_weights + attention_mask

  sinks = module.sinks.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)
  combined_logits = torch.cat([attn_weights, sinks], dim=-1)

  # This was not in the original implementation and slightly affect results; it prevents overflow in BF16/FP16
  # when training with bsz>1 we clamp max values.

  combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
  probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
  scores = probs[..., :-1]  # we drop the sink here
  attn_weights = nn.functional.dropout(scores, p=dropout, training=module.training).to(value_states.dtype)
  attn_output = torch.matmul(attn_weights, value_states)
  attn_output = attn_output.transpose(1, 2).contiguous()
  return attn_output, attn_weights


COMPRESSOR_CLASSES = {
    "sliding_attention": None,
    "compressed_sparse_attention": DeepseekV4CSACompressor_PT,
    "heavily_compressed_attention": DeepseekV4HCACompressor_PT,
}


class DeepseekV4Attention_PT(nn.Module):
  r"""
  Diff with classic attentions:
  * Shared-KV Multi-Query Attention: `num_key_value_heads = 1`; `kv_proj` projects
    directly to that single KV head and the same tensor is read as both key and
    value.
  * Partial RoPE on the first `rope_head_dim` of each head ("Partial Rotary
    Positional Embedding"). RoPE is also applied with position `-i` to the
    attention output's rope slice, so the contribution of each KV entry stays a
    function of the *relative* distance to the query.
    * Per-head learnable attention sink like gpt OSS.
    * Grouped low-rank output projection for perfs.
    * 3 different cache mechanisms, sliding, sliding+CSA, sliding+HCA.
  """

  def __init__(self, config: DeepseekV4Config, layer_idx: int):
    super().__init__()
    self.config = config
    self.layer_idx = layer_idx
    self.layer_type = config.layer_types[layer_idx]
    # Sliding-only layers use the "main" (plain θ=10000) rope; CSA/HCA layers
    # share the same yarn-scaled "compress" rope as their compressor.
    self.rope_layer_type = "main" if self.layer_type == "sliding_attention" else "compress"
    self.num_heads = config.num_attention_heads
    self.num_key_value_groups = config.num_attention_heads  # single KV head, broadcast to all
    self.head_dim = config.head_dim
    self.sliding_window = config.sliding_window
    self.attention_dropout = config.attention_dropout
    self.is_causal = True
    self.scaling = self.head_dim**-0.5

    self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
    self.q_a_norm = DeepseekV4RMSNorm_PT(config.q_lora_rank, eps=config.rms_norm_eps)
    self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False)
    self.q_b_norm = DeepseekV4UnweightedRMSNorm_PT(eps=config.rms_norm_eps)
    self.kv_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
    self.kv_norm = DeepseekV4RMSNorm_PT(self.head_dim, eps=config.rms_norm_eps)
    self.o_a_proj = DeepseekV4GroupedLinear_PT(
        self.num_heads * self.head_dim // config.o_groups, config.o_groups * config.o_lora_rank, config.o_groups
    )
    self.o_b_proj = nn.Linear(config.o_groups * config.o_lora_rank, config.hidden_size, bias=False)
    self.sinks = nn.Parameter(torch.empty(self.num_heads))
    self.compressor = COMPRESSOR_CLASSES[self.layer_type](config) if self.layer_type != "sliding_attention" else None

  def forward(
      self,
      hidden_states: torch.Tensor,
      position_embeddings: tuple[torch.Tensor, torch.Tensor],
      position_ids: torch.Tensor,
      attention_mask: torch.Tensor | None,
      past_key_values: Cache | None = None,
      **kwargs: Unpack[FlashAttentionKwargs],
  ) -> tuple[torch.Tensor, torch.Tensor | None]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    cos, sin = position_embeddings

    q_residual = self.q_a_norm(self.q_a_proj(hidden_states))
    q = self.q_b_proj(q_residual).view(*hidden_shape).transpose(1, 2)
    q = self.q_b_norm(q)
    q = apply_rotary_pos_emb_PT(q, cos, sin)

    kv = self.kv_norm(self.kv_proj(hidden_states)).view(*hidden_shape).transpose(1, 2)
    kv = apply_rotary_pos_emb_PT(kv, cos, sin)

    if past_key_values is not None:  # sliding where K==V
      kv = past_key_values.update(kv, kv, self.layer_idx)[0]

    block_bias = None
    if self.compressor is not None:  # Compressed KV (CSA or HCA)
      compressed_kv, block_bias = self.compressor(
          hidden_states, q_residual, position_ids, past_key_values, self.layer_idx
      )
      kv = torch.cat([kv, compressed_kv], dim=2)

    # compressor returns a `block_bias` carrying per-query causality + indexer
    # selections, which needs to be concatenated to the right of `attention_mask`.
    # Eager/flash interfaces consume the combined mask directly.
    if isinstance(attention_mask, torch.Tensor):
      if block_bias is not None:
        attention_mask = torch.cat([attention_mask, block_bias.to(attention_mask.dtype)], dim=-1)
      elif kv.shape[2] > attention_mask.shape[-1]:
        attention_mask = F.pad(attention_mask, (0, kv.shape[2] - attention_mask.shape[-1]), value=0.0)

    attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, eager_attention_forward
    )
    attn_output, attn_weights = attention_interface(
        self,
        q,
        kv,
        kv,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        s_aux=self.sinks,
        **kwargs,
    )

    # K=V in V4, so V picked up rope on its trailing rope slice. Apply the conjugate
    # rotation (`-sin`) at the query position to undo it on the rope slice of the
    # output before the grouped output projection mixes heads. The transpose pair is
    # just a layout fix-up: apply_rotary_pos_emb expects `[B, S, H, D]` (its
    # `unsqueeze_dim=1` adds a head-broadcast dim to cos/sin); attention gave us
    # `[B, H, S, D]`.
    attn_output = apply_rotary_pos_emb_PT(attn_output.transpose(1, 2), cos, -sin).transpose(1, 2)

    grouped = attn_output.reshape(*input_shape, self.config.o_groups, -1)
    grouped = self.o_a_proj(grouped).flatten(2)
    output = self.o_b_proj(grouped)
    return output, attn_weights


# ==============================================================================
# 2.2 PyTorch Decoder Reference Blocks
# ==============================================================================


class GradientCheckpointingLayer_PT(nn.Module):
  pass


class DeepseekV4HyperConnection_PT(nn.Module):
  r"""
  Manifold-Constrained Hyper-Connections
  (mHC) (Xie et al., 2026) to strengthen the conventional residual connections between adjacent
  Transformer blocks

  Owns the learned (`fn`, `base`, `scale`)
  parameters that turn the incoming `hc_mult` residual streams into collapse / expand
  weights. The decoder layer instantiates two of these (one for the attention site,
  one for the mlp site).

  ASCII shape guide — `B` = batch, `S` = seq, `H` = hc_mult, `D` = hidden_size::

            hidden_streams        flatten(2)        RMSNorm-rescale + F.linear(fn)
       [B, S, H, D]  ──────────►  [B, S, H*D]  ─────────────────────────────────►
                                                           mix-logits
                                                           [B, S, (2+H)*H]
                                                                  │
                          ┌───────────────────────────────────────┴──────────────────────────────┐
                          ▼                          ▼                                           ▼
                      pre logits                post logits                               comb logits
                      [B, S, H]                 [B, S, H]                                 [B, S, H, H]
                      × scale[0]                × scale[1]                                × scale[2]
                      + base[:H]                + base[H:2H]                              + base[2H:]
                      σ() + eps                 σ() + eps                                 σ() + eps
                      │                         │                                         │
                      pre                        post                                     Sinkhorn(iters)
                      (stream collapse weights)  (block-output placement)                 row/col normalise
                                                                                          │
                                                                                          comb
                                                                                          (stream mixer)
  """

  def __init__(self, config: DeepseekV4Config):
    super().__init__()
    self.hc_mult = config.hc_mult
    self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
    self.hc_eps = config.hc_eps
    self.input_norm = DeepseekV4UnweightedRMSNorm_PT(eps=config.rms_norm_eps)
    mix = (2 + self.hc_mult) * self.hc_mult
    self.fn = nn.Parameter(torch.empty(mix, self.hc_mult * config.hidden_size))
    self.base = nn.Parameter(torch.empty(mix))
    # 3 = number of outputs from the mHC mapping: `pre` (input projection
    # weights), `post` (sublayer output projection weights), `comb` (the
    # H×H residual combine matrix that gets Sinkhorn-projected onto the
    # doubly-stochastic manifold). Each output gets its own learned scale.
    self.scale = nn.Parameter(torch.empty(3))

  def forward(self, hidden_streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Compute `pre`, `post`, `comb` from the mHC mapping (paper §2.2 eq. 8).
    `comb` is projected onto the doubly-stochastic manifold via Sinkhorn-
    Knopp: starting from the sigmoid-positive matrix, alternate row and
    column normalisation for `hc_sinkhorn_iters` steps. `pre` then collapses
    the `hc_mult` parallel streams into a single sequence (input projection
    into the sublayer); `post` and `comb` are returned for the caller to
    apply on the sublayer output.
    """
    hc = self.hc_mult
    flat = self.input_norm(hidden_streams.flatten(start_dim=2).float())
    pre_w, post_w, comb_w = F.linear(flat, self.fn.float()).split([hc, hc, hc * hc], dim=-1)
    pre_b, post_b, comb_b = self.base.split([hc, hc, hc * hc])
    pre_scale, post_scale, comb_scale = self.scale.unbind(0)

    pre = torch.sigmoid(pre_w * pre_scale + pre_b) + self.hc_eps
    post = 2 * torch.sigmoid(post_w * post_scale + post_b)
    comb_logits = comb_w.view(*comb_w.shape[:-1], hc, hc) * comb_scale + comb_b.view(hc, hc)
    comb = torch.softmax(comb_logits, dim=-1) + self.hc_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
    for _ in range(self.hc_sinkhorn_iters - 1):
      comb = comb / (comb.sum(dim=-1, keepdim=True) + self.hc_eps)
      comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
    # Collapse the `hc_mult` parallel streams down to a single sequence using
    # the `pre` weights: one weighted sum across the stream axis, ready for
    # the sublayer (attn / MLP).
    collapsed = (pre.unsqueeze(-1) * hidden_streams).sum(dim=2).to(hidden_streams.dtype)
    return post, comb, collapsed


DeepseekV4UnweightedRMSNorm = DeepseekV4UnweightedRMSNorm_PT


class DeepseekV4HyperHead_PT(nn.Module):

  def __init__(self, config: DeepseekV4Config):
    super().__init__()
    self.hc_mult = config.hc_mult
    self.input_norm = DeepseekV4UnweightedRMSNorm(eps=config.rms_norm_eps)
    self.eps = config.hc_eps
    self.hc_fn = nn.Parameter(torch.empty(self.hc_mult, self.hc_mult * config.hidden_size))
    self.hc_base = nn.Parameter(torch.empty(self.hc_mult))
    self.hc_scale = nn.Parameter(torch.empty(1))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    flat = self.input_norm(x.flatten(2).float())
    mixes = F.linear(flat, self.hc_fn.float())
    pre = torch.sigmoid(mixes * self.hc_scale.float() + self.hc_base.float()) + self.eps
    return (pre.unsqueeze(-1) * x).sum(dim=2).to(x.dtype)


class DeepseekV4MLP_PT(nn.Module):

  def __init__(self, config):
    super().__init__()
    self.config = config
    self.hidden_size = config.hidden_size
    self.intermediate_size = config.intermediate_size
    self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
    self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
    self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
    self.act_fn = ACT2FN[config.hidden_act]

  def forward(self, x):
    down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    return down_proj


@use_experts_implementation
class DeepseekV4Experts_PT(nn.Module):
  """Collection of expert weights stored as 3D tensors."""

  def __init__(self, config: DeepseekV4Config):
    super().__init__()
    self.num_experts = config.num_local_experts
    self.hidden_dim = config.hidden_size
    self.intermediate_dim = config.intermediate_size
    self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
    self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
    self.act_fn = ACT2FN[config.hidden_act]
    self.limit = config.swiglu_limit

  def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor) -> torch.Tensor:
    final = torch.zeros_like(hidden_states)
    with torch.no_grad():
      mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
      hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
    for expert_idx in hit:
      expert_idx = expert_idx[0]
      if expert_idx == self.num_experts:
        continue
      top_k_pos, token_idx = torch.where(mask[expert_idx])
      current = self._apply_gate(F.linear(hidden_states[token_idx], self.gate_up_proj[expert_idx]))
      current = F.linear(current, self.down_proj[expert_idx]) * top_k_weights[token_idx, top_k_pos, None]
      final.index_add_(0, token_idx, current.to(final.dtype))
    return final

  def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
    gate, up = gate_up.chunk(2, dim=-1)
    gate = gate.clamp(max=self.limit)
    up = up.clamp(min=-self.limit, max=self.limit)
    return self.act_fn(gate) * up


class DeepseekV4TopKRouter_PT(nn.Module):

  def __init__(self, config: DeepseekV4Config):
    super().__init__()
    self.top_k = config.num_experts_per_tok
    self.num_experts = config.num_local_experts
    self.hidden_dim = config.hidden_size
    self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))
    self.score_fn = ACT2FN[config.scoring_func]
    self.routed_scaling_factor = config.routed_scaling_factor
    self.register_buffer("e_score_correction_bias", torch.zeros(self.num_experts), persistent=True)

  def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat = hidden_states.reshape(-1, self.hidden_dim)
    logits = F.linear(flat, self.weight)
    scores = self.score_fn(logits)
    indices = torch.topk(scores + self.e_score_correction_bias, self.top_k, dim=-1, sorted=False).indices
    weights = scores.gather(1, indices)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    return logits, weights * self.routed_scaling_factor, indices


class DeepseekV4HashRouter_PT(nn.Module):
  r"""
  Hash routing for the first `mlp_layer_types == "hash_moe"` MoE layers (paper
  §2.1). Expert selection is determined by a fixed `tid2eid[input_ids]` lookup —
  a frozen token-id → expert-id table — instead of a learned argmax. The learned
  gate `weight` still produces the per-expert scores that weight the selected
  experts' activations; only the *which-experts* selection is static.
  """

  def __init__(self, config: DeepseekV4Config):
    super().__init__()
    self.top_k = config.num_experts_per_tok
    self.num_experts = config.num_local_experts
    self.hidden_dim = config.hidden_size
    self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))
    self.score_fn = ACT2FN[config.scoring_func]
    self.routed_scaling_factor = config.routed_scaling_factor
    self.register_buffer("tid2eid", torch.zeros(config.vocab_size, self.top_k, dtype=torch.long), persistent=True)

  def forward(
      self, hidden_states: torch.Tensor, input_ids: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat = hidden_states.reshape(-1, self.hidden_dim)
    logits = F.linear(flat, self.weight)
    scores = self.score_fn(logits)
    indices = self.tid2eid[input_ids.reshape(-1)].long()
    weights = scores.gather(1, indices)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    return logits, weights * self.routed_scaling_factor, indices


class DeepseekV4SparseMoeBlock_PT(nn.Module):

  def __init__(self, config: DeepseekV4Config, layer_idx: int):
    super().__init__()
    self.is_hash = config.mlp_layer_types[layer_idx] == "hash_moe"
    self.gate = DeepseekV4HashRouter_PT(config) if self.is_hash else DeepseekV4TopKRouter_PT(config)
    self.experts = DeepseekV4Experts_PT(config)
    self.shared_experts = DeepseekV4MLP_PT(config)

  def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None = None) -> torch.Tensor:
    batch, seq_len, hidden_dim = hidden_states.shape
    residual = hidden_states
    flat = hidden_states.view(-1, hidden_dim)
    if self.is_hash:
      _, weights, indices = self.gate(hidden_states, input_ids)
    else:
      _, weights, indices = self.gate(hidden_states)
    routed = self.experts(flat, indices, weights).view(batch, seq_len, hidden_dim)
    return routed + self.shared_experts(residual)


class DeepseekV4DecoderLayer_PT(GradientCheckpointingLayer_PT):
  r"""DeepSeek-V4 decoder block (paper §2). Differs from a classic residual block in
  two places:

  The residual is a stack of `hc_mult` parallel streams kept in shape
  `[B, S, hc_mult, D]` throughout the block, mixed in and out via two
  :class:`DeepseekV4HyperConnection` modules (Manifold-Constrained Hyper-
  Connections / mHC, paper §2.2; Xie et al., 2026). The mHC mappings constrain
  the residual transform to the manifold of doubly-stochastic matrices via the
  Sinkhorn-Knopp projection — making signal propagation non-expansive across
  deep stacks.

  """

  def __init__(self, config: DeepseekV4Config, layer_idx: int):
    super().__init__()
    self.layer_idx = layer_idx
    self.self_attn = DeepseekV4Attention_PT(config, layer_idx)
    self.mlp = DeepseekV4SparseMoeBlock_PT(config, layer_idx)
    self.input_layernorm = DeepseekV4RMSNorm_PT(config.hidden_size, eps=config.rms_norm_eps)
    self.post_attention_layernorm = DeepseekV4RMSNorm_PT(config.hidden_size, eps=config.rms_norm_eps)
    self.attn_hc = DeepseekV4HyperConnection_PT(config)
    self.ffn_hc = DeepseekV4HyperConnection_PT(config)

  def forward(
      self,
      hidden_states: torch.Tensor,
      input_ids: torch.Tensor | None = None,
      **kwargs,
  ) -> torch.Tensor:
    # hidden_states throughout: [B, S, hc_mult, hidden].
    # `post` / `comb` come out of the HC modules in fp32 (Sinkhorn projection runs
    # in float); the .to(dtype) puts everything back to the input dtype before mixing
    # so both sites stay consistent with `hidden_states`'s entry dtype.
    # comb is consumed transposed: indexed as sum_j comb[j, k] * residual[j, d]
    # (sum over the FIRST hc axis), equivalent to comb.T @ residual. Sinkhorn
    # produces a doubly-stochastic but non-symmetric matrix, so the direction matters.
    dtype = hidden_states.dtype
    post, comb, collapsed = self.attn_hc(hidden_states)
    attn_output, _ = self.self_attn(self.input_layernorm(collapsed), **kwargs)
    hidden_states = post.to(dtype).unsqueeze(-1) * attn_output.unsqueeze(-2) + torch.matmul(
        comb.to(dtype).transpose(-1, -2), hidden_states
    )

    post, comb, collapsed = self.ffn_hc(hidden_states)
    mlp_output = self.mlp(self.post_attention_layernorm(collapsed), input_ids=input_ids)
    return post.to(dtype).unsqueeze(-1) * mlp_output.unsqueeze(-2) + torch.matmul(
        comb.to(dtype).transpose(-1, -2), hidden_states
    )


def _make_config(config_pt, B, S, D, **kwargs):
  """Return a pyconfig Config object suitable for unit tests."""
  kwargs.pop("layer_types", None)
  kwargs.pop("attention_type", None)
  num_heads = kwargs.pop("num_attention_heads", config_pt.num_attention_heads)
  overrides = {
      "run_name": "test_run",
      "enable_checkpointing": False,
      "model_name": "deepseek_v4-tiny",
      "decoder_block": "deepseek_v4",
      "dtype": "float32",
      "weight_dtype": "float32",
      "matmul_precision": "highest",
      "per_device_batch_size": B,
      "max_target_length": S,
      "max_prefill_predict_length": S,
      "emb_dim": D,
      "mhc_expansion_rate": getattr(config_pt, "hc_mult", 4),
      "hc_eps": getattr(config_pt, "hc_eps", 1e-6),
      "sinkhorn_iterations": getattr(config_pt, "hc_sinkhorn_iters", 20),
      "normalization_layer_epsilon": 1e-6,
      "head_dim": config_pt.head_dim,
      "dropout_rate": 0.0,
      "o_groups": config_pt.o_groups,
      "o_lora_rank": config_pt.o_lora_rank,
      "compress_ratios": [4] * 43,
      "compress_rope_theta": 160000.0,
      "sliding_window": config_pt.sliding_window,
      "index_n_heads": config_pt.index_n_heads,
      "index_head_dim": config_pt.index_head_dim,
      "index_topk": config_pt.index_topk,
      "base_num_query_heads": num_heads,
      "q_lora_rank": config_pt.q_lora_rank,
      "qk_rope_head_dim": getattr(config_pt, "qk_rope_head_dim", 64),
      "routed_score_func": getattr(config_pt, "scoring_func", "sqrtsoftplus"),
      "num_hash_layers": 43,
      "rope_max_timescale": config_pt.rope_theta,
      "rope_type": "default",
      "max_position_embeddings": config_pt.max_position_embeddings,
      "shard_mode": "auto",
      "debug_sharding": False,
      "scan_layers": False,
      "remat_policy": "full",
      "num_vocab_tiling": 1,
      "base_mlp_dim": config_pt.moe_intermediate_size,
      "mlp_activations": ["silu"],
      "fused_mlp": False,
      "megablox": False,
      "sparse_matmul": False,
      "use_gather_mosaic_kernel": False,
      "load_balance_loss_weight": 0.0,
      "routed_bias": False,
      "dense_init_scale": 1.0,
      "moe_expert_input_dim": -1,
      "num_experts": 16,
      "num_experts_per_tok": 1,
      "mlp_bias": False,
      "float32_gate_logits": False,
      "use_random_routing": False,
      "routed_scaling_factor": 1.0,
      "attention": "dot_product",
      "shared_experts": 1,
      "base_moe_mlp_dim": config_pt.moe_intermediate_size,
      "vocab_size": getattr(config_pt, "vocab_size", 128),
      **kwargs,
  }
  extra_args = get_decoupled_parallelism_overrides()
  merged = {**overrides, **extra_args}
  cfg = pyconfig.initialize([sys.argv[0], get_test_config_path()], override_model_config=True, **merged)
  if not hasattr(cfg, "trainable_position_size"):
    cfg.trainable_position_size = 0
  if not hasattr(cfg, "original_max_position_embeddings"):
    cfg.original_max_position_embeddings = cfg.max_position_embeddings
  return cfg


class DeepSeekV4ParityTest(unittest.TestCase):

  def test_unweighted_rms_norm_parity(self):
    # Generate identical input vectors across frameworks.
    # Setting a deterministic seed ensures mathematically identical input distributions.
    np.random.seed(42)
    x_np = np.random.randn(4, 64, 512).astype(np.float32)

    x_torch = torch.tensor(x_np)
    x_jax = jnp.array(x_np)

    # Execute PyTorch reference unweighted RMS normalization.
    # Unweighted RMSNorm contains no learnable parameters. Epsilon is set to 1e-6.
    torch_model = DeepseekV4UnweightedRMSNorm_PT(eps=1e-6)
    out_torch = torch_model(x_torch).detach().numpy()

    # Execute JAX equivalent target unweighted RMS normalization.
    # Target module instantiated from top-level imports to optimize namespace lookup.
    jax_model = DeepSeekV4UnweightedRMSNorm(eps=1e-6)
    out_jax = jax_model(x_jax)

    # Compare outputs within numerical precision tolerance limits.
    np.testing.assert_allclose(out_torch, out_jax, atol=1e-5, rtol=1e-5)

  def test_rms_norm_parity(self):
    # Generate identical input and scaling weights across frameworks.
    # Identical weight assignments ensure learnable scale features are verified equivalently.
    np.random.seed(42)
    x_np = np.random.randn(4, 64, 512).astype(np.float32)
    weight_np = np.random.randn(512).astype(np.float32)

    x_torch = torch.tensor(x_np)
    x_jax = jnp.array(x_np)

    # Execute PyTorch reference RMS normalization.
    torch_model = DeepseekV4RMSNorm_PT(hidden_size=512, eps=1e-6)
    # Copy matching scale weights to the reference model parameter state.
    torch_model.weight.data.copy_(torch.tensor(weight_np))
    out_torch = torch_model(x_torch).detach().numpy()

    # Execute JAX equivalent target RMS normalization.
    # JAX model state parameters are explicitly updated to match the generated weights.
    jax_model = DeepSeekV4RMSNorm(hidden_size=512, eps=1e-6)
    jax_model.weight.value = jnp.array(weight_np)
    out_jax = jax_model(x_jax)

    # Assert numerical parity between output states.
    np.testing.assert_allclose(out_torch, out_jax, atol=1e-5, rtol=1e-5)

  def test_rotary_embedding_parity(self):
    # Generate identical input sequences, positional values, and batch layouts.
    # The sequence is constructed to test interleaved rotary mappings and broadcasting.
    np.random.seed(42)
    B, S, H, D = 4, 64, 8, 512
    x_np = np.random.randn(B, S, H, D).astype(np.float32)
    position_ids_np = np.random.randint(0, 1000, size=(B, S)).astype(np.int64)

    x_torch = torch.tensor(x_np)
    position_ids_torch = torch.tensor(position_ids_np)

    x_jax = jnp.array(x_np)
    position_ids_jax = jnp.array(position_ids_np)

    # Setup configuration parameters.
    config = DeepseekV4Config()

    # Execute PyTorch reference rotary embeddings.
    # PyTorch default layout is [B, H, S, D], which requires inputs to be transposed
    # prior to calling the embedding layer, then transposed back to native [B, S, H, D].
    torch_emb = DeepseekV4RotaryEmbedding_PT(config)
    cos_torch, sin_torch = torch_emb(x_torch, position_ids_torch, layer_type="main")

    x_torch_transposed = x_torch.transpose(1, 2)  # [B, H, S, D]
    out_torch = apply_rotary_pos_emb_PT(x_torch_transposed, cos_torch, sin_torch, unsqueeze_dim=1)
    out_torch_np = out_torch.transpose(1, 2).detach().numpy()  # [B, S, H, D]

    # Execute JAX equivalent target rotary embeddings.
    # The target JAX layer operates natively on [B, S, H, D] layouts, applying
    # dimensional unsqueezing at axis 2 to broadcast across heads.
    jax_emb = DeepSeekV4RotaryEmbedding(head_dim=D, partial_rotary_factor=64.0 / 512.0, rope_theta=10000.0)
    cos_jax, sin_jax = jax_emb(x_jax, position_ids_jax)

    # Execute JAX target application.
    out_jax = apply_rotary_pos_emb(x_jax, cos_jax, sin_jax, unsqueeze_dim=2)
    out_jax_np = np.array(out_jax)

    # Compare both the intermediate cos/sin sinusoids and the final rotated values.
    np.testing.assert_allclose(cos_torch.detach().numpy(), cos_jax, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(sin_torch.detach().numpy(), sin_jax, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(out_torch_np, out_jax_np, atol=1e-5, rtol=1e-5)

  def test_grouped_linear_parity(self):
    # Generate identical input arrays and weight matrices across frameworks.
    # Segmented group dimensions and feature boundaries map directly.
    np.random.seed(42)
    B, S, g, i, o = 2, 8, 4, 128, 256
    out_features_per_group = o // g

    # Input shape layout is [B, S, g, i]
    x_np = np.random.randn(B, S, g, i).astype(np.float32)
    # PyTorch standard linear weight layout is [o, i]
    weight_np = np.random.randn(o, i).astype(np.float32)

    x_torch = torch.tensor(x_np)
    x_jax = jnp.array(x_np)

    # Execute PyTorch reference grouped linear block projection.
    torch_model = DeepseekV4GroupedLinear_PT(
        in_features_per_group=i,
        out_features=o,
        n_groups=g,
        bias=False,
    )
    torch_model.weight.data.copy_(torch.tensor(weight_np))
    out_torch = torch_model(x_torch).detach().numpy()

    # Execute JAX equivalent target grouped linear block projection.
    # JAX weights are initialized using the deterministic key context.
    rngs = nnx.Rngs(42)
    jax_model = DeepSeekGroupedLinear(
        in_features_per_group=i,
        out_features=o,
        n_groups=g,
        rngs=rngs,
    )

    # Copy the reshaped and transposed weight matrix matching PyTorch's view mapping
    # [o, i] -> [g, o_g, i] -> [g, i, o_g]
    jax_model.kernel.value = jnp.array(weight_np.reshape(g, out_features_per_group, i).transpose(0, 2, 1))
    out_jax = jax_model(x_jax)

    # Verify numerical output parity between frameworks
    np.testing.assert_allclose(out_torch, out_jax, atol=1e-5, rtol=1e-5)

  def test_hca_compressor_parity(self):
    # Configure deterministic seeds for parity reproducibility
    np.random.seed(42)
    B, S, D, D_head, compress_rate = 2, 128, 512, 256, 32

    # hidden_states: [B, S, D]
    x_np = np.random.randn(B, S, D).astype(np.float32)
    positions_np = np.broadcast_to(np.arange(S)[np.newaxis, :], (B, S)).astype(np.int32)

    x_torch = torch.tensor(x_np)
    positions_torch = torch.tensor(positions_np, dtype=torch.long)

    x_jax = jnp.array(x_np)
    positions_jax = jnp.array(positions_np)

    # Initialize PyTorch configurations matching parameter spaces
    config = DeepseekV4Config()
    config.hidden_size = D
    config.head_dim = D_head
    config.qk_rope_head_dim = int(D_head * (64 / 512))
    config.compress_rates["heavily_compressed_attention"] = compress_rate
    config.rms_norm_eps = 1e-6

    # Initialize PyTorch HCA Compressor model
    torch_model = DeepseekV4HCACompressor_PT(config)
    torch.nn.init.normal_(torch_model.position_bias, std=0.02)

    # Map JAX layer using matching parameters
    jax_config = _make_config(config, B, S, D, compress_ratios=[compress_rate] * 43)

    rngs = nnx.Rngs(42)
    jax_model = attention_compressed.HCACompressor(
        hidden_size=D,
        head_dim=D_head,
        config=jax_config,
        layer_idx=0,
        eps=1e-6,
        rngs=rngs,
    )

    # Set JAX parameters identical to PyTorch states to guarantee numerical parity
    jax_model.kv_proj.kernel[...] = jnp.array(torch_model.kv_proj.weight.detach().numpy().T)
    jax_model.gate_proj.kernel[...] = jnp.array(torch_model.gate_proj.weight.detach().numpy().T)
    jax_model.position_bias[...] = jnp.array(torch_model.position_bias.detach().numpy())
    jax_model.kv_norm.weight[...] = jnp.array(torch_model.kv_norm.weight.detach().numpy())

    # Execute PyTorch stateless compressor path
    # Shape out_torch: [B, 1, W, D_head] where W = S // compress_rate = 4
    out_torch, block_bias_torch = torch_model(
        hidden_states=x_torch,
        q_residual=None,
        position_ids=positions_torch,
        past_key_values=None,
        layer_idx=0,
    )
    out_torch = out_torch.detach().numpy()
    if block_bias_torch is not None:
      block_bias_torch = block_bias_torch.detach().numpy()

    # Execute JAX equivalent stateless compressor path
    # Shape out_jax: [B, 1, W, D_head]
    out_jax, block_bias_jax = jax_model(
        hidden_states=x_jax,
        position_ids=positions_jax,
    )
    out_jax_np = np.array(out_jax)
    if block_bias_jax is not None:
      block_bias_jax = np.array(block_bias_jax)

    # Validate bit-accurate state outputs matching numerical tolerance thresholds
    np.testing.assert_allclose(out_torch, out_jax_np, atol=1e-5, rtol=1e-5)
    if block_bias_torch is not None or block_bias_jax is not None:
      np.testing.assert_allclose(block_bias_torch, block_bias_jax, atol=1e-5, rtol=1e-5)

  def test_indexer_parity(self):
    np.random.seed(42)
    B, S, D, D_rank = 2, 128, 512, 1024
    num_heads, index_head_dim, index_topk, compress_rate = 64, 128, 8, 4

    # hidden_states: [B, S, D]
    x_np = np.random.randn(B, S, D).astype(np.float32)
    # q_residual: [B, S, D_rank]
    q_res_np = np.random.randn(B, S, D_rank).astype(np.float32)
    # position_ids: [B, S]
    positions_np = np.broadcast_to(np.arange(S)[np.newaxis, :], (B, S)).astype(np.int32)

    x_torch = torch.tensor(x_np)
    q_res_torch = torch.tensor(q_res_np)
    positions_torch = torch.tensor(positions_np, dtype=torch.long)

    x_jax = jnp.array(x_np)
    q_res_jax = jnp.array(q_res_np)
    positions_jax = jnp.array(positions_np)

    # Initialize PyTorch indexer configurations
    config = DeepseekV4Config()
    config.hidden_size = D
    config.q_lora_rank = D_rank
    config.index_n_heads = num_heads
    config.index_head_dim = index_head_dim
    config.index_topk = index_topk
    config.compress_rates["compressed_sparse_attention"] = compress_rate
    config.rms_norm_eps = 1e-6

    torch_model = DeepseekV4Indexer_PT(config)
    torch.nn.init.normal_(torch_model.position_bias, std=0.02)

    # Map JAX equivalent Indexer module
    jax_config = _make_config(
        config,
        B,
        S,
        D,
        index_n_heads=num_heads,
        index_head_dim=index_head_dim,
        index_topk=index_topk,
        compress_ratios=[compress_rate] * 43,
    )

    rngs = nnx.Rngs(42)
    jax_model = attention_compressed.DeepSeekV4Indexer(
        hidden_size=D,
        q_lora_rank=D_rank,
        config=jax_config,
        layer_idx=0,
        eps=1e-6,
        rngs=rngs,
    )

    # Synchronize parameter values
    jax_model.kv_proj.kernel[...] = jnp.array(torch_model.kv_proj.weight.detach().numpy().T)
    jax_model.gate_proj.kernel[...] = jnp.array(torch_model.gate_proj.weight.detach().numpy().T)
    jax_model.position_bias[...] = jnp.array(torch_model.position_bias.detach().numpy())
    jax_model.kv_norm.weight[...] = jnp.array(torch_model.kv_norm.weight.detach().numpy())
    jax_model.q_b_proj.kernel[...] = jnp.array(torch_model.q_b_proj.weight.detach().numpy().T)
    jax_model.weights_proj.kernel[...] = jnp.array(torch_model.weights_proj.weight.detach().numpy().T)

    # Execute models
    out_torch = (
        torch_model(
            hidden_states=x_torch,
            q_residual=q_res_torch,
            position_ids=positions_torch,
            past_key_values=None,
            layer_idx=0,
        )
        .detach()
        .numpy()
    )

    out_jax = jax_model(
        hidden_states=x_jax,
        q_residual=q_res_jax,
        position_ids=positions_jax,
    )
    out_jax_np = np.array(out_jax)

    # Check mathematical equivalence of top-k selection indices
    np.testing.assert_allclose(out_torch, out_jax_np, atol=1e-5, rtol=1e-5)

  def test_csa_compressor_parity(self):
    np.random.seed(42)
    B, S, D, D_rank, D_head = 2, 128, 512, 1024, 256
    num_heads, index_head_dim, index_topk, compress_rate = 64, 128, 8, 4

    # Inputs
    x_np = np.random.randn(B, S, D).astype(np.float32)
    q_res_np = np.random.randn(B, S, D_rank).astype(np.float32)
    positions_np = np.broadcast_to(np.arange(S)[np.newaxis, :], (B, S)).astype(np.int32)

    x_torch = torch.tensor(x_np)
    q_res_torch = torch.tensor(q_res_np)
    positions_torch = torch.tensor(positions_np, dtype=torch.long)

    x_jax = jnp.array(x_np)
    q_res_jax = jnp.array(q_res_np)
    positions_jax = jnp.array(positions_np)

    # Configurations
    config = DeepseekV4Config()
    config.hidden_size = D
    config.q_lora_rank = D_rank
    config.head_dim = D_head
    config.qk_rope_head_dim = int(D_head * (64 / 512))
    config.index_n_heads = num_heads
    config.index_head_dim = index_head_dim
    config.index_topk = index_topk
    config.compress_rates["compressed_sparse_attention"] = compress_rate
    config.rms_norm_eps = 1e-6

    torch_model = DeepseekV4CSACompressor_PT(config)
    torch.nn.init.normal_(torch_model.position_bias, std=0.02)
    torch.nn.init.normal_(torch_model.indexer.position_bias, std=0.02)

    jax_config = _make_config(
        config,
        B,
        S,
        D,
        index_n_heads=num_heads,
        index_head_dim=index_head_dim,
        index_topk=index_topk,
        compress_ratios=[compress_rate] * 43,
    )

    rngs = nnx.Rngs(42)
    jax_model = attention_compressed.CSACompressor(
        hidden_size=D,
        q_lora_rank=D_rank,
        head_dim=D_head,
        config=jax_config,
        layer_idx=0,
        eps=1e-6,
        rngs=rngs,
    )

    # Synchronize outer compressor states
    jax_model.kv_proj.kernel[...] = jnp.array(torch_model.kv_proj.weight.detach().numpy().T)
    jax_model.gate_proj.kernel[...] = jnp.array(torch_model.gate_proj.weight.detach().numpy().T)
    jax_model.position_bias[...] = jnp.array(torch_model.position_bias.detach().numpy())
    jax_model.kv_norm.weight[...] = jnp.array(torch_model.kv_norm.weight.detach().numpy())

    # Synchronize inner indexer states
    jax_model.indexer.kv_proj.kernel[...] = jnp.array(torch_model.indexer.kv_proj.weight.detach().numpy().T)
    jax_model.indexer.gate_proj.kernel[...] = jnp.array(torch_model.indexer.gate_proj.weight.detach().numpy().T)
    jax_model.indexer.position_bias[...] = jnp.array(torch_model.indexer.position_bias.detach().numpy())
    jax_model.indexer.kv_norm.weight[...] = jnp.array(torch_model.indexer.kv_norm.weight.detach().numpy())
    jax_model.indexer.q_b_proj.kernel[...] = jnp.array(torch_model.indexer.q_b_proj.weight.detach().numpy().T)
    jax_model.indexer.weights_proj.kernel[...] = jnp.array(torch_model.indexer.weights_proj.weight.detach().numpy().T)

    # Execute
    out_torch, block_bias_torch = torch_model(
        hidden_states=x_torch,
        q_residual=q_res_torch,
        position_ids=positions_torch,
        past_key_values=None,
        layer_idx=0,
    )
    out_torch = out_torch.detach().numpy()
    if block_bias_torch is not None:
      block_bias_torch = block_bias_torch.detach().numpy()

    out_jax, block_bias_jax = jax_model(
        hidden_states=x_jax,
        q_residual=q_res_jax,
        position_ids=positions_jax,
    )
    out_jax_np = np.array(out_jax)
    if block_bias_jax is not None:
      block_bias_jax = np.array(block_bias_jax)

    # Diagnose indexer parity
    topk_torch = (
        torch_model.indexer(
            hidden_states=x_torch,
            q_residual=q_res_torch,
            position_ids=positions_torch,
            past_key_values=None,
            layer_idx=0,
        )
        .detach()
        .numpy()
    )

    topk_jax = jax_model.indexer(
        hidden_states=x_jax,
        q_residual=q_res_jax,
        position_ids=positions_jax,
    )
    topk_jax_np = np.array(topk_jax)

    np.testing.assert_allclose(topk_torch, topk_jax_np, atol=1e-5, rtol=1e-5)

    # Check complete parity of gathered/indexed keys
    np.testing.assert_allclose(out_torch, out_jax_np, atol=1e-5, rtol=1e-5)
    if block_bias_torch is not None or block_bias_jax is not None:
      np.testing.assert_allclose(block_bias_torch, block_bias_jax, atol=1e-5, rtol=1e-5)

  def test_attention_layer_parity(self):
    np.random.seed(42)
    B, S, D, D_rank, D_head, num_heads = 2, 128, 512, 1024, 256, 16
    compress_rate = 32

    # Inputs
    x_np = np.random.randn(B, S, D).astype(np.float32)
    position_ids_np = np.broadcast_to(np.arange(S)[np.newaxis, :], (B, S)).astype(np.int32)

    x_torch = torch.tensor(x_np)
    position_ids_torch = torch.tensor(position_ids_np, dtype=torch.long)

    x_jax = jnp.array(x_np)
    position_ids_jax = jnp.array(position_ids_np)

    # Configurations
    config = DeepseekV4Config()
    config.hidden_size = D
    config.q_lora_rank = D_rank
    config.head_dim = D_head
    config.qk_rope_head_dim = int(D_head * (64 / 512))
    config.num_attention_heads = num_heads
    config.num_key_value_heads = 1
    config.compress_rates["heavily_compressed_attention"] = compress_rate
    config.rms_norm_eps = 1e-6
    config.layer_types = ["heavily_compressed_attention"] * 10

    # Generate reference position embeddings (cos, sin)
    torch_emb = DeepseekV4RotaryEmbedding_PT(config)
    cos_torch, sin_torch = torch_emb(x_torch, position_ids_torch, layer_type="compress")

    cos_jax = jnp.array(cos_torch.detach().numpy())
    sin_jax = jnp.array(sin_torch.detach().numpy())

    # Initialize PyTorch and JAX coordinate attention layers
    torch_model = DeepseekV4Attention_PT(config, layer_idx=0)
    torch.nn.init.normal_(torch_model.sinks, std=0.02)
    if torch_model.compressor is not None:
      torch.nn.init.normal_(torch_model.compressor.position_bias, std=0.02)

    jax_config = _make_config(
        config,
        B,
        S,
        D,
        num_attention_heads=num_heads,
        compress_ratios=[compress_rate] * 10,
        layer_types=["heavily_compressed_attention"] * 10,
        o_groups=config.o_groups,
        o_lora_rank=config.o_lora_rank,
        # Disabling hardware MXU grid alignment padding (sa_block_kv=0).
        # By default, AttentionOp enforces sa_block_kv=512 grid bounds, automatically padding trailing sequence length
        # (S=128 + W=32 = 160) to 512 with zero vectors. Under dot-product attention without explicit causal padding masks
        # (attention_mask=None), Softmax evaluates unmasked zero vectors to positive probability weightings (e^{0.0} = 1.0),
        # artificially inflating the local exponential normalizer sum denominator and distorting numerical parity bounds.
        sa_block_kv=0,
    )

    devices = jax.devices()
    mesh = Mesh(np.array(devices), ("data",))
    rngs = nnx.Rngs(42)
    jax_model = attention_compressed.DeepSeekV4Attention(
        hidden_size=D,
        q_lora_rank=D_rank,
        head_dim=D_head,
        num_heads=num_heads,
        config=jax_config,
        layer_idx=0,
        mesh=mesh,
        eps=1e-6,
        attention_type="heavily_compressed_attention",
        rngs=rngs,
    )

    # Copy projections and normalize weights from PyTorch to JAX
    jax_model.q_a_proj.kernel[...] = jnp.array(torch_model.q_a_proj.weight.detach().numpy().T)
    jax_model.q_a_norm.weight[...] = jnp.array(torch_model.q_a_norm.weight.detach().numpy())
    jax_model.q_b_proj.kernel[...] = jnp.array(torch_model.q_b_proj.weight.detach().numpy().T)

    jax_model.kv_proj.kernel[...] = jnp.array(torch_model.kv_proj.weight.detach().numpy().T)
    jax_model.kv_norm.weight[...] = jnp.array(torch_model.kv_norm.weight.detach().numpy())

    # Handle Grouped Output Projection mapping
    w_o_a_np = torch_model.o_a_proj.weight.detach().numpy()
    in_features_per_group = num_heads * D_head // config.o_groups
    w_o_a_np = w_o_a_np.reshape(config.o_groups, -1, in_features_per_group).transpose(0, 2, 1)
    jax_model.o_a_proj.kernel[...] = jnp.array(w_o_a_np)

    jax_model.o_b_proj.kernel[...] = jnp.array(torch_model.o_b_proj.weight.detach().numpy().T)
    jax_model.sinks[...] = jnp.array(torch_model.sinks.detach().numpy())

    # Copy Compressor weights if present
    if torch_model.compressor is not None:
      jax_model.compressor.kv_proj.kernel[...] = jnp.array(torch_model.compressor.kv_proj.weight.detach().numpy().T)
      jax_model.compressor.gate_proj.kernel[...] = jnp.array(torch_model.compressor.gate_proj.weight.detach().numpy().T)
      jax_model.compressor.position_bias[...] = jnp.array(torch_model.compressor.position_bias.detach().numpy())
      jax_model.compressor.kv_norm.weight[...] = jnp.array(torch_model.compressor.kv_norm.weight.detach().numpy())

    # Execute PyTorch attention layer
    out_torch, _ = torch_model(
        hidden_states=x_torch,
        position_embeddings=(cos_torch, sin_torch),
        position_ids=position_ids_torch,
        attention_mask=None,
    )
    out_torch_np = out_torch.detach().numpy()

    # Execute JAX attention layer
    out_jax, _ = jax_model(
        hidden_states=x_jax,
        cos=cos_jax,
        sin=sin_jax,
        position_ids=position_ids_jax,
        attention_mask=None,
    )
    out_jax_np = np.array(out_jax)

    # Check complete numerical parity of coordination attention layers
    np.testing.assert_allclose(out_torch_np, out_jax_np, atol=1e-5, rtol=1e-5)

  def test_topk_router_parity(self):
    # Generate deterministic random inputs for the router comparison.
    np.random.seed(42)
    B, S, D = 2, 8, 64
    num_experts = 16
    top_k = 6
    routed_scaling_factor = 1.5

    hidden_states_np = np.random.randn(B, S, D).astype(np.float32)
    weight_np = np.random.randn(num_experts, D).astype(np.float32)
    e_score_correction_bias_np = np.random.randn(num_experts).astype(np.float32)

    # 1. Setup PyTorch Reference Router
    config_pt = DeepseekV4Config(
        num_experts_per_tok=top_k,
        num_local_experts=num_experts,
        hidden_size=D,
        routed_scaling_factor=routed_scaling_factor,
        scoring_func="sqrtsoftplus",
    )
    py_router = DeepseekV4TopKRouter_PT(config_pt)
    py_router.weight.data = torch.tensor(weight_np)
    py_router.e_score_correction_bias.data = torch.tensor(e_score_correction_bias_np)

    # Run forward on PyTorch router
    hidden_states_torch = torch.tensor(hidden_states_np)
    py_logits, py_weights, py_indices = py_router(hidden_states_torch)

    # 2. Setup JAX/Flax NNX Equivalent Router
    class MockJaxConfig:

      def __init__(self):
        self.num_experts_per_tok = top_k
        self.num_experts = num_experts
        self.emb_dim = D
        self.moe_expert_input_dim = D
        self.routed_scaling_factor = routed_scaling_factor
        self.routed_score_func = "sqrtsoftplus"
        self.dtype = jnp.float32
        self.weight_dtype = jnp.float32

    config_jax = MockJaxConfig()
    rngs = nnx.Rngs(42)
    jax_router = DeepSeekV4TopKRouter(config=config_jax, mesh=None, rngs=rngs)

    # Copy weight and correction bias parameters using Flax NNX attribute variable assignments.
    jax_router.kernel[...] = jnp.array(weight_np.T)
    jax_router.e_score_correction_bias[...] = jnp.array(e_score_correction_bias_np)

    # Run forward on JAX router
    hidden_states_jax = jnp.array(hidden_states_np)
    jax_logits, jax_weights, jax_indices = jax_router(hidden_states_jax)

    # 3. Parity assertions
    # Compare raw logits directly.
    np.testing.assert_allclose(py_logits.detach().numpy(), jax_logits, atol=1e-5, rtol=1e-5)

    # Symmetrically, the order of the chosen top-k experts can differ (unsorted vs JAX sort).
    # Sort both index selections and weight selections row-by-row (token-by-token) before comparison.
    py_ind_np = py_indices.numpy()
    py_w_np = py_weights.detach().numpy()
    jax_ind_np = np.array(jax_indices)
    jax_w_np = np.array(jax_weights)

    # Sort index arrays row-by-row, and order the corresponding weights array matching the index sort order.
    for i in range(py_ind_np.shape[0]):
      py_sort_order = np.argsort(py_ind_np[i])
      py_ind_np[i] = py_ind_np[i][py_sort_order]
      py_w_np[i] = py_w_np[i][py_sort_order]

      jax_sort_order = np.argsort(jax_ind_np[i])
      jax_ind_np[i] = jax_ind_np[i][jax_sort_order]
      jax_w_np[i] = jax_w_np[i][jax_sort_order]

    # Assert sorted indices and weights are mathematically identical!
    np.testing.assert_array_equal(jax_ind_np, py_ind_np)
    np.testing.assert_allclose(py_w_np, jax_w_np, atol=1e-5, rtol=1e-5)

  def test_hash_router_parity(self):
    # Generate deterministic random inputs for static hash router comparison.
    np.random.seed(42)
    B, S, D = 2, 8, 64
    num_experts = 16
    top_k = 6
    routed_scaling_factor = 1.5
    vocab_size = 32

    hidden_states_np = np.random.randn(B, S, D).astype(np.float32)
    input_ids_np = np.random.randint(0, vocab_size, size=(B, S)).astype(np.int32)
    weight_np = np.random.randn(num_experts, D).astype(np.float32)
    tid2eid_np = np.random.randint(0, num_experts, size=(vocab_size, top_k)).astype(np.int32)

    # 1. Setup PyTorch Reference Router
    config_pt = DeepseekV4Config(
        num_experts_per_tok=top_k,
        num_local_experts=num_experts,
        hidden_size=D,
        routed_scaling_factor=routed_scaling_factor,
        vocab_size=vocab_size,
        scoring_func="sqrtsoftplus",
    )
    py_router = DeepseekV4HashRouter_PT(config_pt)
    py_router.weight.data = torch.tensor(weight_np)
    py_router.tid2eid.data = torch.tensor(tid2eid_np).long()

    # Run forward on PyTorch router
    hidden_states_torch = torch.tensor(hidden_states_np)
    input_ids_torch = torch.tensor(input_ids_np)
    py_logits, py_weights, py_indices = py_router(hidden_states_torch, input_ids_torch)

    # 2. Setup JAX/Flax NNX Equivalent Router
    class MockJaxConfig:

      def __init__(self):
        self.num_experts_per_tok = top_k
        self.num_experts = num_experts
        self.emb_dim = D
        self.moe_expert_input_dim = D
        self.routed_scaling_factor = routed_scaling_factor
        self.routed_score_func = "sqrtsoftplus"
        self.vocab_size = vocab_size
        self.dtype = jnp.float32
        self.weight_dtype = jnp.float32

    config_jax = MockJaxConfig()
    rngs = nnx.Rngs(42)
    jax_router = DeepSeekV4HashRouter(config=config_jax, mesh=None, rngs=rngs)

    # Copy weight and lookup table parameter states using clean Flax NNX assignments.
    jax_router.kernel[...] = jnp.array(weight_np.T)
    jax_router.tid2eid[...] = jnp.array(tid2eid_np, dtype=jnp.int32)

    # Run forward on JAX router
    hidden_states_jax = jnp.array(hidden_states_np)
    input_ids_jax = jnp.array(input_ids_np)
    jax_logits, jax_weights, jax_indices = jax_router(hidden_states_jax, input_ids_jax)

    # 3. Parity assertions
    # Logits, weights, and selected index array checks.
    np.testing.assert_allclose(py_logits.detach().numpy(), jax_logits, atol=1e-5, rtol=1e-5)
    np.testing.assert_array_equal(jax_indices, py_indices.numpy())
    np.testing.assert_allclose(py_weights.detach().numpy(), jax_weights, atol=1e-5, rtol=1e-5)

  def test_hyperhead_parity(self):
    # Verify isolated parametric collapse HyperHead parity E2E!
    np.random.seed(42)
    B, S, k, D = 2, 4, 4, 128
    x_np = np.random.randn(B, S, k, D).astype(np.float32)
    hc_fn_np = np.random.randn(k, k * D).astype(np.float32)
    hc_base_np = np.random.randn(k).astype(np.float32)
    hc_scale_np = np.random.randn(1).astype(np.float32)

    config_pt = DeepseekV4Config(
        hc_mult=k,
        hidden_size=D,
        rms_norm_eps=1e-6,
        hc_eps=1e-6,
    )
    py_head = DeepseekV4HyperHead_PT(config_pt)
    py_head.hc_fn.data = torch.tensor(hc_fn_np)
    py_head.hc_base.data = torch.tensor(hc_base_np)
    py_head.hc_scale.data = torch.tensor(hc_scale_np)

    # Run forward on PyTorch reference
    x_torch = torch.tensor(x_np)
    out_torch = py_head(x_torch)

    # Setup JAX DeepSeekV4HyperHead equivalent NNX module
    class MockJaxConfig:

      def __init__(self):
        self.emb_dim = D
        self.mhc_expansion_rate = k
        self.hc_eps = 1e-6
        self.normalization_layer_epsilon = 1e-6
        self.dtype = jnp.float32
        self.weight_dtype = jnp.float32
        self.matmul_precision = "default"

    config_jax = MockJaxConfig()
    rngs = nnx.Rngs(42)
    jax_head = DeepSeekV4HyperHead(config=config_jax, rngs=rngs)

    # Copy weight matrices and parameter states cleanly
    # Shape mappings:
    # PyTorch: hc_fn has shape [k, k * D], mixes = F.linear(flat, hc_fn) -> flat @ hc_fn.T
    # JAX: hc_fn has shape [k * D, k], mixes = flat @ hc_fn
    # Therefore, JAX weight = PyTorch weight.T
    jax_head.hc_fn[...] = jnp.array(hc_fn_np.T)
    jax_head.hc_base[...] = jnp.array(hc_base_np)
    jax_head.hc_scale[...] = jnp.array(hc_scale_np)

    # Run forward passes on identical random batch stream inputs [B, S, k, D]
    x_jax = jnp.array(x_np)
    out_jax = jax_head(x_jax)

    # Assert bit-accurate numerical parity down to atol=1e-5 E2E!
    np.testing.assert_allclose(out_torch.detach().numpy(), np.array(out_jax), atol=1e-5, rtol=1e-5)

  def test_full_model_stack_parity(self):
    """Verifies complete, scannable multi-layer decoder stack E2E logits parity.

    This E2E test validates that:
      1. Parallel stream transformations [B, S, hc_mult, D] sequence correctly.
      2. Manifold-Constrained Hyper-Connections (mHC) perform identical Sinkhorn
         projections across frameworks.
      3. The JAX scanned compiler (scan_layers = True) constructs and executes
         identical stacked loop parameters compared to unrolled modes (scan_layers = False).
    """
    np.random.seed(42)
    B, S, D, H_mult, vocab_size, num_layers = 2, 8, 128, 4, 32, 3

    # Generate identical input token IDs across frameworks
    input_ids_np = np.random.randint(0, vocab_size, size=(B, S)).astype(np.int32)
    position_ids_np = np.broadcast_to(np.arange(S)[np.newaxis, :], (B, S)).astype(np.int32)
    input_ids_torch = torch.tensor(input_ids_np).long()
    position_ids_torch = torch.tensor(position_ids_np).long()
    input_ids_jax = jnp.array(input_ids_np)

    # 1. Build identical configuration configurations
    config_pt = DeepseekV4Config()
    config_pt.hidden_size = D
    config_pt.intermediate_size = 64
    config_pt.moe_intermediate_size = 64
    config_pt.hc_mult = H_mult
    config_pt.hc_sinkhorn_iters = 8
    config_pt.rms_norm_eps = 1e-6
    config_pt.vocab_size = vocab_size
    config_pt.num_hash_layers = 2
    config_pt.num_local_experts = 4
    config_pt.num_experts_per_tok = 2
    config_pt.num_attention_heads = 4
    config_pt.num_key_value_heads = 1
    config_pt.head_dim = 32
    config_pt.qk_rope_head_dim = 32
    config_pt.rope_parameters["main"]["partial_rotary_factor"] = 1.0
    config_pt.rope_parameters["compress"]["partial_rotary_factor"] = 1.0
    config_pt.q_lora_rank = 64
    config_pt.o_groups = 2
    config_pt.o_lora_rank = 64
    config_pt.index_n_heads = 4
    config_pt.index_head_dim = 32
    config_pt.index_topk = 2
    config_pt.layer_types = ["compressed_sparse_attention", "heavily_compressed_attention", "compressed_sparse_attention"]
    config_pt.mlp_layer_types = ["hash_moe", "hash_moe", "topk_moe"]

    class DeepseekV4DecoderStack_PT(nn.Module):

      def __init__(self, config: DeepseekV4Config, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([DeepseekV4DecoderLayer_PT(config, lyr) for lyr in range(num_layers)])
        self.hc_head = DeepseekV4HyperHead_PT(config)
        self.norm = DeepseekV4RMSNorm_PT(config.hidden_size, eps=config.rms_norm_eps)
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.logits_dense = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.rotary_emb = DeepseekV4RotaryEmbedding_PT(config)

      def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        y = self.embeddings(input_ids)
        y = y.unsqueeze(2).expand(-1, -1, 4, -1)
        cos, sin = self.rotary_emb(y[:, :, 0, :], position_ids, layer_type="compress")
        for layer in self.layers:
          y = layer(
              y, input_ids=input_ids, position_embeddings=(cos, sin), position_ids=position_ids, attention_mask=None
          )
        collapsed = self.hc_head(y)
        normed = self.norm(collapsed)
        logits = self.logits_dense(normed)
        return logits

    decoder_pt = DeepseekV4DecoderStack_PT(config_pt, num_layers)

    torch.nn.init.normal_(decoder_pt.embeddings.weight, std=0.02)
    torch.nn.init.normal_(decoder_pt.norm.weight, std=0.02)
    torch.nn.init.normal_(decoder_pt.logits_dense.weight, std=0.02)
    torch.nn.init.normal_(decoder_pt.hc_head.hc_fn, std=0.02)
    torch.nn.init.normal_(decoder_pt.hc_head.hc_base, std=0.02)
    torch.nn.init.normal_(decoder_pt.hc_head.hc_scale, std=0.02)

    for layer_pt in decoder_pt.layers:
      for param in [
          layer_pt.attn_hc.fn,
          layer_pt.attn_hc.base,
          layer_pt.attn_hc.scale,
          layer_pt.ffn_hc.fn,
          layer_pt.ffn_hc.base,
          layer_pt.ffn_hc.scale,
          layer_pt.self_attn.q_a_proj.weight,
          layer_pt.self_attn.q_a_norm.weight,
          layer_pt.self_attn.q_b_proj.weight,
          layer_pt.self_attn.kv_proj.weight,
          layer_pt.self_attn.kv_norm.weight,
          layer_pt.self_attn.o_a_proj.weight,
          layer_pt.self_attn.o_b_proj.weight,
          layer_pt.self_attn.sinks,
          layer_pt.mlp.gate.weight,
          layer_pt.mlp.experts.gate_up_proj,
          layer_pt.mlp.experts.down_proj,
          layer_pt.mlp.shared_experts.gate_proj.weight,
          layer_pt.mlp.shared_experts.up_proj.weight,
          layer_pt.mlp.shared_experts.down_proj.weight,
          layer_pt.input_layernorm.weight,
          layer_pt.post_attention_layernorm.weight,
      ]:
        torch.nn.init.normal_(param, std=0.02)
      if layer_pt.self_attn.compressor is not None:
        comp_pt = layer_pt.self_attn.compressor
        for param in [comp_pt.kv_proj.weight, comp_pt.gate_proj.weight, comp_pt.position_bias, comp_pt.kv_norm.weight]:
          torch.nn.init.normal_(param, std=0.02)
        if hasattr(comp_pt, "indexer"):
          for param in [
              comp_pt.indexer.kv_proj.weight,
              comp_pt.indexer.gate_proj.weight,
              comp_pt.indexer.position_bias,
              comp_pt.indexer.kv_norm.weight,
              comp_pt.indexer.q_b_proj.weight,
              comp_pt.indexer.weights_proj.weight,
          ]:
            torch.nn.init.normal_(param, std=0.02)

    logits_torch = decoder_pt(input_ids_torch, position_ids_torch).detach().numpy()

    devices = jax.devices()
    mesh = Mesh(np.array(devices), ("data",))

    for scan_mode in [False, True]:
      jax_config = _make_config(
          config_pt,
          B,
          S,
          D,
          base_num_decoder_layers=num_layers,
          logits_via_embedding=False,
          logits_dot_in_fp32=True,
          parameter_memory_host_offload=False,
          param_scan_axis=0,
          use_iota_embed=False,
          num_experts=config_pt.num_local_experts,
          num_experts_per_tok=config_pt.num_experts_per_tok,
          num_hash_layers=config_pt.num_hash_layers,
          gradient_accumulation_steps=1,
          hardware="cpu",
          megablox=False,
          sparse_matmul=False,
          use_gather_mosaic_kernel=False,
          num_vocab_tiling=1,
          compress_ratios=[4, 128, 4] * 15,
          mlp_dim=config_pt.intermediate_size,
          num_attention_heads=config_pt.num_attention_heads,
          q_lora_rank=config_pt.q_lora_rank,
          head_dim=config_pt.head_dim,
          o_groups=config_pt.o_groups,
          o_lora_rank=config_pt.o_lora_rank,
          index_n_heads=config_pt.index_n_heads,
          index_head_dim=config_pt.index_head_dim,
          index_topk=config_pt.index_topk,
          mlp_activations=["silu", "linear"],
          scan_layers=scan_mode,
          # Explicitly disable hardware MXU grid sequence padding (sa_block_kv=0) to ensure dot-product Softmax
          # normalization sums match unpadded PyTorch reference bounds precisely without exponential denominator drift.
          sa_block_kv=0,
      )
      decoder_jax = NNXDecoder(config=jax_config, mesh=mesh, rngs=nnx.Rngs(0))

      scan_length = (jax_config.num_decoder_layers - jax_config.num_hash_layers) // 2

      def get_jax_layer(decoder, lyr):
        if not scan_mode:
          return decoder.layers[lyr]
        if lyr < jax_config.num_hash_layers:
          return getattr(decoder.pre_layers, f"layers_{lyr}")
        elif lyr < jax_config.num_hash_layers + 2 * scan_length:
          return getattr(decoder.layers, f"layers_{(lyr - jax_config.num_hash_layers) % 2}")
        else:
          return getattr(
              decoder.post_layers,
              f"layers_{lyr - (jax_config.num_hash_layers + 2 * scan_length)}",
          )

      shared_embedding = Embed(vocab_size, D, config=jax_config, mesh=mesh, rngs=nnx.Rngs(0))
      shared_embedding.embedding[...] = jnp.array(decoder_pt.embeddings.weight.detach().numpy())
      decoder_jax.decoder_norm.scale[...] = jnp.array(decoder_pt.norm.weight.detach().numpy())
      decoder_jax.logits_dense.kernel[...] = jnp.array(decoder_pt.logits_dense.weight.detach().numpy().T)
      decoder_jax.hc_head.hc_fn[...] = jnp.array(decoder_pt.hc_head.hc_fn.detach().numpy().T)
      decoder_jax.hc_head.hc_base[...] = jnp.array(decoder_pt.hc_head.hc_base.detach().numpy())
      decoder_jax.hc_head.hc_scale[...] = jnp.array(decoder_pt.hc_head.hc_scale.detach().numpy())

      def assign_param(jax_param, pt_value, lyr):
        if hasattr(jax_param, "val"):
          jax_param[...] = pt_value
        else:
          is_scanned = scan_mode and (jax_config.num_hash_layers <= lyr < jax_config.num_hash_layers + 2 * scan_length)
          if is_scanned:
            block_step = (lyr - jax_config.num_hash_layers) // 2
            jax_param[block_step, ...] = pt_value
          else:
            jax_param[...] = pt_value

      hc = H_mult
      for lyr in range(num_layers):
        layer_jax, layer_pt = get_jax_layer(decoder_jax, lyr), decoder_pt.layers[lyr]
        assign_param(
            layer_jax.pre_self_attention_layer_norm.scale,
            jnp.array(layer_pt.input_layernorm.weight.detach().numpy()),
            lyr,
        )
        assign_param(
            layer_jax.post_self_attention_layer_norm.scale,
            jnp.array(layer_pt.post_attention_layernorm.weight.detach().numpy()),
            lyr,
        )

        assign_param(
            layer_jax.self_attention.q_a_proj.kernel,
            jnp.array(layer_pt.self_attn.q_a_proj.weight.detach().numpy().T),
            lyr,
        )
        assign_param(
            layer_jax.self_attention.q_a_norm.weight, jnp.array(layer_pt.self_attn.q_a_norm.weight.detach().numpy()), lyr
        )
        assign_param(
            layer_jax.self_attention.q_b_proj.kernel,
            jnp.array(layer_pt.self_attn.q_b_proj.weight.detach().numpy().T),
            lyr,
        )
        assign_param(
            layer_jax.self_attention.kv_proj.kernel, jnp.array(layer_pt.self_attn.kv_proj.weight.detach().numpy().T), lyr
        )
        assign_param(
            layer_jax.self_attention.kv_norm.weight, jnp.array(layer_pt.self_attn.kv_norm.weight.detach().numpy()), lyr
        )

        w_o_a_np = layer_pt.self_attn.o_a_proj.weight.detach().numpy()
        in_features_per_group = config_pt.num_attention_heads * config_pt.head_dim // config_pt.o_groups
        w_o_a_np = w_o_a_np.reshape(config_pt.o_groups, -1, in_features_per_group).transpose(0, 2, 1)
        assign_param(layer_jax.self_attention.o_a_proj.kernel, jnp.array(w_o_a_np), lyr)

        assign_param(
            layer_jax.self_attention.o_b_proj.kernel,
            jnp.array(layer_pt.self_attn.o_b_proj.weight.detach().numpy().T),
            lyr,
        )
        assign_param(layer_jax.self_attention.sinks, jnp.array(layer_pt.self_attn.sinks.detach().numpy()), lyr)

        if layer_pt.self_attn.compressor is not None:
          comp_pt = layer_pt.self_attn.compressor
          comp_jax = layer_jax.self_attention.compressor
          assign_param(comp_jax.kv_proj.kernel, jnp.array(comp_pt.kv_proj.weight.detach().numpy().T), lyr)
          assign_param(comp_jax.gate_proj.kernel, jnp.array(comp_pt.gate_proj.weight.detach().numpy().T), lyr)
          assign_param(comp_jax.position_bias, jnp.array(comp_pt.position_bias.detach().numpy()), lyr)
          assign_param(comp_jax.kv_norm.weight, jnp.array(comp_pt.kv_norm.weight.detach().numpy()), lyr)
          if hasattr(comp_pt, "indexer"):
            assign_param(
                comp_jax.indexer.kv_proj.kernel, jnp.array(comp_pt.indexer.kv_proj.weight.detach().numpy().T), lyr
            )
            assign_param(
                comp_jax.indexer.gate_proj.kernel, jnp.array(comp_pt.indexer.gate_proj.weight.detach().numpy().T), lyr
            )
            assign_param(comp_jax.indexer.position_bias, jnp.array(comp_pt.indexer.position_bias.detach().numpy()), lyr)
            assign_param(comp_jax.indexer.kv_norm.weight, jnp.array(comp_pt.indexer.kv_norm.weight.detach().numpy()), lyr)
            assign_param(
                comp_jax.indexer.q_b_proj.kernel, jnp.array(comp_pt.indexer.q_b_proj.weight.detach().numpy().T), lyr
            )
            assign_param(
                comp_jax.indexer.weights_proj.kernel,
                jnp.array(comp_pt.indexer.weights_proj.weight.detach().numpy().T),
                lyr,
            )

        moe_pt = layer_pt.mlp
        moe_jax = layer_jax.mlp
        assign_param(moe_jax.MoeBlock_0.gate.kernel, jnp.array(moe_pt.gate.weight.detach().numpy().T), lyr)
        if moe_pt.is_hash:
          assign_param(
              moe_jax.MoeBlock_0.gate.tid2eid, jnp.array(moe_pt.gate.tid2eid.detach().numpy(), dtype=jnp.int32), lyr
          )
        else:
          assign_param(
              moe_jax.MoeBlock_0.gate.e_score_correction_bias,
              jnp.array(moe_pt.gate.e_score_correction_bias.detach().numpy()),
              lyr,
          )

        gate_up_np = moe_pt.experts.gate_up_proj.detach().numpy()
        intermediate_dim = config_pt.intermediate_size
        wi_0_np = gate_up_np[:, :intermediate_dim, :].transpose(0, 2, 1)
        wi_1_np = gate_up_np[:, intermediate_dim:, :].transpose(0, 2, 1)
        wo_np = moe_pt.experts.down_proj.detach().numpy().transpose(0, 2, 1)

        assign_param(moe_jax.MoeBlock_0.wi_0, jnp.array(wi_0_np), lyr)
        assign_param(moe_jax.MoeBlock_0.wi_1, jnp.array(wi_1_np), lyr)
        assign_param(moe_jax.MoeBlock_0.wo, jnp.array(wo_np), lyr)

        assign_param(
            moe_jax.shared_experts.wi_0.kernel, jnp.array(moe_pt.shared_experts.gate_proj.weight.detach().numpy().T), lyr
        )
        assign_param(
            moe_jax.shared_experts.wi_1.kernel, jnp.array(moe_pt.shared_experts.up_proj.weight.detach().numpy().T), lyr
        )
        assign_param(
            moe_jax.shared_experts.wo.kernel, jnp.array(moe_pt.shared_experts.down_proj.weight.detach().numpy().T), lyr
        )

        assign_param(layer_jax.mhc_attention.pre_alpha, jnp.array(layer_pt.attn_hc.fn.detach().numpy()[:hc].T), lyr)
        assign_param(
            layer_jax.mhc_attention.post_alpha, jnp.array(layer_pt.attn_hc.fn.detach().numpy()[hc : 2 * hc].T), lyr
        )
        assign_param(layer_jax.mhc_attention.res_alpha, jnp.array(layer_pt.attn_hc.fn.detach().numpy()[2 * hc :].T), lyr)
        assign_param(layer_jax.mhc_attention.pre_beta, jnp.array(layer_pt.attn_hc.base.detach().numpy()[:hc]), lyr)
        assign_param(
            layer_jax.mhc_attention.post_beta, jnp.array(layer_pt.attn_hc.base.detach().numpy()[hc : 2 * hc]), lyr
        )
        assign_param(
            layer_jax.mhc_attention.res_beta,
            jnp.array(layer_pt.attn_hc.base.detach().numpy()[2 * hc :].reshape(hc, hc)),
            lyr,
        )
        assign_param(layer_jax.mhc_attention.pre_alpha_scale, jnp.array([layer_pt.attn_hc.scale[0].item()]), lyr)
        assign_param(layer_jax.mhc_attention.post_alpha_scale, jnp.array([layer_pt.attn_hc.scale[1].item()]), lyr)
        assign_param(layer_jax.mhc_attention.res_alpha_scale, jnp.array([layer_pt.attn_hc.scale[2].item()]), lyr)

        assign_param(layer_jax.mhc_mlp.pre_alpha, jnp.array(layer_pt.ffn_hc.fn.detach().numpy()[:hc].T), lyr)
        assign_param(layer_jax.mhc_mlp.post_alpha, jnp.array(layer_pt.ffn_hc.fn.detach().numpy()[hc : 2 * hc].T), lyr)
        assign_param(layer_jax.mhc_mlp.res_alpha, jnp.array(layer_pt.ffn_hc.fn.detach().numpy()[2 * hc :].T), lyr)
        assign_param(layer_jax.mhc_mlp.pre_beta, jnp.array(layer_pt.ffn_hc.base.detach().numpy()[:hc]), lyr)
        assign_param(layer_jax.mhc_mlp.post_beta, jnp.array(layer_pt.ffn_hc.base.detach().numpy()[hc : 2 * hc]), lyr)
        assign_param(
            layer_jax.mhc_mlp.res_beta, jnp.array(layer_pt.ffn_hc.base.detach().numpy()[2 * hc :].reshape(hc, hc)), lyr
        )
        assign_param(layer_jax.mhc_mlp.pre_alpha_scale, jnp.array([layer_pt.ffn_hc.scale[0].item()]), lyr)
        assign_param(layer_jax.mhc_mlp.post_alpha_scale, jnp.array([layer_pt.ffn_hc.scale[1].item()]), lyr)
        assign_param(layer_jax.mhc_mlp.res_alpha_scale, jnp.array([layer_pt.ffn_hc.scale[2].item()]), lyr)

      if not scan_mode:
        y_pt = decoder_pt.embeddings(input_ids_torch)
        y_pt = y_pt.unsqueeze(2).expand(-1, -1, 4, -1)
        cos_pt, sin_pt = decoder_pt.rotary_emb(y_pt[:, :, 0, :], position_ids_torch, layer_type="compress")

        y_jax = shared_embedding(input_ids_jax.astype("int32"), model_mode="train")
        y_jax = jnp.repeat(jnp.expand_dims(y_jax, axis=2), 4, axis=2).astype(y_jax.dtype)

        np.testing.assert_allclose(
            y_pt.detach().numpy(), np.array(y_jax), atol=1e-5, rtol=1e-5, err_msg="Embedding mismatch"
        )

        for lyr in range(num_layers):
          layer_pt = decoder_pt.layers[lyr]
          layer_jax = decoder_jax.layers[lyr]

          y_pt = layer_pt(
              y_pt,
              input_ids=input_ids_torch,
              position_embeddings=(cos_pt, sin_pt),
              position_ids=position_ids_torch,
              attention_mask=None,
          )
          y_jax, _ = layer_jax(
              y_jax,
              decoder_segment_ids=jnp.zeros((B, S), dtype=jnp.int32),
              decoder_positions=jnp.array(position_ids_np, dtype=jnp.int32),
              deterministic=True,
              model_mode="train",
              decoder_input_tokens=input_ids_jax,
          )

      logits_jax, _, _ = decoder_jax(
          shared_embedding=shared_embedding,
          decoder_input_tokens=input_ids_jax,
          decoder_positions=jnp.array(position_ids_np, dtype=jnp.int32),
          decoder_segment_ids=jnp.zeros((B, S), dtype=jnp.int32),
          deterministic=True,
      )

      np.testing.assert_allclose(logits_torch, np.array(logits_jax), atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
  unittest.main()
