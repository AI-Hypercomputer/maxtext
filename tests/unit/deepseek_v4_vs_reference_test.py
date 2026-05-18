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

from collections.abc import Callable
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import jax
import jax.numpy as jnp
from flax import nnx
import maxtext.layers.normalizations as jax_norm_module
import maxtext.layers.embeddings as jax_emb_module
import maxtext.layers.linears as jax_linear_module


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
    self.max_position_embeddings = 1048576
    self.rope_theta = 10000.0
    self.compress_rope_theta = 160000.0
    self.compress_rates = {
        "compressed_sparse_attention": 4,
        "heavily_compressed_attention": 128,
    }
    self.sliding_window = 128
    self.o_groups = 8
    self.o_lora_rank = 1024
    self.index_n_heads = 64
    self.index_head_dim = 128
    self.index_topk = 512
    self.rms_norm_eps = 1.0e-6
    self.attention_dropout = 0.0
    self.layer_types = ["heavily_compressed_attention"] * 43

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
  ) -> torch.Tensor:
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
    return compressed.unsqueeze(1)


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
    topk = min(self.index_topk, compressed_kv.shape[1])
    return index_scores.topk(topk, dim=-1).indices


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
  ) -> torch.Tensor:
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
    topk = self.indexer(hidden_states, q_residual, position_ids, past_key_values, layer_idx)  # [B, S, k]
    expanded = compressed_kv.unsqueeze(2).expand(-1, -1, seq_len, -1, -1)
    idx = topk.unsqueeze(1).unsqueeze(-1).expand(-1, 1, -1, -1, self.head_dim)
    return torch.gather(expanded, 3, idx).reshape(batch, 1, -1, self.head_dim)


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

    if self.compressor is not None:  # Compressed KV (CSA or HCA)
      compressed_kv = self.compressor(hidden_states, q_residual, position_ids, past_key_values, self.layer_idx)
      kv = torch.cat([kv, compressed_kv], dim=2)

    # The compressor path concatenates extra entries onto the KV axis after the
    # standard sliding-window cache update, so a tensor `attention_mask` (built
    # for the pre-concat KV length) needs to be right-padded to cover them.
    # Flex-attention passes a `BlockMask` whose KV-length axis comes from its
    # own `mask_mod`, not from a dense tensor — skip the pad in that case.
    if isinstance(attention_mask, torch.Tensor) and kv.shape[2] > attention_mask.shape[-1]:
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


import unittest


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
    jax_model = jax_norm_module.DeepSeekV4UnweightedRMSNorm(eps=1e-6)
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
    jax_model = jax_norm_module.DeepSeekV4RMSNorm(hidden_size=512, eps=1e-6)
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
    jax_emb = jax_emb_module.DeepSeekV4RotaryEmbedding(head_dim=D, partial_rotary_factor=64.0 / 512.0, rope_theta=10000.0)
    cos_jax, sin_jax = jax_emb(x_jax, position_ids_jax)

    # Execute JAX target application.
    out_jax = jax_emb_module.apply_rotary_pos_emb(x_jax, cos_jax, sin_jax, unsqueeze_dim=2)
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
    jax_model = jax_linear_module.DeepSeekGroupedLinear(
        in_features_per_group=i,
        out_features=o,
        n_groups=g,
        rngs=rngs,
    )

    # Copy the reshaped and transposed weight matrix matching PyTorch's view mapping
    # [o, i] -> [g, o_g, i] -> [g, i, o_g]
    jax_model.weight.value = jnp.array(weight_np.reshape(g, out_features_per_group, i).transpose(0, 2, 1))
    out_jax = jax_model(x_jax)

    # Verify numerical output parity between frameworks
    np.testing.assert_allclose(out_torch, out_jax, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
  unittest.main()
