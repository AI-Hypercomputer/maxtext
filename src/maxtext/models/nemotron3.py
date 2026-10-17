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

"""Nemotron-H hybrid decoder (NVIDIA Nemotron-3-Nano): Mamba-2, NoPE attention and MoE layers.

Each decoder layer is `x + mixer(norm(x))`, where the mixer is chosen by the layer's character in
`config.hybrid_override_pattern`: 'M' Mamba-2, '*' attention without RoPE, 'E' routed + shared MoE.
"""

from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from maxtext.common.common_types import (
    CACHE_BATCH,
    CACHE_BATCH_PREFILL,
    CACHE_HEADS,
    Config,
    MODEL_MODE_AUTOREGRESSIVE,
    MODEL_MODE_PREFILL,
    MODEL_MODE_TRAIN,
)
from maxtext.layers import initializers
from maxtext.layers import linears
from maxtext.layers import moe
from maxtext.layers import quantizations
from maxtext.layers.attentions import Attention
from maxtext.layers.normalizations import RMSNorm
from maxtext.layers.quantizations import AqtQuantization as Quant
from maxtext.utils import max_utils

MAMBA, ATTENTION, MOE = "M", "*", "E"


class Nemotron3RMSNormGated(nnx.Module):
  """Gate-then-normalize grouped RMSNorm (HF Zamba2RMSNormGated).

  Computes `x * silu(gate)` in fp32, normalizes each group of `group_size` channels by its RMS,
  multiplies by the scale and returns the input dtype.
  """

  def __init__(
      self,
      num_features: int,
      group_size: int,
      *,
      epsilon: float,
      weight_dtype: Any,
      kernel_axes: tuple[None | str, ...] = ("norm",),
      rngs: nnx.Rngs,
  ):
    if num_features % group_size:
      raise ValueError(f"num_features {num_features} is not a multiple of group_size {group_size}")
    self.num_features = num_features
    self.group_size = group_size
    self.epsilon = epsilon
    self.scale = nnx.Param(
        jax.nn.initializers.ones(rngs.params(), (num_features,), weight_dtype),
        out_sharding=kernel_axes,
    )

  def __call__(self, x: jax.Array, gate: jax.Array | None = None) -> jax.Array:
    input_dtype = x.dtype
    h = x.astype(jnp.float32)
    if gate is not None:
      h = h * jax.nn.silu(gate.astype(jnp.float32))
    grouped = h.reshape(*h.shape[:-1], self.num_features // self.group_size, self.group_size)
    variance = jnp.mean(jnp.square(grouped), axis=-1, keepdims=True)
    h = (grouped * jax.lax.rsqrt(variance + self.epsilon)).reshape(h.shape)
    return (self.scale.get_value() * h).astype(input_dtype)


def _segment_sum(x: jax.Array) -> jax.Array:
  """HF `segment_sum`: out[..., i, j] = sum(x[..., j + 1 : i + 1]) for i > j, 0 elsewhere (unmasked)."""
  t = x.shape[-1]
  expanded = jnp.broadcast_to(x[..., :, None], x.shape + (t,))
  expanded = jnp.where(jnp.tril(jnp.ones((t, t), dtype=bool), -1), expanded, 0.0)
  return jnp.cumsum(expanded, axis=-2)


def _masked_exp(x: jax.Array, mask: jax.Array) -> jax.Array:
  """exp(x) where mask, else 0; the inner where keeps masked entries out of the gradient."""
  return jnp.where(mask, jnp.exp(jnp.where(mask, x, 0.0)), 0.0)


def _run_ids(decoder_segment_ids: None | jax.Array, shape: tuple[int, int]) -> jax.Array:
  """Ids of contiguous same-document runs; padding (id 0) joins the preceding document.

  Mamba state and conv taps never cross a run boundary (packed documents, ruling 2.2 A). Pad
  positions extend the preceding run so a right-padded prefill keeps its true-length state; they
  carry no input (zeroed x, dt = 0).
  """
  if decoder_segment_ids is None:
    return jnp.zeros(shape, jnp.int32)
  filled = jax.lax.associative_scan(lambda a, b: jnp.where(b != 0, b, a), decoder_segment_ids, axis=1)
  starts = (filled[:, 1:] != filled[:, :-1]).astype(jnp.int32)
  return jnp.concatenate([jnp.zeros_like(filled[:, :1]), jnp.cumsum(starts, axis=1)], axis=1)


class Nemotron3Mamba2Mixer(nnx.Module):
  """Mamba-2 mixer (HF NemotronHMamba2Mixer, Phase 1 T04) with its decode cache (T10).

  in_proj -> [gate | xBC | dt]; depthwise causal conv + SiLU on xBC; fp32 SSD chunked scan
  (prefill/train) or one recurrent step (autoregressive); D skip; gated norm; out_proj. The scan
  follows HF torch_forward's algorithm order; the chunk-end decay uses the window sum (last row of
  the intra-chunk decay matrix) so no packed document's values enter another's arithmetic.
  """

  def __init__(self, config: Config, mesh: Mesh, *, model_mode: str, quant: None | Quant = None, rngs: nnx.Rngs):
    del mesh
    cfg = config
    self.config = cfg
    self.num_heads = cfg.mamba_num_heads
    self.head_dim = cfg.mamba_head_dim
    self.n_groups = cfg.mamba_n_groups
    self.state_size = cfg.mamba_ssm_state_size
    self.conv_kernel = cfg.mamba_conv_kernel
    self.chunk_size = cfg.mamba_chunk_size
    self.time_step_limit = tuple(float(v) for v in cfg.mamba_time_step_limit)
    self.intermediate_size = self.num_heads * self.head_dim
    self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.state_size
    self.precision = jax.lax.Precision(cfg.matmul_precision)
    if self.num_heads % self.n_groups:
      raise ValueError(f"mamba_num_heads {self.num_heads} is not a multiple of mamba_n_groups {self.n_groups}")

    self.in_proj = linears.DenseGeneral(
        in_features_shape=cfg.emb_dim,
        out_features_shape=self.intermediate_size + self.conv_dim + self.num_heads,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        kernel_axes=("embed", "mlp"),
        quant=quant,
        shard_mode=cfg.shard_mode,
        matmul_precision=cfg.matmul_precision,
        rngs=rngs,
    )
    conv_init = jax.nn.initializers.variance_scaling(1.0, "fan_in", "truncated_normal", in_axis=0, out_axis=1)
    self.conv_weight = nnx.Param(
        conv_init(rngs.params(), (self.conv_kernel, self.conv_dim), cfg.weight_dtype), out_sharding=(None, "mlp")
    )
    self.conv_bias = (
        nnx.Param(jnp.zeros((self.conv_dim,), cfg.weight_dtype), out_sharding=("mlp",)) if cfg.mamba_conv_bias else None
    )
    # HF __init__ values; checkpoints overwrite them.
    self.dt_bias = nnx.Param(jnp.ones((self.num_heads,), cfg.weight_dtype), out_sharding=("heads",))
    self.A_log = nnx.Param(
        jnp.log(jnp.arange(1, self.num_heads + 1, dtype=jnp.float32)).astype(cfg.weight_dtype),
        out_sharding=("heads",),
    )
    self.D = nnx.Param(jnp.ones((self.num_heads,), cfg.weight_dtype), out_sharding=("heads",))
    self.norm = Nemotron3RMSNormGated(
        self.intermediate_size,
        self.intermediate_size // self.n_groups,
        epsilon=cfg.normalization_layer_epsilon,
        weight_dtype=cfg.weight_dtype,
        rngs=rngs,
    )
    self.out_proj = linears.DenseGeneral(
        in_features_shape=self.intermediate_size,
        out_features_shape=cfg.emb_dim,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        kernel_axes=("mlp", "embed"),
        quant=quant,
        shard_mode=cfg.shard_mode,
        matmul_precision=cfg.matmul_precision,
        rngs=rngs,
    )
    # Decode state (HF NemotronHHybridDynamicCache): the last K-1 conv inputs and the SSM state, in
    # the model dtype. Names match MaxEngine's fixed-size state insert path.
    if model_mode == MODEL_MODE_TRAIN:
      self.conv_state = None
      self.recurrent_state = None
    else:
      batch, _ = max_utils.get_batch_seq_len_for_mode(cfg, model_mode)
      batch_axis = CACHE_BATCH_PREFILL if model_mode == MODEL_MODE_PREFILL else CACHE_BATCH
      self.conv_state = nnx.Cache(
          jnp.zeros((batch, self.conv_kernel - 1, self.conv_dim), cfg.dtype), out_sharding=(batch_axis, None, None)
      )
      self.recurrent_state = nnx.Cache(
          jnp.zeros((batch, self.num_heads, self.head_dim, self.state_size), cfg.dtype),
          out_sharding=(batch_axis, CACHE_HEADS, None, None),
      )

  def _causal_conv(self, padded: jax.Array, run: None | jax.Array) -> jax.Array:
    """Depthwise causal conv over `padded` (K-1 leading context positions); taps stay inside a run."""
    k, s = self.conv_kernel, padded.shape[1] - self.conv_kernel + 1
    weight = self.conv_weight.get_value().astype(jnp.float32)
    padded = padded.astype(jnp.float32)
    run_padded = None if run is None else jnp.pad(run, ((0, 0), (k - 1, 0)), constant_values=-1)
    out = jnp.zeros(padded.shape[:1] + (s, self.conv_dim), jnp.float32)
    for tap in range(k):
      window = padded[:, tap : tap + s]
      if run_padded is not None:
        window = jnp.where((run_padded[:, tap : tap + s] == run)[..., None], window, 0.0)
      out = out + window * weight[tap]
    if self.conv_bias is not None:
      out = out + self.conv_bias.get_value().astype(jnp.float32)
    return out

  def _ssd_scan(self, x, dt, a, b, c, run):  # pylint: disable=too-many-positional-arguments
    """HF torch_forward chunked scan from a zero state, fp32; returns (y [B, S, H, P], final state).

    Pairs of positions in different runs get zero decay, so state never crosses a run boundary.
    """
    batch, seq_len = x.shape[:2]
    lc = self.chunk_size
    pad = (-seq_len) % lc
    nc = (seq_len + pad) // lc

    def chunks(t):
      t = jnp.pad(t, ((0, 0), (0, pad)) + ((0, 0),) * (t.ndim - 2))
      return t.reshape((batch, nc, lc) + t.shape[2:])

    d_residual = self.D.get_value().astype(jnp.float32)[:, None] * jnp.pad(x, ((0, 0), (0, pad), (0, 0), (0, 0)))
    x = x * dt[..., None]
    a = a * dt
    x, a, b, c = (chunks(t) for t in (x, a, b, c))
    run = jnp.pad(run, ((0, 0), (0, pad)), mode="edge").reshape(batch, nc, lc)

    a = a.transpose(0, 3, 1, 2)  # [B, H, nc, L]
    a_cumsum = jnp.cumsum(a, axis=-1)
    causal = jnp.tril(jnp.ones((lc, lc), dtype=bool))
    same_run = run[:, :, :, None] == run[:, :, None, :]  # [B, nc, L(t), L(s)]
    decay = _masked_exp(_segment_sum(a), causal & same_run[:, None])  # [B, H, nc, L, L]

    # 1. Intra-chunk (diagonal blocks).
    g = jnp.einsum("bclhn,bcshn->bclsh", c, b, precision=self.precision)
    m = g * decay.transpose(0, 2, 3, 4, 1)
    y_diag = jnp.einsum("bclsh,bcshp->bclhp", m, x, precision=self.precision)

    # 2. Per-chunk end states (B terms).
    decay_states = decay[:, :, :, -1, :]  # exp(sum of a after s to chunk end), same run only
    b_decay = b * decay_states.transpose(0, 2, 3, 1)[..., None]
    states = jnp.einsum("bclhn,bclhp->bchpn", b_decay, x, precision=self.precision)

    # 3. Inter-chunk recurrence; state z enters chunk z. The zero initial state belongs to the first run.
    states = jnp.concatenate([jnp.zeros_like(states[:, :1]), states], axis=1)
    chunk_run = jnp.concatenate([run[:, 0, :1], run[:, :, -1]], axis=1)  # [B, nc + 1]
    same_chunk_run = chunk_run[:, :, None] == chunk_run[:, None, :]
    chunk_causal = jnp.tril(jnp.ones((nc + 1, nc + 1), dtype=bool))
    a_chunk = jnp.pad(a_cumsum[..., -1], ((0, 0), (0, 0), (1, 0)))
    decay_chunk = _masked_exp(_segment_sum(a_chunk), chunk_causal & same_chunk_run[:, None])
    new_states = jnp.einsum("bhzc,bchpn->bzhpn", decay_chunk, states, precision=self.precision)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. State -> output (C terms), only for positions still in the run that entered the chunk.
    enters = run == chunk_run[:, :-1, None]  # [B, nc, L]
    state_decay_out = _masked_exp(a_cumsum, enters[:, None])
    y_off = jnp.einsum("bclhn,bchpn->bclhp", c, states, precision=self.precision)
    y_off = y_off * state_decay_out.transpose(0, 2, 3, 1)[..., None]

    y = (y_diag + y_off).reshape(batch, nc * lc, self.num_heads, self.head_dim) + d_residual
    return y[:, :seq_len], final_state

  def __call__(self, hidden_states, decoder_segment_ids=None, model_mode=MODEL_MODE_TRAIN, kv_cache=None):
    if kv_cache is not None:
      raise NotImplementedError("Nemotron3Mamba2Mixer keeps its own state; an external kv_cache is not supported.")
    cfg = self.config
    batch, seq_len, _ = hidden_states.shape
    decode = model_mode == MODEL_MODE_AUTOREGRESSIVE
    if decode and seq_len != 1:
      raise ValueError(f"Autoregressive Mamba steps take one token, got {seq_len}.")
    if model_mode != MODEL_MODE_TRAIN and self.conv_state is None:
      raise ValueError(f"model_mode {model_mode!r} needs a mixer built with a non-train model_mode.")
    valid = None if decoder_segment_ids is None or decode else (decoder_segment_ids != 0)[..., None]
    x = hidden_states if valid is None else jnp.where(valid, hidden_states, 0).astype(hidden_states.dtype)
    gate, xbc, dt = jnp.split(self.in_proj(x), [self.intermediate_size, self.intermediate_size + self.conv_dim], -1)

    k = self.conv_kernel
    run = None if decode else _run_ids(decoder_segment_ids, (batch, seq_len))
    if decode:
      padded = jnp.concatenate([self.conv_state.get_value().astype(xbc.dtype), xbc], axis=1)
      self.conv_state.set_value(padded[:, 1:].astype(self.conv_state.get_value().dtype))
    else:
      padded = jnp.pad(xbc, ((0, 0), (k - 1, 0), (0, 0)))
    xbc = jax.nn.silu(self._causal_conv(padded, run).astype(cfg.dtype).astype(jnp.float32)).astype(cfg.dtype)
    if valid is not None:
      xbc = jnp.where(valid, xbc, 0).astype(cfg.dtype)

    gn = self.n_groups * self.state_size
    xs, b, c = jnp.split(xbc.astype(jnp.float32), [self.intermediate_size, self.intermediate_size + gn], axis=-1)
    xs = xs.reshape(batch, seq_len, self.num_heads, self.head_dim)
    rep = self.num_heads // self.n_groups
    b = jnp.repeat(b.reshape(batch, seq_len, self.n_groups, self.state_size), rep, axis=2)
    c = jnp.repeat(c.reshape(batch, seq_len, self.n_groups, self.state_size), rep, axis=2)
    a = -jnp.exp(self.A_log.get_value().astype(jnp.float32))
    dt = jax.nn.softplus(dt.astype(jnp.float32) + self.dt_bias.get_value().astype(jnp.float32))
    dt = jnp.clip(dt, *self.time_step_limit)
    if valid is not None:
      dt = jnp.where(valid, dt, 0.0)

    if decode:
      dt1, x1, b1, c1 = dt[:, 0], xs[:, 0], b[:, 0], c[:, 0]
      cache_dtype = self.recurrent_state.get_value().dtype
      state = self.recurrent_state.get_value().astype(jnp.float32) * jnp.exp(dt1 * a)[..., None, None]
      state = state + (dt1[..., None, None] * b1[:, :, None, :]) * x1[..., None]
      state = state.astype(cache_dtype)
      self.recurrent_state.set_value(state)
      y = jnp.einsum("bhpn,bhn->bhp", state.astype(jnp.float32), c1, precision=self.precision)
      y = y + x1 * self.D.get_value().astype(jnp.float32)[:, None]
      # HF decode feeds the gated norm a model-dtype y (trap 2b).
      y = y.reshape(batch, 1, self.intermediate_size).astype(cfg.dtype)
    else:
      y, final_state = self._ssd_scan(xs, dt, a, b, c, run)
      y = y.reshape(batch, seq_len, self.intermediate_size)
      if model_mode == MODEL_MODE_PREFILL:
        end = seq_len if valid is None else jnp.max(jnp.arange(1, seq_len + 1) * valid[..., 0], axis=1)
        end = jnp.broadcast_to(end, (batch,))
        conv_state = jax.vmap(lambda p, e: jax.lax.dynamic_slice_in_dim(p, e, k - 1, axis=0))(padded, end)
        self.conv_state.set_value(conv_state.astype(self.conv_state.get_value().dtype))
        self.recurrent_state.set_value(final_state.astype(self.recurrent_state.get_value().dtype))

    return self.out_proj(self.norm(y, gate).astype(cfg.dtype))


class Nemotron3DecoderLayer(nnx.Module):
  """Nemotron-H block: `x + mixer(norm(x))`, mixer chosen by hybrid_override_pattern[layer_idx]."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      quant: None | Quant = None,
      *,
      layer_idx: int,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.block_type = config.hybrid_override_pattern[layer_idx]
    self.norm = RMSNorm(
        num_features=config.emb_dim,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        shard_mode=config.shard_mode,
        kernel_axes=("norm",),
        epsilon=config.normalization_layer_epsilon,
        rngs=rngs,
    )
    if self.block_type == MAMBA:
      self.mixer = Nemotron3Mamba2Mixer(config, mesh, model_mode=model_mode, quant=quant, rngs=rngs)
    elif self.block_type == ATTENTION:
      batch_size, seq_len = max_utils.get_batch_seq_len_for_mode(config, model_mode)
      inputs_shape = (batch_size, seq_len, config.emb_dim)
      self.mixer = Attention(
          config=config,
          num_query_heads=config.num_query_heads,
          num_kv_heads=config.num_kv_heads,
          head_dim=config.head_dim,
          max_target_length=config.max_target_length,
          max_prefill_predict_length=config.max_prefill_predict_length,
          attention_kernel=config.attention,
          inputs_q_shape=inputs_shape,
          inputs_kv_shape=inputs_shape,
          mesh=mesh,
          dtype=config.dtype,
          weight_dtype=config.weight_dtype,
          dropout_rate=config.dropout_rate,
          float32_qk_product=config.float32_qk_product,
          float32_logits=config.float32_logits,
          quant=quant,
          kv_quant=quantizations.configure_kv_quant(config),
          use_ragged_attention=config.use_ragged_attention,
          ragged_block_size=config.ragged_block_size,
          use_bias_in_projections=config.attention_bias,
          use_qk_norm=False,
          query_pre_attn_scalar=config.head_dim**-0.5,
          is_nope_layer=True,
          model_mode=model_mode,
          rngs=rngs,
      )
    elif self.block_type == MOE:
      self.mixer = moe.RoutedAndSharedMoE(
          config=config,
          mesh=mesh,
          kernel_init=initializers.nd_dense_init(config.dense_init_scale, "fan_in", "truncated_normal"),
          kernel_axes=("embed", None),
          dtype=config.dtype,
          weight_dtype=config.weight_dtype,
          quant=quant,
          rngs=rngs,
      )
    else:
      raise ValueError(f"Unsupported Nemotron-H block type {self.block_type!r} at layer {layer_idx}")

  def __call__(  # pylint: disable=too-many-positional-arguments
      self,
      inputs: jax.Array,
      decoder_segment_ids: None | jax.Array,
      decoder_positions: None | jax.Array,
      deterministic: bool,
      model_mode: str,
      previous_chunk=None,
      slot: None | int = None,
      kv_cache: None | jax.Array = None,
      attention_metadata: None | dict[str, Any] = None,
      forced_routed_experts: None | jax.Array = None,
  ):
    if isinstance(inputs, tuple):
      inputs = inputs[0]
    hidden = self.norm(inputs)
    if self.block_type == MAMBA:
      hidden = self.mixer(hidden, decoder_segment_ids=decoder_segment_ids, model_mode=model_mode, kv_cache=kv_cache)
    elif self.block_type == ATTENTION:
      hidden, kv_cache = self.mixer(
          hidden,
          hidden,
          decoder_positions,
          decoder_segment_ids=decoder_segment_ids,
          deterministic=deterministic,
          model_mode=model_mode,
          previous_chunk=previous_chunk,
          slot=slot,
          kv_cache=kv_cache,
          attention_metadata=attention_metadata,
      )
    else:
      hidden, load_balance_loss, moe_bias_updates = self.mixer(hidden, forced_routed_experts=forced_routed_experts)
      if self.config.load_balance_loss_weight > 0.0 and load_balance_loss is not None:
        self.sow(nnx.Intermediate, "moe_lb_loss", load_balance_loss)
      if self.config.routed_bias and self.config.routed_bias_update_rate > 0.0 and moe_bias_updates is not None:
        self.sow(nnx.Intermediate, "moe_bias_updates", moe_bias_updates)
    return inputs + hidden, kv_cache
