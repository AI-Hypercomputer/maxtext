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

"""RWKV-7 ("Goose", the x070 architecture) decoder layer.

The math follows BlinkDL's reference implementation (`RWKV-LM/RWKV-v7/
rwkv_v7_demo.py`, the non-CUDA `RWKV7_OP` fallback). Per head, per timestep,
with an N x N recurrent state S:

    S_t = S_{t-1} * diag(w_t) + (S_{t-1} @ (-kk_t)) (kk_t * a_t)^T + v_t k_t^T
    y_t = S_t @ r_t

Unlike attention, there is no key/value cache: the entire history is summarized
by S plus the previous token's hidden state (RWKV's "token shift"). Both are
carried in `nnx.Cache` variables for autoregressive decoding, named after
MaxEngine's fixed-size linear-attention states: `recurrent_state` for S and
`conv_state` for the token shift (a width-2 depthwise convolution whose window
is the previous token), so MaxEngine inserts them into decode slots the same way
it does Gated DeltaNet's.

Two structural details differ from every other decoder block in MaxText and are
handled by a dedicated application path in `nnx_decoders.py`:

  * `v_first` — layer 0's value projection is fed to every later layer, so a
    second value (besides the residual stream) is threaded through the stack.
  * `ln0` — layer 0 applies an extra LayerNorm to the embedding output.
"""

import functools
import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from flax import nnx

from maxtext.common.common_types import (
    CACHE_BATCH,
    CACHE_BATCH_PREFILL,
    CACHE_HEADS,
    Config,
    KV_BATCH,
    KV_HEAD,
    MODEL_MODE_PREFILL,
    MODEL_MODE_TRAIN,
    ShardMode,
)
from maxtext.kernels.rwkv7_wkv import wkv7_pallas
from maxtext.kernels.rwkv7_wkv_gpu import MAX_CHUNK_SIZE as MAX_GPU_CHUNK_SIZE, wkv7_pallas_gpu
from maxtext.layers.linears import DenseGeneral
from maxtext.layers.quantizations import AqtQuantization as Quant
from maxtext.utils import max_utils
from maxtext.utils.sharding import (
    get_logical_axis_rules,
    logical_to_mesh_axes,
    remove_incompatible_mesh_axes_from_partition_spec,
)


class Rwkv7LayerNorm(nnx.Module):
  """Plain LayerNorm (mean-centering, learned scale and bias) over the last axis.

  MaxText's only non-RMS norm, `gpt3.Gpt3LayerNorm`, stores `weight - 1` and
  applies `scale + 1`; RWKV-7 checkpoints store the weight directly, so a
  straight 1:1 module keeps checkpoint conversion honest.
  """

  def __init__(
      self,
      num_features: int,
      epsilon: float = 1e-5,
      dtype: Any = jnp.float32,
      weight_dtype: Any = jnp.float32,
      kernel_axes: tuple[None | str, ...] = (),
      parameter_memory_host_offload: bool = False,
      *,
      rngs: nnx.Rngs,
  ):
    del rngs, parameter_memory_host_offload
    self.epsilon = epsilon
    self.dtype = dtype
    self.scale = nnx.Param(jnp.ones((num_features,), weight_dtype), sharding=kernel_axes)
    self.bias = nnx.Param(jnp.zeros((num_features,), weight_dtype), sharding=kernel_axes)

  def __call__(self, x: jnp.ndarray, out_sharding=None) -> jnp.ndarray:
    del out_sharding
    x = jnp.asarray(x, jnp.float32)
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    y = (x - mean) * jax.lax.rsqrt(var + self.epsilon)
    y = y * jnp.asarray(self.scale[...], jnp.float32) + jnp.asarray(self.bias[...], jnp.float32)
    return y.astype(self.dtype)


def _head_group_norm(x: jnp.ndarray, scale: jnp.ndarray, bias: jnp.ndarray, epsilon: float) -> jnp.ndarray:
  """GroupNorm with one group per head.

  Equivalent to `torch.nn.GroupNorm(n_head, emb_dim, eps)` applied to a
  `(tokens, emb_dim)` tensor: statistics are taken within each head's channels
  only. Written out rather than using `flax.nnx.GroupNorm`, which also reduces
  over leading "spatial" axes and would mix statistics across timesteps.

  Args:
    x: `(..., n_head, head_size)` activations.
    scale: `(emb_dim,)` learned scale.
    bias: `(emb_dim,)` learned bias.
    epsilon: variance epsilon.

  Returns:
    `(..., emb_dim)` normalized activations.
  """
  x = jnp.asarray(x, jnp.float32)
  mean = jnp.mean(x, axis=-1, keepdims=True)
  var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
  y = (x - mean) * jax.lax.rsqrt(var + epsilon)
  y = y.reshape(y.shape[:-2] + (y.shape[-2] * y.shape[-1],))
  return y * scale + bias


def segment_resets(decoder_segment_ids: jnp.ndarray) -> jnp.ndarray:
  """`(B, T)` True where a new document starts inside a packed row.

  That is, where the segment id changes to a nonzero id mid-row. Padding (id 0)
  never resets; it's handled separately as identity steps. Position 0 never
  resets either: the row continues from the incoming state (zero in training,
  the cache in prefill).
  """
  previous = jnp.concatenate([decoder_segment_ids[:, :1], decoder_segment_ids[:, :-1]], axis=1)
  return (decoder_segment_ids != previous) & (decoder_segment_ids != 0)


def _cache_batch_axis(model_mode: str) -> str:
  return CACHE_BATCH_PREFILL if model_mode == MODEL_MODE_PREFILL else CACHE_BATCH


def _last_valid(x: jnp.ndarray, prev: jnp.ndarray, valid_lens: jnp.ndarray | None) -> jnp.ndarray:
  """Each example's last valid timestep of `x` (B, T, E), or `prev` (B, E) if it has none.

  Assumes contiguous right padding (valid tokens first, then padding), which is
  what MaxEngine's prefill produces (`position < true_length`). Indexing the
  sequence prefixed with `prev` at `valid_len` returns `prev` itself for an
  example with no valid tokens, so its cache comes through unchanged.
  """
  if valid_lens is None:
    return x[:, -1, :]
  with_prev = jnp.concatenate([prev[:, None, :], x], axis=1)
  return jax.vmap(lambda seq, n: jax.lax.dynamic_index_in_dim(seq, n, axis=0, keepdims=False))(with_prev, valid_lens)


def wkv7_naive(
    r: jnp.ndarray,
    w: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    a: jnp.ndarray,
    kk: jnp.ndarray,
    state: jnp.ndarray,
    reset: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Sequential WKV7 recurrence — the reference implementation.

  This is the permanent correctness oracle for every faster WKV7 path (e.g. the
  Pallas kernel in `maxtext.kernels.rwkv7_wkv`); it must stay selectable via
  `config.rwkv7_wkv_impl` rather than becoming dead code.

  Args:
    r, w, k, v, a, kk: `(B, T, H, N)` per-timestep projections. `w` is the
      already-exponentiated decay in (0, 1).
    state: `(B, H, N, N)` incoming recurrent state (rows = output dim).
    reset: optional `(B, T)` bool; True at step t zeroes the state entering it
      (a document boundary in a packed row).

  Returns:
    `(out, final_state)` with `out` of shape `(B, T, H, N)`.
  """
  if reset is None:
    reset = jnp.zeros(r.shape[:2], bool)

  def step(carry_state, xs):
    w_t, k_t, v_t, a_t, kk_t, r_t, reset_t = xs
    carry_state = jnp.where(reset_t[:, None, None, None], 0.0, carry_state)
    b_t = kk_t * a_t
    # state @ (-kk): contract over the state's input (column) axis.
    # wkv7_naive serves as a mathematical reference / numerical oracle. Contractions
    # explicitly request Precision.HIGHEST so accelerator backends do not use
    # reduced-precision dot products internally.
    sab = jnp.einsum("bhij,bhj->bhi", carry_state, -kk_t, precision=jax.lax.Precision.HIGHEST)
    carry_state = (
        carry_state * w_t[..., None, :]
        + jnp.einsum("bhi,bhj->bhij", sab, b_t, precision=jax.lax.Precision.HIGHEST)
        + jnp.einsum("bhi,bhj->bhij", v_t, k_t, precision=jax.lax.Precision.HIGHEST)
    )
    y_t = jnp.einsum("bhij,bhj->bhi", carry_state, r_t, precision=jax.lax.Precision.HIGHEST)
    return carry_state, y_t

  time_major = tuple(jnp.swapaxes(t, 0, 1) for t in (w, k, v, a, kk, r, reset))
  final_state, out = jax.lax.scan(step, state, time_major)
  return jnp.swapaxes(out, 0, 1), final_state


# ---------------------------------------------------------------------------
# Initialization, ported from BlinkDL's from-scratch training recipe
# (`RWKV-LM/RWKV-v7/train_temp/src/model.py`): `RWKV_Tmix_x070.__init__` and
# `RWKV_CMix_x070.__init__` set the per-channel, layer-dependent ramps and the
# low-rank factors; `RWKV.generate_init_weight` then re-initializes the
# projection matrices and the GroupNorm scale. Two properties matter for
# training: every low-rank gate has an orthogonal second factor (with both
# factors zero it would be a saddle with zero gradient forever), and `k_k` is
# nonzero (the delta-rule term uses kk twice, so `k_k = 0` would never learn).
# The zero first factors and the zeroed output projections are deliberate:
# they stage activation, they don't disable it.
# ---------------------------------------------------------------------------


def time_mix_init_values(layer_idx: int, num_layers: int, emb: int, head_size: int) -> dict[str, np.ndarray]:
  """Deterministic initial values of a time mix's per-channel parameters."""
  ratio_0_to_1 = layer_idx / (num_layers - 1) if num_layers > 1 else 0.0
  ratio_1_to_almost0 = 1.0 - layer_idx / num_layers
  n = np.arange(emb, dtype=np.float64)
  ddd = n / emb
  linear = n / (emb - 1) - 0.5
  zigzag = ((n % head_size) - (head_size - 1) / 2) / ((head_size - 1) / 2)
  zigzag = zigzag * np.abs(zigzag)
  www = -6 + 6 * (n / (emb - 1)) ** (1 + 1 * ratio_0_to_1**0.3)
  values = {
      "x_r": 1.0 - ddd ** (0.2 * ratio_1_to_almost0),
      "x_w": 1.0 - ddd ** (0.9 * ratio_1_to_almost0),
      "x_k": 1.0 - ddd ** (0.7 * ratio_1_to_almost0),
      "x_v": 1.0 - ddd ** (0.7 * ratio_1_to_almost0),
      "x_a": 1.0 - ddd ** (0.9 * ratio_1_to_almost0),
      "x_g": 1.0 - ddd ** (0.2 * ratio_1_to_almost0),
      "w0": www + 0.5 + zigzag * 2.5,
      "a0": -0.19 + zigzag * 0.3 + linear * 0.4,
      "v0": 0.73 - linear * 0.4,
      "k_k": 0.71 - linear * 0.1,
      "k_a": np.full(emb, 1.02),
      "r_k": np.full((emb // head_size, head_size), -0.04),
      # From `generate_init_weight`: the GroupNorm scale grows with depth.
      "ln_x_scale": np.full(emb, ((1 + layer_idx) / num_layers) ** 0.7),
  }
  return {name: value.astype(np.float32) for name, value in values.items()}


def channel_mix_init_values(layer_idx: int, num_layers: int, emb: int) -> dict[str, np.ndarray]:
  """Deterministic initial values of a channel mix's per-channel parameters."""
  ratio_1_to_almost0 = 1.0 - layer_idx / num_layers
  ddd = np.arange(emb, dtype=np.float64) / emb
  return {"x_k": (1.0 - ddd ** (ratio_1_to_almost0**4)).astype(np.float32)}


def lora_up_gain(rank: int, emb: int) -> float:
  """Gain of BlinkDL's `ortho_init(x, 0.1)` for a `(rank, emb)` second factor."""
  return 0.1 * (math.sqrt(rank / emb) if rank > emb else 1.0)


def _orthogonal(gain: float, key, shape, dtype):
  """An orthogonal matrix times `gain`, computed in float32 (QR rejects bfloat16)."""
  return jax.nn.initializers.orthogonal(gain)(key, shape, jnp.float32).astype(dtype)


def _orthogonal_kernel_init(gain: float):
  """`DenseGeneral` kernel initializer: an orthogonal matrix times `gain`."""

  def init(key, shape, dtype, in_axis, out_axis):
    del in_axis, out_axis
    return _orthogonal(gain, key, shape, dtype)

  return init


def _zeros_kernel_init(key, shape, dtype, in_axis, out_axis):
  del key, in_axis, out_axis
  return jnp.zeros(shape, dtype)


def token_embedding_init(key, shape, dtype=jnp.float32):
  """`generate_init_weight`'s `emb.weight`: uniform in [-1e-4, 1e-4]."""
  return jax.random.uniform(key, shape, dtype, minval=-1e-4, maxval=1e-4)


def logits_kernel_init(vocab_size: int, emb: int):
  """`generate_init_weight`'s `head.weight`: orthogonal, gain `0.5 * sqrt(V / C)` if V > C, else 0.5."""
  return _orthogonal_kernel_init(0.5 * (math.sqrt(vocab_size / emb) if vocab_size > emb else 1.0))


WKV7_IMPLEMENTATIONS = {
    "naive": wkv7_naive,
    "pallas": functools.partial(wkv7_pallas, algorithm="recurrent"),
    "pallas_chunked": functools.partial(wkv7_pallas, algorithm="chunked"),
    "pallas_gpu": wkv7_pallas_gpu,
}


def resolve_wkv7_impl(config: Config, mesh: Mesh | None) -> str:
  """`rwkv7_wkv_impl`, with "autoselected" resolved to the target's kernel.

  The same rule as `attention_kernel="autoselected"`: pick from the mesh's
  platform, not the default backend, so AOT-compiling for a TPU topology from
  a CPU host still selects the TPU kernel. The GPU kernel has no packed
  sequence resets, so it is only chosen when none are asked for.
  """
  if config.rwkv7_wkv_impl != "autoselected":
    return config.rwkv7_wkv_impl
  platform = "cpu" if mesh is None else mesh.devices.flat[0].platform
  if platform == "tpu":
    return "pallas_chunked"
  if platform == "gpu" and not config.rwkv7_segment_resets:
    return "pallas_gpu"
  return "naive"


class Rwkv7TimeMix(nnx.Module):
  """RWKV-7 time mixing: token shift, dynamic (LoRA-parameterized) gates, WKV7."""

  def __init__(
      self,
      config: Config,
      model_mode: str,
      mesh: Mesh,
      rngs: nnx.Rngs,
      layer_idx: int,
      quant: None | Quant = None,
  ):
    self.config = config
    self.mesh = mesh
    self.quant = quant
    self.is_first_layer = layer_idx == 0

    cfg = config
    emb = cfg.emb_dim
    self.head_size = cfg.rwkv7_head_size
    self.num_heads = emb // self.head_size
    init = time_mix_init_values(layer_idx, cfg.num_decoder_layers, emb, self.head_size)

    def mix_param(name, axes=("embed",)):
      return nnx.Param(jnp.asarray(init[name], cfg.weight_dtype), sharding=axes)

    # Token-shift interpolation weights, one per projection.
    self.x_r = mix_param("x_r")
    self.x_w = mix_param("x_w")
    self.x_k = mix_param("x_k")
    self.x_v = mix_param("x_v")
    self.x_a = mix_param("x_a")
    self.x_g = mix_param("x_g")

    # Low-rank ("LoRA") parameterizations of the per-token dynamic gates, each a
    # bias plus a rank-limited MLP: decay w, in-context learning rate a, value
    # residual mix v, output gate g. The first factor starts at zero and the
    # second orthogonal, so each gate starts at its bias and still gets a gradient.
    def lora(rank, bias_name=None):
      down = nnx.Param(jnp.zeros((emb, rank), cfg.weight_dtype), sharding=("embed", None))
      up = nnx.Param(
          _orthogonal(lora_up_gain(rank, emb), rngs.params(), (rank, emb), cfg.weight_dtype), sharding=(None, "embed")
      )
      bias = mix_param(bias_name) if bias_name else None
      return down, up, bias

    self.w1, self.w2, self.w0 = lora(cfg.rwkv7_decay_lora_rank, "w0")
    self.a1, self.a2, self.a0 = lora(cfg.rwkv7_iclr_lora_rank, "a0")
    self.v1, self.v2, self.v0 = lora(cfg.rwkv7_value_lora_rank, "v0")
    self.g1, self.g2, _ = lora(cfg.rwkv7_gate_lora_rank)

    # Per-channel key shaping and the per-head r*k bonus term.
    self.k_k = mix_param("k_k")
    self.k_a = mix_param("k_a")
    self.r_k = mix_param("r_k", axes=("heads", "kv"))

    def proj(kernel_axes, kernel_init):
      return DenseGeneral(
          in_features_shape=emb,
          out_features_shape=emb,
          dtype=cfg.dtype,
          weight_dtype=cfg.weight_dtype,
          kernel_axes=kernel_axes,
          kernel_init=kernel_init,
          use_bias=False,
          shard_mode=cfg.shard_mode,
          matmul_precision=cfg.matmul_precision,
          mesh=mesh,
          quant=quant,
          rngs=rngs,
      )

    self.receptance = proj(("embed", "heads"), _orthogonal_kernel_init(1.0))
    self.key = proj(("embed", "heads"), _orthogonal_kernel_init(0.1))
    self.value = proj(("embed", "heads"), _orthogonal_kernel_init(1.0))
    self.output = proj(("heads", "embed"), _zeros_kernel_init)

    self.ln_x_scale = mix_param("ln_x_scale", axes=("norm",))
    self.ln_x_bias = nnx.Param(jnp.zeros((emb,), cfg.weight_dtype), sharding=("norm",))

    self.wkv_impl_name = resolve_wkv7_impl(cfg, mesh)
    self.wkv_impl = WKV7_IMPLEMENTATIONS[self.wkv_impl_name]
    if self.wkv_impl_name != "naive":
      # Decide interpret mode from the mesh rather than the default backend, so
      # that AOT-compiling for a TPU topology from a CPU host still lowers the
      # real kernel (same rule as megablox in `layers/moe.py`).
      self.wkv_impl = functools.partial(
          self.wkv_impl,
          # The GPU kernel's chunk size is bounded by its backward's
          # conditioning; under "autoselected" a larger request is clamped
          # rather than rejected (asking for `pallas_gpu` by name is not).
          chunk_size=min(cfg.rwkv7_wkv_chunk_size, MAX_GPU_CHUNK_SIZE)
          if self.wkv_impl_name == "pallas_gpu"
          else cfg.rwkv7_wkv_chunk_size,
          # Off-TPU the kernel is interpreted. The model uses the discharge
          # interpreter because the TPU interpreter's IO callbacks can't be
          # differentiated under remat (`jax.checkpoint`, on by default); the
          # kernel tests keep the stricter TPU interpreter.
          interpret=False if mesh.devices.flat[0].platform in ("tpu", "gpu") else "discharge",
      )

    # Decode state: the recurrent matrix plus the previous token's hidden state.
    # The cache-batch axis annotation is what MaxEngine's insert paths look up.
    if model_mode != MODEL_MODE_TRAIN:
      batch_size, _ = max_utils.get_batch_seq_len_for_mode(cfg, model_mode)
      cache_batch = _cache_batch_axis(model_mode)
      self.recurrent_state = nnx.Cache(
          jnp.zeros((batch_size, self.num_heads, self.head_size, self.head_size), jnp.float32),
          out_sharding=(cache_batch, CACHE_HEADS, None, None),
      )
      self.conv_state = nnx.Cache(jnp.zeros((batch_size, emb), cfg.dtype), out_sharding=(cache_batch, None))
    else:
      self.recurrent_state = None
      self.conv_state = None

  def _apply_wkv(self, r, w, k, v, a, kk, state, reset):
    """Runs the WKV7 recurrence, under `shard_map` for the Pallas kernels.

    Mosaic kernels can't be partitioned automatically, so on a multi-device mesh
    the kernel runs per shard: batch and heads are split like GDN's
    (`KV_BATCH`, `KV_HEAD`). The sequence stays whole on every shard, since the
    recurrence carries state across it. The reference scan is plain XLA and
    partitions on its own.
    """
    if self.wkv_impl_name == "naive" or self.mesh is None:
      return self.wkv_impl(r, w, k, v, a, kk, state, reset=reset)

    rules = get_logical_axis_rules()
    seq_pspec = logical_to_mesh_axes((KV_BATCH, None, KV_HEAD, None), mesh=self.mesh, rules=rules)
    state_pspec = logical_to_mesh_axes((KV_BATCH, KV_HEAD, None, None), mesh=self.mesh, rules=rules)
    reset_pspec = logical_to_mesh_axes((KV_BATCH, None), mesh=self.mesh, rules=rules)
    # The GPU kernel takes no resets at all (config validation pairs it with
    # `rwkv7_segment_resets=false`), so it is not handed an all-zero column.
    takes_reset = self.wkv_impl_name != "pallas_gpu"
    if reset is None:
      reset = jnp.zeros(r.shape[:2], bool)
    # Batch axes that don't divide this call's batch (e.g. batch-1 prefill) fall back to replication.
    seq_pspec, state_pspec, reset_pspec = (
        remove_incompatible_mesh_axes_from_partition_spec(spec, x.shape, self.mesh, dims=(0,), allow_remove_axes=True)
        for spec, x in ((seq_pspec, r), (state_pspec, state), (reset_pspec, reset))
    )
    seq_inputs = (r, w, k, v, a, kk)
    if self.config.shard_mode == ShardMode.EXPLICIT:
      # shard_map doesn't reshard operands whose layout differs from in_specs.
      seq_inputs = tuple(jax.sharding.reshard(x, seq_pspec) for x in seq_inputs)
      state = jax.sharding.reshard(state, state_pspec)
      reset = jax.sharding.reshard(reset, reset_pspec)

    sharded = jax.shard_map(
        (lambda *xs: self.wkv_impl(*xs[:7], reset=xs[7])) if takes_reset else self.wkv_impl,
        mesh=self.mesh,
        in_specs=(seq_pspec,) * 6 + (state_pspec,) + ((reset_pspec,) if takes_reset else ()),
        out_specs=(seq_pspec, state_pspec),
        check_vma=False,
    )
    return sharded(*seq_inputs, state, *((reset,) if takes_reset else ()))

  def _token_shift(self, x: jnp.ndarray, shift_in: jnp.ndarray) -> jnp.ndarray:
    """Returns the sequence shifted right by one, seeded with `shift_in` (B, E)."""
    return jnp.concatenate([shift_in[:, None, :], x[:, :-1, :]], axis=1)

  def __call__(
      self,
      x: jnp.ndarray,
      v_first: jnp.ndarray | None,
      valid: jnp.ndarray | None = None,
      reset: jnp.ndarray | None = None,
  ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Applies time mixing.

    Args:
      x: `(B, T, E)` normalized layer input.
      v_first: `(B, T, E)` layer-0 value projection, or None on layer 0.
      valid: optional `(B, T)` bool mask of real (non-padding) tokens, contiguous
        from the start of each row. Padding steps leave the recurrent state and
        the token-shift cache untouched.
      reset: optional `(B, T)` bool, True where a new document starts in a packed
        row: both the recurrent state and the token shift start from zero there.

    Returns:
      `(output, v_first)`.
    """
    cfg = self.config
    x = jnp.asarray(x, jnp.float32)
    batch, seq_len, emb = x.shape
    heads, head_size = self.num_heads, self.head_size

    if self.conv_state is not None:
      shift_in = jnp.asarray(self.conv_state[...], jnp.float32)[:batch]
    else:
      shift_in = jnp.zeros((batch, emb), jnp.float32)
    x_prev = self._token_shift(x, shift_in)
    if reset is not None:
      x_prev = jnp.where(reset[..., None], 0.0, x_prev)

    # Token shift: interpolate between this token and the previous one, with a
    # separately learned interpolation weight per projection.
    delta = x_prev - x
    xr = x + delta * self.x_r[...]
    xw = x + delta * self.x_w[...]
    xk = x + delta * self.x_k[...]
    xv = x + delta * self.x_v[...]
    xa = x + delta * self.x_a[...]
    xg = x + delta * self.x_g[...]

    r = self.receptance(xr)
    k = self.key(xk)
    v = self.value(xv)

    def _contract(a, b):
      return jnp.matmul(a, b, precision=cfg.matmul_precision)

    # Decay: w in (0, 1), applied multiplicatively to the state each step. The
    # double exponential keeps w strictly inside (0, 1) for any input.
    w_gate = _contract(jnp.tanh(_contract(xw, self.w1[...])), self.w2[...])
    w = -jax.nn.softplus(-(self.w0[...] + w_gate)) - 0.5
    w = jnp.exp(-jnp.exp(w))

    if self.is_first_layer:
      v_first = v
    else:
      # Later layers mix in layer 0's value, gated per channel.
      v_gate = _contract(_contract(xv, self.v1[...]), self.v2[...])
      v = v + (v_first - v) * jax.nn.sigmoid(self.v0[...] + v_gate)

    # In-context learning rate: strength of the per-step removal along kk.
    a_gate = _contract(_contract(xa, self.a1[...]), self.a2[...])
    a = jax.nn.sigmoid(self.a0[...] + a_gate)
    g = _contract(jax.nn.sigmoid(_contract(xg, self.g1[...])), self.g2[...])

    kk = k * self.k_k[...]
    kk = kk.reshape(batch, seq_len, heads, head_size)
    # Equivalent to torch's `F.normalize(kk, p=2, dim=-1)` (which divides by
    # `max(norm, 1e-12)`), but clamped before the square root so the gradient
    # stays finite for an all-zero vector instead of going through d/dx sqrt(0).
    kk = kk * jax.lax.rsqrt(jnp.maximum(jnp.sum(jnp.square(kk), axis=-1, keepdims=True), 1e-24))
    k = k * (1 + (a - 1) * self.k_a[...])

    r_h, w_h, k_h, v_h, a_h = (t.reshape(batch, seq_len, heads, head_size) for t in (r, w, k, v, a))

    if self.recurrent_state is not None:
      state_in = jnp.asarray(self.recurrent_state[...], jnp.float32)[:batch]
    else:
      state_in = jnp.zeros((batch, heads, head_size, head_size), jnp.float32)

    wkv_w, wkv_k, wkv_v, wkv_kk = w_h, k_h, v_h, kk
    if valid is not None:
      # Padding steps become identities (w = 1, and k = v = kk = 0 so neither the
      # removal nor the write term touches the state), the same steps the
      # kernels pad ragged tails with.
      keep = valid[:, :, None, None]
      wkv_w = jnp.where(keep, w_h, 1.0)
      wkv_k, wkv_v, wkv_kk = (jnp.where(keep, t, 0.0) for t in (k_h, v_h, kk))

    out, new_state = self._apply_wkv(r_h, wkv_w, wkv_k, wkv_v, a_h, wkv_kk, state_in, reset)

    out = _head_group_norm(out, self.ln_x_scale[...], self.ln_x_bias[...], cfg.rwkv7_groupnorm_epsilon)

    # Bonus term: a direct per-head r*k contribution bypassing the state.
    bonus = jnp.sum(r_h * k_h * self.r_k[...], axis=-1, keepdims=True) * v_h
    out = out + bonus.reshape(batch, seq_len, emb)

    if self.recurrent_state is not None:
      valid_lens = None if valid is None else jnp.sum(valid, axis=1)
      self.recurrent_state[...] = new_state
      self.conv_state[...] = jnp.asarray(_last_valid(x, shift_in, valid_lens), self.conv_state[...].dtype)

    return self.output(out * g), v_first


class Rwkv7ChannelMix(nnx.Module):
  """RWKV-7 channel mixing: token-shifted ReLU-squared MLP."""

  def __init__(
      self,
      config: Config,
      model_mode: str,
      mesh: Mesh,
      rngs: nnx.Rngs,
      layer_idx: int,
      quant: None | Quant = None,
  ):
    self.config = config
    cfg = config
    emb = cfg.emb_dim
    init = channel_mix_init_values(layer_idx, cfg.num_decoder_layers, emb)

    self.x_k = nnx.Param(jnp.asarray(init["x_k"], cfg.weight_dtype), sharding=("embed",))
    self.key = DenseGeneral(
        in_features_shape=emb,
        out_features_shape=cfg.mlp_dim,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        kernel_axes=("embed", "mlp"),
        kernel_init=_orthogonal_kernel_init(1.0),
        use_bias=False,
        shard_mode=cfg.shard_mode,
        matmul_precision=cfg.matmul_precision,
        mesh=mesh,
        quant=quant,
        rngs=rngs,
    )
    self.value = DenseGeneral(
        in_features_shape=cfg.mlp_dim,
        out_features_shape=emb,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        kernel_axes=("mlp", "embed"),
        kernel_init=_zeros_kernel_init,
        use_bias=False,
        shard_mode=cfg.shard_mode,
        matmul_precision=cfg.matmul_precision,
        mesh=mesh,
        quant=quant,
        rngs=rngs,
    )

    if model_mode != MODEL_MODE_TRAIN:
      batch_size, _ = max_utils.get_batch_seq_len_for_mode(cfg, model_mode)
      self.conv_state = nnx.Cache(
          jnp.zeros((batch_size, emb), cfg.dtype), out_sharding=(_cache_batch_axis(model_mode), None)
      )
    else:
      self.conv_state = None

  def __call__(self, x: jnp.ndarray, valid: jnp.ndarray | None = None, reset: jnp.ndarray | None = None) -> jnp.ndarray:
    """Applies channel mixing; `valid` and `reset` are as in `Rwkv7TimeMix.__call__`."""
    x = jnp.asarray(x, jnp.float32)
    batch, _, emb = x.shape
    if self.conv_state is not None:
      shift_in = jnp.asarray(self.conv_state[...], jnp.float32)[:batch]
    else:
      shift_in = jnp.zeros((batch, emb), jnp.float32)
    x_prev = jnp.concatenate([shift_in[:, None, :], x[:, :-1, :]], axis=1)
    if reset is not None:
      x_prev = jnp.where(reset[..., None], 0.0, x_prev)

    k = x + (x_prev - x) * self.x_k[...]
    k = jnp.square(jax.nn.relu(self.key(k)))

    if self.conv_state is not None:
      valid_lens = None if valid is None else jnp.sum(valid, axis=1)
      self.conv_state[...] = jnp.asarray(_last_valid(x, shift_in, valid_lens), self.conv_state[...].dtype)
    return self.value(k)


class Rwkv7DecoderLayer(nnx.Module):
  """One RWKV-7 block: LayerNorm -> time mix -> residual -> LayerNorm -> channel mix -> residual."""

  def __init__(
      self,
      config: Config,
      model_mode: str,
      mesh: Mesh,
      rngs: nnx.Rngs,
      layer_idx: int,
      quant: None | Quant = None,
  ):
    self.config = config
    self.mesh = mesh
    self.is_first_layer = is_first_layer = layer_idx == 0

    if model_mode == MODEL_MODE_PREFILL:
      self.activation_axis_names = ("activation_batch", "prefill_activation_norm_length", "activation_embed")
    else:
      self.activation_axis_names = ("activation_batch", "activation_norm_length", "activation_embed")

    def norm():
      return Rwkv7LayerNorm(
          num_features=config.emb_dim,
          epsilon=config.normalization_layer_epsilon,
          dtype=config.dtype,
          weight_dtype=config.weight_dtype,
          kernel_axes=("norm",),
          rngs=rngs,
      )

    # Layer 0 normalizes the embedding output before anything else.
    self.ln0 = norm() if is_first_layer else None
    self.ln1 = norm()
    self.ln2 = norm()

    self.att = Rwkv7TimeMix(config=config, model_mode=model_mode, mesh=mesh, rngs=rngs, layer_idx=layer_idx, quant=quant)
    self.ffn = Rwkv7ChannelMix(
        config=config, model_mode=model_mode, mesh=mesh, rngs=rngs, layer_idx=layer_idx, quant=quant
    )

  def __call__(
      self,
      inputs: jnp.ndarray,
      decoder_segment_ids=None,
      decoder_positions=None,
      deterministic: bool = True,
      model_mode: str = MODEL_MODE_TRAIN,
      v_first: jnp.ndarray | None = None,
      **unused_kwargs,
  ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Runs one block.

    Args:
      inputs: `(B, T, E)` residual stream.
      decoder_segment_ids: optional `(B, T)`; 0 marks padding. Padding must be
        contiguous at the end of each row (MaxEngine's prefill layout), and
        leaves the recurrent caches as the row's real tokens left them. In a
        packed row (several nonzero ids) each new id starts from a zero state
        and a zero token shift, unless `rwkv7_segment_resets=false` (BlinkDL's
        continuous-stream training). Multi-prompt `MaxEngine.prefill_concat` is
        still rejected: it would need one cache per prompt.
      v_first: layer 0's value projection, threaded through the stack by the
        decoder; None on layer 0, where it is produced.

    Returns:
      `(layer_output, v_first)`.
    """
    del decoder_positions, deterministic, model_mode, unused_kwargs
    valid = None if decoder_segment_ids is None else decoder_segment_ids != 0
    reset = None
    if decoder_segment_ids is not None and self.config.rwkv7_segment_resets:
      reset = segment_resets(decoder_segment_ids)

    x = inputs
    if self.ln0 is not None:
      x = self.ln0(x)

    att_out, v_first = self.att(self.ln1(x), v_first, valid, reset)
    x = x + att_out
    x = x + self.ffn(self.ln2(x), valid, reset)
    return x, v_first
