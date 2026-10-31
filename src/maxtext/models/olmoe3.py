# Copyright 2023-2026 Google LLC
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

"""OLMoE3 decoder layer (AI2's hybrid KDA + latent-MoE architecture).

Reference: ``allenai/OLMo-core@akshitab/standalone``,
``src/scripts/standalone/standalone_model.py``.

Layer structure, matching the reference block:

* Mixer alternates Kimi Delta Attention (KDA, linear attention) with full
  attention. Full-attention layers are those where
  ``(layer_idx + 1) % inhomogeneous_layer_cycle_interval == 0``, which at
  interval 5 reproduces the reference's ``range(4, n_layers, 5)``.
* Four RMSNorms per block (pre and post, around both the mixer and the FFN).
  The post-norm is applied to the sublayer output *before* the residual add.
* FFN is a shared SwiGLU plus, on all layers past ``first_num_dense_layers``,
  routed experts that run entirely in a compressed latent space: the block
  projects ``emb_dim -> moe_expert_input_dim``, runs the experts there, and
  projects back. The router still scores the full-width residual.

Routing matches the reference: top-k weights renormalized to sum to ``top_k``
(``restore_weight_scale``), a batch-level globally-balanced load-balancing
loss, a 1e-5 router z-loss, and the EMo document-pool router (``emo_enabled``,
with the canonical pools min=top_k, max=eval=num_experts).
"""

# pylint: disable=unnecessary-lambda-assignment

import contextlib
import contextvars
import functools
from typing import Any

from jax.sharding import Mesh, PartitionSpec
import jax
import jax.numpy as jnp
import numpy as np

from flax import linen as nn
from flax import nnx
from jax.ad_checkpoint import checkpoint_name

from maxtext.common.common_types import Config, BATCH, LENGTH, EMBED
from maxtext.kernels import head_norm  # pylint: disable=no-name-in-module
from maxtext.kernels import kda  # pylint: disable=no-name-in-module
from maxtext.kernels.kda import intra as kda_intra  # pylint: disable=no-name-in-module
from maxtext.kernels.kda import scan as kda_scan  # pylint: disable=no-name-in-module
from maxtext.layers import attentions
from maxtext.layers import initializers as max_initializers
from maxtext.layers import moe
from maxtext.layers import nnx_wrappers
from maxtext.layers import linears
from maxtext.layers import quantizations
from maxtext.layers.linears import DenseGeneral, MlpBlock
from maxtext.layers.normalizations import RMSNorm
from maxtext.layers.quantizations import AqtQuantization as Quant
from maxtext.models import olmoe3_roles
from maxtext.utils import max_utils
from maxtext.utils.sharding import mesh_axes_for_dim

# The reference uses a separate, looser epsilon for the KDA output norm than for
# the block RMSNorms (kda_norm_eps=1e-5 vs rms_norm_eps=1e-6). Kept as a module
# constant rather than a config key until a model needs to vary it.
_KDA_NORM_EPS = 1e-5

# Reference init std and router z-loss weight (fixed across the ladder).
_INIT_STD = 0.02
_ROUTER_Z_LOSS_WEIGHT = 1e-5

# Set while tracing an eval program that should route on the EMO eval pool. MaxText calls the model the same way in
# train and eval steps (and enable_dropout is off in training too), so the eval step says so explicitly.
_EMO_EVAL_POOLS = contextvars.ContextVar("olmoe3_emo_eval_pools", default=False)


@contextlib.contextmanager
def emo_eval_pools():
  """Within this context, EMO routing uses ``emo_eval_document_expert_pool`` instead of sampling pool sizes."""
  token = _EMO_EVAL_POOLS.set(True)
  try:
    yield
  finally:
    _EMO_EVAL_POOLS.reset(token)


def olmoe3_init(key, shape, dtype=jnp.float32, in_axis=None, out_axis=None):
  """The reference init: truncated normal, std ``_INIT_STD``, bounds +/-3 std.

  Every embedding, projection, router, and expert weight in the reference uses
  this fixed-std init (fan-in plays no part). The trailing axis arguments are
  accepted and ignored so the same function satisfies both MaxText initializer
  signatures (``NdInitializer`` and plain ``Initializer``).
  """
  del in_axis, out_axis
  return jax.nn.initializers.truncated_normal(stddev=_INIT_STD, lower=-3.0, upper=3.0)(key, shape, dtype)


def nanochat_std(fan_in: int, fan_out: int) -> float:
  """OLMo-core's ``nanochat_std``: ``1/sqrt(fan_in) * min(1, sqrt(fan_out/fan_in))``."""
  return fan_in**-0.5 * min(1.0, (fan_out / fan_in) ** 0.5)


def nanochat_init(key, shape, dtype=jnp.float32, in_axis=None, out_axis=None):
  """OLMo-core's ``InitMethod.nanochat`` for one weight: truncated normal, bounds +/-3 std, std ``nanochat_std``.

  Fan-in and fan-out are the products of the kernel's in and out axes (a 2-D kernel without
  axes is ``[in, out]``). A kernel with more dims than those axes cover is a stack of matrices
  over its leading dims: MaxText's MoE passes its ``[experts, in, out]`` kernels
  ``in_axis=(0,)`` and ``out_axis=(1,)``, and the reference inits each expert as its own
  ``in -> out`` matrix, so the last two dims are used.
  """
  if in_axis is None or out_axis is None:
    in_axis, out_axis = (0,), (1,)
  in_axis, out_axis = tuple(np.atleast_1d(in_axis)), tuple(np.atleast_1d(out_axis))
  if len(in_axis) + len(out_axis) < len(shape):
    fan_in, fan_out = shape[-2], shape[-1]
  else:
    fan_in = int(np.prod([shape[i] for i in in_axis]))
    fan_out = int(np.prod([shape[i] for i in out_axis]))
  std = nanochat_std(fan_in, fan_out)
  return jax.nn.initializers.truncated_normal(stddev=std, lower=-3.0, upper=3.0)(key, shape, dtype)


def olmoe3_kernel_init(config: Config):
  """The weight-matrix init that ``olmo_init_method`` selects (also the LM head's)."""
  return nanochat_init if config.olmo_init_method == "nanochat" else olmoe3_init


def olmoe3_embedding_init(config: Config):
  """The token-embedding init: the block init under "normal"; an untruncated std-1.0 normal under "nanochat"."""
  return jax.nn.initializers.normal(stddev=1.0) if config.olmo_init_method == "nanochat" else olmoe3_init


def _olmoe3_attention_init(config: Config):
  """The attention projections' init.

  Under "nanochat" the query kernel ``[emb, heads, 2 * head_dim]`` holds each head's query and
  output gate, the reference's separate ``w_q`` and ``w_g``: two projections of ``heads * head_dim``
  outputs each, so its fan-out is half the stored width. Key, value and output kernels use their axes.
  """
  if config.olmo_init_method != "nanochat":
    return olmoe3_init
  head_dim = config.head_dim

  def init(key, shape, dtype=jnp.float32, in_axis=None, out_axis=None):
    if len(shape) == 3 and shape[-1] == 2 * head_dim and tuple(np.atleast_1d(in_axis)) == (0,):
      std = nanochat_std(shape[0], shape[1] * head_dim)
      return jax.nn.initializers.truncated_normal(stddev=std, lower=-3.0, upper=3.0)(key, shape, dtype)
    return nanochat_init(key, shape, dtype, in_axis, out_axis)

  return init


def _attention_logit_scale(cfg):
  """Full attention's query scale: head_dim^-0.5, times sqrt(base head_dim / head_dim) under `head_dim_transfer` with
  a muP or CompletedP base model (muP's 1/head_dim beyond the base's head_dim; applied after the QK norm)."""
  scale = cfg.head_dim**-0.5
  name = getattr(cfg, "completedp_base_model_name", "") or getattr(cfg, "mup_base_model_name", "")
  if getattr(cfg, "head_dim_transfer", False) and name:
    scale *= (olmoe3_roles.load_base_config(name).head_dim / cfg.head_dim) ** 0.5
  return scale


def _completedp_base(config):
  """The base model's config under CompletedP (`completedp_base_model_name`), else None."""
  name = getattr(config, "completedp_base_model_name", "")
  if not name:
    return None
  if config.olmo_init_method != "normal":
    raise ValueError(f"CompletedP rescales the normal init; olmo_init_method={config.olmo_init_method} isn't supported.")
  return olmoe3_roles.load_base_config(name)


def _scaled(init, factor):
  if factor == 1.0:
    return init

  def scaled_init(*args, **kwargs):
    return init(*args, **kwargs) * factor

  return scaled_init


def _matrix_init(config, sub, dense=False):
  """The init of the in-layer matrix at `sub` (e.g. "mixer/w_out/kernel"): `olmo_init_method`'s, times CompletedP's
  factor (`sqrt(base fan-in / fan-in)`, or `base fan-in / fan-in` for a readout matrix) when it's on."""
  base = _completedp_base(config)
  init = olmoe3_kernel_init(config)
  return init if base is None else _scaled(init, olmoe3_roles.completedp_init_factor(config, base, sub, dense))


def _matrix_init_by_shape(config, pick, dense=False):
  """`_matrix_init` for a module that inits several matrices with one initializer: `pick(shape)` names the matrix."""
  base = _completedp_base(config)
  init = olmoe3_kernel_init(config)
  if base is None:
    return init

  def by_shape(key, shape, dtype=jnp.float32, in_axis=None, out_axis=None):
    factor = olmoe3_roles.completedp_init_factor(config, base, pick(shape), dense)
    w = init(key, shape, dtype, in_axis, out_axis)
    return w if factor == 1.0 else w * factor

  return by_shape


def olmoe3_lm_head_init(config):
  """The untied LM head's init: the block weights', times CompletedP's base emb_dim / emb_dim when it's on."""
  base = _completedp_base(config)
  init = olmoe3_kernel_init(config)
  return init if base is None else _scaled(init, olmoe3_roles.completedp_init_factor(config, base, olmoe3_roles.LM_HEAD_KEY))


def _dense(config, in_features, out_features, kernel_axes, quant, rngs, use_bias=False, mesh=None, kernel_init=None):
  """DenseGeneral with the config-derived arguments this model always passes."""
  return DenseGeneral(
      in_features_shape=in_features,
      out_features_shape=out_features,
      axis=-1,
      kernel_init=kernel_init or olmoe3_kernel_init(config),
      kernel_axes=kernel_axes,
      dtype=config.dtype,
      weight_dtype=config.weight_dtype,
      quant=quant,
      shard_mode=config.shard_mode,
      matmul_precision=config.matmul_precision,
      use_bias=use_bias,
      mesh=mesh,
      rngs=rngs,
  )


def causal_depthwise_conv(x: jnp.ndarray, weight: jnp.ndarray, segment_ids: None | jnp.ndarray) -> jnp.ndarray:
  """Short causal depthwise convolution over time, followed by SiLU.

  Implemented as a sum of lagged copies rather than a conv op so that packed
  documents can be masked per lag: a token never convolves with tokens from a
  preceding document. This is the unfused equivalent of passing ``cu_seqlens``
  to OLMo-core's packed-document convolution kernel.

  Args:
    x: Input of shape ``[batch, length, width]``.
    weight: Depthwise kernel of shape ``[width, kernel_size]``.
    segment_ids: Optional ``[batch, length]`` packing segment ids.

  Returns:
    ``[batch, length, width]``, SiLU applied.
  """
  seq_len = x.shape[1]
  kernel_size = weight.shape[-1]
  acc = jnp.zeros_like(x)
  for tap in range(kernel_size):
    lag = kernel_size - 1 - tap
    shifted = jnp.pad(x, ((0, 0), (lag, 0), (0, 0)))[:, :seq_len, :]
    if segment_ids is not None and lag > 0:
      # -1 marks the left pad so it never matches a real segment id.
      shifted_ids = jnp.pad(segment_ids, ((0, 0), (lag, 0)), constant_values=-1)[:, :seq_len]
      shifted = jnp.where((shifted_ids == segment_ids)[..., None], shifted, 0.0)
    acc = acc + shifted * weight[:, tap]
  return jax.nn.silu(acc)


class OLMoE3KimiDeltaAttention(nnx.Module):
  """Kimi Delta Attention: gated delta rule with per-channel (vector) decay.

  This differs from MaxText's existing GatedDeltaNet in the decay term. GDN
  learns one decay scalar per head (``A_log`` shaped ``[num_heads]``); KDA
  produces a decay per key channel from a low-rank projection, shaped
  ``[batch, length, heads, key_head_dim]``. With scalar decay the intra-chunk
  term collapses to a mask on ``K K^T``; per channel it does not, but it still
  factors once the cumulative decay is folded into the operands, which is what
  ``_delta_rule_chunked`` does.

  ``_delta_rule_scan`` is the unfused reference. It is kept because the chunked
  rule is checked against it, and because it handles sequence lengths that are
  not a multiple of ``gdn_chunk_size``.
  """

  def __init__(self, config: Config, mesh: Mesh, quant: None | Quant = None, *, rngs: nnx.Rngs):
    self.config = config
    self.mesh = mesh
    cfg = config

    self.num_heads = cfg.gdn_num_value_heads
    self.head_k_dim = cfg.gdn_key_head_dim
    self.head_v_dim = cfg.gdn_value_head_dim
    key_width = self.num_heads * self.head_k_dim
    value_width = self.num_heads * self.head_v_dim
    conv_size = cfg.gdn_conv_kernel_dim

    self.w_q = _dense(cfg, cfg.emb_dim, key_width, ("embed", "mlp"), quant, rngs, mesh=mesh, kernel_init=_matrix_init(cfg, "mixer/w_q/kernel"))
    self.w_k = _dense(cfg, cfg.emb_dim, key_width, ("embed", "mlp"), quant, rngs, mesh=mesh, kernel_init=_matrix_init(cfg, "mixer/w_k/kernel"))
    self.w_v = _dense(cfg, cfg.emb_dim, value_width, ("embed", "mlp"), quant, rngs, mesh=mesh, kernel_init=_matrix_init(cfg, "mixer/w_v/kernel"))

    # Under "nanochat" each depthwise conv is a square conv_size -> conv_size map per channel.
    conv_std = nanochat_std(conv_size, conv_size) if cfg.olmo_init_method == "nanochat" else _INIT_STD
    conv_init = jax.nn.initializers.truncated_normal(stddev=conv_std, lower=-3.0, upper=3.0)
    self.q_conv = nnx.Param(conv_init(rngs.params(), (key_width, conv_size), cfg.weight_dtype))
    self.k_conv = nnx.Param(conv_init(rngs.params(), (key_width, conv_size), cfg.weight_dtype))
    self.v_conv = nnx.Param(conv_init(rngs.params(), (value_width, conv_size), cfg.weight_dtype))

    # Low-rank decay projection: emb -> head_v_dim -> key_width.
    self.f_proj_1 = _dense(cfg, cfg.emb_dim, self.head_v_dim, ("embed", "mlp"), quant, rngs, mesh=mesh, kernel_init=_matrix_init(cfg, "mixer/f_proj_1/kernel"))
    self.f_proj_2 = _dense(cfg, self.head_v_dim, key_width, ("mlp", "embed"), quant, rngs, mesh=mesh, kernel_init=_matrix_init(cfg, "mixer/f_proj_2/kernel"))
    self.w_b = _dense(cfg, cfg.emb_dim, self.num_heads, ("embed", None), quant, rngs, mesh=mesh, kernel_init=_matrix_init(cfg, "mixer/w_b/kernel"))

    a_log_init = nnx.initializers.uniform(scale=15.0)  # U[1, 16) after the +1 below
    self.A_log = nnx.Param(jnp.log(1.0 + a_log_init(rngs.params(), (self.num_heads,), jnp.float32)))
    self.dt_bias = nnx.Param(jnp.zeros((key_width,), jnp.float32))

    # Low-rank output gate: emb -> head_v_dim -> value_width (this one has a bias).
    self.g_proj_1 = _dense(cfg, cfg.emb_dim, self.head_v_dim, ("embed", "mlp"), quant, rngs, mesh=mesh, kernel_init=_matrix_init(cfg, "mixer/g_proj_1/kernel"))
    self.g_proj_2 = _dense(
        cfg,
        self.head_v_dim,
        value_width,
        ("mlp", "embed"),
        quant,
        rngs,
        use_bias=True,
        mesh=mesh,
        kernel_init=_matrix_init(cfg, "mixer/g_proj_2/kernel"),
    )

    self.o_norm = RMSNorm(
        num_features=self.head_v_dim,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        kernel_axes=("norm",),
        epsilon=_KDA_NORM_EPS,
        rngs=rngs,
    )
    self.w_out = _dense(
        cfg, value_width, cfg.emb_dim, ("mlp", "embed"), quant, rngs, mesh=mesh, kernel_init=_matrix_init(cfg, "mixer/w_out/kernel")
    )
    # Quantized kernels go through DenseGeneral's own dot_general, so they stay separate.
    self.fused_input_proj = bool(cfg.kda_fused_input_proj) and quant is None

  def __call__(self, x: jnp.ndarray, decoder_segment_ids: None | jnp.ndarray = None) -> jnp.ndarray:
    batch, seq_len, _ = x.shape
    heads, dk, dv = self.num_heads, self.head_k_dim, self.head_v_dim

    # An fp32 conv weight otherwise promotes q/k/v (and their kernel residuals) to fp32.
    cast = self.config.kda_conv_in_compute_dtype
    q_conv, k_conv, v_conv = (w[...].astype(x.dtype) if cast else w[...] for w in (self.q_conv, self.k_conv, self.v_conv))
    if self.fused_input_proj:
      q_in, k_in, v_in, f_in, g_in, b_in = self._fused_input_proj(x)
    else:
      q_in, k_in, v_in = self.w_q(x), self.w_k(x), self.w_v(x)
      f_in, g_in, b_in = self.f_proj_1(x), self.g_proj_1(x), self.w_b(x)
      # Named so the custom remat policy can keep the projections (query/key/value_proj=device)
      # and skip recomputing them and their weight all-gathers in the bwd.
      q_in, k_in, v_in = (
          checkpoint_name(t, n) for t, n in ((q_in, "query_proj"), (k_in, "key_proj"), (v_in, "value_proj"))
      )
    if self._use_flat_head_layout(seq_len):
      return self._flat_head_forward(q_in, k_in, v_in, f_in, g_in, b_in, q_conv, k_conv, v_conv, decoder_segment_ids)
    q = causal_depthwise_conv(q_in, q_conv, decoder_segment_ids).reshape(batch, seq_len, heads, dk)
    k = causal_depthwise_conv(k_in, k_conv, decoder_segment_ids).reshape(batch, seq_len, heads, dk)
    v = causal_depthwise_conv(v_in, v_conv, decoder_segment_ids).reshape(batch, seq_len, heads, dv)

    raw_g = self.f_proj_2(f_in).reshape(batch, seq_len, heads, dk)
    # Reference uses allow_neg_eigval (beta in [0, 2)); False clamps to [0, 1].
    beta_scale = 2.0 if self.config.kda_allow_neg_eigval else 1.0
    beta = beta_scale * jax.nn.sigmoid(b_in.astype(jnp.float32))

    if self.config.use_tokamax_kda:
      out = self._tokamax_kda(q, k, v, raw_g, beta, decoder_segment_ids)
    else:
      # Unfused path: L2-norm and query scale explicit (the kernel does them internally).
      q = _l2_normalize(q.astype(jnp.float32)) * (dk**-0.5)
      k = _l2_normalize(k.astype(jnp.float32))

      dt = self.dt_bias[...].reshape(1, 1, heads, dk)
      log_decay = -jnp.exp(self.A_log[...]).reshape(1, 1, heads, 1) * jax.nn.softplus(raw_g.astype(jnp.float32) + dt)

      if decoder_segment_ids is None:
        resets = jnp.zeros((batch, seq_len), dtype=bool)
      else:
        prev_ids = jnp.pad(decoder_segment_ids, ((0, 0), (1, 0)), constant_values=-1)[:, :seq_len]
        resets = decoder_segment_ids != prev_ids

      chunk = self.config.gdn_chunk_size
      if chunk > 0 and seq_len % chunk == 0:
        if self.config.kda_chunked_impl == "subblock" and self.config.kda_pallas_fused:
          # The whole chunk pipeline in one Pallas kernel per direction (kernels/kda).
          out = self._chunk_kda(q, k, v, log_decay, beta, resets, out_dtype=x.dtype)
        elif self.config.kda_chunked_impl == "subblock":
          # v goes in as is: the sub-block path casts to the compute dtype itself (bf16 -> f32 is exact).
          out = _delta_rule_chunked_subblock(
              q,
              k,
              v,
              log_decay,
              beta,
              resets,
              chunk,
              self.config.gdn_state_dtype,
              sub_block=self.config.kda_sub_block,
              state_scan=self._pallas_state_scan if self.config.kda_pallas_scan else None,
              intra_vjp=self.config.kda_intra_custom_vjp,
              intra_fn=self._pallas_intra if self.config.kda_pallas_intra else None,
              chunk_kernel=self._pallas_chunk if self.config.kda_pallas_chunk else None,
              inverse_precision=getattr(jax.lax.Precision, self.config.kda_inverse_precision.upper()),
          )
        else:
          out = _delta_rule_chunked(
              q, k, v.astype(jnp.float32), log_decay, beta, resets, chunk, self.config.gdn_state_dtype
          )
      else:
        out = _delta_rule_scan(q, k, v.astype(jnp.float32), jnp.exp(log_decay), beta, resets)

    gate = jax.nn.sigmoid(self.g_proj_2(g_in).reshape(batch, seq_len, heads, dv))
    out = self.o_norm(out.astype(x.dtype)) * gate
    out = self.w_out(out.reshape(batch, seq_len, heads * dv))
    out = checkpoint_name(out, "out_proj")
    return out

  def _use_flat_head_layout(self, seq_len: int) -> bool:
    """Whether the `kda_flat_head_layout` path applies (token-major fused Pallas KDA, chunk-aligned length)."""
    cfg = self.config
    chunk = cfg.gdn_chunk_size
    return bool(
        cfg.kda_flat_head_layout
        and not cfg.use_tokamax_kda
        and cfg.kda_chunked_impl == "subblock"
        and cfg.kda_pallas_fused
        and cfg.kda_fused_token_major
        and chunk > 0
        and seq_len % chunk == 0
    )

  def _flat_head_forward(self, q_in, k_in, v_in, f_in, g_in, b_in, q_conv, k_conv, v_conv, decoder_segment_ids):
    """KDA on head-flattened `[B, T, H * d]` activations end to end (`kda_flat_head_layout`).

    Same math as the token-major fused path, but q, k, v, the log-decay, the kernel
    output and the output norm/gate never take the `[B, T, H, d]` shape. On TPU
    a `[T, H, d]` array is tiled on its two minor dims (H, d) while `[T, H * d]` is
    tiled on (T, H * d), so every reshape between them (and the transposes of
    their gradients) is a full relayout copy; the token-major kernel already
    reads and writes the flat layout. Per-head reductions (q/k L2 norm and the
    per-head output RMSNorm) run in `kernels/head_norm.py`, which reads column
    blocks of the flat array; per-head parameters (A_log, the o_norm gain) are
    tiled to `[H * d]` vectors.
    """
    cfg = self.config
    batch, seq_len, _ = q_in.shape
    heads, dk = self.num_heads, self.head_k_dim
    f32 = jnp.float32
    q = causal_depthwise_conv(q_in, q_conv, decoder_segment_ids)
    k = causal_depthwise_conv(k_in, k_conv, decoder_segment_ids)
    v = causal_depthwise_conv(v_in, v_conv, decoder_segment_ids)
    raw_g = self.f_proj_2(f_in)  # [B, T, H * dk]
    beta_scale = 2.0 if cfg.kda_allow_neg_eigval else 1.0
    beta = beta_scale * jax.nn.sigmoid(b_in.astype(f32))  # [B, T, H]

    q = self._head_norm(q, mean=False, eps=1e-6, scale=dk**-0.5, out_dtype=f32)
    k = self._head_norm(k, mean=False, eps=1e-6, out_dtype=f32)
    neg_a = jnp.repeat(-jnp.exp(self.A_log[...].astype(f32)), dk)  # [H * dk]
    log_decay = neg_a * jax.nn.softplus(raw_g.astype(f32) + self.dt_bias[...].astype(f32))

    resets = kda.segment_resets(decoder_segment_ids, batch, seq_len)

    c = cfg.gdn_chunk_size
    n = seq_len // c
    compute_dtype = jnp.bfloat16 if cfg.gdn_state_dtype == "bfloat16" else f32
    seg = jnp.cumsum(resets.reshape(batch, n, c).astype(jnp.int32), axis=-1)[:, None]  # [B, 1, N, C]
    g_last = log_decay.reshape(batch, n, c, heads * dk).sum(axis=2)  # [B, N, H * dk]
    g_last = g_last.reshape(batch, n, heads, dk).transpose(0, 2, 1, 3)  # [B, H, N, dk] (small)
    decay = jnp.where((seg[..., -1] > 0)[..., None], 0.0, jnp.exp(g_last))
    beta_c = beta.reshape(batch, n, c, heads).transpose(0, 3, 1, 2)[..., None]
    q, k, v = (t.astype(compute_dtype) for t in (q, k, v))
    out = self._pallas_fused_token_major_flat(
        q, k, v, log_decay, beta_c, seg, decay, out_dtype=q_in.dtype
    )  # [B, T, H * dv]

    gate = jax.nn.sigmoid(self.g_proj_2(g_in))  # [B, T, H * dv]
    out = self._o_norm_flat(out.astype(q_in.dtype)) * gate
    out = self.w_out(out)
    out = checkpoint_name(out, "out_proj")
    return out

  def _head_norm(self, x, *, mean, eps, scale=1.0, out_dtype=None):
    """`kernels.head_norm.head_norm_flat` on `[B, T, H * d]`, split over the batch axes (Mosaic needs a shard_map)."""
    batch_axes = nn.logical_to_mesh_axes(("activation_batch",), self.config.logical_axis_rules)[0]
    spec = PartitionSpec(batch_axes)
    heads = self.num_heads

    @functools.partial(jax.shard_map, mesh=self.mesh, in_specs=(spec,), out_specs=spec, check_vma=False)
    def run(x_):
      return head_norm.head_norm_flat(x_, heads, mean=mean, eps=eps, scale=scale, out_dtype=out_dtype)

    return run(x)

  def _o_norm_flat(self, x):
    """`self.o_norm` (per-head RMSNorm over dv, shared gain) on `[B, T, H * dv]`; same arithmetic and dtypes."""
    norm = self.o_norm
    if norm.scale_offset != 0.0:
      raise NotImplementedError("kda_flat_head_layout supports the o_norm without a scale offset only.")
    y = self._head_norm(x, mean=True, eps=norm.epsilon, out_dtype=norm.dtype)
    scale = norm.scale.get_value().astype(norm.dtype)
    return y * jnp.tile(scale, self.num_heads)

  def _pallas_fused_token_major_flat(self, q, k, v, log_decay, beta_c, seg, decay, out_dtype=jnp.float32):
    """`kda.fused.kda_fused_token_major` on head-flattened `[B, T, H * d]` inputs; returns `[B, T, H * dv]`."""
    cfg = self.config
    batch_axes = nn.logical_to_mesh_axes(("activation_batch",), cfg.logical_axis_rules)[0]
    spec = PartitionSpec(batch_axes)
    interpret = jax.devices()[0].platform != "tpu"
    precision = getattr(jax.lax.Precision, cfg.kda_inverse_precision.upper())
    h = self.num_heads
    save = cfg.remat_policy == "custom" and cfg.kda_fused != "remat"
    res_name = (kda.RESIDUAL_NAME + ":wy" if cfg.kda_fused_residuals == "wy" else kda.RESIDUAL_NAME) if save else None
    stride = cfg.kda_fused_state_stride

    @functools.partial(jax.shard_map, mesh=self.mesh, in_specs=(spec,) * 7, out_specs=spec, check_vma=False)
    def run(q_, k_, v_, ld_, beta_, seg_, dl_):
      flat = lambda x: x.reshape(x.shape[0] * h, *x.shape[2:])
      seg_b = seg_[:, 0]  # [b, n, C], shared by the heads
      return kda.fused.kda_fused_token_major(
          q_,
          k_,
          v_,
          ld_,
          flat(beta_),
          seg_b[..., None],
          seg_b[:, :, None, :],
          flat(dl_)[..., None],
          h,
          precision,
          interpret,
          cfg.gdn_state_dtype == "float32",
          cfg.kda_outer_block or None,
          out_dtype,
          res_name,
          stride,
      )

    return run(q, k, v, log_decay, beta_c, seg, decay)

  def _pallas_intra(self, q_c, k_c, g):
    """Intra-chunk (scores, kk) of `[B, H, N, C, d]` chunks with the Pallas forward, split over the batch axes."""
    batch_axes = nn.logical_to_mesh_axes(("activation_batch",), self.config.logical_axis_rules)[0]
    spec = PartitionSpec(batch_axes)
    interpret = jax.devices()[0].platform != "tpu"
    sub_block = self.config.kda_sub_block
    compute_dtype = jnp.bfloat16 if self.config.gdn_state_dtype == "bfloat16" else jnp.float32

    @functools.partial(jax.shard_map, mesh=self.mesh, in_specs=(spec,) * 3, out_specs=(spec, spec), check_vma=False)
    def intra(q_, k_, g_):
      b, h, n = q_.shape[:3]
      flat = lambda t: t.reshape(b * h, n, *t.shape[3:])
      scores, kk = _kda_intra_chunk_pallas(
          flat(q_), flat(k_), flat(g_), sub_block, compute_dtype, interpret, self.config.kda_pallas_intra_bwd
      )
      return scores.reshape(b, h, n, *scores.shape[2:]), kk.reshape(b, h, n, *kk.shape[2:])

    return intra(q_c, k_c, g)

  def _chunk_kda(self, q, k, v, log_decay, beta, resets, out_dtype):
    """`kda.chunk_kda` (fused Pallas fwd/bwd) on `[B, T, H, d]` inputs, split over the batch axes only.

    With `kda_fused` = device/offload the kernel output and its VJP residuals are named "kda_fused" for the
    custom remat policy, so per-layer remat does not run the forward kernel again in the bwd; the chunk states
    saved are those of every `kda_fused_state_stride`-th chunk (the bwd rebuilds the others; 0: none saved).
    """
    cfg = self.config
    batch_axes = nn.logical_to_mesh_axes(("activation_batch",), cfg.logical_axis_rules)[0]
    spec = PartitionSpec(batch_axes)
    save = cfg.remat_policy == "custom" and cfg.kda_fused != "remat"
    run_kda = functools.partial(
        kda.chunk_kda,
        chunk_size=cfg.gdn_chunk_size,
        state_dtype=jnp.float32 if cfg.gdn_state_dtype == "float32" else jnp.bfloat16,
        layout="token_major" if cfg.kda_fused_token_major else "chunk_major",
        outer_block=cfg.kda_outer_block or None,
        inverse_precision=getattr(jax.lax.Precision, cfg.kda_inverse_precision.upper()),
        output_dtype=out_dtype,
        save_residuals=cfg.kda_fused_residuals if save else False,
        state_stride=cfg.kda_fused_state_stride,
    )

    @functools.partial(jax.shard_map, mesh=self.mesh, in_specs=(spec,) * 6, out_specs=spec, check_vma=False)
    def run(q_, k_, v_, ld_, beta_, resets_):
      return run_kda(q_, k_, v_, ld_, beta_, resets=resets_)

    return run(q, k, v, log_decay, beta, resets)

  def _pallas_chunk(self, x, vb, kb, kc, decay, qg, scores):
    """`kda_scan.kda_chunk_scan` on `[B, H, N, ...]` operands, split over the batch axes only."""
    batch_axes = nn.logical_to_mesh_axes(("activation_batch",), self.config.logical_axis_rules)[0]
    spec = PartitionSpec(batch_axes)
    interpret = jax.devices()[0].platform != "tpu"

    @functools.partial(jax.shard_map, mesh=self.mesh, in_specs=(spec,) * 7, out_specs=spec, check_vma=False)
    def run(x_, vb_, kb_, kc_, dl_, qg_, sc_):
      b, h, n = x_.shape[:3]
      flat = lambda t: t.reshape(b * h, n, *t.shape[3:])
      out = kda_scan.kda_chunk_scan(
          flat(x_), flat(vb_), flat(kb_), flat(kc_), flat(dl_)[..., None], flat(qg_), flat(sc_), interpret
      )
      return out.reshape(b, h, n, *out.shape[2:])

    return run(x, vb, kb, kc, decay, qg, scores)

  def _pallas_state_scan(self, uw, k_carry, decay_last):
    """`kda_scan.kda_state_scan` on `[B, H, N, ...]` operands, split over the batch axes only.

    Pallas kernels can't be auto-partitioned; batch rows are independent, so each shard scans its own.
    """
    batch_axes = nn.logical_to_mesh_axes(("activation_batch",), self.config.logical_axis_rules)[0]
    spec = PartitionSpec(batch_axes)
    interpret = jax.devices()[0].platform != "tpu"

    @functools.partial(jax.shard_map, mesh=self.mesh, in_specs=(spec,) * 3, out_specs=(spec, spec), check_vma=False)
    def scan(uw_, kc_, dl_):
      b, h, n = uw_.shape[:3]
      flat = lambda t: t.reshape(b * h, n, *t.shape[3:])
      states, delta = kda_scan.kda_state_scan(flat(uw_), flat(kc_), flat(dl_)[..., None], interpret)
      return states.reshape(b, h, n, *states.shape[2:]), delta.reshape(b, h, n, *delta.shape[2:])

    return scan(uw, k_carry, decay_last)

  def _fused_input_proj(self, x):
    """The six projections of x as one matmul over the concatenated kernels.

    Same parameters and the same arithmetic per output column, so checkpoints are
    unchanged; one wide matmul fills the MXU better than six narrow ones and
    turns six bwd input-gradient matmuls into one.
    """
    mods = (self.w_q, self.w_k, self.w_v, self.f_proj_1, self.g_proj_1, self.w_b)
    dtype = self.config.dtype
    kernels = [jnp.asarray(m.kernel[...], dtype) for m in mods]
    kernel = jnp.concatenate(kernels, axis=-1)
    if self.config.kda_fused_proj_reduce_scatter:
      out = linears.dot_general_with_gathered_kernel(
          jnp.asarray(x, dtype),
          kernel,
          (x.ndim - 1,),
          (0,),
          self.config.matmul_precision,
          self.mesh,
          ("embed", None),
          self.config.shard_mode,
          save_gathered_kernel=(
              self.config.gathered_weights != "remat"
              and (
                  self.config.kda_proj == "remat"
                  or (self.mesh.shape.get("data", 1) * self.mesh.shape.get("fsdp", 1)) > 1
                  or self.config.kda_fused != "remat"
              )
          ),
          out_checkpoint_name="kda_proj",
      )
    else:
      x_cast = jnp.asarray(x, dtype)
      out = jax.lax.dot_general(
          x_cast,
          kernel,
          (((x_cast.ndim - 1,), (0,)), ((), ())),
          precision=self.config.matmul_precision,
      )
      out = checkpoint_name(out, "kda_proj")
    bounds = np.cumsum([k.shape[-1] for k in kernels])[:-1]
    return jnp.split(out, bounds, axis=-1)

  def _tokamax_kda(self, q, k, v, raw_g, beta, decoder_segment_ids) -> jnp.ndarray:
    """Fused KDA via tokamax (PR #1103, experimental). Raw q/k/v/gate in: the

    kernel does the q/k L2-norm, K**-0.5 scale, and decay activation. beta in
    [0, 2) matches the reference; parity checked against _delta_rule_scan.
    """
    # pylint: disable=g-import-not-at-top,import-outside-toplevel
    # Lazy import: releases before the KDA PR lack this module.
    from tokamax._src.ops.experimental.kda import api as kda_api

    # Pallas kernels can't be auto-partitioned, so run under shard_map with only
    # the batch axis split (sequence and heads stay whole; packing preserved).
    batch_axes = nn.logical_to_mesh_axes(("activation_batch",), self.config.logical_axis_rules)[0]
    # Segments only for packed data; the varlen path pads to seq + 63*max_segments.
    has_segments = decoder_segment_ids is not None and bool(self.config.packing)
    in_specs = [PartitionSpec(batch_axes, None, None, None)] * 4 + [PartitionSpec(batch_axes, None, None)]
    if has_segments:
      in_specs.append(PartitionSpec(batch_axes, None))
    in_specs += [PartitionSpec(), PartitionSpec()]
    max_num_segments = int(self.config.tokamax_kda_max_num_segments)
    decay_floor = float(self.config.tokamax_kda_log_decay_floor)
    l2norm_outside = bool(self.config.tokamax_kda_l2norm_outside)
    resets_in_gate = has_segments and bool(self.config.tokamax_kda_resets_in_gate)
    if resets_in_gate and decay_floor <= 0:
      raise ValueError("tokamax_kda_resets_in_gate needs tokamax_kda_log_decay_floor > 0")

    @functools.partial(
        jax.shard_map,
        mesh=self.mesh,
        in_specs=tuple(in_specs),
        out_specs=PartitionSpec(batch_axes, None, None, None),
        check_vma=False,
    )
    def call_kernel(q_, k_, v_, g_, beta_, *rest):
      def head_first(t):
        return t.transpose(2, 0, 1, 3)

      seg_kwargs = {}
      if has_segments and not resets_in_gate:
        # tokamax expects 1-indexed segment ids with 0 reserved for padding,
        # which is MaxText's packed-data convention already.
        seg_kwargs = {"segment_ids": rest[0], "max_num_segments": max_num_segments}
      a_log_, dt_bias_ = rest[-2], rest[-1]
      g_hf = head_first(g_)
      if decay_floor > 0:
        # With the gate activated in-kernel the kernel skips its safe-gate centering and
        # overflows fp32 exp once a step decays past ~30 nats (NaN). Activate here and floor.
        heads, dk = g_hf.shape[0], g_hf.shape[-1]
        log_decay = -jnp.exp(a_log_).reshape(heads, 1, 1, 1) * jax.nn.softplus(
            g_hf.astype(jnp.float32) + dt_bias_.reshape(heads, 1, 1, dk)
        )
        gate_kwargs = {"use_gate_in_kernel": False}
        g_hf = jnp.maximum(log_decay, -decay_floor)
        if resets_in_gate:
          # A document's first token decays the carried state by exp(-floor) in every channel,
          # which stands in for the varlen reset. Its own decay only ever multiplies a zero
          # state, so pinning it loses no gradient the exact reset would keep.
          seg = rest[0]
          starts = seg != jnp.pad(seg, ((0, 0), (1, 0)), constant_values=-1)[:, :-1]
          g_hf = jnp.where(starts[None, :, :, None], -decay_floor, g_hf)
      else:
        gate_kwargs = {"use_gate_in_kernel": True, "a_log": a_log_, "delta_time_bias": dt_bias_}
      if l2norm_outside:
        q_, k_ = _kernel_l2norm(q_), _kernel_l2norm(k_)
      out, _ = kda_api.kimi_delta_attention(  # pylint: disable=assignment-from-no-return
          head_first(q_),
          head_first(k_),
          head_first(v_),
          g_hf,
          beta_.transpose(2, 0, 1),
          use_qk_l2norm=not l2norm_outside,
          **gate_kwargs,
          **seg_kwargs,
      )
      return out.transpose(1, 2, 0, 3)

    args = [q, k, v, raw_g, beta]
    if has_segments:
      args.append(decoder_segment_ids.astype(jnp.int32))
    args += [self.A_log[...], self.dt_bias[...]]
    return call_kernel(*args)


def _kernel_l2norm(x: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
  """The tokamax KDA kernel's q/k L2-norm (fp32 math, eps inside the rsqrt, input dtype out)."""
  x_f = x.astype(jnp.float32)
  return (x_f * jax.lax.rsqrt(jnp.sum(x_f * x_f, axis=-1, keepdims=True) + eps)).astype(x.dtype)


def _l2_normalize(x: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
  """The q/k L2-norm of the unfused and fused-Pallas paths: FLA's ``l2norm`` (rsqrt(sum + 1e-6), fp32 math).

  It used to be rsqrt(max(sum, 1e-12)), the standalone reference's form. On trained OLMo-core weights that
  epsilon alone puts the KDA layers ~8e-4 off the PyTorch/FLA model's logits; with FLA's it is ~7e-5.
  """
  return _kernel_l2norm(x, eps)


def _delta_rule_chunked(q, k, v, log_decay, beta, resets, chunk_size: int, state_dtype: str = "float32") -> jnp.ndarray:
  """Chunked delta rule. Mathematically identical to ``_delta_rule_scan``.

  The per-token scan is bound by HBM bandwidth, not compute: it reloads the whole
  ``[batch, heads, dk, dv]`` state every timestep to do a couple of FLOPs per
  element, which measures out around 0.5 FLOP/byte against a machine that needs
  ~348 to saturate the MXU. Chunking loads the state once per chunk and turns
  the intra-chunk work into ``C x C`` matmuls, so intensity rises roughly with
  ``C`` and the sequential chain shortens from ``T`` to ``T/C``.

  The algebra works because KDA's decay, though per key channel rather than per
  head, still factors. With chunk-local ``A_t = prod_{i<=t} a_i``::

      S_t[d,v] = sum_{j<=t} (A_t[d] / A_j[d]) k_j[d] delta_j[v]
      o_t[v]   = sum_{j<=t} ( (q_t * A_t) . (k_j / A_j) ) delta_j[v]

  The pairwise ratio ``A_t[d] / A_j[d]`` is <= 1 for t >= j and is computed
  exactly per pair; see the overflow note in the body for why it must not be
  folded into the operands.
  """
  batch, seq_len, heads, dk = q.shape
  dv = v.shape[-1]
  num_chunks = seq_len // chunk_size
  c = chunk_size
  # Decay/cumulative math stays float32; the operands and the carried state use
  # the compute dtype, which halves the traffic on the bandwidth-bound path.
  compute_dtype = jnp.bfloat16 if state_dtype == "bfloat16" else jnp.float32

  def to_chunks(x):
    return x.reshape(batch, num_chunks, c, heads, -1).transpose(1, 0, 3, 2, 4)

  q_c = to_chunks(q)
  k_c = to_chunks(k)
  v_c = to_chunks(v)
  log_a_c = to_chunks(log_decay)
  beta_c = beta.reshape(batch, num_chunks, c, heads).transpose(1, 0, 3, 2)
  reset_c = resets.reshape(batch, num_chunks, c).transpose(1, 0, 2)

  eye = jnp.eye(c, dtype=q.dtype)
  causal = jnp.tril(jnp.ones((c, c), dtype=bool))
  strict = jnp.tril(jnp.ones((c, c), dtype=bool), k=-1)
  positions = jnp.arange(c)

  def body(state, xs):
    q_i, k_i, v_i, log_a, beta_i, reset_i = xs

    # A document boundary inside a chunk restarts the recurrence, so the
    # cumulative decay restarts with it. Zeroing the decay at the boundary
    # instead would poison every later cumprod and silently drive the tail of
    # the chunk to zero.
    seg = jnp.cumsum(reset_i.astype(jnp.int32), axis=-1)
    seg_start = jax.lax.cummax(jnp.where(reset_i, positions, 0), axis=1)

    log_cum = jnp.cumsum(log_a, axis=-2)
    gather = seg_start[:, None, :, None]
    base = jnp.take_along_axis(log_cum, gather, axis=-2) - jnp.take_along_axis(log_a, gather, axis=-2)
    rel = log_cum - base  # <= 0, so exp(rel) is always safe
    cum = jnp.exp(rel)
    q_i = q_i.astype(compute_dtype)
    k_i = k_i.astype(compute_dtype)
    v_i = v_i.astype(compute_dtype)

    same_doc = seg[:, None, :, None] == seg[:, None, None, :]
    from_prev = (seg[:, None, :, None] == 0).astype(v_i.dtype)

    # Exact per-pair exp(rel_i - rel_j) (bounded by 1 for same-doc i >= j).
    # Folding decay into the operands would restore a matmul but overflows f32
    # (A_log up to 16 -> |rel| ~1e3); the fused kernel is the real fix. Mask
    # excluded pairs BEFORE the exp: their exponent can be +inf, and inf*0=NaN
    # would poison the VJP.
    pair_rel = rel[..., :, None, :] - rel[..., None, :, :]
    pair_mask = (causal[None, None] & same_doc)[..., None]
    pair_decay = jnp.exp(jnp.where(pair_mask, pair_rel, -jnp.inf)).astype(compute_dtype)

    # (I + diag(beta) M) delta = diag(beta) (v - carried prediction)
    m = jnp.einsum("bhid,bhjd,bhijd->bhij", k_i, k_i, pair_decay)
    m = jnp.where(strict & same_doc, m, 0.0) * beta_i[..., None]
    # (I + M)^-1 via Newton (X <- X(2I - AX)); M nilpotent so it converges in
    # log2(C) steps. Batched matmuls stay on the MXU; solve_triangular's TPU
    # lowering forced a ~2.4s/step relayout at the 3p5b shape (profiled).
    a_mat = eye + m
    m_inv = eye - m
    for _ in range(max(0, (c - 1).bit_length() - 1)):
      m_inv = m_inv @ (2.0 * eye - a_mat @ m_inv)
    carried = jnp.einsum("bhid,bhdv->bhiv", k_i * cum.astype(compute_dtype), state) * from_prev
    delta = jnp.einsum(
        "bhij,bhjv->bhiv", m_inv.astype(compute_dtype), (v_i - carried) * beta_i[..., None].astype(compute_dtype)
    )

    scores = jnp.einsum("bhid,bhjd,bhijd->bhij", q_i, k_i, pair_decay)
    scores = jnp.where(causal & same_doc, scores, 0.0).astype(compute_dtype)
    out = jnp.einsum("bhid,bhdv->bhiv", q_i * cum.astype(compute_dtype), state) * from_prev
    out = out + jnp.einsum("bhij,bhjv->bhiv", scores, delta)

    # Carry: the incoming state survives only if this chunk held no reset, and
    # only the final segment's updates propagate past the chunk boundary.
    any_reset = (seg[:, -1] > 0)[:, None, None, None]
    decayed = state * cum[..., -1, :][..., None].astype(compute_dtype)
    state = jnp.where(any_reset, jnp.zeros_like(decayed), decayed)
    # Pre-exp masking again: positions before an intra-chunk reset mix segments.
    last_seg = (seg == seg[:, -1:])[:, None, :, None]
    k_carry = k_i * jnp.exp(jnp.where(last_seg, rel[..., -1:, :] - rel, -jnp.inf)).astype(compute_dtype)
    # The carry dtype has to match the scan's init exactly or lax.scan rejects it.
    state = (state + jnp.einsum("bhid,bhiv->bhdv", k_carry, delta)).astype(compute_dtype)
    return state, out.astype(jnp.float32)

  init_state = jnp.zeros((batch, heads, dk, dv), compute_dtype)
  # Remat the body: otherwise the scan stores O(C^2 d) pairwise residuals
  # (~544GB HBM at the 3p5b shape). Recompute them in the bwd instead.
  body_remat = jax.checkpoint(body, policy=jax.checkpoint_policies.nothing_saveable)
  _, outputs = jax.lax.scan(body_remat, init_state, (q_c, k_c, v_c, log_a_c, beta_c, reset_c))
  return outputs.transpose(1, 0, 3, 2, 4).reshape(batch, seq_len, heads, dv)


@functools.partial(jax.custom_vjp, nondiff_argnums=(1,))
def _invert_unit_lower(m: jnp.ndarray, precision=jax.lax.Precision.HIGHEST) -> jnp.ndarray:
  """(I + M)^-1 for strictly lower triangular M over the last two axes.

  Exact block doubling from 1 x 1 blocks, as the fused Pallas kernel (`kernels/kda_fused._inverse`) and FLA's
  ``solve_tril`` do: with X_s the exact inverse of the s x s block-diagonal part, X_2s = X_s - X_s L_s X_s, where
  L_s keeps the entries of M in the lower-left s x s block of each 2s x 2s diagonal block. Every iterate is the
  exact inverse of a block-diagonal part of (I + M), so no intermediate grows past the answer. (The power series
  (I - N)(I + N^2)... used before is algebraically the same but its partial sums are Neumann sums of (-M)^i; on
  real keys, which are far more alike than random ones, with beta up to 2, they reach huge values in fp32 before
  cancelling.) All batched matmuls; the analytic VJP below replaces differentiating through the iterations.
  """
  return _invert_unit_lower_fwd(m, precision)[0]


def _invert_unit_lower_fwd(m, precision=jax.lax.Precision.HIGHEST):
  """Forward of `_invert_unit_lower`; saves the inverse for the backward."""
  c = m.shape[-1]
  f32 = jnp.float32
  # Tiny matrices: full precision costs little, and a 1-pass bf16 inverse drifts.
  mm = functools.partial(jnp.matmul, precision=precision)
  m = m.astype(f32)
  row = jnp.arange(c)[:, None]
  col = jnp.arange(c)[None, :]
  eye = (row == col).astype(f32)
  # X_2 = I - L_1: the 2 x 2 diagonal blocks need no matmul.
  first = ((row // 2) == (col // 2)) & (row % 2 == 1) & (col % 2 == 0)
  x = eye - jnp.where(first, m, 0.0)
  size = 2
  while size < c:
    lower_left = ((row // (2 * size)) == (col // (2 * size))) & (((row // size) % 2) == 1) & (((col // size) % 2) == 0)
    x = x - mm(mm(x, jnp.where(lower_left, m, 0.0)), x)
    size *= 2
  return x, x


def _invert_unit_lower_bwd(precision, x, dx):
  # d(T^-1) = -X dT X, so dT = -X^T dX X^T; T's free entries are strictly lower.
  xt = jnp.swapaxes(x, -1, -2)
  mm = functools.partial(jnp.matmul, precision=precision)
  dm = -mm(mm(xt, dx.astype(jnp.float32)), xt)
  return (jnp.tril(dm, k=-1),)


_invert_unit_lower.defvjp(_invert_unit_lower_fwd, _invert_unit_lower_bwd)


def _kda_wy_solve(m, rhs, compute_dtype):
  """Solves `(I + M) uw = rhs` using block unit-lower-triangular inversion."""
  x = _invert_unit_lower(m).astype(compute_dtype)
  return jnp.einsum("...ij,...jd->...id", x, rhs.astype(compute_dtype), preferred_element_type=jnp.float32)


def _kda_intra_chunk(q, k, g, sub_block: int, compute_dtype):
  """Per-chunk ``sum_d a_id b_jd exp(g_id - g_jd)`` for (a, b) = (q, k) and (k, k).

  Shapes ``[..., C, d]`` in, ``[..., C, C]`` out (unmasked, f32). ``g`` is the
  cumulative log-decay from the chunk start, so it is non-increasing in time.

  Rows are split into sub-blocks of ``sub_block``. Against an earlier column,
  a row's decay factors through its sub-block's first row p: exp(g_i - g_j) =
  exp(g_i - g_p) exp(g_p - g_j), and both exponents are <= 0 because g only
  falls. That turns every below-diagonal sub-block into a matmul that cannot
  overflow. Only the diagonal sub-blocks need the exact per-pair tensor, which
  is 1/(C/sub_block) of the full one.
  """
  c, d = k.shape[-2], k.shape[-1]
  s = min(sub_block, c)
  nb = c // s
  lead = k.shape[:-2]
  f32 = jnp.float32
  pos = jnp.arange(c)
  piv = jnp.arange(nb) * s
  g_piv = g[..., piv, :]  # [..., nb, d]
  # Right operand for row block r: column j scaled by exp(g_p - g_j), kept only for j < p.
  right_exp = jnp.where((pos[None, :] < piv[:, None])[..., None], g_piv[..., :, None, :] - g[..., None, :, :], -jnp.inf)
  right = (k[..., None, :, :].astype(f32) * jnp.exp(right_exp)).astype(compute_dtype)  # [..., nb, C, d]
  g_rows = g.reshape(*lead, nb, s, d)
  left_scale = jnp.exp(g_rows - g_piv[..., :, None, :])
  # q and k share the right operand and the per-pair decays, so they are stacked on a
  # leading axis: one matmul with twice the rows, and one pass over the pair tensor.
  k_rows = k.reshape(*lead, nb, s, d).astype(f32)
  qk_rows = jnp.stack([q.reshape(*lead, nb, s, d).astype(f32), k_rows], axis=-4)  # [..., 2, nb, s, d]
  lqk = (qk_rows * left_scale[..., None, :, :, :]).astype(compute_dtype)
  off = jnp.einsum("...xrsd,...rjd->...xrsj", lqk, right, preferred_element_type=f32).reshape(*lead, 2, c, c)
  # Diagonal sub-blocks: exact per pair, masked before the exp (upper pairs can be +inf).
  tri = jnp.tril(jnp.ones((s, s), dtype=bool))
  pair = jnp.exp(jnp.where(tri[..., None], g_rows[..., :, None, :] - g_rows[..., None, :, :], -jnp.inf))
  # The default-precision einsum rounds its operands to bf16 anyway, so feeding it bf16 (half the bytes of
  # the per-pair tensor) is bit-identical and faster.
  diag = jnp.einsum(
      "...xrid,...rjd,...rijd->...xrij",
      qk_rows.astype(compute_dtype),
      k_rows.astype(compute_dtype),
      pair.astype(compute_dtype),
      preferred_element_type=f32,
  )
  # Place each sub-block on the block diagonal with a mask (elementwise) rather than an identity einsum.
  eye_nb = jnp.eye(nb, dtype=f32)
  both = off + (diag[..., :, :, None, :] * eye_nb[:, None, :, None]).reshape(*lead, 2, c, c)
  return both[..., 0, :, :], both[..., 1, :, :]


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def _kda_intra_chunk_vjp(q, k, g, sub_block: int, compute_dtype):
  """``_kda_intra_chunk`` with a hand-derived backward (same forward, same outputs).

  Every pairwise factor is exp(g_i - g_j), so the gradients share three contractions per chunk:
  DQ_i = sum_j dS_ij k_j E_ij, DKrow_i = sum_j dK_ij k_j E_ij, DKcol_j = sum_i (dS_ij q_i + dK_ij k_i) E_ij,
  and then dq = DQ, dk = DKrow + DKcol, dg = q*DQ + k*(DKrow - DKcol) (the factor depends on g only through
  g_i - g_j). Below-diagonal sub-blocks use the pivot factorization of the forward, so they are matmuls; the
  diagonal sub-blocks reuse one per-pair tensor for all three. Autodiff instead differentiates the exps and
  masks one by one and costs 2.8x the forward on v4; this costs about 1.5x.
  """
  return _kda_intra_chunk(q, k, g, sub_block, compute_dtype)


def _kda_intra_chunk_vjp_fwd(q, k, g, sub_block, compute_dtype):
  return _kda_intra_chunk(q, k, g, sub_block, compute_dtype), (q, k, g)


def _kda_intra_chunk_vjp_bwd(sub_block, compute_dtype, res, cts):
  """Backward of `_kda_intra_chunk_vjp`: recomputes the intra-chunk terms from (q, k, g)."""
  q, k, g = res
  d_s, d_k = cts
  c, d = k.shape[-2], k.shape[-1]
  s = min(sub_block, c)
  nb = c // s
  lead = k.shape[:-2]
  f32 = jnp.float32
  pos = jnp.arange(c)
  piv = jnp.arange(nb) * s
  g_piv = g[..., piv, :]  # [..., nb, d]
  g_rows = g.reshape(*lead, nb, s, d)
  left = jnp.exp(g_rows - g_piv[..., :, None, :])  # [..., nb, s, d], <= 1
  right = jnp.exp(
      jnp.where((pos[None, :] < piv[:, None])[..., None], g_piv[..., :, None, :] - g[..., None, :, :], -jnp.inf)
  )
  q_rows = q.reshape(*lead, nb, s, d).astype(f32)
  k_rows = k.reshape(*lead, nb, s, d).astype(f32)
  # Rows of dS and dK per sub-block, stacked like the forward's q/k: [..., 2, nb, s, C].
  dsk = jnp.stack([d_s.reshape(*lead, nb, s, c), d_k.reshape(*lead, nb, s, c)], axis=-4).astype(f32)

  # Below-diagonal sub-blocks: E_ij = left_i * right_r[j] (right is zero for j >= the block's pivot).
  kr = (k[..., None, :, :].astype(f32) * right).astype(compute_dtype)  # [..., nb, C, d]
  rows_off = jnp.einsum("...xrsj,...rjd->...xrsd", dsk.astype(compute_dtype), kr, preferred_element_type=f32)
  rows_off = rows_off * left[..., None, :, :, :]  # [..., 2, nb, s, d]: DQ, DKrow
  qkl = (jnp.stack([q_rows, k_rows], axis=-4) * left[..., None, :, :, :]).astype(compute_dtype)
  col = jnp.einsum("...xrsj,...xrsd->...rjd", dsk.astype(compute_dtype), qkl, preferred_element_type=f32)
  col_off = jnp.sum(col * right, axis=-3)  # [..., C, d]

  # Diagonal sub-blocks: exact per pair, lower triangle including the diagonal.
  tri = jnp.tril(jnp.ones((s, s), dtype=bool))
  pair = jnp.exp(jnp.where(tri[..., None], g_rows[..., :, None, :] - g_rows[..., None, :, :], -jnp.inf))
  dsk_blocks = dsk.reshape(*lead, 2, nb, s, nb, s)
  dsk_diag = jnp.moveaxis(jnp.diagonal(dsk_blocks, axis1=-4, axis2=-2), -1, -3)  # [..., 2, nb, s, s]
  # bf16 operands: what the default-precision einsum rounds to anyway, at half the bytes.
  cd = compute_dtype
  pair_c, dsk_c = pair.astype(cd), dsk_diag.astype(cd)
  rows_diag = jnp.einsum("...xrij,...rjd,...rijd->...xrid", dsk_c, k_rows.astype(cd), pair_c, preferred_element_type=f32)
  col_diag = jnp.einsum(
      "...xrij,...xrid,...rijd->...rjd",
      dsk_c,
      jnp.stack([q_rows, k_rows], axis=-4).astype(cd),
      pair_c,
      preferred_element_type=f32,
  )

  dq_full = (rows_off[..., 0, :, :, :] + rows_diag[..., 0, :, :, :]).reshape(*lead, c, d)
  dk_row = (rows_off[..., 1, :, :, :] + rows_diag[..., 1, :, :, :]).reshape(*lead, c, d)
  dk_col = col_off + col_diag.reshape(*lead, c, d)
  dg = q.astype(f32) * dq_full + k.astype(f32) * (dk_row - dk_col)
  # Pivot terms. The matrices depend on g only through differences, so the intra part of dg sums to zero over
  # a chunk. Exactly, the row terms of a below-diagonal block (from bf16(k e^(g_p - g_j))) and its column
  # terms (from bf16(q e^(g_i - g_p))) cancel; rounded differently, they leave a bias that the log-decay
  # gradient (a reverse cumsum of dg) accumulates: 3.6% error at a chunk's first row on v4, against 0.46% for
  # autodiff. Autodiff carries the difference on the pivot row g_p, through e^(+-g_p); so does this.
  # The diagonal blocks' row and column einsums round their intermediate products differently too.
  rows_all = rows_off + rows_diag
  row_tot = jnp.sum(q_rows * rows_all[..., 0, :, :, :] + k_rows * rows_all[..., 1, :, :, :], axis=-2)  # [.., nb, d]
  col_tot = jnp.sum(k[..., None, :, :].astype(f32) * col * right, axis=-2) + jnp.sum(k_rows * col_diag, axis=-2)
  pivot = jnp.concatenate([(col_tot - row_tot)[..., :, None, :], jnp.zeros((*lead, nb, s - 1, d), f32)], axis=-2)
  dg = dg + pivot.reshape(*lead, c, d)
  return dq_full.astype(q.dtype), (dk_row + dk_col).astype(k.dtype), dg.astype(g.dtype)


_kda_intra_chunk_vjp.defvjp(_kda_intra_chunk_vjp_fwd, _kda_intra_chunk_vjp_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6))
def _kda_intra_chunk_pallas(q, k, g, sub_block: int, compute_dtype, interpret: bool, pallas_bwd: bool = False):
  """``_kda_intra_chunk`` on `[BH, N, C, d]` with the Pallas forward of ``kernels/kda_intra.py``.

  The bwd is the hand-derived VJP in XLA, or with `pallas_bwd` the Pallas bwd kernel. The kernel alone is 2x
  faster, but in the full model XLA fuses the hand VJP with its neighbours and wins (v4-128: 1.317 vs
  1.341 s at seq 4096, 2.754 vs 2.784 s at 8192).
  """
  del sub_block, pallas_bwd
  return kda_intra.kda_intra_fwd(q, k, g, compute_dtype=compute_dtype, interpret=interpret)


def _kda_intra_chunk_pallas_fwd(q, k, g, sub_block, compute_dtype, interpret, pallas_bwd):
  del sub_block, pallas_bwd
  return kda_intra.kda_intra_fwd(q, k, g, compute_dtype=compute_dtype, interpret=interpret), (q, k, g)


def _kda_intra_chunk_pallas_bwd(sub_block, compute_dtype, interpret, pallas_bwd, res, cts):
  if not pallas_bwd:
    return _kda_intra_chunk_vjp_bwd(sub_block, compute_dtype, res, cts)
  q, k, g = res
  d_scores, d_kk = cts
  return kda_intra.kda_intra_bwd(q, k, g, d_scores, d_kk, compute_dtype=compute_dtype, interpret=interpret)


_kda_intra_chunk_pallas.defvjp(_kda_intra_chunk_pallas_fwd, _kda_intra_chunk_pallas_bwd)


def _delta_rule_chunked_subblock(
    q,
    k,
    v,
    log_decay,
    beta,
    resets,
    chunk_size: int,
    state_dtype: str = "float32",
    sub_block: int = 16,
    state_scan=None,
    intra_vjp: bool = False,
    intra_fn=None,
    chunk_kernel=None,
    inverse_precision=jax.lax.Precision.HIGHEST,
) -> jnp.ndarray:
  """``_delta_rule_chunked`` restructured for the MXU. Same math, same contract.

  Three changes. The intra-chunk matrices use ``_kda_intra_chunk``'s sub-block
  factorization instead of the full per-pair decay tensor. ``(I + M)^-1`` uses
  block doubling with an analytic VJP. And everything that does not depend on
  the carried state (the matrices, the inverse, the WY products u and w, the
  output readout) runs for all chunks at once outside the scan, which keeps
  only two matmuls per chunk on the sequential path. The scan returns each
  chunk's incoming state and delta; the readout uses them afterwards.

  ``state_scan(uw, k_carry, decay)`` replaces that scan when given (bf16 state only), e.g. the Pallas
  kernel in ``kernels/kda_scan.py``; ``uw`` is u and w side by side, and ``decay`` is already zeroed for
  chunks that hold a reset.

  The fused Pallas version of this pipeline (one kernel per direction) is ``kernels/kda.chunk_kda``.
  """
  batch, seq_len, heads, dk = q.shape
  dv = v.shape[-1]
  c = chunk_size
  n = seq_len // c
  f32 = jnp.float32
  compute_dtype = jnp.bfloat16 if state_dtype == "bfloat16" else f32

  def to_chunks(x):  # [B, T, H, D] -> [B, H, N, C, D]
    return x.reshape(batch, n, c, heads, -1).transpose(0, 3, 1, 2, 4)

  # Cast before chunking so the saved activations are in the compute dtype.
  q_c, k_c, v_c = (to_chunks(t.astype(compute_dtype)) for t in (q, k, v))
  # Cumulative decay as a lower-triangular matmul: jnp.cumsum lowers to an O(C^2) reduce_window on TPU.
  tri_ones = jnp.tril(jnp.ones((c, c), f32))
  g = jnp.einsum("ij,...jd->...id", tri_ones, to_chunks(log_decay).astype(f32), precision=jax.lax.Precision.HIGHEST)
  beta_c = beta.reshape(batch, n, c, heads).transpose(0, 3, 1, 2).astype(f32)[..., None]
  # Named "context" so a custom remat policy (context=device) can keep the chunked inputs and the
  # intra-chunk results below; the bwd then skips the projections, conv, norm, decay activation,
  # intra-chunk build and inverse that per-layer remat would otherwise recompute.
  q_c, k_c, v_c, g, beta_c = (checkpoint_name(t, "context") for t in (q_c, k_c, v_c, g, beta_c))

  # Segment layout per chunk, broadcast over heads.
  seg = jnp.cumsum(resets.reshape(batch, n, c).astype(jnp.int32), axis=-1)[:, None]  # [B, 1, N, C]
  same_doc = seg[..., :, None] == seg[..., None, :]
  causal = jnp.tril(jnp.ones((c, c), dtype=bool))
  strict = jnp.tril(jnp.ones((c, c), dtype=bool), k=-1)
  from_prev = (seg == 0)[..., None].astype(f32)
  last_seg = (seg == seg[..., -1:])[..., None]
  any_reset = seg[..., -1] > 0  # [B, 1, N]

  # The [.., nb, C, d] and per-pair intermediates are large; recompute them in the bwd.
  if intra_fn is not None:
    # e.g. the Pallas forward with the hand-derived bwd (kda_pallas_intra)
    scores, kk = intra_fn(q_c, k_c, g)
  elif intra_vjp:
    # Hand-derived bwd; its residuals are only q, k and g, so it recomputes like nothing_saveable.
    scores, kk = _kda_intra_chunk_vjp(q_c, k_c, g, sub_block, compute_dtype)
  else:
    intra = jax.checkpoint(
        functools.partial(_kda_intra_chunk, sub_block=sub_block, compute_dtype=compute_dtype),
        policy=jax.checkpoint_policies.nothing_saveable,
    )
    scores, kk = intra(q_c, k_c, g)  # pylint: disable=too-many-function-args
  scores = jnp.where(causal & same_doc, scores, 0.0)
  m = jnp.where(strict & same_doc, kk, 0.0) * beta_c
  x = _invert_unit_lower(m, inverse_precision)

  # WY: delta = X diag(beta) (v - from_prev * (k e^g) S) = u - w S.
  eg = jnp.exp(g)
  if chunk_kernel is not None and compute_dtype == jnp.bfloat16:
    # One kernel for u = X vb, w = X kb, the recurrence and the readout (kda_pallas_chunk).
    cd = compute_dtype
    g_last = g[..., -1:, :]
    k_carry = k_c.astype(f32) * jnp.exp(jnp.where(last_seg, g_last - g, -jnp.inf))
    decay = jnp.where(any_reset[..., None], 0.0, jnp.exp(g_last[..., 0, :]))
    vb = (v_c.astype(f32) * beta_c).astype(cd)
    kb = (k_c.astype(f32) * eg * beta_c * from_prev).astype(cd)
    qg = (q_c.astype(f32) * eg * from_prev).astype(cd)
    x, scores = (checkpoint_name(t, "kda_wy") for t in (x, scores))
    out = chunk_kernel(x.astype(cd), vb, kb, k_carry.astype(cd), decay, qg, scores.astype(cd))
    return out.transpose(0, 2, 3, 1, 4).reshape(batch, seq_len, heads, dv)
  rhs = jnp.concatenate([v_c.astype(f32) * beta_c, k_c.astype(f32) * eg * beta_c * from_prev], axis=-1)
  uw = jnp.einsum("...ij,...jd->...id", x.astype(compute_dtype), rhs.astype(compute_dtype), preferred_element_type=f32)
  # Saved as kda_wy=device (34 MB per layer at seq 4096), the remat forward skips the intra-chunk build and the
  # inverse; the intra-chunk VJP still recomputes its own forward.
  x, scores = (checkpoint_name(t, "kda_wy") for t in (x, scores))
  uw = checkpoint_name(uw, "context")
  u, w = uw[..., :dv], uw[..., dv:]
  # Only the chunk's last segment writes the carried state.
  g_last = g[..., -1:, :]
  k_carry = k_c.astype(f32) * jnp.exp(jnp.where(last_seg, g_last - g, -jnp.inf))
  decay_last = jnp.exp(g_last[..., 0, :])  # [B, H, N, dk]

  def scan_layout(t):  # [B, H, N, ...] -> [N, B, H, ...]
    return jnp.moveaxis(t, 2, 0)

  def body(state, xs):
    u_i, w_i, kc_i, dl_i, reset_i = xs
    delta = u_i - jnp.einsum("bhcd,bhdv->bhcv", w_i.astype(compute_dtype), state, preferred_element_type=f32)
    delta = delta.astype(compute_dtype)
    kept = jnp.where(reset_i[:, :, None, None], 0.0, state.astype(f32) * dl_i[..., None])
    new_state = kept + jnp.einsum("bhcd,bhcv->bhdv", kc_i.astype(compute_dtype), delta, preferred_element_type=f32)
    return new_state.astype(compute_dtype), (state, delta)

  if state_scan is not None and compute_dtype == jnp.bfloat16:
    decay = jnp.where(any_reset[..., None], 0.0, decay_last)
    states, delta = state_scan(uw, k_carry.astype(compute_dtype), decay)
  else:
    xs = tuple(scan_layout(t) for t in (u, w, k_carry, decay_last, any_reset))
    init_state = jnp.zeros((batch, heads, dk, dv), compute_dtype)
    _, (states, delta) = jax.lax.scan(body, init_state, xs)
    states, delta = jnp.moveaxis(states, 0, 2), jnp.moveaxis(delta, 0, 2)  # [B, H, N, ...]

  qg = (q_c.astype(f32) * eg * from_prev).astype(compute_dtype)
  out = jnp.einsum("...cd,...dv->...cv", qg, states, preferred_element_type=f32)
  out = out + jnp.einsum("...ij,...jv->...iv", scores.astype(compute_dtype), delta, preferred_element_type=f32)
  return out.transpose(0, 2, 3, 1, 4).reshape(batch, seq_len, heads, dv)


def _delta_rule_scan(q, k, v, decay, beta, resets) -> jnp.ndarray:
  """Unfused delta-rule recurrence, scanned over time.

  State is ``[batch, heads, key_head_dim, value_head_dim]`` and is zeroed at
  document boundaries so a packed batch behaves like independent sequences.
  """

  def step(state, xs):
    q_t, k_t, v_t, decay_t, beta_t, reset_t = xs
    state = jnp.where(reset_t[:, None, None, None], 0.0, state)
    state = state * decay_t[..., None]
    prediction = jnp.einsum("bhkv,bhk->bhv", state, k_t)
    delta = (v_t - prediction) * beta_t[..., None]
    state = state + jnp.einsum("bhk,bhv->bhkv", k_t, delta)
    return state, jnp.einsum("bhkv,bhk->bhv", state, q_t)

  batch, _, heads, dk = q.shape
  dv = v.shape[-1]
  init_state = jnp.zeros((batch, heads, dk, dv), jnp.float32)
  # Scan over time, so move the length axis to the front.
  xs = (
      jnp.swapaxes(q, 0, 1),
      jnp.swapaxes(k, 0, 1),
      jnp.swapaxes(v, 0, 1),
      jnp.swapaxes(decay, 0, 1),
      jnp.swapaxes(beta, 0, 1),
      jnp.swapaxes(resets, 0, 1),
  )
  _, outputs = jax.lax.scan(step, init_state, xs)
  return jnp.swapaxes(outputs, 0, 1)


class OLMoE3Attention(nnx.Module):
  """Full attention for OLMoE3: GQA, NoPE, per-head QK-norm, sigmoid output gate.

  The gate is produced by widening the query projection to ``2 * head_dim`` and
  splitting, which is how MaxText's shared attention implements gated hybrids.
  Parameter count matches the reference's separate ``w_g`` projection; only the
  storage layout differs, which the checkpoint converter must account for.
  """

  def __init__(self, config: Config, mesh: Mesh, model_mode: str, quant: None | Quant = None, *, rngs: nnx.Rngs):
    self.config = config
    cfg = config
    batch_size, seq_len = max_utils.get_batch_seq_len_for_mode(config, model_mode)
    dummy_inputs_shape = (batch_size, seq_len, cfg.emb_dim)

    self.attention = attentions.Attention(
        config=cfg,
        num_query_heads=cfg.num_query_heads,
        num_kv_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim,
        max_target_length=cfg.max_target_length,
        max_prefill_predict_length=cfg.max_prefill_predict_length,
        attention_kernel=cfg.attention,
        inputs_q_shape=dummy_inputs_shape,
        inputs_kv_shape=dummy_inputs_shape,
        out_axis_names=(BATCH, LENGTH, EMBED),
        mesh=mesh,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        dropout_rate=cfg.dropout_rate,
        # Query, key and value kernels are [emb, heads, dims]; the output kernel is [heads * head_dim, emb].
        kernel_init=_matrix_init_by_shape(
            cfg, lambda shape: "mixer/attention/query/kernel" if len(shape) == 3 else "mixer/attention/out/kernel"
        )
        if _completedp_base(cfg) is not None
        else _olmoe3_attention_init(cfg),
        name="self_attention",
        quant=quant,
        kv_quant=quantizations.configure_kv_quant(cfg),
        use_qk_norm=cfg.use_qk_norm,
        query_pre_attn_scalar=_attention_logit_scale(cfg),
        model_mode=model_mode,
        is_nope_layer=True,  # OLMoE3 uses no positional embedding at all.
        rngs=rngs,
    )

  def __call__(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      kv_cache=None,
      attention_metadata=None,
  ):
    return self.attention(
        inputs_q=inputs,
        inputs_kv=inputs,
        inputs_positions=decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
        model_mode=model_mode,
        kv_cache=kv_cache,
        attention_metadata=attention_metadata,
    )


class OLMoE3LatentRoutedMoE(moe.RoutedMoE):
  """RoutedMoE whose router scores the full-width residual.

  The experts run at ``moe_expert_input_dim`` (the latent), but OLMoE3 routes on
  the uncompressed ``emb_dim`` residual, so the gate has to be sized separately
  from the expert input. ``RoutedMoE`` already accepts ``gate_inputs``; only the
  gate's input width needs overriding.
  """

  def get_topk(
      self,
      gate_logits,
      pre_bias_logits,
      rngs=None,
      input_ids=None,
      forced_routed_experts=None,
      per_expert_scale=None,
  ):
    """Top-k routing weights, rescaled to sum to ``top_k`` as in the reference.

    ``input_ids`` carries the packed segment ids (EMo routes per document); it
    is consumed here and never forwarded, so the base class's hash-routing
    interpretation of the argument can't engage. ``forced_routed_experts`` is
    passed straight through to the base class.

    The base class's softmax over the top-k logits already equals the
    reference's L1-normalized gather of full-softmax scores
    (``normalize_expert_weights=1.0``); ``restore_weight_scale=True`` then
    multiplies by ``top_k``, which is the only piece missing here. The masked
    logits leave that unchanged: every selected expert is inside the pool, so
    the gathered scores are the unmasked ones.
    """
    # Forced experts arrive already routed (moe_a2a_token_chunks reuses the full-sequence EMo routing per
    # chunk). They lie inside their pools, so masking again changes nothing but would rebuild the pools from a
    # partial sequence.
    if self.config.emo_enabled and forced_routed_experts is None:
      gate_logits = self._emo_mask_logits(gate_logits, input_ids, rngs)
    top_k_weights, top_k_indices = super().get_topk(
        gate_logits,
        pre_bias_logits,
        rngs,
        input_ids=None,
        forced_routed_experts=forced_routed_experts,
        per_expert_scale=per_expert_scale,
    )
    return top_k_weights * self.num_experts_per_tok, top_k_indices

  def _emo_mask_logits(self, gate_logits, segment_ids, rngs):
    """Masks router logits to each document's expert pool (the EMo router).

    Reference: ``EMoRouter`` in the standalone model. A document aggregates its
    tokens' full-softmax scores, keeps its top ``pool`` experts by that total,
    and every token of the document then routes within the pool. ``pool`` is
    sampled per document during training, uniformly from the integer range
    [emo_min, emo_max] or, with ``emo_document_expert_pool_choices`` set,
    from that set of sizes (uniformly, or with ``emo_document_expert_pool_probs``);
    outside training it is fixed at ``emo_eval_document_expert_pool``.

    MaxText packs documents contiguously, so the per-document score totals are
    cumulative-sum differences at document boundaries rather than a scatter
    over a static document count — this also makes the masking independent of
    the segment-id numbering (only boundaries matter). The auxiliary losses
    intentionally see the unmasked logits, as in the reference.
    """
    cfg = self.config
    batch, seq_len, _ = gate_logits.shape
    if segment_ids is None:
      segment_ids = jnp.zeros((batch, seq_len), dtype=jnp.int32)
    positions = jnp.arange(seq_len)
    prev_ids = jnp.pad(segment_ids, ((0, 0), (1, 0)), constant_values=-1)[:, :seq_len]
    is_first = segment_ids != prev_ids  # position 0 always starts a document
    next_ids = jnp.pad(segment_ids, ((0, 0), (0, 1)), constant_values=-1)[:, 1:]
    is_last = segment_ids != next_ids  # the final position always ends one

    start = jax.lax.cummax(jnp.where(is_first, positions, 0), axis=1)
    end = jnp.flip(jax.lax.cummin(jnp.flip(jnp.where(is_last, positions, seq_len), axis=1), axis=1), axis=1)

    scores = jax.nn.softmax(gate_logits.astype(jnp.float32), axis=-1)
    csum = jnp.cumsum(scores, axis=1)
    csum_end = jnp.take_along_axis(csum, end[..., None], axis=1)
    before = jnp.take_along_axis(csum, jnp.maximum(start - 1, 0)[..., None], axis=1)
    csum_before = jnp.where((start == 0)[..., None], 0.0, before)
    document_scores = csum_end - csum_before  # [b, s, E]: each token sees its document's totals

    if rngs is not None and cfg.model_call_mode != "inference" and not _EMO_EVAL_POOLS.get():
      rng = rngs.params() if hasattr(rngs, "params") and callable(getattr(rngs, "params")) else rngs
      if cfg.emo_document_expert_pool_choices:
        choices = jnp.asarray(cfg.emo_document_expert_pool_choices, dtype=jnp.int32)
        if cfg.emo_document_expert_pool_probs:
          probs = jnp.asarray(cfg.emo_document_expert_pool_probs, dtype=jnp.float32)
          draws = jax.random.choice(rng, choices, (batch, seq_len), p=probs)
        else:
          draws = choices[jax.random.randint(rng, (batch, seq_len), 0, choices.shape[0])]
      else:
        draws = jax.random.randint(
            rng, (batch, seq_len), cfg.emo_min_document_expert_pool, cfg.emo_max_document_expert_pool + 1
        )
      # One draw per document: every token reads the draw at its document start.
      pool = jnp.take_along_axis(draws, start, axis=1)
    else:
      pool = jnp.full((batch, seq_len), cfg.emo_eval_document_expert_pool, dtype=jnp.int32)

    if cfg.moe_lean_routing:
      keep = self._emo_keep_by_threshold(
          jax.lax.stop_gradient(document_scores), pool, bisect=cfg.emo_threshold_by_bisection
      )
      keep = checkpoint_name(keep, "moe_routing")
    else:
      order = jnp.argsort(-document_scores, axis=-1)
      rank = jnp.argsort(order, axis=-1)
      keep = rank < pool[..., None]
    return jnp.where(keep, gate_logits, -jnp.inf)

  @staticmethod
  def _emo_keep_by_threshold(scores, pool, bisect=False):
    """``argsort(argsort(-scores)) < pool`` with one value sort instead of two argsorts.

    The pool-th largest score is the threshold. Everything above it is kept, and
    ties at the threshold fill the remaining slots lowest index first, which is
    the order the stable argsort gives. ``bisect`` finds the threshold with 32
    counting passes over the scores' order-preserving uint32 keys instead of a sort.
    """
    if bisect:
      return moe.keep_top_by_bisection(scores, pool[..., None])
    num_experts = scores.shape[-1]
    ascending = jnp.sort(scores, axis=-1)
    thr = jnp.take_along_axis(ascending, (num_experts - pool)[..., None], axis=-1)
    above = scores > thr
    at = scores == thr
    slots = pool[..., None] - jnp.sum(above, axis=-1, keepdims=True)
    return above | (at & (jnp.cumsum(at, axis=-1) <= slots))

  def load_balance_loss(self, top_k_indices, logits) -> jax.Array:  # pylint: disable=arguments-differ
    """OLMo-core's batch-level load-balancing loss.

    ``lb = (E / K) * sum_e mean_{b,s}(probs_e) * counts_e / (B * S)``, scaled by
    ``load_balance_loss_weight``, with counts summed over the batch rather than
    per sequence.

    On the dense path the arrays here span the global batch. On the sparse path
    this runs inside the routed MoE's ``shard_map``, where they are one device's
    tokens, so without ``moe_global_load_balancing`` each device balances its
    own tokens and the value returned through ``out_specs=P()`` is device 0's.
    With it, the counts are averaged over the axes that shard the tokens first
    (OLMo-core's ``global_load_balancing``: all-reduced counts, local
    probabilities, no gradient through the counts), and the loss is then
    averaged too, so every device holds the global-batch value. The loss is
    linear in the probabilities, so the mean of the per-device terms is exactly
    the global-batch loss; the gradient through ``P()`` is already the mean of
    the per-device gradients, and the final ``pmean`` leaves it unchanged. The
    sparse path sets those axes in ``moe.LB_REDUCE_AXES`` around its
    ``shard_map`` (the same axes it uses for the overflow flag); outside it the
    variable is unset and nothing is reduced.
    """
    flat_indices = jax.lax.stop_gradient(top_k_indices).reshape(-1)
    counts = jnp.bincount(flat_indices, length=self.num_experts).astype(jnp.float32)
    reduce_axes = moe.LB_REDUCE_AXES.get() if getattr(self.config, "moe_global_load_balancing", False) else None
    if reduce_axes:
      counts = jax.lax.pmean(jax.lax.stop_gradient(counts), reduce_axes)
    tokens = top_k_indices.shape[0] * top_k_indices.shape[1]
    mean_probs = logits.astype(jnp.float32).mean(axis=(0, 1))
    lb = (mean_probs * counts).sum() * self.num_experts / (self.num_experts_per_tok * tokens)
    if reduce_axes:
      lb = jax.lax.pmean(lb, reduce_axes)
    lb = lb * self.config.load_balance_loss_weight
    if not getattr(self.config, "olmoe3_router_metrics", False):
      return lb
    # olmoe3_router_metrics: [lb, load imbalance, global load imbalance] instead of the scalar; the decoder layer
    # splits it, so the loss term is unchanged. The vector rides the scalar's path out of the routed MoE's
    # shard_map (out_specs P(): every device computes the same values from the gathered counts). The per-sequence
    # counts are built here only (the loss above uses bincount). With olmoe3_router_metrics_interval the train step
    # sets moe.ROUTER_IMBALANCE_ON, and steps it's False for skip the imbalance (zeros; the logger drops them).
    top_k = jax.lax.stop_gradient(top_k_indices)
    imbalance_on = moe.ROUTER_IMBALANCE_ON.get()
    if imbalance_on is None:
      local_imbalance, global_imbalance = self._router_load_imbalance(top_k)
    else:
      local_imbalance, global_imbalance = jax.lax.cond(
          imbalance_on,
          self._router_load_imbalance,
          lambda _: (jnp.zeros((), jnp.float32), jnp.zeros((), jnp.float32)),
          top_k,
      )
    return jnp.stack([lb, local_imbalance, global_imbalance])

  def _router_load_imbalance(self, top_k_indices):
    """OLMo-core's per-layer "load imbalance" and "global load imbalance" (max / mean tokens per expert).

    OLMo-core counts each rank's routed tokens per expert over the step (all micro-batches) and reports the
    worst rank's max/mean ("load imbalance", reduced with max), and the same ratio for the counts summed over
    the batch ("global load imbalance"). A rank there is ``router_metrics_group_seqs`` sequences (16 for the
    hero: 2048 sequences on 128 GPUs), so the local ratio here groups the global batch into consecutive
    groups of that many sequences, independent of how the TPU mesh splits the batch. On the sparse path this
    runs inside the routed MoE's shard_map: per-sequence counts are summed over any sequence-sharding axes
    and gathered over the batch axes, so every device holds the whole batch's counts.
    """
    cfg = self.config
    if getattr(cfg, "olmoe3_router_metrics_scatter_counts", False):
      # Scatter-add each sequence's routed experts, instead of a [batch, seq, top_k, experts] one-hot mask.
      batch = top_k_indices.shape[0]
      seq_counts = (
          jnp.zeros((batch, self.num_experts), jnp.float32)
          .at[jnp.arange(batch)[:, None], top_k_indices.reshape(batch, -1)]
          .add(1.0)
      )
    else:
      expert_mask = jax.nn.one_hot(top_k_indices, num_classes=self.num_experts, dtype=jnp.float32)
      seq_counts = expert_mask.sum(axis=(1, 2))  # [batch, experts]
    if moe.LB_REDUCE_AXES.get() is not None:  # inside the sparse routed MoE's shard_map
      spec = self._logical_to_mesh_axes(("activation_batch", "activation_norm_length", None))
      batch_axes = tuple(mesh_axes_for_dim(spec[0])) if spec and spec[0] is not None else ()
      seq_axes = tuple(mesh_axes_for_dim(spec[1])) if spec and len(spec) > 1 and spec[1] is not None else ()
      if seq_axes:
        seq_counts = jax.lax.psum(seq_counts, seq_axes)
      if batch_axes:
        seq_counts = jax.lax.all_gather(seq_counts, batch_axes, axis=0, tiled=True)
    num_seqs = seq_counts.shape[0]
    group = int(getattr(cfg, "router_metrics_group_seqs", 0) or 0)
    if group <= 0 or num_seqs % group:
      group = num_seqs
    grouped = seq_counts.reshape(num_seqs // group, group, -1).sum(axis=1)  # [groups, experts]
    local_imbalance = jnp.max(jnp.max(grouped, axis=-1) / jnp.maximum(jnp.mean(grouped, axis=-1), 1e-9))
    total = seq_counts.sum(axis=0)
    global_imbalance = jnp.max(total) / jnp.maximum(jnp.mean(total), 1e-9)
    return local_imbalance, global_imbalance

  def __init__(self, *args, gate_in_features: int, **kwargs):
    super().__init__(*args, **kwargs)
    cfg = self.config
    if cfg.emo_enabled:
      if cfg.emo_min_document_expert_pool < self.num_experts_per_tok:
        raise ValueError("emo_min_document_expert_pool must be >= num_experts_per_tok.")
      pool_ok = cfg.emo_min_document_expert_pool <= cfg.emo_max_document_expert_pool <= self.num_experts
      eval_ok = self.num_experts_per_tok <= cfg.emo_eval_document_expert_pool <= self.num_experts
      if not pool_ok or not eval_ok:
        raise ValueError("EMo pool sizes must satisfy top_k <= min <= max/eval <= num_experts.")
      for size in cfg.emo_document_expert_pool_choices:
        if not self.num_experts_per_tok <= size <= self.num_experts:
          raise ValueError(
              f"emo_document_expert_pool_choices entry {size} must satisfy top_k <= size <= num_experts"
              f" ({self.num_experts_per_tok} <= size <= {self.num_experts})."
          )
      probs = cfg.emo_document_expert_pool_probs
      if probs:
        if len(probs) != len(cfg.emo_document_expert_pool_choices):
          raise ValueError("emo_document_expert_pool_probs must have one entry per emo_document_expert_pool_choices.")
        if min(probs) < 0 or abs(sum(probs) - 1.0) > 1e-6:
          raise ValueError("emo_document_expert_pool_probs must be non-negative and sum to 1.")
    if gate_in_features != self.moe_expert_input_dim:
      self.gate = moe.GateLogit(
          in_features_shape=gate_in_features,
          out_features_shape=self.num_experts,
          mesh=self.mesh,
          model_name=self.config.model_name,
          dtype=jnp.float32 if self.config.float32_gate_logits else self.dtype,
          weight_dtype=self.weight_dtype,
          quant=self.quant,
          kernel_init=self.kernel_init,
          kernel_axes=self.kernel_axes,
          use_bias=self.config.routed_bias,
          score_func=self.config.routed_score_func,
          matmul_precision=self.config.matmul_precision,
          shard_mode=self.config.shard_mode,
          rngs=self.rngs,
      )


class OLMoE3DecoderLayer(nnx.Module):
  """One OLMoE3 block: KDA or full attention, then shared FFN plus latent MoE."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      layer_idx: int,
      quant: None | Quant = None,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.layer_idx = layer_idx
    cfg = config
    self.activation_axis_names = ("activation_batch", "activation_norm_length", "activation_embed")
    # CompletedP multiplies both residual branches by (layers / base layers)^-alpha.
    base = _completedp_base(cfg)
    self.residual_multiplier = (
        1.0 if base is None else (cfg.num_decoder_layers / base.num_decoder_layers) ** -cfg.completedp_alpha
    )
    if base is not None and (cfg.moe_mlp_dim == cfg.moe_expert_input_dim or cfg.shared_expert_mlp_dim == cfg.emb_dim):
      # The MoE and shared-FFN inits tell their matrices apart by shape.
      raise NotImplementedError("CompletedP needs moe_mlp_dim != moe_expert_input_dim and a shared expert width != emb_dim.")
    if cfg.olmo_init_method == "nanochat" and (cfg.fused_mlp or cfg.prefuse_moe_weights):
      # Both store gate and up as one kernel, whose fan-out would count both halves.
      raise ValueError("olmo_init_method=nanochat does not support fused_mlp or prefuse_moe_weights.")

    def block_norm():
      return RMSNorm(
          num_features=cfg.emb_dim,
          dtype=cfg.dtype,
          weight_dtype=cfg.weight_dtype,
          kernel_axes=("norm",),
          epsilon=cfg.normalization_layer_epsilon,
          rngs=rngs,
      )

    # Four norms per block: pre and post around each sublayer.
    self.attn_in_norm = block_norm()
    self.attn_out_norm = block_norm()
    self.ffn_in_norm = block_norm()
    self.ffn_out_norm = block_norm()

    self.is_full_attention_layer = (layer_idx + 1) % cfg.inhomogeneous_layer_cycle_interval == 0
    if self.is_full_attention_layer:
      self.mixer = OLMoE3Attention(cfg, mesh, model_mode, quant, rngs=rngs)
    else:
      self.mixer = OLMoE3KimiDeltaAttention(cfg, mesh, quant, rngs=rngs)

    # Layer 0 is dense with a wide SwiGLU; MoE layers keep a narrow shared expert,
    # whose width is `shared_expert_mlp_dim` (defaults to the routed-expert width).
    self.is_dense_layer = layer_idx < cfg.first_num_dense_layers
    shared_dim = cfg.mlp_dim if self.is_dense_layer else cfg.shared_expert_mlp_dim
    self.shared_ffn = MlpBlock(
        config=cfg,
        mesh=mesh,
        in_features=cfg.emb_dim,
        intermediate_dim=shared_dim,
        activations=cfg.mlp_activations,
        # wi_0 / wi_1 are [emb, hidden], wo is [hidden, emb].
        kernel_init=_matrix_init_by_shape(
            cfg,
            lambda shape: "shared_ffn/wi_0/kernel" if shape[0] == cfg.emb_dim else "shared_ffn/wo/kernel",
            dense=self.is_dense_layer,
        ),
        intermediate_dropout_rate=cfg.dropout_rate,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        quant=quant,
        rngs=rngs,
    )

    if self.is_dense_layer:
      self.latent_down = self.latent_up = self.moe_block = None
    else:
      latent_dim = cfg.moe_expert_input_dim
      if latent_dim <= 0:
        raise ValueError("olmoe3 requires moe_expert_input_dim > 0 (the routed-expert latent width).")
      self.latent_down = _dense(
          cfg, cfg.emb_dim, latent_dim, ("embed", "mlp"), quant, rngs, mesh=mesh, kernel_init=_matrix_init(cfg, "latent_down/kernel")
      )
      self.latent_up = _dense(
          cfg, latent_dim, cfg.emb_dim, ("mlp", "embed"), quant, rngs, mesh=mesh, kernel_init=_matrix_init(cfg, "latent_up/kernel")
      )
      self.moe_block = OLMoE3LatentRoutedMoE(
          config=cfg,
          num_experts=cfg.num_experts,
          num_experts_per_tok=cfg.num_experts_per_tok,
          mesh=mesh,
          # The router is [emb, experts]; the experts' wi_0 / wi_1 are [experts, latent, hidden], wo [experts, hidden, latent].
          kernel_init=_matrix_init_by_shape(
              cfg,
              lambda shape: "moe_block/gate/kernel"
              if len(shape) == 2
              else ("moe_block/wo" if shape[1:] == (cfg.moe_mlp_dim, cfg.moe_expert_input_dim) else "moe_block/wi_0"),
          ),
          # The router weight is small (emb x experts); replicated, it needs no all-gather per use.
          kernel_axes=(None, None) if cfg.olmoe3_replicate_router else ("embed", None),
          intermediate_dim=cfg.moe_mlp_dim,
          dtype=cfg.dtype,
          weight_dtype=cfg.weight_dtype,
          quant=quant,
          rngs=rngs,
          gate_in_features=cfg.emb_dim,
      )

    if cfg.gathered_weights != "remat":
      # Keep every dense kernel's all-gathered copy as a residual (FSDP without reshard-after-forward).
      for _, module in nnx.iter_modules(self):
        if isinstance(module, DenseGeneral):
          module.save_gathered_kernel = True

  def __call__(
      self,
      inputs: jnp.ndarray,
      decoder_segment_ids: None | jnp.ndarray,
      decoder_positions: None | jnp.ndarray,
      deterministic: bool,
      model_mode: str,
      previous_chunk=None,
      slot: None | int = None,
      kv_cache: None | dict[str, Any] = None,
      attention_metadata: None | dict[str, Any] = None,
  ):
    del previous_chunk, slot
    if isinstance(inputs, tuple):
      inputs = inputs[0]

    inputs = nn.with_logical_constraint(inputs, self.activation_axis_names)
    inputs = checkpoint_name(inputs, "decoder_layer_input")

    normed = self.attn_in_norm(inputs)
    # OLMo-core trains with whole-instance mixers; the segment ids then only shape the EMo pools.
    mixer_segment_ids = None if self.config.olmoe3_segments_for_emo_only else decoder_segment_ids
    if self.is_full_attention_layer:
      mixer_out, kv_cache = self.mixer(
          normed,
          mixer_segment_ids,
          decoder_positions,
          deterministic,
          model_mode,
          kv_cache=kv_cache,
          attention_metadata=attention_metadata,
      )
    else:
      mixer_out = self.mixer(normed, mixer_segment_ids)

    attn_branch = self.attn_out_norm(mixer_out)
    hidden = inputs + (attn_branch if self.residual_multiplier == 1.0 else self.residual_multiplier * attn_branch)
    hidden = nn.with_logical_constraint(hidden, self.activation_axis_names)

    ffn_in = self.ffn_in_norm(hidden)
    precomputed_gate = None
    if self.moe_block is not None:
      # Compute the router logits once and share them between routing and the router z-loss below, instead of
      # relying on XLA to fold a second `gate(ffn_in)` call.
      gate_dtype = jnp.float32 if self.config.float32_gate_logits else self.config.dtype
      router_logits, pre_bias_logits = self.moe_block.gate(ffn_in.astype(gate_dtype))
      router_logits = checkpoint_name(router_logits, "moe_router_logits")
      precomputed_gate = (router_logits, pre_bias_logits)
    if self._can_fuse_ffn_in():
      ffn_out, latent_in = self._fused_ffn_in_and_shared_ffn(ffn_in, deterministic=deterministic)
    else:
      ffn_out = self.shared_ffn(ffn_in, deterministic=deterministic)
      latent_in = self.latent_down(ffn_in) if self.moe_block is not None else None
    if self.moe_block is not None:
      # input_ids carries the packed segment ids for the EMo document pools.
      routed, load_balance_loss, _ = self.moe_block(
          latent_in,
          gate_inputs=ffn_in,
          input_ids=decoder_segment_ids,
          precomputed_gate_logits=precomputed_gate,
      )
      if self.config.load_balance_loss_weight > 0.0 and load_balance_loss is not None:
        # Reference auxiliary loss adds a router z-loss: 1e-5 * mean(logsumexp(router logits)^2).
        z_loss = jnp.mean(jax.scipy.special.logsumexp(router_logits.astype(jnp.float32), axis=-1) ** 2)
        if self.config.olmoe3_router_metrics:
          # [lb, load imbalance, global load imbalance] from the routed MoE (see load_balance_loss).
          metrics = load_balance_loss
          load_balance_loss = metrics[0]
          self.sow(
              nnx.Intermediate,
              "olmoe3_router_metrics",
              jnp.stack([load_balance_loss, _ROUTER_Z_LOSS_WEIGHT * z_loss, metrics[1], metrics[2]]),
          )
        self.sow(nnx.Intermediate, "moe_lb_loss", load_balance_loss + _ROUTER_Z_LOSS_WEIGHT * z_loss)
      ffn_out = ffn_out + self.latent_up(routed)

    ffn_branch = self.ffn_out_norm(ffn_out)
    layer_output = hidden + (ffn_branch if self.residual_multiplier == 1.0 else self.residual_multiplier * ffn_branch)
    layer_output = nn.with_logical_constraint(layer_output, self.activation_axis_names)

    if self.config.scan_layers:
      return layer_output, None
    return layer_output, kv_cache

  def _can_fuse_ffn_in(self) -> bool:
    """Whether `shared_ffn.wi_0`, `shared_ffn.wi_1`, and `latent_down` can share one matmul over `ffn_in`."""
    if not self.config.kda_fused_input_proj or self.moe_block is None:
      return False
    if self.shared_ffn.quant is not None or self.shared_ffn.mlp_layer_norm is not None:
      return False
    if self.shared_ffn.use_bias or self.config.fused_mlp or len(self.shared_ffn.activations) != 2:
      return False
    if self.shared_ffn.wi_0.quant_dot_general is not None or self.shared_ffn.wi_1.quant_dot_general is not None:
      return False
    if self.latent_down is not None and (self.latent_down.quant_dot_general is not None or self.latent_down.use_bias):
      return False
    if int(self.shared_ffn.wi_0.kernel[...].shape[-1]) > 7168:
      return False
    mlp_axes = nn.logical_to_mesh_axes(("mlp",), self.config.logical_axis_rules)[0]
    if mlp_axes is not None:
      axes = mlp_axes if isinstance(mlp_axes, tuple) else (mlp_axes,)
      if any(self.mesh.shape.get(a, 1) > 1 for a in axes if a is not None):
        return False
    return True

  def _fused_ffn_in_and_shared_ffn(self, ffn_in: jnp.ndarray, *, deterministic: bool):
    """Fuses `(shared_ffn.wi_0, shared_ffn.wi_1, latent_down)` into one matmul over `ffn_in`."""
    mods = (self.shared_ffn.wi_0, self.shared_ffn.wi_1, self.latent_down)
    dtype = self.config.dtype
    kernels = [jnp.asarray(m.kernel[...], dtype) for m in mods]
    kernel = jnp.concatenate(kernels, axis=-1)
    out = linears.dot_general_with_gathered_kernel(
        jnp.asarray(ffn_in, dtype),
        kernel,
        (ffn_in.ndim - 1,),
        (0,),
        self.config.matmul_precision,
        self.mesh,
        ("embed", None),
        self.config.shard_mode,
        save_gathered_kernel=(self.config.gathered_weights != "remat"),
    )
    mlp_dim = int(kernels[0].shape[-1])

    @functools.partial(jax.custom_vjp, nondiff_argnums=(1,))
    def _split_3_last_axis(arr: jnp.ndarray, n: int):
      return arr[..., :n], arr[..., n : 2 * n], arr[..., 2 * n :]

    def _split_3_fwd(arr: jnp.ndarray, n: int):
      return _split_3_last_axis(arr, n), ()

    def _split_3_bwd(n: int, res, grads):
      del n, res
      g0, g1, g2 = grads
      return (jnp.concatenate([g0, g1, g2], axis=-1),)

    _split_3_last_axis.defvjp(_split_3_fwd, _split_3_bwd)
    wi_0_out, wi_1_out, latent_in = _split_3_last_axis(out, mlp_dim)
    wi_0_out = checkpoint_name(wi_0_out, "mlpwi_0")
    wi_1_out = checkpoint_name(wi_1_out, "mlpwi_1")
    if self.config.activations_in_float32:
      wi_0_out = wi_0_out.astype(jnp.float32)
      wi_1_out = wi_1_out.astype(jnp.float32)
    to_act = linears._convert_to_activation_function  # pylint: disable=protected-access
    act_0 = to_act(self.shared_ffn.activations[0])(wi_0_out)
    act_1 = to_act(self.shared_ffn.activations[1])(wi_1_out)
    ffn_mid = (act_0 * act_1).astype(self.shared_ffn.dtype)
    ffn_mid = self.shared_ffn.dropout(ffn_mid, deterministic=deterministic)
    ffn_mid = self.shared_ffn._maybe_shard_with_logical(  # pylint: disable=protected-access
        ffn_mid, self.shared_ffn.intermediate_logical
    )
    ffn_out = checkpoint_name(self.shared_ffn.wo(ffn_mid), "mlpwo")
    return ffn_out, latent_in


class OLMoE3ScannableBlock(nnx.Module):
  """One full mixer cycle (``inhomogeneous_layer_cycle_interval`` layers).

  OLMoE3 layers are not homogeneous (the mixer alternates and layer 0 is dense),
  so scanning happens over whole cycles rather than single layers.

  ``first_layer_idx`` is the global index of this block's first layer. Scanned
  copies share one parameter set, so the dense prefix cycle must be built and
  applied separately with ``first_layer_idx=0`` while the scanned cycles start
  past ``first_num_dense_layers``. Passing the within-cycle index instead would
  make layer 0 of *every* cycle dense.

  With ``olmoe3_per_layer_remat`` each layer is rematerialized on its own under
  ``remat_policy_fn``, so the backward holds one layer's recomputed activations
  instead of the whole cycle's.
  """

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      quant=None,
      *,
      first_layer_idx: int = 0,
      remat_policy_fn: Any | None = None,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.remat_policy_fn = remat_policy_fn
    self.remat_layers = config.olmoe3_per_layer_remat and config.remat_policy != "none"
    for i in range(config.inhomogeneous_layer_cycle_interval):
      setattr(
          self,
          f"layer_{i}",
          OLMoE3DecoderLayer(
              config=config,
              mesh=mesh,
              model_mode=model_mode,
              layer_idx=first_layer_idx + i,
              quant=quant,
              rngs=rngs.fork(),
          ),
      )

  def __call__(
      self,
      carry: jnp.ndarray,
      decoder_segment_ids: None | jnp.ndarray,
      decoder_positions: None | jnp.ndarray,
      deterministic: bool,
      model_mode: str,
      previous_chunk=None,
      slot: None | int = None,
  ):
    x = carry
    for i in range(self.config.inhomogeneous_layer_cycle_interval):
      layer = getattr(self, f"layer_{i}")
      args = (decoder_segment_ids, decoder_positions, deterministic, model_mode, previous_chunk, slot)
      if not self.remat_layers:
        x, _ = layer(x, *args)
        continue
      graphdef, state = nnx.split(layer)

      def run_layer(state_in, x_in, graphdef=graphdef, args=args):
        merged = nnx.merge(graphdef, state_in)
        out, _ = merged(x_in, *args)
        return out, nnx.state(merged)

      # The layers are unrolled here, so CSE must be blocked or XLA merges the
      # recompute back into the forward and saves everything.
      # pylint: disable-next=too-many-function-args
      x, new_state = jax.checkpoint(run_layer, policy=self.remat_policy_fn, prevent_cse=True)(state, x)
      nnx.update(layer, new_state)
    return x, None


OLMoE3DecoderLayerToLinen = nnx_wrappers.to_linen_class(
    OLMoE3DecoderLayer,
    base_metadata_fn=max_initializers.variable_to_logically_partitioned,
)

OLMoE3ScannableBlockToLinen = nnx_wrappers.to_linen_class(
    OLMoE3ScannableBlock,
    base_metadata_fn=max_initializers.variable_to_logically_partitioned,
)
