# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wan 2.1 UMT5 text encoder (Flax NNX)."""

import math
from typing import Optional

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.m3.core.sharding import Sharding, TensorType
from maxtext.m3.models.wan.utils_wan import constrain, param_metadata

BATCH = "activation_batch"
LENGTH = "activation_length"
EMBED = "activation_embed"
HEADS = "activation_heads"
MLP = "activation_mlp"


def check_tensor_parallel_divides(sharding: Sharding, mesh: jax.sharding.Mesh, divisor: int) -> None:
  """Raises unless the mesh axes `sharding` gives `heads` multiply to a divisor of `divisor` (gcd(heads, d_ff, vocab)).

  The text encoder is tensor parallel (`text_encoder_logical_axis_rules`: heads, MLP width and vocabulary over every
  mesh axis by default), so those axes must divide gcd(num_heads, d_ff, vocab_size) -- 64 for umt5-xxl. On a bigger
  mesh drop axes in the table; the encoder is then replicated over them.
  """
  tp_axes = sharding.map_axis("heads", "heads", TensorType.WEIGHT) or ()
  tp_axes = (tp_axes,) if isinstance(tp_axes, str) else tp_axes
  size = 1
  for axis in tp_axes:
    size *= mesh.shape[axis]
  if divisor % size:
    raise ValueError(
        f"The text encoder splits heads / mlp / vocab over {tp_axes} ({size} devices), which does not divide "
        f"gcd(num_heads, d_ff, vocab_size) = {divisor}. Drop mesh axes in `text_encoder_logical_axis_rules`."
    )


_ACTIVATIONS = {
    "gelu_new": lambda x: jax.nn.gelu(x, approximate=True),
    "gelu_pytorch_tanh": lambda x: jax.nn.gelu(x, approximate=True),
    "gelu": lambda x: jax.nn.gelu(x, approximate=False),
    "relu": jax.nn.relu,
}


def relative_position_bucket(
    relative_position: np.ndarray, num_buckets: int, max_distance: int, bidirectional: bool = True
) -> np.ndarray:
  """T5 relative position -> bucket index, as `UMT5Attention._relative_position_bucket` (encoder: bidirectional).

  Computed on the host in numpy with float32 arithmetic (like the torch reference), so the bucket table does not
  depend on the accelerator's `log` implementation.
  """
  relative_position = np.asarray(relative_position, dtype=np.int64)
  relative_buckets = np.zeros_like(relative_position)
  if bidirectional:
    num_buckets //= 2
    relative_buckets += (relative_position > 0).astype(np.int64) * num_buckets
    relative_position = np.abs(relative_position)
  else:
    relative_position = -np.minimum(relative_position, 0)
  max_exact = num_buckets // 2
  is_small = relative_position < max_exact
  with np.errstate(divide="ignore"):  # log(0) for small positions, discarded by `is_small` below.
    log_ratio = np.log(relative_position.astype(np.float32) / np.float32(max_exact)) / np.float32(
        math.log(max_distance / max_exact)
    )
  log_ratio = log_ratio * np.float32(num_buckets - max_exact)
  if_large = max_exact + np.where(is_small, 0, log_ratio).astype(np.int64)
  if_large = np.minimum(if_large, num_buckets - 1)
  return relative_buckets + np.where(is_small, relative_position, if_large)


class UMT5LayerNorm(nnx.Module):
  """T5 RMS norm: scale only, no mean subtraction, variance accumulated in float32."""

  def __init__(self, dim: int, eps: float, weights_dtype=jnp.float32, sharding: Sharding | None = None):
    self.eps = eps
    self.scale = nnx.Param(
        nnx.initializers.ones(None, (dim,), weights_dtype), **param_metadata(sharding, "norm_scale", ("norm",))
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    variance = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
    x = x * jax.lax.rsqrt(variance + self.eps)
    scale = self.scale[...]
    if scale.dtype in (jnp.float16, jnp.bfloat16):
      x = x.astype(scale.dtype)
    return scale * x


def _linear(in_features, out_features, name, axes, *, dtype, weights_dtype, precision, rngs, sharding):
  return nnx.Linear(
      in_features,
      out_features,
      use_bias=False,
      dtype=dtype,
      param_dtype=weights_dtype,
      precision=precision,
      kernel_init=nnx.initializers.lecun_normal(),
      kernel_metadata=param_metadata(sharding, name, axes),
      rngs=rngs,
  )


class UMT5Attention(nnx.Module):
  """Bidirectional self-attention with a per-layer relative position bias and unscaled scores."""

  def __init__(
      self,
      d_model: int,
      d_kv: int,
      num_heads: int,
      relative_attention_num_buckets: int,
      relative_attention_max_distance: int,
      *,
      dtype,
      weights_dtype,
      precision,
      rngs: nnx.Rngs,
      sharding: Sharding | None = None,
  ):
    self.num_heads = num_heads
    self.d_kv = d_kv
    self.num_buckets = relative_attention_num_buckets
    self.max_distance = relative_attention_max_distance
    self.dtype = dtype
    self.precision = precision
    self.sharding = sharding
    inner_dim = num_heads * d_kv
    kw = {"dtype": dtype, "weights_dtype": weights_dtype, "precision": precision, "rngs": rngs, "sharding": sharding}
    self.query = _linear(d_model, inner_dim, "attn_q", ("embed", "heads"), **kw)
    self.key = _linear(d_model, inner_dim, "attn_k", ("embed", "heads"), **kw)
    self.value = _linear(d_model, inner_dim, "attn_v", ("embed", "heads"), **kw)
    self.proj_attn = _linear(inner_dim, d_model, "attn_out", ("heads", "embed"), **kw)
    self.relative_attention_bias = nnx.Embed(
        relative_attention_num_buckets,
        num_heads,
        dtype=dtype,
        param_dtype=weights_dtype,
        embedding_init=nnx.initializers.normal(1.0),
        embedding_metadata=param_metadata(sharding, "relative_attention_bias", (None, "heads")),
        rngs=rngs,
    )

  def position_bias(self, length: int) -> jax.Array:
    """(num_heads, length, length) additive bias of `UMT5Attention.compute_bias`."""
    position = np.arange(length)
    buckets = relative_position_bucket(position[None, :] - position[:, None], self.num_buckets, self.max_distance)
    values = self.relative_attention_bias(jnp.asarray(buckets, dtype=jnp.int32))  # (q, k, heads)
    return jnp.transpose(values, (2, 0, 1))

  def __call__(self, x: jax.Array, attention_mask: jax.Array) -> jax.Array:
    batch, length, _ = x.shape

    def heads(proj):
      y = proj(x).reshape(batch, length, self.num_heads, self.d_kv)
      return constrain(self.sharding, y, "attn_heads", (BATCH, LENGTH, HEADS, None))

    query, key, value = heads(self.query), heads(self.key), heads(self.value)
    scores = jnp.einsum("bqhd,bkhd->bhqk", query, key, precision=self.precision)  # UMT5: no 1/sqrt(d) scaling.
    scores = scores + self.position_bias(length)[None].astype(scores.dtype)
    scores = jnp.where(attention_mask[:, None, None, :], scores, jnp.finfo(scores.dtype).min)
    weights = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(value.dtype)
    out = jnp.einsum("bhqk,bkhd->bqhd", weights, value, precision=self.precision)
    out = constrain(self.sharding, out, "attn_heads", (BATCH, LENGTH, HEADS, None))
    return self.proj_attn(out.reshape(batch, length, self.num_heads * self.d_kv))


class UMT5FeedForward(nnx.Module):
  """T5 FFN: `wo(act(wi_0(x)) * wi_1(x))` when gated, else `wo(act(wi(x)))`."""

  def __init__(
      self,
      d_model: int,
      d_ff: int,
      dense_act_fn: str,
      is_gated_act: bool,
      *,
      dtype,
      weights_dtype,
      precision,
      rngs,
      sharding: Sharding | None = None,
  ):
    if dense_act_fn not in _ACTIVATIONS:
      raise ValueError(f"Unsupported dense_act_fn {dense_act_fn!r}; supported: {sorted(_ACTIVATIONS)}")
    self.act = _ACTIVATIONS[dense_act_fn]
    self.is_gated_act = is_gated_act
    self.sharding = sharding
    kw = {"dtype": dtype, "weights_dtype": weights_dtype, "precision": precision, "rngs": rngs, "sharding": sharding}
    if is_gated_act:
      self.wi_0 = _linear(d_model, d_ff, "ffn_wi_0", ("embed", "mlp"), **kw)
      self.wi_1 = _linear(d_model, d_ff, "ffn_wi_1", ("embed", "mlp"), **kw)
    else:
      self.wi = _linear(d_model, d_ff, "ffn_wi", ("embed", "mlp"), **kw)
    self.wo = _linear(d_ff, d_model, "ffn_wo", ("mlp", "embed"), **kw)

  def __call__(self, x: jax.Array) -> jax.Array:
    if self.is_gated_act:
      hidden = self.act(self.wi_0(x)) * self.wi_1(x)
    else:
      hidden = self.act(self.wi(x))
    hidden = constrain(self.sharding, hidden, "ffn_hidden", (BATCH, LENGTH, MLP))
    return self.wo(hidden)


class UMT5Block(nnx.Module):
  """Pre-norm residual block: self-attention then FFN (HF `UMT5Block` with `layer[0]` / `layer[1]`)."""

  def __init__(self, config: dict, *, dtype, weights_dtype, precision, rngs: nnx.Rngs, sharding: Sharding | None = None):
    kw = {"dtype": dtype, "weights_dtype": weights_dtype, "precision": precision, "rngs": rngs, "sharding": sharding}
    self.sharding = sharding
    self.attn_norm = UMT5LayerNorm(config["d_model"], config["layer_norm_epsilon"], weights_dtype, sharding)
    self.attn = UMT5Attention(
        config["d_model"],
        config["d_kv"],
        config["num_heads"],
        config["relative_attention_num_buckets"],
        config["relative_attention_max_distance"],
        **kw,
    )
    self.ffn_norm = UMT5LayerNorm(config["d_model"], config["layer_norm_epsilon"], weights_dtype, sharding)
    self.ffn = UMT5FeedForward(config["d_model"], config["d_ff"], config["dense_act_fn"], config["is_gated_act"], **kw)

  def __call__(self, x: jax.Array, attention_mask: jax.Array) -> jax.Array:
    x = x + self.attn(self.attn_norm(x), attention_mask)
    x = constrain(self.sharding, x, "residual", (BATCH, LENGTH, EMBED))
    x = x + self.ffn(self.ffn_norm(x))
    return constrain(self.sharding, x, "residual", (BATCH, LENGTH, EMBED))


class UMT5EncoderModel(nnx.Module):
  """Flax NNX port of `transformers.UMT5EncoderModel` (the umt5-xxl text encoder of Wan 2.1).

  The architecture arguments are the fields of the diffusers `text_encoder/config.json` (Hugging Face
  `UMT5Config`), so `UMT5EncoderModel(**config, rngs=...)` accepts the config stored by the checkpoint converter.
  `__call__(input_ids, attention_mask)` returns the last hidden state `[batch, length, d_model]`.

  Inference-only (no dropout). Differences from T5 that matter here: every layer has its own relative position
  bias (T5 shares the first layer's), the query/key dot product is not scaled, and the FFN is gated-GELU
  (`gelu_new`, tanh approximation). Parameters and activations are sharded by `sharding` (a `WanSharding` from
  `text_encoder_logical_axis_rules` in the pipeline, i.e. tensor parallel over the mesh); `sharding=None` leaves them
  unannotated, for mesh-free use only (checkpoint conversion, abstract shapes).
  """

  def __init__(
      self,
      vocab_size: int = 256384,
      d_model: int = 4096,
      d_kv: int = 64,
      d_ff: int = 10240,
      num_layers: int = 24,
      num_heads: int = 64,
      relative_attention_num_buckets: int = 32,
      relative_attention_max_distance: int = 128,
      layer_norm_epsilon: float = 1e-6,
      dense_act_fn: str = "gelu_new",
      is_gated_act: bool = True,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: Optional[jax.lax.Precision] = None,
      *,
      rngs: nnx.Rngs,
      sharding: Sharding | None = None,
  ):
    self.vocab_size = vocab_size
    self.d_model = d_model
    self.d_ff = d_ff
    self.num_heads = num_heads
    self.dtype = jnp.dtype(dtype)
    self.sharding = sharding
    weights_dtype = jnp.dtype(weights_dtype)
    block_config = {
        "d_model": d_model,
        "d_kv": d_kv,
        "d_ff": d_ff,
        "num_heads": num_heads,
        "relative_attention_num_buckets": relative_attention_num_buckets,
        "relative_attention_max_distance": relative_attention_max_distance,
        "layer_norm_epsilon": layer_norm_epsilon,
        "dense_act_fn": dense_act_fn,
        "is_gated_act": is_gated_act,
    }
    self.token_embedding = nnx.Embed(
        vocab_size,
        d_model,
        dtype=self.dtype,
        param_dtype=weights_dtype,
        embedding_init=nnx.initializers.normal(1.0),
        embedding_metadata=param_metadata(sharding, "token_embedding", ("vocab", "embed")),
        rngs=rngs,
    )
    self.blocks = nnx.data(
        [
            UMT5Block(
                block_config,
                dtype=self.dtype,
                weights_dtype=weights_dtype,
                precision=precision,
                rngs=rngs,
                sharding=sharding,
            )
            for _ in range(num_layers)
        ]
    )
    self.final_norm = UMT5LayerNorm(d_model, layer_norm_epsilon, weights_dtype, sharding)

  def __call__(self, input_ids: jax.Array, attention_mask: Optional[jax.Array] = None) -> jax.Array:
    """Returns the encoder's last hidden state `[batch, length, d_model]`.

    Args:
      input_ids: `[batch, length]` token ids.
      attention_mask: `[batch, length]`, nonzero for real tokens; padded keys are masked out. Defaults to all ones.
    """
    if attention_mask is None:
      attention_mask = jnp.ones(input_ids.shape, dtype=jnp.bool_)
    attention_mask = attention_mask.astype(jnp.bool_)
    x = self.token_embedding(input_ids)
    x = constrain(self.sharding, x, "residual", (BATCH, LENGTH, EMBED))
    for block in self.blocks:
      x = block(x, attention_mask)
    return self.final_norm(x)
