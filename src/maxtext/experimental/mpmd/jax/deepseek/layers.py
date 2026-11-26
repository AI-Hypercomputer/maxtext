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

"""Primitive layers, as plain functions over explicit parameter pytrees.

Every function here reproduces the corresponding MaxText NNX module's forward
pass exactly, including where activations are rounded to bf16.  The rule
throughout MaxText is: weights are stored in `weight_dtype` (fp32) and cast to
`dtype` (bf16) immediately before every matmul; RMSNorm is the only place that
promotes activations to fp32, and it drops back to bf16 *before* applying the
scale.

Every op whose result type explicit sharding cannot infer goes through
`maxtext.experimental.mpmd.jax.sharding_mode` (see its docstring), which reads the mode
off the mesh in context.  Under auto each helper is the expression this file
used to write inline, so that program is unchanged.
Activations are batch-major throughout, so a result's spec is always
`mode.act_spec(ndim)`: the batch on the stage's axes, nothing else sharded.
"""

import contextlib
import contextvars

import jax
import jax.numpy as jnp

from maxtext.experimental.mpmd.jax.deepseek.config import DeepSeekConfig
from maxtext.experimental.mpmd.jax import sharding_mode as mode


def precision_of(config: DeepSeekConfig):
  return jax.lax.Precision(config.matmul_precision)


def precision_or_none(config: DeepSeekConfig):
  """`precision_of`, but None at the default precision.

  For the matmuls that ignored `matmul_precision` until a TPU parity check at
  'highest' tripped on them: the grouped expert matmuls, RMSNorm's scale and
  the reference attention's QK product.  The grouped matmuls and the QK
  product were lowered at DEFAULT, which on TPU means one bf16 pass, so an
  fp32 model at 'highest' still rounded them, and that noise was enough to
  flip the router's top-k choices.  (RMSNorm's scale product contracts
  nothing: XLA:TPU makes it an elementwise multiply at any precision.)  They
  used to pass None, and still do at the default: an explicit DEFAULT would
  lower the same, but it would also override `jax_default_matmul_precision`,
  which governs a None.

  MaxText leaves its own grouped matmuls at the default whatever
  `matmul_precision` says (moe.py: megablox's Pallas dot and the
  `jax.lax.ragged_dot` fallback pass none, tokamax passes DEFAULT -- on TPU
  each is one bf16-rounding pass over fp32 operands), so at a non-default
  precision the port's grouped matmuls are more precise than the MaxText
  they mirror.  The CPU parity tests cannot show it: XLA:CPU computes an
  fp32 dot in fp32 at any precision.
  """
  precision = precision_of(config)
  return None if precision == jax.lax.Precision.DEFAULT else precision


# Lean residuals (`lean_residuals()`, which the pipeline's saved-residual forwards trace under): RMSNorm, its
# scale's product included, and the gated MLPs' activation product run under jax.checkpoint, so a VJP saves their
# bf16 inputs (and the matmuls' operands) instead of the f32 and bf16 normalised activations and the silu / sigmoid /
# product intermediates -- Transformer Engine's RMSNorm and Megatron's fused SwiGLU keep about as little.  Where a
# norm's output is a matmul's lhs, `rms_norm_read` takes the matmul into the norm's checkpoint, so that its VJP does
# not keep the output either (Megatron's layernorm recompute).  The recomputation is elementwise.  attention.mla and
# rope.apply_yarn read the mode too.
# (The pipeline's routed experts go further under it: their grouped matmuls keep what their operand is recomputed
# from, the gated product's inputs or the dispatched tokens, instead of the operand -- mpmd/jax/ds_ep.Recompute.)
_LEAN = contextvars.ContextVar("lean_residuals", default=False)


@contextlib.contextmanager
def lean_residuals(on=True):
  token = _LEAN.set(on)
  try:
    yield
  finally:
    _LEAN.reset(token)


def lean():
  return _LEAN.get()


def gated_product(h0, h1, act, dtype):
  """act(h0) * h1 in `dtype` -- under jax.checkpoint when lean (its VJP then keeps h0 and h1 only)."""
  f = lambda a, b: (activation_fn(act)(a) * b).astype(dtype)
  return (jax.checkpoint(f) if lean() else f)(h0, h1)


def rms_norm(params, x, config: DeepSeekConfig):
  """`normalizations.RMSNorm.__call__`.

  The scale is FSDP-sharded on its only axis, the feature axis the batch-
  sharded activation leaves whole.  Under explicit their product would put
  the same mesh axis on the batch and on the features, which is illegal, so
  the result is typed like the activation and XLA aligns the scale (MaxText
  reshards it to the activation's last-dim spec, `normalizations.align_scale_with_normalized_axis`).
  The product is an einsum, i.e. a dot_general, so it takes the precision.

  Under lean residuals the whole norm runs under jax.checkpoint: its VJP
  keeps x and the scale, and its backward recomputes the bf16 normalised
  activation the scale's gradient reads (a VJP of the product alone would
  keep it, beside x).
  """

  def norm(x, scale):
    x = jnp.asarray(x, jnp.float32)
    mean2 = jnp.mean(jax.lax.square(x), axis=-1, keepdims=True)
    y = jnp.asarray(x * jax.lax.rsqrt(mean2 + config.normalization_layer_epsilon), config.dtype)
    scale = jnp.asarray(scale, config.dtype)
    return mode.einsum("...k,k->...k", y, scale, out_spec=mode.act_spec(y.ndim), precision=precision_or_none(config))

  return (jax.checkpoint(norm) if lean() else norm)(x, params["scale"])


def rms_norm_read(params, x, config: DeepSeekConfig, read):
  """`read(rms_norm(params, x, config))`, `read` being what reads the norm's
  output: the projections that take it as their lhs, then anything
  elementwise.

  Under lean residuals one jax.checkpoint runs both, saving only dot
  products without batch dims -- the projections' outputs, where read's
  backward needs them.  A projection's VJP would keep its lhs, the norm's
  output, beside the norm's own input; here the backward recomputes it from
  that input (Megatron's layernorm recompute), and read's elementwise values
  with it, but no projection.  (The scale's product has a batch dim, the
  feature axis: it is recomputed, elementwise as it is.)  So `read` holds
  matmuls and elementwise ops only: a collective or a custom-VJP kernel's
  outputs would be recomputed too.  Inside, the norm and `read` trace
  without lean residuals: the checkpoint's policy, not theirs, decides what
  nested checkpoints keep, and theirs would only recompute the same values a
  second time.  Outside lean residuals this is the plain composition.
  """
  if not lean():
    return read(rms_norm(params, x, config))

  def f(x, params):
    with lean_residuals(False):
      return read(rms_norm(params, x, config))

  return jax.checkpoint(f, policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable)(x, params)


def dense_general(params, inputs, config: DeepSeekConfig, axis=(-1,)):
  """`linears.DenseGeneral.__call__`.

  The kernel is `in_features_shape + out_features_shape`; `axis` names the
  input axes contracted against the leading kernel axes.

  FSDP stores the kernel sharded on its biggest axis.  That is an OUTPUT
  axis for MLA's `out`, the head, the dense layers' `wi_*` and, at DS-S and
  DeepSeek-V3 sizes, the shared expert's `wo` (at some test sizes `wq_b` /
  `wkv_b` too).  Under explicit an output-split matmul has no legal result
  type of its own, so it is typed like the activation: the kernel stays
  sharded and XLA gathers it.
  """
  kernel = jnp.asarray(params["kernel"], config.dtype)
  inputs = jnp.asarray(inputs, config.dtype)
  norm_axis = tuple(a % inputs.ndim for a in axis)
  contract_ind = tuple(range(len(axis)))
  out_ndim = inputs.ndim + kernel.ndim - 2 * len(axis)
  out = mode.dot_general(
      inputs,
      kernel,
      ((norm_axis, contract_ind), ((), ())),
      out_spec=mode.act_spec(out_ndim),
      precision=precision_of(config),
  )
  if "bias" in params:
    out += jnp.asarray(params["bias"], config.dtype)
  return out


def embed_lookup(params, tokens, config: DeepSeekConfig):
  """`embeddings.Embed.__call__` -- cast the table to bf16 first, then gather.

  The table is never gathered: a row lookup in a vocab-sharded table is a
  local masked gather plus a reduction of the `[B, L, emb]` result, and
  gathering a 129k-row table to replicated per lookup would cost far more
  (MaxText's explicit lookup leaves it sharded too).

  Under explicit the table is looked up with `.at[].get(out_sharding=)`,
  which jax runs as an auto_axes island (`mode.take`).
  """
  table = jnp.asarray(params["embedding"], config.dtype)
  ids = tokens.astype(jnp.int32)
  return mode.take(table, ids, out_spec=mode.act_spec(ids.ndim + 1))


_ACTIVATIONS = {
    "silu": jax.nn.silu,
    "swish": jax.nn.silu,
    "gelu": jax.nn.gelu,
    "relu": jax.nn.relu,
    "sigmoid": jax.nn.sigmoid,
    "tanh": jnp.tanh,
    "linear": lambda x: x,
}


def activation_fn(name: str):
  return _ACTIVATIONS[name]


def mlp_block(params, inputs, config: DeepSeekConfig):
  """`linears.MlpBlock.__call__` -- gated MLP with `wi_0`/`wi_1`/`wo`."""
  activated = None
  for idx, act in enumerate(config.mlp_activations):
    y = dense_general(params[f"wi_{idx}"], inputs, config)
    y = activation_fn(act)(y)
    activated = y if activated is None else activated * y
  activated = jnp.asarray(activated, config.dtype)
  return dense_general(params["wo"], activated, config)
