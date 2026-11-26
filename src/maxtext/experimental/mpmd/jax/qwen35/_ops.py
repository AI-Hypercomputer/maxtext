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

"""Every sharding-sensitive call site of the Qwen3.5 port, one small function each.

The model modules (`layers`, `gdn`, `attention`, `moe`, `model`) never touch a
weight, gather an embedding, start a scan carry, regroup heads, run the conv,
dispatch the MoE, place the GDN core or call an attention kernel except
through this module, and make no sharding call (`reshard`,
`with_sharding_constraint`, `out_sharding=`, `shard_map`, `auto_axes`) of
their own; `qwen35_model_test.CallSiteTest` enforces both, and
`qwen35_explicit_test.OpsCallSiteTest` that this module's own sharding calls
are `mode`'s.  The bodies here go through
`src/maxtext/experimental/mpmd/jax/sharding_mode.py` (whose docstring has the modes and the
three helper groups), as the DeepSeek port's do, so that

  * under auto -- and with no mesh at all -- every body is the plain
    expression the port traced before these bodies existed: the same auto
    program, byte for byte (a canonical-HLO A/B showed it when they came in);
  * under explicit every value the model makes has a type, which is what
    lets the same code trace there;
  * `config.gdn_core='manual'` forces the SAME per-device GDN core in both
    modes, so an auto-vs-explicit comparison compares typing.

What explicit needs here, found by probing an Explicit mesh with batch-
sharded activations (jax 0.10.2 / 0.11 / 0.12): an FSDP weight sharded on an
output dim needs a gather or an `out_sharding` (DeepSeek's
`dense_general`); the GDN head regroup reshape (E3) and the scan carry (E2)
need a type; a weight used elementwise (A_log, dt_bias, the conv kernel) must
be replicated; the embedding gather needs `out_sharding`; the MoE global
argsort cannot run on a batch-sharded axis (DeepSeek's dispatch); and
`solve_triangular`'s VJP is not explicit-clean, hence the custom-VJP inverse
in `gdn.py` (E1, used in EVERY mode), whose two typing points are below.

The wrappers around `maxtext.experimental.mpmd.jax.deepseek` functions (`dense`, `mlp`, `norm`,
`embed`, `routed_experts`, `attention`) are converted on the DeepSeek side;
here they only mark where Qwen depends on them.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from maxtext.experimental.mpmd.jax.deepseek import attention as ds_attention
from maxtext.experimental.mpmd.jax.deepseek import layers as ds_layers
from maxtext.experimental.mpmd.jax.deepseek import moe as ds_moe
from maxtext.experimental.mpmd.jax import sharding_mode as mode


# ------------------------------------------------------------------ weights
def use_weight(w, dtype, config):
  """A weight used by a matmul-like op (norm scales, the router kernel): cast
  to `dtype`, its storage sharding left to the consuming op's `out_spec`.
  """
  del config
  return jnp.asarray(w, dtype)


def small_weight(w, dtype, config):
  """A weight used ELEMENTWISE against activations: GDN `A_log`, `dt_bias`,
  and the depthwise conv kernel.

  `use_weight`, then replicated under explicit: a broadcasting multiply has
  no `out_sharding`, an FSDP-sharded `[Hv]` against a batch-sharded
  `[B, S, Hv]` would put one mesh axis on two dims, and a channel-sharded
  conv kernel is rejected outright.  A few KB per layer.  A type obligation
  (`mode.retype`): auto and none emit nothing beyond the cast.
  """
  return mode.retype(use_weight(w, dtype, config), P())


def dense(params, x, config, axis=(-1,)):
  """Every projection (`in_proj_*`, `out_proj`, q/k/v/out, shared-expert gate,
  head): `maxtext.experimental.mpmd.jax.deepseek.layers.dense_general` (weights cast to `dtype` right
  before the dot, MaxText's rule; then `dot_general(out_spec=act)`)."""
  return ds_layers.dense_general(params, x, config, axis)


def mlp(params, x, config):
  """The shared expert: `maxtext.experimental.mpmd.jax.deepseek.layers.mlp_block` (SwiGLU)."""
  return ds_layers.mlp_block(params, x, config)


def norm(params, x, config):
  """Plain RMSNorm, `maxtext.experimental.mpmd.jax.deepseek.layers.rms_norm` -- the normalised value
  rounded to `dtype`, then a `dtype` scale multiply: both norm kinds under
  `norm_numerics='bf16'` (the zero-centred ones once `layers.rms_norm` has
  folded the `+ 1` into the scale).  The scale realignment against a
  batch-sharded activation is DeepSeek's (the cast + `einsum(out_spec=act)`)."""
  return ds_layers.rms_norm(params, x, config)


def norm_fp32(params, x, config):
  """RMSNorm with the scale multiply in fp32, NOT rounded: `fp32(x) * rsqrt(
  mean(fp32(x)^2) + eps) * fp32(scale)`, as MaxText's RMSNorm computes it with a
  `scale_offset` (zero-centred) or with `dtype=float32` (the GDN gated norm's
  inner norm) since 2026-09-10 -- both norm kinds under `norm_numerics='fp32'`.
  The caller rounds (after the `silu(z)` gate, for the gated norm).

  DeepSeek's `rms_norm` conversion, in fp32: the scale through `use_weight`,
  the multiply an einsum typed like the activation (the FSDP-sharded scale
  would otherwise put the stage's mesh axis on the batch AND the features),
  at `matmul_precision` (`precision_or_none`: None, as before, at 'default').
  """
  x = jnp.asarray(x, jnp.float32)
  mean2 = jnp.mean(jax.lax.square(x), axis=-1, keepdims=True)
  y = x * jax.lax.rsqrt(mean2 + config.normalization_layer_epsilon)
  scale = use_weight(params["scale"], jnp.float32, config)
  return mode.einsum(
      "...k,k->...k", y, scale, out_spec=mode.act_spec(y.ndim), precision=ds_layers.precision_or_none(config)
  )


def dot_general(lhs, rhs, dimension_numbers, *, precision, preferred_element_type=None):
  """The router logits `x @ gate.kernel` (batch on lhs axis 0), typed like the
  activation: the kernel can be FSDP-sharded on its expert (output) axis."""
  (lc, _), (lb, _) = dimension_numbers
  out_ndim = lhs.ndim + rhs.ndim - 2 * len(lc) - len(lb)
  return mode.dot_general(
      lhs,
      rhs,
      dimension_numbers,
      out_spec=mode.act_spec(out_ndim),
      precision=precision,
      preferred_element_type=preferred_element_type,
  )


# --------------------------------------------------------------- activations
def embed(params, tokens, config):
  """The token embedding gather: `maxtext.experimental.mpmd.jax.deepseek.layers.embed_lookup`
  (`mode.take(table, ids, out_spec=act)`)."""
  return ds_layers.embed_lookup(params, tokens, config)


def zeros(shape, dtype, *, batch_dim=0):
  """A fresh `lax.scan` carry (the GDN state `[B, Hv, Dk, Dv]`), E2.

  `mode.zeros`, typed like the activation under explicit: a scan's carry must
  have one type on the way in and out, and a replicated init against a
  batch-sharded update would make them differ.
  """
  return mode.zeros(shape, dtype, mode.act_spec(len(shape), batch_dim))


def regroup(x, shape, *, batch_dim=0):
  """The GDN per-key-head -> per-value-head regroup `[B,S,Hk,Vpk*D] -> [B,S,Hv,D]`
  (and `[B,S,Hk,Vpk] -> [B,S,Hv]`), which splits one axis and merges it with
  another, E3: explicit rejects that reshape without an `out_sharding`, so it
  gets the activation's (`mode.out_sharding`; under auto/none the plain
  reshape)."""
  sh = mode.out_sharding(mode.act_spec(len(shape), batch_dim))
  if sh is None:
    return x.reshape(shape)
  return jnp.reshape(x, shape, out_sharding=sh)


def broadcast_like(x, like):
  """`jnp.broadcast_to(x, like.shape)`, typed like `like`: the identity of the
  GDN inverse's forward in the custom VJP (E1) -- the triangular solve's RHS,
  the block doubling's starting inverse.

  The type comes from `like`'s own aval (`mode.sharding_of`: typed exactly
  like `like`, which `mode.bwd_out_sharding`, a cotangent's type, is not for
  a reduced or unreduced value), the rule for custom-VJP code: a replicated
  RHS against a batch-sharded matrix breaks the batched solve's vmap under
  explicit.
  """
  sh = mode.sharding_of(like)
  if sh is None:
    return jnp.broadcast_to(x, like.shape)
  return jnp.broadcast_to(x, like.shape, out_sharding=sh)


def retype_like(x, like):
  """A custom-VJP cotangent, typed as its primal's cotangent (the GDN
  inverse's backward, E1): `mode.retype_like`, which emits nothing when the
  types agree and nothing at all under auto."""
  return mode.retype_like(x, like)


def conv_general_dilated(lhs, rhs, *, window_strides, padding, dimension_numbers, feature_group_count, precision):
  """The GDN depthwise causal conv (batch on axis 0 of `lhs` and the output).

  The kernel comes in through `small_weight`, i.e. replicated under explicit,
  which is all the conv's sharding rule needs; the result is batch-sharded
  like `lhs`.  No `out_sharding` is passed in any mode.
  """
  return jax.lax.conv_general_dilated(
      lhs,
      rhs,
      window_strides=window_strides,
      padding=padding,
      dimension_numbers=dimension_numbers,
      feature_group_count=feature_group_count,
      precision=precision,
  )


def segment_cumsum(mask, g):
  """The packed GDN's per-document cumulative decay, `einsum('bnhij,bnhj->bnhi', mask, g)` at HIGHEST: `mask
  [B, N, 1, C, C]` the fp32 lower-triangular same-document mask of each chunk, `g [B, N, H, C]` fp32 (MaxText's
  gdn_packing.prepare_jax_gdn_segment_masks, the same contraction).  Batch-major, typed like the activation under
  explicit."""
  return mode.einsum("bnhij,bnhj->bnhi", mask, g, out_spec=mode.act_spec(g.ndim), precision=jax.lax.Precision.HIGHEST)


def conv_taps(taps, w):
  """The packed causal conv's rows at a document edge, `einsum('bskc,kc->bsc', taps, w)` at HIGHEST with fp32
  accumulation: `taps [B, S, K, C]` the window of each such row with the other documents' tokens zeroed, `w [K, C]`
  the fp32 kernel (through `small_weight`: replicated under explicit), as MaxText's
  gdn_packing._compute_segmented_conv1d_core computes them.  Typed like the activation under explicit."""
  return mode.einsum(
      "bskc,kc->bsc",
      taps,
      w,
      out_spec=mode.act_spec(3),
      precision=jax.lax.Precision.HIGHEST,
      preferred_element_type=jnp.float32,
  )


# ---------------------------------------------------------------- dispatch
def routed_experts(params, x, weights, indices, config):
  """MoE dispatch: `maxtext.experimental.mpmd.jax.deepseek.moe.routed_experts`, the DeepSeek
  dispatcher, which picks the kernel by `config.moe_impl` (sort + grouped
  matmuls, or dense); under explicit its global argsort runs on replicated
  tokens, a correctness path."""
  return ds_moe.routed_experts(params, x, weights, indices, config)


def attention(query, key, value, segment_ids, config):
  """The attention kernel call: `maxtext.experimental.mpmd.jax.deepseek.attention.attention_op`.

  GQA: the dot-product and splash kernels map query head h to kv head
  h // (num_query_heads / num_kv_heads) themselves; the fp32 'reference'
  stand-in is MHA-only, so it gets the kv heads repeated in exactly that
  order (`jnp.repeat`), which is the definition of GQA.
  """
  if config.attention_kernel == "reference" and key.shape[2] != query.shape[2]:
    rep = query.shape[2] // key.shape[2]
    key = jnp.repeat(key, rep, axis=2)
    value = jnp.repeat(value, rep, axis=2)
  return ds_attention.attention_op(query, key, value, segment_ids, config)


def gdn_core(core, query, key, value, g, beta, config, segment_ids=None):
  """Where the chunked gated delta rule `core(query, key, value, g, beta)` runs
  (`core(query, key, value, g, beta, segment_ids=segment_ids)` with packed
  documents' segment ids `[B, S]`), per `config.gdn_core`:

    'inline'  in the surrounding program: today's program under auto, typed
              op by op under explicit (E1 + E2 + the regroups above).
    'manual'  per device, in a `shard_map` over the stage's batch axes
              (`mode.manual`, scope 'gdn_core'), in BOTH modes: MaxText's
              layout (its GDN kernel runs in a shard_map), and the same
              per-device program under auto and explicit, so the core's time
              is mode-invariant.  The cumsum, the solve and the state scan
              are batch-local by construction, where auto on its own may
              pick a head- or feature-sharded layout and move between them
              with all-to-alls.

  Every operand is batch-major `[B, S, ...]`, the result too.
  """
  args = (query, key, value, g, beta)
  if segment_ids is not None:
    args, plain = args + (segment_ids,), core
    core = lambda q, k, v, g, beta, seg: plain(q, k, v, g, beta, segment_ids=seg)  # noqa: E731
  if config.gdn_core == "inline":
    return core(*args)
  b = mode.act_spec(1)  # the batch dim of every operand
  return mode.manual(core, in_specs=(b,) * len(args), out_specs=b, name="gdn_core")(*args)
