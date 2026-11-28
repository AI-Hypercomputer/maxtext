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

"""DeepSeek-V3 MoE block: sigmoid router + routed experts + one shared expert.

Mirrors MaxText's `layers/moe.py` for `decoder_block: deepseek` and
`model_name: deepseek3*`.  The two facts that are easy to get wrong:

  * top-k *selects* on the post-bias scores but *weighs* with the pre-bias
    scores (`moe.deepseek_routing`), and
  * the renormalisation `w / (sum(w) + 1e-20) * routed_scaling_factor` runs
    entirely in bf16 (`moe.deepseek_scale_weights`) -- there is no softmax.

The routed experts are dispatched by one argsort over every token of the
stage (`routed_experts`; `src/maxtext/experimental/mpmd/jax/sharding_mode.py` for the modes).
Under auto XLA implements it by all-gathering the routing to every device and
running a weight-stationary MoE.  Under explicit the tokens are made
replicated first, since a sort over a sharded axis has no type: a correctness
path.
"""

import functools

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from maxtext.experimental.mpmd.jax.deepseek import layers
from maxtext.experimental.mpmd.jax.deepseek.config import DeepSeekConfig
from maxtext.experimental.mpmd.jax import sharding_mode as mode


def route(params, inputs, config: DeepSeekConfig):
  """`GateLogit.__call__` + `get_topk` for DeepSeek-V3.

  Returns `(weights, indices)` with shapes `[b, s, k]`.
  """
  gate_dtype = jnp.float32 if config.float32_gate_logits else config.dtype
  x = jnp.asarray(inputs, gate_dtype)
  kernel = jnp.asarray(params["kernel"], gate_dtype)
  logits = _gate_logits(x, kernel, config)
  pre_bias_logits = layers.activation_fn(config.routed_score_func)(logits)

  gate_logits = pre_bias_logits
  if config.routed_bias:
    gate_logits = gate_logits + jnp.asarray(params["bias"], gate_dtype)
  indices = select_experts(gate_logits, config)
  weights = jnp.take_along_axis(pre_bias_logits, indices, axis=-1)

  # deepseek_scale_weights -- bf16 throughout, epsilon outside the sum.
  if config.routed_score_func in ("sigmoid", "sqrtsoftplus"):
    weights = weights / (weights.sum(-1, keepdims=True) + 1e-20)
  weights = weights * config.routed_scaling_factor
  return weights, indices


def _top_2_in_group_sum(scores_grouped):
  """`moe._top_2_in_group_sum`: the two largest scores of each group, summed in
  fp32 -- two masked maxima, not a top-k sort (the lowest index of a tie is
  the one masked, so a tied pair sums to twice the maximum, as top_k's)."""
  max_1 = jnp.max(scores_grouped, axis=-1)
  argmax_1 = jnp.argmax(scores_grouped, axis=-1)[..., None]
  lane = jax.lax.broadcasted_iota(argmax_1.dtype, scores_grouped.shape, scores_grouped.ndim - 1)
  max_2 = jnp.max(jnp.where(lane == argmax_1, -jnp.inf, scores_grouped), axis=-1)
  return max_1.astype(jnp.float32) + max_2.astype(jnp.float32)


def expert_group_mask(gate_logits, config: DeepSeekConfig):
  """`RoutedMoE.expert_group_mask`: 1 for the experts of each token's
  `topk_routing_group` best groups (a group scored by its two best post-bias
  scores), 0 elsewhere; `[..., num_experts]` float32."""
  g = config.n_routing_groups
  grouped = jnp.reshape(gate_logits, gate_logits.shape[:-1] + (g, -1))
  _, group_idx = jax.lax.top_k(_top_2_in_group_sum(grouped), k=config.topk_routing_group)
  group_mask = jnp.sum(jax.nn.one_hot(group_idx, num_classes=g, dtype=jnp.float32), axis=-2)
  mask = jnp.broadcast_to(group_mask[..., None], group_mask.shape + (config.num_experts // g,))
  return jnp.reshape(mask, mask.shape[:-2] + (config.num_experts,))


def select_experts(gate_logits, config: DeepSeekConfig):
  """`RoutedMoE.deepseek_routing`'s choice: the top-k experts of the post-bias
  `gate_logits`, among each token's best groups when `n_routing_groups` is
  set (> 0)."""
  if config.n_routing_groups > 0:
    gate_logits = jnp.where(expert_group_mask(gate_logits, config) > 0, gate_logits, -jnp.inf)
  _, indices = jax.lax.top_k(gate_logits, config.num_experts_per_tok)
  return indices


def _gate_logits(x, kernel, config: DeepSeekConfig):
  """`x @ gate.kernel`, typed like the activation: the kernel is FSDP-sharded
  on whichever of `[emb, num_experts]` is bigger, an output dim when it is
  the expert count."""
  return mode.dot_general(
      x, kernel, (((x.ndim - 1,), (0,)), ((), ())), out_spec=mode.act_spec(x.ndim), precision=layers.precision_of(config)
  )


def load_balance_loss(indices, probs, config: DeepSeekConfig):
  """`moe.RoutedMoE.load_balance_loss`."""
  expert_mask = jax.nn.one_hot(indices, num_classes=config.num_experts, dtype=jnp.int32)
  summed = jnp.sum(expert_mask, axis=2)
  density = jnp.mean(summed, axis=1) / config.num_experts_per_tok
  density_prob = jnp.mean(probs, axis=1)
  return jnp.mean(density * density_prob) * (config.num_experts**2) * config.load_balance_loss_weight


def bias_update(indices, config: DeepSeekConfig):
  """`moe.calculate_load_balance_updates` -- sign rule, no lr."""
  flat = indices.ravel()
  counts = jnp.sum(jax.nn.one_hot(flat, config.num_experts, dtype=jnp.int32), axis=0)
  average_load = jnp.sum(counts) / config.num_experts
  return jnp.sign(average_load - counts) * config.routed_bias_update_rate


def _experts_dense(params, inputs, weights, indices, config: DeepSeekConfig):
  """`moe.RoutedMoE.dense_matmul` with `capacity_factor <= 0`.

  Every expert on every token, so nothing here crosses the batch shards and
  each op is simply typed like the activation: the expert kernels can be
  FSDP-sharded on an output dim, and the routing-weight scatter writes at
  batch-sharded indices.
  """
  b, s, _ = inputs.shape
  w0 = jnp.asarray(params["wi_0"], config.dtype)
  w1 = jnp.asarray(params["wi_1"], config.dtype)
  wo = jnp.asarray(params["wo"], config.dtype)
  precision = layers.precision_of(config)
  act3, act4 = mode.act_spec(3), mode.act_spec(4)

  layer_w0 = mode.einsum("BSM,EMH->BSEH", inputs, w0, out_spec=act4, precision=precision)
  layer_w1 = mode.einsum("BSM,EMH->BSEH", inputs, w1, out_spec=act4, precision=precision)
  hidden = (jax.nn.silu(layer_w0) * layer_w1).astype(config.dtype)
  intermediate = mode.einsum("BSEH,EHM->BSEM", hidden, wo, out_spec=act4, precision=precision)

  # RoutedMoE.reshape_and_update_weights
  dense_weights = jnp.zeros((b, s, config.num_experts), dtype=config.dtype)
  batch_idx = jnp.arange(b)[:, None, None]
  seq_idx = jnp.arange(s)[None, :, None]
  dense_weights = mode.set_at(dense_weights, (batch_idx, seq_idx, indices), weights, out_spec=act3)

  if config.float32_weight_sum:
    intermediate = intermediate.astype(jnp.float32)
    dense_weights = dense_weights.astype(jnp.float32)
  out = mode.einsum("BSEM,BSE->BSM", intermediate, dense_weights, out_spec=act3, precision=precision)
  return out.astype(config.dtype)


@jax.custom_vjp
def _sort_activations(inputs, sort_indices):
  """`moe._sort_activations_custom`.

  The custom VJP matters numerically as well as for speed: the backward pass is
  the inverse *permutation* rather than a scatter-add, so no summation happens
  where the compiler-generated transpose would have summed.

  Under explicit sharding with rows sharded (the dispatches never hand it
  such rows, but a custom VJP must hold for any typed primal) the gather is
  typed like `inputs`, so the cotangent comes back typed like `inputs` too,
  and the backward undoes the permutation with a scatter rather than with an
  argsort, which cannot sort over a sharded axis.  Both are pure data
  movement, so the values are bit-for-bit those of the gather by the inverse.
  """
  sharding = mode.sharding_of(inputs)
  if sharding is None:
    return inputs[sort_indices, ...]
  return inputs.at[sort_indices, ...].get(out_sharding=sharding)


def _sort_activations_fwd(inputs, sort_indices):
  return _sort_activations(inputs, sort_indices), sort_indices


def _sort_activations_bwd(sort_indices, grads):
  # `inputs`' cotangent is typed like `grads`, the cotangent of a result
  # typed like `inputs` (above).
  sharding = mode.sharding_of(grads)
  if sharding is None:
    return _sort_activations(grads, jnp.argsort(sort_indices)), None
  # A permutation writes every row exactly once, so `set` sums nothing.
  return jnp.zeros_like(grads).at[sort_indices, ...].set(grads, out_sharding=sharding), None


_sort_activations.defvjp(_sort_activations_fwd, _sort_activations_bwd)


# `lhs[m, k] * dout[m, n] -> out[g, k, n]`, contracting over the *ragged* axis:
# `megablox`'s `tgmm`, and the only grouped-matmul mode XLA exposes that does
# not cost `num_experts` times the work.
_RAGGED_CONTRACTING = jax.lax.RaggedDotDimensionNumbers(
    dot_dimension_numbers=(((0,), (0,)), ((), ())),
    lhs_ragged_dimensions=[0],
    rhs_group_dimensions=[],
)


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5))
def _grouped_matmul(lhs, rhs, group_sizes, fp32_accum, exact_drhs, precision=None):
  """`kernels/megablox/ops.py:gmm`, expressed with `jax.lax.ragged_dot`.

  megablox is a `jax.custom_vjp` around three Pallas kernels -- `gmm` forward,
  `gmm` again for `dlhs`, `tgmm` for `drhs` -- so its three matmuls round
  independently and each has to be matched independently.  What reproduces
  MaxText, measured leaf by leaf against MaxText's `RoutedMoE` (one MoE
  layer, both fed the same activations and the same output cotangent):

    * forward: accumulate in fp32 and round once.  The kernel accumulates a
      tile in fp32 and stores through `preferred_element_type`, so with a
      single contraction tile that is exactly an fp32 `ragged_dot` rounded to
      bf16.  Asking XLA for a bf16 `ragged_dot` instead is off by up to a bf16
      ulp per layer, which the 58-layer stack turns into a visible loss delta.
    * `dlhs`: XLA's plain bf16 `ragged_dot` transpose.  Deriving it from the
      fp32 forward costs ~1.5 orders of magnitude of accuracy against MaxText
      (2/31 leaves bit-exact end to end rather than 27/31), so the asymmetry
      with the forward is deliberate.
    * `drhs`: fp32 `ragged_dot_general` over the ragged axis leaves only `wo`
      inexact, and by 2e-5.  A masked bf16 `einsum` is bit-exact on every leaf
      but contracts `num_experts` slices instead of one, so it is opt-in via
      `exact_drhs` (`DeepSeekConfig.gmm_exact_drhs`) -- a parity check turns
      it on, training leaves it off.

  `fp32_accum=False` recovers MaxText's `jax_ragged_dot_gmm` fallback
  (`megablox=false use_tokamax_gmm=false`), which is bit-exact in both
  directions without any of the above.

  `precision` (`layers.precision_or_none`) goes to all three matmuls, the
  hand-written transposes included: on TPU a DEFAULT-precision matmul of
  fp32 operands is a bf16 pass, which an fp32 parity run at 'highest' would
  otherwise inherit from these three alone.

  Under explicit sharding (only the global dispatch reaches it, with the rows
  replicated) `ragged_dot` has no sharding rule at all, so the result is
  typed like `lhs`; and its transpose would call `ragged_dot_general` without
  one, which is why the backward spells the two transposes out.  The ragged
  (row) axis of `lhs` must not be sharded: XLA's ragged-dot lowering on CPU
  rejects it.
  """
  accum = jnp.float32 if fp32_accum else lhs.dtype
  out = jax.lax.ragged_dot(
      lhs, rhs, group_sizes, precision=precision, preferred_element_type=accum, out_sharding=mode.sharding_of(lhs)
  )
  return out.astype(lhs.dtype)


def _grouped_matmul_fwd(lhs, rhs, group_sizes, fp32_accum, exact_drhs, precision):
  out = _grouped_matmul(lhs, rhs, group_sizes, fp32_accum, exact_drhs, precision)
  return out, (lhs, rhs, group_sizes)


def _grouped_matmul_bwd(fp32_accum, exact_drhs, precision, res, dout):
  lhs, rhs, group_sizes = res
  # Under explicit, `jax.vjp` of `ragged_dot` fails (its transpose has no
  # out_sharding to give the rule-less op), so both transposes are written
  # out, typed like the primals: the same products, bit for bit.
  lhs_sh, rhs_sh = mode.bwd_out_sharding(lhs), mode.bwd_out_sharding(rhs)
  if lhs_sh is None:
    dlhs = jax.vjp(
        lambda l: jax.lax.ragged_dot(l, rhs, group_sizes, precision=precision, preferred_element_type=lhs.dtype),
        lhs,
    )[1](dout)[0]
  else:
    dlhs = jax.lax.ragged_dot(
        dout,
        jnp.swapaxes(rhs, 1, 2),
        group_sizes,
        precision=precision,
        preferred_element_type=lhs.dtype,
        out_sharding=lhs_sh,
    )

  if not fp32_accum and rhs_sh is None:
    drhs = jax.vjp(
        lambda r: jax.lax.ragged_dot(lhs, r, group_sizes, precision=precision, preferred_element_type=lhs.dtype),
        rhs,
    )[1](dout)[0]
  elif not fp32_accum:
    drhs = jax.lax.ragged_dot_general(
        lhs,
        dout,
        group_sizes,
        _RAGGED_CONTRACTING,
        precision=precision,
        preferred_element_type=lhs.dtype,
        out_sharding=rhs_sh,
    )
  elif exact_drhs:
    bounds = jnp.cumsum(group_sizes)
    starts = jnp.concatenate([jnp.zeros((1,), bounds.dtype), bounds[:-1]])
    rows = jnp.arange(lhs.shape[0])[:, None]
    mask = ((rows >= starts) & (rows < bounds)).astype(lhs.dtype)
    drhs = jnp.einsum(
        "mk,mn,mg->gkn", lhs, dout, mask, precision=precision, preferred_element_type=lhs.dtype, out_sharding=rhs_sh
    )
  else:
    drhs = jax.lax.ragged_dot_general(
        lhs,
        dout,
        group_sizes,
        _RAGGED_CONTRACTING,
        precision=precision,
        preferred_element_type=jnp.float32,
        out_sharding=rhs_sh,
    )
  return dlhs.astype(lhs.dtype), drhs.astype(rhs.dtype), None


_grouped_matmul.defvjp(_grouped_matmul_fwd, _grouped_matmul_bwd)


def _experts_ragged(params, inputs, weights, indices, config: DeepSeekConfig):
  """`moe.RoutedMoE.sparse_matmul` with EP=TP=1.

  Sort tokens by expert, run three grouped matmuls (`_grouped_matmul`), unsort,
  then combine in fp32.

  Numerics note: MaxText casts the expert kernels to bf16 *inside* `gmm`, so
  their cotangents are bf16 too and every grouped-matmul implementation rounds
  them to the same 8-bit mantissa.  The forward and every other cotangent match
  MaxText bit-for-bit; `wi_0`, `wi_1` and `wo` land on the adjacent bf16 value
  for a fraction of their entries because the Pallas `tgmm` sums each group in
  a different order -- worst relative 1 ulp (~2.5e-3) in the leaf-by-leaf
  comparison against MaxText's `RoutedMoE` (see `_grouped_matmul`), and the
  floor for anything that is not that kernel.
  """
  b, s, d = inputs.shape
  tokens = b * s
  k = config.num_experts_per_tok
  w0 = jnp.asarray(params["wi_0"], config.dtype)
  w1 = jnp.asarray(params["wi_1"], config.dtype)
  wo = jnp.asarray(params["wo"], config.dtype)

  # The argsort below is over EVERY token of the stage.  Under explicit a sort
  # over a batch-sharded axis has no type, so the tokens are made replicated
  # first and the result is handed back batch-sharded: the global dispatch,
  # spelled out.  Under auto XLA makes the same choice by itself (it gathers
  # the routing to every device), so this is a type obligation and emits
  # nothing there; inside a caller's shard_map it is not explicit either.
  out_spec = mode.act_spec(inputs.ndim)
  inputs, weights, indices = (mode.retype(t, P()) for t in (inputs, weights, indices))

  inputs_2d = inputs.reshape(tokens, d)
  flat_experts = indices.reshape(-1)
  sort_indices = jnp.argsort(flat_experts)
  if config.moe_use_direct_token_gather:
    # `_route_activations` -- gathers straight into expert order, so the
    # backward pass sums the k copies of each token.
    sorted_inputs = inputs_2d[sort_indices // k].astype(config.dtype)
  else:
    replicated_inputs_2d = jnp.repeat(inputs_2d, k, axis=0)
    sorted_inputs = _sort_activations(replicated_inputs_2d, sort_indices).astype(config.dtype)
  group_sizes = mode.bincount(flat_experts, length=config.num_experts)

  def gmm(lhs, rhs):
    return _grouped_matmul(
        lhs, rhs, group_sizes, config.gmm_fp32_accum, config.gmm_exact_drhs, layers.precision_or_none(config)
    )

  hidden = (jax.nn.silu(gmm(sorted_inputs, w0)) * gmm(sorted_inputs, w1)).astype(config.dtype)
  expert_out = gmm(hidden, wo)

  unsorted = _sort_activations(expert_out, jnp.argsort(sort_indices))
  reshaped_intermediate = unsorted.reshape(tokens, k, -1)
  reshaped_weights = weights.reshape(tokens, k)
  if config.float32_weight_sum:
    reshaped_intermediate = reshaped_intermediate.astype(jnp.float32)
    reshaped_weights = reshaped_weights.astype(jnp.float32)
  out = jnp.einsum("BKE,BK->BE", reshaped_intermediate, reshaped_weights, precision=layers.precision_of(config))
  return mode.retype(out.reshape(b, s, -1).astype(config.dtype), out_spec)


def routed_experts(params, inputs, weights, indices, config: DeepSeekConfig):
  """The routed experts of one MoE layer (see the module docstring).

  `params` is `MoeBlock_0`; `inputs` is `[b, s, emb]`, `weights` and
  `indices` the router's `[b, s, k]`.  `config.moe_impl` picks the kernel:
  `_experts_ragged` (sort + grouped matmuls) or `_experts_dense`.
  """
  experts = _experts_dense if config.moe_impl == "dense" else _experts_ragged
  return experts(params, inputs, weights, indices, config)


def routed_and_shared_moe(params, inputs, config: DeepSeekConfig):
  """`moe.RoutedAndSharedMoE.__call__`.

  Returns `(output, lb_loss, bias_updates)`; the latter two are `None` when the
  corresponding weights are zero, matching MaxText.
  """
  inputs = inputs.astype(config.dtype)
  routed = params["MoeBlock_0"]
  weights, indices = route(routed["gate"], inputs, config)

  lb_loss = None
  if config.load_balance_loss_weight > 0.0:
    gate_dtype = jnp.float32 if config.float32_gate_logits else config.dtype
    kernel = jnp.asarray(routed["gate"]["kernel"], gate_dtype)
    logits = _gate_logits(jnp.asarray(inputs, gate_dtype), kernel, config)
    pre_bias_logits = layers.activation_fn(config.routed_score_func)(logits)
    probs = jax.nn.softmax(pre_bias_logits.astype(jnp.float32), axis=-1).astype(config.dtype)
    lb_loss = load_balance_loss(indices, probs, config)

  updates = None
  if config.routed_bias and config.routed_bias_update_rate > 0.0:
    updates = bias_update(indices, config)

  routed_out = routed_experts(routed, inputs, weights, indices, config)

  shared_out = layers.mlp_block(params["shared_experts"], inputs, config)
  return routed_out + shared_out, lb_loss, updates
