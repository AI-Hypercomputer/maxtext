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

"""The DeepSeek-V3 decoder stack, as plain functions over parameter pytrees.

Mirrors MaxText's `models.models.Transformer` + `layers.nnx_decoders.NNXDecoder` +
`models.deepseek` for `MODEL_MODE_TRAIN`.

Parameter layout differs from MaxText in exactly one way: scanned layers are
stacked at **axis 0** (natural for `jax.lax.scan`), whereas MaxText moves
`nnx.Param` leaves to `param_scan_axis = 1`.  `convert.py` bridges the two.

The router bias (`gate.bias`) lives in a separate `aux` pytree rather than in
`params`, because MaxText stores it as a `MoEBiasVar` -- not an `nnx.Param` --
and therefore never differentiates it.

The model code imports jax and numpy (and, for the 'tokamax_splash' kernel,
tokamax) and never MaxText; only the reference it is held to imports MaxText
(`maxtext_reference.py`, which deepseek_maxtext_parity_test drives), which is
what lets MaxText run next to it as a reference.

Nothing in this file shards: the ops that need a sharding under explicit
(embedding and RoPE gathers, FSDP matmuls, norm scales, the MoE dispatch)
live in `layers` / `rope` / `moe` and go through
`maxtext.experimental.mpmd.jax.sharding_mode`, which reads the mode off the mesh in
context.  The same `forward` runs on one device, under auto and under
explicit.
"""

import jax
import jax.numpy as jnp

from maxtext.experimental.mpmd.jax.deepseek import attention
from maxtext.experimental.mpmd.jax.deepseek import layers
from maxtext.experimental.mpmd.jax.deepseek import moe
from maxtext.experimental.mpmd.jax.deepseek import rope
from maxtext.experimental.mpmd.jax.deepseek.config import DeepSeekConfig


# The remat policies by MaxText name
# (`nnx_decoders.NNXDecoder.get_remat_policy`), as data: the
# `checkpoint_name`s each one saves HERE (`save_only_these_names`).  Two
# differ from MaxText's: its 'minimal' also saves the MLP / MoE outputs
# (`mlpwi_0`, `mlpwi_1`, `mlpwi`, `mlpwo`), which neither port emits, and its
# 'save_dot_except_mlp' saves no `context` (that is its
# 'save_dot_with_context_except_mlp').  (MaxText's lists also hold `kv_proj`,
# which no port here emits.)  'none' and 'full' save nothing, i.e. recompute
# everything (`NNXDecoder.get_remat_policy`).  The one table: `remat_policy_for` builds
# the policy from it, for this port and for the Qwen3.5 port alike.
REMAT_POLICIES = {
    "none": (),
    "full": (),
    "save_out_proj": ("out_proj",),
    "minimal": ("query_proj", "value_proj", "key_proj", "qkv_proj", "out_proj"),
    "save_dot_except_mlp": ("query_proj", "value_proj", "key_proj", "qkv_proj", "context", "out_proj"),
    "save_qkv_proj": ("query_proj", "value_proj", "key_proj", "qkv_proj"),
}


def remat_policy_for(config: DeepSeekConfig):
  """`nnx_decoders.NNXDecoder.get_remat_policy`."""
  name = config.remat_policy or "none"
  if name not in REMAT_POLICIES:
    raise ValueError(f"unsupported remat_policy: {name}")
  names = REMAT_POLICIES[name]
  return jax.checkpoint_policies.save_only_these_names(*names) if names else None


def _merge(base, extra):
  """Shallow-recursive dict merge; `extra` wins.  Used to splice `aux` in."""
  if not extra:
    return base
  out = dict(base)
  for key, value in extra.items():
    if isinstance(out.get(key), dict) and isinstance(value, dict):
      out[key] = _merge(out[key], value)
    else:
      out[key] = value
  return out


def decoder_layer(params, x, positions, segment_ids, freqs_cis, config, is_moe):
  """`DeepSeekGenericLayer.self_attention_with_norm_op` + the MLP/MoE branch.

  `deepseek.DeepSeekGenericLayer` (shared body), `DeepSeekDenseLayer.__call__`
  (dense), `DeepSeekMoELayer.__call__` (MoE).
  The residual stream stays bf16 end to end; nothing here promotes to fp32.
  """
  x = jax.ad_checkpoint.checkpoint_name(x, "decoder_layer_input")

  lnx = layers.rms_norm(params["pre_self_attention_layer_norm"], x, config)
  attention_lnx = attention.mla(params["self_attention"], lnx, positions, segment_ids, freqs_cis, config)
  intermediate_inputs = x + attention_lnx  # residual #1: adds `x`, not `lnx`.

  hidden_states = layers.rms_norm(params["post_self_attention_layer_norm"], intermediate_inputs, config)
  if is_moe:
    mlp_lnx, lb_loss, updates = moe.routed_and_shared_moe(params["DeepSeekMoeBlock_0"], hidden_states, config)
    # Neither is plumbed out of the layer stack: there is no aux-loss term in
    # `train.compute_loss` and no router-bias update after the step.  Dropping
    # them would train a different model from MaxText without a word, so a
    # config that asks for either is refused until they are.
    if lb_loss is not None or updates is not None:
      raise NotImplementedError(
          "load_balance_loss_weight > 0 and routed_bias_update_rate > 0 are not "
          "supported: the aux loss and the bias update are computed per layer but "
          "never reach the loss or the optimizer"
      )
  else:
    mlp_lnx = layers.mlp_block(params["mlp"], hidden_states, config)

  return mlp_lnx + intermediate_inputs  # residual #2: adds the post-attention residual.


def _apply_stack(stack_params, stack_aux, x, positions, segment_ids, freqs_cis, config, is_moe):
  """`nnx_decoders.NNXDecoder._apply_layers_sequentially`.

  `jax.lax.scan` + `jax.checkpoint` are forward-identity; this is a plain
  sequential loop over the layers.
  """

  def body(carry, layer):
    layer_params, layer_aux = layer
    y = decoder_layer(
        _merge(layer_params, layer_aux),
        carry,
        positions,
        segment_ids,
        freqs_cis,
        config,
        is_moe,
    )
    return y, None

  if config.scan_layers:
    fn = jax.checkpoint(body, policy=remat_policy_for(config), prevent_cse=False)
    out, _ = jax.lax.scan(fn, x, (stack_params, stack_aux), unroll=1)
    return out

  num_layers = jax.tree.leaves(stack_params)[0].shape[0]
  for i in range(num_layers):
    sliced = jax.tree.map(lambda leaf, i=i: leaf[i], (stack_params, stack_aux))
    x, _ = body(x, sliced)
  return x


def forward(params, aux, tokens, positions, segment_ids, config: DeepSeekConfig, freqs_cis=None):
  """`Transformer.__call__` for training.  Returns `[B, L, vocab]` logits.

  Args:
    params: differentiable parameter pytree, structured as `param_avals`
      (`init.init_params` draws one, `convert.py` reads one off MaxText).
    aux: non-differentiable state -- the MoE router biases.
    tokens: `[B, L]` int32 input token ids.
    positions: `[B, L]` int32 token positions.
    segment_ids: `[B, L]` int32 packing segment ids (all ones when not packing).
    config: model config.
    freqs_cis: optional precomputed YaRN table; built here if omitted.

  MaxText rebuilds the (163840, 32) YaRN table inside every attention call; we
  build the `max_target_length` prefix once, which is numerically identical and
  avoids 42 MB of redundant work per layer.

  That one is worth spelling out, because it is the single largest term in the
  step-time gap.  XLA does not fold the table build into the position gather:
  the optimized HLO materialises both `f32[163840, 32]` halves of
  `exp(1j * outer(arange(163840), freqs))` and only then gathers the 2048 rows
  the batch actually uses, which takes MLA's HBM traffic from 109 MB to 275 MB.
  Slicing the table afterwards does not help -- it has to be built short.  On
  one v6e chip at MaxText's deepseek3-tiny shapes (seq 2048, 61 layers,
  remat 'full', batch 1) that is 0.39 ms per layer, bit-identical output
  either way: 23 ms per forward pass, doubled to 47 ms per step by full
  remat, out of a 74 ms total step-time gap to MaxText.
  """
  if freqs_cis is None:
    freqs_cis = rope.freqs_cis_for(config, config.max_target_length)

  y = layers.embed_lookup(params["token_embedder"], tokens, config)
  y = y.astype(config.dtype)  # NNXDecoder._apply_embedding

  decoder = params["decoder"]
  aux_decoder = (aux or {}).get("decoder", {})

  # `first_num_dense_layers=0` is a legal MaxText config (a fully-MoE model);
  # the converter then emits no `dense_layers` key at all.
  if "dense_layers" in decoder:
    y = _apply_stack(
        decoder["dense_layers"],
        aux_decoder.get("dense_layers", {}),
        y,
        positions,
        segment_ids,
        freqs_cis,
        config,
        is_moe=False,
    )
  y = _apply_stack(
      decoder["moe_layers"],
      aux_decoder.get("moe_layers", {}),
      y,
      positions,
      segment_ids,
      freqs_cis,
      config,
      is_moe=True,
  )

  y = layers.rms_norm(decoder["decoder_norm"], y, config)
  logits = layers.dense_general(decoder["logits_dense"], y, config)
  if config.cast_logits_to_fp32:
    logits = logits.astype(jnp.float32)  # NNXDecoder.apply_output_head
  return logits


# ------------------------------------------------------------- param avals
# The two layer stacks under `decoder`, each with the layer axis at 0.
DENSE = "dense_layers"
MOE = "moe_layers"


def _sds(shape, dtype):
  """`jax.ShapeDtypeStruct` with plain-int dims."""
  return jax.ShapeDtypeStruct(tuple(int(d) for d in shape), dtype)


def _layer_avals(config: DeepSeekConfig, n: int, is_moe: bool):
  """One stacked layer group, `[n, ...]`, matching `convert.from_nnx_state`."""
  wd, emb, heads = config.weight_dtype, config.emb_dim, config.num_query_heads
  tree = {
      "pre_self_attention_layer_norm": {"scale": _sds((n, emb), wd)},
      "post_self_attention_layer_norm": {"scale": _sds((n, emb), wd)},
      "self_attention": {
          "wkv_a": {"kernel": _sds((n, emb, config.kv_lora_rank + config.qk_rope_head_dim), wd)},
          "kv_norm": {"scale": _sds((n, config.kv_lora_rank), wd)},
          "wkv_b": {"kernel": _sds((n, config.kv_lora_rank, heads, config.qk_nope_head_dim + config.v_head_dim), wd)},
          "out": {"kernel": _sds((n, heads, config.v_head_dim, emb), wd)},
      },
  }
  att = tree["self_attention"]
  if config.q_lora_rank == 0:  # a plain projection, see `attention.mla`
    att["query"] = {"kernel": _sds((n, emb, heads, config.qk_head_dim), wd)}
  else:
    att["wq_a"] = {"kernel": _sds((n, emb, config.q_lora_rank), wd)}
    att["q_norm"] = {"scale": _sds((n, config.q_lora_rank), wd)}
    att["wq_b"] = {"kernel": _sds((n, config.q_lora_rank, heads, config.qk_head_dim), wd)}
  if not is_moe:
    tree["mlp"] = {
        "wi_0": {"kernel": _sds((n, emb, config.mlp_dim), wd)},
        "wi_1": {"kernel": _sds((n, emb, config.mlp_dim), wd)},
        "wo": {"kernel": _sds((n, config.mlp_dim, emb), wd)},
    }
    return tree
  n_exp, fm = config.num_experts, config.moe_mlp_dim
  shared = config.shared_experts * config.shared_expert_mlp_dim
  tree["DeepSeekMoeBlock_0"] = {
      "MoeBlock_0": {
          "gate": {"kernel": _sds((n, emb, n_exp), wd)},
          "wi_0": _sds((n, n_exp, emb, fm), wd),
          "wi_1": _sds((n, n_exp, emb, fm), wd),
          "wo": _sds((n, n_exp, fm, emb), wd),
      },
      "shared_experts": {
          "wi_0": {"kernel": _sds((n, emb, shared), wd)},
          "wi_1": {"kernel": _sds((n, emb, shared), wd)},
          "wo": {"kernel": _sds((n, shared, emb), wd)},
      },
  }
  return tree


def param_avals(config: DeepSeekConfig):
  """`jax.ShapeDtypeStruct` tree for the FULL model parameters (`forward`'s
  `params`).

  Never materialized; this is the thing that lets a 671B model be described
  on a host that could not hold one copy of it.
  """
  dec = {
      "decoder_norm": {"scale": _sds((config.emb_dim,), config.weight_dtype)},
      "logits_dense": {"kernel": _sds((config.emb_dim, config.vocab_size), config.weight_dtype)},
      MOE: _layer_avals(config, config.num_moe_layers, True),
  }
  if config.first_num_dense_layers:
    dec[DENSE] = _layer_avals(config, config.first_num_dense_layers, False)
  return {
      "token_embedder": {"embedding": _sds((config.vocab_size, config.emb_dim), config.weight_dtype)},
      "decoder": dec,
  }


def aux_avals(config: DeepSeekConfig):
  """Router-bias tree (`forward`'s `aux`): `[n_moe_layers, num_experts]`,
  layer axis at 0."""
  if not config.routed_bias:
    return {}
  return {
      "decoder": {
          MOE: {
              "DeepSeekMoeBlock_0": {
                  "MoeBlock_0": {"gate": {"bias": _sds((config.num_moe_layers, config.num_experts), config.weight_dtype)}}
              }
          }
      }
  }
