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

"""The Qwen3.5 text decoder, as plain functions over parameter pytrees.

Mirrors MaxText's `decoder_block: qwen3_5` training forward pass
(models/qwen3_5.py: `Qwen3_5ScannableBlock` + `Qwen3_5DecoderLayer`, scanned by
layers/nnx_decoders.py):

    y = embedding.astype(bf16)[tokens]
    scan over NB = L/4 blocks, each = 4 decoder layers [GDN, GDN, GDN, gated attn]:
        h = zc_rms_norm(x);  x = x + mixer(h)                      (pre-norm)
        h = zc_rms_norm(x);  x = x + sparse_moe(h)                 (both residuals bf16)
    logits = zc_rms_norm(y) @ logits_dense   (untied), cast to fp32

Parameter layout = MaxText's NNX tree before its nested Qwen3 scan (2026-10-21),
with the scanned block axis moved from `param_scan_axis = 1` to axis 0, as the
DeepSeek port does.  `convert.from_nnx_state` maps that tree, and the
nested-scan tree of later MaxText (this one: `local_layers`
[.., NB, cycle - 1, ..] + `global_layer`), onto it:

    token_embedder/embedding                    [V, E]
    decoder/decoder_norm/scale                  [E]          zero-centred, init 0
    decoder/logits_dense/kernel                 [E, V]
    decoder/layers/layer_{0,1,2}/...            [NB, ...]    Gated DeltaNet layers
    decoder/layers/layer_3/...                  [NB, ...]    gated full attention

Remat: on the scanned path one `jax.checkpoint` per BLOCK
(`remat_granularity='block'`, MaxText's placement before its nested scan, policy
`remat_policy`), one per LAYER with an un-checkpointed scan body (`'layer'`,
policy `layer_remat_policy`: MaxText's placement since the nested scan, a
checkpoint per sub-layer inside its nested scan and none around the block), or
both, nested (`'block+layer'`); MaxText's policy names, though 'minimal' and
'save_dot_except_mlp' save other sets than MaxText's
(`maxtext.experimental.mpmd.jax.deepseek.model.REMAT_POLICIES`).  As in the DeepSeek port,
`scan_layers=False` unrolls with no checkpoint at all.

This package imports jax/numpy (and `maxtext.experimental.mpmd.jax.deepseek`'s shared kernels) only,
never MaxText.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from maxtext.experimental.mpmd.jax.deepseek import model as ds_model
from maxtext.experimental.mpmd.jax.qwen35 import _ops
from maxtext.experimental.mpmd.jax.qwen35 import attention
from maxtext.experimental.mpmd.jax.qwen35 import gdn
from maxtext.experimental.mpmd.jax.qwen35 import layers
from maxtext.experimental.mpmd.jax.qwen35 import moe

LAYERS = "layers"


def remat_policy_for(config):
  """MaxText's `get_remat_policy` (nnx_decoders.py), shared with the DeepSeek port.

  `full` and `none` both mean "save nothing inside the checkpoint".  The names
  the policies save that this model emits are the gated attention's
  `query_proj` / `key_proj` / `value_proj` / `out_proj` (and `context` from the
  splash kernels); the GDN emits none, as in MaxText, and so does the MoE,
  unlike MaxText's: its routed experts tag `mlpwi_0` / `mlpwo` (the dense MoE
  also `mlpwi_1`) and its shared expert `mlpwi_0` / `mlpwi_1` / `mlpwo`, which
  MaxText's 'minimal' saves and this table's does not.
  """
  return ds_model.remat_policy_for(config)


def layer_remat_policy_for(config):
  """The per-layer checkpoints' policy (`layer_remat_policy`, default `remat_policy`)."""
  return ds_model.remat_policy_for(config.replace(remat_policy=config.layer_remat_policy_name))


def decoder_layer(params, x, positions, segment_ids, inv_freq, config, is_attention):
  """`Qwen3_5DecoderLayer.__call__`: pre-norm mixer + pre-norm MoE, two residuals."""
  h = layers.rms_norm(params["input_layernorm"], x, config)
  if is_attention:
    a = attention.gated_attention(params["attention"]["attention"], h, positions, segment_ids, config, inv_freq)
  else:
    a = gdn.gated_delta_net(params["attention"], h, segment_ids, config)
  x = x + a
  h = layers.rms_norm(params["post_attention_layernorm"], x, config)
  return x + moe.sparse_moe_block(params["mlp"], h, config)


def _layer_fns(config):
  """One callable per position in the block, checkpointed under layer remat."""
  fns = []
  for i in range(config.inhomogeneous_layer_cycle_interval):
    f = functools.partial(decoder_layer, config=config, is_attention=config.is_attention_layer(i))
    if config.layer_remat:
      # With the CSE barrier.  The layers' recomputes are independent of one
      # another (each starts from its own stored input), and without it XLA
      # hoists them together and the per-layer checkpoint no longer bounds the
      # working set: one 397B-dims block's backward program on v5p (4096
      # tokens, jax 0.12) holds 22.6 GB of temporaries without the barrier,
      # 12.0 GB with it, 29.3 GB under block remat (norm_numerics='bf16':
      # 20.8 / 14.2 / 29.0 GB).  (The block checkpoint keeps
      # prevent_cse=False, as MaxText: the scan already separates it.)
      f = jax.checkpoint(f, policy=layer_remat_policy_for(config), prevent_cse=True)
    fns.append(f)
  return fns


def apply_blocks(stack, x, positions, segment_ids, config, inv_freq=None):
  """Run the `[nb, ...]` stacked blocks of `stack` (`decoder/layers`) over `x`."""
  if inv_freq is None:
    inv_freq = attention.rope_inv_freq(config)
  fns = _layer_fns(config)

  def body(carry, blk):
    for i, f in enumerate(fns):
      carry = f(blk[f"layer_{i}"], carry, positions, segment_ids, inv_freq)
    return carry, None

  if config.scan_layers:
    if config.block_remat:
      body = jax.checkpoint(body, policy=remat_policy_for(config), prevent_cse=False)
    out, _ = jax.lax.scan(body, x, stack)
    return out
  nb = int(jax.tree.leaves(stack)[0].shape[0])
  for i in range(nb):
    x, _ = body(x, jax.tree.map(lambda leaf, i=i: leaf[i], stack))
  return x


def embed(params, tokens, config):
  """`[B, L]` int tokens -> `[B, L, E]` residual stream in `dtype`."""
  return _ops.embed(params["token_embedder"], tokens, config).astype(config.dtype)


def head(params, y, config):
  """Final zero-centred norm + untied unembedding (bf16 dot), logits cast to fp32."""
  dec = params["decoder"]
  y = layers.rms_norm(dec["decoder_norm"], y, config)
  logits = _ops.dense(dec["logits_dense"], y, config)
  return logits.astype(jnp.float32) if config.cast_logits_to_fp32 else logits


def forward(params, tokens, positions, segment_ids, config, inv_freq=None):
  """`Transformer.__call__` for training.  Returns `[B, L, vocab]` logits.

  `inv_freq` is the rotary table (`attention.rope_inv_freq`); None computes it
  in-graph, as MaxText does.
  """
  y = embed(params, tokens, config)
  y = apply_blocks(params["decoder"][LAYERS], y, positions, segment_ids, config, inv_freq)
  return head(params, y, config)


# ------------------------------------------------------------- param avals
def _sds(shape, dtype):
  return jax.ShapeDtypeStruct(tuple(int(d) for d in shape), dtype)


def _moe_avals(config, n):
  wd, e, x, f = config.weight_dtype, config.emb_dim, config.num_experts, config.moe_mlp_dim
  tree = {
      "routed_experts": {
          "gate": {"kernel": _sds((n, e, x), wd)},
          "wi_0": _sds((n, x, e, f), wd),
          "wi_1": _sds((n, x, e, f), wd),
          "wo": _sds((n, x, f, e), wd),
      }
  }
  if config.shared_experts:
    fs = config.shared_expert_mlp_dim
    tree["shared_expert"] = {
        "wi_0": {"kernel": _sds((n, e, fs), wd)},
        "wi_1": {"kernel": _sds((n, e, fs), wd)},
        "wo": {"kernel": _sds((n, fs, e), wd)},
    }
    tree["shared_expert_gate"] = {"kernel": _sds((n, e, 1), wd)}
  return tree


def gdn_layer_avals(config, n):
  """One stacked GDN layer position, `[n, ...]`."""
  wd, e = config.weight_dtype, config.emb_dim
  hv, k = config.gdn_num_value_heads, config.gdn_conv_kernel_dim
  return {
      "input_layernorm": {"scale": _sds((n, e), wd)},
      "post_attention_layernorm": {"scale": _sds((n, e), wd)},
      "attention": {
          "in_proj_qkvz": {"kernel": _sds((n, e, config.gdn_qkvz_dim), wd)},
          "in_proj_ba": {"kernel": _sds((n, e, 2 * hv), wd)},
          "conv1d": {"kernel": _sds((n, k, 1, config.gdn_conv_dim), wd)},
          "A_log": _sds((n, hv), wd),
          "dt_bias": _sds((n, hv), wd),
          "norm": {"rms_norm": {"scale": _sds((n, config.gdn_value_head_dim), wd)}},
          "out_proj": {"kernel": _sds((n, config.gdn_value_dim, e), wd)},
      },
      "mlp": _moe_avals(config, n),
  }


def attention_layer_avals(config, n):
  """One stacked gated-attention layer position, `[n, ...]`."""
  wd, e = config.weight_dtype, config.emb_dim
  h, kv, d = config.num_query_heads, config.num_kv_heads, config.head_dim
  return {
      "input_layernorm": {"scale": _sds((n, e), wd)},
      "post_attention_layernorm": {"scale": _sds((n, e), wd)},
      "attention": {
          "attention": {
              "query": {"kernel": _sds((n, e, h, 2 * d), wd)},
              "key": {"kernel": _sds((n, e, kv, d), wd)},
              "value": {"kernel": _sds((n, e, kv, d), wd)},
              "query_norm": {"scale": _sds((n, d), wd)},
              "key_norm": {"scale": _sds((n, d), wd)},
              "out": {"kernel": _sds((n, h * d, e), wd)},
          }
      },
      "mlp": _moe_avals(config, n),
  }


def block_avals(config, n):
  """`decoder/layers` for `n` stacked blocks."""
  return {
      f"layer_{i}": (attention_layer_avals(config, n) if config.is_attention_layer(i) else gdn_layer_avals(config, n))
      for i in range(config.inhomogeneous_layer_cycle_interval)
  }


def param_avals(config):
  """`jax.ShapeDtypeStruct` tree of the FULL model; never materialised."""
  wd = config.weight_dtype
  return {
      "token_embedder": {"embedding": _sds((config.vocab_size, config.emb_dim), wd)},
      "decoder": {
          LAYERS: block_avals(config, config.num_blocks),
          "decoder_norm": {"scale": _sds((config.emb_dim,), wd)},
          "logits_dense": {"kernel": _sds((config.emb_dim, config.vocab_size), wd)},
      },
  }
