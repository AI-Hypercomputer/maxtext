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

"""Qwen3.5's sparse MoE block: softmax-over-top-k router, routed experts, and one
sigmoid-gated shared expert.

Mirrors `Qwen3NextSparseMoeBlock.__call__` (models/qwen3.py) and the routing
path `GateLogit` + `RoutedMoE.get_topk` takes for it (layers/moe.py):

    logits = x @ gate.kernel        [B,S,X], `dtype` (fp32 with float32_gate_logits),
                                    no bias, no score function
    w, idx = top_k(logits, k)       top-k on the RAW logits
    w = bf16(softmax(fp32(w)))      softmax over the k selected only
    w = w / sum(w)                  norm_topk_prob, in bf16 (not an identity)
    out = experts(x, w, idx) + sigmoid(x @ shared_expert_gate) * shared_mlp(x)

That is HF's "softmax over all experts, top-k, renormalise", up to rounding.
The routed experts and the shared expert are the DeepSeek port's
(`moe.routed_experts` and `layers.mlp_block`).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from maxtext.experimental.mpmd.jax.qwen35 import _ops
from maxtext.experimental.mpmd.jax.qwen35 import layers


def route(params, x, config):
  """Returns `(weights [B,S,k] dtype, indices [B,S,k] int32)`."""
  gate_dtype = jnp.float32 if config.float32_gate_logits else config.dtype
  kernel = _ops.use_weight(params["kernel"], gate_dtype, config)
  logits = _ops.dot_general(
      jnp.asarray(x, gate_dtype), kernel, (((x.ndim - 1,), (0,)), ((), ())), precision=layers.precision_of(config)
  )
  weights, indices = jax.lax.top_k(logits, config.num_experts_per_tok)
  weights = jax.nn.softmax(weights.astype(jnp.float32), axis=-1).astype(config.dtype)
  if config.norm_topk_prob:
    weights = weights / weights.sum(axis=-1, keepdims=True)
  return weights, indices


def sparse_moe_block(params, x, config):
  """`[B, S, E]` -> `[B, S, E]`.  (No load-balance loss: the config refuses it.)"""
  x = x.astype(config.dtype)
  routed = params["routed_experts"]
  weights, indices = route(routed["gate"], x, config)
  out = _ops.routed_experts(routed, x, weights, indices, config)
  if config.shared_experts:
    shared = _ops.mlp(params["shared_expert"], x, config)
    gate = _ops.dense(params["shared_expert_gate"], x, config)  # [B, S, 1]
    out = out + jax.nn.sigmoid(gate) * shared
  return out
