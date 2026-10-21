# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Qwen3-MoE (Qwen3-30B-A3B, Qwen3-235B-A22B) in m3 format.

Qwen3-MoE is Qwen3 with the dense MLP replaced by a routed mixture of
experts: a softmax router picks `num_experts_per_tok` experts per token, and
a Switch-style load-balance loss keeps the routing spread out. Attention and
the decoder stack are the dense model's, imported from `..qwen3`; what is
defined here is exactly the part that differs.

Routing semantics live in this file. The expert matmuls themselves come from
`maxtext.m3.core.moe`, which knows nothing about softmax or top-k
normalization -- it receives the decision and executes it.

The load-balance loss is returned, not `sow`n: each MoE layer adds its term
to an accumulator carried alongside the hidden states through the decoder
stack, and the model hands the mean over layers back as
`CausalLMOutput.aux_loss`. The training loss adds it to the cross entropy;
nothing in the model's non-parameter state changes between calls.

Parameter names match the legacy checkpoint layout: `moe_block.gate.kernel`,
`moe_block.wi_0`, `moe_block.wi_1`, `moe_block.wo`.
"""

from flax import nnx
import jax
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp
from jax.sharding import Mesh

from maxtext.common.common_types import Config
from maxtext.m3.core import moe
from maxtext.m3.core.interfaces import CausalLMOutput
from maxtext.m3.core.sharding import DefaultSharding, Sharding
from maxtext.m3.models.qwen3.modeling_qwen3 import Qwen3Attention, Qwen3Decoder, norm_kwargs

# What flows through the decoder stack: hidden states and the running sum of
# per-layer load-balance losses.
Carry = tuple[jax.Array, jax.Array]


class Qwen3MoeBlock(nnx.Module):
  """Softmax top-k router and routed SwiGLU experts."""

  def __init__(self, config: Config, *, sharding: Sharding, rngs: nnx.Rngs):
    self.config = config
    self.sharding = sharding
    self.executor = moe.executor_from_config(config, sharding)
    num_experts, emb, mlp = config.num_experts, config.emb_dim, config.moe_mlp_dim

    # The router's expert dimension is replicated (legacy: `kernel_axes=("embed", None)`):
    # every device needs every expert's score to route.
    self.gate = nnx.Linear(
        emb,
        num_experts,
        use_bias=False,
        dtype=jnp.float32 if config.float32_gate_logits else config.dtype,
        param_dtype=config.weight_dtype,
        kernel_metadata={"out_sharding": sharding("gate", ("embed", None))},
        rngs=rngs,
    )
    # One 3-D kernel per projection rather than a list of expert modules: the
    # grouped matmuls want the experts stacked, and so does the checkpoint.
    init = nnx.initializers.variance_scaling(config.dense_init_scale, "fan_in", "truncated_normal", in_axis=1, out_axis=2)
    self.wi_0 = nnx.Param(
        init(rngs.params(), (num_experts, emb, mlp), config.weight_dtype),
        out_sharding=sharding("wi_0", ("experts", "expert_embed", "expert_mlp")),
    )
    self.wi_1 = nnx.Param(
        init(rngs.params(), (num_experts, emb, mlp), config.weight_dtype),
        out_sharding=sharding("wi_1", ("experts", "expert_embed", "expert_mlp")),
    )
    self.wo = nnx.Param(
        init(rngs.params(), (num_experts, mlp, emb), config.weight_dtype),
        out_sharding=sharding("wo", ("experts", "expert_mlp", "expert_embed")),
    )

  def route(self, x: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Returns `(weights, indices, probs)`: top-k combine weights, their expert ids, full router probabilities."""
    cfg = self.config
    logits = self.sharding.constrain(self.gate(x), "router_logits", ("batch", "length", "experts"))
    top_logits, indices = jax.lax.top_k(logits, cfg.num_experts_per_tok)
    # Softmax over the selected logits only, as Qwen3 does (not a renormalized
    # slice of the full softmax). `norm_topk_prob` then makes the weights sum
    # to one, which after a softmax they already do; legacy keeps the step
    # for parity with the reference implementation, and so does this.
    weights = jax.nn.softmax(top_logits.astype(jnp.float32), axis=-1).astype(cfg.dtype)
    if cfg.norm_topk_prob:
      weights = weights / jnp.sum(weights, axis=-1, keepdims=True)
    probs = jax.nn.softmax(logits.astype(jnp.float32), axis=-1).astype(cfg.dtype)
    return weights, indices, probs

  def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Returns `(output, load_balance_loss)` for `x` of shape `(batch, length, embed)`."""
    cfg = self.config
    x = x.astype(cfg.dtype)
    weights, indices, probs = self.route(x)
    out = moe.routed_experts(x, weights, indices, self.wi_0[...], self.wi_1[...], self.wo[...], executor=self.executor)
    out = checkpoint_name(out, "mlpwo")
    if cfg.load_balance_loss_weight > 0:
      aux = moe.load_balance_loss(indices, probs, weight=cfg.load_balance_loss_weight)
    else:
      aux = jnp.zeros((), jnp.float32)
    return out, aux


class Qwen3MoeDecoderLayer(nnx.Module):
  """Pre-norm attention block followed by a pre-norm MoE block.

  Takes and returns a `Carry`: the running load-balance loss rides alongside
  the hidden states so that a `Layers` container -- scanned or not -- needs no
  side channel.
  """

  def __init__(self, config: Config, *, sharding: Sharding, rngs: nnx.Rngs):
    self.config = config
    self.sharding = sharding
    self.pre_self_attention_layer_norm = nnx.RMSNorm(
        config.emb_dim, **norm_kwargs(config, sharding, "pre_self_attention_layer_norm", rngs)
    )
    self.self_attention = Qwen3Attention(config, sharding=sharding, rngs=rngs)
    self.post_self_attention_layer_norm = nnx.RMSNorm(
        config.emb_dim, **norm_kwargs(config, sharding, "post_self_attention_layer_norm", rngs)
    )
    self.moe_block = Qwen3MoeBlock(config, sharding=sharding, rngs=rngs)

  def __call__(
      self, carry: Carry, *, decoder_positions: jax.Array, decoder_segment_ids: jax.Array | None = None
  ) -> Carry:
    x, aux_loss = carry
    x = checkpoint_name(
        self.sharding.constrain(x, "decoder_layer_input", ("batch", "length", "embed")), "decoder_layer_input"
    )
    attn_out = self.self_attention(
        self.pre_self_attention_layer_norm(x),
        decoder_positions=decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
    )
    x = (x + attn_out).astype(self.config.dtype)
    moe_out, layer_aux = self.moe_block(self.post_self_attention_layer_norm(x))
    return (x + moe_out).astype(self.config.dtype), aux_loss + layer_aux


class Qwen3MoeModel(nnx.Module):
  """Qwen3-MoE language model."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh | None = None,
      *,
      sharding: Sharding | None = None,
      rngs: nnx.Rngs,
      **unused_kwargs,
  ):
    del mesh, unused_kwargs  # See `Qwen3Model.__init__`.
    self.config = config
    self.sharding = sharding or DefaultSharding()
    self.token_embedder = nnx.Embed(
        config.vocab_size,
        config.emb_dim,
        dtype=config.dtype,
        param_dtype=config.weight_dtype,
        embedding_metadata={"out_sharding": self.sharding("token_embedder", ("vocab", "embed"))},
        rngs=rngs,
    )
    self.decoder = Qwen3Decoder(config, sharding=self.sharding, rngs=rngs, layer_cls=Qwen3MoeDecoderLayer)

  def __call__(
      self,
      decoder_input_tokens: jax.Array,
      decoder_positions: jax.Array,
      decoder_segment_ids: jax.Array | None = None,
      **unused_kwargs,
  ) -> CausalLMOutput:
    """See `Qwen3Model.__call__`. `aux_loss` is the mean load-balance loss over layers."""
    del unused_kwargs
    y = self.token_embedder(decoder_input_tokens)
    carry: Carry = (y, jnp.zeros((), jnp.float32))
    y, aux_loss = self.decoder.layers(carry, decoder_positions=decoder_positions, decoder_segment_ids=decoder_segment_ids)
    logits = self.decoder.apply_output_head(self.token_embedder, y)
    return CausalLMOutput(logits=logits, aux_loss=aux_loss / self.config.num_decoder_layers)


def create_qwen3_moe_model(
    config: Config, mesh: Mesh | None = None, *, rngs: nnx.Rngs | None = None, **kwargs
) -> Qwen3MoeModel:
  """Registry entry point; see `maxtext.m3.models.create_model`."""
  return Qwen3MoeModel(config, mesh, rngs=rngs or nnx.Rngs(0), **kwargs)
