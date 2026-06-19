# Copyright 2023–2026 Google LLC
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

"""Transformer model definition."""
# pylint: disable=arguments-differ
# pylint: disable=no-name-in-module

import contextlib
import functools
from typing import Optional

from flax import linen as flax_linen
from flax import nnx
import jax
from jax.ad_checkpoint import checkpoint_name
import jax.experimental.xla_metadata
import jax.numpy as jnp
from jax.sharding import Mesh
from maxtext.common.common_types import Config
from maxtext.common.common_types import HyperConnectionType, MODEL_MODE_PREFILL
from maxtext.layers import attention_mla
from maxtext.layers import initializers
from maxtext.layers import linears
from maxtext.layers import mhc
from maxtext.layers import moe
from maxtext.layers import nnx_wrappers
from maxtext.layers import quantizations
from maxtext.layers.linears import Dropout
from maxtext.layers.engram import Engram
from maxtext.layers.engram import NgramHashMapping
from maxtext.layers.normalizations import RMSNorm
from maxtext.models import deepseek_batchsplit
from maxtext.models import deepseek_batchsplit_fp8
from maxtext.utils import max_utils
from maxtext.utils.sharding import create_sharding
from maxtext.utils.sharding import maybe_shard_with_logical

import transformers

# -----------------------------------------
# The Decoder Layer for DeepSeek v3
# -----------------------------------------


@contextlib.contextmanager
def _detached_linen_module_stack():
  """Temporarily clear flax's linen module stack so linen↔nnx bridge wrappers take
  their no-context passthrough branch.

  The to_linen bridge installs a qwix-quantization fixup on every nnx submodule's
  __call__ that reads ``linen.module._context.module_stack[-1].path``. The stack's
  resting value is ``[None]`` (a sentinel base), and a valid linen module is only on
  it while ``ToLinen.__call__`` is executing the forward. The hand-written layer
  backward (``_handwritten_moe_layer``) re-traces the bridged layer methods OUTSIDE
  that forward (during the custom_vjp transpose), where ``module_stack[-1]`` is the
  ``None`` sentinel -> ``None.path`` AttributeError. Clearing the stack to ``[]`` makes
  the fixup short-circuit to ``call_fn(...)``. This only drops qwix path tracking
  (a no-op when qwix quantization is off, which the hand-written path requires); the
  traced computation is identical, so numerics are unchanged.
  """
  ctx = flax_linen.module._context  # pylint: disable=protected-access
  saved = list(ctx.module_stack)
  ctx.module_stack.clear()
  try:
    yield
  finally:
    ctx.module_stack.clear()
    ctx.module_stack.extend(saved)


class DeepSeekGenericLayer(nnx.Module):
  """Generic DeepSeek layer with Multi-Head Latent Attention.

  This is to be used as a base class for DeepSeek layers with dense/sparse MLPs.
  This class follows a pattern of separating module creation from execution.
  """

  def __init__(
      self,
      config: Config,
      model_mode: str,
      mesh: Mesh,
      rngs: nnx.Rngs,
      quant: Optional[quantizations.AqtQuantization] = None,
      layer_idx: int = -1,
  ) -> None:
    self.config = config
    self.model_mode = model_mode
    self.mesh = mesh
    self.quant = quant
    self.rngs = rngs
    self.is_mhc_enabled = config.mhc_expansion_rate > 1
    self.layer_idx = layer_idx
    self.is_engram_enabled = config.engram_layers and layer_idx in config.engram_layers

    batch_size, sequence_length = max_utils.get_batch_seq_len_for_mode(self.config, self.model_mode)
    self.dummy_inputs_shape = (batch_size, sequence_length, self.config.emb_dim)

    self.out_sharding = create_sharding(self.mesh, self.logical_axis_names, rules=self.config.logical_axis_rules)
    self.mlp_intermediate_sharding = create_sharding(
        self.mesh, self.mlp_logical_axis_names, rules=self.config.logical_axis_rules
    )

    self.pre_self_attention_layer_norm = RMSNorm(
        num_features=self.dummy_inputs_shape[-1],
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        kernel_axes=("norm",),
        epsilon=self.config.normalization_layer_epsilon,
        rngs=rngs,
    )

    self.post_self_attention_layer_norm = RMSNorm(
        num_features=self.dummy_inputs_shape[-1],
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        kernel_axes=("norm",),
        epsilon=self.config.normalization_layer_epsilon,
        rngs=rngs,
    )

    if self.is_engram_enabled:
      self.engram_layer_norm = RMSNorm(
          num_features=self.dummy_inputs_shape[-1],
          dtype=self.config.dtype,
          weight_dtype=self.config.weight_dtype,
          kernel_axes=("norm",),
          epsilon=self.config.normalization_layer_epsilon,
          rngs=rngs,
      )
      tokenizer = transformers.AutoTokenizer.from_pretrained(config.tokenizer_path, token=config.hf_access_token)
      # TODO(ranran): Refactor NgramHashMapping to initialize once globally or at the model level.
      # Moving this to decoders.py currently causes JAX initialization errors.
      self.ngram_hash_mapping = NgramHashMapping(
          engram_vocab_bases=config.engram_vocab_bases,
          max_ngram_size=config.engram_max_ngram_size,
          engram_num_heads=config.engram_num_heads,
          layer_ids=config.engram_layers,
          tokenizer=tokenizer,
          pad_id=tokenizer.pad_token_id,
          seed=config.engram_seed,
      )
      self.engram = Engram(
          config=config,
          mesh=mesh,
          vocab_sizes=self.ngram_hash_mapping.get_vocab_sizes(layer_idx),
          engram_num_heads=config.engram_num_heads,
          engram_head_dim=config.engram_head_dim,
          engram_max_ngram_size=config.engram_max_ngram_size,
          engram_kernel_size=config.engram_kernel_size,
          mhc_expansion_rate=config.mhc_expansion_rate,
          quant=quant,
          rngs=rngs,
      )
    else:
      self.engram_layer_norm = None
      self.engram = None

    self.self_attention = attention_mla.MLA(
        config=self.config,
        num_query_heads=self.config.num_query_heads,
        num_kv_heads=self.config.num_kv_heads,
        head_dim=self.config.head_dim,
        max_target_length=self.config.max_target_length,
        max_prefill_predict_length=self.config.max_prefill_predict_length,
        attention_kernel=self.config.attention,
        attention_type=self.config.attention_type,
        inputs_q_shape=self.dummy_inputs_shape,
        inputs_kv_shape=self.dummy_inputs_shape,
        mesh=mesh,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        dropout_rate=self.config.dropout_rate,
        name="self_attention",
        quant=quant,
        kv_quant=quantizations.configure_kv_quant(config),
        q_lora_rank=self.config.q_lora_rank,
        kv_lora_rank=self.config.kv_lora_rank,
        qk_nope_head_dim=self.config.qk_nope_head_dim,
        qk_rope_head_dim=self.config.qk_rope_head_dim,
        v_head_dim=self.config.v_head_dim,
        max_position_embeddings=self.config.max_position_embeddings,
        original_max_position_embeddings=self.config.original_max_position_embeddings,
        mscale=self.config.mscale,
        rope_factor=self.config.rope_factor,
        model_mode=model_mode,
        rngs=rngs,
        attn_logits_soft_cap=self.config.attn_logits_soft_cap,
    )

    self.dropout = Dropout(rate=self.config.dropout_rate, broadcast_dims=(-2,), rngs=self.rngs)
    if self.is_mhc_enabled:
      self.mhc_attention = mhc.ManifoldConstrainedHyperConnections(self.config, self.config.emb_dim, self.mesh, self.rngs)
      self.mhc_mlp = mhc.ManifoldConstrainedHyperConnections(self.config, self.config.emb_dim, self.mesh, self.rngs)

  def mlp_op(self, x, deterministic, *args, **kwargs):
    """Executes the MLP operation. To be implemented by subclasses."""
    raise NotImplementedError()

  def with_logical_constraint(self, x):
    return maybe_shard_with_logical(
        x,
        logical_axes=self.logical_axis_names,
        mesh=self.mesh,
        shard_mode=self.config.shard_mode,
        debug_sharding=self.config.debug_sharding,
        extra_stack_level=1,
        rules=self.config.logical_axis_rules,
    )

  def dropout_op(self, x, deterministic):
    dropout = self.dropout(x, deterministic=deterministic)
    return self.with_logical_constraint(dropout)

  def pre_attention_norm_op(self, x):
    pre_attention_norm = self.pre_self_attention_layer_norm(x)
    return self.with_logical_constraint(pre_attention_norm)

  def post_attention_norm_op(self, x):
    post_attention_norm = self.post_self_attention_layer_norm(x)
    return self.with_logical_constraint(post_attention_norm)

  def attention_op(
      self,
      x,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      previous_chunk=None,
      slot: None | int = None,
      wag_cell=None,
  ):
    """Executes the attention layer."""
    # Splash attention is NOT tagged. The weight-AG scheduling group must contain ONLY
    # the gather (a contiguous, self-contained collective block, already tagged inside
    # gather_weights). Co-tagging splash makes the group span the un-annotated QKV
    # dot_generals between the gather and splash -> "annotation group with gaps" ->
    # XLA UNIMPLEMENTED. Overlap is emergent (batchsplit pattern): the grouped gather
    # is emitted before attention and the latency-hiding scheduler floats it over the
    # (ungrouped) splash compute while the SparseCore is idle.
    _sched = contextlib.nullcontext()
    with _sched:
      attention_result, _ = self.self_attention(
          x,
          x,
          decoder_positions,
          decoder_segment_ids=decoder_segment_ids,
          deterministic=deterministic,
          model_mode=self.model_mode,
          out_sharding=self.out_sharding,
          previous_chunk=previous_chunk,
          slot=slot,
          wag_cell=wag_cell,
      )
    return self.with_logical_constraint(attention_result)

  @property
  def logical_axis_names(self):
    """Generate logical names for activations generally."""
    length_name = "prefill_activation_norm_length" if self.model_mode == MODEL_MODE_PREFILL else "activation_norm_length"
    axis_names = ["activation_batch", length_name, "activation_embed"]
    return axis_names

  @property
  def mlp_logical_axis_names(self):
    """Generate logical names for activations in MLP."""
    length_name = "prefill_activation_norm_length" if self.model_mode == MODEL_MODE_PREFILL else "activation_norm_length"
    axis_names = ["activation_batch", length_name, "activation_mlp"]
    return axis_names

  def post_process(self, layer_output, load_balance_loss, moe_bias_updates, kv_cache=None):
    """postprocessing."""

    if self.config.load_balance_loss_weight > 0.0 and load_balance_loss is not None:
      self.sow(nnx.Intermediate, "moe_lb_loss", load_balance_loss)

    if self.config.routed_bias and self.config.routed_bias_update_rate > 0.0 and moe_bias_updates is not None:
      self.sow(nnx.Intermediate, "moe_bias_updates", moe_bias_updates)

    if self.config.record_internal_nn_metrics:
      self.sow(nnx.Intermediate, "activation_mean", jnp.mean(layer_output))
      self.sow(nnx.Intermediate, "activation_stdev", jnp.std(layer_output))
      self.sow(
          nnx.Intermediate,
          "activation_fraction_zero",
          jnp.sum(layer_output == 0) / jnp.size(layer_output),
      )

    if self.config.scan_layers:
      return layer_output, None
    return layer_output, kv_cache

  def self_attention_with_norm_op(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      previous_chunk=None,
      slot: None | int = None,
      wag_cell=None,
  ):
    """self-attention with normalization"""
    if self.is_mhc_enabled:
      intermediate_inputs, _ = self.mhc_attention(
          self.pre_attention_norm_op,
          self.self_attention,
          x=inputs,
          mhc_type=HyperConnectionType.ATTENTION,
          decoder_segment_ids=decoder_segment_ids,
          inputs_positions=decoder_positions,
          deterministic=deterministic,
          model_mode=self.model_mode,
          out_sharding=self.out_sharding,
          previous_chunk=previous_chunk,
          slot=slot,
      )
    else:
      lnx = self.pre_attention_norm_op(inputs)
      attention_lnx = self.attention_op(
          lnx,
          decoder_segment_ids,
          decoder_positions,
          deterministic,
          previous_chunk,
          slot,
          wag_cell=wag_cell,
      )
      intermediate_inputs = inputs + attention_lnx
    # Normalization
    hidden_states = self.post_attention_norm_op(intermediate_inputs)
    return hidden_states, intermediate_inputs

  def engram_op(self, x, decoder_input_tokens):
    normed_x = self.engram_layer_norm(x)
    hash_ids = self.ngram_hash_mapping(decoder_input_tokens)[self.layer_idx]
    return self.engram(normed_x, hash_ids)


class DeepSeekDenseLayer(DeepSeekGenericLayer):
  """DeepSeek-style dense layer with Multi-Head Latent Attention."""

  def __init__(
      self,
      config: Config,
      model_mode: str,
      mesh: Mesh,
      rngs: nnx.Rngs,
      quant: Optional[quantizations.AqtQuantization] = None,
      layer_idx: int = -1,
  ) -> None:
    super().__init__(config, model_mode, mesh, rngs, quant, layer_idx)
    self.mlp = linears.MlpBlock(
        in_features=self.dummy_inputs_shape[-1],
        intermediate_dim=self.config.mlp_dim,
        activations=self.config.mlp_activations,
        intermediate_dropout_rate=self.config.dropout_rate,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        config=self.config,
        quant=quant,
        model_mode=model_mode,
        mesh=mesh,
        rngs=self.rngs,
    )

  def mlp_op(self, x, deterministic):
    mlp = self.mlp(x, deterministic, intermediate_sharding=self.mlp_intermediate_sharding, out_sharding=self.out_sharding)
    return self.with_logical_constraint(mlp)

  def __call__(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      previous_chunk=None,
      slot: None | int = None,
      kv_cache=None,
      attention_metadata=None,
      decoder_input_tokens=None,
  ):
    # Unpack inputs if it's a tuple (e.g. from a previous layer returning (hidden_states, kv_cache))
    if isinstance(inputs, tuple):
      inputs = inputs[0]
    x = self.with_logical_constraint(inputs)
    x = checkpoint_name(x, "decoder_layer_input")

    if self.is_engram_enabled:
      engram_output = self.engram_op(x, decoder_input_tokens)
      x = x + engram_output

    hidden_states, intermediate_inputs = self.self_attention_with_norm_op(
        x,
        decoder_segment_ids,
        decoder_positions,
        deterministic,
        previous_chunk,
        slot,
    )

    if self.is_mhc_enabled:
      layer_output, _ = self.mhc_mlp(
          self.post_attention_norm_op,
          self.mlp,
          x=intermediate_inputs,
          mhc_type=HyperConnectionType.MLP_DENSE,
          deterministic=deterministic,
      )
    else:
      mlp_lnx = self.mlp_op(hidden_states, deterministic)
      layer_output = mlp_lnx + intermediate_inputs
    layer_output = self.dropout_op(layer_output, deterministic=deterministic)

    return self.post_process(layer_output, None, None, kv_cache)


DeepSeekDenseLayerToLinen = nnx_wrappers.to_linen_class(
    DeepSeekDenseLayer,
    base_metadata_fn=initializers.variable_to_logically_partitioned,
)


class DeepSeekMoELayer(DeepSeekGenericLayer):
  """DeepSeek-style MoE layer with Multi-Head Latent Attention.

  Supports dropless and dropping base on configs. Uses a bias in routing instead
  of load balancing loss.
  """

  def __init__(
      self,
      config: Config,
      model_mode: str,
      mesh: Mesh,
      rngs: nnx.Rngs,
      quant: Optional[quantizations.AqtQuantization] = None,
      layer_idx: int = -1,
  ) -> None:
    super().__init__(config, model_mode, mesh, rngs, quant, layer_idx)
    self.DeepSeekMoeBlock_0 = moe.RoutedAndSharedMoE(
        config=self.config,
        mesh=mesh,
        kernel_init=initializers.nd_dense_init(self.config.dense_init_scale, "fan_in", "truncated_normal"),
        kernel_axes=("embed", None),
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        quant=quant,
        rngs=self.rngs,
    )

  def __call__(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      previous_chunk=None,
      slot: None | int = None,
      kv_cache=None,
      attention_metadata=None,
      decoder_input_tokens=None,
  ):
    # Unpack inputs if it's a tuple (e.g. from a previous layer returning (hidden_states, kv_cache))
    if isinstance(inputs, tuple):
      inputs = inputs[0]

    # This code should only be traced during initialization when using
    # batch-split schedule. It is never run during model execution, since
    # `Decoder` directly calls `batch_split_schedule` during execution.
    # That is also why we can split/merge activations here as well as
    # in `Decoder`, since they will never be executed together.
    if self.config.use_batch_split_schedule:
      # The older version of batch-split that fully uses qwix quantization.
      if self.config.use_qwix_quantization and not self.config.use_manual_quantization:
        activation_pspec = jax.sharding.PartitionSpec(
            ("data", "fsdp", "fsdp_transpose", "expert", "context"),
            None,
            None,
        )
        inputs = jax.shard_map(
            functools.partial(
                deepseek_batchsplit_fp8.split,
                split_factor=self.config.batch_split_factor,
            ),
            mesh=self.mesh,
            in_specs=activation_pspec,
            out_specs=[activation_pspec] * self.config.batch_split_factor,
        )(inputs)
        dpos = deepseek_batchsplit_fp8.split(decoder_positions, self.config.batch_split_factor)
        dseg = deepseek_batchsplit_fp8.split(decoder_segment_ids, self.config.batch_split_factor)
        weights = deepseek_batchsplit_fp8.fetch_weights(nnx.to_pure_dict(nnx.state(self, nnx.Param)), self.config.dtype)
        outputs = deepseek_batchsplit_fp8.batch_split_schedule(
            inputs,
            weights,
            dpos,
            dseg,
            model_mode=model_mode,
            mesh=self.mesh,
            quant=self.quant,
            cfg=self.config,
        )
        outputs = jax.shard_map(
            functools.partial(
                deepseek_batchsplit_fp8.merge,
                split_factor=self.config.batch_split_factor,
            ),
            mesh=self.mesh,
            in_specs=([activation_pspec] * self.config.batch_split_factor,),
            out_specs=activation_pspec,
        )(outputs)
        return outputs, None

      # bf16 and fp8 code path for pure-JAX batch-split.
      # fp8 code path supports both manual quantization and qwix
      # quantization.
      input_sharding = jax.typeof(inputs).sharding
      activation_pspec = jax.sharding.PartitionSpec(
          ("data", "fsdp", "expert"),
          None,
          None,
      )
      inputs = jax.reshard(inputs, jax.sharding.NamedSharding(self.mesh, activation_pspec))
      yarn_freqs = deepseek_batchsplit.initialize_yarn_freqs(
          decoder_positions,
          embedding_dims=self.config.qk_rope_head_dim,
          rope_theta=self.config.rope_max_timescale,
          max_position_embeddings=self.config.max_position_embeddings,
          original_max_position_embeddings=self.config.original_max_position_embeddings,
          beta_fast=self.config.beta_fast,
          beta_slow=self.config.beta_slow,
          rope_factor=self.config.rope_factor,
          mesh=self.mesh,
          activation_pspec=activation_pspec,
      )
      yarn_mask = deepseek_batchsplit.initialize_yarn_mask(self.config.qk_rope_head_dim)
      splash_kernel = deepseek_batchsplit.init_splash_kernel(self.config)
      inputs = jax.shard_map(
          functools.partial(
              deepseek_batchsplit.split,
              split_factor=self.config.batch_split_factor,
          ),
          mesh=self.mesh,
          in_specs=activation_pspec,
          out_specs=[activation_pspec] * self.config.batch_split_factor,
      )(inputs)
      yarn_freqs = deepseek_batchsplit.split(yarn_freqs, self.config.batch_split_factor)

      def extract_fn(x):
        if isinstance(x, nnx.variablelib.Variable):
          return maybe_shard_with_logical(
              x.value,
              x.sharding_names,
              self.mesh,
              shard_mode=self.config.shard_mode,
              rules=self.config.logical_axis_rules,
          )
        return x

      weights = deepseek_batchsplit.fetch_weights(
          nnx.to_pure_dict(nnx.state(self, nnx.Param), extract_fn), self.config.dtype
      )
      weights = deepseek_batchsplit.gather_weights(weights, self.mesh)
      outputs, _ = deepseek_batchsplit.batch_split_schedule(
          inputs,
          weights,
          yarn_freqs,
          mesh=self.mesh,
          cfg=self.config,
          splash_kernel=splash_kernel,
          activation_pspec=activation_pspec,
          pairwise_swap_and_negate_mask=yarn_mask,
      )
      moe_inputs, routed_expert_out, shared_expert_out, selected_experts = outputs[1]
      outputs[1], _ = deepseek_batchsplit.unroute_ubatch_shard_mapped(
          moe_inputs,
          routed_expert_out,
          shared_expert_out,
          selected_experts,
          expert_axis_name="expert",
          use_gather_mosaic_kernel=False,
          target_length=self.config.max_target_length,
          mesh=self.mesh,
          activation_pspec=activation_pspec,
      )
      outputs = jax.shard_map(
          functools.partial(
              deepseek_batchsplit.merge,
              split_factor=self.config.batch_split_factor,
          ),
          mesh=self.mesh,
          in_specs=([activation_pspec] * self.config.batch_split_factor,),
          out_specs=activation_pspec,
      )(outputs)
      outputs = jax.reshard(outputs, input_sharding)
      return outputs, None

    x = self.with_logical_constraint(inputs)
    x = checkpoint_name(x, "decoder_layer_input")

    if self.is_engram_enabled:
      engram_output = self.engram_op(x, decoder_input_tokens)
      x = x + engram_output

    # Hand-written layer backward (flag: moe_handwritten_bwd): wrap gather + attention + MoE in a
    # jax.custom_vjp so WE own the backward schedule (place the annotated weight re-gather adjacent
    # to the recomputed attention forward) and there is no auto-remat to cycle against. Restricted
    # to the plain ring path (no mhc/engram, no splash/attn over-grouping) where gather_routed_weights
    # returns a real (w0, w1, wo) tuple. Bit-exact to the autodiff path below.
    if (
        self.config.moe_handwritten_bwd
        and self.config.moe_weight_ag_scheduling_group
        and not self.is_mhc_enabled
        and not self.is_engram_enabled
        and not self.config.moe_wag_splash_group
        and not self.config.moe_wag_attn_group
    ):
      layer_output, load_balance_loss, moe_bias_updates = self._handwritten_moe_layer(
          x, decoder_segment_ids, decoder_positions, deterministic
      )
      return self.post_process(layer_output, load_balance_loss, moe_bias_updates, kv_cache)

    # Pre-gather the routed MoE FSDP weights HERE, before the splash attention, so
    # the (SC-offloaded) weight all-gather is emitted in program order during the
    # attention phase where the SparseCore is idle -> the scheduler overlaps it with
    # the splash kernel (both share the _scheduling_group_id tagged in attention_op).
    # Returns None unless the plain bf16 ring path holds, in which case the MoE falls
    # back to its in-block gather.
    pregathered_weights = None
    wag_cell = None
    if self.config.moe_weight_ag_scheduling_group:
      pregathered_weights = self.DeepSeekMoeBlock_0.gather_routed_weights()
      # w1-in-splash (moe_wag_splash_group): gather_weights returns a zero-arg CLOSURE in the w1 slot
      # instead of the gathered weight. Thread it into the attention via wag_cell; tpu_flash_attention
      # invokes it adjacent to the splash kernel (same _scheduling_group -> contiguous -> overlap) and
      # writes the gathered w1 back into the cell, which we splice into pregathered_weights below.
      if pregathered_weights is not None and callable(pregathered_weights[1]):
        wag_cell = {"gather_w1": pregathered_weights[1]}

    # moe_wag_attn_group: tag the ENTIRE attention forward with the same _scheduling_group_id as the
    # expert weight gathers, so {gathers + attention} forms ONE contiguous scheduling region and the
    # scheduler can overlap the gathers across the whole attention phase (incl. the splash kernel),
    # not only behind the QKV matmuls. Wrapping the whole call (not just the splash kernel) keeps the
    # region gap-free (no ungrouped QKV/norm wedged inside the group).
    _attn_grp_ctx = (
        moe._scheduling_group(moe._WEIGHT_AG_SCHED_GROUP)
        if (self.config.moe_weight_ag_scheduling_group and self.config.moe_wag_attn_group)
        else contextlib.nullcontext()
    )
    with _attn_grp_ctx:
      hidden_states, intermediate_inputs = self.self_attention_with_norm_op(
          x,
          decoder_segment_ids,
          decoder_positions,
          deterministic,
          previous_chunk,
          slot,
          wag_cell=wag_cell,
      )

    if wag_cell is not None and "w1_gathered" in wag_cell:
      _w0, _w1c, _wo = pregathered_weights
      pregathered_weights = (_w0, wag_cell["w1_gathered"], _wo)

    if self.is_mhc_enabled:
      layer_output, metadata = self.mhc_mlp(
          self.post_attention_norm_op,
          self.DeepSeekMoeBlock_0,
          x=intermediate_inputs,
          mhc_type=HyperConnectionType.MLP_MOE,
      )
      load_balance_loss = metadata["load_balance_loss"]
      moe_bias_updates = metadata["moe_bias_updates"]
    else:
      mlp_lnx, load_balance_loss, moe_bias_updates = self.mlp_op(
          hidden_states, deterministic, pregathered_weights=pregathered_weights
      )
      layer_output = mlp_lnx + intermediate_inputs
    layer_output = self.dropout_op(layer_output, deterministic=deterministic)

    return self.post_process(layer_output, load_balance_loss, moe_bias_updates, kv_cache)

  def _handwritten_moe_layer(self, x, decoder_segment_ids, decoder_positions, deterministic):
    """Layer custom_vjp with a hand-written backward (flag: moe_handwritten_bwd).

    Wraps gather + attention + MoE so WE own the backward schedule. The forward runs the existing
    forward (gather emitted in program order BEFORE attention -> keeps the structural fwd overlap)
    and saves only {decoder_layer_input x, sharded params} as residuals -- no gathered weights are
    held (no OOM) and there is no auto-remat decision to cycle against. The backward replays the
    three pure pieces, emitting the annotated weight RE-gather adjacent to the recomputed attention
    forward (SC regather || TC splash-fwd remat), then MoE-bwd -> gather-bwd (psum_scatter ->
    FSDP-sharded weight grads, via the existing _make_cv_gather custom_vjp) -> attn-bwd (splash dkv).

    Numerics are identical to the autodiff path: the pieces compose to the same forward and each VJP
    is the kernel's own; the per-piece param cotangents (disjoint usage) sum to the full gradient.
    seg/pos are closed over in the primal/fwd (same trace, legal) and read from residuals in the bwd
    (avoids a tracer leak). Preconditions are enforced by the caller's gate (plain ring path).
    """
    graphdef, params, rest = nnx.split(self, nnx.Param, ...)
    det = deterministic  # static python bool
    seg0, pos0 = decoder_segment_ids, decoder_positions

    # `rest` (non-Param nnx state, e.g. RNG keys) holds forward-trace tracers, so it must be
    # THREADED through residuals -- never closed over in the bwd -- or it leaks across the
    # nn.scan boundary ("No constant handler for DynamicJaxprTracer"). graphdef/det are static.
    def _gather(p, rest_):
      m = nnx.merge(graphdef, p, rest_)
      return m.DeepSeekMoeBlock_0.gather_routed_weights()  # (w0, w1, wo); annotated custom_vjp gather

    def _attn(p, x_in, seg, pos, rest_):
      m = nnx.merge(graphdef, p, rest_)
      return m.self_attention_with_norm_op(x_in, seg, pos, det)  # (hidden_states, intermediate_inputs)

    def _moe(p, hidden_states, intermediate_inputs, weights, rest_):
      m = nnx.merge(graphdef, p, rest_)
      mlp_lnx, load_balance_loss, moe_bias_updates = m.mlp_op(hidden_states, det, pregathered_weights=weights)
      layer_output = m.dropout_op(mlp_lnx + intermediate_inputs, deterministic=det)
      return layer_output, load_balance_loss, moe_bias_updates

    @jax.custom_vjp
    def fused(p, x_in):
      weights = _gather(p, rest)  # gather FIRST (program-order before attention) -> forward overlap
      hidden_states, intermediate_inputs = _attn(p, x_in, seg0, pos0, rest)
      return _moe(p, hidden_states, intermediate_inputs, weights, rest)

    def fused_fwd(p, x_in):
      weights = _gather(p, rest)
      hidden_states, intermediate_inputs = _attn(p, x_in, seg0, pos0, rest)
      out = _moe(p, hidden_states, intermediate_inputs, weights, rest)
      # Residuals: sharded params + decoder_layer_input + (seg, pos, rest). NO gathered weights / activations.
      return out, (p, x_in, seg0, pos0, rest)

    def fused_bwd(res, cotangents):
      p, x_in, seg, pos, rest_ = res
      # Re-tracing the bridged layer methods happens OUTSIDE the linen forward, so detach the
      # linen module stack to avoid the qwix-fixup None.path crash (see helper docstring).
      with _detached_linen_module_stack():
        # 1) recompute attention forward + build its VJP (the splash-fwd remat; weight-independent)
        (hidden_states, intermediate_inputs), vjp_attn = jax.vjp(
            lambda pp, xx: _attn(pp, xx, seg, pos, rest_), p, x_in
        )
        # 2) OUR placed annotated weight re-gather + its VJP (bwd = tiled psum_scatter -> sharded grads)
        weights, vjp_gather = jax.vjp(lambda pp: _gather(pp, rest_), p)
        # 3) recompute MoE forward + build its VJP
        _out, vjp_moe = jax.vjp(
            lambda pp, hh, ii, ww: _moe(pp, hh, ii, ww, rest_), p, hidden_states, intermediate_inputs, weights
        )
        dp_moe, d_hidden, d_inter, d_weights = vjp_moe(cotangents)
        (dp_gather,) = vjp_gather(d_weights)
        dp_attn, dx = vjp_attn((d_hidden, d_inter))
      # disjoint per-piece param cotangents (zeros elsewhere) -> sum reconstructs the full gradient
      dp = jax.tree.map(lambda a, b, c: a + b + c, dp_attn, dp_moe, dp_gather)
      return dp, dx

    fused.defvjp(fused_fwd, fused_bwd)
    return fused(params, x)

  def mlp_op(self, x, deterministic, *args, pregathered_weights=None, **kwargs):
    mlp_lnx, load_balance_loss, moe_bias_updates = self.DeepSeekMoeBlock_0(
        x,
        intermediate_sharding=self.mlp_intermediate_sharding,
        out_sharding=self.out_sharding,
        pregathered_weights=pregathered_weights,
    )
    return self.with_logical_constraint(mlp_lnx), load_balance_loss, moe_bias_updates


DeepSeekMoELayerToLinen = nnx_wrappers.to_linen_class(
    DeepSeekMoELayer,
    base_metadata_fn=initializers.variable_to_logically_partitioned,
)
