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

"""MaxText-native Kimi-K3 decoder layer built on the Track A layer modules.

Note on layer indexing:
`full_attn_layers` in the MaxText YAML configuration is 0-indexed (the Hugging Face
`config.json` is 1-indexed; the YAML configuration has already been converted).
"""

from typing import Optional, Sequence, Tuple

from flax import nnx
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp
from jax.sharding import Mesh

from maxtext.common.common_types import (
    AttentionType,
    Config,
    MODEL_MODE_PREFILL,
)
from maxtext.layers import (
    attention_kda,
    attention_mla,
    initializers,
    latent_moe,
    linears,
    nnx_wrappers,
    quantizations,
)
from maxtext.layers.attn_res import _apply_attn_res
from maxtext.layers.normalizations import RMSNorm
from maxtext.utils import max_utils
from maxtext.utils.sharding import (
    create_sharding,
    get_logical_axis_rules,
    maybe_shard_with_logical,
)


def build_kimi_layer_spec(
    num_decoder_layers: int,
    full_attn_layers: Sequence[int],
    first_num_dense_layers: int,
) -> list[Tuple[bool, bool]]:
  """Returns a list of (is_linear_attn, is_moe) per layer index."""
  full_attn_set = set(full_attn_layers)
  spec = []
  for idx in range(num_decoder_layers):
    is_linear_attn = idx not in full_attn_set
    is_moe = idx >= first_num_dense_layers
    spec.append((is_linear_attn, is_moe))
  return spec


class KimiK3DecoderLayer(nnx.Module):
  """Kimi-K3 decoder layer with KDA/MLA attention, Latent MoE/Dense MLP, and AttnRes highway."""

  def __init__(
      self,
      config: Config,
      model_mode: str,
      mesh: Mesh,
      rngs: nnx.Rngs,
      quant: Optional[quantizations.AqtQuantization] = None,
      layer_idx: int = -1,
      is_linear_attn: bool = True,
      is_moe: bool = True,
  ):
    self.config = config
    self.model_mode = model_mode
    self.mesh = mesh
    self.quant = quant
    self.rngs = rngs
    self.layer_idx = layer_idx
    self.is_linear_attn = is_linear_attn
    self.is_moe = is_moe

    self.eps = config.normalization_layer_epsilon
    self.attn_res_block_size = config.attn_res_block_size

    batch_size, sequence_length = max_utils.get_batch_seq_len_for_mode(self.config, self.model_mode)
    self.dummy_inputs_shape = (batch_size, sequence_length, self.config.emb_dim)

    self.out_sharding = create_sharding(self.mesh, self.logical_axis_names, rules=get_logical_axis_rules())

    self.pre_self_attention_layer_norm = RMSNorm(
        num_features=self.config.emb_dim,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        kernel_axes=("norm",),
        epsilon=self.config.normalization_layer_epsilon,
        rngs=rngs,
    )

    self.post_self_attention_layer_norm = RMSNorm(
        num_features=self.config.emb_dim,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        kernel_axes=("norm",),
        epsilon=self.config.normalization_layer_epsilon,
        rngs=rngs,
    )

    self.self_attention_res_norm = RMSNorm(
        num_features=self.config.emb_dim,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        kernel_axes=("norm",),
        epsilon=self.config.normalization_layer_epsilon,
        rngs=rngs,
    )

    self.self_attention_res_proj = linears.DenseGeneral(
        in_features_shape=self.config.emb_dim,
        out_features_shape=(1,),
        use_bias=False,
        kernel_axes=("embed", None),
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        rngs=rngs,
    )

    self.mlp_res_norm = RMSNorm(
        num_features=self.config.emb_dim,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        kernel_axes=("norm",),
        epsilon=self.config.normalization_layer_epsilon,
        rngs=rngs,
    )

    self.mlp_res_proj = linears.DenseGeneral(
        in_features_shape=self.config.emb_dim,
        out_features_shape=(1,),
        use_bias=False,
        kernel_axes=("embed", None),
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        rngs=rngs,
    )

    if is_linear_attn:
      self.self_attention = attention_kda.KimiDeltaAttention(config=config, rngs=rngs)
    else:
      self.self_attention = attention_mla.MLA(
          config=self.config,
          num_query_heads=self.config.num_query_heads,
          num_kv_heads=self.config.num_kv_heads,
          head_dim=self.config.head_dim,
          max_target_length=self.config.max_target_length,
          max_prefill_predict_length=self.config.max_prefill_predict_length,
          attention_kernel=self.config.attention,
          attention_type=AttentionType(self.config.attention_type),
          inputs_q_shape=self.dummy_inputs_shape,
          inputs_kv_shape=self.dummy_inputs_shape,
          mesh=mesh,
          dtype=self.config.dtype,
          weight_dtype=self.config.weight_dtype,
          dropout_rate=self.config.dropout_rate,
          name="self_attention",
          quant=quant,
          kv_quant=quantizations.configure_kv_quant(self.config),
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
          mla_use_output_gate=True,
          # Kimi-K3 sets `mla_use_nope: true`: the rope slice is projected but the
          # rotary transform is never applied.
          is_nope_layer=True,
      )

    if is_moe:
      self.mlp = latent_moe.KimiLatentMoEBlock(
          hidden_size=self.config.emb_dim,
          num_experts=self.config.num_experts,
          top_k=self.config.num_experts_per_tok,
          routed_expert_hidden_size=self.config.routed_expert_hidden_size,
          moe_intermediate_size=self.config.moe_intermediate_size,
          num_shared_experts=self.config.num_shared_experts,
          shared_intermediate_size=self.config.shared_intermediate_size,
          routed_scaling_factor=self.config.routed_scaling_factor,
          rms_norm_eps=self.config.normalization_layer_epsilon,
          dtype=self.config.dtype,
          weight_dtype=self.config.weight_dtype,
          quant=quant,
          shard_mode=self.config.shard_mode,
          matmul_precision=self.config.matmul_precision,
          mesh=mesh,
          rngs=rngs,
      )
    else:
      self.mlp = latent_moe.KimiDenseMLP(
          in_features=self.config.emb_dim,
          intermediate_dim=self.config.mlp_dim,
          dtype=self.config.dtype,
          weight_dtype=self.config.weight_dtype,
          quant=quant,
          shard_mode=self.config.shard_mode,
          matmul_precision=self.config.matmul_precision,
          mesh=mesh,
          rngs=rngs,
      )

  @property
  def logical_axis_names(self):
    """Generate logical names for activations generally."""
    length_name = "prefill_activation_norm_length" if self.model_mode == MODEL_MODE_PREFILL else "activation_norm_length"
    axis_names = ["activation_batch", length_name, "activation_embed"]
    return axis_names

  def with_logical_constraint(self, x):
    return maybe_shard_with_logical(
        x,
        logical_axes=self.logical_axis_names,
        mesh=self.mesh,
        shard_mode=self.config.shard_mode,
        debug_sharding=self.config.debug_sharding,
        extra_stack_level=1,
        rules=get_logical_axis_rules(),
    )

  def __call__(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      block_residual=None,
      previous_chunk=None,
      slot=None,
      kv_cache=None,
      attention_metadata=None,
  ):
    x = self.with_logical_constraint(inputs)
    x = checkpoint_name(x, "decoder_layer_input")
    prefix_sum = x  # [B, S, D]
    hidden_states = x

    # STEP 1: pre-attention AttnRes pooling (only if blocks exist)
    if block_residual is not None and block_residual.shape[-2] > 0:
      hidden_states = _apply_attn_res(
          prefix_sum=prefix_sum,
          block_residual=block_residual,
          proj_weight=self.self_attention_res_proj.kernel.value,
          norm_weight=self.self_attention_res_norm.scale.value,
          epsilon=self.eps,
          num_blocks=None,  # dynamic-concat path; unscanned only
      )

    # STEP 2: block-boundary checkpoint
    if self.layer_idx % self.attn_res_block_size == 0:
      new_block = prefix_sum[..., None, :]  # [B, S, 1, D]
      block_residual = (
          new_block
          if (block_residual is None or block_residual.shape[-2] == 0)
          else jnp.concatenate([block_residual, new_block], axis=-2)
      )
      prefix_sum = None

    # STEP 3: attention
    normed_attn_in = self.pre_self_attention_layer_norm(hidden_states)
    if self.is_linear_attn:
      attn_out, _, _ = self.self_attention(normed_attn_in, model_mode=model_mode)
    else:
      attn_out, kv_cache = self.self_attention(
          normed_attn_in,
          normed_attn_in,
          decoder_positions,
          decoder_segment_ids=decoder_segment_ids,
          deterministic=deterministic,
          model_mode=model_mode,
          out_sharding=self.out_sharding,
          previous_chunk=previous_chunk,
          slot=slot,
      )
    attn_out = self.with_logical_constraint(attn_out)

    # STEP 4
    prefix_sum = attn_out if prefix_sum is None else prefix_sum + attn_out

    # STEP 5: pre-MLP AttnRes pooling (ALWAYS, using mlp_res_* weights)
    pooled = _apply_attn_res(
        prefix_sum=prefix_sum,
        block_residual=block_residual,
        proj_weight=self.mlp_res_proj.kernel.value,
        norm_weight=self.mlp_res_norm.scale.value,
        epsilon=self.eps,
        num_blocks=None,
    )

    # STEP 6
    normed_mlp_in = self.post_self_attention_layer_norm(pooled)
    mlp_out = self.mlp(normed_mlp_in)
    mlp_out = self.with_logical_constraint(mlp_out)

    # STEP 7
    prefix_sum = mlp_out if prefix_sum is None else prefix_sum + mlp_out
    return prefix_sum, block_residual, kv_cache


KimiK3DecoderLayerToLinen = nnx_wrappers.to_linen_class(
    KimiK3DecoderLayer,
    base_metadata_fn=initializers.variable_to_logically_partitioned,
)
