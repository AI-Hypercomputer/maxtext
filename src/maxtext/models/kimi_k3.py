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

"""MaxText-native Kimi-K3 decoder layer (KDA/MLA attention, dense MLP/latent MoE, AttnRes).

Supported modes: train and prefill on unpacked inputs. KDA layers carry no recurrent or
conv state between calls yet, so autoregressive decode raises `NotImplementedError`.

Note on layer indexing:
`full_attn_layers` in the MaxText YAML configuration is 0-indexed (the Hugging Face
`config.json` is 1-indexed; the YAML configuration has already been converted).
"""

import functools
from typing import Optional, Sequence, Tuple

from flax import nnx
import jax
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp
from jax.sharding import Mesh

from maxtext.common.common_types import (
    AttentionType,
    Config,
    MODEL_MODE_AUTOREGRESSIVE,
    MODEL_MODE_PREFILL,
)
from maxtext.layers import (
    attention_kda,
    attention_mla,
    latent_moe,
    linears,
    quantizations,
)
from maxtext.layers.attn_res import AttnResBuffer, _apply_attn_res, commit_attn_res_block, flush_attn_res_pending
from maxtext.layers.normalizations import RMSNorm
from maxtext.models.kimi_k3_layout import KimiScanLayout, kimi_scan_layout_from_spec
from maxtext.models.kimi_k3_layout import build_kimi_layer_spec  # pylint: disable=unused-import  # re-exported
from maxtext.utils import max_utils, maxtext_utils
from maxtext.utils.sharding import (
    create_sharding,
    get_logical_axis_rules,
    maybe_shard_with_logical,
)


class KimiK3DecoderLayer(nnx.Module):
  """Kimi-K3 decoder layer with KDA/MLA attention, Latent MoE/Dense MLP, and AttnRes highway."""

  def __init__(
      self,
      config: Config,
      model_mode: str,
      mesh: Mesh,
      rngs: nnx.Rngs,
      quant: Optional[quantizations.AqtQuantization] = None,
      *,
      layer_idx: int,
      is_linear_attn: bool,
      is_moe: bool,
  ):
    # `layer_idx`, `is_linear_attn` and `is_moe` decide the layer's structure (attention type,
    # MLP type, AttnRes block boundaries), so they are required: there is no safe default.
    if layer_idx < 0:
      raise ValueError(f"KimiK3DecoderLayer needs a non-negative layer_idx, got {layer_idx}.")
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
      self.self_attention = attention_kda.KimiDeltaAttention(config=config, mesh=mesh, rngs=rngs)
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
          # Every Kimi-K3 MLA layer is output-gated; MLA ORs this with `config.mla_use_output_gate`.
          mla_use_output_gate=True,
          # Kimi-K3 sets `mla_use_nope: true`: the rope slice is projected but the
          # rotary transform is never applied.
          is_nope_layer=True,
          # HF `KimiMLAAttention` builds `q_a_layernorm` / `kv_a_layernorm` with `KimiRMSNorm`'s
          # default `eps=1e-6` rather than `config.rms_norm_eps` (`1e-5`).
          lora_norm_epsilon=1e-6,
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
          config=self.config,
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
      *,
      layer_idx=None,
  ):
    """Runs one layer.

    Args:
      block_residual: AttnRes highway state. Either the dynamically grown [B, S, K, D]
        block stack (or None) of the unscanned stack, or an `AttnResBuffer` (fixed slot
        count + committed count) for the scanned stack.
      layer_idx: Optional global layer index overriding `self.layer_idx`; may be a traced
        int32 scalar inside lax.scan (only with an `AttnResBuffer`).

    Returns:
      (prefix_sum, block_residual, kv_cache), with `block_residual` of the same kind as given.
    """
    x = self.with_logical_constraint(inputs)
    x = checkpoint_name(x, "decoder_layer_input")
    prefix_sum = x  # [B, S, D]
    hidden_states = x

    use_buffer = isinstance(block_residual, AttnResBuffer)
    if layer_idx is None:
      layer_idx = self.layer_idx
    elif not use_buffer and not isinstance(layer_idx, int):
      raise ValueError("A traced layer_idx requires the static AttnResBuffer highway (scanned stack).")
    # Python bool for a static index, traced bool scalar inside the scanned block.
    is_commit = layer_idx % self.attn_res_block_size == 0

    def pool(prefix, blocks, proj, norm):
      if use_buffer:
        return _apply_attn_res(
            prefix_sum=prefix,
            block_residual=blocks.blocks,
            proj_weight=proj.kernel.value,
            norm_weight=norm.scale.value,
            epsilon=self.eps,
            num_blocks=blocks.blocks.shape[-2],
            num_committed=blocks.count,
            pending=blocks.pending,
            has_pending=blocks.has_pending,
        )
      return _apply_attn_res(
          prefix_sum=prefix,
          block_residual=blocks,
          proj_weight=proj.kernel.value,
          norm_weight=norm.scale.value,
          epsilon=self.eps,
          num_blocks=None,  # dynamic-concat path; unscanned only
      )

    # STEP 1: pre-attention AttnRes pooling (only if blocks exist; the masked pooling of an
    # empty buffer is the identity, so the buffer path needs no branch).
    if use_buffer or (block_residual is not None and block_residual.shape[-2] > 0):
      hidden_states = pool(prefix_sum, block_residual, self.self_attention_res_proj, self.self_attention_res_norm)

    # STEP 2: block-boundary checkpoint
    if use_buffer:
      block_residual = commit_attn_res_block(block_residual, prefix_sum, is_commit)
    elif is_commit:
      new_block = prefix_sum[..., None, :]  # [B, S, 1, D]
      block_residual = (
          new_block
          if (block_residual is None or block_residual.shape[-2] == 0)
          else jnp.concatenate([block_residual, new_block], axis=-2)
      )

    # STEP 3: attention
    normed_attn_in = self.pre_self_attention_layer_norm(hidden_states)
    if self.is_linear_attn:
      if model_mode == MODEL_MODE_AUTOREGRESSIVE:
        raise NotImplementedError(
            "Kimi-K3 autoregressive decode is not supported yet: KDA recurrent/conv state is not "
            "cached between steps, so each decode step would silently restart from zero state."
        )
      # Only the tokamax KDA kernel honors segment ids (kda_max_num_segments > 0); otherwise packed
      # inputs would leak state across documents, which the config validator rejects.
      attn_out, _, _ = self.self_attention(normed_attn_in, model_mode=model_mode, decoder_segment_ids=decoder_segment_ids)
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

    # STEP 4: a commit restarts the prefix sum at the attention output.
    if isinstance(is_commit, bool):
      prefix_sum = attn_out if is_commit else prefix_sum + attn_out
    else:
      prefix_sum = jnp.where(is_commit, attn_out, prefix_sum + attn_out)

    # STEP 5: pre-MLP AttnRes pooling (ALWAYS, using mlp_res_* weights)
    pooled = pool(prefix_sum, block_residual, self.mlp_res_proj, self.mlp_res_norm)

    # STEP 6
    normed_mlp_in = self.post_self_attention_layer_norm(pooled)
    mlp_out = self.mlp(normed_mlp_in)
    mlp_out = self.with_logical_constraint(mlp_out)

    # STEP 7
    prefix_sum = prefix_sum + mlp_out
    return prefix_sum, block_residual, kv_cache


def kimi_scan_layout(config: Config) -> KimiScanLayout:
  """Splits the Kimi-K3 stack into an unrolled prefix, scanned blocks and an unrolled suffix.

  The scanned region starts at the first cycle boundary at or after the dense-MLP layers
  (so every scanned layer is MoE) and spans as many consecutive whole cycles of
  `inhomogeneous_layer_cycle_interval` layers as repeat the first cycle's (KDA/MLA,
  dense/MoE) pattern. For the released 93-layer model: prefix 0..3, 22 blocks of
  [KDA, KDA, KDA, MLA] over layers 4..91, suffix [92]. Checkpoint conversion uses the same
  helper (`kimi_k3_layout.kimi_scan_layout_from_spec`).

  Args:
    config: Kimi-K3 config.

  Returns:
    A `KimiScanLayout` (num_prefix, num_blocks, suffix_start, pattern), where `pattern`
    holds one (is_linear_attn, is_moe) per layer of a block.

  Raises:
    ValueError: If the layers after the periodic region still contain a whole cycle,
      i.e. the layer pattern is not periodic.
  """
  return kimi_scan_layout_from_spec(
      config.num_decoder_layers,
      config.full_attn_layers,
      config.first_num_dense_layers,
      config.inhomogeneous_layer_cycle_interval,
  )


class KimiK3ScannableBlock(nnx.Module):
  """One cycle of Kimi-K3 layers (e.g. [KDA, KDA, KDA, MLA], all latent MoE) for lax.scan.

  The scan carry is (hidden, AttnResBuffer, layer_idx), with `layer_idx` the traced global
  index of the block's first layer, so the sub-layers decide AttnRes block commits from it.
  With `apply_internal_remat`, each sub-layer is rematerialized on its own (the outer scan
  then skips block-level remat), so only one sub-layer's activations are live at a time.
  """

  def __init__(  # pylint: disable=too-many-positional-arguments  # `_create_scanned_layers` signature
      self,
      config: Config,
      model_mode: str,
      mesh: Mesh,
      rngs: nnx.Rngs,
      quant: Optional[quantizations.AqtQuantization] = None,
      *,
      layer_pattern: Sequence[Tuple[bool, bool]],
      layer_idx_offset: int,
      remat_policy_fn=None,
      apply_internal_remat: bool = False,
  ):
    self.config = config
    self.num_of_layers = len(layer_pattern)
    self.remat_policy_fn = remat_policy_fn
    self.apply_internal_remat = apply_internal_remat
    for j, (is_linear_attn, is_moe) in enumerate(layer_pattern):
      # The static layer_idx only fixes the sub-layer's structure; AttnRes commits use the
      # traced index passed at call time.
      layer = KimiK3DecoderLayer(
          config=config,
          model_mode=model_mode,
          mesh=mesh,
          rngs=rngs,
          quant=quant,
          layer_idx=layer_idx_offset + j,
          is_linear_attn=is_linear_attn,
          is_moe=is_moe,
      )
      setattr(self, f"layers_{j}", layer)

  def __call__(  # pylint: disable=too-many-positional-arguments  # scanned-layer call signature
      self,
      carry,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      previous_chunk=None,
      slot=None,
      attention_metadata=None,
  ):
    y, attn_res_state, layer_idx = carry
    n = self.num_of_layers
    block_size = self.config.attn_res_block_size
    remat = self.apply_internal_remat and self.config.remat_policy != "none"
    layer_kwargs = {
        "decoder_segment_ids": decoder_segment_ids,
        "decoder_positions": decoder_positions,
        "deterministic": deterministic,
        "model_mode": model_mode,
        "previous_chunk": previous_chunk,
        "slot": slot,
        "attention_metadata": attention_metadata,
    }
    # With at most one AttnRes commit per cycle, the committed block is kept in a `pending`
    # slot until the end of the cycle, so every sub-layer sees the *same* `blocks` array and
    # (with remat) the scan saves it for backward once per cycle instead of once per sub-layer.
    # The pending block is the input of the committing sub-layer, which is saved anyway.
    use_pending = block_size >= n

    def run_sublayer(j, layer, state_in, y_in, attn_res_in, prev_inputs, idx0):
      merged = layer if state_in is None else nnx.merge(layer, state_in)
      if use_pending:
        commit_pos = jnp.mod(-idx0, block_size)  # sub-layer that commits in this cycle (if < n)
        pending = y_in
        for k, prev in enumerate(prev_inputs):
          pending = jnp.where(commit_pos == k, prev, pending)
        attn_res_in = attn_res_in._replace(pending=pending, has_pending=commit_pos < j)
      out_y, out_attn_res, _ = merged(y_in, block_residual=attn_res_in, layer_idx=idx0 + j, **layer_kwargs)
      if use_pending:
        # Only the (small) pending slot leaves the sub-layer; `blocks` is unchanged.
        out_attn_res = (out_attn_res.pending, out_attn_res.has_pending)
      new_state = None if state_in is None else nnx.state(merged)
      return (out_y, out_attn_res), new_state

    sub_inputs = []
    out_attn_res = None
    for j in range(n):
      layer = getattr(self, f"layers_{j}")
      sub_inputs.append(y)
      prev_inputs = tuple(sub_inputs[:-1]) if use_pending else ()
      if remat:
        graphdef, state = nnx.split(layer)
        run_layer_remat = jax.checkpoint(
            functools.partial(run_sublayer, j, graphdef),
            policy=self.remat_policy_fn,
            prevent_cse=maxtext_utils.should_prevent_cse_in_remat(self.config),
        )
        # pylint: disable-next=too-many-function-args  # jax.checkpoint(partial(...)) confuses pylint
        (y, out_attn_res), new_state = run_layer_remat(state, y, attn_res_state, prev_inputs, layer_idx)
        nnx.update(layer, new_state)
      else:
        (y, out_attn_res), _ = run_sublayer(j, layer, None, y, attn_res_state, prev_inputs, layer_idx)
      if not use_pending:
        attn_res_state = out_attn_res
    if use_pending:
      pending, has_pending = out_attn_res
      attn_res_state = flush_attn_res_pending(attn_res_state._replace(pending=pending, has_pending=has_pending))
    return (y, attn_res_state, layer_idx + n), None
