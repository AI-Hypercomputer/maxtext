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

"""DeepSeek-V4 model definition."""

from typing import Any, Optional

from flax import nnx
from jax.sharding import Mesh

from maxtext.common.common_types import Config, AttentionType
from maxtext.common.common_types import HyperConnectionType
from maxtext.layers import attention_compressed
from maxtext.layers import initializers
from maxtext.layers import moe
from maxtext.layers import quantizations
from maxtext.models import deepseek
from maxtext.utils.sharding import with_logical_constraint
from jax.ad_checkpoint import checkpoint_name
import jax
import jax.numpy as jnp
from jax.experimental import xla_metadata


def _run_deepseek4_layer_remat(
    layer: nnx.Module,
    y: jax.Array,
    *,
    config: Config,
    policy: Any = None,
    prevent_cse: Optional[bool] = None,
    **layer_kwargs,
):
  """Runs one DeepSeek4DecoderLayer under jax.checkpoint inside a length-1 scan.

  The trip-count-one scan boundary keeps only one layer's backward working set
  live at a time, as in Gemma4ScannableBlock._scan_global_layer. Params go in as
  scan xs: carrying them, as Gemma4 does, multiplies the DeepSeek-V4 train-step
  temporaries in TPU AOT. All other state, including RNG counters and variables
  the layer creates during the call (sown aux losses, `indexer_losses`), goes
  through xs/ys and is written back. `policy` is the decoder's resolved remat
  policy (None means full remat).
  """
  params_filter = nnx.All(nnx.Param, nnx.Not(nnx.LoRAParam))
  graphdef, params, other = nnx.split(layer, params_filter, ...)
  params_xs = jax.tree.map(lambda x: x[None], params)
  other_xs = jax.tree.map(lambda x: x[None], other)

  def run_single_layer(carry, xs_slice):
    params_slice, other_slice = xs_slice
    merged_layer = nnx.merge(graphdef, params_slice, other_slice)
    out_y, _ = merged_layer(carry, **layer_kwargs)
    _, _, new_other = nnx.split(merged_layer, params_filter, ...)
    return out_y, new_other

  if prevent_cse is None:
    prevent_cse = not config.scan_layers and not (
        config.gradient_accumulation_steps > 1 and config.hardware in ("gpu", "gpu_multiprocess")
    )
  run_single_layer = jax.checkpoint(run_single_layer, policy=policy, prevent_cse=prevent_cse)
  with xla_metadata.set_xla_metadata(**{"skip-simplify-while-loops_trip-count-one": "true"}):
    y_out, stacked_other = jax.lax.scan(run_single_layer, y, (params_xs, other_xs), length=1)
  nnx.update(layer, jax.tree.map(lambda x: x[0], stacked_other))
  return y_out


class DeepSeek4DecoderLayer(deepseek.DeepSeekGenericLayer):
  """DeepSeek-V4 specific decoder layer.

  Note: V4 does not utilize purely dense layers in the initial transformer blocks.
  Every layer is a Sparse MoE layer (which internally contains shared dense experts).

  Args:
    config: Configuration for the model.
    model_mode: The mode of the model (e.g. 'train', 'inference').
    mesh: JAX sharding mesh.
    rngs: NNX Rngs.
    quant: Optional AQT quantization config.
    layer_idx: The index of the layer.
    compress_ratio: DeepSeek V4 specific parameter defining the KV cache compression
      ratio. Expected values are 0 (no compression, sliding window), 4 (CSA), or 128 (HCA).
    is_hash_routing: DeepSeek V4 specific parameter defining if this layer uses
      static deterministic hash routing (used in prefix layers).
  """

  def __init__(
      self,
      config: Config,
      model_mode: str,
      mesh: Mesh,
      rngs: nnx.Rngs,
      quant: Optional[quantizations.AqtQuantization] = None,
      layer_idx: int = -1,
      compress_ratio: Optional[int] = None,
      is_hash_routing: Optional[bool] = None,
  ) -> None:
    super().__init__(
        config=config,
        model_mode=model_mode,
        mesh=mesh,
        rngs=rngs,
        quant=quant,
        layer_idx=layer_idx,
    )

    # DeepSeek V4 applies Hash Routing to the first `config.first_num_hash_layers` layers.
    # For the unscannable prefix layers, we can safely determine this using `layer_idx`.
    # However, for layers inside scanned blocks, `layer_idx` is a dynamic JAX tracer
    # and cannot be evaluated as a boolean condition. Since all scannable layers occur
    # after the hash-routed prefix, the scannable block explicitly passes
    # `is_hash_routing=False` to safely bypass this check.
    if is_hash_routing is None:
      is_hash_routing = layer_idx < config.first_num_hash_layers
    self.mlp = moe.RoutedAndSharedMoE(
        config=self.config,
        mesh=self.mesh,
        kernel_init=initializers.nd_dense_init(self.config.dense_init_scale, "fan_in", "truncated_normal"),
        kernel_axes=("embed", None),
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        quant=quant,
        is_hash_routing=is_hash_routing,
        rngs=rngs,
    )

    if compress_ratio is None:
      compress_ratio = config.compress_ratios[layer_idx]

    self.self_attention = attention_compressed.CompressedAttention(
        config=self.config,
        compress_ratio=compress_ratio,
        num_query_heads=self.config.num_query_heads,
        num_kv_heads=self.config.num_kv_heads,
        head_dim=self.config.head_dim,
        max_target_length=self.config.max_target_length,
        max_prefill_predict_length=self.config.max_prefill_predict_length,
        attention_kernel=self.config.attention,
        attention_type=AttentionType(self.config.attention_type),
        inputs_q_shape=self.dummy_inputs_shape,
        inputs_kv_shape=self.dummy_inputs_shape,
        mesh=self.mesh,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        dropout_rate=self.config.dropout_rate,
        sliding_window_size=self.config.sliding_window_size,
        q_lora_rank=self.config.q_lora_rank,
        name=f"compressed_attention_layer_{layer_idx}",
        quant=quant,
        kv_quant=quantizations.configure_kv_quant(config),
        model_mode=model_mode,
        rngs=rngs,
    )

  # pylint: disable=arguments-differ
  def mlp_op(self, inputs, deterministic, *args, **kwargs):
    input_ids = kwargs.get("input_ids")
    mlp_lnx, load_balance_loss, moe_bias_updates = self.mlp(
        inputs=inputs,
        input_ids=input_ids,
        token_mask=kwargs.get("token_mask"),
    )
    return self.with_logical_constraint(mlp_lnx), load_balance_loss, moe_bias_updates

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
    if isinstance(inputs, tuple):
      inputs = inputs[0]

    x = self.with_logical_constraint(inputs)
    x = checkpoint_name(x, "decoder_layer_input")

    _, intermediate_inputs = self.self_attention_with_norm_op(
        x,
        decoder_segment_ids,
        decoder_positions,
        deterministic,
        model_mode,
        previous_chunk,
        slot,
    )

    layer_output, metadata = self.mhc_mlp(
        self.post_attention_norm_op,
        self.mlp_op,
        x=intermediate_inputs,
        mhc_type=HyperConnectionType.MLP_MOE,
        deterministic=deterministic,
        input_ids=decoder_input_tokens,
        token_mask=moe.routing_token_mask(self.config, decoder_segment_ids, model_mode),
    )
    load_balance_loss = metadata.get("load_balance_loss", None)
    moe_bias_updates = metadata.get("moe_bias_updates", None)

    layer_output = self.dropout_op(layer_output, deterministic=deterministic)
    return self.post_process(layer_output, load_balance_loss, moe_bias_updates, kv_cache)

  # ---------------------------------------------------------------------------
  # Generalized Batch-Split layer adapter (used only when config.enable_batch_split)
  # ---------------------------------------------------------------------------

  def batch_split_supported(self) -> bool:
    """Returns True if this DeepSeek-V4 layer configuration supports `batch_split`."""
    cfg = self.config
    return (
        self.is_mhc_enabled
        and not self.mlp.is_hash_routing
        and cfg.sparse_matmul
        and cfg.use_ring_of_experts
        and not cfg.te_moe_block
        and not self.is_engram_enabled
        and not getattr(cfg, "te_mhc", False)
        and not (cfg.enable_mhc_lite and getattr(cfg, "use_mhc_pallas_kernel", False))
        and not (cfg.use_indexer and cfg.indexer_loss_scaling_factor > 0.0)
        and not getattr(cfg, "moe_use_fused_gmm3d", False)
        and cfg.num_moe_emb_chunks <= 0
        and not getattr(cfg, "record_internal_nn_metrics", False)
        and cfg.dropout_rate == 0.0
    )

  def batch_split_attn_and_route(
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
    """Phase 1/2 TC compute: CompressedAttention (HCA/CSA) + mHC pre/res/post mappings + router gate."""
    del kv_cache, attention_metadata
    if isinstance(inputs, tuple):
      inputs = inputs[0]
    x = self.with_logical_constraint(inputs)
    _, intermediate_inputs = self.self_attention_with_norm_op(
        x,
        decoder_segment_ids,
        decoder_positions,
        deterministic,
        model_mode,
        previous_chunk,
        slot,
    )

    mhc_mod = self.mhc_mlp
    b, s, k, d = intermediate_inputs.shape
    with jax.named_scope("mhc_norm"):
      norm_x = mhc_mod.mhc_norm(jnp.reshape(intermediate_inputs, (b, s, k * d)))
    pre_alpha = jnp.asarray(mhc_mod.pre_alpha[...], mhc_mod.dtype)
    post_alpha = jnp.asarray(mhc_mod.post_alpha[...], mhc_mod.dtype)
    res_alpha = jnp.asarray(mhc_mod.res_alpha[...], mhc_mod.dtype)
    alpha_concat = jnp.concatenate([pre_alpha, post_alpha, res_alpha], axis=-1)
    h_concat = jnp.einsum("bsm,mn -> bsn", norm_x, alpha_concat, precision=mhc_mod.matmul_precision)
    h_pre = h_concat[..., :k]
    h_post = h_concat[..., k : 2 * k]
    h_res = h_concat[..., 2 * k :]

    pre_mapping = mhc_mod.mapping(
        h_pre,
        mhc_mod.pre_alpha_scale[...],
        mhc_mod.pre_beta[...],
        1.0,
        eps=1e-6,
    )
    layer_input = jnp.einsum("bsk,bskd->bsd", pre_mapping, intermediate_inputs, precision=mhc_mod.matmul_precision)
    hidden_states = self.post_attention_norm_op(layer_input)

    post_mapping = mhc_mod.mapping(
        h_post,
        mhc_mod.post_alpha_scale[...],
        mhc_mod.post_beta[...],
        2.0,
    )
    res_mapping = mhc_mod.res_mapping(h_res)
    res_out = jnp.einsum("bskm,bskd->bsmd", res_mapping, intermediate_inputs, precision=mhc_mod.matmul_precision)
    h_skip = (res_out, post_mapping)

    moe_in, route_in, seq_lb_loss = self.mlp.routed_moe.batch_split_route(
        hidden_states,
        input_ids=decoder_input_tokens,
    )
    return h_skip, moe_in, route_in, seq_lb_loss

  def batch_split_dispatch(self, moe_in, route_in):
    """Weight-free EP All-Gather of microbatch tokens and routing metadata on SC0."""
    return self.mlp.routed_moe.batch_split_dispatch(moe_in, route_in)

  def batch_split_dispatch_bwd(self, moe_in, route_in, g_x_val_ag, g_route_ag_diff):
    """Weight-free backward of `batch_split_dispatch` (EP Reduce-Scatter on SC0)."""
    return self.mlp.routed_moe.batch_split_dispatch_bwd(moe_in, route_in, g_x_val_ag, g_route_ag_diff)

  def batch_split_experts(
      self, moe_in_ref, x_val_ag, x_scale_ag, route_ag, decoder_input_tokens=None, return_wgrad_hooks: bool = False
  ):
    """Local expert permute + GMMs + unpermute without EP dispatch/combine."""
    return self.mlp.routed_moe.batch_split_experts(
        moe_in_ref,
        x_val_ag,
        x_scale_ag,
        route_ag,
        input_ids=decoder_input_tokens,
        return_wgrad_hooks=return_wgrad_hooks,
    )

  def batch_split_shared_expert(self, moe_in):
    """Shared expert MLP on local microbatch tokens."""
    shared_out = self.mlp.shared_experts(
        moe_in,
        intermediate_sharding=self.mlp_intermediate_sharding,
        out_sharding=self.out_sharding,
    )
    return self.with_logical_constraint(shared_out)

  def batch_split_combine(self, y_partial, moe_in_ref):
    """Weight-free EP Reduce-Scatter of routed expert partial outputs on SC0."""
    routed_out = self.mlp.routed_moe.batch_split_combine(y_partial, moe_in_ref)
    return self.with_logical_constraint(routed_out)

  def batch_split_post_combine(self, mlp_sum, h_skip):
    """Weight-free mHC stream expansion and residual combination after routed + shared experts."""
    res_out, post_mapping = h_skip
    mlp_out = self.with_logical_constraint(mlp_sum)
    post_out = jnp.expand_dims(mlp_out, axis=2) * jnp.expand_dims(post_mapping, axis=3)
    return self.with_logical_constraint(res_out + post_out)

  def batch_split_combine_bwd(self, g_out, moe_in_ref, *, defer_dequant: bool = False):
    """Weight-free backward of `batch_split_combine` (EP All-Gather on SC0)."""
    return self.mlp.routed_moe.batch_split_combine_bwd(g_out, moe_in_ref, defer_dequant=defer_dequant)

  def batch_split_finalize_bias_updates(self, summed_counts):
    """Finalizes microbatch-summed expert counts into routed bias updates."""
    return self.mlp.routed_moe.batch_split_finalize_bias_updates(summed_counts)


class DeepSeek4ScannableBlock(nnx.Module):
  """A scannable block containing exactly two DeepSeek V4 layers (HCA and CSA).

  DeepSeek V4 layers alternate `compress_ratio=128` (HCA) and `compress_ratio=4` (CSA)
  throughout the middle of the network. This block encapsulates one full `[128, 4]`
  cycle so it can be perfectly scanned using `jax.lax.scan`.
  """

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      rngs: nnx.Rngs,
      quant: None | quantizations.AqtQuantization = None,
      remat_policy_fn: Any = None,
      prevent_cse_in_remat: Optional[bool] = None,
  ):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.quant = quant
    self.rngs = rngs
    # Resolved remat policy for the per-layer remat in train mode (None means full remat).
    self.remat_policy_fn = remat_policy_fn
    self.prevent_cse_in_remat = prevent_cse_in_remat

    # Layer 0 in the block: HCA (compress_ratio=128) with Standard MoE (is_hash_routing=False)
    self.layers_0 = DeepSeek4DecoderLayer(
        config=self.config,
        mesh=self.mesh,
        model_mode=self.model_mode,
        rngs=self.rngs,
        quant=self.quant,
        compress_ratio=128,
        is_hash_routing=False,
    )

    # Layer 1 in the block: CSA (compress_ratio=4) with Standard MoE (is_hash_routing=False)
    self.layers_1 = DeepSeek4DecoderLayer(
        config=self.config,
        mesh=self.mesh,
        model_mode=self.model_mode,
        rngs=self.rngs,
        quant=self.quant,
        compress_ratio=4,
        is_hash_routing=False,
    )

  def __call__(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      slot=None,
      previous_chunk=None,
      attention_metadata=None,
      kv_cache=None,
      decoder_input_tokens=None,
  ):
    inputs = with_logical_constraint(inputs, ("activation_batch", "activation_norm_length", "activation_embed"))
    inputs = checkpoint_name(inputs, "decoder_layer_input")
    y = inputs

    override = getattr(self, "remat_policy_override", None)
    remat_policy = self.config.remat_policy if override is None else override
    if model_mode == "train" and bool(remat_policy) and remat_policy != "none":
      call_kwargs = dict(  # pylint: disable=use-dict-literal
          decoder_segment_ids=decoder_segment_ids,
          decoder_positions=decoder_positions,
          deterministic=deterministic,
          model_mode=model_mode,
          previous_chunk=previous_chunk,
          slot=slot,
          kv_cache=kv_cache,
          attention_metadata=attention_metadata,
          decoder_input_tokens=decoder_input_tokens,
      )
      if override is None:
        policy_fn = self.remat_policy_fn
      else:
        policy_fn = getattr(self, "scan_safe_remat_policy_fn_override", None)
      remat_kwargs = {"config": self.config, "policy": policy_fn, "prevent_cse": self.prevent_cse_in_remat}
      y = _run_deepseek4_layer_remat(self.layers_0, y, **remat_kwargs, **call_kwargs)
      y = _run_deepseek4_layer_remat(self.layers_1, y, **remat_kwargs, **call_kwargs)
    else:
      y, _ = self.layers_0(
          y,
          decoder_segment_ids,
          decoder_positions,
          deterministic,
          model_mode,
          previous_chunk=previous_chunk,
          slot=slot,
          kv_cache=kv_cache,
          attention_metadata=attention_metadata,
          decoder_input_tokens=decoder_input_tokens,
      )
      y, _ = self.layers_1(
          y,
          decoder_segment_ids,
          decoder_positions,
          deterministic,
          model_mode,
          previous_chunk=previous_chunk,
          slot=slot,
          kv_cache=kv_cache,
          attention_metadata=attention_metadata,
          decoder_input_tokens=decoder_input_tokens,
      )

    return y, None

  def batch_split_sublayer_names(self) -> tuple[str, str]:
    """Sub-layers of this block in execution order for `batch_split_inhomogeneous`."""
    return ("layers_0", "layers_1")
