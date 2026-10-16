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
  graphdef, params, other = nnx.split(layer, nnx.Param, ...)
  params_xs = jax.tree.map(lambda x: x[None], params)
  other_xs = jax.tree.map(lambda x: x[None], other)

  def run_single_layer(carry, xs_slice):
    params_slice, other_slice = xs_slice
    merged_layer = nnx.merge(graphdef, params_slice, other_slice)
    out_y, _ = merged_layer(carry, **layer_kwargs)
    _, _, new_other = nnx.split(merged_layer, nnx.Param, ...)
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
    )
    load_balance_loss = metadata.get("load_balance_loss", None)
    moe_bias_updates = metadata.get("moe_bias_updates", None)

    layer_output = self.dropout_op(layer_output, deterministic=deterministic)
    return self.post_process(layer_output, load_balance_loss, moe_bias_updates, kv_cache)


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

    if model_mode == "train" and bool(self.config.remat_policy) and self.config.remat_policy != "none":
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
      remat_kwargs = {"config": self.config, "policy": self.remat_policy_fn, "prevent_cse": self.prevent_cse_in_remat}
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
