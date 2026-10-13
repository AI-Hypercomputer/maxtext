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

"""GLM-5.3-Flash (glm5_next) decoder layers.

Layer schedule (period `inhomogeneous_layer_cycle_interval` = 4):
  * attention: KDA linear attention, except every 4th layer (i % 4 == 3) which is
    DeepSeek Sparse Attention (NoPE MLA + k-pool indexer);
  * MLP: dense SwiGLU for the first `first_num_dense_layers` layers, MoE afterwards;
  * residuals: manifold-constrained hyper-connections (mHC) around attention and MLP.

For scanning, the decoder is split into an unscanned prefix (the dense layers rounded up to a full
period, e.g. layers 0-3), `Glm5NextScannableBlock`s of one period each (3 KDA+MoE, 1 DSA+MoE), and
an unscanned remainder (e.g. layer 44 of 45).

Reference: `Glm5NextTextDecoderLayer` in transformers/models/glm5_next/modeling_glm5_next.py.
"""

from typing import Optional

from flax import nnx
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import Mesh

from maxtext.common.common_types import Config, HyperConnectionType, MODEL_MODE_PREFILL, get_weight_dtype
from maxtext.layers import initializers
from maxtext.layers import linears
from maxtext.layers import mhc
from maxtext.layers import moe
from maxtext.layers import quantizations
from maxtext.layers.attention_glm5 import Glm5SparseMLA
from maxtext.layers.kimi_delta_attention import KimiDeltaAttention
from maxtext.layers.normalizations import RMSNorm
from maxtext.utils import max_utils
from maxtext.utils.sharding import maybe_shard_with_logical, get_logical_axis_rules


def is_dsa_layer(config: Config, layer_idx: int) -> bool:
  cycle = config.inhomogeneous_layer_cycle_interval
  return layer_idx % cycle == cycle - 1


def is_dense_layer(config: Config, layer_idx: int) -> bool:
  return layer_idx < config.first_num_dense_layers


def scan_layout(config: Config) -> tuple[int, int, int]:
  """Returns (num_prefix_layers, num_scanned_blocks, num_remainder_layers) for scanned execution."""
  cycle = config.inhomogeneous_layer_cycle_interval
  num_prefix = -(-config.first_num_dense_layers // cycle) * cycle
  num_prefix = min(num_prefix, config.num_decoder_layers)
  rest = config.num_decoder_layers - num_prefix
  return num_prefix, rest // cycle, rest % cycle


class Glm5NextDecoderLayer(nnx.Module):
  """One GLM-5.3-Flash decoder layer (KDA or DSA attention; dense or MoE MLP; mHC residuals)."""

  def __init__(
      self,
      config: Config,
      model_mode: str,
      mesh: Mesh,
      rngs: nnx.Rngs,
      quant: Optional[quantizations.AqtQuantization] = None,
      layer_idx: int = -1,
      is_dsa: Optional[bool] = None,
      is_dense: Optional[bool] = None,
  ):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.quant = quant
    self.layer_idx = layer_idx
    self.is_dsa = is_dsa_layer(config, layer_idx) if is_dsa is None else is_dsa
    self.is_dense = is_dense_layer(config, layer_idx) if is_dense is None else is_dense
    assert config.mhc_expansion_rate > 1, "GLM-5.3-Flash requires mHC (mhc_expansion_rate > 1)."

    batch_size, seq_len = max_utils.get_batch_seq_len_for_mode(config, model_mode)
    del batch_size, seq_len
    emb = config.emb_dim

    def norm():
      return RMSNorm(
          num_features=emb,
          dtype=config.dtype,
          weight_dtype=get_weight_dtype(config, "norm"),
          kernel_axes=("norm",),
          epsilon=config.normalization_layer_epsilon,
          rngs=rngs,
      )

    self.pre_self_attention_layer_norm = norm()
    self.post_self_attention_layer_norm = norm()

    if self.is_dsa:
      self.self_attention = Glm5SparseMLA(config, mesh, config.dtype, quant, rngs=rngs)
    else:
      self.self_attention = KimiDeltaAttention(config, mesh, config.dtype, quant, rngs=rngs)

    if self.is_dense:
      self.mlp = linears.MlpBlock(
          config=config,
          mesh=mesh,
          in_features=emb,
          intermediate_dim=config.mlp_dim,
          activations=config.mlp_activations,
          intermediate_dropout_rate=config.dropout_rate,
          dtype=config.dtype,
          weight_dtype=config.weight_dtype,
          quant=quant,
          model_mode=model_mode,
          rngs=rngs,
      )
    else:
      self.mlp = moe.RoutedAndSharedMoE(
          config=config,
          mesh=mesh,
          kernel_init=initializers.nd_dense_init(config.dense_init_scale, "fan_in", "truncated_normal"),
          kernel_axes=("embed", None),
          dtype=config.dtype,
          weight_dtype=config.weight_dtype,
          quant=quant,
          rngs=rngs,
      )

    self.mhc_attention = mhc.ManifoldConstrainedHyperConnections(config, emb, mesh, rngs)
    self.mhc_mlp = mhc.ManifoldConstrainedHyperConnections(config, emb, mesh, rngs)

  @property
  def logical_axis_names(self):
    length = "prefill_activation_norm_length" if self.model_mode == MODEL_MODE_PREFILL else "activation_norm_length"
    return ("activation_batch", length, "activation_embed")

  def _constrain(self, x):
    return maybe_shard_with_logical(
        x,
        logical_axes=self.logical_axis_names,
        mesh=self.mesh,
        shard_mode=self.config.shard_mode,
        debug_sharding=self.config.debug_sharding,
        rules=get_logical_axis_rules(),
    )

  def _pre_attention_norm(self, x):
    return self._constrain(self.pre_self_attention_layer_norm(x))

  def _post_attention_norm(self, x):
    return self._constrain(self.post_self_attention_layer_norm(x))

  def _dense_mlp(self, inputs, deterministic=True):
    return self._constrain(self.mlp(inputs, deterministic=deterministic))

  def _moe_mlp(self, inputs, deterministic=True):
    del deterministic
    out, load_balance_loss, moe_bias_updates = self.mlp(inputs=inputs)
    return self._constrain(out), load_balance_loss, moe_bias_updates

  def __call__(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      previous_chunk=None,
      slot=None,
      kv_cache=None,
      attention_metadata=None,
      decoder_input_tokens=None,
      **kwargs,
  ):
    del decoder_positions, previous_chunk, slot, attention_metadata, decoder_input_tokens, kwargs
    if isinstance(inputs, tuple):
      inputs = inputs[0]
    x = checkpoint_name(inputs, "decoder_layer_input")  # [B, S, hc_mult, E]

    x, _ = self.mhc_attention(
        self._pre_attention_norm,
        self.self_attention,
        x=x,
        mhc_type=HyperConnectionType.ATTENTION,
        decoder_segment_ids=decoder_segment_ids,
        model_mode=model_mode,
    )

    if self.is_dense:
      x, metadata = self.mhc_mlp(
          self._post_attention_norm, self._dense_mlp, x=x, mhc_type=HyperConnectionType.MLP_DENSE,
          deterministic=deterministic,
      )
    else:
      x, metadata = self.mhc_mlp(
          self._post_attention_norm, self._moe_mlp, x=x, mhc_type=HyperConnectionType.MLP_MOE,
          deterministic=deterministic,
      )
      if self.config.load_balance_loss_weight > 0.0 and metadata.get("load_balance_loss") is not None:
        self.sow(nnx.Intermediate, "moe_lb_loss", metadata["load_balance_loss"])

    if self.config.scan_layers:
      return x, None
    return x, kv_cache


class Glm5NextScannableBlock(nnx.Module):
  """One schedule period of MoE layers: (cycle - 1) KDA layers followed by one DSA layer."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      rngs: nnx.Rngs,
      quant: Optional[quantizations.AqtQuantization] = None,
  ):
    self.config = config
    self.cycle = config.inhomogeneous_layer_cycle_interval
    for i in range(self.cycle):
      layer = Glm5NextDecoderLayer(
          config=config,
          model_mode=model_mode,
          mesh=mesh,
          rngs=rngs,
          quant=quant,
          is_dsa=(i == self.cycle - 1),
          is_dense=False,
      )
      setattr(self, f"layers_{i}", layer)

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
      **kwargs,
  ):
    y = inputs
    for i in range(self.cycle):
      y, _ = getattr(self, f"layers_{i}")(y, decoder_segment_ids, decoder_positions, deterministic, model_mode)
    return y, None
