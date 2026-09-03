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

"""Qwen3.5 family of model decoder layers."""
# pylint: disable=arguments-differ
# pylint: disable=no-name-in-module

from flax import nnx

from maxtext.layers import initializers as max_initializers
from maxtext.layers import nnx_wrappers
from maxtext.utils import max_utils

from maxtext.models.qwen3 import (
    Qwen3NextDecoderLayer,
    Qwen3NextGatedDeltaNet,
    Qwen3NextFullAttention,
    Qwen3NextScannableBlock,
    Qwen3NextSparseMoeBlock,
)


# -----------------------------------------
# Qwen3.5 Layer Implementations
# -----------------------------------------


class Qwen3_5GatedDeltaNet(Qwen3NextGatedDeltaNet):
  """Qwen3.5 GatedDeltaNet layer that is identical to Qwen3-Next GatedDeltaNet"""


class Qwen3_5FullAttention(Qwen3NextFullAttention):
  """Qwen3.5 Gated Attention layer that is identical to Qwen3-Next"""


class Qwen3_5SparseMoEBlock(Qwen3NextSparseMoeBlock):
  """Shares same MoE code as Qwen3-Next"""


class Qwen3_5ScannableBlock(Qwen3NextScannableBlock):
  """Scanned Structure for Text-only Architecture, explicitly invoking Qwen3_5 layers.

  Qwen3.5 repeats the same hybrid attention period as Qwen3-Next -- several
  GatedDeltaNet layers followed by one full-attention layer -- so it reuses
  Qwen3-Next's hierarchical nested scans (an inner scan over the homogeneous
  linear-attention layers plus a trip-count-one scan over the full-attention
  layer) and only swaps in the Qwen3.5 decoder layer.
  """

  def _make_decoder_layer(self, *, layer_idx, is_full_attention_layer, rngs):
    return Qwen3_5DecoderLayer(
        config=self.config,
        mesh=self.mesh,
        model_mode=self.model_mode,
        quant=self.quant,
        layer_idx=layer_idx,
        is_full_attention_layer=is_full_attention_layer,
        rngs=rngs,
    )


class Qwen3_5DecoderLayer(Qwen3NextDecoderLayer):
  """Qwen3.5 hybrid decoder layer.

  Qwen3.5's decoder layer is structurally identical to Qwen3-Next's -- norm,
  either full or linear attention, norm, sparse MoE, with residuals around the
  attention and MoE halves -- so the whole forward pass is inherited. Only the
  sub-block factories are overridden, to build the Qwen3.5 sub-classes. The
  attribute names (`input_layernorm`, `attention`, `post_attention_layernorm`,
  `mlp`) are therefore shared, and so are the checkpoint parameter paths.

  Attributes:
    config: The model configuration object.
    mesh: The device mesh for sharding.
    model_mode: The operational mode (e.g., 'train', 'prefill').
    layer_idx: The index of the current layer in the transformer stack.
    quant: Optional quantization configuration.
  """

  def _make_full_attention(self, *, rngs: nnx.Rngs):
    return Qwen3_5FullAttention(
        config=self.config,
        mesh=self.mesh,
        quant=self.quant,
        model_mode=self.model_mode,
        layer_idx=self.layer_idx,
        rngs=rngs,
    )

  def _make_linear_attention(self, *, rngs: nnx.Rngs):
    cfg = self.config
    batch_size, seq_len = max_utils.get_batch_seq_len_for_mode(cfg, self.model_mode)
    dummy_inputs_shape = (batch_size, seq_len, cfg.emb_dim)
    return Qwen3_5GatedDeltaNet(
        config=cfg,
        inputs_shape=dummy_inputs_shape,
        mesh=self.mesh,
        dtype=cfg.dtype,
        model_mode=self.model_mode,
        rngs=rngs,
    )

  def _make_mlp(self, *, rngs: nnx.Rngs):
    return Qwen3_5SparseMoEBlock(config=self.config, mesh=self.mesh, quant=self.quant, rngs=rngs)


Qwen3_5DecoderLayerToLinen = nnx_wrappers.to_linen_class(
    Qwen3_5DecoderLayer,
    base_metadata_fn=max_initializers.variable_to_logically_partitioned,
)


Qwen3_5ScannableBlockToLinen = nnx_wrappers.to_linen_class(
    Qwen3_5ScannableBlock,
    base_metadata_fn=max_initializers.variable_to_logically_partitioned,
)
