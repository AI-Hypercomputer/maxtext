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

"""GLM model definitions (GLM-5.1 & GLM-5.2 with Cross-Layer IndexShare)."""

import jax
from maxtext.layers import initializers
from maxtext.layers import nnx_wrappers
from maxtext.models import deepseek
from maxtext.utils import index_share_utils
from maxtext.utils import max_logging


class GLMGenericLayer(deepseek.DeepSeekGenericLayer):  # pylint: disable=abstract-method
  """Generic GLM layer with Multi-Head Latent Attention and Cross-Layer IndexShare support."""

  def _init_self_attention(self) -> None:
    """Resolves GLM-5.2 IndexShare layer role before constructing MLA once."""
    self.is_shared_layer = False
    self.served_group_size = 1
    if self.config.use_index_share and self.layer_idx >= 0:
      pattern = index_share_utils.parse_index_share_pattern(
          self.config.index_share_pattern, self.config.num_decoder_layers
      )
      self.is_shared_layer = index_share_utils.is_shared_layer(self.layer_idx, pattern)
      self.served_group_size = index_share_utils.get_served_group_sizes(pattern)[self.layer_idx]
      if self.layer_idx == 0 and jax.process_index() == 0:
        num_f = pattern.count("F")
        num_s = pattern.count("S")
        pruned_state = "pruned" if (self.config.prune_shared_indexers and not self.config.scan_layers) else "retained"
        shared_pct = num_s / self.config.num_decoder_layers * 100
        max_logging.info(
            f"[GLM-5.2 IndexShare Active] Total layers: {self.config.num_decoder_layers} | "
            f"Pattern: {self.config.index_share_pattern} | Full (F) layers: {num_f} | "
            f"Shared (S) layers ({pruned_state} indexers): {num_s} ({shared_pct:.1f}% shared)"
        )
    super()._init_self_attention()

  def _format_layer_return(self, layer_output, kv_cache, cached_indexer_state=None):
    """Returns uniform 3-tuple (layer_output, kv_cache_or_none, cached_indexer_state)."""
    return layer_output, (None if self.config.scan_layers else kv_cache), cached_indexer_state


class GLMDenseLayer(GLMGenericLayer, deepseek.DeepSeekDenseLayer):
  """GLM dense decoder layer reusing DeepSeekDenseLayer with IndexShare support."""


class GLMMoELayer(GLMGenericLayer, deepseek.DeepSeekMoELayer):
  """GLM MoE decoder layer reusing DeepSeekMoELayer with IndexShare support."""


GLMDenseLayerToLinen = nnx_wrappers.to_linen_class(
    GLMDenseLayer,
    base_metadata_fn=initializers.variable_to_logically_partitioned,
)

GLMMoELayerToLinen = nnx_wrappers.to_linen_class(
    GLMMoELayer,
    base_metadata_fn=initializers.variable_to_logically_partitioned,
)
