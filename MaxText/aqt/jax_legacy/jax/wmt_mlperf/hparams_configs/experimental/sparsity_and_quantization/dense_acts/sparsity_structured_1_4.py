# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Small-sized model with 1:4 sparsity for dense activations only."""

from aqt.jax_legacy.jax.wmt_mlperf.hparams_configs import base_config
from aqt.jax_legacy.jax.wmt_mlperf.hparams_configs.experimental import small_model_bfloat16


def get_config(quant_target=base_config.QuantTarget.NONE):
  """Returns configuration for a small transformer model with 1:4 sparsity for dense activations only."""
  config = small_model_bfloat16.get_config(
      quant_target=quant_target)
  config.mlp_block_dense_1.act_sparsity.type = 'STRUCTURED_NM'
  config.mlp_block_dense_1.act_sparsity.prune_rate = (1, 4)
  config.mlp_block_dense_2.act_sparsity.type = 'STRUCTURED_NM'
  config.mlp_block_dense_2.act_sparsity.prune_rate = (1, 4)
  config.metadata.hyper_str = 'mlp_acts_sparsity_structured_1_4'
  return config
