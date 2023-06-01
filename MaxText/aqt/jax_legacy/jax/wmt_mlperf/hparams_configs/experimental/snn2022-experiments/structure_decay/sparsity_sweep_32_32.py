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

"""Creates a sweep over all the leaderboard models.

Model size: full model
Cadence: 100K, 150K
Sweep:
Parts: {encoder, decoder, encoder + decoder}
Inputs: {weights}
Sparsity type: {1:4, 2:4}
"""

import copy

from aqt.jax_legacy.jax.wmt_mlperf.hparams_configs.experimental.sparsity import full_model_bfloat16_sparse_dense_weights_only_structured_nm


import ml_collections


def get_config():
  """Returns sweep configuration (see module docstring)."""
  sweep_config = ml_collections.ConfigDict()
  base_configs = [
      full_model_bfloat16_sparse_dense_weights_only_structured_nm,
  ]
  configs = []
  for base_config_file in base_configs:
    for n, m in zip([32], [32]):
      base_config = base_config_file.get_config(n, m)
      base_config.mlp_block.dense_1.weight_sparsity.structure_decay = True
      base_config.mlp_block.dense_2.weight_sparsity.structure_decay = True
      config = copy.deepcopy(base_config)
      config.metadata.hyper_str = f"{config.metadata.hyper_str}_N({n})_M({m})"
      configs.append(config)
  sweep_config.configs = configs
  return sweep_config
