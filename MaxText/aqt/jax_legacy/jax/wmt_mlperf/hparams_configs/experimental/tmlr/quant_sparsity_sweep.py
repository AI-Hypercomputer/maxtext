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

"""Multiple sized model with varied quantization and sparsity.

Quantization is enabled since starting
Sparsity is one shot pruning, updated and applied at 100K steps.
"""

import copy

from aqt.jax_legacy.jax.wmt_mlperf.hparams_configs.experimental.tmlr import model_size_sweep
import ml_collections


def get_config():
  """Sweep for multiple sized model with varied quantization and sparsity."""
  sweep_config = ml_collections.ConfigDict()
  configs = []

  for model_config in model_size_sweep.get_model_size_configs():
    for prec, prune_rate in [
        (None, None),
        (8, None),
        (None, (2, 4)),
        (4, None),
        (None, (1, 4)),
        (8, (2, 4)),
        (2, None),
        (None, (1, 8)),
        (4, (2, 4)),
        (8, (1, 4)),
    ]:
      config = copy.deepcopy(model_config)
      config.quant_type = 'aqt'

      # mlp + attention weights + acts  quantization
      config.dense.weight_prec = prec
      config.dense.quant_act.prec = prec

      # mlp weights sparsity
      config.mlp_block.dense_1.weight_sparsity.type = 'STRUCTURED_NM'
      config.mlp_block.dense_1.weight_sparsity.prune_rate = prune_rate
      config.mlp_block.dense_2.weight_sparsity.type = 'STRUCTURED_NM'
      config.mlp_block.dense_2.weight_sparsity.prune_rate = prune_rate
      # attn_weights sparsity
      config.attention.dense_kqv.weight_sparsity.type = 'STRUCTURED_NM'
      config.attention.dense_kqv.weight_sparsity.prune_rate = prune_rate
      config.attention.dense_out.weight_sparsity.type = 'STRUCTURED_NM'
      config.attention.dense_out.weight_sparsity.prune_rate = prune_rate
      config.metadata.hyper_str = (
          f'{config.metadata.hyper_str}_prec({prec})_NM({prune_rate})'
      )
      configs.append(config)

  sweep_config.configs = configs
  return sweep_config
