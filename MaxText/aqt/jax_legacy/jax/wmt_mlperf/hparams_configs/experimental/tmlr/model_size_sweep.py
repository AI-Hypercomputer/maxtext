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

"""Creates a sweep over multiple sized models.

Sweep:
[{number of layers x emb dim x (heads,  qkv,  mlp)} x quantization x sparsity]
Sweep for {3, 6} x {1024} x {(8, 512, 1024), (8, 1024, 2048), (16, 512, 2048),
(16, 1024, 4096)}
"""

import copy
import itertools

from aqt.jax_legacy.jax.wmt_mlperf.hparams_configs import base_config


def get_model_size_configs():
  """Sweeps over multiple num_layers, heads dims, qkv_dims and mlp_dims."""
  configs = []

  for num_layers, (num_heads, qkv_dim, mlp_dim) in itertools.product(
      [3, 6],
      [(8, 512, 1024), (8, 1024, 2048), (16, 512, 2048), (16, 1024, 4096)],
  ):
    base_config_file = base_config.get_config(
        n_layers=num_layers,
        quant_target=base_config.QuantTarget.WEIGHTS_AND_AUTO_ACTS,
    )
    config = copy.deepcopy(base_config_file)
    config.metadata.hyper_str = f"wmt_L({num_layers})_H({num_heads})_QKV({qkv_dim})_MLP({mlp_dim})_emb(1024)"
    model = config.model_hparams
    model.num_heads = num_heads
    model.qkv_dim = qkv_dim
    model.mlp_dim = mlp_dim
    configs.append(config)

  return configs
