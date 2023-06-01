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

"""Different sparsity and quantization strategies for dense weights in small-sized model."""

import copy

from aqt.jax_legacy.jax.wmt_mlperf.hparams_configs.experimental.sparsity_and_quantization import baseline
from aqt.jax_legacy.jax.wmt_mlperf.hparams_configs.experimental.sparsity_and_quantization.dense_weights import quant_4bit
from aqt.jax_legacy.jax.wmt_mlperf.hparams_configs.experimental.sparsity_and_quantization.dense_weights import quant_8bit
from aqt.jax_legacy.jax.wmt_mlperf.hparams_configs.experimental.sparsity_and_quantization.dense_weights import quant_8bit_sparsity_structured_2_4
from aqt.jax_legacy.jax.wmt_mlperf.hparams_configs.experimental.sparsity_and_quantization.dense_weights import sparsity_structured_1_4
from aqt.jax_legacy.jax.wmt_mlperf.hparams_configs.experimental.sparsity_and_quantization.dense_weights import sparsity_structured_2_4
import ml_collections


def get_config():
  """Returns sweep configuration (see module docstring)."""
  sweep_config = ml_collections.ConfigDict()
  base_configs = [
      baseline,
      quant_4bit,
      sparsity_structured_1_4,
      quant_8bit_sparsity_structured_2_4,
      sparsity_structured_2_4,
      quant_8bit,
  ]
  configs = []
  for base_config_file in base_configs:
    base_config = base_config_file.get_config()
    config = copy.deepcopy(base_config)
    config.metadata.hyper_str = f"{config.metadata.hyper_str}"
    configs.append(config)
  sweep_config.configs = configs
  return sweep_config
