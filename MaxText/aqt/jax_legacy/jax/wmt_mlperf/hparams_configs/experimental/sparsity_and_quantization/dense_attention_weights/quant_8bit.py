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

"""Small-sized model with 8-bit quantization for attention weights only."""

from aqt.jax_legacy.jax.wmt_mlperf.hparams_configs import base_config
from aqt.jax_legacy.jax.wmt_mlperf.hparams_configs.experimental import small_model_bfloat16


def get_config():
  """Returns configuration for a small transformer model with 8-bit quantization for attention weights only."""
  config = small_model_bfloat16.get_config(
      quant_target=base_config.QuantTarget.WEIGHTS_ONLY)
  config.quant_type = "aqt"
  config.attention.dense_kqv.weight_prec = 8
  config.attention.dense_out.weight_prec = 8
  config.metadata.hyper_str = "attn_weights_quant_8bit"
  return config
