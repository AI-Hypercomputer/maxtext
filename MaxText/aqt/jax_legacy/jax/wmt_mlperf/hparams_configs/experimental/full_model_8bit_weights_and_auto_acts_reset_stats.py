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

"""Full-sized transformer with 8-bit weights-and-acts, resetting activation stats every update."""

from aqt.jax_legacy.jax.wmt_mlperf.hparams_configs import base_config
from aqt.jax_legacy.jax.wmt_mlperf.hparams_configs.leaderboard import full_model_bfloat16


def get_config():
  config = full_model_bfloat16.get_config(
      quant_target=base_config.QuantTarget.WEIGHTS_AND_AUTO_ACTS)
  config.bounds.reset_stats = True
  config.weight_prec = 8
  config.quant_act.prec = 8
  return config
