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

"""Resnet101 unquantized model."""

from aqt.jax_legacy.jax.imagenet.configs import base_config


def get_config(quant_target=base_config.QuantTarget.NONE):
  return base_config.get_config(
      imagenet_type=base_config.ImagenetType.RESNET101,
      quant_target=quant_target)
