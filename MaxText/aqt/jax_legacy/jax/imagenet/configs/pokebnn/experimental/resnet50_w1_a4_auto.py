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

"""Resnet50 quantized model.

# pylint: disable=line-too-long

Results at the last batch from TB:
- Eval accuracy: 0.7271
- Eval loss: 1.077
- Train accuracy: 0.7365
- Training loss: 1.102
- Tensorboard link: http://shortn/_wpbXEvjgmx
- Xmanager link: http://shortn/_KAzZi6QzHI
- CNS link: http://shortn/_kamgdDcOKe
- Metrics from reporting tool: http://shortn/_Ey7SVZ1jeY
- Bash file for launching: ./resnet50_w1_a4_auto.sh
"""

from aqt.jax_legacy.jax.imagenet.configs import base_config


def get_config(quant_target=base_config.QuantTarget.WEIGHTS_AND_AUTO_ACT):
  """Gets Resnet50 config for 1 bit weights and 4 bits auto activation quantization.

  conv_init and last dense layer not quantized as these are the most
  sensitive layers in the model.

  Args:
   quant_target: quantization target, of type QuantTarget.

  Returns:
   ConfigDict instance.
  """
  config = base_config.get_config(
      imagenet_type=base_config.ImagenetType.RESNET50,
      quant_target=quant_target)
  config.weight_prec = 1
  config.quant_act.prec = 4
  config.half_shift = True

  # The first conv layer and the dense layer still use 8-bit weights and acts
  config.model_hparams.conv_init.weight_prec = 8
  config.model_hparams.conv_init.quant_act.prec = 8

  config.model_hparams.dense_layer.weight_prec = 8
  config.model_hparams.dense_layer.quant_act.prec = 8

  config.metadata.hyper_str = "resnet50_w1_a4_auto"

  return config
