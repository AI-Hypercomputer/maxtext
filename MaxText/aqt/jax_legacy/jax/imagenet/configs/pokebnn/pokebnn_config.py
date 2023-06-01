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

"""PokeBNN model configuration."""

import copy

from aqt.jax_legacy.jax.imagenet.configs import base_config
from aqt.jax_legacy.jax.imagenet.configs.paper import resnet50_w4_a4_init8_dense8_auto

import ml_collections


def get_config(quant_target=base_config.QuantTarget.WEIGHTS_AND_AUTO_ACT):
  """Gets Resnet50 config for 8 bits weights and 1 bit auto activation quantization.

  conv_init and last dense layer not quantized as these are the most
  sensitive layers in the model.

  Args:
   quant_target: quantization target, of type QuantTarget.

  Returns:
   ConfigDict instance.
  """

  def set_init_bound_coeff(field):
    # input should be a class field so that the changes in this function
    # will be global to the class even without a return value
    field.cams_coeff = 0.0
    field.cams_stddev_coeff = 0.0
    field.mean_of_max_coeff = 0.0
    field.stddev_coeff = 0.0
    field.absdev_coeff = 0.0
    field.fixed_bound = 0.0
    field.granularity = "per_channel"
    field.use_old_code = False

  def reset_bound_for_convinit_dense(config):
    # reset bound haparams for conv_init and dense layers
    # use mean_of_max to automatically calculate the bound values
    set_init_bound_coeff(config.model_hparams.dense_layer.quant_act.bounds)
    config.model_hparams.dense_layer.quant_act.bounds.initial_bound = -1
    config.model_hparams.dense_layer.quant_act.bounds.mean_of_max_coeff = 1.0
    config.model_hparams.dense_layer.weight_half_shift = False
    config.model_hparams.dense_layer.quant_act.half_shift = False
    set_init_bound_coeff(config.model_hparams.conv_init.quant_act.bounds)
    config.model_hparams.conv_init.quant_act.bounds.initial_bound = -1
    config.model_hparams.conv_init.quant_act.bounds.mean_of_max_coeff = 1.0
    config.model_hparams.conv_init.weight_half_shift = False
    config.model_hparams.conv_init.quant_act.half_shift = False
    return config

  def set_conv_proj_precision(config, bits):
    for residual_block in config.model_hparams.residual_blocks:
      set_init_bound_coeff(residual_block.conv_se.quant_act.bounds)
      residual_block.conv_se.quant_act.bounds.initial_bound = -1
      residual_block.conv_se.quant_act.bounds.mean_of_max_coeff = 1.0
      residual_block.conv_se.weight_prec = bits
      residual_block.conv_se.quant_act.prec = bits
      residual_block.conv_se.weight_half_shift = False
      residual_block.conv_se.quant_act.half_shift = False
      if residual_block.conv_proj is not None:
        set_init_bound_coeff(residual_block.conv_proj.quant_act.bounds)
        residual_block.conv_proj.quant_act.bounds.initial_bound = -1
        residual_block.conv_proj.quant_act.bounds.mean_of_max_coeff = 1.0
        residual_block.conv_proj.weight_prec = bits
        residual_block.conv_proj.quant_act.prec = bits
        residual_block.conv_proj.weight_half_shift = False
        residual_block.conv_proj.quant_act.half_shift = False
    return config

  # create an init config which the sweep configs will be based on
  config_init = base_config.get_config(
      imagenet_type=base_config.ImagenetType.RESNET50,
      quant_target=quant_target)
  config_init.teacher_model = "resnet50-8bit"
  config_init.is_teacher = False
  config_init.weight_prec = 1
  config_init.quant_act.prec = 1
  config_init.half_shift = True
  config_init.base_learning_rate = 2e-5
  config_init.activation_bound_start_step = 7500
  config_init.weight_quant_start_step = 50
  config_init.weight_decay = 5e-5
  # set act function and shortcut method to each conv layer
  config_init.act_function = "none"
  config_init.shortcut_ch_shrink_method = "none"
  config_init.shortcut_ch_expand_method = "none"
  config_init.shortcut_spatial_method = "none"
  # set learning rate scheduler
  config_init.lr_scheduler.num_epochs = 800
  config_init.lr_scheduler.warmup_epochs = 0
  config_init.lr_scheduler.cooldown_epochs = 0
  config_init.lr_scheduler.scheduler = "piecewise"
  config_init.lr_scheduler.knee_epochs = 750
  config_init.lr_scheduler.knee_lr = 0.0
  config_init.lr_scheduler.endlr = 0.0
  # -1 means no early stopping by default
  config_init.early_stop_steps = -1
  # optimizer params
  config_init.optimizer = "adam"
  config_init.adam.beta1 = 0.9
  config_init.adam.beta2 = 0.99
  # Conv_init and dense layers will have floating-point weights and acts
  config_init.model_hparams.conv_init.weight_prec = 8
  config_init.model_hparams.conv_init.quant_act.prec = 8
  config_init.model_hparams.dense_layer.weight_prec = 8
  config_init.model_hparams.dense_layer.quant_act.prec = 8
  # set all of the input distributions to "symmetric"
  config_init.model_hparams.dense_layer.quant_act.input_distribution = "symmetric"
  config_init.model_hparams.conv_init.quant_act.input_distribution = "symmetric"
  for residual_block in config_init.model_hparams.residual_blocks:
    residual_block.conv_1.quant_act.input_distribution = "symmetric"
    residual_block.conv_2.quant_act.input_distribution = "symmetric"
    residual_block.conv_3.quant_act.input_distribution = "symmetric"
    residual_block.conv_se.quant_act.input_distribution = "symmetric"
    if residual_block.conv_proj is not None:
      residual_block.conv_proj.quant_act.input_distribution = "symmetric"
      residual_block.conv_proj.weight_prec = None
      residual_block.conv_proj.quant_act.prec = None
  # set bound hparams to all zero for activations
  # will update one of the bound hparams at a time in sweep configs
  set_init_bound_coeff(config_init.quant_act.bounds)
  # set initial bound value
  config_init.quant_act.bounds.initial_bound = 3.0
  # name of the experiment on TB
  config_init.metadata.hyper_str = "w1a1_KD"

  # create a collection of config files for sweeping
  sweep_config = ml_collections.ConfigDict()
  configs = []

  # sweep filter multiplier
  for init_group in [32]:
    for shortcut_spatial_method in ["avg_pool"]:
      for shortcut_shrink_method in ["consecutive"]:
        for shortcut_expand_method in ["zeropad"]:
          config = copy.deepcopy(config_init)
          config.model_hparams.filter_multiplier = 1.0
          config.model_hparams.init_group = init_group
          config.model_hparams.se_ratio = 0.125
          config.act_function = "bprelu"
          config.quant_act.bounds.fixed_bound = 3.0
          config.shortcut_ch_shrink_method = shortcut_shrink_method
          config.shortcut_ch_expand_method = shortcut_expand_method
          config.shortcut_spatial_method = shortcut_spatial_method
          config = set_conv_proj_precision(config, 4)
          # reset bound haparams for conv_init and dense layers
          config = reset_bound_for_convinit_dense(config)
          config.metadata.hyper_str = "sweep_pokebnn"
          configs.append(config)

  # w4a4_init8_dense8 model
  configs.append(resnet50_w4_a4_init8_dense8_auto.get_config())

  sweep_config.configs = configs
  return sweep_config
