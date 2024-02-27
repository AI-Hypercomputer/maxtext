"""
 Copyright 2023 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

# pylint: disable=missing-module-docstring, bare-except, consider-using-generator
from collections import OrderedDict

import max_logging

import accelerator_to_spec_map
import math
import max_utils
import os
import sys
import yaml

import jax

from typing import Any, Union

_MAX_PREFIX = "M_"

def yaml_key_to_env_key(s: str) -> str:
  return _MAX_PREFIX + s.upper()

def string_to_bool(s: str) -> bool:
  if s.lower() == "true":
    return True
  if s.lower() == "false":
    return False
  raise ValueError(f"Can't convert {s} to bool")

_yaml_types_to_parser = {str : str, int : int, float : float, bool : string_to_bool}

def validate_attention_type(s: str) -> None:
  valid_attention_types = ('dot_product', 'flash', 'cudnn_flash_te')
  if s not in valid_attention_types: # currently supported attention
    raise ValueError(
      "Invalid attention type was passed. Valid options ", valid_attention_types
    )

def validate_keys(keys):
  validate_attention_type(keys['attention'])

  assert ((keys["load_parameters_path"]=="" and keys["load_full_state_path"]=="") or
    keys["enable_checkpointing"]), "You must set enable_checkpointing to load a checkpoint"
  assert keys["load_parameters_path"]=="" or keys["load_full_state_path"]=="",\
    "At most one of `load_parameters_path` or `load_full_state_path` should be set"

def validate_model_name(s: str) -> bool:
  # currently supported models
  valid_model_names = ('default', 'llama2-7b', 'llama2-70b', 'mistral-7b',
                       'mixtral-8x7b', 'gemma-7b','gemma-2b',
                       'gpt3-175b', 'gpt3-22b', 'gpt3-6b', 'gpt3-52k')
  if s not in valid_model_names:
    raise ValueError(
      "Invalid model name was passed. Valid options ", valid_model_names
    )

def validate_no_keys_overwritten_twice(keys1: list[str], keys2: list[str]):
  overwritten_keys = [k for k in keys1 if k in keys2]
  if overwritten_keys:
    raise ValueError(
        f"Keys {overwritten_keys} are overwritten from both the model" 
        " and the environment/command line. This isn't allowed.")

_config = None
config = None

def print_system_information():
  max_logging.log(f"System Information: Jax Version: {jax.__version__}")
  max_logging.log(f"System Information: Jaxlib Version: {jax.lib.__version__}")
  max_logging.log(f"System Information: Jax Backend: {jax.lib.xla_bridge.get_backend().platform_version}")

def _lists_to_tuples(l: list[Any]) -> Union[tuple[Any],list[Any]]:
  return tuple(_lists_to_tuples(x) for x in l) if isinstance(l, list) else l

class _HyperParameters():
  # pylint: disable=missing-class-docstring
  def _validate_env_variables(self, raw_data_from_yaml):
    for environment_var in os.environ:
      if environment_var[:len(_MAX_PREFIX)] == _MAX_PREFIX:
        proposed_key = environment_var[len(_MAX_PREFIX):].lower()
        if proposed_key not in raw_data_from_yaml:
          raise ValueError(f"We received env `{environment_var}` but it doesn't match a key, so it is assumed a mistake.")
        if not environment_var[len(_MAX_PREFIX):].isupper():
          raise ValueError(f"We received env `{environment_var}` but it isn't all uppercase.")

  def _load_kwargs(self, argv: list[str], **kwargs):
    args_dict = dict(a.split("=", 1) for a in argv[2:])
    args_dict.update(kwargs)
    return args_dict

  def _update_from_env_and_command_line(self, raw_keys, raw_data_from_yaml, argv, **kwargs) -> list[str]:
    ''' Update model config from environment and command line
    '''
    raw_data_from_cmd_line = self._load_kwargs(argv, **kwargs)
    updated_keys = []

    for k in raw_data_from_cmd_line:
      if k not in raw_data_from_yaml:
        raise ValueError(
            f"Key {k} was passed at the command line but isn't in config."
        )

    for k in raw_data_from_yaml:
      if k in raw_data_from_cmd_line and yaml_key_to_env_key(k) in os.environ:
        raise ValueError(
            f"You are passing overrides by both CLI and ENV for `{k}`. This isn't allowed.")

      if not k in raw_data_from_cmd_line and not yaml_key_to_env_key(k) in os.environ:
        raw_keys[k] = raw_data_from_yaml[k]
        continue

      updated_keys.append(k)
      if k in raw_data_from_cmd_line:
        new_proposal = raw_data_from_cmd_line[k]
      else:
        new_proposal = os.environ.get(yaml_key_to_env_key(k))

      if (not isinstance(new_proposal, type(raw_data_from_yaml[k]))) and \
              (type(raw_data_from_yaml[k]) not in _yaml_types_to_parser):
        raise ValueError(
            f"For key '{k}', type {type(raw_data_from_yaml[k])} not in {_yaml_types_to_parser.keys()}, can't pass"
            " at the CLI or ENV"
        )

      if isinstance(new_proposal, type(raw_data_from_yaml[k])):
        raw_keys[k] = new_proposal  # take the raw data, no type conversion
      else:
        try:
          raw_keys[k] = _yaml_types_to_parser[type(raw_data_from_yaml[k])](
              new_proposal
          )  # take the command line value, but type it like the config value.
        except ValueError as e:
          raise ValueError(
              f"Couldn't parse value from CLI or ENV '{new_proposal}' for key '{k}'") from e

    return updated_keys

  def __init__(self, argv: list[str], **kwargs):
    with open(argv[1], "r", encoding="utf-8") as yaml_file:
      raw_data_from_yaml = yaml.safe_load(yaml_file)
    self._validate_env_variables(raw_data_from_yaml)

    raw_keys = OrderedDict()
    keys_from_env_and_command_line = self._update_from_env_and_command_line(raw_keys, raw_data_from_yaml, argv, **kwargs)
    max_logging.log(
        f"Updating keys from env and command line: {keys_from_env_and_command_line}")
    keys_from_model = _HyperParameters.update_model_vars(raw_keys)
    max_logging.log(f"Updating keys from model: {keys_from_model}")
    validate_no_keys_overwritten_twice(keys_from_env_and_command_line, keys_from_model)

    # We initialize the jax distributed system here because it must be done before device backend is initialized.
    max_utils.maybe_initialize_jax_distributed_system(raw_keys)

    if raw_keys['model_name'] == "gpt3-175b":
      _HyperParameters.configure_gpt3_task(raw_keys)
    _HyperParameters.user_init(raw_keys)

    self.keys = raw_keys
    keys = [k for k in raw_keys] # pylint: disable=unnecessary-comprehension
    keys.sort()
    for k in keys:
      max_logging.log(f"Config param {k}: {raw_keys[k]}")

  @staticmethod
  def user_init(raw_keys):
    '''Transformations between the config data and configs used at runtime'''
    if raw_keys["run_name"] == "":
      raw_keys["run_name"] = os.environ.get("JOBSET_NAME") #using XPK default
    run_name = raw_keys["run_name"]
    base_output_directory = raw_keys["base_output_directory"]
    if run_name:
      raw_keys["tensorboard_dir"] = os.path.join(base_output_directory, run_name, "tensorboard", "")
      raw_keys["checkpoint_dir"] = os.path.join(base_output_directory, run_name, "checkpoints", "")
      raw_keys["metrics_dir"] = os.path.join(base_output_directory, run_name, "metrics", "")

    if raw_keys["learning_rate_schedule_steps"]==-1:
      raw_keys["learning_rate_schedule_steps"] = raw_keys["steps"]
    if raw_keys["steps"]==-1:
      raw_keys["steps"] = raw_keys["learning_rate_schedule_steps"]

    emb_scale, num_head_scale, mlp_dim_scale, layer_scale = get_individual_scales(raw_keys['global_parameter_scale'])
    raw_keys['emb_dim'] = 2**emb_scale * raw_keys['base_emb_dim']
    raw_keys['num_query_heads'] = 2**num_head_scale * raw_keys['base_num_query_heads']
    raw_keys['num_kv_heads'] = 2**num_head_scale * raw_keys['base_num_kv_heads']
    raw_keys['mlp_dim'] = 2**mlp_dim_scale * raw_keys['base_mlp_dim']
    raw_keys['num_decoder_layers'] = 2**layer_scale * raw_keys['base_num_decoder_layers']

    raw_keys['global_batch_size_to_load'], raw_keys['global_batch_size_to_train_on'] = \
      calculate_global_batch_sizes(raw_keys)
    raw_keys['num_slices'] = get_num_slices(raw_keys)
    raw_keys['quantization_local_shard_count'] = get_quantization_local_shard_count(raw_keys)

    print_system_information()

    # Write raw_keys to GCS before type conversions
    max_utils.write_config_raw_keys_for_gcs(raw_keys)

    # Type conversions
    raw_keys["dtype"] = jax.numpy.dtype(raw_keys["dtype"])
    raw_keys["logical_axis_rules"] = _lists_to_tuples(raw_keys["logical_axis_rules"])
    raw_keys["data_sharding"] = _lists_to_tuples(raw_keys["data_sharding"])

    validate_keys(raw_keys)

  @staticmethod
  def configure_gpt3_task(raw_keys):
    '''dynamically configure gpt3 task based on training rules'''
    # follow https://github.com/google/paxml/blob/19db52eed85ae0d2365339b83a97cd0b873bbf73/paxml/tasks/lm/params/c4.py#L280
    #   according to training_rules of mlperf gpt3 training
    global_batch_size = calculate_global_batch_sizes(raw_keys)[1]
    if global_batch_size <= 3584:
      raw_keys['learning_rate'] = 2e-5
    else:
      raw_keys['learning_rate'] = 3e-5
    warmup_steps = math.ceil(265.0 * 1536 / global_batch_size - 1e-6)
    decay_end_step = math.ceil(108600.0 * 1536 / global_batch_size - 1e-6)
    raw_keys['learning_rate_schedule_steps'] = decay_end_step
    raw_keys['warmup_steps_fraction'] = warmup_steps / decay_end_step
    global_batch_size_to_train_on = calculate_global_batch_sizes(raw_keys)[1]
    raw_keys['eval_interval'] = math.ceil(24567 / global_batch_size_to_train_on)

  @staticmethod
  def update_model_vars(raw_keys):
    ''' Update model config variables
    '''
    validate_model_name(raw_keys['model_name'])
    max_logging.log(f"Running Model: {raw_keys['model_name']}")

    updated_keys = []
    if raw_keys['model_name'] != 'default':
      dir_path = os.path.dirname(os.path.realpath(__file__))
      file_path = os.path.join(dir_path, f"configs/models/{raw_keys['model_name']}.yml")
      with open(file_path, 'r', encoding="utf-8") as file:
        model_vars = yaml.safe_load(file)
        updated_keys = list(model_vars.keys())
      raw_keys = validate_and_update_keys(raw_keys, model_vars)
    return updated_keys

def validate_and_update_keys(raw_keys, model_keys):
  ''' Validate and update model specific config keys
  '''
  max_logging.log("Updating following parameters in config\n")
  for k in model_keys:
    max_logging.log(f"{k}: {model_keys[k]}")
    if k not in raw_keys:
      raise ValueError(f'Key {k} does not exist in config/base.yml.')
    elif not isinstance(raw_keys[k], type(model_keys[k])):
      raise ValueError(f'Type of key:{k} does not match with {type(model_keys[k])}')
    else:
      raw_keys[k] = model_keys[k]
  return raw_keys

def get_individual_scales(scale):
  '''Choose appropriate scales for individual dimensions based on global scale
  We choose to rotate between doubling:
    num_head and mlp_dim
    embed_dim
    num_layers
  Any one of these steps is not a perfect doubling, although going through a cycle
  of three is a near perfect 8x scaling except for the linear -> softmax -> output step'''


  log_2_scale = math.floor((math.log2(scale)))
  if 2**log_2_scale != scale:
    raise ValueError("Global parameter scale should be a power of 2. If you want finer grained control of the model sizes "
      "then you can explicitly set base_embed_dim, base_num_heads, base_mlp_dim, base_num_decoder_layers and/or head_dim.")
  base_scale, rem = divmod(log_2_scale, 3)
  num_head_scale = base_scale + int(rem > 0)
  mlp_dim_scale = num_head_scale
  emb_scale = base_scale + int(rem > 1)
  layer_scale = base_scale
  return emb_scale, num_head_scale, mlp_dim_scale, layer_scale

def calculate_global_batch_sizes(raw_keys):
  """ Calculates target global batch size from target devices and per_device_batch"""
  per_device_batch_size = raw_keys['per_device_batch_size']
  num_devices = get_num_target_devices(raw_keys)
  if per_device_batch_size < 1.0:
    # For per_device_batch_size<1, we load the data as if per_device_batch_size=1
    global_batch_size_to_load = num_devices
  else:
    global_batch_size_to_load = int(num_devices * per_device_batch_size)

  global_batch_size_to_train_on = int(num_devices * per_device_batch_size)
  return global_batch_size_to_load, global_batch_size_to_train_on

def get_num_target_devices(raw_keys):
  compile_topology = accelerator_to_spec_map.get_system_characteristics(raw_keys.get('compile_topology', ""))
  if compile_topology is not None:
    devices_per_slice = compile_topology.devices_per_slice
    return int(devices_per_slice * raw_keys['compile_topology_num_slices'])
  else:
    return len(jax.devices())

def get_num_slices(raw_keys):
  if int(raw_keys['compile_topology_num_slices']) > 0:
    return raw_keys['compile_topology_num_slices']
  else:
    devices = jax.devices()
    try:
      return 1 + max([d.slice_index for d in devices])
    except:
      return 1

def get_quantization_local_shard_count(raw_keys):
  if raw_keys['quantization_local_shard_count'] == -1:
    return raw_keys['num_slices']
  else:
    return raw_keys['quantization_local_shard_count']

class HyperParameters(): # pylint: disable=missing-class-docstring
  def __init__(self):
    pass

  def __getattr__(self, attr):
    if attr not in _config.keys:
      raise ValueError(f"Requested key {attr}, not in config")
    return _config.keys[attr]

  def __setattr__(self, attr, value):
    raise ValueError

  def get_keys(self):
    return _config.keys

def initialize(argv, **kwargs):
  global _config, config
  _config = _HyperParameters(argv, **kwargs)
  config = HyperParameters()

if __name__ == "__main__":
  initialize(sys.argv)
  print(config.steps)
  r = range(config.steps)
