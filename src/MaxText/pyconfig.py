# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pytype: skip-file
# pylint: disable=missing-module-docstring, bare-except, consider-using-generator, missing-function-docstring
from collections import OrderedDict
from typing import Any
from math import prod
import math
import os
import sys
import datetime

import jax
from jax.experimental.compilation_cache import compilation_cache
from jax.tree_util import register_pytree_node_class

import omegaconf

from MaxText import accelerator_to_spec_map
from MaxText import max_logging
from MaxText import max_utils
from MaxText.common_types import DecoderBlockType
from MaxText.globals import MAXTEXT_ASSETS_ROOT, MAXTEXT_REPO_ROOT, MAXTEXT_PKG_DIR
from MaxText.layers.attentions import AttentionType
from MaxText.utils import gcs_utils


# pylint: disable=line-too-long

_MAX_PREFIX = "M_"

# YAML attribute to specify inheritance.
_BASE_CONFIG_ATTR = "base_config"


def yaml_key_to_env_key(s: str) -> str:
  return _MAX_PREFIX + s.upper()


def string_to_bool(s: str) -> bool:
  if s.lower() == "true":
    return True
  if s.lower() == "false":
    return False
  raise ValueError(f"Can't convert {s} to bool")


_yaml_types_to_parser = {str: str, int: int, float: float, bool: string_to_bool}


def validate_compute_axis_order(s: str) -> None:
  valid_compute_axis_order = ("0,1,2,3", "0,2,1,3")
  if s not in valid_compute_axis_order:  # currently supported compute_axis_order
    raise ValueError("Invalid compute_axis_order was passed. Valid options are ", valid_compute_axis_order)


def validate_kv_quant_axis(s: str, quantize_kvcache: bool) -> None:
  valid_kv_quant_axis = ("", "dkv", "heads_and_dkv")
  if s not in valid_kv_quant_axis:  # currently supported kv_quant_axis
    raise ValueError("Invalid kv_quant_axis was passed. Valid options ", valid_kv_quant_axis)
  if quantize_kvcache and s == "":
    raise ValueError("kv_quant_axis cannot be '' when quantize_kvcache is True")


def validate_attention_kernel(s: str) -> None:
  valid_attention_kernels = ("autoselected", "dot_product", "flash", "cudnn_flash_te", "cudnn_flash_jax", "paged")
  if s not in valid_attention_kernels:  # currently supported attention
    raise ValueError("Invalid attention kernel was passed. Valid options ", valid_attention_kernels)


def validate_attention_type(s: str) -> None:
  valid_attention_types = (attention_type.value for attention_type in AttentionType)
  if s not in valid_attention_types:  # currently supported attention
    raise ValueError("Invalid attention type was passed. Valid options ", valid_attention_types)


def validate_attention_window_params(
    attention_type: str,
    chunk_attn_window_size: int,
    sliding_window_size: int,
) -> None:
  """
  Validates window size parameters for attention types 'chunk' and 'local'.
  """
  if attention_type == AttentionType.CHUNK.value:
    # Validate chunk_attn_window_size for 'chunk' attention
    if not isinstance(chunk_attn_window_size, int) or chunk_attn_window_size <= 0:
      raise ValueError(
          f"When attention_type is 'chunk', chunk_attn_window_size must be an integer greater than 0. "
          f"Got: {chunk_attn_window_size}"
      )
  elif attention_type == AttentionType.LOCAL_SLIDING.value:
    if not isinstance(sliding_window_size, int) or sliding_window_size <= 0:
      raise ValueError(
          f"When attention_type is 'local', sliding_window_size must be an integer greater than 0. "
          f"Got: {sliding_window_size}"
      )


def validate_profiler_type(s: str) -> None:
  valid_profiler_types = ("", "nsys", "xplane")
  if s not in valid_profiler_types:  # currently supported attention
    raise ValueError("Invalid profiler type was passed. Valid options ", valid_profiler_types)


def validate_periodic_profiler(profiler, profile_periodically_period, profiler_steps):
  if profile_periodically_period <= 0:
    return
  if not profiler:
    raise ValueError("Periodic profiler requested but no profiler was set, set it via profiler=xplane or profiler=nsys")
  if profile_periodically_period < profiler_steps:
    raise ValueError(
        f"You must set the profile_periodically_period {profile_periodically_period} at least as long as profiler_steps {profiler_steps}."
    )


def validate_model_call_mode(s: str) -> None:
  valid_model_call_modes = ("", "inference")
  if s not in valid_model_call_modes:  # currently supported attention
    raise ValueError(f"Invalid model call mode {s}. Valid options are {valid_model_call_modes}")


def validate_prefill_and_target_lengths(max_prefill_length: int, max_target_length: int) -> None:
  if max_prefill_length <= 0:
    raise ValueError(f"Invalid max_prefill_predict_length {max_prefill_length}, it should be a positive number")
  if max_target_length < max_prefill_length:
    # valid max_target_length = max_prefill_length for existing logit checks
    raise ValueError(
        f"Invalid max_target_length {max_target_length}, this should be the sum of "
        f"max_prefill_predict_length ({max_prefill_length}) and the expected max output length."
    )


def validate_rope_type(rope_type: str) -> None:
  valid_rope_types = ("default", "yarn", "llama3.1")
  if rope_type not in valid_rope_types:
    raise ValueError(f"Invalid RoPE type was passed. Got: {rope_type}. Valid options: {valid_rope_types}")


def validate_expert_shard_attention_option(expert_shard_attention_option: str) -> None:
  valid_expert_shard_attention_option = ("fsdp", "context")
  if expert_shard_attention_option not in valid_expert_shard_attention_option:
    raise ValueError(
        f"Invalid expert_shard_attention_option was passed. Got: {expert_shard_attention_option}. Valid options: {valid_expert_shard_attention_option}"
    )
    
    
def validate_vocab_tiling(num_vocab_tiling: int, per_device_batch_size: int, max_target_length: int, enable_nnx: bool):
  if (per_device_batch_size * max_target_length) % num_vocab_tiling != 0:
    raise ValueError(
      "Per device batch size times sequence length should be divisible by the number of vocab tiles."
    )
  if num_vocab_tiling > 1 and enable_nnx: #TODO (chengnuojin) enable vocab tiling on NNX after NNX migration
    raise ValueError(
      "We currently don't support vocab tiling on NNX module."
    )


def validate_keys(keys):
  validate_attention_kernel(keys["attention"])
  validate_attention_type(keys["attention_type"])
  validate_attention_window_params(
      keys["attention_type"], keys.get("chunk_attn_window_size"), keys.get("sliding_window_size")
  )
  validate_profiler_type(keys["profiler"])
  validate_periodic_profiler(keys["profiler"], keys["profile_periodically_period"], keys["profiler_steps"])
  validate_compute_axis_order(keys["compute_axis_order"])
  validate_kv_quant_axis(keys["kv_quant_axis"], keys["quantize_kvcache"])
  validate_model_call_mode(keys["model_call_mode"])
  validate_prefill_and_target_lengths(keys["max_prefill_predict_length"], keys["max_target_length"])
  validate_rope_type(keys["rope_type"])
  validate_vocab_tiling(keys["num_vocab_tiling"], keys["per_device_batch_size"], keys["max_target_length"], keys["enable_nnx"])

  # TODO remove after b/435512699 resolved
  if keys["context_parallel_size"] > 1 and keys["context_parallel_load_balance"] and keys["attention_type"] == "chunk":
    raise ValueError("Currently load-balanced context parallelism is not supported for chunk attention.")

  if keys["mtp_eval_target_module"] < 0:
    raise ValueError("mtp_eval_target_module cannot be negative. Set to 0 to disable evaluation.")

  if keys["mtp_num_layers"] > 0 and keys["model_call_mode"] == "inference":
    raise ValueError(
        "Multi-Token Prediction (MTP) is enabled (mtp_num_layers > 0), but it is not supported in inference mode. "
        "Please disable MTP by setting mtp_num_layers=0 for inference."
    )

  assert (keys["load_parameters_path"] == "" and keys["load_full_state_path"] == "") or keys[
      "enable_checkpointing"
  ], "You must set enable_checkpointing to load a checkpoint"
  assert (
      keys["load_parameters_path"] == "" or keys["load_full_state_path"] == ""
  ), "At most one of `load_parameters_path` or `load_full_state_path` should be set"

  if keys["enable_multi_tier_checkpointing"]:
    assert (
        keys["local_checkpoint_directory"]
    ), "A local checkpoint directory must be specified when using multi-tier checkpointing"
    assert (keys["local_checkpoint_period"] > 0), "A positive local checkpoint period must be specified when using multi-tier checkpointing"
    assert (
        keys["multi_tier_checkpointing_backup_interval_minutes"] > 0
    ), "A positive multi-tier checkpointing backup interval minutes must be specified when using multi-tier checkpointing"

  if keys["enable_emergency_checkpoint"]:
    assert (
        keys["local_checkpoint_directory"] != ""
    ), "A local checkpoint directory must be specified when using emergency checkpoint"
    assert (
        keys["local_checkpoint_period"] > 0
    ), "A positive local checkpoint period must be specified when using emergency checkpoint"

  else:
    max_logging.log(
        "Not using emergency checkpoint, ignoring local_checkpoint_directory, local_checkpoint_period,"
        " use_replicator_service and replicator_backup_interval_minutes"
    )

  validate_multiple_slices(keys)
  if keys["num_experts"] > 1:
    validate_mlp_dim(keys)
    validate_sparse_matmul_parallelism(keys)
    validate_ring_of_experts_parallelism(keys)
    validate_shard_fsdp_on_expert_parallelism(keys)
    validate_ragged_dot(keys)
    validate_deepseek_moe(keys)
    validate_gpt_oss_moe(keys)
    validate_expert_shard_attention_option(keys["expert_shard_attention_option"])

  if keys["use_multimodal"]:
    validate_multimodal_model_name(keys["model_name"])
    if keys["use_sft"]:
      assert keys[
          "sft_train_on_completion_only"
      ], "In multimodal SFT (use_multimodal=True, use_sft=True), sft_train_on_completion_only must be set to True"
      # TODO(aireenmei, hengtaoguo): support packing
      assert not keys["packing"], "In multimodal SFT (use_multimodal=True, use_sft=True), packing is not supported yet"

  if keys["decoder_block"] == "llama4":
    validate_llama4_config(keys)


def validate_tokenizer(keys):
  assert keys[
      "tokenizer_path"
  ], "Please provide tokenizer_path. Even if using pre-tokenized data, tokenizer is required to process special tokens."


def validate_constant_bound(keys):
  if keys["constant_bound_config"] == "":
    keys["constant_bound_config"] = []
  else:
    value_list = keys["constant_bound_config"].split(",")
    keys["constant_bound_config"] = list(map(float, value_list))
  assert (
      len(keys["constant_bound_config"]) == 0 or len(keys["constant_bound_config"]) == 6
  ), "Please specify competete constant bound or none"


def validate_quantization_methods(keys):
  """Validate quantization methods
  """
  valid_quant_methods = (
    "", "int8", "fp8", "fp8_full", "fp8_gpu", "fp8_nanoo"
  )
  if keys["use_qwix_quantization"]:
    if keys["quantization"] not in valid_quant_methods:
      raise ValueError(
          f"Invalid quantization method {keys['quantization']}. Valid options are {valid_quant_methods}"
      )


def validate_data_input(keys):
  """validate provided parameters for data input"""
  if not keys["hf_access_token"]:
    keys["hf_access_token"] = None
  if keys["dataset_type"] == "hf":
    max_logging.log(
        f"dataset_type set to hf, will use {keys['hf_path']=}, {keys['hf_data_dir']=} and {keys['hf_train_files']=} to read data"
    )
    assert keys["hf_path"] != "", "hf_path can't be empty when dataset_type=hf"
    if not keys["hf_train_files"]:
      keys["hf_train_files"] = None
    if not keys["hf_eval_files"]:
      keys["hf_eval_files"] = None
    if keys["hf_eval_files"]:
      keys["hf_eval_split"] = "train"
    if keys["eval_interval"] > 0:
      assert keys["hf_eval_split"], "Please specify hf_eval_split or set eval_interval to <=0."
    assert keys["num_epoch"] == 1, f"hf pipeline only supports num_epoch=1, but num_epoch={keys['num_epoch']} is given."

  elif keys["dataset_type"] == "grain":
    max_logging.log(
        f"dataset_type set to grain, will use {keys['grain_train_files']=} as data files, and {keys['grain_worker_count']} workers"
    )
    assert keys["grain_train_files"] != "", "grain_train_files can't be empty when dataset_type=grain"
    if keys["eval_interval"] > 0:
      assert keys["grain_eval_files"], "Please specify grain_eval_files or set eval_interval to <=0."
    assert keys["tokenizer_type"] in (
        "sentencepiece",
        "huggingface",
    ), f"grain pipeline only supports tokenizer_type: sentencepiece, huggingface, but got {keys['tokenizer_type']}"
  elif keys["dataset_type"] == "tfds":
    max_logging.log(f"dataset_type set to tfds, will use {keys['dataset_path']=} and {keys['dataset_name']=}")
    assert keys["dataset_name"] != "", "dataset_name can't be empty when dataset_type=tfds"
    if keys["eval_interval"] > 0:
      assert keys["eval_split"], "Please specify eval_split or set eval_interval to <=0."

  if "tokenizer_llama3.tiktoken" in keys["tokenizer_path"]:
    assert (
        keys["tokenizer_type"] == "tiktoken"
    ), "tokenizer_type must be tiktoken when using tokenizer=tokenizer_llama3.tiktoken"

  if keys["sharding_tolerance"] > 1.0 or keys["sharding_tolerance"] < 0.0:
    max_logging.log(
        "WARNING: 'sharding_tolerance: allowed percentage of non-sharded parameters' should be between 0.0 and 1.0"
    )

  if keys["eval_interval"] > 0 and keys["generate_padding_batch_eval"]:
    assert keys["eval_steps"] > 0, "eval_steps must be > 0 when generate_padding_batch_eval is True"


def validate_llama4_config(keys: dict):
  """
  Validates the following checks for Llama4 models:

  Args:
    keys: the raw config in dict form

  """
  if keys["capacity_factor"] >= 0:
    raise ValueError(
        "Llama4 decoder has not been tested with capacity_factor >= 0 -- please set that value to -1 for now!"
    )
  if keys["num_experts_per_tok"] > 1:
    raise ValueError("Only top-1 routing is supported for Llama4 for now!")
  if keys["base_num_decoder_layers"] % keys["interleave_moe_layer_step"] != 0:
    raise ValueError(
        f"The number of decoder layers ({keys['base_num_decoder_layers']}) must be divisible by interleave moe layer step ({keys['interleave_moe_layer_step']})"
    )


def validate_model_name(s: str) -> bool:
  """Validate provided model name."""
  # currently supported models
  valid_model_names = (
      "default",
      "llama2-7b",
      "llama2-13b",
      "llama2-70b",
      "llama3-8b",
      "llama3-70b",
      "llama3.1-8b",
      "llama3.1-70b",
      "llama3.1-405b",
      "llama3.3-70b",
      "mistral-7b",
      "mixtral-8x7b",
      "mixtral-8x22b",
      "deepseek2-16b",
      "deepseek2-236b",
      "deepseek3-671b",
      "deepseek3-test",
      "kimi-k2-1t",
      "gemma-7b",
      "gemma-2b",
      "gemma2-2b",
      "gemma2-9b",
      "gemma2-27b",
      "gemma3-4b",
      "gemma3-12b",
      "gemma3-27b",
      "qwen3-0.6b",
      "qwen3-4b",
      "qwen3-8b",
      "qwen3-14b",
      "qwen3-32b",
      "qwen3-235b-a22b",
      "qwen3-30b-a3b",
      "qwen3-480b-a35b",
      "gpt3-175b",
      "gpt3-22b",
      "gpt3-6b",
      "gpt3-52k",
      "gpt-oss-20b",
      "gpt-oss-120b",
      "llama4-17b-16e",
      "llama4-17b-128e",
  )
  if s not in valid_model_names:
    raise ValueError(f"Invalid model name was passed. Got {s}, Valid options {valid_model_names}")


def validate_multimodal_model_name(s: str) -> bool:
  valid_model_names = (
      "gemma3-4b",
      "gemma3-12b",
      "gemma3-27b",
      "llama4-17b-16e",
      "llama4-17b-128e",
  )
  if s not in valid_model_names:
    raise ValueError(
        f"Invalid multimodal model name was passed. Got {s}. Valid options which support multimodal inputs are: {valid_model_names}"
    )


def validate_no_keys_overwritten_twice(keys1: list[str], keys2: list[str]):
  overwritten_keys = [k for k in keys1 if k in keys2]
  if overwritten_keys:
    raise ValueError(
        f"Keys {overwritten_keys} are overwritten from both the model"
        " and the environment/command line. This isn't allowed."
    )


def validate_and_assign_remat_tensors(keys):
  # list of allowed tensors for custom remat policy
  tensors = [
      "decoder_layer_input",
      "context",
      "mlpwi",
      "mlpwi_0",
      "mlpwi_1",
      "mlpwo",
      "query_proj",
      "key_proj",
      "value_proj",
      "out_proj",
  ]
  assert keys["decoder_layer_input"] != "remat", "Cannot remeterialize this tensor with scan_layers=True"
  tensors_on_device = []
  tensors_to_offload = []
  for t in tensors:
    if keys[t] == "device":
      tensors_on_device.append(t)
    elif keys[t] == "offload":
      tensors_to_offload.append(t)
    elif keys[t] != "remat":
      raise ValueError(f"Invalid value chosen for tensor {t}")
  keys["tensors_on_device"] = tensors_on_device
  keys["tensors_to_offload"] = tensors_to_offload
  return keys


def _lists_to_tuples(l: list[Any]) -> tuple[Any] | list[Any]:
  return tuple(_lists_to_tuples(x) for x in l) if isinstance(l, list) else l


# TODO: remove in future when MaxText commands are no longer used
def resolve_config_path(param: str) -> str:
    """Resolve config path to auto rewrite to use new src folder.
    This ensures backwards compatibility with older versions of MaxText."""
    return param if os.path.isfile(param) else os.path.join("src", param)


class _HyperParameters:
  # pylint: disable=missing-class-docstring
  # This class is responsible for loading, merging, and overriding the configuration.

  def _validate_env_variables(self, raw_data_from_yaml: dict[str, Any]):
    for environment_var in os.environ:
      if environment_var[: len(_MAX_PREFIX)] == _MAX_PREFIX:
        proposed_key = environment_var[len(_MAX_PREFIX) :].lower()
        if proposed_key not in raw_data_from_yaml:
          raise ValueError(f"We received env `{environment_var}` but it doesn't match a key, so it is assumed a mistake.")
        if not environment_var[len(_MAX_PREFIX) :].isupper():
          raise ValueError(f"We received env `{environment_var}` but it isn't all uppercase.")

  def _update_from_env_and_command_line(self, raw_keys, raw_data_from_yaml, argv, **kwargs) -> list[str]:
    """Update model config from environment and command line using omegaconf.OmegaConf overrides."""
    # Use omegaconf.OmegaConf.from_cli to capture CLI arguments.
    cli_cfg = omegaconf.OmegaConf.from_cli(argv[2:])
    # Also create a configuration from any extra keyword arguments.
    kwargs_cfg = omegaconf.OmegaConf.create(kwargs)
    # Merge command-line and keyword arguments.
    cmdline_cfg = omegaconf.OmegaConf.merge(cli_cfg, kwargs_cfg)
    raw_data_from_cmd_line = omegaconf.OmegaConf.to_container(cmdline_cfg, resolve=True)

    updated_keys = []

    for k in raw_data_from_cmd_line:
      if k not in raw_data_from_yaml:
        raise ValueError(f"Key {k} was passed at the command line but isn't in config.")

    for k in raw_data_from_yaml:
      if k in raw_data_from_cmd_line and yaml_key_to_env_key(k) in os.environ:
        raise ValueError(f"You are passing overrides by both CLI and ENV for `{k}`. This isn't allowed.")

      if k not in raw_data_from_cmd_line and yaml_key_to_env_key(k) not in os.environ:
        raw_keys[k] = raw_data_from_yaml[k]
        continue

      updated_keys.append(k)
      if k in raw_data_from_cmd_line:
        new_proposal = raw_data_from_cmd_line[k]
      else:
        new_proposal = os.environ.get(yaml_key_to_env_key(k))

      if (not isinstance(new_proposal, type(raw_data_from_yaml[k]))) and (
          type(raw_data_from_yaml[k]) not in _yaml_types_to_parser
      ):
        raise ValueError(
            f"For key '{k}', type {type(raw_data_from_yaml[k])} not in {_yaml_types_to_parser.keys()}, can't pass"
            " at the CLI or ENV"
        )

      if new_proposal is None:
        raw_keys[k] = None  # This allows users to set empty strings via CLI, otherwise parsed as "None" - b/405981568
      elif isinstance(new_proposal, type(raw_data_from_yaml[k])):
        raw_keys[k] = new_proposal  # take the raw data, no type conversion
      else:
        try:
          raw_keys[k] = _yaml_types_to_parser[type(raw_data_from_yaml[k])](
              new_proposal
          )  # take the command line value, but type it like the config value.
        except ValueError as e:
          raise ValueError(f"Couldn't parse value from CLI or ENV '{new_proposal}' for key '{k}'") from e

    return updated_keys

  def _load_config(self, config_name: str) -> dict[str, Any]:
    """Loads the YAML config from a file using omegaconf.OmegaConf, and resolves inheritance."""
    base_cfg = omegaconf.OmegaConf.load(config_name)
    raw_data_from_yaml = omegaconf.OmegaConf.to_container(base_cfg, resolve=True)

    # Load data from parent config. Note that inheritance has override
    # semantics, and the path is relative to the current config.
    if _BASE_CONFIG_ATTR in raw_data_from_yaml:
      parent_config_filename = raw_data_from_yaml[_BASE_CONFIG_ATTR]
      if not os.path.isabs(parent_config_filename):
        loaded_parent_config_filename = os.path.join(os.path.dirname(config_name), parent_config_filename)
        if not os.path.isfile(loaded_parent_config_filename):
          dir_path = os.path.dirname(os.path.realpath(__file__))
          loaded_parent_config_filename = os.path.join(dir_path, "configs", parent_config_filename)
      else:
        loaded_parent_config_filename = parent_config_filename

      base_config = self._load_config(loaded_parent_config_filename)
      # Override base_config with values from raw_data_from_yaml.
      for key, value in raw_data_from_yaml.items():
        base_config[key] = value
      return base_config
    return raw_data_from_yaml

  def __init__(self, argv: list[str], **kwargs):
    config_name: str = resolve_config_path(argv[1])
    raw_data_from_yaml = self._load_config(config_name)

    self._validate_env_variables(raw_data_from_yaml)

    raw_keys = OrderedDict()
    keys_from_env_and_command_line = self._update_from_env_and_command_line(raw_keys, raw_data_from_yaml, argv, **kwargs)
    max_logging.log(f"Updating keys from env and command line: {keys_from_env_and_command_line}")
    keys_from_model = _HyperParameters.update_model_vars(argv[1], raw_keys, config_name, keys_from_env_and_command_line)
    max_logging.log(f"Updating keys from model: {keys_from_model}")
    if not raw_keys["override_model_config"]:
      validate_no_keys_overwritten_twice(keys_from_env_and_command_line, keys_from_model)

    # This must be invoked before initializing the backend
    raw_keys = validate_and_set_hlo_dump_defaults(raw_keys)

    # We initialize the jax distributed system here because it must be done before device backend is initialized.
    if raw_keys["jax_debug_log_modules"]:
      jax.config.update("jax_debug_log_modules", raw_keys["jax_debug_log_modules"])
    max_utils.maybe_initialize_jax_distributed_system(raw_keys)

    if raw_keys["jax_cache_dir"]:
      compilation_cache.set_cache_dir(os.path.expanduser(raw_keys["jax_cache_dir"]))

    _HyperParameters.user_init(raw_keys)
    if raw_keys["dataset_type"] == "c4_mlperf" and raw_keys["model_name"] == "gpt3-175b":
      _HyperParameters.configure_gpt3_task(raw_keys)

    if raw_keys["dataset_type"] == "c4_mlperf" and raw_keys["model_name"] != "gpt3-175b":
      _HyperParameters.configure_c4_mlperf_task(raw_keys)

    if not os.path.isfile(raw_keys["tokenizer_path"]):
      # Try and find the tokenizer path relative to the config file.
      for search_root in (
          MAXTEXT_ASSETS_ROOT,
          os.path.dirname(MAXTEXT_ASSETS_ROOT),
          os.path.join(MAXTEXT_REPO_ROOT, "assets"),
          MAXTEXT_REPO_ROOT,
          os.path.join(MAXTEXT_REPO_ROOT, "src", "MaxText"),
          MAXTEXT_PKG_DIR,
          os.path.dirname(config_name)
      ):
        tokenizer_path = os.path.join(
            search_root,
            raw_keys["tokenizer_path"],
        )

        if os.path.isfile(tokenizer_path):
          raw_keys["tokenizer_path"] = tokenizer_path
          break

    self.keys = raw_keys
    keys = [k for k in raw_keys]  # pylint: disable=unnecessary-comprehension
    keys.sort()

    if raw_keys["log_config"]:
      for k in keys:
        if k != "hf_access_token":
          max_logging.log(f"Config param {k}: {raw_keys[k]}")

  @staticmethod
  def user_init(raw_keys):
    """Transformations between the config data and configs used at runtime"""
    if raw_keys["run_name"] == "":
      raw_keys["run_name"] = os.environ.get("JOBSET_NAME")  # using XPK default
      if raw_keys["run_name"] == "":
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y-%m-%d-%H-%M")
        raw_keys["run_name"] = f'{raw_keys["model_name"]}_{timestamp}'
    run_name = raw_keys["run_name"]
    base_output_directory = raw_keys["base_output_directory"]
    if run_name:
      raw_keys["tensorboard_dir"] = os.path.join(base_output_directory, run_name, "tensorboard", "")
      raw_keys["checkpoint_dir"] = os.path.join(base_output_directory, run_name, "checkpoints", "")
      raw_keys["metrics_dir"] = os.path.join(base_output_directory, run_name, "metrics", "")

    if raw_keys["learning_rate_schedule_steps"] == -1:
      raw_keys["learning_rate_schedule_steps"] = raw_keys["steps"]
    if raw_keys["steps"] == -1:
      raw_keys["steps"] = raw_keys["learning_rate_schedule_steps"]

    if raw_keys["attn_logits_soft_cap"] == 0.0:
      raw_keys["attn_logits_soft_cap"] = None
    if raw_keys["final_logits_soft_cap"] == 0.0:
      raw_keys["final_logits_soft_cap"] = None

    emb_scale, num_head_scale, mlp_dim_scale, layer_scale = get_individual_scales(raw_keys["global_parameter_scale"])
    raw_keys["emb_dim"] = 2**emb_scale * raw_keys["base_emb_dim"]
    raw_keys["num_query_heads"] = 2**num_head_scale * raw_keys["base_num_query_heads"]
    raw_keys["num_kv_heads"] = 2**num_head_scale * raw_keys["base_num_kv_heads"]
    raw_keys["mlp_dim"] = 2**mlp_dim_scale * raw_keys["base_mlp_dim"]
    raw_keys["moe_mlp_dim"] = 2**mlp_dim_scale * raw_keys["base_moe_mlp_dim"]
    raw_keys["num_decoder_layers"] = 2**layer_scale * raw_keys["base_num_decoder_layers"]

    # This is the first command that initializes the backend - it calls
    # jax.devices()
    (
        raw_keys["global_batch_size_to_load"],
        raw_keys["global_batch_size_to_train_on"],
        raw_keys["micro_batch_size_to_train_on"],
    ) = calculate_global_batch_sizes(
        raw_keys["per_device_batch_size"],
        raw_keys["expansion_factor_real_data"],
        get_num_target_devices(raw_keys),
        raw_keys["gradient_accumulation_steps"],
    )

    if raw_keys["eval_per_device_batch_size"] <= 0:
      raw_keys["eval_per_device_batch_size"] = raw_keys["per_device_batch_size"]

    (
        raw_keys["global_batch_size_to_load_eval"],
        raw_keys["global_batch_size_to_eval_on"],
        raw_keys["micro_batch_size_to_eval_on"],
    ) = calculate_global_batch_sizes(
        raw_keys["eval_per_device_batch_size"],
        raw_keys["expansion_factor_real_data"],
        get_num_target_devices(raw_keys),
        1,
    )

    if raw_keys["pagedattn_max_pages_per_group"] <= 0:
      raw_keys["pagedattn_max_pages_per_group"] = (
          raw_keys["max_target_length"] + raw_keys["pagedattn_tokens_per_page"] - 1
      ) // raw_keys["pagedattn_tokens_per_page"]

    raw_keys["num_slices"] = max_utils.get_num_slices(raw_keys)
    raw_keys["quantization_local_shard_count"] = get_quantization_local_shard_count(raw_keys)
    raw_keys["context_parallel_size"] = get_context_parallel_size(raw_keys)
    raw_keys = create_parallelisms_list(raw_keys)
    raw_keys = set_and_validate_pipeline_config(raw_keys)

    if raw_keys["dataset_type"] == "c4_mlperf":
      raw_keys["add_bos"] = False
      raw_keys["add_eos"] = False
      max_logging.log("Override add_bos and add_eos to False when dataset_type=c4_mlperf")

    # Write raw_keys to GCS before type conversions
    gcs_utils.write_config_raw_keys_for_gcs(raw_keys)

    # Type conversions
    raw_keys["dtype"] = jax.numpy.dtype(raw_keys["dtype"])
    raw_keys["weight_dtype"] = jax.numpy.dtype(raw_keys["weight_dtype"])
    raw_keys["mu_dtype"] = set_mu_dtype(raw_keys)
    raw_keys["logical_axis_rules"] = _lists_to_tuples(raw_keys["logical_axis_rules"])
    raw_keys["data_sharding"] = _lists_to_tuples(raw_keys["data_sharding"])

    if raw_keys["remat_policy"] == "custom":
      raw_keys = validate_and_assign_remat_tensors(raw_keys)
    validate_keys(raw_keys)
    validate_tokenizer(raw_keys)
    validate_data_input(raw_keys)
    validate_constant_bound(raw_keys)
    validate_quantization_methods(raw_keys)

    raw_keys["decoder_block"] = DecoderBlockType(raw_keys["decoder_block"])

  @staticmethod
  def configure_gpt3_task(raw_keys):
    """dynamically configure gpt3 task based on training rules"""
    # follow https://github.com/google/paxml/blob/19db52eed85ae0d2365339b83a97cd0b873bbf73/paxml/tasks/lm/params/c4.py#L280
    #   according to training_rules of mlperf gpt3 training
    global_batch_size_to_train_on = raw_keys["global_batch_size_to_train_on"]
    if global_batch_size_to_train_on <= 3584:
      raw_keys["learning_rate"] = 2e-5
    else:
      raw_keys["learning_rate"] = 3e-5
    warmup_steps = math.ceil(265.0 * 1536 / global_batch_size_to_train_on - 1e-6)
    decay_end_step = math.ceil(108600.0 * 1536 / global_batch_size_to_train_on - 1e-6)
    raw_keys["learning_rate_schedule_steps"] = decay_end_step
    raw_keys["warmup_steps_fraction"] = warmup_steps / decay_end_step
    raw_keys["eval_interval"] = math.ceil(24567 / global_batch_size_to_train_on)

  @staticmethod
  def configure_c4_mlperf_task(raw_keys):
    """dynamically configure based on training rules"""
    # follow https://github.com/google/paxml/blob/19db52eed85ae0d2365339b83a97cd0b873bbf73/paxml/tasks/lm/params/c4.py#L280
    #   according to training_rules of mlperf
    global_batch_size_to_train_on = raw_keys["global_batch_size_to_train_on"]
    global_batch_size_to_eval_on = raw_keys["global_batch_size_to_eval_on"]
    max_target_length = raw_keys["max_target_length"]

    learning_rate = (8.0e-5 * global_batch_size_to_train_on) / 1152
    warmup_steps = math.ceil(8000.0 * 1152 / global_batch_size_to_train_on - 1e-6)
    decay_end_step = math.ceil(1200000.0 * 1152 / global_batch_size_to_train_on - 1e-6)
    raw_keys["learning_rate"] = learning_rate
    raw_keys["learning_rate_schedule_steps"] = decay_end_step
    raw_keys["warmup_steps_fraction"] = warmup_steps / decay_end_step

    raw_keys["eval_steps"] = math.ceil(5760 * 8192 / max_target_length / global_batch_size_to_eval_on)
    raw_keys["eval_interval"] = math.ceil(377487360 / max_target_length / global_batch_size_to_train_on)

  @staticmethod
  def update_model_vars(base_config_path, raw_keys, config_name: str, keys_from_env_and_command_line):
    """Update model config variables"""
    validate_model_name(raw_keys["model_name"])
    max_logging.log(f"Running Model: {raw_keys['model_name']}")

    updated_keys = []
    if raw_keys["model_name"] != "default":
      model_name = raw_keys["model_name"]
      # First look at the model configs next to the base_config_path, and
      # fallback to the python codebase if the config cannot be found.
      file_path = os.path.join(os.path.dirname(base_config_path), "models", f"{model_name}.yml")
      if not os.path.isfile(file_path):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(dir_path, "configs", "models", f"{model_name}.yml")
      # Use omegaconf.OmegaConf to load the model-specific configuration.
      model_vars = omegaconf.OmegaConf.load(file_path)
      model_vars = omegaconf.OmegaConf.to_container(model_vars, resolve=True)
      if raw_keys["override_model_config"]:
        model_vars = {key: value for key, value in model_vars.items() if key not in keys_from_env_and_command_line}
      updated_keys = list(model_vars.keys())
      raw_keys = validate_and_update_keys(raw_keys, model_vars, config_name)
    return updated_keys


def create_parallelisms_list(raw_keys):
  ici_parallelism = [
      raw_keys["ici_data_parallelism"],
      raw_keys["ici_pipeline_parallelism"],
      raw_keys["ici_fsdp_parallelism"],
      raw_keys["ici_fsdp_transpose_parallelism"],
      raw_keys["ici_sequence_parallelism"],
      raw_keys["ici_context_parallelism"],
      raw_keys["ici_context_autoregressive_parallelism"],
      raw_keys["ici_tensor_parallelism"],
      raw_keys["ici_tensor_transpose_parallelism"],
      raw_keys["ici_tensor_sequence_parallelism"],
      raw_keys["ici_expert_parallelism"],
      raw_keys["ici_autoregressive_parallelism"],
  ]
  dcn_parallelism = [
      raw_keys["dcn_data_parallelism"],
      raw_keys["dcn_pipeline_parallelism"],
      raw_keys["dcn_fsdp_parallelism"],
      raw_keys["dcn_fsdp_transpose_parallelism"],
      raw_keys["dcn_sequence_parallelism"],
      raw_keys["dcn_context_parallelism"],
      raw_keys["dcn_context_autoregressive_parallelism"],
      raw_keys["dcn_tensor_parallelism"],
      raw_keys["dcn_tensor_transpose_parallelism"],
      raw_keys["dcn_tensor_sequence_parallelism"],
      raw_keys["dcn_expert_parallelism"],
      raw_keys["dcn_autoregressive_parallelism"],
  ]
  raw_keys["ici_parallelism"] = ici_parallelism
  raw_keys["dcn_parallelism"] = dcn_parallelism
  return raw_keys


def set_mu_dtype(raw_keys):
  # Default mu_dtype to weight_dtype if unset
  if raw_keys["mu_dtype"]:
    assert raw_keys["opt_type"] != "adam_pax", "opt_type adam_pax doesn't support explicitly setting mu_dtype"

  if raw_keys["mu_dtype"] == "":
    return raw_keys["weight_dtype"]
  else:
    return jax.numpy.dtype(raw_keys["mu_dtype"])


def validate_and_set_hlo_dump_defaults(raw_keys):
  if not raw_keys["dump_hlo"]:
    return raw_keys
  if os.environ.get("XLA_FLAGS") and raw_keys["dump_hlo_xla_flags"]:
    raise ValueError("You must set either XLA_FLAGS or dump_hlo_xla_flags to dump HLO, but not both.")
  if not os.environ.get("XLA_FLAGS") and not raw_keys["dump_hlo_xla_flags"]:
    raw_keys["dump_hlo_xla_flags"] = f"--xla_dump_to={raw_keys['dump_hlo_local_dir']} --xla_dump_large_constants"
    if raw_keys["dump_hlo_local_module_name"]:
      raw_keys["dump_hlo_xla_flags"] = (
          f"{raw_keys['dump_hlo_xla_flags']} --xla_dump_hlo_module_re={raw_keys['dump_hlo_local_module_name']}"
      )
  if not raw_keys["dump_hlo_gcs_dir"]:
    raw_keys["dump_hlo_gcs_dir"] = os.path.join(raw_keys["base_output_directory"], raw_keys["run_name"], "xla_dump")
  else:
    raw_keys["dump_hlo_gcs_dir"] = gcs_utils.add_trailing_slash(raw_keys["dump_hlo_gcs_dir"])
  if not os.environ.get("XLA_FLAGS"):
    os.environ["XLA_FLAGS"] = raw_keys["dump_hlo_xla_flags"]
  return raw_keys


def validate_multiple_slices(raw_keys):
  if (
      math.fabs(
          math.prod(
              [
                  raw_keys["dcn_data_parallelism"],
                  raw_keys["dcn_pipeline_parallelism"],
                  raw_keys["dcn_fsdp_parallelism"],
                  raw_keys["dcn_fsdp_transpose_parallelism"],
                  raw_keys["dcn_sequence_parallelism"],
                  raw_keys["dcn_context_parallelism"],
                  raw_keys["dcn_tensor_parallelism"],
                  raw_keys["dcn_tensor_sequence_parallelism"],
                  raw_keys["dcn_expert_parallelism"],
                  raw_keys["dcn_context_autoregressive_parallelism"],
                  raw_keys["dcn_autoregressive_parallelism"],
              ]
          )
      )
      > 1
  ):
    assert raw_keys["num_slices"] > 1, "DCN parallelism requested but only one slice available."


def set_and_validate_pipeline_config(raw_keys):
  if using_pipeline_parallelism(raw_keys):
    # For pipeline parallelism, model_fsdp_ag_once should be False, and pipeline_fsdp_ag_once is typically True.
    if raw_keys["model_fsdp_ag_once"]:
      raise ValueError(
          "You should only set pipeline_fsdp_once=True, leave model_fsdp_ag_once=False with pipeline parallelism."
      )

    def modify_activation_embed_and_logits_batch(logical_axis_rules):
      for idx, logical_rule in enumerate(logical_axis_rules):
        if logical_rule[0] == "activation_embed_and_logits_batch":
          # For pipeline parallelism the pre and post decoder layer tensors' batch dimension is sharded by stages.
          # Microbatches are sharded by stage, so moving out of and into this sharding should be a local reshape.
          # The "stage" needs to be listed first since the microbatch dimension is first before the reshape.
          logical_axis_rules[idx] = [
              "activation_embed_and_logits_batch",
              ["stage", "data", "fsdp", "fsdp_transpose", "expert"],
          ]
          break  # Exit the loop after modifying the list
      return logical_axis_rules

    def pipeline_first_axis(raw_keys):
      # We have seen better performance when axes used for DCN are earlier in this list than ICI, see (b/339009148) for details
      ici_parallelism = [
          raw_keys["ici_pipeline_parallelism"],
          raw_keys["ici_data_parallelism"],
          raw_keys["ici_fsdp_parallelism"],
          raw_keys["ici_fsdp_transpose_parallelism"],
          raw_keys["ici_sequence_parallelism"],
          raw_keys["ici_context_parallelism"],
          raw_keys["ici_context_autoregressive_parallelism"],
          raw_keys["ici_tensor_parallelism"],
          raw_keys["ici_tensor_transpose_parallelism"],
          raw_keys["ici_tensor_sequence_parallelism"],
          raw_keys["ici_expert_parallelism"],
          raw_keys["ici_autoregressive_parallelism"],
      ]
      dcn_parallelism = [
          raw_keys["dcn_pipeline_parallelism"],
          raw_keys["dcn_data_parallelism"],
          raw_keys["dcn_fsdp_parallelism"],
          raw_keys["dcn_fsdp_transpose_parallelism"],
          raw_keys["dcn_sequence_parallelism"],
          raw_keys["dcn_context_parallelism"],
          raw_keys["dcn_context_autoregressive_parallelism"],
          raw_keys["dcn_tensor_parallelism"],
          raw_keys["dcn_tensor_transpose_parallelism"],
          raw_keys["dcn_tensor_sequence_parallelism"],
          raw_keys["dcn_expert_parallelism"],
          raw_keys["dcn_autoregressive_parallelism"],
      ]
      mesh_axes = [
          "stage",
          "data",
          "fsdp",
          "fsdp_transpose",
          "sequence",
          "context",
          "context_autoregressive",
          "tensor",
          "tensor_transpose",
          "tensor_sequence",
          "expert",
          "autoregressive",
      ]
      data_sharding = [
          [
              "stage",
              "data",
              "fsdp",
              "fsdp_transpose",
              "sequence",
              "context",
              "context_autoregressive",
              "tensor",
              "tensor_transpose",
              "tensor_sequence",
              "expert",
              "autoregressive",
          ]
      ]

      raw_keys["ici_parallelism"] = ici_parallelism
      raw_keys["dcn_parallelism"] = dcn_parallelism
      raw_keys["mesh_axes"] = mesh_axes
      raw_keys["data_sharding"] = data_sharding
      return raw_keys

    raw_keys["using_pipeline_parallelism"] = True
    raw_keys["logical_axis_rules"] = modify_activation_embed_and_logits_batch(raw_keys["logical_axis_rules"])
    raw_keys = pipeline_first_axis(raw_keys)
    num_stages = int(raw_keys["ici_pipeline_parallelism"] * raw_keys["dcn_pipeline_parallelism"])
    if raw_keys["pipeline_parallel_layers"] == -1:
      if raw_keys["decoder_block"] == "deepseek":
        moe_layers = raw_keys["num_decoder_layers"] - raw_keys["first_num_dense_layers"]
        raw_keys["pipeline_parallel_layers"] = moe_layers
      else:
        raw_keys["pipeline_parallel_layers"] = raw_keys["num_decoder_layers"]
    else:
      if raw_keys["decoder_block"] == "deepseek":
        moe_layers = raw_keys["num_decoder_layers"] - raw_keys["first_num_dense_layers"]
        assert (
            raw_keys["pipeline_parallel_layers"] <= moe_layers
        ), f"You can only pipeline a subset of the moe decoder layers for deepseek, but you requested to pipeline {raw_keys['pipeline_parallel_layers']} with pipeline_parallel_layers and there are only {moe_layers} decoder layers."
      else:
        assert (
            raw_keys["pipeline_parallel_layers"] <= raw_keys["num_decoder_layers"]
        ), f"You can only pipeline a subset of the decoder layers, but you requested to pipeline {raw_keys['pipeline_parallel_layers']} with pipeline_parallel_layers and there are only {raw_keys['num_decoder_layers']} decoder layers."
    assert (
        raw_keys["scan_layers"] or raw_keys["pipeline_parallel_layers"] == raw_keys["num_decoder_layers"]
    ), "Currently we only support scan_layers=True when pipelining a subset of layers."
    assert (
        raw_keys["pipeline_parallel_layers"] > 0
    ), "You must set pipeline_parallel_layers to a positive integer less than or equal to the number of layers"

    if raw_keys["num_pipeline_repeats"] == -1:
      num_pipeline_repeats, remainder = divmod(
          raw_keys["pipeline_parallel_layers"], num_stages * raw_keys["num_layers_per_pipeline_stage"]
      )
      assert (
          not remainder
      ), f"The number of layers per stage ({raw_keys['num_layers_per_pipeline_stage']}) times the number of stages ({num_stages}) must divide the number of pipeline_parallel_layers which defaults to decoder layers  ({raw_keys['pipeline_parallel_layers']}) "
      raw_keys["num_pipeline_repeats"] = num_pipeline_repeats
    assert (
        num_stages * raw_keys["num_pipeline_repeats"] * raw_keys["num_layers_per_pipeline_stage"]
        == raw_keys["pipeline_parallel_layers"]
    ), f"The product of pipeline stages ({num_stages}), repeats ({raw_keys['num_pipeline_repeats']}), and layers per stage ({raw_keys['num_layers_per_pipeline_stage']}) must be equal to pipeline_parallel_layers which defaults to decoder layers ({raw_keys['pipeline_parallel_layers']})"
    if raw_keys["num_pipeline_microbatches"] == -1:
      if raw_keys["pipeline_delay_activation_forwarding"]:
        raw_keys["num_pipeline_microbatches"] = 2 * num_stages
      else:
        raw_keys["num_pipeline_microbatches"] = num_stages
    assert (
        raw_keys["num_pipeline_microbatches"] % num_stages == 0
    ), f"The number of microbatches ({raw_keys['num_pipeline_microbatches']}) must be divisible by the number of stages ({num_stages})"
    assert (
        raw_keys["micro_batch_size_to_train_on"] % raw_keys["num_pipeline_microbatches"] == 0
    ), f"The batch size ({raw_keys['micro_batch_size_to_train_on']}) must be divisible by the number of microbatches ({raw_keys['num_pipeline_microbatches']})"
    if raw_keys["pipeline_delay_activation_forwarding"]:
      assert (
          raw_keys["num_pipeline_microbatches"] >= 2 * num_stages
      ), f"Delayed activation forwarding requires at least 2 * num_stages microbatches, but {num_stages} stages are used with {raw_keys['num_pipeline_microbatches']} microbatches"
  else:
    raw_keys["using_pipeline_parallelism"] = False
  return raw_keys


def validate_deepseek_moe(raw_keys):
  if raw_keys["n_routing_groups"] != -1:
    if raw_keys["topk_routing_group"] == -1:
      raise ValueError(f'config topk_routing_group: {raw_keys["topk_routing_group"]} is not defined')
    if raw_keys["n_routing_groups"] <= raw_keys["topk_routing_group"]:
      raise ValueError(
          f'config n_routing_groups: {raw_keys["n_routing_groups"]} must be greter than topk_routing_group: {raw_keys["topk_routing_group"]}'
      )
    if raw_keys["num_experts"] % raw_keys["n_routing_groups"] != 0:
      raise ValueError(
          f'config num_experts: {raw_keys["num_experts"]} must be divisible by n_routing_groups: {raw_keys["n_routing_groups"]}'
      )

def validate_mlp_dim(raw_keys):
  """Validates that MLP dimensions are consistent for fully MoE models."""
  is_fully_moe_model = (raw_keys["interleave_moe_layer_step"] == 1 and raw_keys["first_num_dense_layers"] == 0)
  base_mlp_dim = raw_keys["base_mlp_dim"]
  base_moe_mlp_dim = raw_keys["base_moe_mlp_dim"]
  if is_fully_moe_model and (base_mlp_dim != base_moe_mlp_dim):
      raise ValueError(f'For a fully MoE model, base_mlp_dim must be equal to base_moe_mlp_dim. Received base_mlp_dim={base_mlp_dim} and base_moe_mlp_dim={base_moe_mlp_dim}.')

def validate_gpt_oss_moe(raw_keys):
  if raw_keys["decoder_block"] == "gpt_oss" and not raw_keys["sparse_matmul"] and raw_keys["capacity_factor"] != -1:
    raise ValueError(f"GPT OSS model only supports dropless MoE. Please use dense matmul with capacity_factor=-1 or sparse matmul.")

def validate_sparse_matmul_parallelism(raw_keys):
  # TODO: remove once b/434699033 resolved
  if raw_keys["sparse_matmul"] and (using_expert_parallelism(raw_keys) and using_pipeline_parallelism(raw_keys)):
    raise ValueError("Sparse matmul doesn't support using expert and pipeline parallelism together.")

  # TODO: remove once b/435539039 resolved
  if raw_keys["sparse_matmul"] and (
      using_fsdp_and_transpose_parallelism(raw_keys)
      and using_expert_parallelism(raw_keys)
      and using_tensor_parallelism(raw_keys)
  ):
    raise ValueError("Sparse matmul doesn't support using fsdp, expert, and tensor parallelism together.")
  tensor_parallelism = (
      raw_keys["ici_tensor_parallelism"]
      * raw_keys["dcn_tensor_parallelism"]
      * raw_keys["ici_tensor_sequence_parallelism"]
      * raw_keys["dcn_tensor_sequence_parallelism"]
      * raw_keys["ici_tensor_transpose_parallelism"]
      * raw_keys["dcn_tensor_transpose_parallelism"]
  )
  if raw_keys["sparse_matmul"] and using_tensor_parallelism(raw_keys) and (raw_keys["emb_dim"] % tensor_parallelism):
    raise ValueError(
        f"The embedding dimension {raw_keys['emb_dim']} is not divisible by tensor parallelism setting {tensor_parallelism}."
    )
  expert_parallelism = raw_keys["ici_expert_parallelism"] * raw_keys["dcn_expert_parallelism"]
  if raw_keys["sparse_matmul"] and using_expert_parallelism(raw_keys) and (raw_keys["num_experts"] % expert_parallelism):
    raise ValueError(
        f"The expert dimension {raw_keys['num_experts']} is not divisible by expert parallelism setting {expert_parallelism}."
    )


def validate_ring_of_experts_parallelism(raw_keys):
  if raw_keys["use_ring_of_experts"] and not using_expert_parallelism(raw_keys):
    raise ValueError("Ring-of-experts requires expert-parallelism to be enabled.")

def validate_shard_fsdp_on_expert_parallelism(raw_keys):
  if raw_keys["fsdp_shard_on_exp"] and raw_keys["num_experts"] % raw_keys["ici_fsdp_parallelism"]!=0: 
    raise ValueError("fsdp_shard_on_exp requires num_experts is divisiable by ici_fsdp_parallelism.")
  if raw_keys["fsdp_shard_on_exp"] and (using_tensor_parallelism(raw_keys) or using_expert_parallelism(raw_keys)): 
    raise ValueError("fsdp_shard_on_exp requires ici_expert_parallelism = 1 and ici_tensor_parallelism/ici_tensor_transpose_parallelism = 1.")

def validate_ragged_dot(raw_keys):
  if raw_keys["sparse_matmul"] and not raw_keys["megablox"]:
    config_flag = "jax_ragged_dot_use_ragged_dot_instruction"
    try:
      jax.config.update(config_flag, True)
    except AttributeError:
      max_logging.log(f"JAX config {config_flag} not found, possibly due to old JAX version.")


def create_new_logical_axis_rules(old_logical_axis_rules, new_logical_axis_rules):
  new_logical_axis = set()
  replacements = []
  for logical_axis, mesh_axes in new_logical_axis_rules:
    logical_axis_exists = any(rule for rule in old_logical_axis_rules if rule[0] == logical_axis)
    if not logical_axis_exists:
      continue
    replacements.append((logical_axis, mesh_axes))
    new_logical_axis.add(logical_axis)
  old_logical_rules_filtered = [
      (old_logical_axis, _lists_to_tuples(old_mesh_axes))
      for old_logical_axis, old_mesh_axes in old_logical_axis_rules
      if old_logical_axis not in new_logical_axis
  ]
  return old_logical_rules_filtered + replacements


def update_model_keys(raw_keys, model_keys, key):
  """Update `key` value in `raw_keys` from the value in `model_keys`."""
  assert key in model_keys and key in raw_keys
  if key == "logical_axis_rules":
    raw_keys[key] = create_new_logical_axis_rules(
        old_logical_axis_rules=raw_keys[key], new_logical_axis_rules=model_keys[key]
    )
    return
  raw_keys[key] = model_keys[key]


def validate_and_update_keys(raw_keys, model_keys, config_name: str):
  """Validate and update model specific config keys"""
  max_logging.log("Updating following parameters in config\n")

  for k in model_keys:
    max_logging.log(f"{k}: {model_keys[k]}")
    if k not in raw_keys:
      raise ValueError(f"Key {k} does not exist in config {config_name}.")
    elif not isinstance(raw_keys[k], type(model_keys[k])):
      raise ValueError(f"Type of key:{k} does not match with {type(model_keys[k])}")
    else:
      update_model_keys(raw_keys, model_keys, k)
  return raw_keys


def get_individual_scales(scale):
  """Choose appropriate scales for individual dimensions based on global scale
  We choose to rotate between doubling:
    num_head and mlp_dim
    embed_dim
    num_layers
  Any one of these steps is not a perfect doubling, although going through a cycle
  of three is a near perfect 8x scaling except for the linear -> softmax -> output step"""

  log_2_scale = math.floor((math.log2(scale)))
  if 2**log_2_scale != scale:
    raise ValueError(
        "Global parameter scale should be a power of 2. If you want finer grained control of the model sizes "
        "then you can explicitly set base_embed_dim, base_num_heads, base_mlp_dim, base_num_decoder_layers and/or head_dim."
    )
  base_scale, rem = divmod(log_2_scale, 3)
  num_head_scale = base_scale + int(rem > 0)
  mlp_dim_scale = num_head_scale
  emb_scale = base_scale + int(rem > 1)
  layer_scale = base_scale
  return emb_scale, num_head_scale, mlp_dim_scale, layer_scale


def calculate_global_batch_sizes(
    per_device_batch_size, expansion_factor_real_data, num_devices, gradient_accumulation_steps
):
  """Calculates target global batch size from target devices and per_device_batch"""
  if per_device_batch_size < 1.0:
    # For per_device_batch_size<1, we load the data as if per_device_batch_size=1
    if expansion_factor_real_data != -1:
      micro_batch_size_to_load = num_devices * expansion_factor_real_data
    else:
      micro_batch_size_to_load = num_devices
  else:
    if expansion_factor_real_data != -1:
      micro_batch_size_to_load = int(num_devices * per_device_batch_size * expansion_factor_real_data)
    else:
      micro_batch_size_to_load = int(num_devices * per_device_batch_size)

  micro_batch_size_to_train_on = int(num_devices * per_device_batch_size)
  global_batch_size_to_load = int(micro_batch_size_to_load * gradient_accumulation_steps)
  global_batch_size_to_train_on = int(micro_batch_size_to_train_on * gradient_accumulation_steps)
  return global_batch_size_to_load, global_batch_size_to_train_on, micro_batch_size_to_train_on


def get_num_target_devices(raw_keys):
  # In AOT case compile_topology is set (e.g. is not the empty string), and we determine the
  # number of devices from the compile_topology. In non-AOT settings we simply can use jax.devices().
  if raw_keys.get("compile_topology"):
    compile_topology = accelerator_to_spec_map.get_system_characteristics(raw_keys["compile_topology"])
    devices_per_slice = compile_topology.devices_per_slice
    return int(devices_per_slice * raw_keys["compile_topology_num_slices"])
  elif raw_keys.get("subslice_shape") and raw_keys.get("enable_single_controller"):
    subslice_shape = tuple(int(x) for x in raw_keys["subslice_shape"].split(","))
    return prod(subslice_shape)
  else:
    return len(jax.devices())


def get_quantization_local_shard_count(raw_keys):
  if raw_keys["quantization_local_shard_count"] == -1:
    return raw_keys["num_slices"]
  else:
    return raw_keys["quantization_local_shard_count"]


def get_context_parallel_size(raw_keys):
  cp_size = raw_keys["ici_context_parallelism"] * raw_keys["dcn_context_parallelism"]
  # ep acts as cp in attention
  if raw_keys["expert_shard_attention_option"] == "context":
    cp_size = cp_size * raw_keys["ici_expert_parallelism"] * raw_keys["dcn_expert_parallelism"]
  return cp_size


def using_pipeline_parallelism(raw_keys) -> bool:
  return int(raw_keys["ici_pipeline_parallelism"]) > 1 or int(raw_keys["dcn_pipeline_parallelism"]) > 1


def using_tensor_parallelism(raw_keys) -> bool:
  return (
      int(raw_keys["ici_tensor_parallelism"]) > 1
      or int(raw_keys["dcn_tensor_parallelism"]) > 1
      or int(raw_keys["ici_tensor_sequence_parallelism"]) > 1
      or int(raw_keys["dcn_tensor_sequence_parallelism"]) > 1
  )


def using_sequence_parallelism(raw_keys) -> bool:
  return int(raw_keys["ici_sequence_parallelism"]) > 1 or int(raw_keys["dcn_sequence_parallelism"]) > 1


def using_expert_parallelism(raw_keys) -> bool:
  if int(raw_keys["ici_expert_parallelism"]) > 1 and int(raw_keys["dcn_expert_parallelism"]) > 1:
    raise ValueError("Expert parallelism can only be enabled on ICI or DCN, not both.")
  return int(raw_keys["ici_expert_parallelism"]) > 1 or int(raw_keys["dcn_expert_parallelism"]) > 1


def using_fsdp_and_transpose_parallelism(raw_keys) -> bool:
  return (
      int(raw_keys["ici_fsdp_parallelism"]) > 1
      or int(raw_keys["dcn_fsdp_parallelism"]) > 1
      or int(raw_keys["ici_fsdp_transpose_parallelism"]) > 1
      or int(raw_keys["dcn_fsdp_transpose_parallelism"]) > 1
  )

@register_pytree_node_class
class HyperParameters:
  """Wrapper class to expose the configuration in a read-only manner."""

  def __init__(self, config):
    object.__setattr__(self, "_config", config)

  def __getattr__(self, attr):
    try:
      # Attempt to perform the normal lookup
      return object.__getattribute__(self, "_config").keys[attr]
    except AttributeError as exc:
      raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'") from exc

  def __setattr__(self, attr, value):
    raise ValueError("Reinitialization of config is not allowed")

  def get_keys(self):
    return self._config.keys
  

  def tree_flatten(self):
    return (), self

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    return aux_data


def initialize(argv, **kwargs):
  _config = _HyperParameters(argv, **kwargs)
  config = HyperParameters(_config)
  return config


if __name__ == "__main__":
  main_config = initialize(sys.argv)
  print(main_config.steps)
  r = range(main_config.steps)
