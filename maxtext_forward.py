"""
# tpu
export LIBTPU_INIT_ARGS='--xla_tpu_scoped_vmem_limit_kib=81920'
python maxtext_forward.py
"""

import jax
import jax.numpy as jnp
import numpy as np
# from importlib import reload
# import MaxText
# reload(MaxText)
import MaxText.layers.models as models
import MaxText.layers.quantizations as quantizations
from MaxText import pyconfig
from MaxText import maxtext_utils
from MaxText.common_types import DECODING_ACTIVE_SEQUENCE_INDICATOR, MODEL_MODE_TRAIN

# jax.config.update("jax_platform_name", "cpu")

import argparse
import sys
from MaxText.tests.forward_pass_logit_checker import get_data
import jsonlines
from MaxText import max_logging


class MockConfigForForwardPass:
  """
  This is a bit hacky, but it's needed because `get_data` expects to be
  passed in an MT config, but in the PT/reference code, this is difficult
  to set up, so we just mock this out for now.
  """

  def __init__(self):
    self.global_batch_size_to_train_on = 1
    self.max_target_length = 4
    self.use_multimodal=False


model_args = [
    "something.py",
    "MaxText/configs/base.yml",
    # "hardware=cpu",
    "base_output_directory=test",
    "run_name=temp-testing-only",
    "skip_jax_distributed_system=true",
    # model specific
    "model_name=gpt-oss-20b",
    "scan_layers=false",
    "attention=dot_product",
    "load_parameters_path=gs://shuningjin-multipod-dev/gpt-oss-20b/unscan-bf16-v2-2025-09-02-01-16-00/0/items",
    # high precision flags
    "weight_dtype=float32",
    "dtype=float32",
    "activations_in_float32=true",
    "matmul_precision=high",
]


# model_args = ['/mnt/disks/jacobplatin/code/maxtext/llama4_maverick_check_weight.py', 'MaxText/configs/base.yml', 'hardware=cpu', 'scan_layers=false', 'base_output_directory=llama4', 'run_name=temp-testing-only', 'model_name=llama4-17b-128e', 'skip_jax_distributed_system=true', 'load_parameters_path=/mnt/disks/jacobplatin/models/llama4/maverick/4-layer-unscanned/0/items/']
config = pyconfig.initialize(model_args)

init_rng = jax.random.PRNGKey(config.init_weights_seed)
init_rng, rng1 = jax.random.split(init_rng)
devices_array = maxtext_utils.create_device_mesh(config)
mesh = jax.sharding.Mesh(devices_array, config.mesh_axes)
quant = quantizations.configure_quantization(config)
# model = models.Transformer(config, mesh=mesh, quant=quant)
model = models.transformer_as_linen(config, mesh, quant=quant, model_mode=MODEL_MODE_TRAIN)
state, _ = maxtext_utils.setup_decode_state(model, config, rng1, mesh, None)


def print_nested_keys(data, prefix=""):
  """
  Prints nested keys of a dictionary-like structure in a directory-like format.
  Args:
      data: The dictionary-like structure to traverse.
      prefix: The current path prefix.
  """
  if isinstance(data, dict):
    for key, value in data.items():
      current_path = f"{prefix}{key}."
      print_nested_keys(value, current_path)
  else:
    print(f"{prefix} | {data.shape} | {data.mean()}")


print_nested_keys(state.params)


# init_rng = jax.random.PRNGKey(config.init_weights_seed)
# init_rng, rng1 = jax.random.split(init_rng)
# devices_array = maxtext_utils.create_device_mesh(config)
# mesh = jax.sharding.Mesh(devices_array, config.mesh_axes)
# quant = quantizations.configure_quantization(config)
# maxtext_model = models.transformer_as_linen(config, mesh, quant=quant, model_mode=MODEL_MODE_TRAIN)
# maxtext_state, _ = maxtext_utils.setup_decode_state(maxtext_model, config, rng1, mesh, None)


# def get_data(golden_data, golden_data_index, config):
#   """Get the golden data for the test indexed at golden_data_index"""

#   max_logging.log(f"Comparing forward pass for golden data index = {golden_data_index} ")
#   max_logging.log(f"config.global_batch_size_to_train_on={config.global_batch_size_to_train_on}")
#   s = (config.global_batch_size_to_train_on, config.max_target_length)
#   ids = np.asarray(golden_data[golden_data_index]["tokens"], dtype=np.int32)

#   logits = np.asarray(golden_data[golden_data_index]["logits"], dtype=np.float32)
#   max_logging.log(f" prompt=\"{golden_data[golden_data_index]['prompt']}\" raw ids={ids}, logits.shape = {logits.shape}")

#   decoder_segment_ids = np.zeros(s) + DECODING_ACTIVE_SEQUENCE_INDICATOR
#   decoder_positions = np.stack(
#       [np.arange(config.max_target_length, dtype=np.int32) for _ in range(config.global_batch_size_to_train_on)]
#   )

#   ids = np.stack([ids for _ in range(config.global_batch_size_to_train_on)])
#   max_logging.log(f"ids={ids}, decoder_segment_ids = {decoder_segment_ids}, decoder_positions= {decoder_positions}")

#   return ids, decoder_segment_ids, decoder_positions, logits


def get_data(golden_data, golden_data_index, config):
  """Get the golden data for the test indexed at golden_data_index"""

  max_logging.log(f"Comparing forward pass for golden data index = {golden_data_index}")
  max_logging.log(f"config.global_batch_size_to_train_on={config.global_batch_size_to_train_on}")

  original_ids = np.asarray(golden_data[golden_data_index]["tokens"], dtype=np.int32)
  seq_len = len(original_ids)

  if seq_len > config.max_target_length:
    raise ValueError(
        f"Golden data sequence length ({seq_len}) is greater than max_target_length ({config.max_target_length})"
    )

  s = (config.global_batch_size_to_train_on, config.max_target_length)

  # Pad ids to max_target_length. MaxText expects 0 for padding.
  padded_ids = np.pad(original_ids, (0, config.max_target_length - seq_len), "constant", constant_values=0)
  ids = np.stack([padded_ids for _ in range(config.global_batch_size_to_train_on)])

  logits = np.asarray(golden_data[golden_data_index]["logits"], dtype=np.float32)
  max_logging.log(
      f" prompt=\"{golden_data[golden_data_index]['prompt']}\" raw ids={original_ids}, logits.shape = {logits.shape}"
  )

  decoder_segment_ids = np.zeros(s, dtype=np.int32)
  decoder_segment_ids[:, :seq_len] = DECODING_ACTIVE_SEQUENCE_INDICATOR
  decoder_positions = np.stack(
      [np.arange(config.max_target_length, dtype=np.int32) for _ in range(config.global_batch_size_to_train_on)]
  )

  return ids, decoder_segment_ids, decoder_positions, logits#, seq_len


def setup_golden_data(input_golden_data_path):
  """
  Sets up the data to run on for both PT/MaxText forward pass.

  Returns:
    ids: tokenized ids to feed as input to the forward pass
    decoder_segment_ids: segment ids to feed as input to the forward pass (used for MaxText / JAX only)
    decoder_positions: position ids to feed as input to the forward pass (used for MaxText / JAX only)
    golden_logits: pre-computed logits from the forward pass to serve as reference
  """

  with jsonlines.open(input_golden_data_path, "r") as f:
    golden_data = list(f)
  print("len(golden_data)", len(golden_data))

  #   assert len(golden_data) == 1
  golden_data_index = 0
  ids, decoder_segment_ids, decoder_positions, golden_logits = get_data(
      golden_data, golden_data_index, MockConfigForForwardPass()
  )
  golden_logits = np.array(golden_logits).astype(np.float32)
  print(
      "Golden token ids have values:",
      ids,
      "Golden decoder_positions have values:",
      decoder_positions,
      "Golden decoder_segment_ids have values:",
      decoder_segment_ids,
  )
  return ids, decoder_segment_ids, decoder_positions, golden_logits


ids, decoder_segment_ids, decoder_positions, golden_logits = setup_golden_data("/tmp/gpt-oss-20b-golden_bf16_v2_debug.jsonl")
# ids, decoder_segment_ids, decoder_positions, golden_logits = setup_golden_data("golden_scout_4layer.jsonl")

full_train_logits = model.apply(
    state.params,
    ids,
    decoder_positions,
    decoder_segment_ids,
    enable_dropout=False,
    rngs={"aqt": init_rng},
)

print(full_train_logits)
