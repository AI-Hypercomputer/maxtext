"""
# {"prompt": "I love to sleep", "tokens": [40, 3047, 316, 8746], "logits": …}
# gcloud storage cp gs://shuningjin-multipod-dev/golden-logits/openai/gpt-oss-20b/golden_bf16_v2_debug.jsonl /tmp/gpt-oss-20b-golden_bf16_v2_debug.jsonl

# tpu
export LIBTPU_INIT_ARGS='--xla_tpu_scoped_vmem_limit_kib=81920'
python maxtext_forward.py

export LIBTPU_INIT_ARGS='--xla_tpu_scoped_vmem_limit_kib=81920'
python maxtext_forward.py \
--attention=dot_product \
--scan_layers=false \
--load_parameters_path=gs://shuningjin-multipod-dev/gpt-oss-20b/unscan-bf16-v2-2025-09-02-01-16-00/0/items
"""

import jax
import jax.numpy as jnp
import numpy as np
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
import os


class MockConfigForForwardPass:
  """
  This is a bit hacky, but it's needed because `get_data` expects to be
  passed in an MT config, but in the PT/reference code, this is difficult
  to set up, so we just mock this out for now.
  """

  def __init__(self):
    self.global_batch_size_to_train_on = 1
    self.max_target_length = 4
    self.use_multimodal = False


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

  return ids, decoder_segment_ids, decoder_positions, logits  # , seq_len


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


if __name__ == "__main__":

  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

  parser = argparse.ArgumentParser(description="Run a MaxText model with specified configurations.")
  # Arguments that were in the original list
  # parser.add_argument("--base_output_directory", type=str, default="test", help="Base directory for output.")
  # parser.add_argument("--run_name", type=str, default="temp-testing-only", help="Name of the run.")
  # parser.add_argument("--skip_jax_distributed_system", type=str, default="true", help="Set to 'true' to skip JAX distributed setup.")
  # parser.add_argument("--model_name", type=str, default="gpt-oss-20b", help="Name of the model to use.")
  parser.add_argument("--scan_layers", type=str, default="false", help="Whether to use scanned layers.")
  parser.add_argument("--attention", type=str, default="dot_product", help="Attention mechanism type.")
  parser.add_argument(
      "--load_parameters_path",
      type=str,
      default="gs://shuningjin-multipod-dev/gpt-oss-20b/unscan-bf16-v2-2025-09-02-01-16-00/0/items",
      help="Path to load model parameters from.",
  )
  # parser.add_argument("--weight_dtype", type=str, default="float32", help="Data type for model weights.")
  # parser.add_argument("--dtype", type=str, default="float32", help="Data type for computations.")
  # parser.add_argument("--activations_in_float32", type=str, default="true", help="Use float32 for activations.")
  # parser.add_argument("--matmul_precision", type=str, default="high", help="Precision for matrix multiplications.")
  args = parser.parse_args()

  golden_data_path = "/tmp/gpt-oss-20b-golden_bf16_v2_debug.jsonl"

  model_args = [
      "something.py",
      "MaxText/configs/base.yml",
      # "hardware=cpu",
      "base_output_directory=test",
      "run_name=temp-testing-only",
      "skip_jax_distributed_system=true",
      # model specific
      "model_name=gpt-oss-20b",
      f"scan_layers={args.scan_layers}",
      f"attention={args.attention}",
      f"load_parameters_path={args.load_parameters_path}",
      # high precision flags
      "weight_dtype=float32",
      "dtype=float32",
      "activations_in_float32=true",
      "matmul_precision=high",
      "max_prefill_predict_length=4",
      "max_target_length=4",
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

  # print_nested_keys(state.params)

  ids, decoder_segment_ids, decoder_positions, logits_hf = setup_golden_data(golden_data_path)

  full_train_logits = model.apply(
      state.params,
      ids,
      decoder_positions,
      decoder_segment_ids,
      enable_dropout=False,
      rngs={"aqt": init_rng},
  )

  full_train_logits = jax.experimental.multihost_utils.process_allgather(full_train_logits)

  token_size = logits_hf.shape[0]
  max_logging.log(f"{token_size=}")
  logits_maxtext = full_train_logits[0, :token_size, :]
  logits_hf = logits_hf[:token_size, :]
  max_logging.log(f"{logits_maxtext.shape=}")
  max_logging.log(f"{logits_hf.shape=}")
  max_logging.log(f"Max Numerical Difference {np.abs(logits_hf - logits_maxtext).max()}")

  max_logging.log(f"{logits_maxtext=}")
  max_logging.log(f"{logits_hf=}")

  maxtext_probabilities = jax.nn.softmax(logits_maxtext, axis=-1)
  hf_probabilities = jax.nn.softmax(logits_hf, axis=-1)

  max_logging.log(f"{maxtext_probabilities=}")
  max_logging.log(f"{hf_probabilities=}")

  kl_div = jax.numpy.sum(jax.scipy.special.kl_div(hf_probabilities, maxtext_probabilities), axis=-1)
  max_logging.log(f"KL divergence = {kl_div}, max KL divergence = {jax.numpy.max(kl_div)}")
