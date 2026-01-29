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

"""Convert weights from a GPT-OSS style model to a MaxText one with scanned format.

Example cmd:

python3 -m MaxText.utils.ckpt_scripts.convert_gpt_oss_ckpt --base-model-path <path/to/hf/ckpt> \
    --maxtext-model-path <GCS/path/to/save/new/maxtext/ckpt> --model-size gpt-oss-20b
"""

# pylint: disable=g-line-too-long
import argparse
import gc
import logging
import os
import pathlib
import absl
import time

os.environ["JAX_PLATFORMS"] = "cpu"

import ml_dtypes
import numpy as np
import psutil
from safetensors import safe_open
from tqdm import tqdm

from MaxText import max_logging
from MaxText.utils.ckpt_scripts.llama_or_mistral_ckpt import save_weights_to_checkpoint
from MaxText.utils.ckpt_scripts.convert_gpt_oss_unscanned_ckpt import MODEL_PARAMS_DICT, _hf_to_maxtext_mapping, _pt_to_np
from MaxText.utils.ckpt_conversion.utils.utils import MemoryMonitorTqdm, print_peak_memory
from maxtext.inference.inference_utils import str2bool

absl.logging.set_verbosity(absl.logging.INFO)  # for max_logging.log

# NOTE: numpy doesn't have native support for bfloat16, so
# we'll use ml_dtypes instead (which is quasi native)
# NOTE: it's incredibly silly but you can't directly cast from
# a torch tensor of type bfloat16 to a numpy array of type bfloat16
# so we have to cast to float32 first
CAST_DTYPE = ml_dtypes.bfloat16


def _convert_huggingface_to_jax_weights(
    base_model_path: str, model_size: str, model_params: dict, mem_info: psutil.Process
):
  """Convert a Huggingface Checkpoint to a dictionary of Numpy arrays representing the weights.

  Args:
    base_model_path (str): Path to the base model checkpoint.
    model_size (str): Size of the base model.
    model_params (dict): Dictionary containing model parameters.
    mem_info (psutil.Process): Process object to track memory usage.

  Returns:
    jax_weights (dict): Dictionary containing the converted weights.
  """
  # model params
  base_num_decoder_layers = model_params["base_num_decoder_layers"]
  base_emb_dim = model_params["base_emb_dim"]
  base_num_query_heads = model_params["base_num_query_heads"]
  base_num_kv_heads = model_params["base_num_kv_heads"]
  head_dim = model_params["head_dim"]
  layer_cycle_interval = model_params["inhomogeneous_layer_cycle_interval"]

  # load model
  max_logging.log(f"Loading the base model from {base_model_path}")
  ckpt_paths = sorted(pathlib.Path(base_model_path).glob("[!.]*.safetensors"))
  chkpt_vars = {}
  for i, ckpt_path in tqdm(enumerate(ckpt_paths), total=len(ckpt_paths)):
    max_logging.log(f"Loading checkpoint {i+1} of {len(ckpt_paths)} ...")

    with safe_open(ckpt_path, framework="pt", device="cpu") as f:
      for key in f.keys():
        parts = key.split(".")
        layer = int(parts[2]) if "layers" in key else 0
        mapped_key = _hf_to_maxtext_mapping(layer)[key]
        chkpt_vars[mapped_key] = f.get_tensor(key)

  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  # initialize the data structure for storing jax_weights
  jax_weights = {
      "token_embedder": {"embedding": None},
      "decoder": {
          "decoder_norm": {"scale": None},
          "logits_dense": {"kernel": None},
          "layers": {},
      },
  }
  # block 0, 1
  for block_idx in range(layer_cycle_interval):
    jax_weights["decoder"]["layers"][f"layers_{block_idx}"] = {
        "pre_self_attention_layer_norm": {"scale": None},
        "post_self_attention_layer_norm": {"scale": None},
        "GptOssAttention": {
            "query": {"kernel": None, "bias": None},
            "key": {"kernel": None, "bias": None},
            "value": {"kernel": None, "bias": None},
            "out": {"kernel": None, "bias": None},
            "sinks": None,
        },
        "GptOssMlp": {
            "gate": {"kernel": None, "bias": None},
            "wi_0": None,
            "wi_0_bias": None,
            "wi_1": None,
            "wi_1_bias": None,
            "wo": None,
            "wo_bias": None,
        },
    }

  # decoder norm scale ###########################################
  max_logging.log("Processing decoder norm scale")
  jax_weights["decoder"]["decoder_norm"]["scale"] = _pt_to_np(chkpt_vars["norm.weight"], cast_dtype=CAST_DTYPE)
  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  # logits dense #################################################
  max_logging.log("Processing logits dense")

  logit_dense = _pt_to_np(chkpt_vars["output.weight"], cast_dtype=CAST_DTYPE)
  jax_weights["decoder"]["logits_dense"]["kernel"] = logit_dense.transpose()  # [:, :vocab_size]

  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  # token embedding ##############################################
  max_logging.log("Processing token embeddings")

  jax_weights["token_embedder"]["embedding"] = _pt_to_np(chkpt_vars["tok_embeddings.weight"], cast_dtype=CAST_DTYPE)

  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  # layer weight: self attention ###############################################
  max_logging.log("Processing self attention")
  for layer_idx in MemoryMonitorTqdm(range(base_num_decoder_layers), desc="layers", leave=True):
    block_layer_idx, block_idx = divmod(layer_idx, layer_cycle_interval)
    stack_shape = (base_num_decoder_layers // layer_cycle_interval,)
    self_attention = jax_weights["decoder"]["layers"][f"layers_{block_idx}"]["GptOssAttention"]

    wq = _pt_to_np(chkpt_vars[f"layers.{layer_idx}.attention.wq.weight"], cast_dtype=CAST_DTYPE)
    wk = _pt_to_np(chkpt_vars[f"layers.{layer_idx}.attention.wk.weight"], cast_dtype=CAST_DTYPE)
    wv = _pt_to_np(chkpt_vars[f"layers.{layer_idx}.attention.wv.weight"], cast_dtype=CAST_DTYPE)
    w_post = _pt_to_np(chkpt_vars[f"layers.{layer_idx}.attention.wo.weight"], cast_dtype=CAST_DTYPE)

    # NOTE: not scale the query weights in checkpoint, but apply query_pre_attn_scalar=1/np.sqrt(head_dim) for attention
    # (num_attention_heads * head_dim, hidden_size) ->
    #     (hidden_size, num_attention_heads * head_dim) -> (hidden_size, num_attention_heads, head_dim)
    # [embed, q, head_dim]
    wq = wq.transpose().reshape([base_emb_dim, base_num_query_heads, head_dim])
    # [embed, kv, head_dim]
    wk = wk.transpose().reshape([base_emb_dim, base_num_kv_heads, head_dim])
    # [embed, kv, head_dim]
    wv = wv.transpose().reshape([base_emb_dim, base_num_kv_heads, head_dim])
    # (hidden_size, num_attention_heads * head_dim) -> (num_attention_heads * head_dim, hidden_size) ->
    #     (num_attention_heads, head_dim, hidden_size)
    # [q, head_dim, embed]
    w_post = w_post.transpose().reshape([base_num_query_heads, head_dim, base_emb_dim])

    sinks = _pt_to_np(chkpt_vars[f"layers.{layer_idx}.attention.sinks"], cast_dtype=CAST_DTYPE)
    wq_bias = _pt_to_np(chkpt_vars[f"layers.{layer_idx}.attention.wq.bias"], cast_dtype=CAST_DTYPE)
    wk_bias = _pt_to_np(chkpt_vars[f"layers.{layer_idx}.attention.wk.bias"], cast_dtype=CAST_DTYPE)
    wv_bias = _pt_to_np(chkpt_vars[f"layers.{layer_idx}.attention.wv.bias"], cast_dtype=CAST_DTYPE)
    w_post_bias = _pt_to_np(chkpt_vars[f"layers.{layer_idx}.attention.wo.bias"], cast_dtype=CAST_DTYPE)

    wq_bias = wq_bias.reshape([base_num_query_heads, head_dim])
    wk_bias = wk_bias.reshape([base_num_kv_heads, head_dim])
    wv_bias = wv_bias.reshape([base_num_kv_heads, head_dim])

    if self_attention["query"]["kernel"] is None:
      self_attention["query"]["kernel"] = np.zeros(stack_shape + wq.shape, dtype=CAST_DTYPE)
      self_attention["key"]["kernel"] = np.zeros(stack_shape + wk.shape, dtype=CAST_DTYPE)
      self_attention["value"]["kernel"] = np.zeros(stack_shape + wv.shape, dtype=CAST_DTYPE)
      self_attention["out"]["kernel"] = np.zeros(stack_shape + w_post.shape, dtype=CAST_DTYPE)
      self_attention["query"]["bias"] = np.zeros(stack_shape + wq_bias.shape, dtype=CAST_DTYPE)
      self_attention["key"]["bias"] = np.zeros(stack_shape + wk_bias.shape, dtype=CAST_DTYPE)
      self_attention["value"]["bias"] = np.zeros(stack_shape + wv_bias.shape, dtype=CAST_DTYPE)
      self_attention["out"]["bias"] = np.zeros(stack_shape + w_post_bias.shape, dtype=CAST_DTYPE)
      self_attention["sinks"] = np.zeros(stack_shape + sinks.shape, dtype=CAST_DTYPE)

    self_attention["query"]["kernel"][block_layer_idx, ...] = wq
    self_attention["key"]["kernel"][block_layer_idx, ...] = wk
    self_attention["value"]["kernel"][block_layer_idx, ...] = wv
    self_attention["out"]["kernel"][block_layer_idx, ...] = w_post
    self_attention["query"]["bias"][block_layer_idx, ...] = wq_bias
    self_attention["key"]["bias"][block_layer_idx, ...] = wk_bias
    self_attention["value"]["bias"][block_layer_idx, ...] = wv_bias
    self_attention["out"]["bias"][block_layer_idx, ...] = w_post_bias
    self_attention["sinks"][block_layer_idx, ...] = sinks

  for block_idx in range(layer_cycle_interval):
    self_attention = jax_weights["decoder"]["layers"][f"layers_{block_idx}"]["GptOssAttention"]
    self_attention["query"]["kernel"] = self_attention["query"]["kernel"].transpose(1, 0, 2, 3)
    self_attention["key"]["kernel"] = self_attention["key"]["kernel"].transpose(1, 0, 2, 3)
    self_attention["value"]["kernel"] = self_attention["value"]["kernel"].transpose(1, 0, 2, 3)
    self_attention["out"]["kernel"] = self_attention["out"]["kernel"].transpose(1, 0, 2, 3)
    self_attention["query"]["bias"] = self_attention["query"]["bias"].transpose(1, 0, 2)
    self_attention["key"]["bias"] = self_attention["key"]["bias"].transpose(1, 0, 2)
    self_attention["value"]["bias"] = self_attention["value"]["bias"].transpose(1, 0, 2)
    self_attention["out"]["bias"] = self_attention["out"]["bias"].transpose(1, 0)
    self_attention["sinks"] = self_attention["sinks"].transpose(1, 0)

  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  # layer weight: pre and post self attention norm ################
  max_logging.log("Processing pre and post self attention norms")
  for layer_idx in MemoryMonitorTqdm(range(base_num_decoder_layers), desc="layers", leave=True):
    block_layer_idx, block_idx = divmod(layer_idx, layer_cycle_interval)
    stack_shape = (base_num_decoder_layers // layer_cycle_interval,)
    layer_weight = jax_weights["decoder"]["layers"][f"layers_{block_idx}"]

    pre_self_attention_layernorm = _pt_to_np(
        chkpt_vars[f"layers.{layer_idx}.attention_norm.weight"], cast_dtype=CAST_DTYPE
    )
    post_self_attention_layernorm = _pt_to_np(chkpt_vars[f"layers.{layer_idx}.ffn_norm.weight"], cast_dtype=CAST_DTYPE)

    if layer_weight["pre_self_attention_layer_norm"]["scale"] is None:
      layer_weight["pre_self_attention_layer_norm"]["scale"] = np.zeros(
          stack_shape + pre_self_attention_layernorm.shape, dtype=CAST_DTYPE
      )
      layer_weight["post_self_attention_layer_norm"]["scale"] = np.zeros(
          stack_shape + post_self_attention_layernorm.shape, dtype=CAST_DTYPE
      )

    layer_weight["pre_self_attention_layer_norm"]["scale"][block_layer_idx, ...] = pre_self_attention_layernorm  # pylint: disable=E1137
    layer_weight["post_self_attention_layer_norm"]["scale"][block_layer_idx, ...] = post_self_attention_layernorm  # pylint: disable=E1137

  for block_idx in range(layer_cycle_interval):
    layer_weight = jax_weights["decoder"]["layers"][f"layers_{block_idx}"]
    layer_weight["pre_self_attention_layer_norm"]["scale"] = layer_weight["pre_self_attention_layer_norm"][
        "scale"
    ].transpose(1, 0)
    layer_weight["post_self_attention_layer_norm"]["scale"] = layer_weight["post_self_attention_layer_norm"][
        "scale"
    ].transpose(1, 0)

  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  # layer weight: mlp ################################################
  max_logging.log("Processing mlp weights")

  for layer_idx in MemoryMonitorTqdm(range(base_num_decoder_layers), desc="layers", leave=True):
    block_layer_idx, block_idx = divmod(layer_idx, layer_cycle_interval)
    stack_shape = (base_num_decoder_layers // layer_cycle_interval,)
    mlp_weight = jax_weights["decoder"]["layers"][f"layers_{block_idx}"]["GptOssMlp"]

    gate = _pt_to_np(chkpt_vars[f"layers.{layer_idx}.feed_forward.gate.weight"], cast_dtype=CAST_DTYPE)
    gate_bias = _pt_to_np(chkpt_vars[f"layers.{layer_idx}.feed_forward.gate.bias"], cast_dtype=CAST_DTYPE)
    wi_0_1 = _pt_to_np(chkpt_vars[f"layers.{layer_idx}.feed_forward.experts.gate_up_proj"], cast_dtype=CAST_DTYPE)
    wi_0_1_bias = _pt_to_np(
        chkpt_vars[f"layers.{layer_idx}.feed_forward.experts.gate_up_proj_bias"], cast_dtype=CAST_DTYPE
    )
    wo = _pt_to_np(chkpt_vars[f"layers.{layer_idx}.feed_forward.experts.down_proj"], cast_dtype=CAST_DTYPE)
    wo_bias = _pt_to_np(chkpt_vars[f"layers.{layer_idx}.feed_forward.experts.down_proj_bias"], cast_dtype=CAST_DTYPE)

    # router
    gate = gate.transpose()
    # experts.gate_up_proj: de-interleave last dim, even for gate, odd for up_proj
    wi_0 = wi_0_1[..., ::2]
    wi_1 = wi_0_1[..., 1::2]
    del wi_0_1
    wi_0_bias = wi_0_1_bias[..., ::2]
    wi_1_bias = wi_0_1_bias[..., 1::2]
    del wi_0_1_bias

    if mlp_weight["gate"]["kernel"] is None:
      mlp_weight["gate"]["kernel"] = np.zeros(stack_shape + gate.shape, dtype=CAST_DTYPE)
      mlp_weight["gate"]["bias"] = np.zeros(stack_shape + gate_bias.shape, dtype=CAST_DTYPE)
      mlp_weight["wi_0"] = np.zeros(stack_shape + wi_0.shape, dtype=CAST_DTYPE)
      mlp_weight["wi_1"] = np.zeros(stack_shape + wi_1.shape, dtype=CAST_DTYPE)
      mlp_weight["wi_0_bias"] = np.zeros(stack_shape + wi_0_bias.shape, dtype=CAST_DTYPE)
      mlp_weight["wi_1_bias"] = np.zeros(stack_shape + wi_1_bias.shape, dtype=CAST_DTYPE)
      mlp_weight["wo"] = np.zeros(stack_shape + wo.shape, dtype=CAST_DTYPE)
      mlp_weight["wo_bias"] = np.zeros(stack_shape + wo_bias.shape, dtype=CAST_DTYPE)

    mlp_weight["gate"]["kernel"][block_layer_idx, ...] = gate
    mlp_weight["gate"]["bias"][block_layer_idx, ...] = gate_bias
    mlp_weight["wi_0"][block_layer_idx, ...] = wi_0
    mlp_weight["wi_1"][block_layer_idx, ...] = wi_1
    mlp_weight["wi_0_bias"][block_layer_idx, ...] = wi_0_bias
    mlp_weight["wi_1_bias"][block_layer_idx, ...] = wi_1_bias
    mlp_weight["wo"][block_layer_idx, ...] = wo
    mlp_weight["wo_bias"][block_layer_idx, ...] = wo_bias

    gc.collect()

  for block_idx in range(layer_cycle_interval):
    mlp_weight = jax_weights["decoder"]["layers"][f"layers_{block_idx}"]["GptOssMlp"]
    mlp_weight["gate"]["kernel"] = mlp_weight["gate"]["kernel"].transpose(1, 0, 2)
    mlp_weight["wi_0"] = mlp_weight["wi_0"].transpose(1, 0, 2, 3)
    mlp_weight["wi_1"] = mlp_weight["wi_1"].transpose(1, 0, 2, 3)
    mlp_weight["wo"] = mlp_weight["wo"].transpose(1, 0, 2, 3)
    mlp_weight["gate"]["bias"] = mlp_weight["gate"]["bias"].transpose(1, 0)
    mlp_weight["wi_0_bias"] = mlp_weight["wi_0_bias"].transpose(1, 0, 2)
    mlp_weight["wi_1_bias"] = mlp_weight["wi_1_bias"].transpose(1, 0, 2)
    mlp_weight["wo_bias"] = mlp_weight["wo_bias"].transpose(1, 0, 2)

  del chkpt_vars
  gc.collect()
  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))
  return jax_weights


def convert_to_jax_weights(base_model_path: str, model_size: str):
  """
  Function to convert the checkpoint at base_model_path into Orbax checkpoint
  for MaxText and output jax_weights ready for MaxText

  Args:
    base_model_path: checkpoint path
    model_size: gpt-oss-20b, gpt-oss-120b
  """
  model_params = MODEL_PARAMS_DICT[model_size]
  mem_info = psutil.Process()
  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))
  max_logging.log(f"Loading the base model from {base_model_path}")
  return _convert_huggingface_to_jax_weights(base_model_path, model_size, model_params, mem_info)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--base-model-path", type=str, required=True)
  parser.add_argument("--maxtext-model-path", type=str, required=True)
  parser.add_argument("--model-size", type=str, required=True)
  parser.add_argument("--simulated-cpu-devices-count", type=int, required=False, default=16)
  parser.add_argument("--use-ocdbt", type=str2bool, required=False, default=True)
  parser.add_argument("--use-zarr3", type=str2bool, required=False, default=True)
  args = parser.parse_args()

  overall_start = time.time()

  if args.model_size not in MODEL_PARAMS_DICT:
    raise NotImplementedError(f"Model '{args.model_size}' is not supported.")

  os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={args.simulated_cpu_devices_count}"
  base_weights_path = args.maxtext_model_path

  # transform
  start = time.time()
  weights = convert_to_jax_weights(args.base_model_path, args.model_size)
  max_logging.log(f"Elapse for transform: {(time.time() - start) / 60:.2f} min")

  # save
  save_weights_to_checkpoint(
      args.maxtext_model_path,
      weights,
      args.simulated_cpu_devices_count,
      args.use_ocdbt,
      args.use_zarr3,
  )
  max_logging.log(f"Successfully saved base_weights to {base_weights_path}.")
  max_logging.log(f"Overall Elapse: {(time.time() - overall_start) / 60:.2f} min")
  print_peak_memory()
