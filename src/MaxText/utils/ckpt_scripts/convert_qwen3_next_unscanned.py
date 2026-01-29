# Copyright 2025 Google LLC
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

"""Convert weights from a Qwen3 Next model to a MaxText one in unscanned orbax format.

Example cmd:

python3 -m MaxText.utils.ckpt_scripts.convert_qwen3_next_unscanned --base-model-path <path/to/hf/ckpt> \
    --maxtext-model-path <GCS/path/to/save/new/maxtext/ckpt> --model-size qwen3-next-80b-a3b
"""

# pylint: disable=g-line-too-long
import argparse
import gc
import logging
import os
import pathlib

os.environ["JAX_PLATFORMS"] = "cpu"

import ml_dtypes
import psutil
import numpy as np
from safetensors import safe_open
import torch
from tqdm import tqdm
from typing import Any, Dict

from MaxText import max_logging
from MaxText.utils.ckpt_scripts.llama_or_mistral_ckpt import save_weights_to_checkpoint
from MaxText.utils.ckpt_scripts.convert_qwen3_next_scanned import MODEL_PARAMS_DICT
from maxtext.inference.inference_utils import str2bool


# NOTE: numpy doesn't have native support for bfloat16, so
# we'll use ml_dtypes instead (which is quasi native)
# NOTE: it's incredibly silly but you can't directly cast from
# a torch tensor of type bfloat16 to a numpy array of type bfloat16
# so we have to cast to float32 first
CAST_DTYPE = ml_dtypes.bfloat16


def _pt_to_np(pt_weight, cast_dtype=None, transpose=False):
  if cast_dtype:
    np_weight = pt_weight.to(torch.float32).numpy().astype(cast_dtype)
  else:
    np_weight = pt_weight.to(torch.float32).numpy()
  if transpose:
    np_weight = np_weight.transpose()
  return np_weight


def create_unscanned_layer_pytree(layer_idx) -> Dict[str, Any]:
  """Creates the nested dictionary for one scanned layer."""
  if layer_idx % 4 == 3:
    return {
        # Common
        "input_layernorm": {"scale": None},
        "post_attention_layernorm": {"scale": None},
        # MoE
        "mlp": {
            "shared_expert": {
                "wi_0": {"kernel": None},
                "wi_1": {"kernel": None},
                "wo": {"kernel": None},
            },
            "shared_expert_gate": {"kernel": None},
            "routed_experts": {
                "gate": {"kernel": None},
                "wi_0": None,
                "wi_1": None,
                "wo": None,
            },
        },
        # Attention (will hold both GA and GDN params)
        "attention": {
            "attention": {
                "query": {"kernel": None},
                "key": {"kernel": None},
                "value": {"kernel": None},
                "out": {"kernel": None},
                "query_norm": {"scale": None},
                "key_norm": {"scale": None},
            },
        },
    }
  else:
    return {
        # Common
        "input_layernorm": {"scale": None},
        "post_attention_layernorm": {"scale": None},
        # MoE
        "mlp": {
            "shared_expert": {
                "wi_0": {"kernel": None},
                "wi_1": {"kernel": None},
                "wo": {"kernel": None},
            },
            "shared_expert_gate": {"kernel": None},
            "routed_experts": {
                "gate": {"kernel": None},
                "wi_0": None,
                "wi_1": None,
                "wo": None,
            },
        },
        # Attention (will hold both GA and GDN params)
        "attention": {
            # GDN Params
            "A_log": None,
            "conv1d": {"kernel": None},
            "dt_bias": None,
            "in_proj_ba": {"kernel": None},
            "in_proj_qkvz": {"kernel": None},
            "norm": {"rms_norm": {"scale": None}},
            "out_proj": {"kernel": None},
        },
    }


def convert_hf_to_maxtext(base_model_path: str, model_size: str, model_params: dict, mem_info: psutil.Process):
  """Convert a Huggingface Checkpoint to a dictionary of Numpy arrays representing the weights.

  Args:
    base_model_path (str): Path to the base model checkpoint.
    model_size (str): Size of the base model.
    model_params (dict): Dictionary containing model parameters.
    mem_info (psutil.Process): Process object to track memory usage.

  Returns:
    jax_weights (dict): Dictionary containing the converted weights.
  """
  # Load all params from config
  num_layers = model_params["num_hidden_layers"]
  hidden_size = model_params["hidden_size"]
  num_experts = model_params["num_experts"]
  ga_num_q_heads = model_params["ga_num_q_heads"]
  head_dim = model_params["head_dim"]
  ga_num_kv_heads = model_params["ga_num_kv_heads"]

  # load model
  max_logging.log(f"Loading the base model from {base_model_path}")
  ckpt_paths = sorted(pathlib.Path(base_model_path).glob("*.safetensors"))
  chkpt_vars = {}

  for i, ckpt_path in enumerate(ckpt_paths):
    max_logging.log(f"Loading checkpoint {i+1} of {len(ckpt_paths)} ...")
    with safe_open(ckpt_path, framework="pt", device="cpu") as f:
      for key in f.keys():
        # Follow up for @rbierneni: Skipping mtp weights until verified
        if key.startswith("model.") or key.startswith("lm_head."):
          chkpt_vars[key] = f.get_tensor(key)

  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  # Part 2: Initialize the nested MaxText weights dictionary
  jax_weights = {
      "token_embedder": {"embedding": None},
      "decoder": {
          "decoder_norm": {"scale": None},
          "logits_dense": {"kernel": None},
      },
  }
  for l in range(num_layers):
    jax_weights["decoder"][f"layers_{l}"] = create_unscanned_layer_pytree(l)

  # Part 3: Populate weights
  # Non-layer weights
  max_logging.log("Populating non-layer weights...")
  jax_weights["decoder"]["decoder_norm"]["scale"] = _pt_to_np(chkpt_vars["model.norm.weight"], cast_dtype=CAST_DTYPE)
  jax_weights["token_embedder"]["embedding"] = _pt_to_np(chkpt_vars["model.embed_tokens.weight"], cast_dtype=CAST_DTYPE)
  jax_weights["decoder"]["logits_dense"]["kernel"] = _pt_to_np(
      chkpt_vars["lm_head.weight"], cast_dtype=CAST_DTYPE
  ).transpose()
  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  # Linear + Gated Attention layers
  max_logging.log("Processing linear & gated attention layers")
  for l in tqdm(range(num_layers), desc="layers", leave=False):
    if l % 4 == 3:
      gated_attn = jax_weights["decoder"][f"layers_{l}"]["attention"]["attention"]

      k_kernel = (
          _pt_to_np(chkpt_vars[f"model.layers.{l}.self_attn.k_proj.weight"], cast_dtype=CAST_DTYPE)
          .transpose()
          .reshape(hidden_size, ga_num_kv_heads, head_dim)
      )
      k_norm = _pt_to_np(chkpt_vars[f"model.layers.{l}.self_attn.k_norm.weight"], cast_dtype=CAST_DTYPE)
      o_kernel = _pt_to_np(chkpt_vars[f"model.layers.{l}.self_attn.o_proj.weight"], cast_dtype=CAST_DTYPE).transpose()
      q_kernel = (
          _pt_to_np(chkpt_vars[f"model.layers.{l}.self_attn.q_proj.weight"], cast_dtype=CAST_DTYPE)
          .transpose()
          .reshape(hidden_size, ga_num_q_heads, head_dim * 2)
      )
      q_norm = _pt_to_np(chkpt_vars[f"model.layers.{l}.self_attn.q_norm.weight"], cast_dtype=CAST_DTYPE)
      v_kernel = (
          _pt_to_np(chkpt_vars[f"model.layers.{l}.self_attn.v_proj.weight"], cast_dtype=CAST_DTYPE)
          .transpose()
          .reshape(hidden_size, ga_num_kv_heads, head_dim)
      )

      gated_attn["key"]["kernel"] = k_kernel
      gated_attn["key_norm"]["scale"] = k_norm
      gated_attn["out"]["kernel"] = o_kernel
      gated_attn["query"]["kernel"] = q_kernel
      gated_attn["query_norm"]["scale"] = q_norm
      gated_attn["value"]["kernel"] = v_kernel
    else:
      lin_attn = jax_weights["decoder"][f"layers_{l}"]["attention"]

      a_log = _pt_to_np(chkpt_vars[f"model.layers.{l}.linear_attn.A_log"], cast_dtype=CAST_DTYPE)
      conv1d_kernel = _pt_to_np(
          chkpt_vars[f"model.layers.{l}.linear_attn.conv1d.weight"], cast_dtype=CAST_DTYPE
      ).transpose(2, 1, 0)
      dt_bias = _pt_to_np(chkpt_vars[f"model.layers.{l}.linear_attn.dt_bias"], cast_dtype=CAST_DTYPE)
      ba_kernel = _pt_to_np(
          chkpt_vars[f"model.layers.{l}.linear_attn.in_proj_ba.weight"], cast_dtype=CAST_DTYPE
      ).transpose()
      qkvz_kernel = _pt_to_np(
          chkpt_vars[f"model.layers.{l}.linear_attn.in_proj_qkvz.weight"], cast_dtype=CAST_DTYPE
      ).transpose()
      gated_rms_norm = _pt_to_np(chkpt_vars[f"model.layers.{l}.linear_attn.norm.weight"], cast_dtype=CAST_DTYPE)
      o_kernel = _pt_to_np(chkpt_vars[f"model.layers.{l}.linear_attn.out_proj.weight"], cast_dtype=CAST_DTYPE).transpose()

      lin_attn["A_log"] = a_log
      lin_attn["conv1d"]["kernel"] = conv1d_kernel
      lin_attn["dt_bias"] = dt_bias
      lin_attn["in_proj_ba"]["kernel"] = ba_kernel
      lin_attn["in_proj_qkvz"]["kernel"] = qkvz_kernel
      lin_attn["norm"]["rms_norm"]["scale"] = gated_rms_norm
      lin_attn["out_proj"]["kernel"] = o_kernel
  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  # layer weight pre and post self attention norm
  max_logging.log("Processing pre and post self attention norms")
  for l in tqdm(range(num_layers), desc="layers", leave=False):
    layer_weight = jax_weights["decoder"][f"layers_{l}"]

    input_layernorm = _pt_to_np(chkpt_vars[f"model.layers.{l}.input_layernorm.weight"], cast_dtype=CAST_DTYPE)
    post_attention_layernorm = _pt_to_np(
        chkpt_vars[f"model.layers.{l}.post_attention_layernorm.weight"], cast_dtype=CAST_DTYPE
    )

    layer_weight["input_layernorm"]["scale"] = input_layernorm
    layer_weight["post_attention_layernorm"]["scale"] = post_attention_layernorm
  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  # mlp weights
  max_logging.log("Processing mlp layer weights")
  for l in tqdm(range(num_layers), desc="layers", leave=False):
    mlp_weights = jax_weights["decoder"][f"layers_{l}"]["mlp"]

    shared_wi_0 = _pt_to_np(
        chkpt_vars[f"model.layers.{l}.mlp.shared_expert.gate_proj.weight"], cast_dtype=CAST_DTYPE
    ).transpose()
    shared_wi_1 = _pt_to_np(
        chkpt_vars[f"model.layers.{l}.mlp.shared_expert.up_proj.weight"], cast_dtype=CAST_DTYPE
    ).transpose()
    shared_wo = _pt_to_np(
        chkpt_vars[f"model.layers.{l}.mlp.shared_expert.down_proj.weight"], cast_dtype=CAST_DTYPE
    ).transpose()
    shared_gate_kernel = _pt_to_np(
        chkpt_vars[f"model.layers.{l}.mlp.shared_expert_gate.weight"], cast_dtype=CAST_DTYPE
    ).transpose()

    mlp_weights["shared_expert_gate"]["kernel"] = shared_gate_kernel
    mlp_weights["shared_expert"]["wi_0"]["kernel"] = shared_wi_0
    mlp_weights["shared_expert"]["wi_1"]["kernel"] = shared_wi_1
    mlp_weights["shared_expert"]["wo"]["kernel"] = shared_wo

    wi_0_list = []
    wi_1_list = []
    wo_list = []
    routed_gate_kernel = _pt_to_np(chkpt_vars[f"model.layers.{l}.mlp.gate.weight"], cast_dtype=CAST_DTYPE).transpose()
    for i in range(num_experts):
      wi_0_list.append(
          _pt_to_np(chkpt_vars[f"model.layers.{l}.mlp.experts.{i}.gate_proj.weight"], cast_dtype=CAST_DTYPE).transpose()
      )
      wi_1_list.append(
          _pt_to_np(chkpt_vars[f"model.layers.{l}.mlp.experts.{i}.up_proj.weight"], cast_dtype=CAST_DTYPE).transpose()
      )
      wo_list.append(
          _pt_to_np(chkpt_vars[f"model.layers.{l}.mlp.experts.{i}.down_proj.weight"], cast_dtype=CAST_DTYPE).transpose()
      )

    mlp_weights["routed_experts"]["gate"]["kernel"] = routed_gate_kernel
    mlp_weights["routed_experts"]["wi_0"] = np.stack(wi_0_list, axis=0)
    mlp_weights["routed_experts"]["wi_1"] = np.stack(wi_1_list, axis=0)
    mlp_weights["routed_experts"]["wo"] = np.stack(wo_list, axis=0)

    gc.collect()
  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  del chkpt_vars
  gc.collect()
  return jax_weights


def convert_to_jax_weights(base_model_path: str, model_size: str):
  """
  Function to convert the checkpoint at base_model_path into Orbax checkpoint
  for MaxText and output jax_weights ready for MaxText

  Attributes:
    base_model_path: checkpoint path
    model_size: qwen3-next-80b-a3b
  """
  model_params = MODEL_PARAMS_DICT[model_size]
  mem_info = psutil.Process()
  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))
  max_logging.log(f"Loading the base model from {base_model_path}")
  return convert_hf_to_maxtext(base_model_path, model_size, model_params, mem_info)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--base-model-path", type=str, required=True)
  parser.add_argument("--maxtext-model-path", type=str, required=True)
  parser.add_argument("--model-size", type=str, required=True)
  parser.add_argument("--simulated-cpu-devices-count", type=int, required=False, default=16)
  parser.add_argument("--use-ocdbt", type=str2bool, required=False, default=True)
  parser.add_argument("--use-zarr3", type=str2bool, required=False, default=True)
  args = parser.parse_args()

  if args.model_size not in MODEL_PARAMS_DICT:
    raise NotImplementedError(f"Model '{args.model_size}' is not supported.")

  os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={args.simulated_cpu_devices_count}"
  base_weights_path = args.maxtext_model_path

  save_weights_to_checkpoint(
      args.maxtext_model_path,
      convert_to_jax_weights(args.base_model_path, args.model_size),
      args.simulated_cpu_devices_count,
      args.use_ocdbt,
      args.use_zarr3,
  )
  max_logging.log(f"Successfully saved base_weights to {base_weights_path}.")
