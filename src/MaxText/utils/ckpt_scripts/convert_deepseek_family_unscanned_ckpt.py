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

r"""Convert weights from a DeepSeek style model to a MaxText one with unscanned format.

Example cmd:

python3 -m MaxText.utils.ckpt_scripts.convert_deepseek_family_unscanned_ckpt --base_model_path <path/to/hf/ckpt> \
    --maxtext_model_path <GCS/path/to/save/new/maxtext/ckpt> --model_size deepseek2-16b

optional flags: [--use-ocdbt True --use-zarr3 True]
"""

# pylint: disable=line-too-long
# pylint: disable=unsupported-assignment-operation
# pytype: disable=unsupported-operands

import argparse
import pathlib
import os
import gc
import logging
import absl
import psutil
from tqdm import tqdm
import time

import numpy as np
from safetensors import safe_open
import torch

from MaxText import max_logging
from MaxText.utils.ckpt_scripts import convert_deepseek_family_ckpt as ds_ckpt
from MaxText.utils.ckpt_scripts.llama_or_mistral_ckpt import save_weights_to_checkpoint
from MaxText.inference_utils import str2bool
from MaxText.utils.ckpt_conversion.utils.utils import MemoryMonitorTqdm, print_peak_memory

absl.logging.set_verbosity(absl.logging.INFO)  # for max_logging.log


def _convert_huggingface_to_jax_weights(base_model_path, model_params, mem_info) -> dict:
  """Convert Huggingface Checkpoint to Jax."""
  base_num_decoder_layers = model_params["num_layers"]
  first_num_dense_layers = model_params["first_num_dense_layers"]
  base_num_query_heads = model_params["base_num_query_heads"]
  base_emb_dim = model_params["base_emb_dim"]
  num_experts = model_params["num_experts"]
  q_lora_rank = model_params["q_lora_rank"]
  kv_lora_rank = model_params["kv_lora_rank"]
  qk_nope_head_dim = model_params["qk_nope_head_dim"]
  qk_rope_head_dim = model_params["qk_rope_head_dim"]
  v_head_dim = model_params["v_head_dim"]
  use_sparse_indexer = model_params.get("use_sparse_indexer", False)

  # load safetensor
  ckpt_paths = sorted(pathlib.Path(base_model_path).glob("[!.]*.safetensors"))
  chkpt_vars = {}
  for i, ckpt_path in tqdm(enumerate(ckpt_paths), total=len(ckpt_paths)):
    max_logging.log(f"Loading checkpoint {i+1} of {len(ckpt_paths)} ...")
    with safe_open(ckpt_path, framework="pt", device="cpu") as f:
      for key in f.keys():
        parts = key.split(".")
        layer = int(parts[2]) if "layers" in key else 0
        if key.endswith("_scale_inv"):
          raise ValueError("fp8 checkpoint is not supported.")
        if ds_ckpt.is_key_allowed(key, ds_ckpt.MTP_KEYS_TO_SKIP):
          mapped_key = ds_ckpt.hf_to_maxtext_mapping(layer, num_experts, first_num_dense_layers, base_num_decoder_layers)[
              key
          ]
          chkpt_vars[mapped_key] = f.get_tensor(key)

  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  # initialize the data structure for storing jax_weights
  jax_weights = {
      "decoder": {
          "decoder_norm": {"scale": None},
          "logits_dense": {"kernel": None},
      },
      "token_embedder": {"embedding": None},
  }

  # decoder norm scale ###########################################
  max_logging.log("Processing decoder norm scale")
  jax_weights["decoder"]["decoder_norm"]["scale"] = chkpt_vars["decoder_norm.scale"].to(torch.float16).numpy()
  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  # logits dense #################################################
  max_logging.log("Processing logits dense")
  jax_weights["decoder"]["logits_dense"]["kernel"] = (
      chkpt_vars["logits_dense.kernel"].to(torch.float16).numpy().transpose()
  )
  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  # token embedding ##############################################
  max_logging.log("Processing token embeddings")
  jax_weights["token_embedder"]["embedding"] = chkpt_vars["token_embedder.embedding"].to(torch.float16).numpy()
  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  layers = {
      "dense_layers": first_num_dense_layers,
      "moe_layers": base_num_decoder_layers - first_num_dense_layers,
  }
  # self attention and normalization ###############################################
  max_logging.log("Processing self attention and normalization in dense layer")
  for layer_key, layer_value in layers.items():
    for layer_idx in MemoryMonitorTqdm(range(layer_value), desc=layer_key, leave=True):
      layer_name = f"{layer_key}_{layer_idx}"
      if layer_key == "dense_layers":
        jax_weights["decoder"].update(
            {
                layer_name: {
                    "mlp": {
                        "wi_0": {"kernel": None},
                        "wi_1": {"kernel": None},
                        "wo": {"kernel": None},
                    },
                    "self_attention": {
                        "kv_norm": {"scale": None},
                        "wkv_a": {"kernel": None},
                        "wkv_b": {"kernel": None},
                        "out": {"kernel": None},
                    },
                    "pre_self_attention_layer_norm": {"scale": None},
                    "post_self_attention_layer_norm": {"scale": None},
                },
            }
        )
      else:
        jax_weights["decoder"].update(
            {
                layer_name: {
                    "DeepSeekMoeBlock_0": {
                        "MoeBlock_0": {
                            "wi_0": None,
                            "wi_1": None,
                            "wo": None,
                            "gate": {"kernel": None},
                        },
                        "shared_experts": {
                            "wi_0": {"kernel": None},
                            "wi_1": {"kernel": None},
                            "wo": {"kernel": None},
                        },
                    },
                    "self_attention": {
                        "kv_norm": {"scale": None},
                        "wkv_a": {"kernel": None},
                        "wkv_b": {"kernel": None},
                        "out": {"kernel": None},
                    },
                    "pre_self_attention_layer_norm": {"scale": None},
                    "post_self_attention_layer_norm": {"scale": None},
                },
            }
        )
      self_attention = jax_weights["decoder"][layer_name]["self_attention"]
      pre_self_attention_layer_norm = jax_weights["decoder"][layer_name]["pre_self_attention_layer_norm"]
      post_self_attention_layer_norm = jax_weights["decoder"][layer_name]["post_self_attention_layer_norm"]

      pre_self_attention = (
          chkpt_vars[f"{layer_key}.{layer_idx}.pre_self_attention_layer_norm.scale"].to(torch.float16).numpy()
      )
      post_self_attention = (
          chkpt_vars[f"{layer_key}.{layer_idx}.post_self_attention_layer_norm.scale"].to(torch.float16).numpy()
      )
      kv_norm = chkpt_vars[f"{layer_key}.{layer_idx}.self_attention.kv_norm.scale"].to(torch.float16).numpy().transpose()
      wkv_a = chkpt_vars[f"{layer_key}.{layer_idx}.self_attention.wkv_a.kernel"].to(torch.float16).numpy().transpose()
      wkv_b = chkpt_vars[f"{layer_key}.{layer_idx}.self_attention.wkv_b.kernel"].to(torch.float16).numpy().transpose()
      out = chkpt_vars[f"{layer_key}.{layer_idx}.self_attention.out.kernel"].to(torch.float16).numpy().transpose()
      if q_lora_rank != 0:
        q_norm = chkpt_vars[f"{layer_key}.{layer_idx}.self_attention.q_norm.scale"].to(torch.float16).numpy()
        wq_a = chkpt_vars[f"{layer_key}.{layer_idx}.self_attention.wq_a.kernel"].to(torch.float16).numpy().transpose()
        wq_b = chkpt_vars[f"{layer_key}.{layer_idx}.self_attention.wq_b.kernel"].to(torch.float16).numpy().transpose()
      else:
        query = chkpt_vars[f"{layer_key}.{layer_idx}.self_attention.query.kernel"].to(torch.float16).numpy().transpose()

      # reshape to match maxtext
      wkv_b = np.reshape(wkv_b, [kv_lora_rank, base_num_query_heads, (qk_nope_head_dim + v_head_dim)])
      out = np.reshape(out, [base_num_query_heads, v_head_dim, base_emb_dim])
      if q_lora_rank != 0:
        wq_b = np.reshape(wq_b, [q_lora_rank, base_num_query_heads, (qk_nope_head_dim + qk_rope_head_dim)])
      else:
        query = np.reshape(query, [base_emb_dim, base_num_query_heads, (qk_nope_head_dim + qk_rope_head_dim)])

      if q_lora_rank != 0:
        self_attention.update(
            {
                "q_norm": {"scale": None},
                "wq_a": {"kernel": None},
                "wq_b": {"kernel": None},
            }
        )
      else:
        self_attention.update({"query": {"kernel": None}})

      self_attention["kv_norm"]["scale"] = kv_norm
      self_attention["wkv_a"]["kernel"] = wkv_a
      self_attention["wkv_b"]["kernel"] = wkv_b
      self_attention["out"]["kernel"] = out
      pre_self_attention_layer_norm["scale"] = pre_self_attention
      post_self_attention_layer_norm["scale"] = post_self_attention
      if q_lora_rank != 0:
        self_attention["q_norm"]["scale"] = q_norm
        self_attention["wq_a"]["kernel"] = wq_a
        self_attention["wq_b"]["kernel"] = wq_b
      else:
        self_attention["query"]["kernel"] = query

      # TODO(shuningjin)
      # BEGIN CHANGE
      if use_sparse_indexer:
        # init weight
        self_attention["indexer"] = {
            "wq_b": {"kernel": None},
            "wk": {"kernel": None},
            "weights_proj": {"kernel": None},
            "k_norm": {"scale": None, "bias": None},
        }
        # read from huggingface ckpt
        # transform weight
        # assign
        self_attention["indexer"]["wq_b"]["kernel"] = None
        self_attention["indexer"]["wk"]["kernel"] = None
        self_attention["indexer"]["weights_proj"]["kernel"] = None
        self_attention["indexer"]["k_norm"]["scale"] = None
        self_attention["indexer"]["k_norm"]["bias"] = None
      # END CHANGE

      jax_weights["decoder"][layer_name]["self_attention"] = self_attention
      jax_weights["decoder"][layer_name]["pre_self_attention_layer_norm"] = pre_self_attention_layer_norm
      jax_weights["decoder"][layer_name]["post_self_attention_layer_norm"] = post_self_attention_layer_norm

  # layer weights: mlp ################################################
  max_logging.log("Processing mlp weights")
  for layer_key, layer_value in layers.items():
    for layer_idx in MemoryMonitorTqdm(range(layer_value), desc=layer_key, leave=True):
      if layer_key == "dense_layers":
        layer_name = f"{layer_key}_{layer_idx}"
        mlp = jax_weights["decoder"][layer_name]["mlp"]
        wi_0 = chkpt_vars[f"{layer_key}.{layer_idx}.mlp.wi_0.kernel"].to(torch.float16).numpy().transpose()
        wi_1 = chkpt_vars[f"{layer_key}.{layer_idx}.mlp.wi_1.kernel"].to(torch.float16).numpy().transpose()
        wo = chkpt_vars[f"{layer_key}.{layer_idx}.mlp.wo.kernel"].to(torch.float16).numpy().transpose()
        mlp["wi_0"]["kernel"] = wi_0
        mlp["wi_1"]["kernel"] = wi_1
        mlp["wo"]["kernel"] = wo
        jax_weights["decoder"][layer_name]["mlp"] = mlp
      else:
        layer_name = f"{layer_key}_{layer_idx}"
        moe = jax_weights["decoder"][layer_name]["DeepSeekMoeBlock_0"]
        if q_lora_rank != 0:
          gate_bias = (
              chkpt_vars[f"{layer_key}.{layer_idx}.DeepSeekMoeBlock_0.MoeBlock_0.gate.bias"]
              .to(torch.float16)
              .numpy()
              .transpose()
          )
        gate = (
            chkpt_vars[f"{layer_key}.{layer_idx}.DeepSeekMoeBlock_0.MoeBlock_0.gate.kernel"]
            .to(torch.float16)
            .numpy()
            .transpose()
        )
        shared_wi_0 = (
            chkpt_vars[f"{layer_key}.{layer_idx}.DeepSeekMoeBlock_0.shared_experts.wi_0.kernel"]
            .to(torch.float16)
            .numpy()
            .transpose()
        )
        shared_wi_1 = (
            chkpt_vars[f"{layer_key}.{layer_idx}.DeepSeekMoeBlock_0.shared_experts.wi_1.kernel"]
            .to(torch.float16)
            .numpy()
            .transpose()
        )
        shared_wo = (
            chkpt_vars[f"{layer_key}.{layer_idx}.DeepSeekMoeBlock_0.shared_experts.wo.kernel"]
            .to(torch.float16)
            .numpy()
            .transpose()
        )

        if q_lora_rank != 0:
          moe["MoeBlock_0"]["gate"]["bias"] = gate_bias
        moe["MoeBlock_0"]["gate"]["kernel"] = gate
        moe["shared_experts"]["wi_0"]["kernel"] = shared_wi_0
        moe["shared_experts"]["wi_1"]["kernel"] = shared_wi_1
        moe["shared_experts"]["wo"]["kernel"] = shared_wo

        for k in tqdm(range(num_experts), desc="experts", leave=False):
          wi_0 = (
              chkpt_vars[f"{layer_key}.{layer_idx}.DeepSeekMoeBlock_0.MoeBlock_0.{k}.wi_0"]
              .to(torch.float16)
              .numpy()
              .transpose()
          )
          wi_1 = (
              chkpt_vars[f"{layer_key}.{layer_idx}.DeepSeekMoeBlock_0.MoeBlock_0.{k}.wi_1"]
              .to(torch.float16)
              .numpy()
              .transpose()
          )
          wo = (
              chkpt_vars[f"{layer_key}.{layer_idx}.DeepSeekMoeBlock_0.MoeBlock_0.{k}.wo"]
              .to(torch.float16)
              .numpy()
              .transpose()
          )

          if moe["MoeBlock_0"]["wi_0"] is None:
            stack_shape = (num_experts,)
            moe["MoeBlock_0"]["wi_0"] = np.zeros(stack_shape + wi_0.shape, dtype=np.float16)
            moe["MoeBlock_0"]["wi_1"] = np.zeros(stack_shape + wi_1.shape, dtype=np.float16)
            moe["MoeBlock_0"]["wo"] = np.zeros(stack_shape + wo.shape, dtype=np.float16)
          moe["MoeBlock_0"]["wi_0"][k, ...] = wi_0
          moe["MoeBlock_0"]["wi_1"][k, ...] = wi_1
          moe["MoeBlock_0"]["wo"][k, ...] = wo

        jax_weights["decoder"][layer_name]["DeepSeekMoeBlock_0"] = moe

  del chkpt_vars
  gc.collect()
  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))
  return jax_weights


def _convert_to_jax_weights(base_model_path, model_size, mem_info) -> dict:
  """
  Function to convert the checkpoint at base_model_path into Orbax checkpoint
  for MaxText and output jax_weights ready for MaxText.

  Args:
      base_model_path: Path to the Hugging Face model checkpoint.
      model_size: Model size key in MODEL_PARAMS_DICT.
      mem_info: A process instance used for memory tracking.

  Returns:
      The converted JAX weights.
  """
  model_params = ds_ckpt.MODEL_PARAMS_DICT[model_size]
  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))
  max_logging.log(f"Loading the base model from {base_model_path}")
  return _convert_huggingface_to_jax_weights(base_model_path, model_params, mem_info)


def main() -> None:
  parser = argparse.ArgumentParser(description="Convert and save DeepSeek model weights.")
  parser.add_argument("--base_model_path", type=str, required=True)
  parser.add_argument("--maxtext_model_path", type=str, required=True)
  parser.add_argument("--model_size", type=str, required=True)
  parser.add_argument("--simulated_cpu_devices_count", type=int, required=False, default=16)
  parser.add_argument("--use-ocdbt", type=str2bool, required=False, default=True)
  parser.add_argument("--use-zarr3", type=str2bool, required=False, default=True)
  args = parser.parse_args()

  overall_start = time.time()

  if args.model_size not in ds_ckpt.MODEL_PARAMS_DICT:
    raise NotImplementedError(f"Model '{args.model_size}' is not supported.")

  os.environ["JAX_PLATFORMS"] = "cpu"
  os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={args.simulated_cpu_devices_count}"
  mem_info = psutil.Process()

  # transform
  start = time.time()
  weights = _convert_to_jax_weights(args.base_model_path, args.model_size, mem_info)
  max_logging.log(f"Elapse for transform: {(time.time() - start) / 60:.2f} min")

  # save
  save_weights_to_checkpoint(
      args.maxtext_model_path,
      weights,
      args.simulated_cpu_devices_count,
      args.use_ocdbt,
      args.use_zarr3,
  )
  max_logging.log(f"Successfully saved base_weights to {args.maxtext_model_path}.")
  max_logging.log(f"Overall Elapse: {(time.time() - overall_start) / 60:.2f} min")
  print_peak_memory()

if __name__ == "__main__":
  main()
