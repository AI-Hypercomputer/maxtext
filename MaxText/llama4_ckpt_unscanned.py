# Copyright 2025 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Convert weights from a Llama4 PyTorch model to a MaxText one.

Usage:

Get LLaMA chkpt_vars from Meta

Example cmd:
To save a ckpt
JAX_PLATFORMS=CPU python -m  MaxText.llama4_ckpt_unscanned --base-model-path [CHKPT_DIR] \
 --maxtext-model-path [OUTPUT_CHKPT_DIR] --model-size llama4-17b-16e

The base model checkpoints should be in the format `{name}.{chkpt_idx}.pth`
For example: `llama4-17b-16e.00.pth`
This script requires a large memory VM (most likely a CPU instance with at least 500GB of memory).

NOTE @jacobplatin: eventually, we'll want to merge this into the regular `llama_or_mistral` script
since the logic is effectively the same.

Also note that we'll want to add HF chkpt support eventually as well.
"""
# pylint: disable=g-line-too-long
import argparse
import gc
import logging
import os
import pathlib
import re
from dataclasses import dataclass

os.environ["JAX_PLATFORMS"] = "cpu"

import ml_dtypes
import numpy as np
import psutil
import torch
from tqdm import tqdm

from MaxText import max_logging
from MaxText.inference_utils import str2bool
from MaxText.llama_or_mistral_ckpt import save_weights_to_checkpoint, permute_to_match_maxtext_rope, MODEL_PARAMS_DICT

SIMULATED_CPU_DEVICES_COUNT = 16

# NOTE: numpy doesn't have native support for bfloat16, so
# we'll use ml_dtypes instead (which is quasi native)
# NOTE: it's incredibly silly but you can't directly cast from
# a torch tensor of type bfloat16 to a numpy array of type bfloat16
# so we have to cast to float32 first
CAST_DTYPE = ml_dtypes.bfloat16


def _incoming_ckpt_to_maxtext_mapping(layer_idx: int = -1, expert_idx: int = -1, model_size="") -> dict:
  """
  Maps from an incoming checkpoint (e.g. downloaded Llama checkpoint (.pth or .saftensors))
  to MaxText model weights.

  Args:
  layer_idx: The layer index of the model.
  expert_idx: The expert index of the model.
  model_size: The model size/name.

  Returns:
  A dictionary mapping from the incoming checkpoint to the MaxText model weights.
  """
  # pylint: disable=line-too-long
  base_mapping = {
      "tok_embeddings.weight": "model.embed_tokens.weight",
      "norm.weight": "model.norm.weight",
      "output.weight": "lm_head.weight",
      # MOE model
      f"layers.{layer_idx}.attention_norm.weight": f"model.layers.{layer_idx}.input_layernorm.weight",
      f"layers.{layer_idx}.ffn_norm.weight": f"model.layers.{layer_idx}.post_attention_layernorm.weight",
      f"layers.{layer_idx}.attention.wq.weight": f"model.layers.{layer_idx}.self_attn.q_proj.weight",
      f"layers.{layer_idx}.attention.wk.weight": f"model.layers.{layer_idx}.self_attn.k_proj.weight",
      f"layers.{layer_idx}.attention.wv.weight": f"model.layers.{layer_idx}.self_attn.v_proj.weight",
      f"layers.{layer_idx}.attention.wo.weight": f"model.layers.{layer_idx}.self_attn.o_proj.weight",
      f"layers.{layer_idx}.feed_forward.gate.weight": f"model.layers.{layer_idx}.block_sparse_moe.gate.weight",
      # MOE model: routed experts
      f"layers.{layer_idx}.feed_forward.experts.{expert_idx}.w1.weight": f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w1.weight",
      f"layers.{layer_idx}.feed_forward.experts.{expert_idx}.w2.weight": f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w2.weight",
      f"layers.{layer_idx}.feed_forward.experts.{expert_idx}.w3.weight": f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w3.weight",
      # MOE model: shared experts
      f"layers.{layer_idx}.feed_forward.w_in_shared_FD.weight": f"model.layers.{layer_idx}.mlp.shared_experts.up_proj.weight",
      f"layers.{layer_idx}.feed_forward.w_out_shared_DF.weight": f"model.layers.{layer_idx}.mlp.shared_experts.gate_proj.weight",
      f"layers.{layer_idx}.feed_forward.w_swiglu_FD.weight": f"model.layers.{layer_idx}.mlp.shared_experts.down_proj.weight",
      # dense model
      f"layers.{layer_idx}.feed_forward.w1.weight": f"model.layers.{layer_idx}.mlp.gate_proj.weight",
      f"layers.{layer_idx}.feed_forward.w2.weight": f"model.layers.{layer_idx}.mlp.down_proj.weight",
      f"layers.{layer_idx}.feed_forward.w3.weight": f"model.layers.{layer_idx}.mlp.up_proj.weight",
      # LoRA Adapter
      f"layers.{layer_idx}.attention.wq.lora_A.weights": f"base_model.model.model.layers.{layer_idx}.self_attn.q_proj.lora_A.weight",
      f"layers.{layer_idx}.attention.wq.lora_B.weights": f"base_model.model.model.layers.{layer_idx}.self_attn.q_proj.lora_B.weight",
      f"layers.{layer_idx}.attention.wk.lora_A.weights": f"base_model.model.model.layers.{layer_idx}.self_attn.k_proj.lora_A.weight",
      f"layers.{layer_idx}.attention.wk.lora_B.weights": f"base_model.model.model.layers.{layer_idx}.self_attn.k_proj.lora_B.weight",
      f"layers.{layer_idx}.attention.wv.lora_A.weights": f"base_model.model.model.layers.{layer_idx}.self_attn.v_proj.lora_A.weight",
      f"layers.{layer_idx}.attention.wv.lora_B.weights": f"base_model.model.model.layers.{layer_idx}.self_attn.v_proj.lora_B.weight",
      f"layers.{layer_idx}.attention.wo.lora_A.weights": f"base_model.model.model.layers.{layer_idx}.self_attn.o_proj.lora_A.weight",
      f"layers.{layer_idx}.attention.wo.lora_B.weights": f"base_model.model.model.layers.{layer_idx}.self_attn.o_proj.lora_B.weight",
  }

  if model_size.startswith("llama4"):
    base_mapping[f"layers.{layer_idx}.attention.wkq.layer_norm_weight"] = base_mapping.pop(
        f"layers.{layer_idx}.attention_norm.weight"
    )
    base_mapping[f"layers.{layer_idx}feed_forward.norm.weight"] = base_mapping.pop(f"layers.{layer_idx}.ffn_norm.weight")

    base_mapping[f"layers.{layer_idx}.feed_forward.experts.moe_w_in_eD_F"] = base_mapping.pop(
        f"layers.{layer_idx}.feed_forward.experts.{expert_idx}.w1.weight"
    )

    base_mapping[f"layers.{layer_idx}.feed_forward.experts.moe_w_out_eF_D"] = base_mapping.pop(
        f"layers.{layer_idx}.feed_forward.experts.{expert_idx}.w2.weight"
    )

    base_mapping[f"layers.{layer_idx}.feed_forward.experts.moe_w_swiglu_eD_F"] = base_mapping.pop(
        f"layers.{layer_idx}.feed_forward.experts.{expert_idx}.w3.weight"
    )


@dataclass
class _NamespaceMapper:
  """A class to dynamically map Mistral/Llama weight names to HF/PT weights."""

  collection: dict
  model_size: str = ""
  delimiter: str = "."

  def __getitem__(self, key):
    if key in self.collection:
      return self.collection[key]  # original key takes precedence
    fields = key.split(self.delimiter)
    num_fields = [int(field) for field in fields if re.match(r"[0-9]+", field) is not None]
    mapping = _incoming_ckpt_to_maxtext_mapping(*num_fields, model_size=self.model_size)
    if key not in mapping:
      raise ValueError(f"Key `{key}` is missing from the original collection and from the mapping.")
    new_key = mapping[key]
    if new_key not in self.collection:
      raise ValueError(f"New key `{new_key}` mapped from `{key}` is missing from the collection.")
    return self.collection[new_key]


def _convert_pytorch_to_jax_weights(base_model_path: str, model_size: str, model_params: dict, mem_info: psutil.Process):
  """Convert a PyTorch checkpoint to a dictionary of Numpy arrays representing the weights.

  Args:
    base_model_path (str): Path to the PyTorch checkpoint file.
    model_size (str): Model size.
    model_params (dict): Model parameters.
    mem_info (psutil.Process): Memory usage information.

  Returns:
    dict: Dictionary of Numpy arrays representing the weights.
  """
  base_num_decoder_layers = model_params["num_layers"]
  base_num_query_heads = model_params["num_heads"]
  head_dim = model_params["dims_per_head"]
  base_num_kv_heads = model_params["num_kv_heads"]
  vocab_size = model_params["vocab"]
  num_experts = model_params["num_experts"] if "num_experts" in model_params else None
  rope_type = model_params.get("rope_type")
  scale_query = model_params.get("scale_query", True)

  chkpt_vars = {}
  ckpt_paths = sorted(pathlib.Path(base_model_path).glob("[!.]*.pth"))
  for i, ckpt_path in enumerate(ckpt_paths):
    max_logging.log(f"Loading checkpoint {i+1} of {len(ckpt_paths)} ...")
    # NOTE: starting in PT2.6, `weights_only` was switched from the default of `False` to `True`
    # thus we need to specify this or else loading will fail
    chkpt_vars[int(ckpt_path.name.split(".", maxsplit=2)[1])] = torch.load(ckpt_path, map_location="cpu", weights_only=False)
  chkpt_vars = [chkpt_vars[i] for i in sorted(list(chkpt_vars.keys()))]
  # map weight names if they use HuggingFace instead of PyTorch convention
  chkpt_vars = [_NamespaceMapper(var, model_size=model_size) for var in chkpt_vars]

  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  # initialize the data structure for storing jax_weights
  is_llama4_model = model_size[:6] == "llama4"
  jax_weights = {
      "decoder": {
          "decoder_norm": {"scale": None},
          "logits_dense": {"kernel": None},
      },
      "token_embedder": {"embedding": None},
  }

  # decoder norm scale ###########################################
  max_logging.log("Processing decoder norm scale")
  decoder_norm_scale = chkpt_vars[0]["norm.weight"].type(torch.float32).numpy().astype(CAST_DTYPE)
  jax_weights["decoder"]["decoder_norm"]["scale"] = decoder_norm_scale

  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  # logits dense #################################################
  max_logging.log("Processing logits dense")
  logits_dense = np.concatenate(
      [var["output.weight"].type(torch.float32).numpy().astype(CAST_DTYPE) for var in chkpt_vars], axis=0
  ).transpose()[:, :vocab_size]
  jax_weights["decoder"]["logits_dense"]["kernel"] = logits_dense

  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  # token embedding ##############################################
  max_logging.log("Processing token embeddings")
  if model_size[:6] in ["llama3", "llama4"]:
    token_embedder = np.concatenate(
        [var["tok_embeddings.weight"].type(torch.float32).numpy().astype(CAST_DTYPE) for var in chkpt_vars], axis=0
    )
  else:
    token_embedder = np.concatenate(
        [var["tok_embeddings.weight"].type(torch.float32).numpy().astype(CAST_DTYPE) for var in chkpt_vars], axis=1
    )[:vocab_size, :]
  jax_weights["token_embedder"]["embedding"] = token_embedder
  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  # self attention ###############################################
  max_logging.log("Processing self attention")

  # llama3.1-405b kv weight is replicated within every two files.
  wkv_step = 1 if model_size != "llama3.1-405b" else 2
  is_llama4_model = model_size[:6] == "llama4"
  has_printed_warning = False

  for layer_idx in tqdm(range(base_num_decoder_layers), desc="layers", leave=False):
    layer_name = f"layers_{layer_idx}"
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
                    "query": {"kernel": None},
                    "key": {"kernel": None},
                    "value": {"kernel": None},
                    "out": {"kernel": None},
                },
                "pre_self_attention_layer_norm": {"scale": None},
                "post_self_attention_layer_norm": {"scale": None},
            },
        }
    )

    self_attention = jax_weights["decoder"][layer_name]["self_attention"]

    # NOTE: llama4 fuses the qkv weights together, so we need to unroll them
    if is_llama4_model:
      # This of shape [base_emb_dim + 2 * kv_hidden_size, hidden_size]
      hidden_size = model_params["base_emb_dim"]
      ratio = base_num_query_heads // base_num_kv_heads
      kv_hidden_size = hidden_size // ratio
      wq, wk, wv = (
          np.zeros((hidden_size, hidden_size), dtype=ml_dtypes.bfloat16),
          np.zeros((kv_hidden_size, hidden_size), dtype=ml_dtypes.bfloat16),
          np.zeros((kv_hidden_size, hidden_size), dtype=ml_dtypes.bfloat16),
      )
      num_chkpt_parts = len(chkpt_vars)
      # NOTE: it's VERY important that the qkv splitting happens first and then the concatenation.
      # Concatenating the qkv weights first and then splitting will result in incorrect weights!
      for i in range(num_chkpt_parts):
        wqkv = chkpt_vars[i][f"layers.{layer_idx}.attention.wqkv.weight"].type(torch.float32).numpy()
        local_d = head_dim // num_chkpt_parts
        local_hidden_size = hidden_size // num_chkpt_parts
        local_kv_hidden_size = kv_hidden_size // num_chkpt_parts
        q = wqkv[:local_hidden_size, :]
        k = wqkv[local_hidden_size : local_hidden_size + local_kv_hidden_size, :]
        v = wqkv[local_hidden_size + local_kv_hidden_size :, :]

        wq[i * local_d * base_num_query_heads : (i + 1) * local_d * base_num_query_heads] = q
        wk[i * local_d * base_num_kv_heads : (i + 1) * local_d * base_num_kv_heads] = k
        wv[i * local_d * base_num_kv_heads : (i + 1) * local_d * base_num_kv_heads] = v

    else:
      wq = np.concatenate(
          [
              var[f"layers.{layer_idx}.attention.wq.weight"].type(torch.float32).numpy().astype(CAST_DTYPE)
              for var in chkpt_vars
          ],
          axis=0,
      )
      wk = np.concatenate(
          [
              var[f"layers.{layer_idx}.attention.wk.weight"].type(torch.float32).numpy().astype(CAST_DTYPE)
              for var in chkpt_vars[::wkv_step]
          ],
          axis=0,
      )
      wv = np.concatenate(
          [
              var[f"layers.{layer_idx}.attention.wv.weight"].type(torch.float32).numpy().astype(CAST_DTYPE)
              for var in chkpt_vars[::wkv_step]
          ],
          axis=0,
      )

    wq = wq.transpose()
    wk = wk.transpose()
    wv = wv.transpose()

    wq = np.reshape(wq, [base_num_query_heads * head_dim, base_num_query_heads, head_dim])
    wk = np.reshape(wk, [base_num_query_heads * head_dim, base_num_kv_heads, head_dim])
    wv = np.reshape(wv, [base_num_query_heads * head_dim, base_num_kv_heads, head_dim])

    if not rope_type.startswith("llama3.1"):
      wq = permute_to_match_maxtext_rope(wq)
      wk = permute_to_match_maxtext_rope(wk)
    else:
      if not has_printed_warning:
        max_logging.log("Skipping permute_to_match_maxtext_rope because model is a Llama3 variant or has RoPE Type Llama3.1")
        has_printed_warning = True

    # This will be of size [hidden_size, hidden_size], but the first dimmension is sharded, so I believe
    # we want to tranpose this and then reshape to head_dim * base_num_query_heads on the first dim
    w_post = np.concatenate(
        [
            var[f"layers.{layer_idx}.attention.wo.weight"].type(torch.float32).numpy().astype(CAST_DTYPE)
            for var in chkpt_vars
        ],
        axis=1,
    ).transpose()

    w_post = np.reshape(w_post, [base_num_query_heads, head_dim, base_num_query_heads * head_dim])

    # scale the query weights
    # NOTE: the np.sqrt here will silently cast to float64, so we add a manual cast to ensure the CAST_DTYPE is respected
    self_attention["query"]["kernel"] = wq / (np.sqrt(head_dim).astype(CAST_DTYPE)) if scale_query else wq  # pylint: disable=E1137

    self_attention["key"]["kernel"] = wk  # pylint: disable=E1137
    self_attention["value"]["kernel"] = wv  # pylint: disable=E1137
    self_attention["out"]["kernel"] = w_post  # pylint: disable=E1137

    jax_weights["decoder"][layer_name]["self_attention"] = self_attention
  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  # layer weight pre and post self attention norm ################
  max_logging.log("Processing pre and post self attention norms")

  # self attention layer norm and swap the layer index
  for layer_idx in tqdm(range(base_num_decoder_layers), desc="layers", leave=False):
    layer_name = f"layers_{layer_idx}"
    layer_weight = {"pre_self_attention_layer_norm": {"scale": None}, "post_self_attention_layer_norm": {"scale": None}}
    pre_self_attention_layernorm_name = (
        f"layers.{layer_idx}.attention.wqkv.layer_norm_weight"
        if is_llama4_model
        else f"layers.{layer_idx}.attention_norm.weight"
    )
    pre_self_attention_layernorm = (
        chkpt_vars[0][pre_self_attention_layernorm_name].type(torch.float32).numpy().astype(CAST_DTYPE)
    )
    post_self_attention_layernorm_name = (
        f"layers.{layer_idx}.feed_forward.norm.weight" if is_llama4_model else f"layers.{layer_idx}.ffn_norm.weight"
    )
    post_self_attention_layernorm = (
        chkpt_vars[0][post_self_attention_layernorm_name].type(torch.float32).numpy().astype(CAST_DTYPE)
    )
    jax_weights["decoder"][layer_name]["pre_self_attention_layer_norm"]["scale"] = pre_self_attention_layernorm
    jax_weights["decoder"][layer_name]["post_self_attention_layer_norm"]["scale"] = post_self_attention_layernorm
  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  # layer weights ################################################
  max_logging.log("Processing layer weights")

  for layer_idx in tqdm(range(base_num_decoder_layers), desc="layers", leave=False):
    # TODO: get rid of layer_weight support for the `num_experts is None` branch
    layer_name = f"layers_{layer_idx}"
    if num_experts is None:
      if is_llama4_model:
        raise NotImplementedError("Non-MoE model conversion logic for Llama4 is not yet supported!")
      wi_0 = np.concatenate(
          [
              var[f"layers.{layer_idx}.feed_forward.w1.weight"].type(torch.float32).numpy().astype(CAST_DTYPE)
              for var in chkpt_vars
          ],
          axis=0,
      ).transpose()
      wi_1 = np.concatenate(
          [
              var[f"layers.{layer_idx}.feed_forward.w3.weight"].type(torch.float32).numpy().astype(CAST_DTYPE)
              for var in chkpt_vars
          ],
          axis=0,
      ).transpose()
      wo = np.concatenate(
          [
              var[f"layers.{layer_idx}.feed_forward.w1.weight"].type(torch.float32).numpy().astype(CAST_DTYPE)
              for var in chkpt_vars
          ],
          axis=1,
      ).transpose()
      if layer_weight["mlp"]["wi_0"]["kernel"] is None:
        stack_shape = (base_num_decoder_layers,)
        layer_weight["mlp"]["wi_0"]["kernel"] = np.zeros(stack_shape + wi_0.shape, dtype=CAST_DTYPE)
        layer_weight["mlp"]["wi_1"]["kernel"] = np.zeros(stack_shape + wi_1.shape, dtype=CAST_DTYPE)
        layer_weight["mlp"]["wo"]["kernel"] = np.zeros(stack_shape + wo.shape, dtype=CAST_DTYPE)

      layer_weight["mlp"]["wi_0"]["kernel"][layer_idx, ...] = wi_0  # pytype: disable=unsupported-operands
      layer_weight["mlp"]["wi_1"]["kernel"][layer_idx, ...] = wi_1  # pytype: disable=unsupported-operands
      layer_weight["mlp"]["wo"]["kernel"][layer_idx, ...] = wo  # pytype: disable=unsupported-operands
    else:

      gate = chkpt_vars[0][f"layers.{layer_idx}.feed_forward.router_DE"].type(torch.float32).numpy().astype(CAST_DTYPE)
      jax_weights["decoder"][layer_name]["DeepSeekMoeBlock_0"]["MoeBlock_0"]["gate"]["kernel"] = gate
      base_emb_dim = model_params["base_emb_dim"]

      wi_0 = np.concatenate(
          [
              var[f"layers.{layer_idx}.feed_forward.experts.moe_w_in_eD_F"]
              .type(torch.float32)
              .numpy()
              .astype(CAST_DTYPE)
              .reshape(num_experts, base_emb_dim, -1)
              for var in chkpt_vars
          ],
          axis=2,
      )
      # NOTE: should probably update this to be more rigorous, but this should be fine for now
      f_dim = wi_0.shape[-1]
      wi_1 = np.concatenate(
          [
              var[f"layers.{layer_idx}.feed_forward.experts.moe_w_swiglu_eD_F"]
              .type(torch.float32)
              .numpy()
              .reshape(num_experts, base_emb_dim, -1)
              .astype(CAST_DTYPE)
              for var in chkpt_vars
          ],
          axis=2,
      )
      wo = np.concatenate(
          [
              var[f"layers.{layer_idx}.feed_forward.experts.moe_w_out_eF_D"]
              .type(torch.float32)
              .numpy()
              .astype(CAST_DTYPE)
              .reshape(num_experts, -1, base_emb_dim)
              for var in chkpt_vars
          ],
          axis=1,
      )

      # ei, li = k, layer_idx
      # routed experts
      jax_weights["decoder"][layer_name]["DeepSeekMoeBlock_0"]["MoeBlock_0"]["wi_0"] = wi_0
      jax_weights["decoder"][layer_name]["DeepSeekMoeBlock_0"]["MoeBlock_0"]["wi_1"] = wi_1
      jax_weights["decoder"][layer_name]["DeepSeekMoeBlock_0"]["MoeBlock_0"]["wo"] = wo

      # shared experts
      # TODO: make this optional for setups that don't use shared experts
      wi_0 = np.concatenate(
          [
              var[f"layers.{layer_idx}.feed_forward.w_in_shared_FD.weight"].type(torch.float32).numpy().astype(CAST_DTYPE)
              for var in chkpt_vars
          ],
          axis=0,
      ).transpose()
      wi_1 = np.concatenate(
          [
              var[f"layers.{layer_idx}.feed_forward.w_swiglu_FD.weight"].type(torch.float32).numpy().astype(CAST_DTYPE)
              for var in chkpt_vars
          ],
          axis=0,
      ).transpose()
      wo = np.concatenate(
          [
              var[f"layers.{layer_idx}.feed_forward.w_out_shared_DF.weight"].type(torch.float32).numpy().astype(CAST_DTYPE)
              for var in chkpt_vars
          ],
          axis=1,
      ).transpose()

      # TODO: make this optional for setups that don't use shared experts
      jax_weights["decoder"][layer_name]["DeepSeekMoeBlock_0"]["shared_experts"]["wi_0"]["kernel"] = wi_0
      jax_weights["decoder"][layer_name]["DeepSeekMoeBlock_0"]["shared_experts"]["wi_1"]["kernel"] = wi_1
      jax_weights["decoder"][layer_name]["DeepSeekMoeBlock_0"]["shared_experts"]["wo"]["kernel"] = wo

      gc.collect()
  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  del chkpt_vars
  gc.collect()
  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))
  return jax_weights


def convert_to_jax_weights(base_model_path: str, model_size: str):
  """
  Function to convert the checkpoint at base_model_path into Orbax checkpoint
  for MaxText and output jax_weights ready for MaxText

  Attributes:
    base_model_path: checkpoint path
    model_size: llama2-7b to 70b, mistral-7b, or mixtral-8x7b, mixtral-8x22b
  """
  model_params = MODEL_PARAMS_DICT[model_size]
  mem_info = psutil.Process()
  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  max_logging.log(f"Loading the base model from {base_model_path}")

  return _convert_pytorch_to_jax_weights(base_model_path, model_size, model_params, mem_info)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--base-model-path", type=str, required=True)
  parser.add_argument("--maxtext-model-path", type=str, required=True)
  parser.add_argument("--model-size", type=str, required=True)
  parser.add_argument("--use-ocdbt", type=str2bool, required=False, default=True)
  parser.add_argument("--use-zarr3", type=str2bool, required=False, default=True)
  args = parser.parse_args()

  if args.model_size not in MODEL_PARAMS_DICT or not args.model_size.startswith("llama4"):
    raise NotImplementedError("Currently, we only support llama4 models but got " + args.model_size)

  os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={SIMULATED_CPU_DEVICES_COUNT}"
  base_weights_path = args.maxtext_model_path

  save_weights_to_checkpoint(
      args.maxtext_model_path,
      convert_to_jax_weights(args.base_model_path, args.model_size),
      SIMULATED_CPU_DEVICES_COUNT,
      args.use_ocdbt,
      args.use_zarr3,
  )
  max_logging.log(f"Successfully saved base_weights to {base_weights_path}.")
