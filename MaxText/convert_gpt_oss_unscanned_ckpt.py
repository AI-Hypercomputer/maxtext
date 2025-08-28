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

"""Convert weights from a GPT-OSS style model to a MaxText one.

Example cmd:

python3 -m MaxText.convert_gpt_oss_unscanned_ckpt --base_model_path <path/to/TODO/ckpt> \
    --maxtext_model_path <GCS/path/to/save/new/maxtext/ckpt> --model_size gpt-oss-20b
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
from safetensors import safe_open
import torch
from tqdm import tqdm

from MaxText import max_logging
from MaxText.inference_utils import str2bool
from MaxText.llama_or_mistral_ckpt import save_weights_to_checkpoint, MODEL_PARAMS_DICT
from MaxText.llama4_ckpt_unscanned import _pt_to_np


SIMULATED_CPU_DEVICES_COUNT = 16

# NOTE: numpy doesn't have native support for bfloat16, so
# we'll use ml_dtypes instead (which is quasi native)
# NOTE: it's incredibly silly but you can't directly cast from
# a torch tensor of type bfloat16 to a numpy array of type bfloat16
# so we have to cast to float32 first
CAST_DTYPE = ml_dtypes.bfloat16


def _hf_to_maxtext_mapping(layer_idx: int = -1, expert_idx: int = -1) -> dict:
  """
  Returns a mapping from HuggingFace model weight names to MaxText model weight names.

  Args:
    layer_idx (int): Layer index.
    expert_idx (int): Expert index.

  Returns:
    dict [str, str]: Mapping from HuggingFace model weight names to MaxText model weight names.
  """
  # pylint: disable=line-too-long
  return {
      "model.embed_tokens.weight": "tok_embeddings.weight",
      "model.norm.weight": "norm.weight",
      "lm_head.weight": "output.weight",
      # layernorm
      f"model.layers.{layer_idx}.input_layernorm.weight": f"layers.{layer_idx}.attention_norm.weight",
      f"model.layers.{layer_idx}.post_attention_layernorm.weight": f"layers.{layer_idx}.ffn_norm.weight",
      # attention
      f"model.layers.{layer_idx}.self_attn.q_proj.weight": f"layers.{layer_idx}.attention.wq.weight",
      f"model.layers.{layer_idx}.self_attn.k_proj.weight": f"layers.{layer_idx}.attention.wk.weight",
      f"model.layers.{layer_idx}.self_attn.v_proj.weight": f"layers.{layer_idx}.attention.wv.weight",
      f"model.layers.{layer_idx}.self_attn.o_proj.weight": f"layers.{layer_idx}.attention.wo.weight",
      # attention additional
      f"model.layers.{layer_idx}.self_attn.sinks": f"layers.{layer_idx}.attention.sinks",
      f"model.layers.{layer_idx}.self_attn.q_proj.bias": f"layers.{layer_idx}.attention.wq.bias",
      f"model.layers.{layer_idx}.self_attn.k_proj.bias": f"layers.{layer_idx}.attention.wk.bias",
      f"model.layers.{layer_idx}.self_attn.v_proj.bias": f"layers.{layer_idx}.attention.wv.bias",
      f"model.layers.{layer_idx}.self_attn.o_proj.bias": f"layers.{layer_idx}.attention.wo.bias",
      # MoE
      f"model.layers.{layer_idx}.mlp.router.weight": f"layers.{layer_idx}.mlp.router.weight",
      f"model.layers.{layer_idx}.mlp.router.bias": f"layers.{layer_idx}.mlp.router.bias",
      f"model.layers.{layer_idx}.mlp.experts.gate_up_proj": f"layers.{layer_idx}.mlp.experts.gate_up_proj",
      f"model.layers.{layer_idx}.mlp.experts.gate_up_proj.bias": f"layers.{layer_idx}.mlp.experts.gate_up_proj.bias",
      f"model.layers.{layer_idx}.mlp.experts.down_proj": f"layers.{layer_idx}.mlp.experts.down_proj",
      f"model.layers.{layer_idx}.mlp.experts.down_proj.bias": f"layers.{layer_idx}.mlp.experts.down_proj",
      # f"language_model.model.layers.{layer_idx}.feed_forward.shared_expert.up_proj.weight": f"layers.{layer_idx}.feed_forward.shared_experts.up_proj.weight",
      # f"language_model.model.layers.{layer_idx}.feed_forward.shared_expert.gate_proj.weight": f"layers.{layer_idx}.feed_forward.shared_experts.gate_proj.weight",
      # f"language_model.model.layers.{layer_idx}.feed_forward.shared_expert.down_proj.weight": f"layers.{layer_idx}.feed_forward.shared_experts.down_proj.weight",
      # # MOE model
      # f"model.layers.{layer_idx}.block_sparse_moe.gate.weight": f"layers.{layer_idx}.feed_forward.gate.weight",
      # f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w1.weight": f"layers.{layer_idx}.feed_forward.experts.{expert_idx}.w1.weight",
      # f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w2.weight": f"layers.{layer_idx}.feed_forward.experts.{expert_idx}.w2.weight",
      # f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w3.weight": f"layers.{layer_idx}.feed_forward.experts.{expert_idx}.w3.weight",
      # # FFN
      # f"model.layers.{layer_idx}.mlp.gate_proj.weight": f"layers.{layer_idx}.feed_forward.w1.weight",
      # f"model.layers.{layer_idx}.mlp.up_proj.weight": f"layers.{layer_idx}.feed_forward.w2.weight",
      # f"model.layers.{layer_idx}.mlp.down_proj.weight": f"layers.{layer_idx}.feed_forward.w3.weight",
  }


model_params = {
    # Attention
    "base_emb_dim": 2880,
    "base_num_query_heads": 64,
    "base_num_kv_heads": 8,
    "head_dim": 64,
    # "sliding_window_size": 128,
    # "attention_bias": True,
    # "attention_sink": True,
    # RoPE
    # "rope_type": "yarn",
    # "rope_max_timescale": 150_000,
    # "max_position_embeddings": 131072,
    # "original_max_position_embeddings": 4096,
    # "rope_factor": 32,
    # "beta_fast": 32,
    # "beta_slow": 1,
    # MLP
    "base_mlp_dim": 2880,
    # "base_moe_mlp_dim": 2880,
    # "mlp_activations": ["sigmoid","linear"],
    # "mlp_activations_limit": 7.0,
    # "routed_bias": True,
    # "mlp_bias": True,
    "num_experts": 32,
    # "num_experts_per_tok": 4,
    # # General
    "base_num_decoder_layers": 24,
    # "vocab_size": 201088,
    # "normalization_layer_epsilon": 1.0e-5,
    # "enable_dropout": False,
    # "logits_via_embedding": False,
    # "decoder_block": "gpt_oss",
    # "inhomogeneous_layer_cycle_interval": 2,
}

base_num_decoder_layers = model_params["base_num_decoder_layers"]


jax_weights = {
    "token_embedder": {"embedding": None},
    "decoder": {
        "decoder_norm": {"scale": None},
        "logits_dense": {"kernel": None},
    },
}

for i in range(base_num_decoder_layers):
  jax_weights["decoder"]["layers_{i}"] = {
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


def _convert_huggingface_to_jax_weights(
    base_model_path: str, model_size: str, model_params: dict, mem_info: psutil.Process
):

  max_logging.log(f"Loading the base model from {base_model_path}")
  ckpt_paths = sorted(pathlib.Path(base_model_path).glob("[!.]*.safetensors"))
  chkpt_vars = {}
  for i, ckpt_path in enumerate(ckpt_paths):
    max_logging.log(f"Loading checkpoint {i+1} of {len(ckpt_paths)} ...")

    with safe_open(ckpt_path, framework="pt", device="cpu") as f:
      for key in f.keys():
        parts = key.split(".")
        # TODO check
        layer = int(parts[2]) if "layers" in key else 0
        mapped_key = _hf_to_maxtext_mapping(layer)[key]
        chkpt_vars[mapped_key] = f.get_tensor(key)

  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))


  # decoder norm scale ###########################################
  max_logging.log("Processing decoder norm scale")
  decoder_norm_scale = chkpt_vars["norm.weight"].to(torch.float32).numpy().astype(CAST_DTYPE)
  jax_weights["decoder"]["decoder_norm"]["scale"] = decoder_norm_scale

  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  # logits dense #################################################
  max_logging.log("Processing logits dense")

  jax_weights["decoder"]["logits_dense"]["kernel"] = (
      chkpt_vars["output.weight"].to(torch.float32).numpy().astype(CAST_DTYPE).transpose()[:, :vocab_size]
  )

  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  # token embedding ##############################################
  max_logging.log("Processing token embeddings")

  jax_weights["token_embedder"]["embedding"] = (
      chkpt_vars["tok_embeddings.weight"].to(torch.float32).numpy().astype(CAST_DTYPE)
  )

  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))


  # layer weight pre and post self attention norm ################
  max_logging.log("Processing pre and post self attention norms")

  for layer_idx in tqdm(range(base_num_decoder_layers), desc="layers", leave=False):
    layer_weight = jax_weights["decoder"][f"layers_{layer_idx}"]

    pre_self_attention_layernorm = (
        chkpt_vars[f"layers.{layer_idx}.attention_norm.weight"].type(torch.float32).numpy().astype(CAST_DTYPE)
    )
    post_self_attention_layernorm = (
        chkpt_vars[f"layers.{layer_idx}.ffn_norm.weight"].type(torch.float32).numpy().astype(CAST_DTYPE)
    )

    layer_weight["pre_self_attention_layer_norm"]["scale"] = pre_self_attention_layernorm  # pylint: disable=E1137
    layer_weight["post_self_attention_layer_norm"]["scale"] = post_self_attention_layernorm  # pylint: disable=E1137

