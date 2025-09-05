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

python3 -m MaxText.convert_gpt_oss_unscanned_ckpt --base-model-path <path/to/hf/ckpt> \
    --maxtext-model-path <GCS/path/to/save/new/maxtext/ckpt> --model-size gpt-oss-20b
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
from MaxText.llama_or_mistral_ckpt import save_weights_to_checkpoint


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


MODEL_PARAMS_DICT = {
    "gpt-oss-20b": {
        "base_emb_dim": 2880,
        "base_num_query_heads": 64,
        "base_num_kv_heads": 8,
        "head_dim": 64,
        "base_num_decoder_layers": 24,
    },
    "gpt-oss-120b": {
        "base_emb_dim": 2880,
        "base_num_query_heads": 64,
        "base_num_kv_heads": 8,
        "head_dim": 64,
        "base_num_decoder_layers": 36,
    },
}


def _hf_to_maxtext_mapping(layer_idx: int = -1) -> dict:
  """
  Returns a mapping from HuggingFace model weight names to MaxText model weight names.

  Args:
    layer_idx (int): Layer index.

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
      f"model.layers.{layer_idx}.self_attn.q_proj.bias": f"layers.{layer_idx}.attention.wq.bias",
      f"model.layers.{layer_idx}.self_attn.k_proj.bias": f"layers.{layer_idx}.attention.wk.bias",
      f"model.layers.{layer_idx}.self_attn.v_proj.bias": f"layers.{layer_idx}.attention.wv.bias",
      f"model.layers.{layer_idx}.self_attn.o_proj.bias": f"layers.{layer_idx}.attention.wo.bias",
      f"model.layers.{layer_idx}.self_attn.sinks": f"layers.{layer_idx}.attention.sinks",
      # MoE
      f"model.layers.{layer_idx}.mlp.router.weight": f"layers.{layer_idx}.feed_forward.gate.weight",
      f"model.layers.{layer_idx}.mlp.router.bias": f"layers.{layer_idx}.feed_forward.gate.bias",
      f"model.layers.{layer_idx}.mlp.experts.gate_up_proj": f"layers.{layer_idx}.feed_forward.experts.gate_up_proj",
      f"model.layers.{layer_idx}.mlp.experts.gate_up_proj_bias": f"layers.{layer_idx}.feed_forward.experts.gate_up_proj_bias",
      f"model.layers.{layer_idx}.mlp.experts.down_proj": f"layers.{layer_idx}.feed_forward.experts.down_proj",
      f"model.layers.{layer_idx}.mlp.experts.down_proj_bias": f"layers.{layer_idx}.feed_forward.experts.down_proj_bias",
  }


def _convert_huggingface_to_jax_weights(base_model_path: str, model_size: str, model_params: dict, mem_info: psutil.Process):
  # model params
  base_num_decoder_layers = model_params["base_num_decoder_layers"]
  base_emb_dim = model_params["base_emb_dim"]
  base_num_query_heads = model_params["base_num_query_heads"]
  base_num_kv_heads = model_params["base_num_kv_heads"]
  head_dim = model_params["head_dim"]

  # load model
  max_logging.log(f"Loading the base model from {base_model_path}")
  ckpt_paths = sorted(pathlib.Path(base_model_path).glob("[!.]*.safetensors"))
  chkpt_vars = {}
  for i, ckpt_path in enumerate(ckpt_paths):
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
      },
  }
  for layer_idx in range(base_num_decoder_layers):
    jax_weights["decoder"][f"layers_{layer_idx}"] = {
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

  # self attention ###############################################
  max_logging.log("Processing self attention")
  for layer_idx in tqdm(range(base_num_decoder_layers), desc="layers", leave=False):
    self_attention = jax_weights["decoder"][f"layers_{layer_idx}"]["GptOssAttention"]

    wq = _pt_to_np(chkpt_vars[f"layers.{layer_idx}.attention.wq.weight"], cast_dtype=CAST_DTYPE)
    wk = _pt_to_np(chkpt_vars[f"layers.{layer_idx}.attention.wk.weight"], cast_dtype=CAST_DTYPE)
    wv = _pt_to_np(chkpt_vars[f"layers.{layer_idx}.attention.wv.weight"], cast_dtype=CAST_DTYPE)
    w_post = _pt_to_np(chkpt_vars[f"layers.{layer_idx}.attention.wo.weight"], cast_dtype=CAST_DTYPE)

    # NOTE: not scale the query weights in checkpoint, but apply query_pre_attn_scalar=1/np.sqrt(head_dim) for attention
    # (num_attention_heads * head_dim, hidden_size) -> (hidden_size, num_attention_heads * head_dim) -> (hidden_size, num_attention_heads, head_dim)
    # [embed, q, head_dim]
    self_attention["query"]["kernel"] = wq.transpose().reshape([base_emb_dim, base_num_query_heads, head_dim])
    # [embed, kv, head_dim]
    self_attention["key"]["kernel"] = wk.transpose().reshape([base_emb_dim, base_num_kv_heads, head_dim])
    # [embed, kv, head_dim]
    self_attention["value"]["kernel"] = wv.transpose().reshape([base_emb_dim, base_num_kv_heads, head_dim])
    # (hidden_size, num_attention_heads * head_dim) -> (num_attention_heads * head_dim, hidden_size) -> (num_attention_heads, head_dim, hidden_size)
    # [q, head_dim, embed]
    self_attention["out"]["kernel"] = w_post.transpose().reshape([base_num_query_heads, head_dim, base_emb_dim])

    sinks = _pt_to_np(chkpt_vars[f"layers.{layer_idx}.attention.sinks"], cast_dtype=CAST_DTYPE)
    wq_bias = _pt_to_np(chkpt_vars[f"layers.{layer_idx}.attention.wq.bias"], cast_dtype=CAST_DTYPE)
    wk_bias = _pt_to_np(chkpt_vars[f"layers.{layer_idx}.attention.wk.bias"], cast_dtype=CAST_DTYPE)
    wv_bias = _pt_to_np(chkpt_vars[f"layers.{layer_idx}.attention.wv.bias"], cast_dtype=CAST_DTYPE)
    w_post_bias = _pt_to_np(chkpt_vars[f"layers.{layer_idx}.attention.wo.bias"], cast_dtype=CAST_DTYPE)

    self_attention["sinks"] = sinks
    self_attention["query"]["bias"] = wq_bias.reshape([base_num_query_heads, head_dim])
    self_attention["key"]["bias"] = wk_bias.reshape([base_num_kv_heads, head_dim])
    self_attention["value"]["bias"] = wv_bias.reshape([base_num_kv_heads, head_dim])
    self_attention["out"]["bias"] = w_post_bias

  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  # layer weight pre and post self attention norm ################
  max_logging.log("Processing pre and post self attention norms")
  for layer_idx in tqdm(range(base_num_decoder_layers), desc="layers", leave=False):
    layer_weight = jax_weights["decoder"][f"layers_{layer_idx}"]
    pre_self_attention_layernorm = _pt_to_np(chkpt_vars[f"layers.{layer_idx}.attention_norm.weight"], cast_dtype=CAST_DTYPE)
    post_self_attention_layernorm = _pt_to_np(chkpt_vars[f"layers.{layer_idx}.ffn_norm.weight"], cast_dtype=CAST_DTYPE)

    layer_weight["pre_self_attention_layer_norm"]["scale"] = pre_self_attention_layernorm  # pylint: disable=E1137
    layer_weight["post_self_attention_layer_norm"]["scale"] = post_self_attention_layernorm  # pylint: disable=E1137

  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  # layer weights ################################################
  max_logging.log("Processing layer weights")

  for layer_idx in tqdm(range(base_num_decoder_layers), desc="layers", leave=False):
    mlp_weight = jax_weights["decoder"][f"layers_{layer_idx}"]["GptOssMlp"]

    gate = _pt_to_np(chkpt_vars[f"layers.{layer_idx}.feed_forward.gate.weight"], cast_dtype=CAST_DTYPE)
    gate_bias = _pt_to_np(chkpt_vars[f"layers.{layer_idx}.feed_forward.gate.bias"], cast_dtype=CAST_DTYPE)
    wi_0_1 = _pt_to_np(chkpt_vars[f"layers.{layer_idx}.feed_forward.experts.gate_up_proj"], cast_dtype=CAST_DTYPE)
    wi_0_1_bias = _pt_to_np(chkpt_vars[f"layers.{layer_idx}.feed_forward.experts.gate_up_proj_bias"], cast_dtype=CAST_DTYPE)
    wo = _pt_to_np(chkpt_vars[f"layers.{layer_idx}.feed_forward.experts.down_proj"], cast_dtype=CAST_DTYPE)
    wo_bias = _pt_to_np(chkpt_vars[f"layers.{layer_idx}.feed_forward.experts.down_proj_bias"], cast_dtype=CAST_DTYPE)

    # router
    mlp_weight["gate"]["kernel"] = gate.transpose()
    mlp_weight["gate"]["bias"] = gate_bias
    # experts.gate_up_proj: de-interleave last dim, even for gate, odd for up_proj
    wi_0 = wi_0_1[..., ::2]
    wi_1 = wi_0_1[..., 1::2]
    del wi_0_1
    wi_0_bias = wi_0_1_bias[..., ::2]
    wi_1_bias = wi_0_1_bias[..., 1::2]
    del wi_0_1_bias
    mlp_weight["wi_0"] = wi_0
    mlp_weight["wi_1"] = wi_1
    mlp_weight["wi_0_bias"] = wi_0_bias
    mlp_weight["wi_1_bias"] = wi_1_bias
    # experts.down_proj
    mlp_weight["wo"] = wo
    mlp_weight["wo_bias"] = wo_bias

    gc.collect()

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
