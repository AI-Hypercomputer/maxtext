"""
Copyright 2025 Google LLC
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

r"""Convert weights from a DeepSeek style model to a MaxText one.

Example cmd:
To save a ckpt
python3 MaxText/convert_deepseek_ckpt.py --base_model_path <path/to/meta/ckpt> \
    --maxtext_model_path <GCS/path/to/save/new/maxtext/ckpt> --model_size deepseek2-16b
"""
# pylint: disable=g-line-too-long
import argparse
import pathlib
import os
import gc
import re
import logging
from dataclasses import dataclass

os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import jax
from jax import tree
from flax.training import train_state
import torch
import psutil
from tqdm import tqdm

import max_logging
from train import save_checkpoint
import checkpointing
from safetensors import safe_open
import max_utils

MODEL_PARAMS_DICT = {
    "deepseek2-16b": {
        "num_layers": 27,
        "first_num_dense_layers": 1,
        "base_num_query_heads": 16,
        "base_num_kv_heads": 16,
        "vocab": 102400,
        "base_emb_dim": 2048,
        "base_mlp_dim": 10944,
        "base_moe_mlp_dim": 1408,
        "num_experts": 64,
        "shared_experts": 2,
        "q_lora_rank": 0,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
    },
    "deepseek3-671b": {
        "num_layers": 61,
        "first_num_dense_layers": 3,
        "base_num_query_heads": 128,
        "base_num_kv_heads": 128,
        "vocab": 129280,
        "base_emb_dim": 7168,
        "base_mlp_dim": 18432,
        "base_moe_mlp_dim": 2048,
        "num_experts": 256,
        "shared_experts": 1,
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
    },
}


SIMULATED_CPU_DEVICES_COUNT = 1


def _hf_to_maxtext_mapping(layer_idx, num_experts, first_num_dense_layers) -> dict:
  # HF MLP self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
  # pylint: disable=line-too-long
  mapping = {
      "model.embed_tokens.weight": "token_embedder.embedding",
      "lm_head.weight": "logits_dense.kernel",
      "model.norm.weight": "decoder_norm.scale",
  }

  if layer_idx < first_num_dense_layers:
    # Dense layers mapping
    mapping.update(
        {
            f"model.layers.{layer_idx}.input_layernorm.weight": f"dense_layers.{layer_idx}.pre_self_attention_layer_norm.scale",
            f"model.layers.{layer_idx}.post_attention_layernorm.weight": f"dense_layers.{layer_idx}.post_self_attention_layer_norm.scale",
            f"model.layers.{layer_idx}.self_attn.q_proj.weight": f"dense_layers.{layer_idx}.self_attention.query.kernel",
            # f"model.layers.{layer_idx}.self_attn.q_a_proj.weight": f"dense_layers.{layer_idx}.self_attention.wq_a.kernel",
            # f"model.layers.{layer_idx}.self_attn.q_a_layernorm.weight": f"dense_layers.{layer_idx}.self_attention.q_norm.kernel",
            # f"model.layers.{layer_idx}.self_attn.q_b_proj.weight": f"dense_layers.{layer_idx}.self_attention.wq_b.kernel",
            f"model.layers.{layer_idx}.self_attn.kv_a_proj_with_mqa.weight": f"dense_layers.{layer_idx}.self_attention.wkv_a.kernel",
            f"model.layers.{layer_idx}.self_attn.kv_b_proj.weight": f"dense_layers.{layer_idx}.self_attention.wkv_b.kernel",
            f"model.layers.{layer_idx}.self_attn.o_proj.weight": f"dense_layers.{layer_idx}.self_attention.out.kernel",
            f"model.layers.{layer_idx}.mlp.gate_proj.weight": f"dense_layers.{layer_idx}.mlp.wi_0.kernel",
            f"model.layers.{layer_idx}.mlp.up_proj.weight": f"dense_layers.{layer_idx}.mlp.wi_1.kernel",
            f"model.layers.{layer_idx}.mlp.down_proj.weight": f"dense_layers.{layer_idx}.mlp.wo.kernel",
            f"model.layers.{layer_idx}.self_attn.kv_a_layernorm.weight": f"dense_layers.{layer_idx}.self_attention.kv_norm.scale",
        }
    )
  else:
    # MoE layers mapping
    moe_layer_idx_in_maxtext = layer_idx - first_num_dense_layers
    for expert_idx in range(num_experts):
      mapping.update(
          {
              f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight": f"moe_layers.{moe_layer_idx_in_maxtext}.DeepSeekMoeBlock_0.MoeBlock_0.{expert_idx}.wo",
              f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight": f"moe_layers.{moe_layer_idx_in_maxtext}.DeepSeekMoeBlock_0.MoeBlock_0.{expert_idx}.wi_0",
              f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight": f"moe_layers.{moe_layer_idx_in_maxtext}.DeepSeekMoeBlock_0.MoeBlock_0.{expert_idx}.wi_1",
          }
      )
    mapping.update(
        {
            f"model.layers.{layer_idx}.mlp.gate.weight": f"moe_layers.{moe_layer_idx_in_maxtext}.DeepSeekMoeBlock_0.MoeBlock_0.gate.kernel",
            f"model.layers.{layer_idx}.mlp.shared_experts.down_proj.weight": f"moe_layers.{moe_layer_idx_in_maxtext}.DeepSeekMoeBlock_0.shared_experts.wo.kernel",
            f"model.layers.{layer_idx}.mlp.shared_experts.gate_proj.weight": f"moe_layers.{moe_layer_idx_in_maxtext}.DeepSeekMoeBlock_0.shared_experts.wi_0.kernel",
            f"model.layers.{layer_idx}.mlp.shared_experts.up_proj.weight": f"moe_layers.{moe_layer_idx_in_maxtext}.DeepSeekMoeBlock_0.shared_experts.wi_1.kernel",
            f"model.layers.{layer_idx}.self_attn.kv_a_layernorm.weight": f"moe_layers.{moe_layer_idx_in_maxtext}.self_attention.kv_norm.scale",
            f"model.layers.{layer_idx}.self_attn.kv_a_proj_with_mqa.weight": f"moe_layers.{moe_layer_idx_in_maxtext}.self_attention.wkv_a.kernel",
            f"model.layers.{layer_idx}.self_attn.kv_b_proj.weight": f"moe_layers.{moe_layer_idx_in_maxtext}.self_attention.wkv_b.kernel",
            f"model.layers.{layer_idx}.self_attn.o_proj.weight": f"moe_layers.{moe_layer_idx_in_maxtext}.self_attention.out.kernel",
            f"model.layers.{layer_idx}.self_attn.q_proj.weight": f"moe_layers.{moe_layer_idx_in_maxtext}.self_attention.query.kernel",
            f"model.layers.{layer_idx}.input_layernorm.weight": f"moe_layers.{moe_layer_idx_in_maxtext}.pre_self_attention_layer_norm.scale",
            f"model.layers.{layer_idx}.post_attention_layernorm.weight": f"moe_layers.{moe_layer_idx_in_maxtext}.post_self_attention_layer_norm.scale",
        }
    )
  return mapping


def _convert_huggingface_to_jax_weights(base_model_path, model_size, model_params, mem_info):
  """Convert Huggingface Checkpoint to Jax."""
  base_num_decoder_layers = model_params["num_layers"]
  first_num_dense_layers = model_params["first_num_dense_layers"]
  base_num_query_heads = model_params["base_num_query_heads"]
  base_num_kv_heads = model_params["base_num_kv_heads"]
  vocab_size = model_params["vocab"]
  base_emb_dim = model_params["base_emb_dim"]
  base_mlp_dim = model_params["base_mlp_dim"]
  base_moe_mlp_dim = model_params["base_moe_mlp_dim"]
  num_experts = model_params["num_experts"]
  shared_experts = model_params["shared_experts"]
  q_lora_rank = model_params["q_lora_rank"]
  kv_lora_rank = model_params["kv_lora_rank"]
  qk_nope_head_dim = model_params["qk_nope_head_dim"]
  qk_rope_head_dim = model_params["qk_rope_head_dim"]
  v_head_dim = model_params["v_head_dim"]

  ckpt_paths = sorted(pathlib.Path(base_model_path).glob("[!.]*.safetensors"))
  chkpt_vars = {}
  for i, ckpt_path in enumerate(ckpt_paths):
    max_logging.log(f"Loading checkpoint {i+1} of {len(ckpt_paths)} ...")
    with safe_open(ckpt_path, framework="pt", device="cpu") as f:
      for key in f.keys():
        parts = key.split(".")
        layer = int(parts[2]) if "layers" in key else 0
        # TODO for now skip the fp8 scaling, need to handle it with fp8 -> bf16 conversion
        if not key.endswith("_scale_inv"):
          mapped_key = _hf_to_maxtext_mapping(layer, num_experts, first_num_dense_layers)[key]
          chkpt_vars[mapped_key] = f.get_tensor(key)

  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  # initialize the data structure for storing jax_weights
  jax_weights = {
      "decoder": {
          "dense_layers": {
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
                  "query": {"kernel": None},
              },
              "pre_self_attention_layer_norm": {"scale": None},
              "post_self_attention_layer_norm": {"scale": None},
          },
          "moe_layers": {
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
                  "query": {"kernel": None},
              },
              "pre_self_attention_layer_norm": {"scale": None},
              "post_self_attention_layer_norm": {"scale": None},
          },
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
  jax_weights["decoder"]["logits_dense"]["kernel"] = chkpt_vars["logits_dense.kernel"].to(torch.float16).numpy().transpose()
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
    self_attention = jax_weights["decoder"][f"{layer_key}"]["self_attention"]
    pre_self_attention_layer_norm = jax_weights["decoder"][f"{layer_key}"]["pre_self_attention_layer_norm"]
    post_self_attention_layer_norm = jax_weights["decoder"][f"{layer_key}"]["post_self_attention_layer_norm"]

    for layer_idx in tqdm(range(layer_value), desc=layer_key, leave=False):
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
      query = chkpt_vars[f"{layer_key}.{layer_idx}.self_attention.query.kernel"].to(torch.float16).numpy().transpose()

      wkv_b = np.reshape(wkv_b, [kv_lora_rank, base_num_query_heads, (qk_nope_head_dim + v_head_dim)])
      out = np.reshape(out, [base_num_query_heads, v_head_dim, base_emb_dim])
      query = np.reshape(query, [base_emb_dim, base_num_query_heads, (qk_nope_head_dim + qk_rope_head_dim)])

      if self_attention["kv_norm"]["scale"] is None:
        stack_shape = (layer_value,)
        self_attention["kv_norm"]["scale"] = np.zeros(stack_shape + kv_norm.shape, dtype=np.float16)
        self_attention["wkv_a"]["kernel"] = np.zeros(stack_shape + wkv_a.shape, dtype=np.float16)
        self_attention["wkv_b"]["kernel"] = np.zeros(stack_shape + wkv_b.shape, dtype=np.float16)
        self_attention["out"]["kernel"] = np.zeros(stack_shape + out.shape, dtype=np.float16)
        self_attention["query"]["kernel"] = np.zeros(stack_shape + query.shape, dtype=np.float16)
        pre_self_attention_layer_norm["scale"] = np.zeros(stack_shape + pre_self_attention.shape, dtype=np.float16)
        post_self_attention_layer_norm["scale"] = np.zeros(stack_shape + post_self_attention.shape, dtype=np.float16)
      self_attention["kv_norm"]["scale"][layer_idx, ...] = kv_norm
      self_attention["wkv_a"]["kernel"][layer_idx, ...] = wkv_a
      self_attention["wkv_b"]["kernel"][layer_idx, ...] = wkv_b
      self_attention["out"]["kernel"][layer_idx, ...] = out
      self_attention["query"]["kernel"][layer_idx, ...] = query
      pre_self_attention_layer_norm["scale"][layer_idx, ...] = pre_self_attention
      post_self_attention_layer_norm["scale"][layer_idx, ...] = post_self_attention

    # Re-order
    self_attention["kv_norm"]["scale"] = np.transpose(self_attention["kv_norm"]["scale"], axes=(1, 0))
    self_attention["wkv_a"]["kernel"] = np.transpose(self_attention["wkv_a"]["kernel"], axes=(1, 0, 2))
    self_attention["wkv_b"]["kernel"] = np.transpose(self_attention["wkv_b"]["kernel"], axes=(1, 0, 2, 3))
    self_attention["out"]["kernel"] = np.transpose(self_attention["out"]["kernel"], axes=(1, 0, 2, 3))
    self_attention["query"]["kernel"] = np.transpose(self_attention["query"]["kernel"], axes=(1, 0, 2, 3))
    pre_self_attention_layer_norm["scale"] = np.transpose(pre_self_attention_layer_norm["scale"], axes=(1, 0))
    post_self_attention_layer_norm["scale"] = np.transpose(post_self_attention_layer_norm["scale"], axes=(1, 0))

    jax_weights["decoder"][f"{layer_key}"]["self_attention"] = self_attention
    jax_weights["decoder"][f"{layer_key}"]["pre_self_attention_layer_norm"] = pre_self_attention_layer_norm
    jax_weights["decoder"][f"{layer_key}"]["post_self_attention_layer_norm"] = post_self_attention_layer_norm
    logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  # layer weights ################################################
  max_logging.log("Processing layer weights")
  for layer_key, layer_value in layers.items():

    if layer_key == "dense_layers":
      mlp = jax_weights["decoder"][f"{layer_key}"]["mlp"]
      for layer_idx in tqdm(range(layer_value), desc=layer_key, leave=False):
        wi_0 = chkpt_vars[f"{layer_key}.{layer_idx}.mlp.wi_0.kernel"].to(torch.float16).numpy().transpose()
        wi_1 = chkpt_vars[f"{layer_key}.{layer_idx}.mlp.wi_1.kernel"].to(torch.float16).numpy().transpose()
        wo = chkpt_vars[f"{layer_key}.{layer_idx}.mlp.wo.kernel"].to(torch.float16).numpy().transpose()

        if mlp["wo"]["kernel"] is None:
          stack_shape = (layer_value,)
          mlp["wi_0"]["kernel"] = np.zeros(stack_shape + wi_0.shape, dtype=np.float16)
          mlp["wi_1"]["kernel"] = np.zeros(stack_shape + wi_1.shape, dtype=np.float16)
          mlp["wo"]["kernel"] = np.zeros(stack_shape + wo.shape, dtype=np.float16)
        mlp["wi_0"]["kernel"][layer_idx, ...] = wi_0
        mlp["wi_1"]["kernel"][layer_idx, ...] = wi_1
        mlp["wo"]["kernel"][layer_idx, ...] = wo

      mlp["wi_0"]["kernel"] = np.transpose(mlp["wi_0"]["kernel"], axes=(1, 0, 2))
      mlp["wi_1"]["kernel"] = np.transpose(mlp["wi_1"]["kernel"], axes=(1, 0, 2))
      mlp["wo"]["kernel"] = np.transpose(mlp["wo"]["kernel"], axes=(1, 0, 2))
      jax_weights["decoder"][f"{layer_key}"]["mlp"] = mlp
      logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))
    else:
      moe = jax_weights["decoder"][f"{layer_key}"]["DeepSeekMoeBlock_0"]
      for layer_idx in tqdm(range(layer_value), desc=layer_key, leave=False):
        gate = chkpt_vars[f"{layer_key}.{layer_idx}.DeepSeekMoeBlock_0.MoeBlock_0.gate.kernel"].to(torch.float16).numpy().transpose()
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

        if moe["MoeBlock_0"]["gate"]["kernel"] is None:
          stack_shape = (layer_value,)
          moe["MoeBlock_0"]["gate"]["kernel"] = np.zeros(stack_shape + gate.shape, dtype=np.float16)
          moe["shared_experts"]["wi_0"]["kernel"] = np.zeros(stack_shape + shared_wi_0.shape, dtype=np.float16)
          moe["shared_experts"]["wi_1"]["kernel"] = np.zeros(stack_shape + shared_wi_1.shape, dtype=np.float16)
          moe["shared_experts"]["wo"]["kernel"] = np.zeros(stack_shape + shared_wo.shape, dtype=np.float16)
        moe["MoeBlock_0"]["gate"]["kernel"][layer_idx, ...] = gate
        moe["shared_experts"]["wi_0"]["kernel"][layer_idx, ...] = shared_wi_0
        moe["shared_experts"]["wi_1"]["kernel"][layer_idx, ...] = shared_wi_1
        moe["shared_experts"]["wo"]["kernel"][layer_idx, ...] = shared_wo

      moe["MoeBlock_0"]["gate"]["kernel"] = np.transpose(moe["MoeBlock_0"]["gate"]["kernel"], axes=(1, 0, 2))
      moe["shared_experts"]["wi_0"]["kernel"] = np.transpose(moe["shared_experts"]["wi_0"]["kernel"], axes=(1, 0, 2))
      moe["shared_experts"]["wi_1"]["kernel"] = np.transpose(moe["shared_experts"]["wi_1"]["kernel"], axes=(1, 0, 2))
      moe["shared_experts"]["wo"]["kernel"] = np.transpose(moe["shared_experts"]["wo"]["kernel"], axes=(1, 0, 2))

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
          stack_shape = (num_experts, layer_value)
          moe["MoeBlock_0"]["wi_0"] = np.zeros(stack_shape + wi_0.shape, dtype=np.float16)
          moe["MoeBlock_0"]["wi_1"] = np.zeros(stack_shape + wi_1.shape, dtype=np.float16)
          moe["MoeBlock_0"]["wo"] = np.zeros(stack_shape + wo.shape, dtype=np.float16)
        moe["MoeBlock_0"]["wi_0"][k, layer_idx, ...] = wi_0
        moe["MoeBlock_0"]["wi_1"][k, layer_idx, ...] = wi_1
        moe["MoeBlock_0"]["wo"][k, layer_idx, ...] = wo
      jax_weights["decoder"][f"{layer_key}"]["DeepSeekMoeBlock_0"] = moe
      logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  del chkpt_vars
  gc.collect()
  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))
  return jax_weights


def convert_to_jax_weights(base_model_path, model_size):
  """
  Function to convert the checkpoint at base_model_path into Orbax checkpoint
  for MaxText and output jax_weights ready for MaxText

  Attributes:
  base_model_path: checkpoint path
  model_size: model size in MODEL_PARAMS_DICT
  """
  model_params = MODEL_PARAMS_DICT[model_size]
  mem_info = psutil.Process()
  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))
  max_logging.log(f"Loading the base model from {base_model_path}")
  return _convert_huggingface_to_jax_weights(base_model_path, model_size, model_params, mem_info)


def save_jax_weights_to_checkpoint(maxtext_model_path, jax_weights):
  """
  Function to save jax_weights ready for MaxText to a parameters checkpoint

  Attributes:
  maxtext_model_path: Path to save the MaxText checkpoint to
  jax_weights: TODO
  """
  """Save maxtext parameter checkpoint."""

  mem_info = psutil.Process()
  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))
  gc.collect()
  mesh = jax.sharding.Mesh(jax.devices(), "checkpoint_sharding_axis")
  s1 = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("checkpoint_sharding_axis"))  # shards first axis
  s2 = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, "checkpoint_sharding_axis"))  # shards second axis
  s3 = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None))  # no sharding

  def checkpoint_device_put(arr):
    if arr.shape[0] % SIMULATED_CPU_DEVICES_COUNT == 0:
      max_logging.log("sharding first axis")
      return jax.device_put(arr, device=s1)
    elif len(arr.shape) > 1 and arr.shape[1] % SIMULATED_CPU_DEVICES_COUNT == 0:
      max_logging.log("sharding second axis")
      return jax.device_put(arr, device=s2)
    else:
      max_logging.log("no sharding was possible, replicating")
      return jax.device_put(arr, device=s3)

  # convert all weights to jax.numpy with sharding if applicable
  jax_weights_flat, jax_weights_struct = tree.flatten(jax_weights)
  jax_weights_new = []
  while len(jax_weights_flat) > 0:
    jax_weight = jax_weights_flat.pop(0)
    jax_weights_new.append(checkpoint_device_put(jax_weight))
    del jax_weight
    gc.collect()
    logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  jax_weights = tree.unflatten(jax_weights_struct, jax_weights_new)

  # dummy configs for the checkpoint_manager
  step_number_to_save_new_ckpt = 0
  enable_checkpointing = True
  async_checkpointing = False
  save_interval_steps = 1

  checkpoint_manager = checkpointing.create_orbax_checkpoint_manager(
      maxtext_model_path, enable_checkpointing, async_checkpointing, save_interval_steps
  )

  state_new = train_state.TrainState(
      step=0, apply_fn=None, params={"params": jax_weights}, tx=None, opt_state={}  # type: ignore
  )

  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))
  if checkpoint_manager is not None:
    if save_checkpoint(checkpoint_manager, step_number_to_save_new_ckpt, state_new):
      max_logging.log(f"saved a checkpoint at step {step_number_to_save_new_ckpt}")
    # Upon preemption, exit when and only when all ongoing saves are complete.
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--base_model_path", type=str, required=True)
  parser.add_argument("--maxtext_model_path", type=str, required=True)
  parser.add_argument("--model_size", type=str, required=True)
  args = parser.parse_args()

  if args.model_size not in MODEL_PARAMS_DICT:
    raise NotImplementedError

  os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={SIMULATED_CPU_DEVICES_COUNT}"
  save_jax_weights_to_checkpoint(args.maxtext_model_path, convert_to_jax_weights(args.base_model_path, args.model_size))
