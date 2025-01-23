"""
Copyright 2023 Google LLC
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

r"""Convert weights from a Llama or Mistral model to a MaxText one.

Usage:

Get LLaMA chkpt_vars from Meta

Example cmd:
To save a ckpt
python3 MaxText/llama_or_mistral_ckpt.py --base-model-path <path/to/meta/ckpt> \
    --maxtext-model-path <GCS/path/to/save/new/maxtext/ckpt> --model-size llama2-7b

The base model checkpoints should be in the format `{name}.{chkpt_idx}.pth` 
For example: `mistral-7b.00.pth`
For large size model (e.g. 70B model), this script requires large memory VM.
The script load and save weights in a single pass.
To fit less memory, modify convert() to load/save weights in multiple passes.
Each pass, load and save partial weights (subset of all weight variables).
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


MODEL_PARAMS_DICT = {
    "llama2-70b": {
        "num_layers": 80,
        "num_heads": 64,
        "num_kv_heads": 8,
        "dims_per_head": 128,
        "vocab": 32000,
    },
    "llama2-13b": {
        "num_layers": 40,
        "num_heads": 40,
        "num_kv_heads": 40,
        "dims_per_head": 128,
        "vocab": 32000,
    },
    "llama2-7b": {
        "num_layers": 32,
        "num_heads": 32,
        "num_kv_heads": 32,
        "dims_per_head": 128,
        "vocab": 32000,
    },
    "llama3-8b": {
        "num_layers": 32,
        "num_heads": 32,
        "num_kv_heads": 8,
        "dims_per_head": 128,
        "vocab": 128256,
    },
    "llama3-70b": {
        "num_layers": 80,
        "num_heads": 64,
        "num_kv_heads": 8,
        "dims_per_head": 128,
        "vocab": 128256,
    },
    "llama3.1-8b": {
        "num_layers": 32,
        "num_heads": 32,
        "num_kv_heads": 8,
        "dims_per_head": 128,
        "vocab": 128256,
    },
    "llama3.1-70b": {
        "num_layers": 80,
        "num_heads": 64,
        "num_kv_heads": 8,
        "dims_per_head": 128,
        "vocab": 128256,
    },
    "llama3.1-405b": {
        "num_layers": 126,
        "num_heads": 128,
        "num_kv_heads": 8,
        "dims_per_head": 128,
        "vocab": 128256,
    },
    "mistral-7b": {
        "num_layers": 32,
        "num_heads": 32,
        "num_kv_heads": 8,
        "dims_per_head": 128,
        "vocab": 32000,
        "base_emb_dim": 4096,
        "base_mlp_dim": 14336,
    },
    "mixtral-8x7b": {
        "num_layers": 32,
        "num_heads": 32,
        "num_kv_heads": 8,
        "dims_per_head": 128,
        "vocab": 32000,
        "base_emb_dim": 4096,
        "base_mlp_dim": 14336,
        "num_experts": 8,
    },
    "mixtral-8x22b": {
        "num_layers": 56,
        "num_heads": 48,
        "num_kv_heads": 8,
        "dims_per_head": 128,
        "vocab": 32768,
        "base_emb_dim": 6144,
        "base_mlp_dim": 16384,
        "num_experts": 8,
    },
}

SIMULATED_CPU_DEVICES_COUNT = 16


def _hf_mapping(layer_idx: int = -1, expert_idx: int = -1) -> dict:
  # pylint: disable=line-too-long
  return {
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
      f"layers.{layer_idx}.feed_forward.experts.{expert_idx}.w1.weight": f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w1.weight",
      f"layers.{layer_idx}.feed_forward.experts.{expert_idx}.w2.weight": f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w2.weight",
      f"layers.{layer_idx}.feed_forward.experts.{expert_idx}.w3.weight": f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w3.weight",
      # dense model
      f"layers.{layer_idx}.feed_forward.w1.weight": f"model.layers.{layer_idx}.mlp.gate_proj.weight",
      f"layers.{layer_idx}.feed_forward.w2.weight": f"model.layers.{layer_idx}.mlp.down_proj.weight",
      f"layers.{layer_idx}.feed_forward.w3.weight": f"model.layers.{layer_idx}.mlp.up_proj.weight",
  }


@dataclass
class _HFNamespaceMapper:
  """A class to dynamically map Mistral/Llama weight names to Huggingface weights
  if the checkpoint is from HF.
  """

  collection: dict
  delimiter: str = "."

  def __getitem__(self, key):
    if key in self.collection:
      return self.collection[key]  # original key takes precedence
    fields = key.split(self.delimiter)
    num_fields = [int(field) for field in fields if re.match(r"[0-9]+", field) is not None]
    mapping = _hf_mapping(*num_fields)
    if key not in mapping:
      raise ValueError(f"Key `{key}` is missing from the original collection and from the mapping.")
    new_key = mapping[key]
    if new_key not in self.collection:
      raise ValueError(f"New key `{new_key}` mapped from `{key}` is missing from the collection.")
    return self.collection[new_key]


def permute_to_match_maxtext_rope(arr):
  evens = arr[..., ::2]
  odds = arr[..., 1::2]
  return np.concatenate((evens, odds), axis=arr.ndim - 1)


def convert_to_jax_weights(base_model_path, model_size):
  """
  Function to convert the checkpoint at base_model_path into Orbax checkpoint
  for MaxText and output jax_weights ready for MaxText

  Attributes:
  base_model_path: checkpoint path
  model_size: llama2-7b to 70b, mistral-7b, or mixtral-8x7b, mixtral-8x22b
  """
  """Convert model to maxtext."""
  model_params = MODEL_PARAMS_DICT[model_size]
  base_num_decoder_layers = model_params["num_layers"]
  base_num_query_heads = model_params["num_heads"]
  head_dim = model_params["dims_per_head"]
  base_num_kv_heads = model_params["num_kv_heads"]
  vocab_size = model_params["vocab"]
  num_experts = model_params["num_experts"] if "num_experts" in model_params else None
  mem_info = psutil.Process()
  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  max_logging.log(f"Loading the base model from {base_model_path}")
  # Skip any hidden files for checkpoints
  ckpt_paths = sorted(pathlib.Path(base_model_path).glob("[!.]*.pth"))
  chkpt_vars = {}
  for i, ckpt_path in enumerate(ckpt_paths):
    max_logging.log(f"Loading checkpoint {i+1} of {len(ckpt_paths)} ...")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    chkpt_vars[int(ckpt_path.name.split(".", maxsplit=2)[1])] = checkpoint
  chkpt_vars = [chkpt_vars[i] for i in sorted(list(chkpt_vars.keys()))]
  # map weight names if they use HuggingFace instead of PyTorch convention
  chkpt_vars = [_HFNamespaceMapper(var) for var in chkpt_vars]

  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  # initialize the data structure for storing jax_weights
  layer_key = "MoeBlock_0" if num_experts else "mlp"
  jax_weights = {
      "decoder": {
          "layers": {
              layer_key: {},
              "pre_self_attention_layer_norm": {},
              "post_self_attention_layer_norm": {},
              "self_attention": {},
          },
          "decoder_norm": {"scale": None},
          "logits_dense": {"kernel": None},
      },
      "token_embedder": {"embedding": None},
  }

  # decoder norm scale ###########################################
  max_logging.log("Processing decoder norm scale")
  decoder_norm_scale = chkpt_vars[0]["norm.weight"].type(torch.float16).numpy()
  jax_weights["decoder"]["decoder_norm"]["scale"] = decoder_norm_scale

  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  # logits dense #################################################
  max_logging.log("Processing logits dense")
  logits_dense = np.concatenate(
      [var["output.weight"].type(torch.float16).numpy() for var in chkpt_vars], axis=0
  ).transpose()[:, :vocab_size]
  jax_weights["decoder"]["logits_dense"]["kernel"] = logits_dense

  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  # token embedding ##############################################
  max_logging.log("Processing token embeddings")
  if model_size[:6] == "llama3":
    token_embedder = np.concatenate([var["tok_embeddings.weight"].type(torch.float16).numpy() for var in chkpt_vars], axis=0)
  else:
    token_embedder = np.concatenate(
        [var["tok_embeddings.weight"].type(torch.float16).numpy() for var in chkpt_vars], axis=1
    )[:vocab_size, :]
  jax_weights["token_embedder"]["embedding"] = token_embedder
  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  # self attention ###############################################
  max_logging.log("Processing self attention")
  self_attention = {
      "query": {"kernel": None},
      "key": {"kernel": None},
      "value": {"kernel": None},
      "out": {"kernel": None},
  }
  for layer_idx in tqdm(range(base_num_decoder_layers), desc="layers", leave=False):
    wq = np.concatenate(
        [var[f"layers.{layer_idx}.attention.wq.weight"].type(torch.float16).numpy() for var in chkpt_vars], axis=0
    ).transpose()
    wk = np.concatenate(
        [var[f"layers.{layer_idx}.attention.wk.weight"].type(torch.float16).numpy() for var in chkpt_vars], axis=0
    ).transpose()
    wv = np.concatenate(
        [var[f"layers.{layer_idx}.attention.wv.weight"].type(torch.float16).numpy() for var in chkpt_vars], axis=0
    ).transpose()

    wq = np.reshape(wq, [base_num_query_heads * head_dim, base_num_query_heads, head_dim])
    wk = np.reshape(wk, [base_num_query_heads * head_dim, base_num_kv_heads, head_dim])
    wv = np.reshape(wv, [base_num_query_heads * head_dim, base_num_kv_heads, head_dim])
    if model_size[:8] != "llama3.1":
      wq = permute_to_match_maxtext_rope(wq)
      wk = permute_to_match_maxtext_rope(wk)

    w_post = np.concatenate(
        [var[f"layers.{layer_idx}.attention.wo.weight"].type(torch.float16).numpy() for var in chkpt_vars],
        axis=1,
    )

    w_post = np.reshape(w_post, [base_num_query_heads * head_dim, base_num_query_heads, head_dim])

    if self_attention["query"]["kernel"] is None:
      stack_shape = (base_num_decoder_layers,)
      self_attention["query"]["kernel"] = np.zeros(stack_shape + wq.shape, dtype=np.float16)
      self_attention["key"]["kernel"] = np.zeros(stack_shape + wk.shape, dtype=np.float16)
      self_attention["value"]["kernel"] = np.zeros(stack_shape + wv.shape, dtype=np.float16)
      self_attention["out"]["kernel"] = np.zeros(stack_shape + w_post.shape, dtype=np.float16)

    self_attention["query"]["kernel"][layer_idx, ...] = wq  # pylint: disable=E1137
    self_attention["key"]["kernel"][layer_idx, ...] = wk  # pylint: disable=E1137
    self_attention["value"]["kernel"][layer_idx, ...] = wv  # pylint: disable=E1137
    self_attention["out"]["kernel"][layer_idx, ...] = w_post  # pylint: disable=E1137

  self_attention["query"]["kernel"] = np.transpose(self_attention["query"]["kernel"], axes=(1, 0, 2, 3))
  self_attention["key"]["kernel"] = np.transpose(self_attention["key"]["kernel"], axes=(1, 0, 2, 3))
  self_attention["value"]["kernel"] = np.transpose(self_attention["value"]["kernel"], axes=(1, 0, 2, 3))
  # layers, base_num_query_heads * head_dim, base_num_query_heads, head_dim =>
  # base_num_query_heads, layers,head_dim, base_num_query_heads * head_dim
  self_attention["out"]["kernel"] = np.transpose(self_attention["out"]["kernel"], axes=(2, 0, 3, 1))

  # scale the query weights
  self_attention["query"]["kernel"] = self_attention["query"]["kernel"] / np.sqrt(head_dim)

  jax_weights["decoder"]["layers"]["self_attention"] = self_attention
  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  # layer weight pre and post self attention norm ################
  max_logging.log("Processing pre and post self attention norms")
  layer_weight = {"pre_self_attention_layer_norm": {"scale": None}, "post_self_attention_layer_norm": {"scale": None}}

  # self attention layer norm and swap the layer index
  for layer_idx in tqdm(range(base_num_decoder_layers), desc="layers", leave=False):
    pre_self_attention_layernorm = chkpt_vars[0][f"layers.{layer_idx}.attention_norm.weight"].type(torch.float16).numpy()
    post_self_attention_layernorm = chkpt_vars[0][f"layers.{layer_idx}.ffn_norm.weight"].type(torch.float16).numpy()
    if layer_weight["pre_self_attention_layer_norm"]["scale"] is None:
      stack_shape = (base_num_decoder_layers,)
      layer_weight["pre_self_attention_layer_norm"]["scale"] = np.zeros(
          stack_shape + pre_self_attention_layernorm.shape, dtype=np.float16
      )
      layer_weight["post_self_attention_layer_norm"]["scale"] = np.zeros(
          stack_shape + post_self_attention_layernorm.shape, dtype=np.float16
      )
    layer_weight["pre_self_attention_layer_norm"]["scale"][layer_idx, ...] = pre_self_attention_layernorm  # pylint: disable=E1137
    layer_weight["post_self_attention_layer_norm"]["scale"][layer_idx, ...] = post_self_attention_layernorm  # pylint: disable=E1137

  layer_weight["pre_self_attention_layer_norm"]["scale"] = np.transpose(
      layer_weight["pre_self_attention_layer_norm"]["scale"], axes=(1, 0)
  )
  layer_weight["post_self_attention_layer_norm"]["scale"] = np.transpose(
      layer_weight["post_self_attention_layer_norm"]["scale"], axes=(1, 0)
  )

  jax_weights["decoder"]["layers"]["pre_self_attention_layer_norm"] = layer_weight["pre_self_attention_layer_norm"]
  jax_weights["decoder"]["layers"]["post_self_attention_layer_norm"] = layer_weight["post_self_attention_layer_norm"]
  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  # layer weights ################################################
  max_logging.log("Processing layer weights")
  if num_experts is None:
    layer_weight["mlp"] = {
        "wi_0": {"kernel": None},
        "wi_1": {"kernel": None},
        "wo": {"kernel": None},
    }
  else:
    layer_weight["gate"] = {"kernel": None}

    for k in range(num_experts):
      jax_weights["decoder"]["layers"]["MoeBlock_0"]["gate"] = {}
    layer_weight["mlp"] = {
        "wi_0": {"kernel": None},
        "wi_1": {"kernel": None},
        "wo": {"kernel": None},
    }

  for layer_idx in tqdm(range(base_num_decoder_layers), desc="layers", leave=False):
    if num_experts is None:
      wi_0 = np.concatenate(
          [var[f"layers.{layer_idx}.feed_forward.w1.weight"].type(torch.float16).numpy() for var in chkpt_vars], axis=0
      ).transpose()
      wi_1 = np.concatenate(
          [var[f"layers.{layer_idx}.feed_forward.w3.weight"].type(torch.float16).numpy() for var in chkpt_vars], axis=0
      ).transpose()
      wo = np.concatenate(
          [var[f"layers.{layer_idx}.feed_forward.w2.weight"].type(torch.float16).numpy() for var in chkpt_vars], axis=1
      ).transpose()
      if layer_weight["mlp"]["wi_0"]["kernel"] is None:
        stack_shape = (base_num_decoder_layers,)
        layer_weight["mlp"]["wi_0"]["kernel"] = np.zeros(stack_shape + wi_0.shape, dtype=np.float16)
        layer_weight["mlp"]["wi_1"]["kernel"] = np.zeros(stack_shape + wi_1.shape, dtype=np.float16)
        layer_weight["mlp"]["wo"]["kernel"] = np.zeros(stack_shape + wo.shape, dtype=np.float16)
      layer_weight["mlp"]["wi_0"]["kernel"][layer_idx, ...] = wi_0
      layer_weight["mlp"]["wi_1"]["kernel"][layer_idx, ...] = wi_1
      layer_weight["mlp"]["wo"]["kernel"][layer_idx, ...] = wo
    else:
      gate = np.concatenate(
          [var[f"layers.{layer_idx}.feed_forward.gate.weight"].type(torch.float16).numpy() for var in chkpt_vars], axis=0
      ).transpose()
      if layer_weight["gate"]["kernel"] is None:
        stack_shape = (base_num_decoder_layers,)
        layer_weight["gate"]["kernel"] = np.zeros(stack_shape + gate.shape, dtype=np.float16)
      layer_weight["gate"]["kernel"][layer_idx, ...] = gate
      for k in tqdm(range(num_experts), desc="experts", leave=False):
        wi_0 = np.concatenate(
            [
                var[f"layers.{layer_idx}.feed_forward.experts.{k}.w1.weight"].type(torch.float16).numpy()
                for var in chkpt_vars
            ],
            axis=0,
        ).transpose()
        wi_1 = np.concatenate(
            [
                var[f"layers.{layer_idx}.feed_forward.experts.{k}.w3.weight"].type(torch.float16).numpy()
                for var in chkpt_vars
            ],
            axis=0,
        ).transpose()
        wo = np.concatenate(
            [
                var[f"layers.{layer_idx}.feed_forward.experts.{k}.w2.weight"].type(torch.float16).numpy()
                for var in chkpt_vars
            ],
            axis=1,
        ).transpose()
        if layer_weight["mlp"]["wi_0"]["kernel"] is None:
          stack_shape = (num_experts, base_num_decoder_layers)
          layer_weight["mlp"]["wi_0"]["kernel"] = np.zeros(stack_shape + wi_0.shape, dtype=np.float16)
          layer_weight["mlp"]["wi_1"]["kernel"] = np.zeros(stack_shape + wi_1.shape, dtype=np.float16)
          layer_weight["mlp"]["wo"]["kernel"] = np.zeros(stack_shape + wo.shape, dtype=np.float16)
        ei, li = k, layer_idx
        layer_weight["mlp"]["wi_0"]["kernel"][ei, li, ...] = wi_0
        layer_weight["mlp"]["wi_1"]["kernel"][ei, li, ...] = wi_1
        layer_weight["mlp"]["wo"]["kernel"][ei, li, ...] = wo
      gc.collect()
  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  if num_experts is None:
    # swap the layer index
    layer_weight["mlp"]["wi_0"]["kernel"] = np.transpose(layer_weight["mlp"]["wi_0"]["kernel"], axes=(1, 0, 2))
    layer_weight["mlp"]["wi_1"]["kernel"] = np.transpose(layer_weight["mlp"]["wi_1"]["kernel"], axes=(1, 0, 2))
    layer_weight["mlp"]["wo"]["kernel"] = np.transpose(layer_weight["mlp"]["wo"]["kernel"], axes=(1, 0, 2))

    jax_weights["decoder"]["layers"]["mlp"] = layer_weight["mlp"]
  else:
    layer_weight["gate"]["kernel"] = np.transpose(layer_weight["gate"]["kernel"], axes=(1, 0, 2))
    jax_weights["decoder"]["layers"]["MoeBlock_0"]["gate"]["kernel"] = layer_weight["gate"]["kernel"]

    jax_weights["decoder"]["layers"]["MoeBlock_0"]["wi_0"] = layer_weight["mlp"]["wi_0"]["kernel"]
    jax_weights["decoder"]["layers"]["MoeBlock_0"]["wi_1"] = layer_weight["mlp"]["wi_1"]["kernel"]
    jax_weights["decoder"]["layers"]["MoeBlock_0"]["wo"] = layer_weight["mlp"]["wo"]["kernel"]
  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  del chkpt_vars
  gc.collect()
  logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))
  return jax_weights


def save_jax_weights_to_checkpoint(maxtext_model_path, jax_weights):
  """
  Function to save jax_weights ready for MaxText to a parameters checkpoint

  Attributes:
  maxtext_model_path: Path to save the MaxText checkpoint to
  model_size: llama2-7b to 70b, mistral-7b, or mixtral-8x7b, mixtral-8x22b
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
  parser.add_argument("--base-model-path", type=str, required=True)
  parser.add_argument("--maxtext-model-path", type=str, required=True)
  parser.add_argument("--model-size", type=str, required=True)

  args = parser.parse_args()

  if args.model_size not in MODEL_PARAMS_DICT:
    raise NotImplementedError

  os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={SIMULATED_CPU_DEVICES_COUNT}"

  save_jax_weights_to_checkpoint(args.maxtext_model_path, convert_to_jax_weights(args.base_model_path, args.model_size))
