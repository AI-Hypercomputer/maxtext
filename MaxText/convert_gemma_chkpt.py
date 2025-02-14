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

# pylint: disable=line-too-long
"""
Convert orbax Gemma checkpoint to MaxText compatible checkpoint.
"""

import jax
import jax.numpy as jnp
import numpy as np
import psutil
import gc

jax.config.update("jax_platform_name", "cpu")
import argparse
import copy
from flax.training import train_state

from typing import Any
import logging
import max_logging


import orbax

import checkpointing
from train import save_checkpoint

Params = dict[str, Any]

SIMULATED_CPU_DEVICES_COUNT = 16


def nest_params(params: Params) -> Params:
  """Nests params as a dict of dicts rather than a flat dict."""
  nested_params = {}
  for path, param in params.items():
    *path, leaf = path.split("/")
    subdict = nested_params
    for key in path:
      subdict = subdict.setdefault(key, {})
    subdict[leaf] = param
  return nested_params


def convert_to_jax_weights(base_model_path, model_size):
  """Convert to MaxText compatible orbax weights."""
  print("Loading checkpoint")
  checkpointer = orbax.checkpoint.PyTreeCheckpointer()
  params = checkpointer.restore(base_model_path)
  params = nest_params(params)
  num_layers = max((int(k.split("_")[1]) for k in params["transformer"].keys() if "layer_" in k)) + 1
  hidden_dim, embed_dim = params["transformer"]["layer_0"]["mlp"]["linear"]["w"].shape
  num_heads, head_dim, _ = params["transformer"]["layer_0"]["attn"]["attn_vec_einsum"]["w"].shape
  print("Model configurations from checkpoint")
  print(f"num_layers: {num_layers}")
  print(f"hidden_dim: {hidden_dim}")
  print(f"embed_dim: {embed_dim}")
  print(f"num_heads: {num_heads}")
  print(f"head_dim: {head_dim}")

  jax_weights = {
      "decoder": {
          "decoder_norm": {"scale": params["transformer"]["final_norm"]["scale"] + 1},
      },
      "token_embedder": {"embedding": params["transformer"]["embedder"]["input_embedding"] * jnp.sqrt(embed_dim)},
  }
  self_attention = dict(
      {
          "query": {"kernel": []},
          "key": {"kernel": []},
          "value": {"kernel": []},
          "out": {"kernel": []},
      }
  )

  layer_weight = dict(
      {
          "mlp": {
              "wi_0": {"kernel": []},
              "wi_1": {"kernel": []},
              "wo": {"kernel": []},
          },
          "pre_self_attention_norm": {"scale": []},
          "pre_ffw_norm": {"scale": []},
      }
  )

  for layer_idx in range(num_layers):
    in_layer_name = "layer_" + str(layer_idx)
    # attention block
    if model_size in ("2b", "9b"):  # MQA
      self_attention["query"]["kernel"].append(
          params["transformer"][in_layer_name]["attn"]["q_einsum"]["w"].transpose((1, 0, 2)) * head_dim**-0.5
      )
      self_attention["key"]["kernel"].append(
          params["transformer"][in_layer_name]["attn"]["kv_einsum"]["w"][0].transpose((1, 0, 2))
      )
      self_attention["value"]["kernel"].append(
          params["transformer"][in_layer_name]["attn"]["kv_einsum"]["w"][1].transpose((1, 0, 2))
      )
    else:
      self_attention["query"]["kernel"].append(
          params["transformer"][in_layer_name]["attn"]["qkv_einsum"]["w"][0].transpose((1, 0, 2)) * head_dim**-0.5
      )
      self_attention["key"]["kernel"].append(
          params["transformer"][in_layer_name]["attn"]["qkv_einsum"]["w"][1].transpose((1, 0, 2))
      )
      self_attention["value"]["kernel"].append(
          params["transformer"][in_layer_name]["attn"]["qkv_einsum"]["w"][2].transpose((1, 0, 2))
      )
    self_attention["out"]["kernel"].append(params["transformer"][in_layer_name]["attn"]["attn_vec_einsum"]["w"])
    # mlp
    layer_weight["mlp"]["wi_0"]["kernel"].append(params["transformer"][in_layer_name]["mlp"]["gating_einsum"]["w"][0])
    layer_weight["mlp"]["wi_1"]["kernel"].append(params["transformer"][in_layer_name]["mlp"]["gating_einsum"]["w"][1])
    layer_weight["mlp"]["wo"]["kernel"].append(params["transformer"][in_layer_name]["mlp"]["linear"]["w"])
    layer_weight["pre_self_attention_norm"]["scale"].append(
        params["transformer"][in_layer_name]["pre_attention_norm"]["scale"] + 1
    )
    layer_weight["pre_ffw_norm"]["scale"].append(params["transformer"][in_layer_name]["pre_ffw_norm"]["scale"] + 1)

  self_attention["query"]["kernel"] = np.array(self_attention["query"]["kernel"]).transpose((1, 0, 2, 3))
  self_attention["key"]["kernel"] = np.array(self_attention["key"]["kernel"]).transpose((1, 0, 2, 3))
  self_attention["value"]["kernel"] = np.array(self_attention["value"]["kernel"]).transpose((1, 0, 2, 3))
  self_attention["out"]["kernel"] = np.array(self_attention["out"]["kernel"]).transpose((1, 0, 2, 3))

  layer_weight["mlp"]["wi_0"]["kernel"] = np.array(layer_weight["mlp"]["wi_0"]["kernel"]).transpose((1, 0, 2))
  layer_weight["mlp"]["wi_1"]["kernel"] = np.array(layer_weight["mlp"]["wi_1"]["kernel"]).transpose((1, 0, 2))
  layer_weight["mlp"]["wo"]["kernel"] = np.array(layer_weight["mlp"]["wo"]["kernel"]).transpose((1, 0, 2))
  layer_weight["pre_self_attention_norm"]["scale"] = np.array(layer_weight["pre_self_attention_norm"]["scale"]).transpose(
      (1, 0)
  )
  layer_weight["pre_ffw_norm"]["scale"] = np.array(layer_weight["pre_ffw_norm"]["scale"]).transpose((1, 0))

  layer_weight["self_attention"] = copy.deepcopy(self_attention)
  jax_weights["decoder"]["layers"] = copy.deepcopy(layer_weight)

  return jax_weights


def save_jax_weights_to_checkpoint(maxtext_model_path, jax_weights):
  """
  Function to save jax_weights ready for MaxText to a parameters checkpoint
  """
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
  jax_weights_flat, jax_weights_struct = jax.tree.flatten(jax_weights)
  jax_weights_new = []
  while len(jax_weights_flat) > 0:
    jax_weight = jax_weights_flat.pop(0)
    jax_weights_new.append(checkpoint_device_put(jax_weight))
    del jax_weight
    gc.collect()
    logging.debug("Memory usage: %f GB", mem_info.memory_info().rss / (1024**3))

  jax_weights = jax.tree.unflatten(jax_weights_struct, jax_weights_new)

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


def main(raw_args=None) -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--base_model_path", type=str, required=True)
  parser.add_argument("--maxtext_model_path", type=str, required=True)
  parser.add_argument("--model_size", type=str, required=True)
  args = parser.parse_args(raw_args)
  if args.model_size not in ("2b", "7b", "9b"):
    raise NotImplementedError

  save_jax_weights_to_checkpoint(args.maxtext_model_path, convert_to_jax_weights(args.base_model_path, args.model_size))


if __name__ == "__main__":
  main()
