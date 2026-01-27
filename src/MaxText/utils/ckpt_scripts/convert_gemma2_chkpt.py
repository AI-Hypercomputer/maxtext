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

# pylint: disable=line-too-long
"""
Convert orbax Gemma checkpoint to MaxText compatible checkpoint.
"""

from typing import Any
import argparse
import copy
import sys

import numpy as np

import jax
import jax.numpy as jnp

from flax.training import train_state

import orbax

from MaxText import max_logging
from maxtext.common import checkpointing

jax.config.update("jax_platform_name", "cpu")

Params = dict[str, Any]


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


def main(raw_args=None) -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--base_model_path", type=str, required=True)
  parser.add_argument("--maxtext_model_path", type=str, required=True)
  parser.add_argument("--model_size", type=str, required=True)
  args = parser.parse_args(raw_args)
  if args.model_size not in ("2b", "9b", "27b"):
    raise NotImplementedError("only implemented for gemma 2 classes")

  print("Loading checkpoint")
  checkpointer = orbax.checkpoint.PyTreeCheckpointer()
  params = checkpointer.restore(args.base_model_path)
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

  query_pre_attn_scalar = None
  if args.model_size in ("2b", "9b"):
    query_pre_attn_scalar = head_dim**-0.5
  elif args.model_size in ("27b"):
    query_pre_attn_scalar = (embed_dim // num_heads) ** -0.5

  transpose_gating_einsum = True
  if args.model_size in ("2b"):
    transpose_gating_einsum = False
  elif args.model_size in ("9b", "27b"):
    transpose_gating_einsum = True

  jax_weights = {
      "decoder": {
          "decoder_norm": {"scale": params["transformer"]["final_norm"]["scale"] + 1},
      },
      "token_embedder": {"embedding": params["transformer"]["embedder"]["input_embedding"] * jnp.sqrt(embed_dim)},
  }
  self_attention_local = dict(
      {
          "query": {"kernel": []},
          "key": {"kernel": []},
          "value": {"kernel": []},
          "out": {"kernel": []},
      }
  )
  self_attention_global = dict(
      {
          "query": {"kernel": []},
          "key": {"kernel": []},
          "value": {"kernel": []},
          "out": {"kernel": []},
      }
  )

  layer_weight = dict(
      {
          "mlp_local": {
              "wi_0": {"kernel": []},
              "wi_1": {"kernel": []},
              "wo": {"kernel": []},
          },
          "mlp_global": {
              "wi_0": {"kernel": []},
              "wi_1": {"kernel": []},
              "wo": {"kernel": []},
          },
          "pre_self_attention_norm_local": {"scale": []},
          "pre_ffw_norm_local": {"scale": []},
          "post_self_attention_norm_local": {"scale": []},
          "post_ffw_norm_local": {"scale": []},
          "pre_self_attention_norm_global": {"scale": []},
          "pre_ffw_norm_global": {"scale": []},
          "post_self_attention_norm_global": {"scale": []},
          "post_ffw_norm_global": {"scale": []},
      }
  )

  for layer_idx in range(0, num_layers, 2):
    in_layer_name_local = "layer_" + str(layer_idx)
    in_layer_name_global = "layer_" + str(layer_idx + 1)

    ######################## layer local attention ########################
    self_attention_local["query"]["kernel"].append(
        params["transformer"][in_layer_name_local]["attn"]["q_einsum"]["w"].transpose((1, 0, 2)) * query_pre_attn_scalar
    )
    self_attention_local["key"]["kernel"].append(
        params["transformer"][in_layer_name_local]["attn"]["kv_einsum"]["w"][0].transpose((1, 0, 2))
    )
    self_attention_local["value"]["kernel"].append(
        params["transformer"][in_layer_name_local]["attn"]["kv_einsum"]["w"][1].transpose((1, 0, 2))
    )
    self_attention_local["out"]["kernel"].append(
        params["transformer"][in_layer_name_local]["attn"]["attn_vec_einsum"]["w"]
    )

    # mlp
    if transpose_gating_einsum:
      layer_weight["mlp_local"]["wi_0"]["kernel"].append(
          np.transpose(params["transformer"][in_layer_name_local]["mlp"]["gating_einsum"]["w"][0])
      )
      layer_weight["mlp_local"]["wi_1"]["kernel"].append(
          np.transpose(params["transformer"][in_layer_name_local]["mlp"]["gating_einsum"]["w"][1])
      )
    else:
      layer_weight["mlp_local"]["wi_0"]["kernel"].append(
          params["transformer"][in_layer_name_local]["mlp"]["gating_einsum"]["w"][0]
      )
      layer_weight["mlp_local"]["wi_1"]["kernel"].append(
          params["transformer"][in_layer_name_local]["mlp"]["gating_einsum"]["w"][1]
      )

    layer_weight["mlp_local"]["wo"]["kernel"].append(params["transformer"][in_layer_name_local]["mlp"]["linear"]["w"])

    layer_weight["pre_self_attention_norm_local"]["scale"].append(
        params["transformer"][in_layer_name_local]["pre_attention_norm"]["scale"] + 1
    )
    layer_weight["pre_ffw_norm_local"]["scale"].append(
        params["transformer"][in_layer_name_local]["pre_ffw_norm"]["scale"] + 1
    )

    layer_weight["post_self_attention_norm_local"]["scale"].append(
        params["transformer"][in_layer_name_local]["post_attention_norm"]["scale"] + 1
    )
    layer_weight["post_ffw_norm_local"]["scale"].append(
        params["transformer"][in_layer_name_local]["post_ffw_norm"]["scale"] + 1
    )

    ######################## layer global attention ########################

    self_attention_global["query"]["kernel"].append(
        params["transformer"][in_layer_name_global]["attn"]["q_einsum"]["w"].transpose((1, 0, 2)) * query_pre_attn_scalar
    )
    self_attention_global["key"]["kernel"].append(
        params["transformer"][in_layer_name_global]["attn"]["kv_einsum"]["w"][0].transpose((1, 0, 2))
    )
    self_attention_global["value"]["kernel"].append(
        params["transformer"][in_layer_name_global]["attn"]["kv_einsum"]["w"][1].transpose((1, 0, 2))
    )
    self_attention_global["out"]["kernel"].append(
        params["transformer"][in_layer_name_global]["attn"]["attn_vec_einsum"]["w"]
    )

    # mlp
    if transpose_gating_einsum:
      layer_weight["mlp_global"]["wi_0"]["kernel"].append(
          np.transpose(params["transformer"][in_layer_name_global]["mlp"]["gating_einsum"]["w"][0])
      )
      layer_weight["mlp_global"]["wi_1"]["kernel"].append(
          np.transpose(params["transformer"][in_layer_name_global]["mlp"]["gating_einsum"]["w"][1])
      )
    else:
      layer_weight["mlp_global"]["wi_0"]["kernel"].append(
          params["transformer"][in_layer_name_global]["mlp"]["gating_einsum"]["w"][0]
      )
      layer_weight["mlp_global"]["wi_1"]["kernel"].append(
          params["transformer"][in_layer_name_global]["mlp"]["gating_einsum"]["w"][1]
      )

    layer_weight["mlp_global"]["wo"]["kernel"].append(params["transformer"][in_layer_name_global]["mlp"]["linear"]["w"])

    layer_weight["pre_self_attention_norm_global"]["scale"].append(
        params["transformer"][in_layer_name_global]["pre_attention_norm"]["scale"] + 1
    )
    layer_weight["pre_ffw_norm_global"]["scale"].append(
        params["transformer"][in_layer_name_global]["pre_ffw_norm"]["scale"] + 1
    )

    layer_weight["post_self_attention_norm_global"]["scale"].append(
        params["transformer"][in_layer_name_global]["post_attention_norm"]["scale"] + 1
    )
    layer_weight["post_ffw_norm_global"]["scale"].append(
        params["transformer"][in_layer_name_global]["post_ffw_norm"]["scale"] + 1
    )

  self_attention_local["query"]["kernel"] = np.array(self_attention_local["query"]["kernel"]).transpose((1, 0, 2, 3))
  self_attention_local["key"]["kernel"] = np.array(self_attention_local["key"]["kernel"]).transpose((1, 0, 2, 3))
  self_attention_local["value"]["kernel"] = np.array(self_attention_local["value"]["kernel"]).transpose((1, 0, 2, 3))
  self_attention_local["out"]["kernel"] = np.array(self_attention_local["out"]["kernel"]).transpose((1, 0, 2, 3))

  self_attention_global["query"]["kernel"] = np.array(self_attention_global["query"]["kernel"]).transpose((1, 0, 2, 3))
  self_attention_global["key"]["kernel"] = np.array(self_attention_global["key"]["kernel"]).transpose((1, 0, 2, 3))
  self_attention_global["value"]["kernel"] = np.array(self_attention_global["value"]["kernel"]).transpose((1, 0, 2, 3))
  self_attention_global["out"]["kernel"] = np.array(self_attention_global["out"]["kernel"]).transpose((1, 0, 2, 3))

  layer_weight["mlp_local"]["wi_0"]["kernel"] = np.array(layer_weight["mlp_local"]["wi_0"]["kernel"]).transpose((1, 0, 2))
  layer_weight["mlp_local"]["wi_1"]["kernel"] = np.array(layer_weight["mlp_local"]["wi_1"]["kernel"]).transpose((1, 0, 2))
  layer_weight["mlp_local"]["wo"]["kernel"] = np.array(layer_weight["mlp_local"]["wo"]["kernel"]).transpose((1, 0, 2))

  layer_weight["mlp_global"]["wi_0"]["kernel"] = np.array(layer_weight["mlp_global"]["wi_0"]["kernel"]).transpose(
      (1, 0, 2)
  )
  layer_weight["mlp_global"]["wi_1"]["kernel"] = np.array(layer_weight["mlp_global"]["wi_1"]["kernel"]).transpose(
      (1, 0, 2)
  )
  layer_weight["mlp_global"]["wo"]["kernel"] = np.array(layer_weight["mlp_global"]["wo"]["kernel"]).transpose((1, 0, 2))

  layer_weight["pre_self_attention_norm_local"]["scale"] = np.array(
      layer_weight["pre_self_attention_norm_local"]["scale"]
  ).transpose((1, 0))
  layer_weight["pre_ffw_norm_local"]["scale"] = np.array(layer_weight["pre_ffw_norm_local"]["scale"]).transpose((1, 0))
  layer_weight["post_self_attention_norm_local"]["scale"] = np.array(
      layer_weight["post_self_attention_norm_local"]["scale"]
  ).transpose((1, 0))
  layer_weight["post_ffw_norm_local"]["scale"] = np.array(layer_weight["post_ffw_norm_local"]["scale"]).transpose((1, 0))

  layer_weight["pre_self_attention_norm_global"]["scale"] = np.array(
      layer_weight["pre_self_attention_norm_global"]["scale"]
  ).transpose((1, 0))
  layer_weight["pre_ffw_norm_global"]["scale"] = np.array(layer_weight["pre_ffw_norm_global"]["scale"]).transpose((1, 0))
  layer_weight["post_self_attention_norm_global"]["scale"] = np.array(
      layer_weight["post_self_attention_norm_global"]["scale"]
  ).transpose((1, 0))
  layer_weight["post_ffw_norm_global"]["scale"] = np.array(layer_weight["post_ffw_norm_global"]["scale"]).transpose(
      (1, 0)
  )

  layer_weight["self_attention_local"] = copy.deepcopy(self_attention_local)
  layer_weight["self_attention_global"] = copy.deepcopy(self_attention_global)

  jax_weights["decoder"]["layers"] = copy.deepcopy(layer_weight)
  jax_weights = jax.tree_util.tree_map(jnp.array, jax_weights)

  def astype_fn(x):
    if isinstance(x, jnp.ndarray):
      return x.astype(jnp.bfloat16)
    else:
      return x

  jax_weights = jax.tree_util.tree_map(astype_fn, jax_weights)

  enable_checkpointing = True
  async_checkpointing = False
  save_interval_steps = 1

  checkpoint_manager = checkpointing.create_orbax_checkpoint_manager(
      args.maxtext_model_path, enable_checkpointing, async_checkpointing, save_interval_steps
  )

  state_new = train_state.TrainState(
      step=0, apply_fn=None, params={"params": jax_weights}, tx=None, opt_state={}  # type: ignore
  )

  if checkpoint_manager is not None:
    if checkpointing.save_checkpoint(checkpoint_manager, 0, state_new):
      max_logging.log("saved a checkpoint at step 0")
    # Upon preemption, exit when and only when all ongoing saves are complete.
    if checkpoint_manager.reached_preemption(0):
      checkpoint_manager.wait_until_finished()
      sys.exit()


if __name__ == "__main__":
  main()
