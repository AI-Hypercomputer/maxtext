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

from typing import Any
import argparse
import copy
import sys

import numpy as np

import jax
import jax.numpy as jnp

from flax.training import train_state

import orbax

from MaxText import checkpointing
from MaxText import max_logging

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


def rename_nested_keys(data, old_key, new_key):
  """
  Recursively renames keys in a nested dictionary.
  Args:
      data (dict): The nested dictionary to process.
      old_key (str): The key to find and rename.
      new_key (str): The new name for the key.
  Returns:
      dict: A new dictionary with the specified keys renamed.
  """
  new_data = {}
  for key, value in data.items():
    new_k = new_key if key == old_key else key
    if isinstance(value, dict):
      new_data[new_k] = rename_nested_keys(value, old_key, new_key)
    elif isinstance(value, list):
      new_data[new_k] = [rename_nested_keys(item, old_key, new_key) if isinstance(item, dict) else item for item in value]
    else:
      new_data[new_k] = value
  return new_data


def main(raw_args=None) -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--base_model_path", type=str, required=True)
  parser.add_argument("--maxtext_model_path", type=str, required=True)
  parser.add_argument("--model_size", type=str, required=True)
  args = parser.parse_args(raw_args)
  if args.model_size not in ("4b", "12b", "27b"):
    raise NotImplementedError("only implemented for gemma 3 classes")

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

  # All gemma 3 models have transpose_gating_einsum == True
  transpose_gating_einsum = True

  jax_weights = {
      "decoder": {
          "decoder_norm": {"scale": params["transformer"]["final_norm"]["scale"] + 1},
      },
      "token_embedder": {"embedding": params["transformer"]["embedder"]["input_embedding"] * jnp.sqrt(embed_dim)},
      "vision_encoder": {
          "Gemma3VisionEncoderLayer_0": {
              "embedding": {
                  "bias": params["SigLiPFromPatches_0"]["siglip_encoder"]["embedding"]["bias"],
                  "kernel": params["SigLiPFromPatches_0"]["siglip_encoder"]["embedding"]["kernel"],
              },
              "pos_embedding": params["SigLiPFromPatches_0"]["siglip_encoder"]["pos_embedding"],
              "Transformer": params["SigLiPFromPatches_0"]["siglip_encoder"]["Transformer"],
          },
          "VisionEmbedder_0": {
              "mm_input_projection": params["transformer"]["embedder"]["mm_input_projection"],
              "mm_soft_embedding_norm": {
                  "scale": params["transformer"]["embedder"]["mm_soft_embedding_norm"]["scale"] + 1
              },
          },
      },
  }
  # Rename MlpBlock_0 to MlpBlockViT_0 in vision encoder
  # This is because the gemma3 model has MlpBlock in the vision encoder,
  # which has the same name as the MlpBlock in the MaxText decoder but different structure.
  # Hence, we need to rename it to avoid confusion.
  vision_encoder_weights = rename_nested_keys(jax_weights["vision_encoder"], "MlpBlock_0", "MlpBlockViT_0")
  jax_weights["vision_encoder"] = vision_encoder_weights
  self_attention = dict(
      {
          "query": {"kernel": []},
          "key": {"kernel": []},
          "value": {"kernel": []},
          "out": {"kernel": []},
          "query_norm": {"scale": []},
          "key_norm": {"scale": []},
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
          "post_self_attention_norm": {"scale": []},
          "post_ffw_norm": {"scale": []},
      }
  )

  for layer_idx in range(0, num_layers):
    in_layer_name = "layer_" + str(layer_idx)

    self_attention["query"]["kernel"].append(
        params["transformer"][in_layer_name]["attn"]["q_einsum"]["w"].transpose((1, 0, 2))  # * query_pre_attn_scalar
    )
    self_attention["key"]["kernel"].append(
        params["transformer"][in_layer_name]["attn"]["kv_einsum"]["w"][0].transpose((1, 0, 2))
    )
    self_attention["value"]["kernel"].append(
        params["transformer"][in_layer_name]["attn"]["kv_einsum"]["w"][1].transpose((1, 0, 2))
    )
    self_attention["out"]["kernel"].append(params["transformer"][in_layer_name]["attn"]["attn_vec_einsum"]["w"])

    self_attention["key_norm"]["scale"].append(params["transformer"][in_layer_name]["attn"]["_key_norm"]["scale"] + 1)
    self_attention["query_norm"]["scale"].append(params["transformer"][in_layer_name]["attn"]["_query_norm"]["scale"] + 1)

    # mlp
    if transpose_gating_einsum:
      layer_weight["mlp"]["wi_0"]["kernel"].append(
          np.transpose(params["transformer"][in_layer_name]["mlp"]["gating_einsum"]["w"][0])
      )
      layer_weight["mlp"]["wi_1"]["kernel"].append(
          np.transpose(params["transformer"][in_layer_name]["mlp"]["gating_einsum"]["w"][1])
      )
    else:
      layer_weight["mlp"]["wi_0"]["kernel"].append(params["transformer"][in_layer_name]["mlp"]["gating_einsum"]["w"][0])
      layer_weight["mlp"]["wi_1"]["kernel"].append(params["transformer"][in_layer_name]["mlp"]["gating_einsum"]["w"][1])

    layer_weight["mlp"]["wo"]["kernel"].append(params["transformer"][in_layer_name]["mlp"]["linear"]["w"])

    layer_weight["pre_self_attention_norm"]["scale"].append(
        params["transformer"][in_layer_name]["pre_attention_norm"]["scale"] + 1
    )
    layer_weight["pre_ffw_norm"]["scale"].append(params["transformer"][in_layer_name]["pre_ffw_norm"]["scale"] + 1)

    layer_weight["post_self_attention_norm"]["scale"].append(
        params["transformer"][in_layer_name]["post_attention_norm"]["scale"] + 1
    )
    layer_weight["post_ffw_norm"]["scale"].append(params["transformer"][in_layer_name]["post_ffw_norm"]["scale"] + 1)

  self_attention["query"]["kernel"] = np.array(self_attention["query"]["kernel"]).transpose((1, 0, 2, 3))
  self_attention["key"]["kernel"] = np.array(self_attention["key"]["kernel"]).transpose((1, 0, 2, 3))
  self_attention["value"]["kernel"] = np.array(self_attention["value"]["kernel"]).transpose((1, 0, 2, 3))
  self_attention["out"]["kernel"] = np.array(self_attention["out"]["kernel"]).transpose((1, 0, 2, 3))
  self_attention["key_norm"]["scale"] = np.array(self_attention["key_norm"]["scale"]).transpose((1, 0))
  self_attention["query_norm"]["scale"] = np.array(self_attention["query_norm"]["scale"]).transpose((1, 0))

  layer_weight["mlp"]["wi_0"]["kernel"] = np.array(layer_weight["mlp"]["wi_0"]["kernel"]).transpose((1, 0, 2))
  layer_weight["mlp"]["wi_1"]["kernel"] = np.array(layer_weight["mlp"]["wi_1"]["kernel"]).transpose((1, 0, 2))
  layer_weight["mlp"]["wo"]["kernel"] = np.array(layer_weight["mlp"]["wo"]["kernel"]).transpose((1, 0, 2))

  layer_weight["pre_self_attention_norm"]["scale"] = np.array(layer_weight["pre_self_attention_norm"]["scale"]).transpose(
      (1, 0)
  )
  layer_weight["pre_ffw_norm"]["scale"] = np.array(layer_weight["pre_ffw_norm"]["scale"]).transpose((1, 0))
  layer_weight["post_self_attention_norm"]["scale"] = np.array(
      layer_weight["post_self_attention_norm"]["scale"]
  ).transpose((1, 0))
  layer_weight["post_ffw_norm"]["scale"] = np.array(layer_weight["post_ffw_norm"]["scale"]).transpose((1, 0))

  layer_weight["self_attention"] = copy.deepcopy(self_attention)

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
      max_logging.log(f"Checkpoint saved to: {args.maxtext_model_path}")
    # Upon preemption, exit when and only when all ongoing saves are complete.
    if checkpoint_manager.reached_preemption(0):
      checkpoint_manager.wait_until_finished()
      sys.exit()


if __name__ == "__main__":
  main()
