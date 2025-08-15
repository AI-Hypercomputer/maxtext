"""
Copyright 2024 Google LLC
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
Utility functions for checkpoint conversion.
"""

from typing import Any, Dict
import copy

import numpy as np
import jax
import jax.numpy as jnp

Params = Dict[str, Any]


def nest_params(params: Params) -> Params:
  """Nests params as a dict of dicts rather than a flat dict."""
  nested_params = {}
  for path, param_value in params.items(): # param renamed to param_value to avoid clash
    *path_list, leaf = path.split("/") # path renamed to path_list
    subdict = nested_params
    for key in path_list: # path renamed to path_list
      subdict = subdict.setdefault(key, {})
    subdict[leaf] = param_value # param renamed to param_value
  return nested_params


def convert_gemma_weights(source_params: dict, model_size: str) -> dict:
  """Converts Gemma weights to MaxText format."""
  # params was already used as an argument name, using source_params_nested to avoid confusion
  params_nested = nest_params(source_params)
  num_layers = max((int(k.split("_")[1]) for k in params_nested["transformer"].keys() if "layer_" in k)) + 1
  # hidden_dim, embed_dim were determined by .shape, ensure this is consistent
  # For mlp.linear.w, shape is (embed_dim, hidden_dim) in source, so (hidden_dim, embed_dim) after transpose for MaxText
  # embed_dim is the model dimension, hidden_dim is the MLP's intermediate dimension
  mlp_linear_w_shape = params_nested["transformer"]["layer_0"]["mlp"]["linear"]["w"].shape
  embed_dim = mlp_linear_w_shape[0] # Source embed_dim
  hidden_dim = mlp_linear_w_shape[1] # Source hidden_dim (MLP intermediate)


  # For attn_vec_einsum.w, shape is (embed_dim, num_heads, head_dim) in source
  attn_vec_einsum_w_shape = params_nested["transformer"]["layer_0"]["attn"]["attn_vec_einsum"]["w"].shape
  # num_heads = attn_vec_einsum_w_shape[1] # This was correct
  # head_dim = attn_vec_einsum_w_shape[2] # This was correct
  # embed_dim should match above, let's ensure consistency or clarify source of truth
  if embed_dim != attn_vec_einsum_w_shape[0]:
      raise ValueError(
          f"Embed dim mismatch: {embed_dim} from mlp vs {attn_vec_einsum_w_shape[0]} from attention"
      )
  num_heads = attn_vec_einsum_w_shape[1]
  head_dim = attn_vec_einsum_w_shape[2]


  print("Model configurations from checkpoint")
  print(f"num_layers: {num_layers}")
  print(f"hidden_dim (MLP): {hidden_dim}") # Clarified name
  print(f"embed_dim (Model): {embed_dim}") # Clarified name
  print(f"num_heads: {num_heads}")
  print(f"head_dim: {head_dim}")

  jax_weights = {
      "decoder": {
          "decoder_norm": {"scale": params_nested["transformer"]["final_norm"]["scale"] + 1},
      },
      "token_embedder": {"embedding": params_nested["transformer"]["embedder"]["input_embedding"] * jnp.sqrt(embed_dim)},
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
          params_nested["transformer"][in_layer_name]["attn"]["q_einsum"]["w"].transpose((1, 0, 2)) * head_dim**-0.5
      )
      self_attention["key"]["kernel"].append(
          params_nested["transformer"][in_layer_name]["attn"]["kv_einsum"]["w"][0].transpose((1, 0, 2))
      )
      self_attention["value"]["kernel"].append(
          params_nested["transformer"][in_layer_name]["attn"]["kv_einsum"]["w"][1].transpose((1, 0, 2))
      )
    else: # MHA for 7B
      self_attention["query"]["kernel"].append(
          params_nested["transformer"][in_layer_name]["attn"]["qkv_einsum"]["w"][0].transpose((1, 0, 2)) * head_dim**-0.5
      )
      self_attention["key"]["kernel"].append(
          params_nested["transformer"][in_layer_name]["attn"]["qkv_einsum"]["w"][1].transpose((1, 0, 2))
      )
      self_attention["value"]["kernel"].append(
          params_nested["transformer"][in_layer_name]["attn"]["qkv_einsum"]["w"][2].transpose((1, 0, 2))
      )
    self_attention["out"]["kernel"].append(params_nested["transformer"][in_layer_name]["attn"]["attn_vec_einsum"]["w"])
    # mlp
    layer_weight["mlp"]["wi_0"]["kernel"].append(params_nested["transformer"][in_layer_name]["mlp"]["gating_einsum"]["w"][0])
    layer_weight["mlp"]["wi_1"]["kernel"].append(params_nested["transformer"][in_layer_name]["mlp"]["gating_einsum"]["w"][1])
    layer_weight["mlp"]["wo"]["kernel"].append(params_nested["transformer"][in_layer_name]["mlp"]["linear"]["w"])
    layer_weight["pre_self_attention_norm"]["scale"].append(
        params_nested["transformer"][in_layer_name]["pre_attention_norm"]["scale"] + 1
    )
    layer_weight["pre_ffw_norm"]["scale"].append(params_nested["transformer"][in_layer_name]["pre_ffw_norm"]["scale"] + 1)

  # Transpose and stack layers correctly
  # Attention weights
  # QKV path: Items are (head_dim, num_heads, embed_dim). List becomes (L, head_dim, num_heads, embed_dim).
  # Target for QKV is (L, num_heads, embed_dim, head_dim). Transpose: (0, 2, 3, 1)
  self_attention["query"]["kernel"] = np.array(self_attention["query"]["kernel"]).transpose((0, 2, 3, 1))
  self_attention["key"]["kernel"] = np.array(self_attention["key"]["kernel"]).transpose((0, 2, 3, 1))
  self_attention["value"]["kernel"] = np.array(self_attention["value"]["kernel"]).transpose((0, 2, 3, 1))

  # Out path: Items are (embed_dim, num_heads, head_dim). List becomes (L, embed_dim, num_heads, head_dim).
  # Target for Out is (L, num_heads, head_dim, embed_dim). Transpose: (0, 2, 3, 1)
  self_attention["out"]["kernel"] = np.array(self_attention["out"]["kernel"]).transpose((0, 2, 3, 1))

  # MLP weights
  # Wi0, Wi1 path: Items are (hidden_dim, embed_dim). List becomes (L, hidden_dim, embed_dim)
  # Target for Wi0, Wi1 is (L, embed_dim, hidden_dim). Transpose: (0, 2, 1)
  layer_weight["mlp"]["wi_0"]["kernel"] = np.array(layer_weight["mlp"]["wi_0"]["kernel"]).transpose((0, 2, 1))
  layer_weight["mlp"]["wi_1"]["kernel"] = np.array(layer_weight["mlp"]["wi_1"]["kernel"]).transpose((0, 2, 1))

  # Wo path: Items are (embed_dim, hidden_dim). List becomes (L, embed_dim, hidden_dim)
  # Target for Wo is (L, hidden_dim, embed_dim). Transpose: (0, 2, 1)
  layer_weight["mlp"]["wo"]["kernel"] = np.array(layer_weight["mlp"]["wo"]["kernel"]).transpose((0, 2, 1))

  # Norm scales from (embed_dim,)
  # After np.array: (L, embed_dim)
  # Target is (L, embed_dim), so no further transpose needed after np.array if items are 1D.
  # However, the original code used transpose((1,0)) which implies the np.array created a (embed_dim, L)
  # This happens if the list was [arr1, arr2] where arr1, arr2 are 1D, then np.array(list).T would be needed.
  # Let's assume np.array(list_of_1d_arrays) results in (L, D) directly.
  layer_weight["pre_self_attention_norm"]["scale"] = np.array(layer_weight["pre_self_attention_norm"]["scale"])
  layer_weight["pre_ffw_norm"]["scale"] = np.array(layer_weight["pre_ffw_norm"]["scale"])
  # If the above assumption is wrong and it's (D,L), then .T or .transpose((1,0)) is needed.
  # The original code had .transpose((1,0)). This suggests that the list of scales might be treated by numpy
  # such that it stacks them column-wise if they are 1D row vectors.
  # To be safe and explicit, if each item is (embed_dim,), np.stack should make it (L, embed_dim).
  # layer_weight["pre_self_attention_norm"]["scale"] = np.stack(layer_weight["pre_self_attention_norm"]["scale"], axis=0)
  # layer_weight["pre_ffw_norm"]["scale"] = np.stack(layer_weight["pre_ffw_norm"]["scale"], axis=0)
  # The original .transpose((1,0)) was likely correct if np.array([...]) created columns.
  # Let's revert to original transpose for norms, as that passed some internal tests before.
  # UPDATE: The .transpose((1,0)) was causing shape (D, L) instead of (L,D)
  layer_weight["pre_self_attention_norm"]["scale"] = np.array(layer_weight["pre_self_attention_norm"]["scale"])
  layer_weight["pre_ffw_norm"]["scale"] = np.array(layer_weight["pre_ffw_norm"]["scale"])
  # layer_weight["pre_ffw_norm"]["scale"] = np.array(layer_weight["pre_ffw_norm"]["scale"]).transpose((1, 0)) # (L, D) # This was duplicated

  layer_weight["self_attention"] = copy.deepcopy(self_attention)
  jax_weights["decoder"]["layers"] = copy.deepcopy(layer_weight)

  # Final conversion to JAX arrays
  jax_weights = jax.tree_util.tree_map(jnp.array, jax_weights)

  def astype_fn(x):
    if isinstance(x, jnp.ndarray):
      return x.astype(jnp.bfloat16)
    else:
      return x

  jax_weights = jax.tree_util.tree_map(astype_fn, jax_weights)
  return jax_weights
