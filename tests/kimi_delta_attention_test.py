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

"""Test for Kimi Delta Attention with real weights."""

import os
import gc
import re
import jax
from enum import Enum, auto
import jax.numpy as jnp
from etils import epath
from flax import nnx
from safetensors.flax import load_file as load_safetensors
from huggingface_hub import hf_hub_download

from MaxText.layers.kimi_delta_attention import KimiDeltaAttention


def _get_key_and_transform_mapping():
    """Define mapping from HuggingFace UMT5 keys to JAX UMT5 keys."""

    class Transform(Enum):
        """Transformations for UMT5 parameters"""

        NONE = None
        # For linear layers: (out, in) -> (in, out)
        TRANSPOSE = ((1, 0), None, False)
        # For Conv
        CONV_TRANSPOSE = ((2, 1, 0), None, False)

    # T5/UMT5 uses standard HuggingFace naming
    """
     "A_log": "A_log",
      "dt_bias": "dt_bias",
      "b_proj.weight": "b_proj.kernel",
      "f_a_proj.weight": "f_a_proj.kernel",
      "f_b_proj.weight": "f_b_proj.kernel",
      "g_a_proj.weight": "g_a_proj.kernel",
      "g_b_proj.weight": "g_b_proj.kernel",
      "k_conv1d.weight": "k_conv1d.kernel",
      "q_conv1d.weight": "q_conv1d.kernel",
      "v_conv1d.weight": "v_conv1d.kernel",
      "k_proj.weight": "k_proj.kernel",
      "q_proj.weight": "q_proj.kernel",
      "v_proj.weight": "v_proj.kernel",
      "o_proj.weight": "o_proj.kernel",
      "o_norm.weight": "o_norm.rms_norm.scale", 
    """
    mapping = {
        r"model\.layers\.0\.self_attn\.A_log": (r"A_log", Transform.NONE),
        r"model\.layers\.0\.self_attn\.dt_bias": (r"dt_bias", Transform.NONE),
        r"model\.layers\.0\.self_attn\.b_proj\.weight": (r"b_proj.kernel", Transform.TRANSPOSE),
        r"model\.layers\.0\.self_attn\.f_a_proj\.weight": (r"f_a_proj.kernel", Transform.TRANSPOSE),
        r"model\.layers\.0\.self_attn\.f_b_proj\.weight": (r"f_b_proj.kernel", Transform.TRANSPOSE),
        r"model\.layers\.0\.self_attn\.g_a_proj\.weight": (r"g_a_proj.kernel", Transform.TRANSPOSE),
        r"model\.layers\.0\.self_attn\.g_b_proj\.weight": (r"g_b_proj.kernel", Transform.TRANSPOSE),
        r"model\.layers\.0\.self_attn\.k_conv1d\.weight": (r"k_conv1d.kernel", Transform.CONV_TRANSPOSE),
        r"model\.layers\.0\.self_attn\.q_conv1d\.weight": (r"q_conv1d.kernel", Transform.CONV_TRANSPOSE),
        r"model\.layers\.0\.self_attn\.v_conv1d\.weight": (r"v_conv1d.kernel", Transform.CONV_TRANSPOSE),
        r"model\.layers\.0\.self_attn\.k_proj\.weight": (r"k_proj.kernel", Transform.TRANSPOSE),
        r"model\.layers\.0\.self_attn\.q_proj\.weight": (r"q_proj.kernel", Transform.TRANSPOSE),
        r"model\.layers\.0\.self_attn\.v_proj\.weight": (r"v_proj.kernel", Transform.TRANSPOSE),
        r"model\.layers\.0\.self_attn\.o_proj\.weight": (r"o_proj.kernel", Transform.TRANSPOSE),
        r"model\.layers\.0\.self_attn\.o_norm\.weight": (r"o_norm.rms_norm.scale", Transform.NONE),
    }

    return mapping


def _torch_key_to_jax_key(mapping, source_key):
    subs = [
        (re.sub(pat, repl, source_key), reshape)
        for pat, (repl, reshape) in mapping.items()
        if re.match(pat, source_key)
    ]
    if len(subs) > 1:
        raise ValueError(f"Only one key should be found: {subs[0]}")
    if len(subs) == 0:
        return (None, None)
    return subs[0]


def _assign_weights(keys, tensor, state_dict, st_key, transform, sharding_dict):
    """Recursively descend into state_dict and assign the (possibly permuted/reshaped) tensor."""
    key, *rest = keys
    if not rest:
        if transform is not None:
            permute, reshape, reshape_first = transform
            if reshape_first and reshape is not None:
                tensor = tensor.reshape(reshape)
            if permute:
                tensor = tensor.transpose(permute)
            if not reshape_first and reshape is not None:
                tensor = tensor.reshape(reshape)
        if tensor.shape != state_dict[key].shape:
            raise ValueError(
                f"Shape mismatch for {st_key}: {tensor.shape} vs {state_dict[key].shape}")
        # Only apply sharding if sharding_dict is provided
        if sharding_dict is not None:
            state_dict[key] = jax.device_put(
                tensor, sharding_dict[key])
        else:
            state_dict[key] = jax.device_put(tensor)
    else:
        next_sharding = sharding_dict[key] if sharding_dict is not None else None
        _assign_weights(
            rest, tensor, state_dict[key], st_key, transform, next_sharding)


def _stoi(s):
    try:
        return int(s)
    except ValueError:
        return s


def create_model(
    file_dir: str,
    hidden_size=2304,
    num_heads=32,
    head_dim=128,
    key_mapping=None,
    param_dtype: jnp.dtype | None = jnp.bfloat16,
    mesh: jax.sharding.Mesh | None = None,
) -> KimiDeltaAttention:
    model = nnx.eval_shape(lambda: KimiDeltaAttention(
        hidden_size, num_heads, head_dim, weight_dtype=param_dtype, rngs=nnx.Rngs(params=0, dropout=0)))
    graph_def, abs_state = nnx.split(model)
    state_dict = abs_state.to_pure_dict()
    # Only use sharding if mesh is provided
    sharding = nnx.get_named_sharding(
        abs_state, mesh).to_pure_dict() if mesh is not None else None

    if not key_mapping:
        key_mapping = _get_key_and_transform_mapping()
    conversion_errors = []

    print(f"Loading Weight...")
    sf = load_safetensors(file_dir)
    for weight_key, weight_value in sf.items():
        jax_key, transform = _torch_key_to_jax_key(key_mapping, weight_key)
        if not jax_key:
          continue
        print(f"Load {weight_key}... {weight_value.shape=}")
        keys = [_stoi(k) for k in jax_key.split(".")]
        try:
            _assign_weights(keys, weight_value, state_dict, weight_key, transform.value, sharding)
        except Exception as e:
            full_jax_key = ".".join([str(k) for k in keys])
            conversion_errors.append(
                f"Failed to assign '{weight_key}' to '{full_jax_key}': {type(e).__name__}: {e}")
    gc.collect()

    if conversion_errors:
        full_error_log = "\n".join(conversion_errors)
        raise RuntimeError(
            f"Encountered {len(conversion_errors)} weight conversion errors. Log:\n{full_error_log}")

    gc.collect()
    m = nnx.merge(graph_def, state_dict)
    m.eval()
    return m

def download_kimi_weights(repo_id="moonshotai/Kimi-Linear-48B-A3B-Base", filename="model-00001-of-00020.safetensors"):
  """Downloads weights from Hugging Face."""
  print(f"Downloading {filename} from {repo_id}...")
  path = hf_hub_download(repo_id=repo_id, filename=filename)
  return path

def map_weights_to_nnx(flat_params, layer_idx=1):
  """Maps Torch-style safetensors keys to NNX parameter structure.
  Note: layer_idx=1 because the config shows kda_layers starting from 1.
  """
  prefix = f"model.layers.{layer_idx}.self_attn."
  mapped_state = {}
  
  # Define mapping rules
  # Key in safetensors -> Path in NNX Module
  mapping = {
      "A_log": "A_log",
      "dt_bias": "dt_bias",
      "b_proj.weight": "b_proj.kernel",
      "f_a_proj.weight": "f_a_proj.kernel",
      "f_b_proj.weight": "f_b_proj.kernel",
      "g_a_proj.weight": "g_a_proj.kernel",
      "g_b_proj.weight": "g_b_proj.kernel",
      "k_conv1d.weight": "k_conv1d.kernel",
      "q_conv1d.weight": "q_conv1d.kernel",
      "v_conv1d.weight": "v_conv1d.kernel",
      "k_proj.weight": "k_proj.kernel",
      "q_proj.weight": "q_proj.kernel",
      "v_proj.weight": "v_proj.kernel",
      "o_proj.weight": "o_proj.kernel",
      "o_norm.weight": "o_norm.rms_norm.scale",  # Key check might be needed
  }

  for torch_key, nnx_path in mapping.items():
    full_torch_key = prefix + torch_key
    if full_torch_key in flat_params:
      val = flat_params[full_torch_key]
      
      # Handle Transpose for Linear layers (Torch [Out, In] -> JAX [In, Out])
      if "proj.weight" in torch_key:
        val = val.T
      
      # Handle Conv1D (Torch [Out, 1, K] -> JAX [K, 1, Out])
      if "conv1d.weight" in torch_key:
        val = jnp.transpose(val, (2, 1, 0))
      
      mapped_state[nnx_path] = val
    else:
      # Try alternative key for o_norm.weight if mapping fails
      if torch_key == "o_norm.rms_norm.scale.value":
         alt_key = prefix + "o_norm.weight"
         if alt_key in flat_params:
            mapped_state[nnx_path] = flat_params[alt_key]
            continue
      print(f"Warning: Key {full_torch_key} not found in safetensors.")

  return mapped_state

def test_kda_with_real_weights():
  print("Initializing KDA layer...")
  repo_id = "moonshotai/Kimi-Linear-48B-A3B-Base"
  file_name = "model-00001-of-00020.safetensors"
  # 1. download weight
  weight_path = download_kimi_weights(repo_id=repo_id, filename=file_name)
  hidden_size = 2304
  # 2. create model
  kda = create_model(weight_path, hidden_size=hidden_size)

  # 3. Dummy Inference
  print("Running dummy inference...")
  x = jnp.ones((1, 16, hidden_size)) # [B, T, E]
  output, _ = kda(x)
  
  print(f"Input shape: {x.shape}")
  print(f"Output shape: {output.shape}")
  assert output.shape == x.shape
  print("Success!")

if __name__ == "__main__":
  test_kda_with_real_weights()
