# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wan model loading, tensor reshaping, and device transfer utilities."""

import concurrent.futures
import json
import os
import re
import shutil
import threading
import time
from typing import Callable, Optional

from flax.linen import Partitioned
from flax.traverse_util import flatten_dict, unflatten_dict
from huggingface_hub import hf_hub_download
import jax
import jax.numpy as jnp
from maxtext.utils import max_logging
import ml_dtypes
import numpy as np
from safetensors import safe_open
import torch

_ACTIVATIONS = {
    "swish": jax.nn.silu,
    "silu": jax.nn.silu,
    "relu": jax.nn.relu,
    "gelu": jax.nn.gelu,
    "gelu_tanh": jax.nn.gelu,
    "mish": jax.nn.mish,
}


def get_activation(name: str):
  func = _ACTIVATIONS.get(name)
  if func is None:
    raise ValueError(f"Unknown activation function: {name}")
  return func


class FlaxModelMixin:
  """Base mixin for Flax models."""

  config_name = "config.json"

  @classmethod
  def _from_config(cls, config, **kwargs):
    return cls(config, **kwargs)


def validate_flax_state_dict(expected_pytree: dict, new_pytree: dict):
  expected_pytree = flatten_dict(expected_pytree)
  if len(expected_pytree.keys()) != len(new_pytree.keys()):
    set1 = set(expected_pytree.keys())
    set2 = set(new_pytree.keys())
    missing_keys = set1 ^ set2
    max_logging.log(f"missing keys : {missing_keys}")
  for key in expected_pytree.keys():
    if key in new_pytree.keys():
      try:
        expected_pytree_shape = expected_pytree[key].shape
      except Exception:
        expected_pytree_shape = expected_pytree[key].value.shape
      if expected_pytree_shape != new_pytree[key].shape:
        max_logging.log(
            f"shape mismatch for {key}: expected {expected_pytree[key].shape}"
            f" but got {new_pytree[key].shape}"
        )
    else:
      max_logging.log(f"key: {key} not found...")


def torch2jax(torch_tensor: torch.Tensor):
  is_bfloat16 = torch_tensor.dtype == torch.bfloat16
  if is_bfloat16:
    torch_tensor = torch_tensor.float()
  if torch_tensor.device.type != "cpu":
    torch_tensor = torch_tensor.to("cpu")
  numpy_value = torch_tensor.numpy()
  local_cpu_device_0 = jax.local_devices(backend="cpu")[0]
  return jnp.array(
      numpy_value,
      dtype=jnp.bfloat16 if is_bfloat16 else None,
      device=local_cpu_device_0,
  )


def rename_key(key):
  regex = r"\w+[.]\d+"
  pats = re.findall(regex, key)
  for pat in pats:
    key = key.replace(pat, "_".join(pat.split(".")))
  return key


def rename_key_and_reshape_tensor(
    pt_tuple_key, pt_tensor, random_flax_state_dict, scan_layers=False
):
  renamed_pt_tuple_key = pt_tuple_key[:-1] + ("scale",)
  if len(pt_tuple_key) > 1:
    for rename_from, rename_to in (
        ("to_out_0", "proj_attn"),
        ("to_k", "key"),
        ("to_v", "value"),
        ("to_q", "query"),
        ("txt_attn_proj", "txt_attn_proj"),
        ("img_attn_proj", "img_attn_proj"),
        ("txt_attn_qkv", "txt_attn_qkv"),
        ("img_attn_qkv", "img_attn_qkv"),
    ):
      if pt_tuple_key[-2] == rename_from:
        weight_name = pt_tuple_key[-1]
        weight_name = "kernel" if weight_name == "weight" else weight_name
        renamed_pt_tuple_key = pt_tuple_key[:-2] + (rename_to, weight_name)
        if renamed_pt_tuple_key in random_flax_state_dict:
          return renamed_pt_tuple_key, pt_tensor.T

  if (
      any("norm" in str_ for str_ in pt_tuple_key)
      and (pt_tuple_key[-1] == "bias")
      and (pt_tuple_key[:-1] + ("bias",) not in random_flax_state_dict)
      and (pt_tuple_key[:-1] + ("scale",) in random_flax_state_dict)
  ):
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("scale",)
    return renamed_pt_tuple_key, pt_tensor
  elif (
      pt_tuple_key[-1] in ["weight", "gamma"]
      and pt_tuple_key[:-1] + ("scale",) in random_flax_state_dict
  ):
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("scale",)
    return renamed_pt_tuple_key, pt_tensor

  if (
      pt_tuple_key[-1] == "weight"
      and pt_tuple_key[:-1] + ("embedding",) in random_flax_state_dict
  ):
    pt_tuple_key = pt_tuple_key[:-1] + ("embedding",)
    return renamed_pt_tuple_key, pt_tensor

  renamed_pt_tuple_key = pt_tuple_key[:-1] + ("kernel",)
  if pt_tuple_key[-1] == "weight" and pt_tensor.ndim == 4:
    pt_tensor = pt_tensor.transpose(2, 3, 1, 0)
    return renamed_pt_tuple_key, pt_tensor

  renamed_pt_tuple_key = pt_tuple_key[:-1] + ("kernel",)
  if pt_tuple_key[-1] == "weight" and pt_tensor.ndim == 5:
    pt_tensor = pt_tensor.transpose(2, 3, 4, 1, 0)
    return renamed_pt_tuple_key, pt_tensor

  renamed_pt_tuple_key = pt_tuple_key[:-1] + ("kernel",)
  if pt_tuple_key[-1] == "weight":
    pt_tensor = pt_tensor.T
    return renamed_pt_tuple_key, pt_tensor

  renamed_pt_tuple_key = pt_tuple_key[:-1] + ("weight",)
  if pt_tuple_key[-1] == "gamma":
    renamed_pt_tuple_key = pt_tuple_key
    pt_tensor = pt_tensor.flatten()
    return renamed_pt_tuple_key, pt_tensor

  renamed_pt_tuple_key = pt_tuple_key[:-1] + ("bias",)
  if pt_tuple_key[-1] == "beta":
    return renamed_pt_tuple_key, pt_tensor

  return pt_tuple_key, pt_tensor


CAUSVID_TRANSFORMER_MODEL_NAME_OR_PATH = "lightx2v/Wan2.1-T2V-14B-CausVid"
WAN_21_FUSION_X_MODEL_NAME_OR_PATH = "vrgamedevgirl84/Wan14BT2VFusioniX"

# WAN 2.2 transformer and transformer_2 have byte-identical index.json files,
# i.e. ONE blob in the HF hub cache. hf_hub revalidates and rewrites cached
# blobs, so parallel transformer loads must not resolve metadata concurrently.
_HF_METADATA_LOCK = threading.Lock()


def _tuple_str_to_int(in_tuple):
  out_list = []
  for item in in_tuple:
    try:
      out_list.append(int(item))
    except ValueError:
      out_list.append(item)
  return tuple(out_list)


def _normalize_animate_list_key(key):
  """Convert flattened animate list names into nnx.List-style tuple paths."""
  if not key:
    return key

  if isinstance(key[0], str) and key[0].startswith("face_adapter_"):
    adapter_idx = int(key[0].split("_")[-1])
    return ("face_adapter", adapter_idx) + key[1:]

  if (
      len(key) >= 2
      and key[0] == "motion_encoder"
      and isinstance(key[1], str)
      and key[1].startswith("motion_network_")
  ):
    layer_idx = int(key[1].split("_")[-1])
    return ("motion_encoder", "motion_network", layer_idx) + key[2:]

  return key


def rename_for_nnx(key):
  new_key = key
  if "norm_k" in key or "norm_q" in key:
    new_key = key[:-1] + ("scale",)
  return new_key


def rename_for_custom_trasformer(key):
  renamed_pt_key = key.replace("model.diffusion_model.", "")

  renamed_pt_key = renamed_pt_key.replace(
      "head.modulation", "scale_shift_table"
  )
  renamed_pt_key = renamed_pt_key.replace("head.head", "proj_out")
  renamed_pt_key = renamed_pt_key.replace(
      "text_embedding_0", "condition_embedder.text_embedder.linear_1"
  )
  renamed_pt_key = renamed_pt_key.replace(
      "text_embedding_2", "condition_embedder.text_embedder.linear_2"
  )
  renamed_pt_key = renamed_pt_key.replace(
      "time_embedding_0", "condition_embedder.time_embedder.linear_1"
  )
  renamed_pt_key = renamed_pt_key.replace(
      "time_embedding_2", "condition_embedder.time_embedder.linear_2"
  )
  renamed_pt_key = renamed_pt_key.replace(
      "time_projection_1", "condition_embedder.time_proj"
  )

  renamed_pt_key = renamed_pt_key.replace("blocks_", "blocks.")
  renamed_pt_key = renamed_pt_key.replace("self_attn", "attn1")
  renamed_pt_key = renamed_pt_key.replace("cross_attn", "attn2")
  renamed_pt_key = renamed_pt_key.replace(".q.", ".query.")
  renamed_pt_key = renamed_pt_key.replace(".k.", ".key.")
  renamed_pt_key = renamed_pt_key.replace(".v.", ".value.")
  renamed_pt_key = renamed_pt_key.replace(".o.", ".proj_attn.")
  renamed_pt_key = renamed_pt_key.replace("ffn_0", "ffn.act_fn.proj")
  renamed_pt_key = renamed_pt_key.replace("ffn_2", "ffn.proj_out")
  renamed_pt_key = renamed_pt_key.replace(".modulation", ".scale_shift_table")
  renamed_pt_key = renamed_pt_key.replace("norm3", "norm2.layer_norm")

  return renamed_pt_key


def get_key_and_value(
    pt_tuple_key,
    tensor,
    flax_state_dict,
    random_flax_state_dict,
    scan_layers,
    num_layers=40,
):
  block_index = None
  if scan_layers:
    if len(pt_tuple_key) >= 2 and pt_tuple_key[0] == "blocks":
      block_index = int(pt_tuple_key[1])
      pt_tuple_key = ("blocks",) + pt_tuple_key[2:]

  flax_key, flax_tensor = rename_key_and_reshape_tensor(
      pt_tuple_key, tensor, random_flax_state_dict, scan_layers
  )

  flax_key = rename_for_nnx(flax_key)
  flax_key = _tuple_str_to_int(flax_key)

  if scan_layers and block_index is not None:
    if flax_key in flax_state_dict:
      new_tensor = flax_state_dict[flax_key]
    else:
      new_tensor = jnp.zeros(
          (num_layers,) + flax_tensor.shape, dtype=flax_tensor.dtype
      )
    flax_tensor = new_tensor.at[block_index].set(flax_tensor)
  return flax_key, flax_tensor


def _build_random_flax_state_dict(eval_shapes):
  flattened_dict = flatten_dict(eval_shapes)
  random_flax_state_dict = {}
  for key, value in flattened_dict.items():
    random_flax_state_dict[tuple(str(item) for item in key)] = value
  return random_flax_state_dict


def _rename_common_wan_transformer_key(renamed_pt_key: str) -> str:
  if "condition_embedder" in renamed_pt_key:
    renamed_pt_key = renamed_pt_key.replace(
        "time_embedding_0", "time_embedder.linear_1"
    )
    renamed_pt_key = renamed_pt_key.replace(
        "time_embedding_2", "time_embedder.linear_2"
    )
    renamed_pt_key = renamed_pt_key.replace("time_projection_1", "time_proj")
    renamed_pt_key = renamed_pt_key.replace(
        "text_embedding_0", "text_embedder.linear_1"
    )
    renamed_pt_key = renamed_pt_key.replace(
        "text_embedding_2", "text_embedder.linear_2"
    )

  if "image_embedder" in renamed_pt_key:
    if "net.0.proj" in renamed_pt_key:
      renamed_pt_key = renamed_pt_key.replace("net.0.proj", "net_0")
    elif "net_0.proj" in renamed_pt_key:
      renamed_pt_key = renamed_pt_key.replace("net_0.proj", "net_0")
    if "net.2" in renamed_pt_key:
      renamed_pt_key = renamed_pt_key.replace("net.2", "net_2")
    renamed_pt_key = renamed_pt_key.replace("norm1", "norm1.layer_norm")
    if "norm1" in renamed_pt_key or "norm2" in renamed_pt_key:
      renamed_pt_key = renamed_pt_key.replace("weight", "scale")
      renamed_pt_key = renamed_pt_key.replace("kernel", "scale")

  renamed_pt_key = renamed_pt_key.replace("blocks_", "blocks.")
  renamed_pt_key = renamed_pt_key.replace(
      ".scale_shift_table", ".adaln_scale_shift_table"
  )
  renamed_pt_key = renamed_pt_key.replace("to_out_0", "proj_attn")
  renamed_pt_key = renamed_pt_key.replace("ffn.net_2", "ffn.proj_out")
  renamed_pt_key = renamed_pt_key.replace("ffn.net_0", "ffn.act_fn")
  renamed_pt_key = renamed_pt_key.replace("norm2", "norm2.layer_norm")

  return renamed_pt_key


def _rename_wan_animate_pt_tuple_key(pt_key: str):
  renamed_pt_key = _rename_common_wan_transformer_key(rename_key(pt_key))
  is_motion_custom_weight = _is_motion_encoder_custom_weight(pt_key)

  renamed_pt_key = renamed_pt_key.replace(".activation.bias", ".act_fn.bias")
  if is_motion_custom_weight and renamed_pt_key.endswith(".kernel"):
    renamed_pt_key = renamed_pt_key[:-7] + ".weight"

  return tuple(renamed_pt_key.split(".")), is_motion_custom_weight


def get_wan_animate_key_and_value(
    pt_tuple_key,
    tensor,
    flax_state_dict,
    random_flax_state_dict,
    scan_layers,
    is_motion_custom_weight=False,
    num_layers=40,
):
  if is_motion_custom_weight:
    flax_key = _normalize_animate_list_key(_tuple_str_to_int(pt_tuple_key))
    return flax_key, tensor

  flax_key, flax_tensor = get_key_and_value(
      pt_tuple_key,
      tensor,
      flax_state_dict,
      random_flax_state_dict,
      scan_layers,
      num_layers,
  )
  flax_key = _normalize_animate_list_key(flax_key)
  return flax_key, flax_tensor


def load_fusionx_transformer(
    pretrained_model_name_or_path: str,
    eval_shapes: dict,
    device: str,
    hf_download: bool = True,
    num_layers: int = 40,
    scan_layers: bool = True,
):
  device = jax.local_devices(backend=device)[0]
  with jax.default_device(device):
    if hf_download:
      ckpt_shard_path = hf_hub_download(
          pretrained_model_name_or_path,
          filename="Wan14BT2VFusioniX_fp16_.safetensors",
      )
      tensors = {}
      with safe_open(ckpt_shard_path, framework="pt") as f:
        for k in f.keys():
          tensors[k] = torch2jax(f.get_tensor(k))

      flax_state_dict = {}
      cpu = jax.local_devices(backend="cpu")[0]
      flattened_dict = flatten_dict(eval_shapes)
      # turn all block numbers to strings just for matching weights.
      # Later they will be turned back to ints.
      random_flax_state_dict = {}
      for key in flattened_dict:
        string_tuple = tuple([str(item) for item in key])
        random_flax_state_dict[string_tuple] = flattened_dict[key]
      for pt_key, tensor in tensors.items():
        renamed_pt_key = rename_key(pt_key)

        renamed_pt_key = rename_for_custom_trasformer(renamed_pt_key)

        pt_tuple_key = tuple(renamed_pt_key.split("."))

        flax_key, flax_tensor = get_key_and_value(
            pt_tuple_key,
            tensor,
            flax_state_dict,
            random_flax_state_dict,
            scan_layers,
            num_layers,
        )
        flax_state_dict[flax_key] = jax.device_put(
            jnp.asarray(flax_tensor), device=cpu
        )

      validate_flax_state_dict(eval_shapes, flax_state_dict)
      flax_state_dict = unflatten_dict(flax_state_dict)
      del tensors
      jax.clear_caches()
      return flax_state_dict


def load_causvid_transformer(
    pretrained_model_name_or_path: str,
    eval_shapes: dict,
    device: str,
    hf_download: bool = True,
    num_layers: int = 40,
    scan_layers: bool = True,
):
  device = jax.local_devices(backend=device)[0]
  with jax.default_device(device):
    if hf_download:
      ckpt_shard_path = hf_hub_download(
          pretrained_model_name_or_path, filename="causal_model.pt"
      )
      loaded_state_dict = torch.load(ckpt_shard_path)

      tensors = {}
      flax_state_dict = {}
      cpu = jax.local_devices(backend="cpu")[0]
      flattened_dict = flatten_dict(eval_shapes)
      # turn all block numbers to strings just for matching weights.
      # Later they will be turned back to ints.
      random_flax_state_dict = {}
      for key in flattened_dict:
        string_tuple = tuple([str(item) for item in key])
        random_flax_state_dict[string_tuple] = flattened_dict[key]
      for pt_key, tensor in loaded_state_dict.items():
        tensor = torch2jax(tensor)
        renamed_pt_key = rename_key(pt_key)
        renamed_pt_key = rename_for_custom_trasformer(renamed_pt_key)

        pt_tuple_key = tuple(renamed_pt_key.split("."))
        flax_key, flax_tensor = get_key_and_value(
            pt_tuple_key,
            tensor,
            flax_state_dict,
            random_flax_state_dict,
            scan_layers,
            num_layers,
        )
        flax_state_dict[flax_key] = jax.device_put(
            jnp.asarray(flax_tensor), device=cpu
        )

      validate_flax_state_dict(eval_shapes, flax_state_dict)
      flax_state_dict = unflatten_dict(flax_state_dict)
      del tensors
      jax.clear_caches()
      return flax_state_dict


def load_wan_transformer(
    pretrained_model_name_or_path: str,
    eval_shapes: dict,
    device: str,
    hf_download: bool = True,
    num_layers: int = 40,
    scan_layers: bool = True,
    subfolder: str = "",
    cast_dtype_fn: Optional[Callable] = None,
    converted_cache_dir: str = "",
):
  if pretrained_model_name_or_path == CAUSVID_TRANSFORMER_MODEL_NAME_OR_PATH:
    return load_causvid_transformer(
        pretrained_model_name_or_path,
        eval_shapes,
        device,
        hf_download,
        num_layers,
        scan_layers,
    )
  elif pretrained_model_name_or_path == WAN_21_FUSION_X_MODEL_NAME_OR_PATH:
    return load_fusionx_transformer(
        pretrained_model_name_or_path,
        eval_shapes,
        device,
        hf_download,
        num_layers,
        scan_layers,
    )
  else:
    return load_base_wan_transformer(
        pretrained_model_name_or_path,
        eval_shapes,
        device,
        hf_download,
        num_layers,
        scan_layers,
        subfolder,
        cast_dtype_fn,
        converted_cache_dir,
    )


def _torch_tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
  """Converts a CPU torch tensor to numpy without copying or upcasting.

  bfloat16 has no native numpy dtype, so it is reinterpreted through uint16
  into ml_dtypes.bfloat16 (bit-identical, zero-copy).
  """
  if tensor.dtype == torch.bfloat16:
    return tensor.view(torch.uint16).numpy().view(ml_dtypes.bfloat16)
  return tensor.numpy()


def _converted_key_to_filename(flax_key: tuple) -> str:
  return ".".join(str(k) for k in flax_key) + ".npy"


def try_load_converted_weights(
    cache_dir: str, eval_shapes: dict, cast_dtype_fn: Optional[Callable]
) -> Optional[dict]:
  """Loads a converted-weights cache as mmapped arrays, or None on mismatch.

  The torch->flax conversion (transpose + scan-stack + cast) is a pure
  function of the checkpoint, so it is paid once and memoized on disk.
  Keys/shapes are validated against eval_shapes and dtypes against
  cast_dtype_fn, so a policy or model change falls back to a fresh
  conversion (which re-saves).
  """
  manifest_path = os.path.join(cache_dir, "manifest.json")
  if not os.path.isfile(manifest_path):
    return None
  try:
    with open(manifest_path, "r") as f:
      manifest = json.load(f)
    expected_keys = set(flatten_dict(eval_shapes).keys())

    def load_one(key_str, meta):
      flax_key = _tuple_str_to_int(tuple(key_str.split(".")))
      logical_dtype = np.dtype(meta["dtype"])
      if cast_dtype_fn is not None and logical_dtype != np.dtype(
          cast_dtype_fn(flax_key)
      ):
        raise ValueError(f"dtype policy changed for {key_str}")
      # Eager parallel read (page-cache/RAM speed): an mmap would defer the
      # read into the device_put as serial page faults, halving put speed.
      value = np.load(os.path.join(cache_dir, meta["file"]))
      if meta.get("bitview"):
        # Non-native dtypes (bf16/fp8) are stored as same-width uints:
        # npy cannot resolve ml_dtypes descriptors on all paths.
        value = value.view(logical_dtype)
      if tuple(value.shape) != tuple(meta["shape"]):
        raise ValueError(f"shape changed for {key_str}")
      return flax_key, value

    flax_state_dict = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
      for flax_key, value in executor.map(
          lambda kv: load_one(*kv), manifest.items()
      ):
        flax_state_dict[flax_key] = value
    if set(flax_state_dict.keys()) != expected_keys:
      return None
    return unflatten_dict(flax_state_dict)
  except (OSError, ValueError, KeyError, TypeError) as e:
    max_logging.log(f"Converted-weights cache unusable ({e}); reconverting")
    return None


def save_converted_weights(cache_dir: str, flat_state_dict: dict) -> None:
  """Writes the converted tree as per-tensor .npy + manifest, atomically."""
  tmp_dir = f"{cache_dir}.tmp.{os.getpid()}"
  os.makedirs(tmp_dir, exist_ok=True)
  manifest = {}
  uint_by_width = {1: np.uint8, 2: np.uint16, 4: np.uint32}
  for flax_key, value in flat_state_dict.items():
    filename = _converted_key_to_filename(flax_key)
    bitview = value.dtype.kind not in "fiub"  # ml_dtypes (bf16/fp8) etc.
    stored = (
        value.view(uint_by_width[value.dtype.itemsize]) if bitview else value
    )
    np.save(os.path.join(tmp_dir, filename), stored)
    key_str = ".".join(str(k) for k in flax_key)
    manifest[key_str] = {
        "file": filename,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "bitview": bitview,
    }
  with open(os.path.join(tmp_dir, "manifest.json"), "w") as f:
    json.dump(manifest, f)
  try:
    os.rename(tmp_dir, cache_dir)
  except OSError:
    shutil.rmtree(tmp_dir, ignore_errors=True)  # another process won the race


def load_base_wan_transformer(
    pretrained_model_name_or_path: str,
    eval_shapes: dict,
    device: str,
    hf_download: bool = True,
    num_layers: int = 40,
    scan_layers: bool = True,
    subfolder: str = "",
    cast_dtype_fn: Optional[Callable] = None,
    converted_cache_dir: str = "",
):
  """Loads WAN transformer weights from diffusers safetensors shards.

  Fast path compared to the historical implementation:
    - tensors are read zero-copy from the safetensors mmap (no bf16->f32
      round trip through torch.float()),
    - scanned block weights are written in place into one preallocated
      (num_layers, ...) numpy buffer per param (the old jnp
      ``at[block].set`` rebuilt the full stacked array once per layer,
      i.e. O(num_layers^2) copies),
    - the optional ``cast_dtype_fn(flax_key) -> np.dtype`` casts each param
      to its final dtype during this single copy, so no later full-tree
      cast pass is needed,
    - shard files are converted in parallel threads (numpy copies release
      the GIL).
  Returns a nested dict of numpy arrays (host memory).
  """
  del device  # weights stay in plain host numpy until device_put by the caller
  if converted_cache_dir:
    t_start = time.perf_counter()
    cached = try_load_converted_weights(
        converted_cache_dir, eval_shapes, cast_dtype_fn
    )
    if cached is not None:
      max_logging.log(
          f"Loaded converted {subfolder or 'transformer'} weights (mmap) in"
          f" {time.perf_counter() - t_start:.1f}s"
      )
      return cached
  filename = "diffusion_pytorch_model.safetensors.index.json"
  local_files = False
  if os.path.isdir(pretrained_model_name_or_path):
    index_file_path = os.path.join(
        pretrained_model_name_or_path, subfolder, filename
    )
    if not os.path.isfile(index_file_path):
      raise FileNotFoundError(
          f"File {index_file_path} not found for local directory."
      )
    local_files = True
  elif hf_download:
    # download the index file for sharded models.
    with _HF_METADATA_LOCK:
      index_file_path = hf_hub_download(
          pretrained_model_name_or_path,
          subfolder=subfolder,
          filename=filename,
      )
  t_start = time.perf_counter()
  with open(index_file_path, "r") as f:
    index_dict = json.load(f)
  model_files = sorted(set(index_dict["weight_map"].values()))

  # turn all block numbers to strings just for matching weights.
  # Later they will be turned back to ints.
  random_flax_state_dict = _build_random_flax_state_dict(eval_shapes)
  flax_state_dict = {}
  dict_lock = threading.Lock()

  def resolve_shard_path(model_file):
    if local_files:
      return os.path.join(pretrained_model_name_or_path, subfolder, model_file)
    return hf_hub_download(
        pretrained_model_name_or_path, subfolder=subfolder, filename=model_file
    )

  def convert_chunk(ckpt_shard_path, chunk_keys):
    # Each task opens its own handle: safetensors mmap open is cheap and
    # per-thread handles avoid serializing get_tensor calls.
    with safe_open(ckpt_shard_path, framework="pt") as f:
      for pt_key in chunk_keys:
        tensor = _torch_tensor_to_numpy(f.get_tensor(pt_key))
        renamed_pt_key = rename_key(pt_key)
        renamed_pt_key = _rename_common_wan_transformer_key(renamed_pt_key)
        pt_tuple_key = tuple(renamed_pt_key.split("."))

        block_index = None
        if (
            scan_layers
            and len(pt_tuple_key) >= 2
            and pt_tuple_key[0] == "blocks"
        ):
          block_index = int(pt_tuple_key[1])
          pt_tuple_key = ("blocks",) + pt_tuple_key[2:]

        # rename_key_and_reshape_tensor only reindexes/transposes views; the
        # single real copy happens on assignment into the target buffer below.
        flax_key, flax_tensor = rename_key_and_reshape_tensor(
            pt_tuple_key, tensor, random_flax_state_dict, scan_layers
        )
        flax_key = rename_for_nnx(flax_key)
        flax_key = _tuple_str_to_int(flax_key)

        if block_index is not None:
          with dict_lock:
            stacked = flax_state_dict.get(flax_key)
            if stacked is None:
              stacked_dtype = (
                  cast_dtype_fn(flax_key)
                  if cast_dtype_fn
                  else flax_tensor.dtype
              )
              stacked = np.empty(
                  (num_layers,) + flax_tensor.shape, dtype=stacked_dtype
              )
              flax_state_dict[flax_key] = stacked
          # Rows are disjoint per block, so concurrent writes need no lock.
          # This assignment fuses transpose + dtype cast (RTNE, matching XLA
          # convert semantics) into one pass.
          stacked[block_index] = flax_tensor
        else:
          target_dtype = (
              cast_dtype_fn(flax_key) if cast_dtype_fn else flax_tensor.dtype
          )
          # Copy (never keep a view) so nothing references the shard mmap.
          value = np.array(
              flax_tensor, dtype=target_dtype, copy=True, order="C"
          )
          with dict_lock:
            flax_state_dict[flax_key] = value

  # Chunk keys per shard so conversion parallelizes across tensors, not just
  # across the ~12 shard files. norm_added_q is explicitly ignored by the
  # diffusers implementation.
  chunk_size = 16
  tasks = []
  for model_file in model_files:
    ckpt_shard_path = resolve_shard_path(model_file)
    with safe_open(ckpt_shard_path, framework="pt") as f:
      shard_keys = [k for k in f.keys() if "norm_added_q" not in k]
    for i in range(0, len(shard_keys), chunk_size):
      tasks.append((ckpt_shard_path, shard_keys[i : i + chunk_size]))
  max_logging.log(
      f"Load and port {pretrained_model_name_or_path} {subfolder}:"
      f" {len(model_files)} shards, {len(tasks)} chunks"
  )
  with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
    futures = [
        executor.submit(convert_chunk, path, keys) for path, keys in tasks
    ]
    for future in concurrent.futures.as_completed(futures):
      future.result()  # re-raise conversion errors

  validate_flax_state_dict(eval_shapes, flax_state_dict)
  if converted_cache_dir and not os.path.isdir(converted_cache_dir):
    t_save = time.perf_counter()
    if jax.process_index() == 0:
      save_converted_weights(converted_cache_dir, flax_state_dict)
      max_logging.log(
          f"Saved converted-weights cache to {converted_cache_dir} in"
          f" {time.perf_counter() - t_save:.1f}s"
      )
  flax_state_dict = unflatten_dict(flax_state_dict)
  max_logging.log(
      f"Converted {subfolder or 'transformer'} weights to host arrays in"
      f" {time.perf_counter() - t_start:.1f}s"
  )
  return flax_state_dict


def _is_motion_encoder_custom_weight(pt_key: str) -> bool:
  """Returns True for FlaxMotionConv2d/FlaxMotionLinear weight keys that must NOT be renamed to kernel."""
  prefixes = (
      "motion_encoder.conv_in.",
      "motion_encoder.conv_out.",
  )
  if any(pt_key.startswith(p) for p in prefixes) and pt_key.endswith(".weight"):
    return True
  if "motion_encoder.res_blocks." in pt_key and pt_key.endswith(".weight"):
    return True
  if "motion_encoder.motion_network." in pt_key and pt_key.endswith(".weight"):
    return True
  return False


def load_wan_animate_transformer(
    pretrained_model_name_or_path: str,
    eval_shapes: dict,
    device: str,
    hf_download: bool = True,
    num_layers: int = 40,
    scan_layers: bool = True,
    subfolder: str = "transformer",
):
  """Loads WanAnimate transformer weights from a HuggingFace checkpoint.

  Handles the additional key mappings for:
    - pose_patch_embedding (nnx.Conv3d → kernel)
    - motion_encoder.* (FlaxMotionConv2d/FlaxMotionLinear → keep as 'weight', no
    transpose)
    - activation.bias → act_fn.bias  (FusedLeakyReLU bias remapping)
    - face_encoder.* (nnx.Conv/Linear → standard rename to kernel)
    - face_adapter.* (nnx.Linear → standard rename to kernel)
  """
  device = jax.local_devices(backend=device)[0]
  filename = "diffusion_pytorch_model.safetensors.index.json"
  local_files = False
  if os.path.isdir(pretrained_model_name_or_path):
    index_file_path = os.path.join(
        pretrained_model_name_or_path, subfolder, filename
    )
    if not os.path.isfile(index_file_path):
      raise FileNotFoundError(
          f"File {index_file_path} not found for local directory."
      )
    local_files = True
  elif hf_download:
    index_file_path = hf_hub_download(
        pretrained_model_name_or_path,
        subfolder=subfolder,
        filename=filename,
    )
  with jax.default_device(device):
    with open(index_file_path, "r") as f:
      index_dict = json.load(f)
    model_files = set()
    for key in index_dict["weight_map"].keys():
      model_files.add(index_dict["weight_map"][key])

    model_files = list(model_files)
    tensors = {}
    for model_file in model_files:
      if local_files:
        ckpt_shard_path = os.path.join(
            pretrained_model_name_or_path, subfolder, model_file
        )
      else:
        ckpt_shard_path = hf_hub_download(
            pretrained_model_name_or_path,
            subfolder=subfolder,
            filename=model_file,
        )
      max_logging.log(
          f"Load and port {pretrained_model_name_or_path} {subfolder} on"
          f" {device}"
      )
      if ckpt_shard_path is not None:
        with safe_open(ckpt_shard_path, framework="pt") as f:
          for k in f.keys():
            tensors[k] = torch2jax(f.get_tensor(k))

    flax_state_dict = {}
    cpu = jax.local_devices(backend="cpu")[0]
    random_flax_state_dict = _build_random_flax_state_dict(eval_shapes)

    for pt_key, tensor in tensors.items():
      if "norm_added_q" in pt_key:
        continue

      pt_tuple_key, is_motion_custom_weight = _rename_wan_animate_pt_tuple_key(
          pt_key
      )
      flax_key, flax_tensor = get_wan_animate_key_and_value(
          pt_tuple_key,
          tensor,
          flax_state_dict,
          random_flax_state_dict,
          scan_layers,
          is_motion_custom_weight=is_motion_custom_weight,
          num_layers=num_layers,
      )

      flax_state_dict[flax_key] = jax.device_put(
          jnp.asarray(flax_tensor), device=cpu
      )

    validate_flax_state_dict(eval_shapes, flax_state_dict)
    flax_state_dict = unflatten_dict(flax_state_dict)
    del tensors
    jax.clear_caches()
    return flax_state_dict


def load_wan_vae(
    pretrained_model_name_or_path: str,
    eval_shapes: dict,
    device: str,
    hf_download: bool = True,
):
  device = jax.devices(device)[0]
  subfolder = "vae"
  filename = "diffusion_pytorch_model.safetensors"
  if os.path.isdir(pretrained_model_name_or_path):
    ckpt_path = os.path.join(pretrained_model_name_or_path, subfolder, filename)
    if not os.path.isfile(ckpt_path):
      raise FileNotFoundError(
          f"File {ckpt_path} not found for local directory."
      )
  elif hf_download:
    ckpt_path = hf_hub_download(
        pretrained_model_name_or_path, subfolder=subfolder, filename=filename
    )
  max_logging.log(
      f"Load and port {pretrained_model_name_or_path} VAE on {device}"
  )
  with jax.default_device(device):
    if ckpt_path is not None:
      tensors = {}
      with safe_open(ckpt_path, framework="pt") as f:
        for k in f.keys():
          tensors[k] = torch2jax(f.get_tensor(k))
      flax_state_dict = {}
      cpu = jax.local_devices(backend="cpu")[0]
      for pt_key, tensor in tensors.items():
        renamed_pt_key = rename_key(pt_key)
        # Order matters
        renamed_pt_key = renamed_pt_key.replace("up_blocks_", "up_blocks.")
        renamed_pt_key = renamed_pt_key.replace("mid_block_", "mid_block.")
        renamed_pt_key = renamed_pt_key.replace("down_blocks_", "down_blocks.")

        renamed_pt_key = renamed_pt_key.replace(
            "conv_in.bias", "conv_in.conv.bias"
        )
        renamed_pt_key = renamed_pt_key.replace(
            "conv_in.weight", "conv_in.conv.weight"
        )
        renamed_pt_key = renamed_pt_key.replace(
            "conv_out.bias", "conv_out.conv.bias"
        )
        renamed_pt_key = renamed_pt_key.replace(
            "conv_out.weight", "conv_out.conv.weight"
        )
        renamed_pt_key = renamed_pt_key.replace("attentions_", "attentions.")
        renamed_pt_key = renamed_pt_key.replace("resnets_", "resnets.")
        renamed_pt_key = renamed_pt_key.replace("upsamplers_", "upsamplers.")
        renamed_pt_key = renamed_pt_key.replace("resample_", "resample.")
        renamed_pt_key = renamed_pt_key.replace("conv1.bias", "conv1.conv.bias")
        renamed_pt_key = renamed_pt_key.replace(
            "conv1.weight", "conv1.conv.weight"
        )
        renamed_pt_key = renamed_pt_key.replace("conv2.bias", "conv2.conv.bias")
        renamed_pt_key = renamed_pt_key.replace(
            "conv2.weight", "conv2.conv.weight"
        )
        renamed_pt_key = renamed_pt_key.replace(
            "time_conv.bias", "time_conv.conv.bias"
        )
        renamed_pt_key = renamed_pt_key.replace(
            "time_conv.weight", "time_conv.conv.weight"
        )
        renamed_pt_key = renamed_pt_key.replace("quant_conv", "quant_conv.conv")
        renamed_pt_key = renamed_pt_key.replace(
            "conv_shortcut", "conv_shortcut.conv"
        )
        if "decoder" in renamed_pt_key:
          renamed_pt_key = renamed_pt_key.replace(
              "resample.1.bias", "resample.layers.1.bias"
          )
          renamed_pt_key = renamed_pt_key.replace(
              "resample.1.weight", "resample.layers.1.weight"
          )
        if "encoder" in renamed_pt_key:
          renamed_pt_key = renamed_pt_key.replace("resample.1", "resample.conv")
        pt_tuple_key = tuple(renamed_pt_key.split("."))
        flax_key, flax_tensor = rename_key_and_reshape_tensor(
            pt_tuple_key, tensor, eval_shapes
        )
        flax_key = _tuple_str_to_int(flax_key)
        flax_state_dict[flax_key] = jax.device_put(
            jnp.asarray(flax_tensor), device=cpu
        )
      validate_flax_state_dict(eval_shapes, flax_state_dict)
      flax_state_dict = unflatten_dict(flax_state_dict)
      del tensors
      jax.clear_caches()
    else:
      raise FileNotFoundError(f"Path {ckpt_path} was not found")

    return flax_state_dict


# ==============================================================================
# Wan Model Hardware, Mesh, GCS, Profiler, and Configuration Utilities
# ==============================================================================

from enum import Enum
import subprocess
import yaml
from jax.experimental import mesh_utils
from maxtext.utils import max_utils


# --- Hardware & TPU Generation Detection ---
class TpuType(Enum):
  TPU_V4 = "v4"
  TPU_V5E = "v5e"
  TPU_V5P = "v5p"
  TPU_V6_LITE = "v6e"
  TPU_7X = "v7x"
  UNKNOWN = "unknown"


def get_tpu_type() -> TpuType:
  """Detects the current TPU hardware generation."""
  try:
    device_kind = jax.devices()[0].device_kind
    if "7x" in device_kind:
      return TpuType.TPU_7X
    elif "v6 lite" in device_kind:
      return TpuType.TPU_V6_LITE
    else:
      return TpuType.UNKNOWN
  except Exception:
    return TpuType.UNKNOWN


def safe_getattr(obj, name, default=None):
  try:
    return getattr(obj, name)
  except (AttributeError, KeyError):
    return default


# --- Attention & Dimension Constants ---
try:
  from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel

  BlockSizes = splash_attention_kernel.BlockSizes
except (ImportError, AttributeError):
  BlockSizes = None

Mesh = jax.sharding.Mesh

# Physical axis names for device meshes.
CONTEXT = "context"

# Logical axis names for diffusion attention.
SELF_ATTN_HEAD = "activation_self_attn_heads"
SELF_ATTN_Q_LENGTH = "activation_self_attn_q_length"
SELF_ATTN_KV_LENGTH = "activation_self_attn_kv_length"
CROSS_ATTN_HEAD = "activation_cross_attn_heads"
CROSS_ATTN_Q_LENGTH = "activation_cross_attn_q_length"
CROSS_ATTN_KV_LENGTH = "activation_cross_attn_kv_length"

# Sequence parallel axis rules
SEQUENCE_PARALLEL_AXIS_RULES = [
    [SELF_ATTN_HEAD, None],
    [SELF_ATTN_Q_LENGTH, CONTEXT],
    [SELF_ATTN_KV_LENGTH, None],
    [CROSS_ATTN_HEAD, None],
    [CROSS_ATTN_Q_LENGTH, CONTEXT],
    [CROSS_ATTN_KV_LENGTH, None],
]


# --- Mesh & Precision Utilities ---
def get_precision(config):
  if not hasattr(config, "precision") or not config.precision:
    return None
  return jax.lax.Precision[config.precision]


def get_flash_block_sizes(config):
  if not hasattr(config, "flash_block_sizes") or not config.flash_block_sizes:
    return None
  if isinstance(config.flash_block_sizes, str):
    flash_block_sizes = json.loads(config.flash_block_sizes)
  else:
    flash_block_sizes = config.flash_block_sizes
  if flash_block_sizes and BlockSizes is not None:
    return BlockSizes(
        block_q=flash_block_sizes["block_q"],
        block_kv_compute=flash_block_sizes["block_kv_compute"],
        block_kv=flash_block_sizes["block_kv"],
        block_q_dkv=flash_block_sizes["block_q_dkv"],
        block_kv_dkv=flash_block_sizes["block_kv_dkv"],
        block_kv_dkv_compute=flash_block_sizes["block_kv_dkv_compute"],
        block_q_dq=flash_block_sizes["block_q_dq"],
        block_kv_dq=flash_block_sizes["block_kv_dq"],
    )
  return None


def device_put_replicated(x, sharding):
  """Although the name indicates replication, this function can be used

  to also shard an array based on sharding.
  """
  from jax.sharding import NamedSharding, PartitionSpec as P

  arr = getattr(x, "value", x)
  shd = getattr(sharding, "value", sharding)
  if hasattr(shd, "devices") and not hasattr(shd, "device_set"):
    shd = NamedSharding(shd, P())
  res = jax.make_array_from_callback(arr.shape, shd, lambda index: arr[index])
  if hasattr(x, "set_value"):
    x.set_value(res)
    return x
  return res


def get_global_batch_size(per_device_batch_size):
  num_devices = len(jax.devices())
  return max(1, int(num_devices * per_device_batch_size))


def create_device_mesh(config, devices=None, logging=True):
  """Creates a device mesh supporting 4D mesh (data, fsdp, context, tensor)."""
  if devices is None:
    devices = jax.devices()
  num_devices = len(devices)
  try:
    num_slices = 1 + max([d.slice_index for d in devices])
  except Exception:
    num_slices = 1
  num_devices_per_slice = num_devices // num_slices
  if logging:
    max_logging.log(f"Devices: {devices} (num_devices: {num_devices})")

  multi_slice_env = num_slices > 1
  config_keys = config.get_keys() if hasattr(config, "get_keys") else config
  if (
      "dcn_context_parallelism" in config_keys
      and "ici_context_parallelism" in config_keys
  ):
    dcn_parallelism = [
        config.dcn_data_parallelism,
        config.dcn_fsdp_parallelism,
        config.dcn_context_parallelism,
        config.dcn_tensor_parallelism,
    ]
    ici_parallelism = [
        config.ici_data_parallelism,
        config.ici_fsdp_parallelism,
        config.ici_context_parallelism,
        config.ici_tensor_parallelism,
    ]
  else:
    dcn_parallelism = [
        getattr(config, "dcn_data_parallelism", 1),
        getattr(config, "dcn_fsdp_parallelism", 1),
        getattr(config, "dcn_tensor_parallelism", 1),
    ]
    ici_parallelism = [
        getattr(config, "ici_data_parallelism", 1),
        getattr(config, "ici_fsdp_parallelism", 1),
        getattr(config, "ici_tensor_parallelism", 1),
    ]

  ici_parallelism = max_utils.fill_unspecified_mesh_axes(
      ici_parallelism, num_devices_per_slice, "ICI"
  )
  allow_split = getattr(config, "allow_split_physical_axes", False)
  if multi_slice_env:
    dcn_parallelism = max_utils.fill_unspecified_mesh_axes(
        dcn_parallelism, num_slices, "DCN"
    )
    mesh = mesh_utils.create_hybrid_device_mesh(
        ici_parallelism,
        dcn_parallelism,
        devices,
        allow_split_physical_axes=allow_split,
    )
  else:
    mesh = mesh_utils.create_device_mesh(
        ici_parallelism, devices, allow_split_physical_axes=allow_split
    )

  if logging:
    max_logging.log(f"Decided on mesh: {mesh}")

  return mesh


# --- File, GCS, and Prompt Utilities ---
def upload_file_to_gcs(output_dir: str, file_path: str, subdir: str = ""):
  """Uploads one generated file to {output_dir}/{subdir}/, logging failures."""
  try:
    from google.cloud import storage

    path_without_scheme = output_dir.removeprefix("gs://")
    parts = path_without_scheme.split("/", 1)
    bucket_name = parts[0]
    folder_name = parts[1] if len(parts) > 1 else ""
    destination_blob_name = os.path.normpath(
        os.path.join(folder_name, subdir, os.path.basename(file_path))
    ).lstrip("/")

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    max_logging.log(
        f"Uploading {file_path} to {bucket_name}/{destination_blob_name}..."
    )
    blob.upload_from_filename(file_path)
    max_logging.log(f"Upload complete {file_path}.")
  except Exception as e:
    max_logging.log(f"An error occurred: {e}")


def upload_file_to_gcs_from_str(output_dir: str, file_name: str, data: str):
  try:
    from google.cloud import storage

    path_without_scheme = output_dir.removeprefix("gs://")
    parts = path_without_scheme.split("/", 1)
    bucket_name = parts[0]
    folder_name = parts[1] if len(parts) > 1 else ""
    destination_blob_name = os.path.normpath(
        os.path.join(folder_name, file_name)
    ).lstrip("/")

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_string(data)
  except Exception as e:
    max_logging.log(f"An error occurred uploading to GCS: {e}")


def write_config_raw_keys_for_gcs(raw_keys):
  if (
      "output_dir" not in raw_keys
      or not raw_keys["output_dir"]
      or not raw_keys["output_dir"].startswith("gs://")
  ):
    return
  run_name = raw_keys.get("run_name")
  if run_name:
    output_dir = os.path.join(raw_keys["output_dir"], run_name)
  else:
    output_dir = raw_keys["output_dir"]
  upload_file_to_gcs_from_str(output_dir, "config.yaml", yaml.dump(raw_keys))


def delete_file(file_path: str):
  """Removes a local file, e.g. after it has been uploaded to GCS."""
  if os.path.exists(file_path):
    try:
      os.remove(file_path)
      max_logging.log(f"Successfully deleted file: {file_path}")
    except OSError as e:
      max_logging.log(f"Error deleting file '{file_path}': {e}")
  else:
    max_logging.log(f"The file '{file_path}' does not exist.")


def get_gcs_output_path(config):
  output_dir = getattr(config, "output_dir", "")
  if not output_dir or not output_dir.startswith("gs://"):
    return None
  run_name = getattr(config, "run_name", "")
  return os.path.join(output_dir, run_name) if run_name else output_dir


def download_blobs(source_blob_name, destination_file_name):
  try:
    from google.cloud import storage

    path_without_scheme = source_blob_name.removeprefix("gs://")
    parts = path_without_scheme.split("/", 1)
    bucket_name = parts[0]
    blob_name = parts[1] if len(parts) > 1 else ""

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    dest_path = os.path.join(destination_file_name, os.path.basename(blob_name))
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    blob.download_to_filename(dest_path)
    return dest_path
  except Exception as e:
    max_logging.log(f"An error occurred downloading blob from GCS: {e}")
    return source_blob_name


def load_prompts(prompt_file_path: str, default_prompt: str = "") -> list:
  if not prompt_file_path:
    if not default_prompt:
      raise ValueError("Neither prompt_file nor prompt was specified.")
    return [default_prompt]
  if prompt_file_path.startswith("http://") or prompt_file_path.startswith(
      "https://"
  ):
    import requests

    response = requests.get(prompt_file_path)
    response.raise_for_status()
    raw_lines = response.text.splitlines()
  else:
    if not os.path.isfile(prompt_file_path):
      raise FileNotFoundError(
          f"Prompt file not found at local path: {prompt_file_path}"
      )
    with open(prompt_file_path, "r", encoding="utf-8") as f:
      raw_lines = f.readlines()
  prompts = [
      line.strip()
      for line in raw_lines
      if line.strip() and not line.strip().startswith("#")
  ]
  if not prompts:
    if default_prompt:
      return [default_prompt]
    raise ValueError(
        f"Prompt file '{prompt_file_path}' contains no valid non-empty prompts."
    )
  return prompts


def chunk_and_pad(items: list, batch_size: int):
  if batch_size <= 0:
    raise ValueError(f"batch_size must be positive, got {batch_size}")
  for i in range(0, len(items), batch_size):
    chunk = items[i : i + batch_size]
    actual_len = len(chunk)
    padded_chunk = (
        chunk + [chunk[-1]] * (batch_size - actual_len)
        if actual_len < batch_size
        else chunk
    )
    yield i, padded_chunk, actual_len


def get_git_commit_hash():
  try:
    return (
        subprocess.check_output(["git", "rev-parse", "HEAD"])
        .decode("ascii")
        .strip()
    )
  except Exception:
    return "UNKNOWN"


# --- Profiler ---
def profiler_enabled(config):
  return (
      bool(getattr(config, "enable_profiler", False))
      and jax.process_index() == 0
  )


class Profiler:

  def __init__(self, config, session_name=None):
    self.config = config
    self.session_name = session_name
    self._active = None

  def start(self):
    if not profiler_enabled(self.config):
      return
    log_dir = getattr(self.config, "tensorboard_dir", "")
    if not log_dir or log_dir.startswith("gs://"):
      run_name = getattr(self.config, "run_name", "run")
      log_dir = os.path.join("/tmp/profiler_traces", run_name)
    if self.session_name:
      log_dir = os.path.join(log_dir, self.session_name)
    os.makedirs(log_dir, exist_ok=True)
    max_logging.log(f"Starting profiler trace in: {log_dir}")
    jax.profiler.start_trace(log_dir)
    self._active = "jax"

  def stop(self):
    if self._active == "jax":
      jax.profiler.stop_trace()
      trace_dir = getattr(self.config, "tensorboard_dir", "")
      if trace_dir.startswith("gs://"):
        run_name = getattr(self.config, "run_name", "run")
        local_dir = os.path.join("/tmp/profiler_traces", run_name)
        if os.path.exists(local_dir):
          try:
            from google.cloud import storage

            path_without_scheme = trace_dir.removeprefix("gs://")
            parts = path_without_scheme.split("/", 1)
            bucket_name = parts[0]
            prefix = parts[1] if len(parts) > 1 else ""
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            for root, _, files in os.walk(local_dir):
              for file in files:
                local_file = os.path.join(root, file)
                rel_path = os.path.relpath(local_file, local_dir)
                blob_name = os.path.join(prefix, rel_path)
                blob = bucket.blob(blob_name)
                blob.upload_from_filename(local_file)
                max_logging.log(
                    f"Uploaded {local_file} to gs://{bucket_name}/{blob_name}"
                )
          except Exception as e:
            max_logging.log(f"Failed to upload profiler traces: {e}")
    self._active = None
