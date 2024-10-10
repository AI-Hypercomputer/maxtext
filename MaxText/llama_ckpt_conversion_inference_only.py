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

r"""Convert weights from a Llama for MaxText inference.

Usage:

Get LLaMA pytorch_vars from Meta

Example cmd:
To save a ckpt
python3 MaxText/llama_or_mistral_ckpt.py --base-model-path <path/to/meta/ckpt> \
    --maxtext-model-path <GCS/path/to/save/new/maxtext/ckpt> --model-size llama2-7b

For large size model (e.g. 70B model), this script requires large memory VM.
The script load and save weights in a single pass.
To fit less memory, modify convert() to load/save weights in multiple passes.
Each pass, load and save partial weights (subset of all weight variables).
"""
# pylint: disable=g-line-too-long
import argparse
import pathlib
from pprint import pprint

import numpy as np

import checkpointing
import jax
from flax.training import train_state
import max_logging
import max_utils
from train import save_checkpoint
import torch
import sys
import os
import pyconfig
from layers import models, quantizations
import pathwaysutils

def permute_to_match_maxtext_rope(arr):
  evens = arr[..., ::2]
  odds = arr[..., 1::2]
  return jax.numpy.concatenate((evens, odds), axis=arr.ndim - 1)


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
    "llama3-405b": {
        "num_layers": 126,
        "num_heads": 128,
        "num_kv_heads": 8,
        "dims_per_head": 128,
        "vocab": 128256,
    },
}


def convert(base_model_path, maxtext_model_path, model_size, params_shardings):
  """
  Function to convert the checkpoint at base_model_path into Orbax checkpoint
  for MaxText and save at maxtext_model_path

  Attributes:
  base_model_path: checkpoint path
  maxtext_model_path: Path to save the MaxText checkpoint to
  model_size: llama3-8b to 405b.
  """
  """Convert model to maxtext."""
  model_params = MODEL_PARAMS_DICT[model_size]
  base_num_decoder_layers = model_params["num_layers"]
  base_num_query_heads = model_params["num_heads"]
  head_dim = model_params["dims_per_head"]
  base_num_kv_heads = model_params["num_kv_heads"]
  vocab_size = model_params["vocab"]

  print(f"Loading the base model from {base_model_path}")
  # Skip any hidden files for checkpoints
  ckpt_paths = sorted(pathlib.Path(base_model_path).glob("[!.]*.pth"))
  pytorch_vars = {}
  for i, ckpt_path in enumerate(ckpt_paths):
    print(f"Loading checkpoint {i+1} of {len(ckpt_paths)} ...")
    import psutil
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    pytorch_vars[int(ckpt_path.name.split(".", maxsplit=2)[1])] = checkpoint
    print("memory usage in GB: ", psutil.Process().memory_info().rss / (1024 * 1024))

  pytorch_vars = [pytorch_vars[i] for i in sorted(list(pytorch_vars.keys()))]

  if model_size[:6] == 'llama3':
    token_embedder = np.concatenate(
      [var["tok_embeddings.weight"].type(torch.float16).numpy() for var in pytorch_vars], axis=0
    )
  else:
    token_embedder = np.concatenate(
      [var["tok_embeddings.weight"].type(torch.float16).numpy() for var in pytorch_vars], axis=1
    )[:vocab_size, :]

  for var in pytorch_vars:
    del var["tok_embeddings.weight"]
  jax_weights = {
      "decoder": {
          "decoder_norm": {"scale": pytorch_vars[0]["norm.weight"].type(torch.float16).numpy()},
          "logits_dense": {
              "kernel": np.concatenate(
                  [var["output.weight"].type(torch.float16).numpy() for var in pytorch_vars], axis=0
              ).transpose()[:, :vocab_size]
          },
      },
      "token_embedder": {
          "embedding": token_embedder
      },
  }
  for i in range(base_num_decoder_layers):
    jax_weights["decoder"][f"layers_{i}"] = {
      "mlp": {
        "wi_0": {"kernel": None},
        "wi_1": {"kernel": None},
        "wo": {"kernel": None},
      },
      "pre_self_attention_layer_norm": {"scale": None},
      "post_self_attention_layer_norm": {"scale": None},
      "self_attention": {
        "query": {"kernel": None},
        "key": {"kernel": None},
        "value": {"kernel": None},
        "out": {"kernel": None},
      },
    }

  # llama3-405b kv weight is replicated within every two files.
  wkv_step = 1 if model_size != "llama3-405b" else 2

  for layer_idx in range(base_num_decoder_layers):
    print("layer idx: ", layer_idx)
    print("memory usage in GB: ", psutil.Process().memory_info().rss / (1024 * 1024))
    wq = np.concatenate(
        [var[f"layers.{layer_idx}.attention.wq.weight"].type(torch.float16).numpy() for var in pytorch_vars], axis=0
    ).transpose()
    wk = np.concatenate(
        [var[f"layers.{layer_idx}.attention.wk.weight"].type(torch.float16).numpy() for var in pytorch_vars[::wkv_step]], axis=0
    ).transpose()
    wv = np.concatenate(
        [var[f"layers.{layer_idx}.attention.wv.weight"].type(torch.float16).numpy() for var in pytorch_vars[::wkv_step]], axis=0
    ).transpose()

    wq = np.reshape(wq, [base_num_query_heads * head_dim, base_num_query_heads, head_dim])
    wk = np.reshape(wk, [base_num_query_heads * head_dim, base_num_kv_heads, head_dim])
    wv = np.reshape(wv, [base_num_query_heads * head_dim, base_num_kv_heads, head_dim])
    wq = permute_to_match_maxtext_rope(wq)
    wk = permute_to_match_maxtext_rope(wk)

    w_post = np.concatenate(
        [var[f"layers.{layer_idx}.attention.wo.weight"].type(torch.float16).numpy() for var in pytorch_vars],
        axis=1,
    )

    w_post = np.reshape(w_post, [base_num_query_heads * head_dim, base_num_query_heads, head_dim])

    jax_weights["decoder"][f"layers_{layer_idx}"]["self_attention"]["query"]["kernel"] = wq / np.sqrt(head_dim)

    jax_weights["decoder"][f"layers_{layer_idx}"]["self_attention"]["key"]["kernel"] = wk
    jax_weights["decoder"][f"layers_{layer_idx}"]["self_attention"]["value"]["kernel"] = wv

    # base_num_query_heads * head_dim, base_num_query_heads, head_dim =>
    # base_num_query_heads, head_dim, base_num_query_heads * head_dim
    jax_weights["decoder"][f"layers_{layer_idx}"]["self_attention"]["out"]["kernel"] = np.transpose(w_post, axes=(1,2,0))

    pre_self_attention_layernorm = pytorch_vars[0][f"layers.{layer_idx}.attention_norm.weight"].type(torch.float16).numpy()
    post_self_attention_layernorm = pytorch_vars[0][f"layers.{layer_idx}.ffn_norm.weight"].type(torch.float16).numpy()
    jax_weights["decoder"][f"layers_{layer_idx}"]["pre_self_attention_layer_norm"]["scale"] = pre_self_attention_layernorm
    jax_weights["decoder"][f"layers_{layer_idx}"]["post_self_attention_layer_norm"]["scale"] = post_self_attention_layernorm


    wi_0 = np.concatenate(
        [var[f"layers.{layer_idx}.feed_forward.w1.weight"].type(torch.float16).numpy() for var in pytorch_vars], axis=0
    ).transpose()
    wi_1 = np.concatenate(
        [var[f"layers.{layer_idx}.feed_forward.w3.weight"].type(torch.float16).numpy() for var in pytorch_vars], axis=0
    ).transpose()
    wo = np.concatenate(
        [var[f"layers.{layer_idx}.feed_forward.w2.weight"].type(torch.float16).numpy() for var in pytorch_vars], axis=1
    ).transpose()
    jax_weights["decoder"][f"layers_{layer_idx}"]["mlp"]["wi_0"]["kernel"] = wi_0
    jax_weights["decoder"][f"layers_{layer_idx}"]["mlp"]["wi_1"]["kernel"] = wi_1
    jax_weights["decoder"][f"layers_{layer_idx}"]["mlp"]["wo"]["kernel"] = wo

    for var in pytorch_vars:
      del var[f"layers.{layer_idx}.attention.wq.weight"]
      del var[f"layers.{layer_idx}.attention.wk.weight"]
      del var[f"layers.{layer_idx}.attention.wv.weight"]
      del var[f"layers.{layer_idx}.attention.wo.weight"]
      del var[f"layers.{layer_idx}.feed_forward.w1.weight"]
      del var[f"layers.{layer_idx}.feed_forward.w2.weight"]
      del var[f"layers.{layer_idx}.feed_forward.w3.weight"]


  # Shard weights in the same way as maxengine will load them
  def checkpoint_device_put(arr, sharding):
    return jax.device_put(arr, sharding)

  jax_weights = jax.tree.map(checkpoint_device_put, jax_weights, params_shardings)

  step_number_to_save_new_ckpt = 0
  checkpoint_manager = checkpointing.create_orbax_checkpoint_manager(
      checkpoint_dir=maxtext_model_path,
      enable_checkpointing=True,
      use_async=False,
      save_interval_steps=1,
      use_ocdbt=False,
      use_zarr3=False,
  )

  state_new = train_state.TrainState(
      step=0, apply_fn=None, params={"params": jax_weights}, tx=None, opt_state={}  # type: ignore
  )

  if checkpoint_manager is not None:
    if save_checkpoint(checkpoint_manager, step_number_to_save_new_ckpt, state_new):
      max_logging.log(f"saved a checkpoint at step {step_number_to_save_new_ckpt}")
    # Upon preemption, exit when and only when all ongoing saves are complete.
    if checkpoint_manager.reached_preemption(0):
      checkpoint_manager.wait_until_finished()
      sys.exit()


if __name__ == "__main__":
  # Do not query VM metadata server for TPU variables.
  os.environ["TPU_SKIP_MDS_QUERY"] = "1"

  parser = argparse.ArgumentParser()
  parser.add_argument("--base-model-path", type=str, required=True)
  parser.add_argument("--maxtext-model-path", type=str, required=True)
  parser.add_argument("--model-size", type=str, required=True, choices=MODEL_PARAMS_DICT.keys())
  parser.add_argument("maxtextargs", nargs="+")
  args = parser.parse_args()

  pyconfig.initialize(["python"] + args.maxtextargs)
  config = pyconfig.config

  mesh = jax.sharding.Mesh(max_utils.create_device_mesh(config), config.mesh_axes)
  print("Mesh:")
  pprint(mesh)

  # Create model in the same way as maxengine.MaxEngine will and get shardings for state.
  model = models.Transformer(config, mesh=mesh, quant=quantizations.configure_quantization(config))
  _, _, train_state_shardings = max_utils.get_abstract_state(model, None, config, jax.random.PRNGKey(0), mesh, False)
  params_shardings = train_state_shardings.params['params']
  print(f"Params shardings:")
  pprint(params_shardings)

  convert(args.base_model_path, args.maxtext_model_path, args.model_size, params_shardings)
