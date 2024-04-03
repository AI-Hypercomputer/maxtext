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
"""Convert weights from a paxml gpt3 model to a MaxText one.

Test cmd for gpt3-52k:
python MaxText/convert_gpt3_ckpt_from_paxml.py \
  --paxml-ckpt-path=gs://maxtext-gpt3/ckpt_test/paxml/checkpoints/checkpoint_00000000/state \
  --maxtext-model-name=gpt3-52k \
  --run-name=$RUN_NAME \
  --base-output-directory=$BASE_OUTPUT_DIR

True cmd for gpt3-175b:

The script is memory demanding, requires at least 250 GiB in cpu and cumulative TPU memory of all devices should be
  above ~4.2 TiB (175 billion param * 4 byte/param * 3 (model var and 2 opt momentums) * 2 copies in converting) 

python MaxText/convert_gpt3_ckpt_from_paxml.py \
  --paxml-ckpt-path=gs://mlperf-llm-public2/gpt3_spmd1x64x24_tpuv4-3072_v84_20221101/checkpoints/checkpoint_00004000 \
  --maxtext-model-name=gpt3-175b \
  --run-name=$RUN_NAME \
  --base-output-directory=$BASE_OUTPUT_DIR
"""
import max_utils
import optimizers
import pyconfig
import os
from jax import random
from jax.sharding import Mesh
from layers.models import Transformer
from layers import quantizations
import checkpointing

import numpy as np
import tensorstore as ts

import sys
import jax
import gc
import max_logging
from psutil import Process
from train import save_checkpoint
import argparse

def fmt_size(num_bytes: int) -> str:
  assert num_bytes > 0
  for unit in ["B", "KiB", "MiB", "GiB"]:
    if num_bytes < 1024.0:
      break
    num_bytes /= 1024.0
  return f"{num_bytes:.2f} {unit}"

def check_memory():
  """print out cpu/tpu memory."""
  cpu_bytes = Process().memory_info().rss
  max_logging.log(f"cpu memory: {fmt_size(cpu_bytes)}")
  for d in jax.local_devices():
    stats = d.memory_stats()
    used = stats['bytes_in_use']
    limit = stats['bytes_limit']
    max_logging.log(f"tpu memory: Using {fmt_size(used)} / {fmt_size(limit)} ({used/limit:%}) on {d}")


def convert(paxml_ckpt_path, maxtext_model_name, base_output_directory, run_name):
  """convert ckpt."""

  base_args = [
    '', 'MaxText/configs/base.yml',  # base arg
    'per_device_batch_size=1',
    'ici_fsdp_parallelism=-1', 'ici_tensor_parallelism=1',
    f'model_name={maxtext_model_name}',
    f'run_name={run_name}', f'base_output_directory={base_output_directory}',
    'checkpoint_period=1',
    'async_checkpointing=false',
  ]
  pyconfig.initialize(base_args)
  cfg = pyconfig.config
  init_rng, _ = random.split(random.PRNGKey(cfg.init_weights_seed), 2)
  devices_array = max_utils.create_device_mesh(cfg)
  mesh = Mesh(devices_array, cfg.mesh_axes)

  quant = quantizations.configure_quantization(cfg)
  model = Transformer(cfg, mesh, quant=quant)
  learning_rate_schedule = max_utils.create_learning_rate_schedule(cfg)
  tx = optimizers.get_optimizer(cfg, learning_rate_schedule)

  checkpoint_manager = checkpointing.create_orbax_checkpoint_manager(
    cfg.checkpoint_dir,
    cfg.enable_checkpointing,
    cfg.async_checkpointing,
    cfg.checkpoint_period,
  )

  state, _, _ = max_utils.setup_training_state(model, None, tx, cfg, init_rng, mesh, checkpoint_manager)
  max_logging.log("start")
  check_memory()

  # maxtext keystr: (paxml keystr, transform_fn)
  keystr_map = {
    "['token_embedder']['embedding']": (".params.lm.softmax.logits_ffn.linear.w", lambda x: x.T),
    "['decoder']['position_embedder']['embedding']": (".params.lm.position_emb.emb_var", None),
    "['decoder']['layers']['pre_self_attention_norm']['scale']": (".params.lm.transformer.repeat.sub.x_layers_0.layer_norm.scale", lambda x: np.moveaxis(x, 0, cfg.param_scan_axis)),
    "['decoder']['layers']['pre_self_attention_norm']['bias']": (".params.lm.transformer.repeat.sub.x_layers_0.layer_norm.bias", lambda x: np.moveaxis(x, 0, cfg.param_scan_axis)),
    "['decoder']['layers']['self_attention']['query']['kernel']": (".params.lm.transformer.repeat.sub.x_layers_0.self_attention.combined_qkv.w", lambda x: np.moveaxis(x[:,0], 0, cfg.param_scan_axis)),
    "['decoder']['layers']['self_attention']['query']['bias']": (".params.lm.transformer.repeat.sub.x_layers_0.self_attention.combined_qkv.b", lambda x: np.moveaxis(x[:,0], 0, cfg.param_scan_axis)),
    "['decoder']['layers']['self_attention']['key']['kernel']": (".params.lm.transformer.repeat.sub.x_layers_0.self_attention.combined_qkv.w", lambda x: np.moveaxis(x[:,1], 0, cfg.param_scan_axis)),
    "['decoder']['layers']['self_attention']['key']['bias']": (".params.lm.transformer.repeat.sub.x_layers_0.self_attention.combined_qkv.b", lambda x: np.moveaxis(x[:,1], 0, cfg.param_scan_axis)),
    "['decoder']['layers']['self_attention']['value']['kernel']": (".params.lm.transformer.repeat.sub.x_layers_0.self_attention.combined_qkv.w", lambda x: np.moveaxis(x[:,2], 0, cfg.param_scan_axis)),
    "['decoder']['layers']['self_attention']['value']['bias']": (".params.lm.transformer.repeat.sub.x_layers_0.self_attention.combined_qkv.b", lambda x: np.moveaxis(x[:,2], 0, cfg.param_scan_axis)),
    "['decoder']['layers']['self_attention']['qkv_proj']['kernel']": (".params.lm.transformer.repeat.sub.x_layers_0.self_attention.combined_qkv.w", lambda x: np.moveaxis(x, [2, 0], [0, cfg.param_scan_axis])),
    "['decoder']['layers']['self_attention']['qkv_proj']['bias']": (".params.lm.transformer.repeat.sub.x_layers_0.self_attention.combined_qkv.b", lambda x: np.moveaxis(x, 0, cfg.param_scan_axis)),
    "['decoder']['layers']['self_attention']['out']['kernel']": (".params.lm.transformer.repeat.sub.x_layers_0.self_attention.post.w", lambda x: np.moveaxis(x, [0, 1], [cfg.param_scan_axis, -1])),
    "['decoder']['layers']['self_attention']['out']['bias']": (".params.lm.transformer.repeat.sub.x_layers_0.self_attention.post.b", lambda x: np.moveaxis(x, 0, cfg.param_scan_axis)),
    "['decoder']['layers']['mlp']['mlp_layer_norm']['scale']": (".params.lm.transformer.repeat.sub.x_layers_0.ff_layer.layer_norm.scale", lambda x: np.moveaxis(x, 0, cfg.param_scan_axis)),
    "['decoder']['layers']['mlp']['mlp_layer_norm']['bias']": (".params.lm.transformer.repeat.sub.x_layers_0.ff_layer.layer_norm.bias", lambda x: np.moveaxis(x, 0, cfg.param_scan_axis)),
    "['decoder']['layers']['mlp']['wi']['kernel']": (".params.lm.transformer.repeat.sub.x_layers_0.ff_layer.ffn_layer1.linear.w", lambda x: np.moveaxis(x, 0, cfg.param_scan_axis)),
    "['decoder']['layers']['mlp']['wi']['bias']": (".params.lm.transformer.repeat.sub.x_layers_0.ff_layer.ffn_layer1.bias.b", lambda x: np.moveaxis(x, 0, cfg.param_scan_axis)),
    "['decoder']['layers']['mlp']['wo']['kernel']": (".params.lm.transformer.repeat.sub.x_layers_0.ff_layer.ffn_layer2.linear.w", lambda x: np.moveaxis(x, 0, cfg.param_scan_axis)),
    "['decoder']['layers']['mlp']['wo']['bias']": (".params.lm.transformer.repeat.sub.x_layers_0.ff_layer.ffn_layer2.bias.b", lambda x: np.moveaxis(x, 0, cfg.param_scan_axis)),
    "['decoder']['decoder_norm']['scale']": (".params.lm.final_ln.scale", lambda x: x.T),
    "['decoder']['decoder_norm']['bias']": (".params.lm.final_ln.bias", None),
  }

  state_map = {
    ".step": ("step", None),
    ".opt_state.count": ("opt_states_0.no_prefix_0.count", None),
  }

  def get_layer_prefix(keystr_pax):
    # different path format between decoder_layer variable
    if "x_layers_0" in keystr_pax:
      # string format for all variables in scanned decoder layer
      prefix_pax_opt_state = f"p#{cfg.base_num_decoder_layers}#i-1_2"
    else:
      prefix_pax_opt_state = "no_prefix_2"
    return prefix_pax_opt_state

  for keystr_maxtext, (keystr_pax, transform_fn) in keystr_map.items():
    # model variable
    state_map[f".params['params']{keystr_maxtext}"] = (f"mdl_vars{keystr_pax}", transform_fn)
    prefix_pax_opt_state = get_layer_prefix(keystr_pax)
    # first momentum in optimizer state
    state_map[f".opt_state.mu['params']{keystr_maxtext}"] = (f"opt_states_0.{prefix_pax_opt_state}.m{keystr_pax}", transform_fn)
    # second momentum in optimizer state
    state_map[f".opt_state.nu['params']{keystr_maxtext}"] = (f"opt_states_0.{prefix_pax_opt_state}.v{keystr_pax}", transform_fn)

  def verify_fn(key_path, _):
    keystr = jax.tree_util.keystr(key_path)
    assert keystr in state_map, f"{keystr} not found"

  jax.tree_util.tree_map_with_path(verify_fn, state)

  memory_metrics = {'max_cpu_bytes': 0}

  bucket_name, paxml_ckpt_prefix = paxml_ckpt_path[len("gs://"):].split('/', 1)

  def map_fn(key_path, value):
    key_path_str = jax.tree_util.keystr(key_path)
    file_path, transform_fn = state_map[key_path_str]
    full_path = os.path.join(paxml_ckpt_prefix, file_path)
    spec = {'driver': 'zarr', 'metadata_key': '.zarray', 'kvstore': {}}
    spec['kvstore'] = {
      'bucket': bucket_name,
      'driver': 'gcs',
      'path': full_path,
    }

    arr = ts.open(ts.Spec(spec), open=True).result().read().result()
    if transform_fn is not None:
      arr = transform_fn(arr)

    assert value.shape == arr.shape, f"{key_path}, {value.shape}, {arr.shape}"
    shape = value.shape
    sharding = value.sharding
    result = jax.make_array_from_single_device_arrays(
      shape,
      sharding,
      [jax.device_put(np.array(arr[index]), d)
      for d, index in sharding.addressable_devices_indices_map(shape).items()],
    )

    # log peak cpu memory
    cpu_bytes = Process().memory_info().rss
    memory_metrics["max_cpu_bytes"] = max(cpu_bytes, memory_metrics["max_cpu_bytes"])

    # collect cpu memory back asap
    arr = None
    del arr
    gc.collect()
    max_logging.log(f"{key_path_str} finished")
    check_memory()
    return result

  converted_state = jax.tree_util.tree_map_with_path(map_fn, state)
  max_logging.log("converted state finished")
  check_memory()

  if save_checkpoint(checkpoint_manager, converted_state.step, converted_state):
    max_logging.log(f"saved a checkpoint at step {converted_state.step}")
  # Upon preemption, exit when and only when all ongoing saves are complete.
  if checkpoint_manager.reached_preemption(converted_state.step):
    checkpoint_manager.wait_until_finished()
    sys.exit()

  max_logging.log(f"Peak cpu memory in a single process: {fmt_size(memory_metrics['max_cpu_bytes'])}")
  max_logging.log("checkpoint converted and saved successfully.")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--paxml-ckpt-path',
                      type=str,
                      default="gs://mlperf-llm-public2/gpt3_spmd1x64x24_tpuv4-3072_v84_20221101/checkpoints/checkpoint_00004000",
                      required=True)
  parser.add_argument('--maxtext-model-name', choices=['gpt3-175b', 'gpt3-52k'],  type=str, required=True)
  parser.add_argument('--base-output-directory', type=str, required=True)
  parser.add_argument('--run-name', type=str, required=True)

  args = parser.parse_args()
  if not args.paxml_ckpt_path.startswith("gs://"):
    raise ValueError("--paxml-ckpt-path should be a gcs path starting with gs://")

  convert(args.paxml_ckpt_path, args.maxtext_model_name, args.base_output_directory, args.run_name)
