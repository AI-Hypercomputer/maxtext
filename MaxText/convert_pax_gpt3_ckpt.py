from absl import app

import max_utils
import maxtext_utils
import pyconfig
import os
from typing import Sequence
from jax import random
from jax.sharding import Mesh
from layers.models import Transformer
import checkpointing

import numpy as np
import tensorstore as ts

from flax.training import train_state
import jax.numpy as jnp
import sys
import jax
import gc
import max_logging
from psutil import Process
import humanize
import functools
from paxml import trainer_lib
from paxml.tasks.lm.params.c4 import C4SpmdGpt3AdamOrgHP
from praxis import py_utils
import optax

NestedMap = py_utils.NestedMap

PEAK_CPU_MEMORY = 0

fmt_size = functools.partial(humanize.naturalsize, binary=True)

def print_memory():
  max_logging.log(f"cpu memory: {fmt_size(Process().memory_info().rss)}")
  for d in jax.local_devices():
    stats = d.memory_stats()
    used = stats['bytes_in_use']
    limit = stats['bytes_limit']
    max_logging.log(f"tpu memory: Using {fmt_size(used)} / {fmt_size(limit)} ({used/limit:%}) on {d}")


def get_ts_array(filename):
  spec = {'driver': 'zarr', 'metadata_key': '.zarray', 'kvstore': {}}
  spec['kvstore'] = {
    'bucket': 'mlperf-llm-public2',
    'driver': 'gcs',
    'path': filename,
  }

  return ts.open(ts.Spec(spec), open=True).result()


def _get_numpy_array(filename, layer_idx):
  t = get_ts_array(filename)
  if layer_idx is None:
    t_v = t.read().result()
  else:
    t_v = t[layer_idx:layer_idx + 1].read().result()
  return t_v


def get_array(filename, layer_idx=None):
  return _get_numpy_array(filename, layer_idx)

MLPerf_GPT3_175B = {
  'base_num_decoder_layers': 96,
  'base_emb_dim': 12288,
  'base_num_heads': 96,
  'base_mlp_dim': 49152,
  'head_dim': 128,
  'vocab_size': 50304,
  'max_target_length': 2048,
  'mlp_activations': ['gelu'],
  'max_trainable_pe_max_seq_len': 16384,
}

base_args = [
  '', 'configs/base.yml',  # base arg
  f'base_emb_dim={MLPerf_GPT3_175B["base_emb_dim"]}',
  f'base_num_heads={MLPerf_GPT3_175B["base_num_heads"]}',
  f'base_mlp_dim={MLPerf_GPT3_175B["base_mlp_dim"]}',
  f'base_num_decoder_layers={MLPerf_GPT3_175B["base_num_decoder_layers"]}',
  f'head_dim={MLPerf_GPT3_175B["base_emb_dim"] // MLPerf_GPT3_175B["base_num_heads"]}',
  f'vocab_size={MLPerf_GPT3_175B["vocab_size"]}',
  f'max_target_length={MLPerf_GPT3_175B["max_target_length"]}',
  f'max_trainable_pe_max_seq_len={MLPerf_GPT3_175B["max_trainable_pe_max_seq_len"]}',
  'per_device_batch_size=0.25',
  'ici_fsdp_parallelism=-1',
  'ici_tensor_parallelism=4',
  'attention=mha',
  'steps=5', 'run_name=convergence_test', 'base_output_directory=gs://lizhiyu-multipods/lizhiyu/colab_adamw',
  'dtype=float32',
  'save_period=1000',
  'async_checkpointing=false',
  # added keys
  'embed_lookup_style=matmul',
  'use_position_embedding=True',
  'use_bias_linear=True',
  'use_pre_norm_mlp=True',
  'apply_padding_mask_mlp=True',
  'add_skip_connection_mlp=True',
  'use_bias_layer_norm=True',
  'use_mean_center_layer_norm=True',
  'reductions_in_fp32_layer_norm=False',
  'epsilon_layer_norm=1.e-5',
  'use_rotary_position_emb=False',
  'use_qk_norm=False',
  'logits_norm=False',
  'stable_cross_entropy_loss=False',
  'query_scale_style=post',
  'skip_connection_style_decoder=GPT3',
  ]


def main(args: Sequence[str]):
  pyconfig.initialize(base_args, mlp_activations=MLPerf_GPT3_175B["mlp_activations"])
  cfg = pyconfig.config
  init_rng, nextrng = random.split(random.PRNGKey(cfg.init_weights_seed), 2)
  devices_array = max_utils.create_device_mesh(cfg)
  mesh = Mesh(devices_array, cfg.mesh_axes)

  model = Transformer(config=cfg, mesh=mesh)
  learning_rate_schedule = max_utils.create_learning_rate_schedule(cfg)
  # tx = maxtext_utils.get_optimizer(cfg, learning_rate_schedule)
  tx = optax.adamw(
    learning_rate_schedule,
    b1=cfg.adam_b1,
    b2=cfg.adam_b2,
    eps=cfg.adam_eps,
    eps_root=cfg.adam_eps_root,
    weight_decay=cfg.adam_weight_decay,
  )

  checkpoint_manager = checkpointing.create_orbax_checkpoint_manager(
    cfg.checkpoint_dir,
    cfg.enable_checkpointing,
    cfg.async_checkpointing,
    cfg.save_period,
  )

  state, state_mesh_annotations = max_utils.setup_training_state(model, tx, cfg, init_rng, mesh, checkpoint_manager)
  max_logging.log("start")
  print_memory()

  MAPS = {
    ".step": ("step", None),
    ".opt_state[0].count": ("opt_states_0.no_prefix_0.count", None),
    ".opt_state[1].count": ("opt_states_0.no_prefix_1.count", None),
    ".opt_state[2].count": ("opt_states_0.no_prefix_2.count", None),
    ".params['token_embedder']['embedding']": ("mdl_vars.params.lm.softmax.logits_ffn.linear.w", lambda x: x.T),
    ".params['position_embedder']['embedding']": ("mdl_vars.params.lm.position_emb.emb_var", None),
    ".params['decoder']['decoder']['pre_self_attention_norm']['scale']": ("mdl_vars.params.lm.transformer.repeat.sub.x_layers_0.layer_norm.scale", lambda x: np.moveaxis(x, 0, cfg.param_scan_axis)),
    ".params['decoder']['decoder']['pre_self_attention_norm']['bias']": ("mdl_vars.params.lm.transformer.repeat.sub.x_layers_0.layer_norm.bias", lambda x: np.moveaxis(x, 0, cfg.param_scan_axis)),
    ".params['decoder']['decoder']['self_attention']['query']['kernel']": ("mdl_vars.params.lm.transformer.repeat.sub.x_layers_0.self_attention.combined_qkv.w", lambda x: np.moveaxis(x[:,0], 0, cfg.param_scan_axis)),
    ".params['decoder']['decoder']['self_attention']['query']['bias']": ("mdl_vars.params.lm.transformer.repeat.sub.x_layers_0.self_attention.combined_qkv.b", lambda x: np.moveaxis(x[:,0], 0, cfg.param_scan_axis)),
    ".params['decoder']['decoder']['self_attention']['key']['kernel']": ("mdl_vars.params.lm.transformer.repeat.sub.x_layers_0.self_attention.combined_qkv.w", lambda x: np.moveaxis(x[:,1], 0, cfg.param_scan_axis)),
    ".params['decoder']['decoder']['self_attention']['key']['bias']": ("mdl_vars.params.lm.transformer.repeat.sub.x_layers_0.self_attention.combined_qkv.b", lambda x: np.moveaxis(x[:,1], 0, cfg.param_scan_axis)),
    ".params['decoder']['decoder']['self_attention']['value']['kernel']": ("mdl_vars.params.lm.transformer.repeat.sub.x_layers_0.self_attention.combined_qkv.w", lambda x: np.moveaxis(x[:,2], 0, cfg.param_scan_axis)),
    ".params['decoder']['decoder']['self_attention']['value']['bias']": ("mdl_vars.params.lm.transformer.repeat.sub.x_layers_0.self_attention.combined_qkv.b", lambda x: np.moveaxis(x[:,2], 0, cfg.param_scan_axis)),
    ".params['decoder']['decoder']['self_attention']['out']['kernel']": ("mdl_vars.params.lm.transformer.repeat.sub.x_layers_0.self_attention.post.w", lambda x: np.transpose(x, (2, 0, 3, 1))),
    ".params['decoder']['decoder']['self_attention']['out']['bias']": ("mdl_vars.params.lm.transformer.repeat.sub.x_layers_0.self_attention.post.b", lambda x: np.moveaxis(x, 0, cfg.param_scan_axis)),
    ".params['decoder']['decoder']['mlp']['mlp_layer_norm']['scale']": ("mdl_vars.params.lm.transformer.repeat.sub.x_layers_0.ff_layer.layer_norm.scale", lambda x: np.moveaxis(x, 0, cfg.param_scan_axis)),
    ".params['decoder']['decoder']['mlp']['mlp_layer_norm']['bias']": ("mdl_vars.params.lm.transformer.repeat.sub.x_layers_0.ff_layer.layer_norm.bias", lambda x: np.moveaxis(x, 0, cfg.param_scan_axis)),
    ".params['decoder']['decoder']['mlp']['wi']['kernel']": ("mdl_vars.params.lm.transformer.repeat.sub.x_layers_0.ff_layer.ffn_layer1.linear.w", lambda x: np.moveaxis(x, 0, cfg.param_scan_axis)),
    ".params['decoder']['decoder']['mlp']['wi']['bias']": ("mdl_vars.params.lm.transformer.repeat.sub.x_layers_0.ff_layer.ffn_layer1.bias.b", lambda x: np.moveaxis(x, 0, cfg.param_scan_axis)),
    ".params['decoder']['decoder']['mlp']['wo']['kernel']": ("mdl_vars.params.lm.transformer.repeat.sub.x_layers_0.ff_layer.ffn_layer2.linear.w", lambda x: np.moveaxis(x, 0, cfg.param_scan_axis)),
    ".params['decoder']['decoder']['mlp']['wo']['bias']": ("mdl_vars.params.lm.transformer.repeat.sub.x_layers_0.ff_layer.ffn_layer2.bias.b", lambda x: np.moveaxis(x, 0, cfg.param_scan_axis)),
    ".params['decoder']['decoder_norm']['scale']": ("mdl_vars.params.lm.final_ln.scale", lambda x: x.T),
    ".params['decoder']['decoder_norm']['bias']": ("mdl_vars.params.lm.final_ln.bias", None),
    ".opt_state[0].mu['token_embedder']['embedding']": ("opt_states_0.no_prefix_2.m.params.lm.softmax.logits_ffn.linear.w", lambda x: x.T),
    ".opt_state[0].mu['position_embedder']['embedding']": ("opt_states_0.no_prefix_2.m.params.lm.position_emb.emb_var", None),
    ".opt_state[0].mu['decoder']['decoder']['pre_self_attention_norm']['scale']": (f"opt_states_0.p#{cfg.base_num_decoder_layers}#i-1_2.m.params.lm.transformer.repeat.sub.x_layers_0.layer_norm.scale", lambda x: np.moveaxis(x, 0, cfg.param_scan_axis)),
    ".opt_state[0].mu['decoder']['decoder']['pre_self_attention_norm']['bias']": (f"opt_states_0.p#{cfg.base_num_decoder_layers}#i-1_2.m.params.lm.transformer.repeat.sub.x_layers_0.layer_norm.bias", lambda x: np.moveaxis(x, 0, cfg.param_scan_axis)),
    ".opt_state[0].mu['decoder']['decoder']['self_attention']['query']['kernel']": (f"opt_states_0.p#{cfg.base_num_decoder_layers}#i-1_2.m.params.lm.transformer.repeat.sub.x_layers_0.self_attention.combined_qkv.w", lambda x: np.moveaxis(x[:,0], 0, cfg.param_scan_axis)),
    ".opt_state[0].mu['decoder']['decoder']['self_attention']['query']['bias']": (f"opt_states_0.p#{cfg.base_num_decoder_layers}#i-1_2.m.params.lm.transformer.repeat.sub.x_layers_0.self_attention.combined_qkv.b", lambda x: np.moveaxis(x[:,0], 0, cfg.param_scan_axis)),
    ".opt_state[0].mu['decoder']['decoder']['self_attention']['key']['kernel']": (f"opt_states_0.p#{cfg.base_num_decoder_layers}#i-1_2.m.params.lm.transformer.repeat.sub.x_layers_0.self_attention.combined_qkv.w", lambda x: np.moveaxis(x[:,1], 0, cfg.param_scan_axis)),
    ".opt_state[0].mu['decoder']['decoder']['self_attention']['key']['bias']": (f"opt_states_0.p#{cfg.base_num_decoder_layers}#i-1_2.m.params.lm.transformer.repeat.sub.x_layers_0.self_attention.combined_qkv.b", lambda x: np.moveaxis(x[:,1], 0, cfg.param_scan_axis)),
    ".opt_state[0].mu['decoder']['decoder']['self_attention']['value']['kernel']": (f"opt_states_0.p#{cfg.base_num_decoder_layers}#i-1_2.m.params.lm.transformer.repeat.sub.x_layers_0.self_attention.combined_qkv.w", lambda x: np.moveaxis(x[:,2], 0, cfg.param_scan_axis)),
    ".opt_state[0].mu['decoder']['decoder']['self_attention']['value']['bias']": (f"opt_states_0.p#{cfg.base_num_decoder_layers}#i-1_2.m.params.lm.transformer.repeat.sub.x_layers_0.self_attention.combined_qkv.b", lambda x: np.moveaxis(x[:,2], 0, cfg.param_scan_axis)),
    ".opt_state[0].mu['decoder']['decoder']['self_attention']['out']['kernel']": (f"opt_states_0.p#{cfg.base_num_decoder_layers}#i-1_2.m.params.lm.transformer.repeat.sub.x_layers_0.self_attention.post.w", lambda x: np.transpose(x, (2, 0, 3, 1))),
    ".opt_state[0].mu['decoder']['decoder']['self_attention']['out']['bias']": (f"opt_states_0.p#{cfg.base_num_decoder_layers}#i-1_2.m.params.lm.transformer.repeat.sub.x_layers_0.self_attention.post.b", lambda x: np.moveaxis(x, 0, cfg.param_scan_axis)),
    ".opt_state[0].mu['decoder']['decoder']['mlp']['mlp_layer_norm']['scale']": (f"opt_states_0.p#{cfg.base_num_decoder_layers}#i-1_2.m.params.lm.transformer.repeat.sub.x_layers_0.ff_layer.layer_norm.scale", lambda x: np.moveaxis(x, 0, cfg.param_scan_axis)),
    ".opt_state[0].mu['decoder']['decoder']['mlp']['mlp_layer_norm']['bias']": (f"opt_states_0.p#{cfg.base_num_decoder_layers}#i-1_2.m.params.lm.transformer.repeat.sub.x_layers_0.ff_layer.layer_norm.bias", lambda x: np.moveaxis(x, 0, cfg.param_scan_axis)),
    ".opt_state[0].mu['decoder']['decoder']['mlp']['wi']['kernel']": (f"opt_states_0.p#{cfg.base_num_decoder_layers}#i-1_2.m.params.lm.transformer.repeat.sub.x_layers_0.ff_layer.ffn_layer1.linear.w", lambda x: np.moveaxis(x, 0, cfg.param_scan_axis)),
    ".opt_state[0].mu['decoder']['decoder']['mlp']['wi']['bias']": (f"opt_states_0.p#{cfg.base_num_decoder_layers}#i-1_2.m.params.lm.transformer.repeat.sub.x_layers_0.ff_layer.ffn_layer1.bias.b", lambda x: np.moveaxis(x, 0, cfg.param_scan_axis)),
    ".opt_state[0].mu['decoder']['decoder']['mlp']['wo']['kernel']": (f"opt_states_0.p#{cfg.base_num_decoder_layers}#i-1_2.m.params.lm.transformer.repeat.sub.x_layers_0.ff_layer.ffn_layer2.linear.w", lambda x: np.moveaxis(x, 0, cfg.param_scan_axis)),
    ".opt_state[0].mu['decoder']['decoder']['mlp']['wo']['bias']": (f"opt_states_0.p#{cfg.base_num_decoder_layers}#i-1_2.m.params.lm.transformer.repeat.sub.x_layers_0.ff_layer.ffn_layer2.bias.b", lambda x: np.moveaxis(x, 0, cfg.param_scan_axis)),
    ".opt_state[0].mu['decoder']['decoder_norm']['scale']": ("opt_states_0.no_prefix_2.m.params.lm.final_ln.scale", lambda x: x.T),
    ".opt_state[0].mu['decoder']['decoder_norm']['bias']": ("opt_states_0.no_prefix_2.m.params.lm.final_ln.bias", None),
    ".opt_state[0].nu['token_embedder']['embedding']": ("opt_states_0.no_prefix_2.v.params.lm.softmax.logits_ffn.linear.w", lambda x: x.T),
    ".opt_state[0].nu['position_embedder']['embedding']": ("opt_states_0.no_prefix_2.v.params.lm.position_emb.emb_var", None),
    ".opt_state[0].nu['decoder']['decoder']['pre_self_attention_norm']['scale']": (f"opt_states_0.p#{cfg.base_num_decoder_layers}#i-1_2.v.params.lm.transformer.repeat.sub.x_layers_0.layer_norm.scale", lambda x: np.moveaxis(x, 0, cfg.param_scan_axis)),
    ".opt_state[0].nu['decoder']['decoder']['pre_self_attention_norm']['bias']": (f"opt_states_0.p#{cfg.base_num_decoder_layers}#i-1_2.v.params.lm.transformer.repeat.sub.x_layers_0.layer_norm.bias", lambda x: np.moveaxis(x, 0, cfg.param_scan_axis)),
    ".opt_state[0].nu['decoder']['decoder']['self_attention']['query']['kernel']": (f"opt_states_0.p#{cfg.base_num_decoder_layers}#i-1_2.v.params.lm.transformer.repeat.sub.x_layers_0.self_attention.combined_qkv.w", lambda x: np.moveaxis(x[:,0], 0, cfg.param_scan_axis)),
    ".opt_state[0].nu['decoder']['decoder']['self_attention']['query']['bias']": (f"opt_states_0.p#{cfg.base_num_decoder_layers}#i-1_2.v.params.lm.transformer.repeat.sub.x_layers_0.self_attention.combined_qkv.b", lambda x: np.moveaxis(x[:,0], 0, cfg.param_scan_axis)),
    ".opt_state[0].nu['decoder']['decoder']['self_attention']['key']['kernel']": (f"opt_states_0.p#{cfg.base_num_decoder_layers}#i-1_2.v.params.lm.transformer.repeat.sub.x_layers_0.self_attention.combined_qkv.w", lambda x: np.moveaxis(x[:,1], 0, cfg.param_scan_axis)),
    ".opt_state[0].nu['decoder']['decoder']['self_attention']['key']['bias']": (f"opt_states_0.p#{cfg.base_num_decoder_layers}#i-1_2.v.params.lm.transformer.repeat.sub.x_layers_0.self_attention.combined_qkv.b", lambda x: np.moveaxis(x[:,1], 0, cfg.param_scan_axis)),
    ".opt_state[0].nu['decoder']['decoder']['self_attention']['value']['kernel']": (f"opt_states_0.p#{cfg.base_num_decoder_layers}#i-1_2.v.params.lm.transformer.repeat.sub.x_layers_0.self_attention.combined_qkv.w", lambda x: np.moveaxis(x[:,2], 0, cfg.param_scan_axis)),
    ".opt_state[0].nu['decoder']['decoder']['self_attention']['value']['bias']": (f"opt_states_0.p#{cfg.base_num_decoder_layers}#i-1_2.v.params.lm.transformer.repeat.sub.x_layers_0.self_attention.combined_qkv.b", lambda x: np.moveaxis(x[:,2], 0, cfg.param_scan_axis)),
    ".opt_state[0].nu['decoder']['decoder']['self_attention']['out']['kernel']": (f"opt_states_0.p#{cfg.base_num_decoder_layers}#i-1_2.v.params.lm.transformer.repeat.sub.x_layers_0.self_attention.post.w", lambda x: np.transpose(x, (2, 0, 3, 1))),
    ".opt_state[0].nu['decoder']['decoder']['self_attention']['out']['bias']": (f"opt_states_0.p#{cfg.base_num_decoder_layers}#i-1_2.v.params.lm.transformer.repeat.sub.x_layers_0.self_attention.post.b", lambda x: np.moveaxis(x, 0, cfg.param_scan_axis)),
    ".opt_state[0].nu['decoder']['decoder']['mlp']['mlp_layer_norm']['scale']": (f"opt_states_0.p#{cfg.base_num_decoder_layers}#i-1_2.v.params.lm.transformer.repeat.sub.x_layers_0.ff_layer.layer_norm.scale", lambda x: np.moveaxis(x, 0, cfg.param_scan_axis)),
    ".opt_state[0].nu['decoder']['decoder']['mlp']['mlp_layer_norm']['bias']": (f"opt_states_0.p#{cfg.base_num_decoder_layers}#i-1_2.v.params.lm.transformer.repeat.sub.x_layers_0.ff_layer.layer_norm.bias", lambda x: np.moveaxis(x, 0, cfg.param_scan_axis)),
    ".opt_state[0].nu['decoder']['decoder']['mlp']['wi']['kernel']": (f"opt_states_0.p#{cfg.base_num_decoder_layers}#i-1_2.v.params.lm.transformer.repeat.sub.x_layers_0.ff_layer.ffn_layer1.linear.w", lambda x: np.moveaxis(x, 0, cfg.param_scan_axis)),
    ".opt_state[0].nu['decoder']['decoder']['mlp']['wi']['bias']": (f"opt_states_0.p#{cfg.base_num_decoder_layers}#i-1_2.v.params.lm.transformer.repeat.sub.x_layers_0.ff_layer.ffn_layer1.bias.b", lambda x: np.moveaxis(x, 0, cfg.param_scan_axis)),
    ".opt_state[0].nu['decoder']['decoder']['mlp']['wo']['kernel']": (f"opt_states_0.p#{cfg.base_num_decoder_layers}#i-1_2.v.params.lm.transformer.repeat.sub.x_layers_0.ff_layer.ffn_layer2.linear.w", lambda x: np.moveaxis(x, 0, cfg.param_scan_axis)),
    ".opt_state[0].nu['decoder']['decoder']['mlp']['wo']['bias']": (f"opt_states_0.p#{cfg.base_num_decoder_layers}#i-1_2.v.params.lm.transformer.repeat.sub.x_layers_0.ff_layer.ffn_layer2.bias.b", lambda x: np.moveaxis(x, 0, cfg.param_scan_axis)),
    ".opt_state[0].nu['decoder']['decoder_norm']['scale']": ("opt_states_0.no_prefix_2.v.params.lm.final_ln.scale", lambda x: x.T),
    ".opt_state[0].nu['decoder']['decoder_norm']['bias']": ("opt_states_0.no_prefix_2.v.params.lm.final_ln.bias", None),
  }

  def verify_fn(key_path, value, prefix='gpt3_spmd1x64x24_tpuv4-3072_v84_20221101/checkpoints/checkpoint_00004000'):
    key_path_str = jax.tree_util.keystr(key_path)
    assert key_path_str in MAPS, f"{key_path_str} not found"

  jax.tree_util.tree_map_with_path(verify_fn, state)

  def map_fn(key_path, value, prefix='gpt3_spmd1x64x24_tpuv4-3072_v84_20221101/checkpoints/checkpoint_00004000'):
    key_path_str = jax.tree_util.keystr(key_path)
    file_path, transform_fn = MAPS[key_path_str]
    full_path = os.path.join(prefix, file_path)
    spec = {'driver': 'zarr', 'metadata_key': '.zarray', 'kvstore': {}}
    spec['kvstore'] = {
    'bucket': 'mlperf-llm-public2',
    'driver': 'gcs',
    'path': full_path,
    }

    arr = ts.open(ts.Spec(spec), open=True).result().read().result()
    if transform_fn is not None:
    arr = transform_fn(arr)

    global PEAK_CPU_MEMORY
    PEAK_CPU_MEMORY = max(PEAK_CPU_MEMORY, Process().memory_info().rss)
    assert value.shape == arr.shape, f"{key_path}, {value.shape}, {arr.shape}"
    shape = value.shape
    sharding = value.sharding
    result = jax.make_array_from_single_device_arrays(
    shape,
    sharding,
    [jax.device_put(np.array(arr[index]), d)
      for d, index in sharding.addressable_devices_indices_map(shape).items()],
    )

    arr = None
    del arr
    gc.collect()
    max_logging.log(f"{key_path_str} finished")
    print_memory()
    return result

  converted_state = jax.tree_util.tree_map_with_path(map_fn, state)
  max_logging.log("converted state finished")
  print_memory()

  if checkpoint_manager.save(converted_state.step, converted_state):
    max_logging.log(f"saved a checkpoint at step {converted_state.step}")
  # Upon preemption, exit when and only when all ongoing saves are complete.
  if checkpoint_manager.reached_preemption(converted_state.step):
    checkpoint_manager.wait_until_finished()
    sys.exit()

  max_logging.log(f"Peak cpu memory in a single process: {fmt_size(PEAK_CPU_MEMORY)}")
  max_logging.log("checkpoint converted and saved successfully.")

if __name__ == "__main__":
  app.run(main)