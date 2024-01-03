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
    tx = maxtext_utils.get_optimizer(cfg, learning_rate_schedule)

    checkpoint_manager_dist = checkpointing.create_orbax_checkpoint_manager(
        cfg.checkpoint_dir,
        cfg.enable_checkpointing,
        cfg.async_checkpointing,
        cfg.save_period,
    )

    state_dist, _ = max_utils.setup_training_state(model, tx, cfg, init_rng, mesh, checkpoint_manager_dist)

    checkpoint_manager_src = checkpointing.create_orbax_checkpoint_manager(
        'gs://lizhiyu-multipods/lizhiyu/colab/convergence_test/checkpoints',
        cfg.enable_checkpointing,
        cfg.async_checkpointing,
        cfg.save_period,
    )
    tx_src = optax.adamw(
        learning_rate_schedule,
        b1=cfg.adam_b1,
        b2=cfg.adam_b2,
        eps=cfg.adam_eps,
        eps_root=cfg.adam_eps_root,
        weight_decay=cfg.adam_weight_decay,
    )
    state_src, _ = max_utils.setup_training_state(model, tx_src, cfg, init_rng, mesh, checkpoint_manager_src)
    state_dist = state_dist.replace(
        step=state_src.step,
        params=state_src.params,
        opt_state=dict(
            count=state_src.opt_state.count,
            m=state_src.opt_state[0].mu,
            v=state_src.opt_state[0].nu,
            ),
        )

    if checkpoint_manager_dist.save(state_dist.step, state_dist):
        max_logging.log(f"saved a checkpoint at step {state_dist.step}")
    # Upon preemption, exit when and only when all ongoing saves are complete.
    if checkpoint_manager_dist.reached_preemption(state_dist.step):
        checkpoint_manager_dist.wait_until_finished()
        sys.exit()

    max_logging.log("checkpoint converted and saved successfully.")

if __name__ == "__main__":
  app.run(main)
