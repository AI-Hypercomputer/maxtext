import functools
import os.path
import sys
import unittest

import pytest

import jax
from jax.sharding import Mesh
import jax.numpy as jnp

from flax.core import meta
from flax import linen as nn

from MaxText import maxtext_utils
from MaxText import pyconfig
from MaxText.common_types import MODEL_MODE_TRAIN
from MaxText.globals import PKG_DIR
from MaxText.layers import pipeline
from MaxText.layers import simple_layer
from MaxText.input_pipeline.input_pipeline_interface import create_data_iterator
from MaxText.train import main as train_main
from MaxText.layers import deepseek
from MaxText.train import (
    check_example_batch,
    create_goodput_recorder,
    eval_step,
    EPS,
    get_first_step,
    load_next_batch,
    record_goodput,
    record_scalar_metrics,
    save_checkpoint,
    setup_mesh_and_model,
    train_step,
    validate_train_config,
)


config = pyconfig.initialize(
    [sys.argv[0], os.path.join(PKG_DIR, "configs", "base.yml")],
    enable_checkpointing=False,
    enable_goodput_recording=False,
    run_name="circular_ag_once",
    max_target_length=128,
    base_emb_dim=28,
    ici_pipeline_parallelism=2,
    base_num_decoder_layers=8,
    num_pipeline_microbatches=8,
    per_device_batch_size=4,
    pipeline_fsdp_ag_once=True,
    dataset_type="synthetic",
)

init_rng, writer, checkpoint_manager, mesh, model, learning_rate_schedule, tx = setup_mesh_and_model(config)
data_iterator, eval_data_iterator = create_data_iterator(config, mesh)
state, _, state_mesh_shardings, data_iterator = maxtext_utils.setup_training_state(
    model, data_iterator, tx, config, init_rng, mesh, checkpoint_manager
)

  # pylint: disable=line-too-long
(
    functional_train,
    in_shard_train,
    out_shard_train,
    static_argnums_train,
    donate_argnums_train,
) = maxtext_utils.get_functional_train_with_signature(train_step, mesh, state_mesh_shardings, model, config)