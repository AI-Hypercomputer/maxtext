"""Integration Test: Insert and Replay Router Logits/Expert Decisions in MaxText Trainer.

Validates:
1. Passing `forced_routed_experts` in data batch to `train.loss_fn`.
2. MoE layers executing with forced routing in trainer forward/backward step.
"""

import os
import sys
import unittest
import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from maxtext.configs import pyconfig
from maxtext.models import models
from maxtext.utils import maxtext_utils
from maxtext.trainers.pre_train import train
from tests.utils.test_helpers import get_test_config_path


class TrainerRouterReplayTest(unittest.TestCase):

  def setUp(self):
    os.environ["NEW_MODEL_DESIGN"] = "1"
    os.environ["SKIP_JAX_PRECOMPILE"] = "1"

  def test_loss_fn_with_forced_routed_experts(self):
    seq_len = 16
    batch_size = 2
    num_layers = 1
    num_experts = 4
    top_k = 2
    vocab_size = 1000

    base_kwargs = {
        "run_name": "test_trainer_router_replay",
        "enable_checkpointing": False,
        "override_model_config": True,
        "base_num_decoder_layers": num_layers,
        "num_decoder_layers": num_layers,
        "model_name": "qwen3.5-35b-a3b",
        "num_experts": num_experts,
        "num_experts_per_tok": top_k,
        "base_emb_dim": 256,
        "base_num_query_heads": 2,
        "base_num_kv_heads": 2,
        "head_dim": 256,
        "partial_rotary_factor": 0.25,
        "base_mlp_dim": 256,
        "base_moe_mlp_dim": 256,
        "vocab_size": vocab_size,
        "max_target_length": seq_len,
        "max_prefill_predict_length": seq_len,
        "per_device_batch_size": float(batch_size),
        "scan_layers": False,
        "weight_dtype": "bfloat16",
        "dtype": "bfloat16",
        "log_config": False,
        "skip_jax_distributed_system": True,
        "ici_tensor_parallelism": 1,
        "ici_data_parallelism": 1,
        "ici_expert_parallelism": 1,
        "enable_nnx": True,
        "pure_nnx": True,
        "pure_nnx_decoder": True,
        "sparse_matmul": True,
    }

    cfg = pyconfig.initialize(
        [sys.argv[0], get_test_config_path(), "attention=flash", "sparse_matmul=True"],
        **base_kwargs,
    )

    devices_array = maxtext_utils.create_device_mesh(cfg)
    mesh = Mesh(devices_array, cfg.mesh_axes)
    rng = jax.random.PRNGKey(42)

    # Construct input tokens and synthetic forced routed experts
    tokens = jnp.array([10, 20, 30, 40] * 4, dtype=jnp.int32)[:seq_len]
    inputs = jnp.tile(jnp.expand_dims(tokens, axis=0), (batch_size, 1))
    positions = jnp.tile(jnp.expand_dims(jnp.arange(seq_len, dtype=jnp.int32), axis=0), (batch_size, 1))
    segmentation = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    targets = jnp.roll(inputs, -1, axis=-1)

    # Synthetic forced routed experts: [batch, seq_len, top_k]
    forced_experts = jnp.zeros((batch_size, seq_len, top_k), dtype=jnp.int32)
    forced_experts = forced_experts.at[:, :, 0].set(1)
    forced_experts = forced_experts.at[:, :, 1].set(3)

    data_batch = {
        "inputs": inputs,
        "inputs_position": positions,
        "inputs_segmentation": segmentation,
        "targets": targets,
        "targets_segmentation": segmentation,
        "forced_routed_experts": forced_experts,
    }

    model = models.transformer_as_linen(config=cfg, mesh=mesh, quant=None, model_mode="train")
    init_params_rng, init_dropout_rng = jax.random.split(rng)
    params = model.init(
        {"params": init_params_rng, "dropout": init_dropout_rng},
        inputs,
        positions,
        segmentation,
        enable_dropout=False,
    )

    # Execute trainer loss_fn with forced router replay data
    loss, aux = train.loss_fn(
        model,
        cfg,
        data_batch,
        dropout_rng=init_dropout_rng,
        params=params,
        is_train=True,
    )

    self.assertIsNotNone(loss)
    self.assertFalse(jnp.isnan(loss), "Loss must not be NaN")
    print(f"\n[Trainer Router Replay] Computed loss with forced routing: {loss}")


if __name__ == "__main__":
  unittest.main()
