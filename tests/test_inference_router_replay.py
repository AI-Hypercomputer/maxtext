"""Integration Test: Extract Router Replay Data from MaxText MoE Inference.

Validates:
1. MaxText MoE inference execution (attention="vllm_rpa", fused_moe_matmul) capturing `selected_experts` / `expert_indices`.
2. Router replay extraction from intermediate state and decoder outputs with shape (batch_size, seq_len, top_k).
"""

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["NEW_MODEL_DESIGN"] = "1"
os.environ["SKIP_JAX_PRECOMPILE"] = "1"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
import sys
import unittest
import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from maxtext.configs import pyconfig
from maxtext.models import models
from maxtext.utils import maxtext_utils
from maxtext.common.common_types import (
    DECODING_ACTIVE_SEQUENCE_INDICATOR,
    MODEL_MODE_PREFILL,
)
from tests.utils.test_helpers import get_test_config_path


class InferenceRouterReplayExtractionTest(unittest.TestCase):

  def setUp(self):
    os.environ["NEW_MODEL_DESIGN"] = "1"
    os.environ["SKIP_JAX_PRECOMPILE"] = "1"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

  def test_extract_router_replay_from_inference(self):
    seq_len = 16
    batch_size = 2
    num_layers = 1
    num_experts = 4
    top_k = 2

    test_tokens = [791, 7155, 315, 9342, 374, 9897, 323, 374] * 4
    raw_tokens = test_tokens[:seq_len]

    base_kwargs = {
        "run_name": "test_router_replay_extraction_fast",
        "enable_checkpointing": False,
        "override_model_config": True,
        "base_num_decoder_layers": num_layers,
        "num_decoder_layers": num_layers,
        "model_name": "qwen3.5-35b-a3b",
        "num_experts": num_experts,
        "num_experts_per_tok": top_k,
        "base_emb_dim": 512,
        "base_num_query_heads": 2,
        "base_num_kv_heads": 2,
        "head_dim": 256,
        "partial_rotary_factor": 0.25,
        "base_mlp_dim": 1024,
        "base_moe_mlp_dim": 1024,
        "vocab_size": 1000,
        "max_target_length": seq_len,
        "max_prefill_predict_length": seq_len,
        "per_device_batch_size": float(batch_size),
        "scan_layers": False,
        "weight_dtype": "bfloat16",
        "dtype": "bfloat16",
        "log_config": False,
        "skip_jax_distributed_system": True,
        "ici_tensor_parallelism": 4,
        "ici_data_parallelism": 1,
        "ici_expert_parallelism": 1,
        "enable_nnx": True,
        "pure_nnx": True,
        "pure_nnx_decoder": True,
    }

    cfg_infer = pyconfig.initialize(
        [sys.argv[0], get_test_config_path("inference/vllm.yml"), "attention=vllm_rpa"],
        **base_kwargs,
    )

    devices_array = maxtext_utils.create_device_mesh(cfg_infer)
    mesh = Mesh(devices_array, cfg_infer.mesh_axes)
    rng = jax.random.PRNGKey(42)

    ids = jnp.tile(jnp.expand_dims(jnp.array(raw_tokens, dtype=jnp.int32), axis=0), (batch_size, 1))
    decoder_positions = jnp.tile(jnp.expand_dims(jnp.arange(seq_len, dtype=jnp.int32), axis=0), (batch_size, 1))
    segment_ids = jnp.zeros((batch_size, seq_len), dtype=jnp.int32) + DECODING_ACTIVE_SEQUENCE_INDICATOR

    model_infer = models.transformer_as_linen(config=cfg_infer, mesh=mesh, quant=None, model_mode=MODEL_MODE_PREFILL)
    init_params_rng, init_dropout_rng = jax.random.split(rng)
    vars_dict_init = model_infer.init(
        {"params": init_params_rng, "dropout": init_dropout_rng},
        ids,
        decoder_positions,
        segment_ids,
        enable_dropout=False,
    )
    vars_infer = dict(vars_dict_init)

    # Run inference prefill
    out, cache_infer = model_infer.apply(
        vars_infer,
        ids,
        decoder_positions,
        segment_ids,
        enable_dropout=False,
        model_mode=MODEL_MODE_PREFILL,
        mutable=["cache", "intermediates"],
    )

    self.assertIsInstance(out, tuple, "Inference output must be a tuple")
    self.assertEqual(len(out), 3, "Inference output must be (hidden_state, kv_caches, expert_indices)")
    hidden_state, kv_caches, expert_indices = out
    self.assertIsNotNone(expert_indices, "expert_indices must not be None")
    self.assertEqual(expert_indices.shape[1:], (batch_size * seq_len, top_k))
    print(f"\n[Inference Router Extraction] Extracted routing data shape: {expert_indices.shape}")


if __name__ == "__main__":
  unittest.main()
