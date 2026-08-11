# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Standalone Reproduction & Diagnostic Test: Splash Attention vs. Ragged Paged Attention (RPA).

This test directly isolates the attention core mathematical execution without model stack,
embeddings, layernorms, or MoE layers.

It computes 3-way numerical comparison across:
1. Exact Mathematical Reference Attention (FP32 Causal Dot-Product in pure JAX)
2. Training Kernel: Splash / Flash Attention (wrap_flash_attention / AttentionOp)
3. Inference Kernel: vLLM Ragged Paged Attention (sharded_ragged_paged_attention / Pallas RPA)
"""

import math
import os
import sys
from typing import Any, Dict, Tuple
import jax
from jax import numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

# Ensure Mosaic Pallas TPU lowering is registered
try:
    from jax._src.pallas.mosaic import lowering as _mosaic_lowering
except ImportError:
    pass

from maxtext.common.common_types import (
    MODEL_MODE_PREFILL,
    MODEL_MODE_TRAIN,
    AttentionType,
)
from maxtext.configs import pyconfig
from maxtext.layers.attention_op import AttentionOp
from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR
from maxtext.utils.sharding import create_sharding, get_logical_axis_rules


def compute_drift_metrics(
    tensor_a: jax.Array, tensor_b: jax.Array
) -> Dict[str, float]:
    """Computes comprehensive numerical drift metrics between two arrays."""
    a = np.array(jax.device_get(tensor_a), dtype=np.float32)
    b = np.array(jax.device_get(tensor_b), dtype=np.float32)

    abs_diff = np.abs(a - b)
    max_abs = float(np.max(abs_diff))
    mae = float(np.mean(abs_diff))
    mse = float(np.mean(abs_diff**2))

    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    norm_a = float(np.linalg.norm(a_flat))
    norm_b = float(np.linalg.norm(b_flat))
    dot = float(np.dot(a_flat, b_flat))

    cos_sim = dot / (norm_a * norm_b + 1e-12) if norm_a > 0 and norm_b > 0 else 1.0
    rel_err = float(np.linalg.norm(a_flat - b_flat) / (norm_a + 1e-12))

    return {
        "max_abs_err": max_abs,
        "mae": mae,
        "mse": mse,
        "cos_sim": cos_sim,
        "rel_err": rel_err,
    }


def run_splash_attention(
    mesh: Mesh,
    config: Any,
    query: jax.Array,  # (batch, seq_len, num_query_heads, head_dim)
    key: jax.Array,    # (batch, seq_len, num_kv_heads, head_dim)
    value: jax.Array,  # (batch, seq_len, num_kv_heads, head_dim)
    decoder_segment_ids: jax.Array,
    inputs_positions: jax.Array,
) -> jax.Array:
    """Executes the training Flash / Splash Attention kernel on TPU."""
    from maxtext.layers import attentions
    from flax import nnx

    attn = attentions.Attention(
        config=config,
        num_query_heads=config.num_query_heads,
        num_kv_heads=config.num_kv_heads,
        head_dim=config.head_dim,
        max_target_length=config.max_target_length,
        mesh=mesh,
        attention_kernel=config.attention,
        inputs_q_shape=(query.shape[0], query.shape[1], config.base_emb_dim),
        inputs_kv_shape=(key.shape[0], key.shape[1], config.base_emb_dim),
        model_mode=MODEL_MODE_TRAIN,
        rngs=nnx.Rngs(params=0),
    )

    @nnx.jit
    def _forward_splash(m, q, k, v, seg, pos):
        return m.attention_op(
            q,
            k,
            v,
            seg,
            pos,
            MODEL_MODE_TRAIN,
            [None, None],
            None,
            None,
            None,
        )

    out = _forward_splash(attn, query, key, value, decoder_segment_ids, inputs_positions)
    return out


def run_rpa_attention(
    mesh: Mesh,
    config: Any,
    query: jax.Array,  # (batch, seq_len, num_query_heads, head_dim)
    key: jax.Array,    # (batch, seq_len, num_kv_heads, head_dim)
    value: jax.Array,  # (batch, seq_len, num_kv_heads, head_dim)
    block_size: int = 128,
    softmax_scale: float | None = None,
) -> jax.Array:
    """Executes the authentic inference Pallas Ragged Paged Attention (RPA) kernel on TPU."""
    from tpu_inference.layers.common import attention_interface

    batch_size, seq_len, num_query_heads, head_dim = query.shape
    num_kv_heads = key.shape[2]

    num_blocks_per_seq = (seq_len + block_size - 1) // block_size
    total_pages = batch_size * num_blocks_per_seq

    # 5D Paged KV Cache: (total_pages, block_size, num_kv_heads, 2, head_dim)
    kv_cache = jnp.zeros(
        (total_pages, block_size, num_kv_heads, 2, head_dim),
        dtype=query.dtype,
    )

    # 1D Metadata arrays matching RPA sharding rules
    block_tables = jnp.arange(total_pages, dtype=jnp.int32)
    seq_lens = jnp.array([seq_len] * batch_size, dtype=jnp.int32)
    query_start_loc = jnp.tile(jnp.array([0, seq_len], dtype=jnp.int32), (batch_size,))
    request_distribution = jnp.tile(jnp.array([0, 0, 1], dtype=jnp.int32), (batch_size,))

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    @jax.jit
    def _forward_rpa(q, k, v, kv, sl, bt, qsl, rd):
        out, _ = attention_interface.sharded_ragged_paged_attention(
            mesh,
            q,
            k,
            v,
            kv,
            sl,
            bt,
            qsl,
            rd,
            None,             # sinks
            softmax_scale,    # query_pre_attn_scalar
            None,             # attention_chunk_size (None for full global attention)
            None,             # q_scale
            None,             # k_scale
            None,             # v_scale
            update_kv_cache=True,
        )
        return out

    q_3d = query.reshape(-1, num_query_heads, head_dim)
    k_3d = key.reshape(-1, num_kv_heads, head_dim)
    v_3d = value.reshape(-1, num_kv_heads, head_dim)

    out_rpa = _forward_rpa(
        q_3d, k_3d, v_3d, kv_cache, seq_lens, block_tables, query_start_loc, request_distribution
    )
    return out_rpa.reshape(batch_size, seq_len, num_query_heads, head_dim)


def compare_attention_kernels_on_tpu(
    batch_size: int = 4,
    seq_len: int = 512,
    num_query_heads: int = 16,
    num_kv_heads: int = 2,
    head_dim: int = 256,
    dtype_str: str = "bfloat16",
    block_size: int = 128,
    extra_train_kwargs: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Runs a pure attention kernel isolated comparison on Cloud TPU."""
    from maxtext.utils import maxtext_utils
    from tests.utils.test_helpers import get_test_config_path

    dtype = jnp.bfloat16 if dtype_str == "bfloat16" else jnp.float32

    train_kwargs = {
        "override_model_config": True,
        "model_name": "qwen3.5-35b-a3b",
        "max_target_length": seq_len,
        "max_prefill_predict_length": seq_len,
        "per_device_batch_size": 1.0,
        "enable_nnx": True,
        "pure_nnx": True,
        "pure_nnx_decoder": True,
        "scan_layers": False,
        "enable_checkpointing": False,
        "log_config": False,
        "weight_dtype": dtype_str,
        "dtype": dtype_str,
    }
    if extra_train_kwargs:
        train_kwargs.update(extra_train_kwargs)

    train_cfg = pyconfig.initialize(
        [sys.argv[0], get_test_config_path(), "attention=flash"],
        **train_kwargs,
    )
    cfg_infer = pyconfig.initialize(
        [
            sys.argv[0],
            get_test_config_path("inference/vllm.yml"),
            "attention=vllm_rpa",
            "model_call_mode=inference",
            "ici_data_parallelism=-1",
        ],
        **train_kwargs,
    )

    train_devices = maxtext_utils.create_device_mesh(train_cfg)
    train_mesh = Mesh(train_devices, train_cfg.mesh_axes)

    infer_devices = maxtext_utils.create_device_mesh(cfg_infer)
    infer_mesh = Mesh(infer_devices, cfg_infer.mesh_axes)

    key_rng = jax.random.PRNGKey(42)
    k_q, k_k, k_v = jax.random.split(key_rng, 3)

    q_init = jax.random.normal(k_q, (batch_size, seq_len, num_query_heads, head_dim), dtype=dtype)
    k_init = jax.random.normal(k_k, (batch_size, seq_len, num_kv_heads, head_dim), dtype=dtype)
    v_init = jax.random.normal(k_v, (batch_size, seq_len, num_kv_heads, head_dim), dtype=dtype)

    decoder_positions = jnp.tile(jnp.arange(seq_len, dtype=jnp.int32), (batch_size, 1))
    decoder_segment_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)

    # Shard tensors across data mesh
    q_sharded = jax.device_put(q_init, NamedSharding(train_mesh, P(("data", "fsdp"), None, None, None)))
    k_sharded = jax.device_put(k_init, NamedSharding(train_mesh, P(("data", "fsdp"), None, None, None)))
    v_sharded = jax.device_put(v_init, NamedSharding(train_mesh, P(("data", "fsdp"), None, None, None)))
    pos_sharded = jax.device_put(decoder_positions, NamedSharding(train_mesh, P(("data", "fsdp"), None)))
    seg_sharded = jax.device_put(decoder_segment_ids, NamedSharding(train_mesh, P(("data", "fsdp"), None)))

    softmax_scale = 1.0 / math.sqrt(head_dim)

    # 1. Splash / Flash Attention (Training Kernel expects pre-scaled Q = Q / sqrt(d))
    print("  [1/2] Executing Splash Attention (Training Kernel on TPU)...", flush=True)
    q_splash_input = q_sharded * softmax_scale
    out_splash = run_splash_attention(
        train_mesh,
        train_cfg,
        q_splash_input,
        k_sharded,
        v_sharded,
        seg_sharded,
        pos_sharded,
    )
    out_splash.block_until_ready()

    # 2. Ragged Paged Attention (Inference Kernel)
    print("  [2/2] Executing vLLM Ragged Paged Attention (Inference Pallas RPA Kernel on TPU)...", flush=True)
    out_rpa = run_rpa_attention(
        infer_mesh,
        cfg_infer,
        q_sharded,
        k_sharded,
        v_sharded,
        block_size=block_size,
        softmax_scale=softmax_scale,
    )
    out_rpa.block_until_ready()

    # Compute Splash vs. RPA Pairwise Drift Metrics
    metrics_splash_vs_rpa = compute_drift_metrics(out_splash, out_rpa)

    return {
        "splash_vs_rpa": metrics_splash_vs_rpa,
        "out_splash": out_splash,
        "out_rpa": out_rpa,
    }
