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

"""Standalone Reproduction & Diagnostic Test: Tokamax GMM v2 (Training) vs. Fused MoE (tpu-inference).

This test isolates the MoE block computation without full attention layers, embeddings, or layernorms.

It computes 3-way numerical comparisons across:
1. Exact Mathematical Reference MoE (FP32 Dense Gated MLP & Router in pure JAX)
2. Training Kernel: Tokamax GMM v2 (RoutedMoE / mblx.gmm with use_tokamax_gmm=True, use_gmm_v2=True)
3. Inference Kernel: Fused MoE Kernel (tpu_inference.layers.common.fused_moe_gmm.fused_moe_func)
"""

import math
import os
import sys
from typing import Any, Dict, Tuple
from flax import nnx
import jax
from jax import numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
import pytest

# Ensure Mosaic Pallas TPU lowering is registered
try:
    from jax._src.pallas.mosaic import lowering as _mosaic_lowering
except ImportError:
    pass

from maxtext.common.common_types import (
    MODEL_MODE_PREFILL,
    MODEL_MODE_TRAIN,
)
from maxtext.configs import pyconfig
from maxtext.layers import initializers as max_initializers
from maxtext.layers import moe
from maxtext.utils import maxtext_utils
from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR
from tests.utils.test_helpers import get_test_config_path


def compute_metrics(a: jax.Array, b: jax.Array) -> dict[str, float]:
    """Computes L_inf, MAE, RMSE, CosSim, and Relative Error between two arrays."""
    a_f32 = np.asarray(a, dtype=np.float32)
    b_f32 = np.asarray(b, dtype=np.float32)

    abs_diff = np.abs(a_f32 - b_f32)
    max_err = float(np.max(abs_diff))
    mae = float(np.mean(abs_diff))
    rmse = float(np.sqrt(np.mean(np.square(abs_diff))))

    norm_a = np.linalg.norm(a_f32.ravel())
    norm_b = np.linalg.norm(b_f32.ravel())
    if norm_a > 1e-12 and norm_b > 1e-12:
        cos_sim = float(np.dot(a_f32.ravel(), b_f32.ravel()) / (norm_a * norm_b))
    else:
        cos_sim = 1.0

    denom = np.maximum(np.abs(b_f32), 1e-6)
    rel_err = float(np.mean(abs_diff / denom))

    return {
        "max_err": max_err,
        "mae": mae,
        "rmse": rmse,
        "cos_sim": cos_sim,
        "rel_err": rel_err,
    }


def run_reference_moe(
    inputs: jax.Array,
    gate_logits: jax.Array,
    w0: jax.Array,  # [E, D, H]
    w1: jax.Array,  # [E, D, H]
    wo: jax.Array,  # [E, H, D]
    topk: int = 8,
    renormalize: bool = True,
) -> tuple[jax.Array, dict[str, Any]]:
    """Exact Mathematical Reference MoE in high-precision Float32."""
    num_tokens, emb_dim = inputs.shape
    num_experts = w0.shape[0]

    # 1. Router Scoring & Top-K Selection
    scores = jax.nn.softmax(gate_logits.astype(jnp.float32), axis=-1)
    topk_weights, topk_indices = jax.lax.top_k(scores, k=topk)
    if renormalize:
        topk_weights = topk_weights / jnp.sum(topk_weights, axis=-1, keepdims=True)
    topk_weights = topk_weights.astype(inputs.dtype)

    # 2. Per-expert exact reference execution (memory-efficient & exact)
    final_out = jnp.zeros_like(inputs, dtype=jnp.float32)
    inputs_f32 = inputs.astype(jnp.float32)
    w0_f32 = w0.astype(jnp.float32)
    w1_f32 = w1.astype(jnp.float32)
    wo_f32 = wo.astype(jnp.float32)

    for e in range(num_experts):
        expert_mask = (topk_indices == e)
        expert_weight = jnp.sum(jnp.where(expert_mask, topk_weights, 0.0), axis=-1)  # [T]

        g_e = jnp.matmul(inputs_f32, w0_f32[e])
        u_e = jnp.matmul(inputs_f32, w1_f32[e])
        act_e = jax.nn.silu(g_e) * u_e
        out_e = jnp.matmul(act_e, wo_f32[e])

        final_out = final_out + out_e * expert_weight[:, None]

    final_out = final_out.astype(inputs.dtype)
    return final_out, {}


def compare_moe_kernels_on_tpu(
    mesh: Mesh,
    batch_size: int = 4,
    seq_len: int = 512,
    emb_dim: int = 2048,
    moe_mlp_dim: int = 512,
    num_experts: int = 8,
    num_experts_per_tok: int = 8,
    dtype: jnp.dtype = jnp.float32,
    train_moe_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compares Training MoE (Tokamax GMM v2) vs Inference MoE (Fused MoE) on Cloud TPU."""
    if train_moe_kwargs is None:
        train_moe_kwargs = {}

    num_tokens = batch_size * seq_len
    dtype_str = "float32" if dtype == jnp.float32 else "bfloat16"

    # Base configuration for MaxText RoutedMoE
    base_kwargs = {
        "override_model_config": True,
        "model_name": "qwen3.5-35b-a3b",
        "base_emb_dim": emb_dim,
        "base_mlp_dim": moe_mlp_dim,
        "base_moe_mlp_dim": moe_mlp_dim,
        "num_experts": num_experts,
        "num_experts_per_tok": num_experts_per_tok,
        "vocab_size": 32000,
        "max_target_length": seq_len,
        "max_prefill_predict_length": seq_len,
        "per_device_batch_size": 1.0,
        "enable_nnx": True,
        "pure_nnx": True,
        "pure_nnx_decoder": True,
        "scan_layers": False,
        "enable_checkpointing": False,
        "log_config": False,
        "megablox": True,
        "use_tokamax_gmm": True,
        "use_gmm_v2": True,
        "sparse_matmul": True,
        "norm_topk_prob": True,
        "routed_score_func": "",
    }
    train_kwargs = dict(base_kwargs)
    train_kwargs.update(train_moe_kwargs)

    cfg_train = pyconfig.initialize(
        [
            sys.argv[0],
            get_test_config_path(),
            "sparse_matmul=True",
            "megablox=True",
            "use_tokamax_gmm=True",
            "use_gmm_v2=True",
        ],
        weight_dtype=dtype_str,
        dtype=dtype_str,
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
        weight_dtype=dtype_str,
        dtype=dtype_str,
        **base_kwargs,
    )

    train_devices = maxtext_utils.create_device_mesh(cfg_train)
    train_mesh = Mesh(train_devices, cfg_train.mesh_axes)

    infer_devices = maxtext_utils.create_device_mesh(cfg_infer)
    infer_mesh = Mesh(infer_devices, cfg_infer.mesh_axes)

    rng = nnx.Rngs(params=42)

    # 1. Instantiate Training RoutedMoE (Tokamax GMM v2)
    train_moe = moe.RoutedMoE(
        config=cfg_train,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        mesh=train_mesh,
        kernel_init=max_initializers.nd_dense_init(cfg_train.dense_init_scale, "fan_in", "truncated_normal"),
        kernel_axes=("embed", None),
        intermediate_dim=moe_mlp_dim,
        dtype=dtype,
        weight_dtype=dtype,
        rngs=rng,
    )

    # 2. Instantiate Inference RoutedMoE (Fused MoE from tpu-inference)
    infer_moe = moe.RoutedMoE(
        config=cfg_infer,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        mesh=infer_mesh,
        kernel_init=max_initializers.nd_dense_init(cfg_infer.dense_init_scale, "fan_in", "truncated_normal"),
        kernel_axes=("embed", None),
        intermediate_dim=moe_mlp_dim,
        dtype=dtype,
        weight_dtype=dtype,
        rngs=rng,
    )

    # 3. Synchronize weights between train and infer RoutedMoE
    infer_moe.gate.kernel = train_moe.gate.kernel
    if hasattr(train_moe.gate, "bias") and train_moe.gate.bias is not None:
        infer_moe.gate.bias = train_moe.gate.bias

    infer_moe.wi_0 = train_moe.wi_0
    infer_moe.wi_1 = train_moe.wi_1
    infer_moe.wo = train_moe.wo

    # Prepare random input activations
    key = jax.random.PRNGKey(42)
    inputs_3d = jax.random.normal(key, (batch_size, seq_len, emb_dim), dtype=dtype)
    inputs_3d = jax.device_put(inputs_3d, NamedSharding(train_mesh, P(("data", "fsdp"), None, None)))

    print("  [1/3] Executing Training MoE (Tokamax GMM v2 on TPU)...")
    out_train, _, _ = train_moe(inputs_3d)
    out_train_2d = out_train.reshape(num_tokens, emb_dim)

    print("  [2/3] Executing Inference MoE (Fused MoE Kernel on TPU)...")
    out_infer, _, _ = infer_moe(inputs_3d)
    out_infer_2d = out_infer.reshape(num_tokens, emb_dim)
    inputs_2d = inputs_3d.reshape(num_tokens, emb_dim)

    print("  [3/3] Executing Exact Math Reference MoE...")
    @jax.jit
    def _run_ref(x, w0, w1, wo, gw):
        logits = jnp.dot(x, gw)
        scores = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
        topk_w, topk_idx = jax.lax.top_k(scores, k=num_experts_per_tok)
        topk_w = topk_w / jnp.sum(topk_w, axis=-1, keepdims=True)
        topk_w = topk_w.astype(x.dtype)

        acc = jnp.zeros_like(x, dtype=jnp.float32)
        x_f32 = x.astype(jnp.float32)
        w0_f32 = w0.astype(jnp.float32)
        w1_f32 = w1.astype(jnp.float32)
        wo_f32 = wo.astype(jnp.float32)

        for e in range(num_experts):
            mask = (topk_idx == e)
            w_e = jnp.sum(jnp.where(mask, topk_w, 0.0), axis=-1)
            g_e = jnp.matmul(x_f32, w0_f32[e])
            u_e = jnp.matmul(x_f32, w1_f32[e])
            act_e = jax.nn.silu(g_e) * u_e
            out_e = jnp.matmul(act_e, wo_f32[e])
            acc = acc + out_e * w_e[:, None]
        return acc.astype(x.dtype)

    out_ref_2d = _run_ref(
        inputs_2d,
        train_moe.wi_0.value,
        train_moe.wi_1.value,
        train_moe.wo.value,
        train_moe.gate.kernel.value,
    )

    # Compute drift metrics
    metrics_train_vs_infer = compute_metrics(out_train_2d, out_infer_2d)
    metrics_train_vs_ref = compute_metrics(out_train_2d, out_ref_2d)
    metrics_infer_vs_ref = compute_metrics(out_infer_2d, out_ref_2d)

    return {
        "train_vs_infer": metrics_train_vs_infer,
        "train_vs_ref": metrics_train_vs_ref,
        "infer_vs_ref": metrics_infer_vs_ref,
    }


@pytest.mark.tpu_only
@pytest.mark.integration_test
def test_tokamax_gmm_v2_vs_fused_moe_small():
    """Small, fast 3-way parity check between the training MoE kernel (Tokamax GMM v2,
    `RoutedMoE` with `use_tokamax_gmm=True, use_gmm_v2=True`) and the real inference
    MoE kernel actually served by vLLM/tpu-inference (`RoutedMoE.fused_moe_matmul`,
    which calls `tpu_inference.layers.common.fused_moe_gmm.fused_moe_func` whenever
    `attention in ("vllm_rpa", "vllm_batched_rpa")`), against an exact FP32 math
    reference. This is the smoke-test counterpart to the larger multi-config sweep
    performed by `tests/run_moe_kernel_repro.py`, run directly on this TPU VM's
    locally-attached chips.
    """
    results = compare_moe_kernels_on_tpu(
        mesh=None,
        batch_size=1,
        seq_len=64,
        emb_dim=256,
        moe_mlp_dim=128,
        num_experts=4,
        num_experts_per_tok=2,
        dtype=jnp.float32,
    )
    assert results["train_vs_infer"]["cos_sim"] > 0.99
    assert results["train_vs_ref"]["cos_sim"] > 0.99
    assert results["infer_vs_ref"]["cos_sim"] > 0.99
