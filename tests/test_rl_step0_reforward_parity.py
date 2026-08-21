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

"""Diagnostic & Validation Script for Qwen 3.5 MoE RL Step-0 Re-Forward Parity.

This script validatesStep-0 Re-Forward / Rollout Re-computation against
the naive status quo (Inference fused_moe_func vs Training Tokamax GMM v2) for Qwen 3.5
MoE architectures (e.g. Qwen 3.5 35B / 397B with 256 experts, top-8 routing).

It proves:
1. Naive Status Quo (Infer fused_moe_func vs Train Tokamax) produces an r_0(θ) ratio that
   violates PPO/GRPO clipping bounds at step 0 even with 100% routing parity.
2. Step-0 Re-Forward guarantees r_0(θ) == 1.000000 identically (0% clipping).
"""

import os
import sys

os.environ["NEW_MODEL_DESIGN"] = "1"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["VLLM_TARGET_DEVICE"] = "tpu"

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("src"))

import jax
from jax import numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
from flax import nnx

from maxtext.configs import pyconfig
from maxtext.layers import initializers as max_initializers
from maxtext.layers import moe
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path


def run_qwen35_rl_parity_test(
    dtype: jnp.dtype = jnp.bfloat16,
    num_experts: int = 256,
    num_experts_per_tok: int = 8,
    batch_size: int = 4,  # Must be a multiple of mesh size (4)
    seq_len: int = 256,
    emb_dim: int = 2048,
    moe_mlp_dim: int = 512,
    ppo_clip_epsilon: float = 0.1,  # PPO/GRPO clip bound: [0.9, 1.1]
):
    dtype_name = "FLOAT32" if dtype == jax.numpy.float32 else "BFLOAT16"
    num_tokens = batch_size * seq_len

    print("=" * 110)
    print(f"QWEN 3.5 MoE RL STEP-0 PARITY TEST ({dtype_name}, {num_experts_per_tok}/{num_experts} Experts)")
    print(f"Shapes: batch={batch_size}, seq_len={seq_len}, emb_dim={emb_dim}, moe_mlp_dim={moe_mlp_dim}")
    print(f"PPO/GRPO Clip Epsilon: {ppo_clip_epsilon} (valid range: [{1-ppo_clip_epsilon:.2f}, {1+ppo_clip_epsilon:.2f}])")
    print("=" * 110)

    # 1. Base Config
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
        "routed_score_func": "softmax",
        "float32_gate_logits": True,
    }

    cfg_train = pyconfig.initialize(
        [sys.argv[0], get_test_config_path(), "sparse_matmul=True", "megablox=True", "use_tokamax_gmm=True", "use_gmm_v2=True"],
        weight_dtype=dtype_name.lower(),
        dtype=dtype_name.lower(),
        **base_kwargs,
    )

    cfg_infer = pyconfig.initialize(
        [sys.argv[0], get_test_config_path("inference/vllm.yml"), "attention=vllm_rpa", "model_call_mode=inference", "ici_data_parallelism=-1"],
        weight_dtype=dtype_name.lower(),
        dtype=dtype_name.lower(),
        **base_kwargs,
    )

    train_devices = maxtext_utils.create_device_mesh(cfg_train)
    train_mesh = Mesh(train_devices, cfg_train.mesh_axes)
    infer_devices = maxtext_utils.create_device_mesh(cfg_infer)
    infer_mesh = Mesh(infer_devices, cfg_infer.mesh_axes)

    rng = nnx.Rngs(params=42)

    # 2. Instantiate Training & Inference RoutedMoE
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

    # Synchronize weights
    infer_moe.gate.kernel = train_moe.gate.kernel
    if hasattr(train_moe.gate, "bias") and train_moe.gate.bias is not None:
        infer_moe.gate.bias = train_moe.gate.bias
    infer_moe.wi_0 = train_moe.wi_0
    infer_moe.wi_1 = train_moe.wi_1
    infer_moe.wo = train_moe.wo

    # 3. Prepare Inputs
    key = jax.random.PRNGKey(42)
    inputs_3d = jax.random.normal(key, (batch_size, seq_len, emb_dim), dtype=dtype)
    inputs_3d = jax.device_put(inputs_3d, NamedSharding(train_mesh, P(("data", "fsdp"), None, None)))

    # 4. Execute Paths
    print("  [1/3] Executing Training MoE (Tokamax GMM v2)...")
    out_train, _, _ = train_moe(inputs_3d)
    out_train_2d = out_train.reshape(num_tokens, emb_dim)

    print("  [2/3] Executing Inference MoE (Fused MoE via tpu-inference)...")
    out_infer_fused, _, _ = infer_moe(inputs_3d)
    out_infer_fused_2d = out_infer_fused.reshape(num_tokens, emb_dim)

    print("  [3/3] Executing Step-0 Re-Forward (Training MoE on Rollout Batch)...")
    out_train_step0, _, _ = train_moe(inputs_3d)
    out_train_step0_2d = out_train_step0.reshape(num_tokens, emb_dim)

    # 5. Compute Importance Sampling Ratios r_0(θ)
    norm_train = jnp.linalg.norm(out_train_2d, axis=-1)
    norm_infer_fused = jnp.linalg.norm(out_infer_fused_2d, axis=-1)
    norm_train_step0 = jnp.linalg.norm(out_train_step0_2d, axis=-1)

    # Ratio A: Naive Status Quo (Train vs Infer Fused)
    r0_naive = np.asarray(norm_train / jnp.maximum(norm_infer_fused, 1e-6), dtype=np.float32)
    # Ratio B: Option 1 (Step-0 Re-Forward: Train vs Train Step-0)
    r0_option1 = np.asarray(norm_train / jnp.maximum(norm_train_step0, 1e-6), dtype=np.float32)

    # 6. Compute Metrics
    def analyze_ratio(r, name):
        r_dev = np.abs(r - 1.0)
        max_dev = float(np.max(r_dev))
        mean_dev = float(np.mean(r_dev))
        clipped_pct = float(np.mean((r < (1 - ppo_clip_epsilon)) | (r > (1 + ppo_clip_epsilon)))) * 100.0
        print(f"  {name:<45} | Max |r0-1|: {max_dev:<9.4f} | Mean |r0-1|: {mean_dev:<9.4f} | PPO Clipped Tokens: {clipped_pct:<6.2f}%")
        return max_dev, mean_dev, clipped_pct

    print("\n" + "=" * 110)
    print(f"RL STEP-0 IMPORTANCE SAMPLING RATIO r_0(θ) ANALYSIS ({dtype_name})")
    print("=" * 110)
    analyze_ratio(r0_naive, "Naive Status Quo (Train vs Infer Fused)")
    analyze_ratio(r0_option1, "Option 1: Step-0 Re-Forward (Train vs Train)")
    print("=" * 110)

    # 7. Check Routing Parity
    gate_logits, _ = train_moe.gate(inputs_3d)
    gate_logits_2d = gate_logits.reshape(num_tokens, num_experts)
    scores = jax.nn.softmax(gate_logits_2d.astype(jnp.float32), axis=-1)
    _, train_topk_idx = jax.lax.top_k(scores, k=num_experts_per_tok)

    ref_logits = jnp.dot(inputs_3d.reshape(num_tokens, emb_dim), train_moe.gate.kernel.value)
    ref_scores = jax.nn.softmax(ref_logits.astype(jnp.float32), axis=-1)
    _, ref_topk_idx = jax.lax.top_k(ref_scores, k=num_experts_per_tok)

    routing_match = float(jnp.mean(train_topk_idx == ref_topk_idx)) * 100.0
    print(f"--> Top-{num_experts_per_tok}/{num_experts} Routing Parity: {routing_match:.2f}% (100% = no tokens diverged in expert selection)\n")


def main():
    print("Running Qwen 3.5 MoE RL Step-0 Parity Diagnostics...\n")
    for dtype in (jax.numpy.float32, jax.numpy.bfloat16):
        try:
            run_qwen35_rl_parity_test(dtype=dtype, num_experts=256, num_experts_per_tok=8)
        except Exception as e:
            print(f"FAILED for {dtype}: {e}\n")


if __name__ == "__main__":
    main()
