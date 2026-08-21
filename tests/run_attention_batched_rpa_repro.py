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

"""Standalone attention kernel comparison: Splash vs Default RPA vs Batched RPA on TPU."""

import math
import os
import sys

os.environ["NEW_MODEL_DESIGN"] = "1"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["VLLM_TARGET_DEVICE"] = "tpu"

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("src"))

import jax
import numpy as np
from jax import numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from maxtext.configs import pyconfig
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path
from tests.unit.attention_kernel_repro_test import (
    compute_drift_metrics,
    run_splash_attention,
    run_reference_attention,
)
import tpu_inference.kernels.ragged_paged_attention.v3.kernel as default_rpa
import tpu_inference.kernels.experimental.batched_rpa.wrapper as batched_rpa
from tpu_inference.layers.common import attention_interface


def run_standalone_rpa(
    mesh: Mesh,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    use_batched_kernel: bool,
    block_size: int = 128,
    softmax_scale: float | None = None,
) -> jax.Array:
    """Runs either Default RPA v3 or Batched RPA in isolation on TPU."""
    batch_size, seq_len, num_query_heads, head_dim = q.shape
    num_kv_heads = k.shape[2]
    num_blocks_per_seq = (seq_len + block_size - 1) // block_size
    total_pages = batch_size * num_blocks_per_seq

    if q.dtype == jnp.float32:
        kv_cache = jnp.zeros((total_pages, block_size, num_kv_heads * 2, 1, head_dim), dtype=q.dtype)
    else:
        kv_cache = jnp.zeros((total_pages, block_size, num_kv_heads, 2, head_dim), dtype=q.dtype)

    block_tables = jnp.arange(total_pages, dtype=jnp.int32)
    seq_lens = jnp.array([seq_len] * batch_size, dtype=jnp.int32)
    # Cumulative per-request token offsets, shape (batch_size+1,).
    query_start_loc = jnp.arange(0, (batch_size + 1) * seq_len, seq_len, dtype=jnp.int32)
    # [num_decode_requests, num_decode_requests, num_total_requests], shape (3,).
    request_distribution = jnp.array([0, 0, batch_size], dtype=jnp.int32)

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    if use_batched_kernel:
        def _batched_rpa_wrapper(*args, **kwargs):
            kwargs.setdefault("vmem_limit_bytes", 64 * 1024 * 1024)
            return batched_rpa.ragged_paged_attention(*args, **kwargs)

        attention_interface.ragged_paged_attention = _batched_rpa_wrapper
    else:
        attention_interface.ragged_paged_attention = default_rpa.ragged_paged_attention

    q_3d = q.reshape(-1, num_query_heads, head_dim)
    k_3d = k.reshape(-1, num_kv_heads, head_dim)
    v_3d = v.reshape(-1, num_kv_heads, head_dim)

    # The RPA kernel's shard_map runs on `mesh`, which (per real vLLM-TPU
    # serving) may use only a subset of the visible devices (tensor-parallel
    # capped at num_kv_heads). Inputs may still be committed to a different
    # mesh's devices (e.g. the training mesh) -- explicitly re-place them
    # (replicated) onto `mesh`'s devices so shard_map's device set matches.
    replicated = NamedSharding(mesh, P())
    q_3d = jax.device_put(q_3d, replicated)
    k_3d = jax.device_put(k_3d, replicated)
    v_3d = jax.device_put(v_3d, replicated)

    out_rpa, _ = attention_interface.sharded_ragged_paged_attention(
        mesh,
        q_3d,
        k_3d,
        v_3d,
        kv_cache,
        seq_lens,
        block_tables,
        query_start_loc,
        request_distribution,
        None,             # sinks
        softmax_scale,    # query_pre_attn_scalar
        None,             # attention_chunk_size
        None,             # q_scale
        None,             # k_scale
        None,             # v_scale
        update_kv_cache=True,
    )
    return out_rpa.reshape(batch_size, seq_len, num_query_heads, head_dim)


def benchmark_attention_kernels(dtype_str: str = "float32"):
    # NOTE: batch_size/seq_len/block_size reduced from the original
    # (4, 512, 128) sweep -- the Batched RPA kernel's internal autotuned
    # decode-shape compilation (e.g. "RPAd-p128-b8-q1-k1152") requested
    # ~84.9MB of scoped VMEM against the real ~64MB TPU v5p VMEM budget at
    # the original sizes, causing a RESOURCE_EXHAUSTED CompileTimeScopedVmemOom
    # even with vmem_limit_bytes raised well past 64MB (the limit kwarg can't
    # exceed the physical budget). This smaller config fits within budget.
    batch_size = 2
    seq_len = 256
    num_query_heads = 16
    num_kv_heads = 2
    head_dim = 256
    block_size = 64
    dtype = jnp.bfloat16 if dtype_str == "bfloat16" else jnp.float32

    train_kwargs = {
        "override_model_config": True,
        "model_name": "qwen3.5-35b-a3b",
        "base_emb_dim": 2048,
        "base_num_query_heads": num_query_heads,
        "base_num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "max_target_length": seq_len,
        "per_device_batch_size": 1.0,
        "enable_nnx": True,
        "pure_nnx": True,
        "pure_nnx_decoder": True,
        "scan_layers": False,
        "enable_checkpointing": False,
        "log_config": False,
        "weight_dtype": dtype_str,
        "dtype": dtype_str,
        "inhomogeneous_layer_cycle_interval": 1,
    }

    train_cfg = pyconfig.initialize(
        [
            sys.argv[0],
            get_test_config_path(),
            "attention=flash",
            "use_tokamax_splash=True",
            "sa_use_base2_exp=False",
            "sa_fuse_reciprocal=False",
        ],
        **train_kwargs,
    )

    train_devices = maxtext_utils.create_device_mesh(train_cfg)
    train_mesh = Mesh(train_devices, train_cfg.mesh_axes)

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
    # Real vLLM-TPU serving shards tensor-parallel across the "model" mesh axis,
    # capped at num_kv_heads -- NOT data-parallel across all devices (the
    # request_distribution/query_start_loc metadata arrays have small fixed
    # shapes that cannot be sharded across >1 "data" replicas). Build the
    # inference mesh manually with model=tp_degree, all other axes=1, rather
    # than via maxtext_utils.create_device_mesh (which requires the ICI
    # product to equal the full visible device count).
    all_devices = jax.devices()
    tp_degree = min(num_kv_heads, len(all_devices))
    infer_devices = np.array(all_devices[:tp_degree])
    infer_mesh_shape = tuple(tp_degree if axis == "model" else 1 for axis in cfg_infer.mesh_axes)
    infer_devices = infer_devices.reshape(infer_mesh_shape)
    infer_mesh = Mesh(infer_devices, cfg_infer.mesh_axes)

    key_rng = jax.random.PRNGKey(42)
    k_q, k_k, k_v = jax.random.split(key_rng, 3)

    q_init = jax.random.normal(k_q, (batch_size, seq_len, num_query_heads, head_dim), dtype=dtype)
    k_init = jax.random.normal(k_k, (batch_size, seq_len, num_kv_heads, head_dim), dtype=dtype)
    v_init = jax.random.normal(k_v, (batch_size, seq_len, num_kv_heads, head_dim), dtype=dtype)

    decoder_positions = jnp.tile(jnp.arange(seq_len, dtype=jnp.int32), (batch_size, 1))
    decoder_segment_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)

    q_sharded = jax.device_put(q_init, NamedSharding(train_mesh, P(("data", "fsdp"), None, None, None)))
    k_sharded = jax.device_put(k_init, NamedSharding(train_mesh, P(("data", "fsdp"), None, None, None)))
    v_sharded = jax.device_put(v_init, NamedSharding(train_mesh, P(("data", "fsdp"), None, None, None)))
    pos_sharded = jax.device_put(decoder_positions, NamedSharding(train_mesh, P(("data", "fsdp"), None)))
    seg_sharded = jax.device_put(decoder_segment_ids, NamedSharding(train_mesh, P(("data", "fsdp"), None)))

    softmax_scale = 1.0 / math.sqrt(head_dim)

    # 1. Tokamax Splash (Training Kernel)
    print("  [1/4] Executing Tokamax Splash (base2_exp=False, fuse_recip=False)...", flush=True)
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

    # 2. Default Pallas RPA v3 (Inference Kernel)
    print("  [2/4] Executing Default Pallas RPA v3...", flush=True)
    out_default_rpa = run_standalone_rpa(
        infer_mesh, q_sharded, k_sharded, v_sharded, use_batched_kernel=False, block_size=block_size, softmax_scale=softmax_scale
    )
    out_default_rpa.block_until_ready()

    # 3. Batched RPA (Target Inference Kernel)
    print("  [3/4] Executing Batched RPA...", flush=True)
    out_batched_rpa = run_standalone_rpa(
        infer_mesh, q_sharded, k_sharded, v_sharded, use_batched_kernel=True, block_size=block_size, softmax_scale=softmax_scale
    )
    out_batched_rpa.block_until_ready()

    # 4. Exact Mathematical Reference (FP32)
    print("  [4/4] Executing Exact Math Reference Attention...", flush=True)
    out_ref = run_reference_attention(q_init, k_init, v_init, softmax_scale)

    m_splash_vs_default_rpa = compute_drift_metrics(out_splash, out_default_rpa)
    m_splash_vs_batched_rpa = compute_drift_metrics(out_splash, out_batched_rpa)
    m_splash_vs_ref = compute_drift_metrics(out_splash, out_ref)
    m_default_rpa_vs_ref = compute_drift_metrics(out_default_rpa, out_ref)
    m_batched_rpa_vs_ref = compute_drift_metrics(out_batched_rpa, out_ref)
    m_batched_vs_default_rpa = compute_drift_metrics(out_batched_rpa, out_default_rpa)

    print(f"\n==================== STANDALONE ATTENTION RESULTS ({dtype_str.upper()}) ====================")
    print(f"1. Tokamax Splash vs Default RPA v3 : L_inf={m_splash_vs_default_rpa['max_abs_err']:.2e}, MAE={m_splash_vs_default_rpa['mae']:.2e}, CosSim={m_splash_vs_default_rpa['cos_sim']:.6f}")
    print(f"2. Tokamax Splash vs Batched RPA    : L_inf={m_splash_vs_batched_rpa['max_abs_err']:.2e}, MAE={m_splash_vs_batched_rpa['mae']:.2e}, CosSim={m_splash_vs_batched_rpa['cos_sim']:.6f}")
    print(f"3. Batched RPA vs Default RPA v3    : L_inf={m_batched_vs_default_rpa['max_abs_err']:.2e}, MAE={m_batched_vs_default_rpa['mae']:.2e}, CosSim={m_batched_vs_default_rpa['cos_sim']:.6f}")
    print(f"4. Tokamax Splash vs Exact Ref Math : L_inf={m_splash_vs_ref['max_abs_err']:.2e}, MAE={m_splash_vs_ref['mae']:.2e}, CosSim={m_splash_vs_ref['cos_sim']:.6f}")
    print(f"5. Batched RPA vs Exact Ref Math    : L_inf={m_batched_rpa_vs_ref['max_abs_err']:.2e}, MAE={m_batched_rpa_vs_ref['mae']:.2e}, CosSim={m_batched_rpa_vs_ref['cos_sim']:.6f}")
    print(f"6. Default RPA v3 vs Exact Ref Math : L_inf={m_default_rpa_vs_ref['max_abs_err']:.2e}, MAE={m_default_rpa_vs_ref['mae']:.2e}, CosSim={m_default_rpa_vs_ref['cos_sim']:.6f}")
    print("================================================================================\n")


def main():
    print("[Local TPU VM] Running directly on locally-attached TPU chips (no SPS proxy).")
    for dt in ["float32", "bfloat16"]:
        benchmark_attention_kernels(dt)


if __name__ == "__main__":
    main()
