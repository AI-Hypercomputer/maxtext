# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runner to benchmark Qwen3.5 MoE 1-Layer Intermediate Tensor & Logits

Dumps directly on locally-attached Cloud TPU v5p chips.
"""

import os
import sys

os.environ["NEW_MODEL_DESIGN"] = "1"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["VLLM_TARGET_DEVICE"] = "tpu"

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("src"))

import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

try:
    from jax._src.pallas.mosaic import lowering as _mosaic_lowering
except Exception:
    pass

from maxtext.common.common_types import MODEL_MODE_PREFILL, MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.models import qwen3_5
from maxtext.utils import maxtext_utils
from tests.unit.qwen3_5_layer_dump_test import (capture_qwen3_5_layer_intermediates, compute_drift_metrics,
                                                dump_tensors_to_npz, generate_comparison_markdown_table,
                                                sync_qwen3_5_layer_weights)
from tests.utils.test_helpers import get_test_config_path


# pylint: disable=too-many-positional-arguments
def benchmark_layer_on_tpu(
    dtype_str: str,
    batch_size: int = 4,
    seq_len: int = 512,
    emb_dim: int = 2048,
    moe_mlp_dim: int = 512,
    num_experts: int = 8,
    num_experts_per_tok: int = 8,
    output_dir: str = "",
    extra_train_kwargs: dict[str, Any] | None = None,
    test_label: str = "Standard",
) -> tuple[str, dict[str, Any]]:
    """Runs 1-layer forward pass on TPU for training (Flash+SparseMoE) and inference (vLLM RPA+FusedMoE)."""
    print(
        f"\n>>> Running Qwen3.5 1-Layer Benchmark [{test_label}] in {dtype_str} on TPU..."
    )
    base_kwargs = {
        "override_model_config": True,
        "num_decoder_layers": 1,
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
        "inhomogeneous_layer_cycle_interval": 1,
        "norm_topk_prob": True,
        "float32_logits": True,
        "float32_gate_logits": True,
        "float32_weight_sum": True,
    }

    train_kwargs = dict(base_kwargs)
    train_kwargs.update({
        "megablox": True,
        "use_tokamax_gmm": True,
        "use_gmm_v2": True,
        "sparse_matmul": True,
        "wi_tile_fwd_batch_seq": 256,
        "wi_tile_fwd_embed_dim": 128,
        "wi_tile_fwd_mlp_dim": 128,
        "use_tokamax_splash": True,
        "sa_use_base2_exp": False,
        "sa_fuse_reciprocal": True,
    })
    if extra_train_kwargs:
        train_kwargs.update(extra_train_kwargs)

    cfg_train = pyconfig.initialize(
        [
            sys.argv[0],
            get_test_config_path(),
            "attention=flash",
            "use_tokamax_splash=True",
            "sa_use_base2_exp=False",
            "sa_fuse_reciprocal=True",
            "sparse_matmul=True",
            "megablox=True",
            "use_tokamax_gmm=True",
            "use_gmm_v2=True",
        ],
        weight_dtype=dtype_str,
        dtype=dtype_str,
        **train_kwargs,
    )

    # Tensor-parallel degree for the inference mesh is capped at
    # `num_kv_heads`: the RPA kernel shards the KV cache's head axis across
    # the ('model', 'expert', 'dcp') mesh axes, and that combined axis size
    # must evenly divide num_kv_heads (2 for qwen3.5-35b-a3b). It also must
    # evenly divide the (batch_size+1,)-shaped `query_start_loc` and (3,)
    # -shaped `request_distribution` AttentionMetadata arrays that shard_map
    # replicates across ('data', 'pcp', 'attn_dp', 'attn_dp_expert') -- so
    # the infer mesh cannot simply reuse the 4-device data-parallel train
    # mesh (see tests/run_qwen3_5_logit_parity.py for the reference pattern).
    infer_tp_degree = min(len(jax.devices()), cfg_train.num_kv_heads)

    cfg_infer = pyconfig.initialize(
        [
            sys.argv[0],
            get_test_config_path("inference/vllm.yml"),
            "attention=vllm_rpa",
            "prefuse_moe_weights=True",
            "model_call_mode=inference",
            f"ici_tensor_parallelism={infer_tp_degree}",
        ],
        weight_dtype=dtype_str,
        dtype=dtype_str,
        **base_kwargs,
    )

    train_devices = maxtext_utils.create_device_mesh(cfg_train)
    train_mesh = Mesh(train_devices, cfg_train.mesh_axes)

    # `create_device_mesh` requires the ICI parallelism product to equal the
    # total visible device count, which does not hold for `infer_tp_degree`
    # (<=4). Build the mesh directly from a device slice instead.
    infer_device_slice = np.array(jax.devices()[:infer_tp_degree])
    infer_mesh_shape = tuple(infer_tp_degree if axis == "model" else 1 for axis in cfg_infer.mesh_axes)
    infer_devices = infer_device_slice.reshape(infer_mesh_shape)
    infer_mesh = Mesh(infer_devices, cfg_infer.mesh_axes)

    actual_batch_size = max(len(jax.devices()), 4)

    rng = nnx.Rngs(params=42)
    train_layer = qwen3_5.Qwen3_5DecoderLayer(
        config=cfg_train,
        mesh=train_mesh,
        model_mode=MODEL_MODE_TRAIN,
        layer_idx=0,
        rngs=rng,
    )
    infer_layer = qwen3_5.Qwen3_5DecoderLayer(
        config=cfg_infer,
        mesh=infer_mesh,
        model_mode=MODEL_MODE_PREFILL,
        layer_idx=0,
        rngs=rng,
    )

    sync_qwen3_5_layer_weights(train_layer, infer_layer)

    # `sync_qwen3_5_layer_weights` does raw attribute assignment
    # (`dst_attn.query = src_attn.query`), which makes `infer_layer` share
    # the *same* nnx.Param/Variable objects as `train_layer` (not copies).
    # `nnx.state`/`nnx.update` mutate a Variable's `.value` in place, so
    # calling them on `infer_layer` after this aliasing would silently also
    # overwrite `train_layer`'s params (observed: it moved train_layer's
    # weights onto the smaller infer device slice, corrupting the training
    # pass). `nnx.split`/`nnx.merge` instead rebuilds `infer_layer` with
    # brand-new Variable objects wrapping the (device-placed) values, which
    # breaks the aliasing cleanly.
    infer_replicated_sharding = NamedSharding(infer_mesh, P())
    _infer_graphdef, _infer_state = nnx.split(infer_layer)
    _infer_state = jax.tree_util.tree_map(
        lambda x: jax.device_put(x, infer_replicated_sharding), _infer_state
    )
    infer_layer = nnx.merge(_infer_graphdef, _infer_state)

    dtype_jax = jnp.bfloat16 if dtype_str == "bfloat16" else jnp.float32
    key = jax.random.PRNGKey(101)
    inputs_np = jax.random.normal(
        key, (actual_batch_size, seq_len, emb_dim), dtype=dtype_jax
    )
    decoder_positions_np = jnp.broadcast_to(
        jnp.arange(seq_len, dtype=jnp.int32), (actual_batch_size, seq_len)
    )
    decoder_segment_ids_np = jnp.ones((actual_batch_size, seq_len), dtype=jnp.int32)

    inputs = jax.device_put(
        inputs_np, NamedSharding(train_mesh, P(("data", "fsdp"), None, None))
    )
    decoder_positions = jax.device_put(
        decoder_positions_np, NamedSharding(train_mesh, P(("data", "fsdp"), None))
    )
    decoder_segment_ids = jax.device_put(
        decoder_segment_ids_np, NamedSharding(train_mesh, P(("data", "fsdp"), None))
    )

    # Separate copies placed on `infer_mesh` for the inference forward pass.
    infer_inputs = jax.device_put(inputs_np, infer_replicated_sharding)
    infer_decoder_positions = jax.device_put(decoder_positions_np, infer_replicated_sharding)
    infer_decoder_segment_ids = jax.device_put(decoder_segment_ids_np, infer_replicated_sharding)

    print("  -> Executing Training pass (Flash Attention + Sparse MoE)...")
    jax.set_mesh(train_mesh)
    _, train_tensors = capture_qwen3_5_layer_intermediates(
        train_layer,
        inputs,
        decoder_segment_ids,
        decoder_positions,
        model_mode=MODEL_MODE_TRAIN,
    )

    print("  -> Executing Inference pass (vLLM RPA + Pallas Fused MoE)...")
    jax.set_mesh(infer_mesh)
    _, infer_tensors = capture_qwen3_5_layer_intermediates(
        infer_layer,
        infer_inputs,
        infer_decoder_segment_ids,
        infer_decoder_positions,
        model_mode=MODEL_MODE_PREFILL,
    )

    print("  -> Computing intermediate tensor drift metrics on TPU...")
    metrics = {}
    for name, t_train in train_tensors.items():
        metrics[name] = compute_drift_metrics(t_train, infer_tensors[name])
        m = metrics[name]
        print(
            f"     [{name:<25}] L_inf={m['max_abs_err']:.6e}, MAE={m['mae']:.6e}, CosSim={m['cos_sim']:.6f}"
        )

    table_md = generate_comparison_markdown_table(metrics)

    if output_dir:
        try:
            os.makedirs(output_dir, exist_ok=True)
            train_dump_path = os.path.join(
                output_dir, f"qwen3_5_train_tensors_{dtype_str}.npz"
            )
            infer_dump_path = os.path.join(
                output_dir, f"qwen3_5_infer_tensors_{dtype_str}.npz"
            )
            dump_tensors_to_npz(train_tensors, train_dump_path)
            dump_tensors_to_npz(infer_tensors, infer_dump_path)
        except Exception as e:
            print(f"     Warning: skipping full npz dump: {e}")

    import gc
    del train_tensors, infer_tensors, train_layer, infer_layer
    gc.collect()
    time.sleep(2)

    return table_md, metrics


def main():
    """Runs full Qwen3.5 1-layer numerical drift benchmarks on the local TPU VM."""
    print("=" * 80)
    print("[Local TPU VM] Running directly on locally-attached TPU chips (no SPS proxy).")
    print("=" * 80)

    results_doc_path = os.path.join(
        os.getcwd(), "docs", "qwen3_5_kernel_drift_results.md"
    )
    os.makedirs(os.path.dirname(results_doc_path), exist_ok=True)

    print(f"  JAX Platforms: {jax.config.jax_platforms}")
    print(f"  Detected TPU Devices ({len(jax.devices())}): {jax.devices()}\n")

    # 1. Baseline: BFloat16 Benchmark
    print(">>> Running Qwen3.5 1-Layer MoE Benchmark in bfloat16 on TPU...")
    b1_table, b1_metrics = benchmark_layer_on_tpu(
        dtype_str="bfloat16",
        batch_size=4,
        seq_len=512,
        emb_dim=2048,
        moe_mlp_dim=512,
        num_experts=8,
        num_experts_per_tok=8,
        output_dir="",
        test_label="BFloat16 (Tokamax Splash base-e vs vLLM RPA)",
    )

    # 2. Float32 Benchmark
    print("\n>>> Running Qwen3.5 1-Layer MoE Benchmark in float32 on TPU...")
    f32_table, f32_metrics = benchmark_layer_on_tpu(
        dtype_str="float32",
        batch_size=4,
        seq_len=512,
        emb_dim=2048,
        moe_mlp_dim=512,
        num_experts=8,
        num_experts_per_tok=8,
        output_dir="",
        test_label="Float32 (Tokamax Splash base-e vs vLLM RPA)",
    )

    time_str = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    num_devs = len(jax.devices())
    doc_content = f"""# Qwen3.5 MoE 1-Decoder Layer Kernel Drift Results

**Date / Timestamp:** {time_str}
**Hardware Platform:** Google Cloud TPU v5p (local TPU VM, locally-attached chips)
**Topology:** {num_devs} TPU Devices
**Model Architecture:** Qwen3.5 MoE (`qwen3.5-35b-a3b` 1-Layer Full Attention + MoE Block)

---

## 1. Key Component Parity Summary (BFloat16 vs. Float32)

| Component | Training Kernel | Inference Kernel | BF16 CosSim | BF16 $L_\\infty$ | BF16 MAE | FP32 CosSim | FP32 $L_\\infty$ | FP32 MAE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Pre-Attention (T01)** | Layer Input | Layer Input | **`{b1_metrics['T01_layer_input']['cos_sim']:.6f}`** | `{b1_metrics['T01_layer_input']['max_abs_err']:.2e}` | `{b1_metrics['T01_layer_input']['mae']:.2e}` | **`{f32_metrics['T01_layer_input']['cos_sim']:.6f}`** | `{f32_metrics['T01_layer_input']['max_abs_err']:.2e}` | `{f32_metrics['T01_layer_input']['mae']:.2e}` |
| **Attention Core (T12)** | Splash / Flash Attention | vLLM RPA (Pallas) | **`{b1_metrics['T12_attn_core_out']['cos_sim']:.6f}`** | `{b1_metrics['T12_attn_core_out']['max_abs_err']:.2e}` | `{b1_metrics['T12_attn_core_out']['mae']:.2e}` | **`{f32_metrics['T12_attn_core_out']['cos_sim']:.6f}`** | `{f32_metrics['T12_attn_core_out']['max_abs_err']:.2e}` | `{f32_metrics['T12_attn_core_out']['mae']:.2e}` |
| **Attention Out Proj (T14)** | Linear Projection | Linear Projection | **`{b1_metrics['T14_attn_out_proj']['cos_sim']:.6f}`** | `{b1_metrics['T14_attn_out_proj']['max_abs_err']:.2e}` | `{b1_metrics['T14_attn_out_proj']['mae']:.2e}` | **`{f32_metrics['T14_attn_out_proj']['cos_sim']:.6f}`** | `{f32_metrics['T14_attn_out_proj']['max_abs_err']:.2e}` | `{f32_metrics['T14_attn_out_proj']['mae']:.2e}` |
| **MoE Routing (T20)** | Top-K Router | Top-K Router | **`{b1_metrics['T20_router_gate_logits']['cos_sim']:.6f}`** | `{b1_metrics['T20_router_gate_logits']['max_abs_err']:.2e}` | `{b1_metrics['T20_router_gate_logits']['mae']:.2e}` | **`{f32_metrics['T20_router_gate_logits']['cos_sim']:.6f}`** | `{f32_metrics['T20_router_gate_logits']['max_abs_err']:.2e}` | `{f32_metrics['T20_router_gate_logits']['mae']:.2e}` |
| **Routed MoE Compute (T23)** | Sparse Matmul | Pallas Fused MoE | **`{b1_metrics['T23_routed_moe_out']['cos_sim']:.6f}`** | `{b1_metrics['T23_routed_moe_out']['max_abs_err']:.2e}` | `{b1_metrics['T23_routed_moe_out']['mae']:.2e}` | **`{f32_metrics['T23_routed_moe_out']['cos_sim']:.6f}`** | `{f32_metrics['T23_routed_moe_out']['max_abs_err']:.2e}` | `{f32_metrics['T23_routed_moe_out']['mae']:.2e}` |
| **Full Layer Output (T25)** | Full Decoder Layer | Full Decoder Layer | **`{b1_metrics['T25_layer_output']['cos_sim']:.6f}`** | `{b1_metrics['T25_layer_output']['max_abs_err']:.2e}` | `{b1_metrics['T25_layer_output']['mae']:.2e}` | **`{f32_metrics['T25_layer_output']['cos_sim']:.6f}`** | `{f32_metrics['T25_layer_output']['max_abs_err']:.2e}` | `{f32_metrics['T25_layer_output']['mae']:.2e}` |

---

## 2. Complete 25-Intermediate Tensor Breakdown (Float32)

{f32_table}

---

## 3. Complete 25-Intermediate Tensor Breakdown (BFloat16)

{b1_table}
"""
    with open(results_doc_path, "w", encoding="utf-8") as f:
        f.write(doc_content)

    print(f"\n✓ Results successfully saved to branch artifact: {results_doc_path}")
    print(
        "NOTE: This dump uses the AttentionMetadata construction in "
        "tests/unit/qwen3_5_layer_dump_test.py, which was fixed to use correct "
        "query_start_loc / request_distribution shapes/values. If this doc was "
        "regenerated after that fix, the RPA-derived tensor numbers above are "
        "current; otherwise treat any pre-fix copy of this file as stale."
    )


if __name__ == "__main__":
    main()
