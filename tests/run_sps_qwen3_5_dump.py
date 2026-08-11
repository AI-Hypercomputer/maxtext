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

"""Programmatic SPS launcher to run and benchmark Qwen3.5 MoE 1-Layer

Intermediate Tensor & Logits Dumps on Cloud TPU v5p over GKE.
"""

import os
import sys

os.environ["NEW_MODEL_DESIGN"] = "1"

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("src"))

import subprocess
import time
from typing import Any

import jax
import jax.numpy as jnp
import pathwaysutils.proxy_backend
from flax import nnx
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from pathwaysutils.experimental.shared_pathways_service import gke_utils, isc_pathways

pathwaysutils.proxy_backend.register_backend_factory()

from maxtext.common.common_types import MODEL_MODE_PREFILL, MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.models import qwen3_5
from maxtext.utils import maxtext_utils
from tests.unit.qwen3_5_layer_dump_test import (capture_qwen3_5_layer_intermediates, compute_drift_metrics,
                                                dump_tensors_to_npz, generate_comparison_markdown_table,
                                                sync_qwen3_5_layer_weights)
from tests.utils.test_helpers import get_test_config_path


# --- Monkey-Patch 300s Pod Timeout ---
def custom_check_pod_ready(pod_name: str) -> str:
    """Extends kubectl wait timeout to 300s for slow image pulls / cluster scheduling."""
    target = f"pod/{pod_name}" if not pod_name.startswith("pod/") else pod_name
    print(f"[SPS Launcher] Waiting up to 300s for {target} to be ready...")
    wait_command = [
        "kubectl",
        "wait",
        "--for=condition=Ready",
        "--timeout=300s",
        "--",
        target,
    ]
    subprocess.run(wait_command, check=True)
    return pod_name


gke_utils.check_pod_ready = custom_check_pod_ready
# -------------------------------------


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
    }

    train_kwargs = dict(base_kwargs)
    if extra_train_kwargs:
        train_kwargs.update(extra_train_kwargs)

    cfg_train = pyconfig.initialize(
        [sys.argv[0], get_test_config_path(), "attention=flash", "sparse_matmul=True"],
        weight_dtype=dtype_str,
        dtype=dtype_str,
        **train_kwargs,
    )

    cfg_infer = pyconfig.initialize(
        [
            sys.argv[0],
            get_test_config_path("inference/vllm.yml"),
            "attention=vllm_rpa",
            "prefuse_moe_weights=True",
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

    dtype_jax = jnp.bfloat16 if dtype_str == "bfloat16" else jnp.float32
    key = jax.random.PRNGKey(101)
    inputs = jax.random.normal(
        key, (actual_batch_size, seq_len, emb_dim), dtype=dtype_jax
    )
    decoder_positions = jnp.broadcast_to(
        jnp.arange(seq_len, dtype=jnp.int32), (actual_batch_size, seq_len)
    )
    decoder_segment_ids = jnp.ones((actual_batch_size, seq_len), dtype=jnp.int32)

    inputs = jax.device_put(
        inputs, NamedSharding(train_mesh, P(("data", "fsdp"), None, None))
    )
    decoder_positions = jax.device_put(
        decoder_positions, NamedSharding(train_mesh, P(("data", "fsdp"), None))
    )
    decoder_segment_ids = jax.device_put(
        decoder_segment_ids, NamedSharding(train_mesh, P(("data", "fsdp"), None))
    )

    print("  -> Executing Training pass (Flash Attention + Sparse MoE)...")
    _, train_tensors = capture_qwen3_5_layer_intermediates(
        train_layer,
        inputs,
        decoder_segment_ids,
        decoder_positions,
        model_mode=MODEL_MODE_TRAIN,
    )

    print("  -> Executing Inference pass (vLLM RPA + Pallas Fused MoE)...")
    _, infer_tensors = capture_qwen3_5_layer_intermediates(
        infer_layer,
        inputs,
        decoder_segment_ids,
        decoder_positions,
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

    return table_md, metrics


def main():
    """Connects to SPS cluster and runs full Qwen3.5 1-layer numerical drift benchmarks."""
    cluster = "auto-v5p-8-bodaborg"
    project = "cloud-tpu-multipod-dev"
    region = "europe-west4"
    gcs_bucket = "gs://cloud-pathways-staging/mohit-scratch"
    pathways_service = "sps-mohit-pathways-head-0-0.sps-mohit:29001"
    tpu_instance_type = "tpuv5:2x2x1"
    tpu_slice_count = 1
    proxy_server_image = (
        "us-docker.pkg.dev/cloud-tpu-v2-images/pathways/proxy_server@"
        "sha256:cca2c7eeb5d6b1f49a7619d078e74ef4d0ef2d6129d7ac9fb36b8c937194204b"
    )

    print("=" * 80)
    print(
        f"[SPS Launcher] Connecting to {cluster} ({tpu_instance_type} x {tpu_slice_count} slice)..."
    )
    print("=" * 80)

    results_doc_path = os.path.join(
        os.getcwd(), "docs", "qwen3_5_kernel_drift_results.md"
    )
    os.makedirs(os.path.dirname(results_doc_path), exist_ok=True)

    with isc_pathways.connect(
        cluster=cluster,
        project=project,
        region=region,
        gcs_bucket=gcs_bucket,
        pathways_service=pathways_service,
        expected_tpu_instances={tpu_instance_type: tpu_slice_count},
        proxy_server_image=proxy_server_image,
        collect_service_metrics=True,
    ):
        print("✓ Successfully connected to SPS Cloud TPU v5p!")
        print(f"  JAX Platforms: {jax.config.jax_platforms}")
        print(f"  Detected TPU Devices ({len(jax.devices())}): {jax.devices()}\n")

        # 1. Baseline: Default Splash Attention (Block 512)
        print(">>> [Run 1/3] Baseline: Default Splash Attention (Block 512)...")
        b1_table, b1_metrics = benchmark_layer_on_tpu(
            dtype_str="bfloat16",
            batch_size=4,
            seq_len=512,
            emb_dim=2048,
            moe_mlp_dim=512,
            num_experts=8,
            num_experts_per_tok=8,
            output_dir="",
            test_label="Baseline (Block 512)",
        )

        # 2. Option 2: Tile Alignment (sa_block_q=128, sa_block_kv=128, sa_block_kv_compute=128)
        print(">>> [Run 2/3] Option 2: Tile Alignment (Block 128x128)...")
        b2_table, b2_metrics = benchmark_layer_on_tpu(
            dtype_str="bfloat16",
            batch_size=4,
            seq_len=512,
            emb_dim=2048,
            moe_mlp_dim=512,
            num_experts=8,
            num_experts_per_tok=8,
            output_dir="",
            extra_train_kwargs={
                "sa_block_q": 128,
                "sa_block_kv": 128,
                "sa_block_kv_compute": 128,
            },
            test_label="Option 2 (Tile Alignment 128)",
        )

        # 3. Option 3: Tile Alignment + Exact Math (use_tokamax_splash, use_base2_exp=False, fuse_reciprocal=False)
        print(">>> [Run 3/3] Option 3: Tile Alignment 128 + Exact Softmax Math...")
        b3_table, b3_metrics = benchmark_layer_on_tpu(
            dtype_str="bfloat16",
            batch_size=4,
            seq_len=512,
            emb_dim=2048,
            moe_mlp_dim=512,
            num_experts=8,
            num_experts_per_tok=8,
            output_dir="",
            extra_train_kwargs={
                "sa_block_q": 128,
                "sa_block_kv": 128,
                "sa_block_kv_compute": 128,
                "use_tokamax_splash": True,
                "sa_use_base2_exp": False,
                "sa_fuse_reciprocal": False,
            },
            test_label="Option 3 (Tile 128 + Exact Math)",
        )

        print("\n" + "=" * 80)
        print("COMPARATIVE EVALUATION SUMMARY (Attention Drift Reduction)")
        print("=" * 80)
        print(
            f"{'Configuration':<35} | {'T12 Core L_inf':<15} | {'T14 OutProj L_inf':<18} | {'T25 Layer CosSim':<16}"
        )
        print("-" * 90)
        for label, m in [
            ("1. Baseline (Block 512)", b1_metrics),
            ("2. Option 2 (Tile 128x128)", b2_metrics),
            ("3. Option 3 (Tile 128 + Exact Math)", b3_metrics),
        ]:
            print(
                f"{label:<35} | {m['T12_attn_core_out']['max_abs_err']:<15.6e} | "
                f"{m['T14_attn_out_proj']['max_abs_err']:<18.6e} | "
                f"{m['T25_layer_output']['cos_sim']:<16.6f}"
            )

        time_str = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        num_devs = len(jax.devices())
        doc_content = f"""# Qwen3.5 MoE 1-Decoder Layer Kernel Drift Results

**Date / Timestamp:** {time_str}  
**Hardware Platform:** Google Cloud TPU v5p (Shared Pathways Service over GKE `auto-v5p-8-bodaborg`)  
**Topology:** 2x2x1 ({num_devs} TPU Devices)  
**Model Architecture:** Qwen3.5 MoE (`qwen3.5-35b-a3b` 1-Layer Full Attention + MoE Block)  
**Evaluated Precision:** `bfloat16`  

---

## 1. Attention Precision & Tiling Comparative Analysis

| Configuration | `T12_attn_core_out` ($L_\\infty$) | `T14_attn_out_proj` ($L_\\infty$) | `T25_layer_output` (CosSim) |
| :--- | :--- | :--- | :--- |
| **Baseline (Splash Block 512)** | `{b1_metrics['T12_attn_core_out']['max_abs_err']:.6e}` | `{b1_metrics['T14_attn_out_proj']['max_abs_err']:.6e}` | `{b1_metrics['T25_layer_output']['cos_sim']:.6f}` |
| **Option 2 (Tile Alignment 128x128)** | `{b2_metrics['T12_attn_core_out']['max_abs_err']:.6e}` | `{b2_metrics['T14_attn_out_proj']['max_abs_err']:.6e}` | `{b2_metrics['T25_layer_output']['cos_sim']:.6f}` |
| **Option 3 (Tile 128 + Exact Math)** | `{b3_metrics['T12_attn_core_out']['max_abs_err']:.6e}` | `{b3_metrics['T14_attn_out_proj']['max_abs_err']:.6e}` | `{b3_metrics['T25_layer_output']['cos_sim']:.6f}` |

---

## 2. Baseline Full 25-Tensor Breakdown (BFloat16)

{b1_table}
"""
        with open(results_doc_path, "w", encoding="utf-8") as f:
            f.write(doc_content)

        print(f"\n✓ Results successfully saved to branch artifact: {results_doc_path}")


if __name__ == "__main__":
    main()
