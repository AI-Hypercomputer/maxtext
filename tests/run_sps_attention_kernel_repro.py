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

"""SPS Runner for Isolated Splash vs RPA Attention Kernel Numerical Parity.

Connects to Google Cloud Shared Pathways Service (SPS) on GKE,
executes the standalone attention kernel test on Cloud TPU v5p,
and outputs the exact 3-way comparative error analysis.
"""

import os
import sys

os.environ["NEW_MODEL_DESIGN"] = "1"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["VLLM_TARGET_DEVICE"] = "tpu"

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("src"))

import time
import jax
import numpy as np
import pathwaysutils.proxy_backend

pathwaysutils.proxy_backend.register_backend_factory()

# Ensure Mosaic Pallas TPU lowering is registered for SPS client
try:
    from jax._src.pallas.mosaic import lowering as _mosaic_lowering
except ImportError:
    pass

from pathwaysutils.experimental.shared_pathways_service import isc_pathways
from tests.unit.attention_kernel_repro_test import compare_attention_kernels_on_tpu


def print_metrics_table(label: str, metrics: dict):
    print(f"\n--- {label} ---")
    print(f"  Max Absolute Error (L_inf): {metrics['max_abs_err']:.6e}")
    print(f"  Mean Absolute Error (MAE)  : {metrics['mae']:.6e}")
    print(f"  Mean Squared Error (MSE)   : {metrics['mse']:.6e}")
    print(f"  Cosine Similarity          : {metrics['cos_sim']:.6f}")
    print(f"  Relative Error (L2 norm)   : {metrics['rel_err']:.6e}")


def main():
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
    print("STANDALONE ATTENTION KERNEL REPRO TEST: SPLASH ATTENTION VS. RPA")
    print(f"Connecting to {cluster} ({tpu_instance_type} x {tpu_slice_count} slice)...")
    print("=" * 80)

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
        print("✓ Connected to SPS Cloud TPU v5p!\n")
        print(">>> Running Attention Kernel Comparison (Qwen3.5 Shape: B=4, S=512, H_q=16, H_kv=2, D=256)...")
        results = compare_attention_kernels_on_tpu(
            batch_size=4,
            seq_len=512,
            num_query_heads=16,
            num_kv_heads=2,
            head_dim=256,
            dtype_str="bfloat16",
            block_size=128,
        )

        m_splash_rpa = results["splash_vs_rpa"]

        print("=" * 80)
        print("ISOLATED ATTENTION KERNEL PARITY: SPLASH ATTENTION VS. RPA")
        print("=" * 80)
        print_metrics_table("Splash Attention (Training) vs. RPA (Inference)", m_splash_rpa)

        print("\n" + "=" * 80)
        print("ATTENTION KERNEL PARITY SUMMARY")
        print("=" * 80)
        print(f"{'Comparison':<42} | {'Max Abs Err (L_inf)':<20} | {'MAE':<15} | {'Cosine Sim':<12}")
        print("-" * 95)
        print(f"{'Splash Attention vs. RPA (Serving)':<42} | {m_splash_rpa['max_abs_err']:<20.6e} | {m_splash_rpa['mae']:<15.6e} | {m_splash_rpa['cos_sim']:<12.6f}")

        # Save standalone report
        doc_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "docs",
            "attention_kernel_repro_results.md",
        )
        time_str = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        doc = f"""# Isolated Attention Kernel Parity: Splash Attention vs. RPA

**Date:** {time_str}  
**Hardware:** Google Cloud TPU v5p (`{cluster}`)  
**Configuration:** `batch_size=4`, `seq_len=512`, `num_query_heads=16`, `num_kv_heads=2`, `head_dim=256`, `dtype=bfloat16`  

---

## 1. Direct Comparative Parity

| Comparison Pair | Max Abs Error ($L_\\infty$) | MAE | MSE | Cosine Similarity | Relative Error |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Splash Attn (Train) vs. RPA (Infer)** | `{m_splash_rpa['max_abs_err']:.6e}` | `{m_splash_rpa['mae']:.6e}` | `{m_splash_rpa['mse']:.6e}` | **`{m_splash_rpa['cos_sim']:.6f}`** | `{m_splash_rpa['rel_err']:.6e}` |

---

## 2. Key Diagnostic Takeaway

1. **Kernel Disparity Root Cause:** By isolating $(Q, K, V)$ to identical synthetic inputs, all outer network operations (projections, layernorms, RoPE, gating, and MoE) are completely eliminated.
2. **Current Metric:** Splash Attention and RPA produce a baseline cosine similarity of **{m_splash_rpa['cos_sim']*100:.2f}%** on identical inputs.
"""
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write(doc)
        print(f"\n✓ Repro results successfully written to: {doc_path}")


if __name__ == "__main__":
    main()
