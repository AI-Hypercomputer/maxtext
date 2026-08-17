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

"""SPS Runner for Isolated Tokamax GMM v2 vs Fused MoE Kernel Numerical Parity in Float32.

Connects to Google Cloud Shared Pathways Service (SPS) on GKE,
executes the standalone MoE kernel tests on Cloud TPU v5p,
and outputs the exact 3-way comparative error analysis across different MoE configurations.
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
from tests.unit.moe_kernel_repro_test import compare_moe_kernels_on_tpu


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
    print("STANDALONE MOE KERNEL REPRO: TOKAMAX GMM V2 VS FUSED MOE (FLOAT32)")
    print(f"Connecting to {cluster} ({tpu_instance_type} x {tpu_slice_count} slice)...")
    print("=" * 80)

    moe_configs_to_test = [
        ("Tokamax GMM v2 (Standard: 128x128 Tile)", {
            "use_tokamax_gmm": True, "use_gmm_v2": True, "megablox": True, "sparse_matmul": True
        }),
        ("Tokamax GMM v2 (Tile 256x128)", {
            "use_tokamax_gmm": True, "use_gmm_v2": True, "megablox": True, "sparse_matmul": True,
            "wi_tile_fwd_batch_seq": 256, "wi_tile_fwd_embed_dim": 128, "wi_tile_fwd_mlp_dim": 128
        }),
        ("Megablox Legacy Pallas GMM", {
            "use_tokamax_gmm": False, "use_gmm_v2": False, "megablox": True, "sparse_matmul": True
        }),
        ("Dense Einsum (XLA Reference Path)", {
            "use_tokamax_gmm": False, "use_gmm_v2": False, "megablox": False, "sparse_matmul": False
        }),
    ]

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

        print("=" * 110)
        print(">>> MOE KERNEL SWEEP (FLOAT32) [Inference = Fused MoE Kernel (tpu-inference)]")
        print("=" * 110)
        print(f"{'Configuration':<45} | {'Vs Infer L_inf':<14} | {'Vs Infer MAE':<14} | {'Vs Infer CosSim':<15} | {'Vs Ref L_inf':<14} | {'Vs Ref MAE':<12}")
        print("-" * 110)

        for name, extra_kwargs in moe_configs_to_test:
            try:
                res = compare_moe_kernels_on_tpu(
                    mesh=None,
                    batch_size=4,
                    seq_len=512,
                    emb_dim=2048,
                    moe_mlp_dim=512,
                    num_experts=8,
                    num_experts_per_tok=8,
                    dtype=jax.numpy.float32,
                    train_moe_kwargs=extra_kwargs,
                )
                m_infer = res["train_vs_infer"]
                m_ref = res["train_vs_ref"]
                print(
                    f"{name:<45} | {m_infer['max_err']:<14.2e} | {m_infer['mae']:<14.2e} | "
                    f"{m_infer['cos_sim']:<15.6f} | {m_ref['max_err']:<14.2e} | {m_ref['mae']:<12.2e}"
                )
            except Exception as e:
                print(f"{name:<45} | FAILED: {e}")

        # Baseline: Fused MoE vs Exact Reference
        try:
            m_infer_ref = res["infer_vs_ref"]
            print("-" * 110)
            print(
                f"--> INFERENCE Fused MoE vs Exact Ref (FLOAT32): L_inf={m_infer_ref['max_err']:.2e}, "
                f"MAE={m_infer_ref['mae']:.2e}, CosSim={m_infer_ref['cos_sim']:.6f}"
            )
        except Exception:
            pass


if __name__ == "__main__":
    main()
