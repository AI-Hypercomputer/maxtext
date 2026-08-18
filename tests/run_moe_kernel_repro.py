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

"""Runner for Isolated Tokamax GMM v2 vs Fused MoE Kernel Numerical Parity in Float32.

Executes the standalone MoE kernel tests directly on locally-attached Cloud
TPU v5p chips and outputs the exact 3-way comparative error analysis across
different MoE configurations.
"""

import os
import sys

os.environ["NEW_MODEL_DESIGN"] = "1"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["VLLM_TARGET_DEVICE"] = "tpu"

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("src"))

import jax

# Ensure Mosaic Pallas TPU lowering is registered
try:
    from jax._src.pallas.mosaic import lowering as _mosaic_lowering
except ImportError:
    pass

from tests.unit.moe_kernel_repro_test import compare_moe_kernels_on_tpu


import argparse

def run_sweep(dtype, topk: int = 2, num_experts: int = 8):
    dtype_name = "FLOAT32" if dtype == jax.numpy.float32 else "BFLOAT16"
    routing_type = f"Sparse Top-{topk}/{num_experts}" if topk < num_experts else f"Dense Top-{topk}/{num_experts}"
    print("=" * 80)
    print(f"STANDALONE MOE KERNEL REPRO: TOKAMAX GMM V2 VS FUSED MOE ({dtype_name}, {routing_type})")
    print("[Local TPU VM] Running directly on locally-attached TPU chips.")
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

    print("=" * 125)
    print(f">>> MOE KERNEL SWEEP ({dtype_name}, {routing_type}) [Inference = Fused MoE Kernel (tpu-inference)]")
    print("=" * 125)
    print(f"{'Configuration':<42} | {'Vs Infer L_inf':<14} | {'Vs Infer MAE':<14} | {'Vs Infer CosSim':<15} | {'Vs Ref L_inf':<14} | {'Vs Ref MAE':<12} | {'Routing Parity':<14}")
    print("-" * 125)

    results = []
    res = None
    for name, extra_kwargs in moe_configs_to_test:
        try:
            res = compare_moe_kernels_on_tpu(
                mesh=None,
                batch_size=4,
                seq_len=512,
                emb_dim=2048,
                moe_mlp_dim=512,
                num_experts=num_experts,
                num_experts_per_tok=topk,
                dtype=dtype,
                train_moe_kwargs=extra_kwargs,
            )
            m_infer = res["train_vs_infer"]
            m_ref = res["train_vs_ref"]
            r_parity = res.get("routing_parity", 1.0)
            print(
                f"{name:<42} | {m_infer['max_err']:<14.2e} | {m_infer['mae']:<14.2e} | "
                f"{m_infer['cos_sim']:<15.6f} | {m_ref['max_err']:<14.2e} | {m_ref['mae']:<12.2e} | {r_parity * 100:<13.2f}%"
            )
            results.append((name, m_infer, m_ref, r_parity))
        except Exception as e:
            print(f"{name:<42} | FAILED: {e}")
            results.append((name, None, None, 0.0, str(e)))

    # Baseline: Fused MoE vs Exact Reference
    infer_vs_ref = None
    if res is not None:
        try:
            infer_vs_ref = res["infer_vs_ref"]
            print("-" * 125)
            print(
                f"--> INFERENCE Fused MoE vs Exact Ref ({dtype_name}): L_inf={infer_vs_ref['max_err']:.2e}, "
                f"MAE={infer_vs_ref['mae']:.2e}, CosSim={infer_vs_ref['cos_sim']:.6f}"
            )
        except Exception:
            pass

    return results, infer_vs_ref


def main():
    parser = argparse.ArgumentParser(description="MoE Kernel Parity Repro Runner")
    parser.add_argument("--topk", type=int, default=None, help="Top-K experts per token to test (e.g. 2 for sparse, 8 for dense). If omitted, sweeps both 2 and 8.")
    parser.add_argument("--num_experts", type=int, default=8, help="Total number of experts (default: 8)")
    args = parser.parse_args()

    topk_list = [args.topk] if args.topk is not None else [2, 8]

    all_results = {}
    for topk in topk_list:
        print("\n" + "#" * 125)
        print(f"### SWEEPING TOP-{topk}/{args.num_experts} ROUTING ({'SPARSE - REALISTIC FOR RL' if topk < args.num_experts else 'DENSE - KERNEL MATH ONLY'}) ###")
        print("#" * 125 + "\n")
        for dtype in (jax.numpy.float32, jax.numpy.bfloat16):
            dtype_name = "float32" if dtype == jax.numpy.float32 else "bfloat16"
            all_results[f"{dtype_name}_top{topk}"] = run_sweep(dtype, topk=topk, num_experts=args.num_experts)
    return all_results


if __name__ == "__main__":
    main()
