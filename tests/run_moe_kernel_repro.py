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


def run_sweep(dtype):
    dtype_name = "FLOAT32" if dtype == jax.numpy.float32 else "BFLOAT16"
    print("=" * 80)
    print(f"STANDALONE MOE KERNEL REPRO: TOKAMAX GMM V2 VS FUSED MOE ({dtype_name})")
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

    print("=" * 110)
    print(f">>> MOE KERNEL SWEEP ({dtype_name}) [Inference = Fused MoE Kernel (tpu-inference)]")
    print("=" * 110)
    print(f"{'Configuration':<45} | {'Vs Infer L_inf':<14} | {'Vs Infer MAE':<14} | {'Vs Infer CosSim':<15} | {'Vs Ref L_inf':<14} | {'Vs Ref MAE':<12}")
    print("-" * 110)

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
                num_experts=8,
                num_experts_per_tok=8,
                dtype=dtype,
                train_moe_kwargs=extra_kwargs,
            )
            m_infer = res["train_vs_infer"]
            m_ref = res["train_vs_ref"]
            print(
                f"{name:<45} | {m_infer['max_err']:<14.2e} | {m_infer['mae']:<14.2e} | "
                f"{m_infer['cos_sim']:<15.6f} | {m_ref['max_err']:<14.2e} | {m_ref['mae']:<12.2e}"
            )
            results.append((name, m_infer, m_ref))
        except Exception as e:
            print(f"{name:<45} | FAILED: {e}")
            results.append((name, None, None, str(e)))

    # Baseline: Fused MoE vs Exact Reference
    infer_vs_ref = None
    if res is not None:
        try:
            infer_vs_ref = res["infer_vs_ref"]
            print("-" * 110)
            print(
                f"--> INFERENCE Fused MoE vs Exact Ref ({dtype_name}): L_inf={infer_vs_ref['max_err']:.2e}, "
                f"MAE={infer_vs_ref['mae']:.2e}, CosSim={infer_vs_ref['cos_sim']:.6f}"
            )
        except Exception:
            pass

    return results, infer_vs_ref


def main():
    all_results = {}
    for dtype in (jax.numpy.float32, jax.numpy.bfloat16):
        dtype_name = "float32" if dtype == jax.numpy.float32 else "bfloat16"
        all_results[dtype_name] = run_sweep(dtype)
    return all_results


if __name__ == "__main__":
    main()
