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

"""Runner for Isolated Splash vs RPA Attention Kernel Numerical Parity.

Executes the standalone attention kernel test directly on locally-attached
Cloud TPU v5p chips and outputs the exact 3-way comparative error analysis.
"""

import os
import sys

os.environ["NEW_MODEL_DESIGN"] = "1"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["VLLM_TARGET_DEVICE"] = "tpu"

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("src"))

# Ensure Mosaic Pallas TPU lowering is registered
try:
    from jax._src.pallas.mosaic import lowering as _mosaic_lowering
except ImportError:
    pass

from tests.unit.attention_kernel_repro_test import compare_attention_kernels_on_tpu


def print_metrics_table(label: str, metrics: dict):
    print(f"\n--- {label} ---")
    print(f"  Max Absolute Error (L_inf): {metrics['max_abs_err']:.6e}")
    print(f"  Mean Absolute Error (MAE)  : {metrics['mae']:.6e}")
    print(f"  Mean Squared Error (MSE)   : {metrics['mse']:.6e}")
    print(f"  Cosine Similarity          : {metrics['cos_sim']:.6f}")
    print(f"  Relative Error (L2 norm)   : {metrics['rel_err']:.6e}")


def main():
    print("=" * 80)
    print("STANDALONE ATTENTION KERNEL REPRO: SPLASH VS RPA CONFIGURATION SWEEP")
    print("[Local TPU VM] Running directly on locally-attached TPU chips.")
    print("=" * 80)

    configs_to_test = [
        ("JAX Splash Attention (Legacy Default)", ["use_tokamax_splash=False"]),
        ("Tokamax Splash (Default: base2_exp=True, fuse_recip=True)", [
            "use_tokamax_splash=True", "sa_use_base2_exp=True", "sa_fuse_reciprocal=True"
        ]),
        ("Tokamax Splash (base2_exp=False, fuse_recip=True)", [
            "use_tokamax_splash=True", "sa_use_base2_exp=False", "sa_fuse_reciprocal=True"
        ]),
        ("Tokamax Splash (base2_exp=True, fuse_recip=False)", [
            "use_tokamax_splash=True", "sa_use_base2_exp=True", "sa_fuse_reciprocal=False"
        ]),
        ("Tokamax Splash (base2_exp=False, fuse_recip=False)", [
            "use_tokamax_splash=True", "sa_use_base2_exp=False", "sa_fuse_reciprocal=False"
        ]),
        ("Tokamax Splash (BlockSize=256)", [
            "use_tokamax_splash=True", "sa_block_q=256", "sa_block_kv=256", "sa_block_kv_compute=256"
        ]),
    ]

    for infer_attn in ["vllm_rpa", "vllm_batched_rpa"]:
        infer_label = "DEFAULT RPA (v3)" if infer_attn == "vllm_rpa" else "BATCHED RPA (Target)"
        print("\n" + "#" * 90)
        print(f"### INFERENCE KERNEL: {infer_label}")
        print("#" * 90)

        for dtype_str in ["float32", "bfloat16"]:
            print("=" * 90)
            print(f">>> ATTENTION KERNEL SWEEP ({dtype_str.upper()}) [Inference = {infer_label}]")
            print("=" * 90)

            print(f"{'Configuration':<52} | {'Vs RPA L_inf':<12} | {'Vs RPA MAE':<12} | {'Vs RPA CosSim':<13} | {'Vs Ref MAE':<12}")
            print("-" * 115)

            for cfg_name, extra_args in configs_to_test:
                try:
                    res = compare_attention_kernels_on_tpu(
                        batch_size=4,
                        seq_len=512,
                        num_query_heads=16,
                        num_kv_heads=2,
                        head_dim=256,
                        dtype_str=dtype_str,
                        block_size=128,
                        extra_train_args=extra_args,
                        infer_attention=infer_attn,
                    )
                    m_rpa = res["splash_vs_rpa"]
                    m_ref = res["splash_vs_ref"]
                    print(
                        f"{cfg_name:<52} | "
                        f"{m_rpa['max_abs_err']:<12.2e} | "
                        f"{m_rpa['mae']:<12.2e} | "
                        f"{m_rpa['cos_sim']:<13.6f} | "
                        f"{m_ref['mae']:<12.2e}"
                    )
                except Exception as e:
                    print(f"{cfg_name:<52} | FAILED: {e}")

            # Print RPA vs Ref
            try:
                res_ref = compare_attention_kernels_on_tpu(
                    batch_size=4,
                    seq_len=512,
                    num_query_heads=16,
                    num_kv_heads=2,
                    head_dim=256,
                    dtype_str=dtype_str,
                    block_size=128,
                    extra_train_args=["use_tokamax_splash=False"],
                    infer_attention=infer_attn,
                )
                rpa_ref = res_ref["rpa_vs_ref"]
                print(f"--> {infer_label} vs Exact Ref ({dtype_str.upper()}): L_inf={rpa_ref['max_abs_err']:.2e}, MAE={rpa_ref['mae']:.2e}, CosSim={rpa_ref['cos_sim']:.6f}")
            except Exception as e:
                print(f"--> {infer_label} vs Exact Ref FAILED: {e}")


if __name__ == "__main__":
    main()
