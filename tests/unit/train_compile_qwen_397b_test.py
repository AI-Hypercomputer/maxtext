# Copyright 2026 Google LLC
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

"""Test running Ahead-of-Time (AOT) train compile for Qwen3.5-397B on TPU v7x Ghostfish."""

import os
import sys
import time
import traceback
from absl.testing import absltest
import jax
from maxtext.trainers.pre_train.train_compile import main as train_compile_main


class QwenTrainCompileGhostfishTest(absltest.TestCase):

  def test_qwen35_397b_train_compile(self):
    print("=== STARTING TRAIN COMPILE ON GHOSTFISH ===", flush=True)
    print(f"JAX backend: {jax.default_backend()}", flush=True)
    print(f"JAX devices: {jax.devices()}", flush=True)

    # Locate configs directory in runfiles
    runfiles_dir = os.environ.get("PYTHON_RUNFILES") or os.environ.get("TEST_SRCDIR") or ""
    possible_paths = [
        "src/maxtext/configs",
        os.path.join(runfiles_dir, "google3/third_party/py/maxtext/src/maxtext/configs"),
        os.path.join(runfiles_dir, "third_party/py/maxtext/src/maxtext/configs"),
        "/build/work/runfiles/google3/third_party/py/maxtext/src/maxtext/configs",
    ]
    configs_dir = None
    for p in possible_paths:
      if os.path.isdir(p) and os.path.exists(os.path.join(p, "base.yml")):
        configs_dir = p
        break
    if configs_dir is None:
      configs_dir = possible_paths[0]

    os.environ["MAXTEXT_CONFIGS_DIR"] = configs_dir
    base_config_path = os.path.join(configs_dir, "base.yml")
    model_config_path = os.path.join(configs_dir, "models/qwen3.5-397b-a17b.yml")
    print(f"Using configs_dir: {configs_dir}", flush=True)
    print(f"base_config_path exists: {os.path.exists(base_config_path)}", flush=True)
    print(f"model_config_path exists: {os.path.exists(model_config_path)}", flush=True)

    compile_xla_flags = (
        "--xla_tpu_use_tc_device_shape_on_sc=true "
        "--xla_sc_disable_megacore_partitioning=true "
        "--xla_tpu_offload_gather_to_sparsecore=true "
        "--xla_tpu_enable_sparse_core_collective_offload_all_gather=true "
        "--xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true "
        "--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true "
        "--xla_tpu_use_single_sparse_core_for_all_gather_offload=true "
        "--xla_tpu_enable_concurrent_sparse_core_offloading=true "
        "--xla_tpu_aggressive_opt_barrier_removal=true "
        "--xla_tpu_scoped_vmem_limit_kib=65472"
    )

    base_args = [
        "",
        base_config_path,
        "model_name=qwen3.5-397b-a17b",
        "base_num_decoder_layers=60",
        "override_model_config=true",
        "scan_layers=True",
        "use_multimodal=false",
        "dataset_type=synthetic",
        "max_target_length=65536",
        "opt_type=adamw",
        "adam_weight_decay=0.0",
        "adam_b1=0.9",
        "adam_b2=0.999",
        "adam_eps=1e-8",
        "dtype=bfloat16",
        "mu_dtype=bfloat16",
        "megablox=true",
        "sparse_matmul=true",
        "use_tokamax_gmm=true",
        "use_gmm_v2=true",
        "use_gmm_v2_heuristic_tiling=true",
        "merge_gating_gmm=false",
        "use_ring_of_experts=true",
        "use_ragged_sort=true",
        "use_custom_sort_vjp=false",
        "ragged_buffer_factor=2.0",
        "use_random_routing=True",
        "num_moe_token_chunks=2",
        "attention=flash",
        "use_tokamax_splash=true",
        "use_splash_scheduler=true",
        "sa_block_q=512",
        "sa_block_kv=1024",
        "sa_block_kv_compute=512",
        "sa_block_q_dkv=1024",
        "sa_block_kv_dkv=2048",
        "sa_block_kv_dkv_compute=1024",
        "sa_fuse_reciprocal=false",
        "sa_use_base2_exp=true",
        "dq_reduction_steps=3",
        "gdn_chunk_size=64",
        "use_gdn_kernel=true",
        "custom_mesh_and_rule=cp-as-ep",
        "ici_tensor_parallelism=1",
        "ici_fsdp_parallelism=16",
        "ici_context_parallelism=16",
        "ici_expert_parallelism=1",
        "context_parallel_strategy=ring",
        "context_parallel_load_balance=False",
        "allow_split_physical_axes=False",
        "remat_policy=custom",
        "decoder_layer_input=device",
        "gdn=device",
        "gdn_conv=remat",
        "num_vocab_tiling=16",
        "use_iota_embed=false",
        "per_device_batch_size=0.0625",
        "gradient_accumulation_steps=32",
        "internal_compile=true",
        "internal_compile_num_devices=256",
        "compile_topology=gf=4x4x8",
        "compile_topology_num_slices=1",
        "enable_checkpointing=false",
        f"compile_xla_flags={compile_xla_flags}",
    ]

    print("\n--- STARTING QWEN3.5-397B TRAIN COMPILE (256 devices, gf=4x4x8) ---", flush=True)
    start_time = time.time()
    try:
      train_compile_main(tuple(base_args))
      compile_duration = time.time() - start_time
      print(f"\n=== QWEN3.5-397B TRAIN COMPILE FINISHED SUCCESSFULLY IN {compile_duration:.2f}s ===", flush=True)
    except Exception as e:
      print(f"\n=== QWEN3.5-397B TRAIN COMPILE FAILED: {e} ===", flush=True)
      traceback.print_exc()
      sys.stdout.flush()
      sys.stderr.flush()
      raise e


if __name__ == "__main__":
  absltest.main()
