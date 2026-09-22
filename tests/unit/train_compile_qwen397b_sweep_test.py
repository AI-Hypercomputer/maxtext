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

"""Parameterized AOT compilation sweep for Qwen3.5-397B on Cloud TPU v7x Ghostfish.

Sweeps across:
1. Hardware mesh partitions: (FSDP=16, CP=16, EP=1), (FSDP=8, CP=32, EP=1), (FSDP=8, CP=16, EP=2)
2. Micro-batch sizes: 1x, 2x, 4x sequences per FSDP replica (GA from 128 down to 16)
3. Rematerialization policies: custom_gdn, custom_gdn_ctx, save_qkv
4. MoE token chunking: chunks=2, chunks=4

Total test combinations: 54. Executed in parallel across 27 Ghostfish shards on Forge.
"""

import os
import sys
import time
import traceback
from absl.testing import absltest
from absl.testing import parameterized
from maxtext.src.maxtext.trainers.pre_train.train_compile import main as train_compile_main


def _build_sweep_parameters():
  """Constructs the 54 sweep configurations."""
  configs = []

  # Mesh Partitions
  mesh_configs = [
      # (name, fsdp, cp, ep, mbs_list, ga_list)
      # Proven winner in XM 291648413 (FSDP*EP=64 requires MBS=64):
      ("m_32_4_2", 32, 4, 2, [64], [16]),
      ("m_16_16_1", 16, 16, 1, [16, 32, 64], [64, 32, 16]),
      ("m_8_32_1", 8, 32, 1, [8, 16, 32], [128, 64, 32]),
      ("m_8_16_2", 8, 16, 2, [8, 16, 32], [128, 64, 32]),
  ]

  # 3 Remat Policies
  remat_options = [
      ("custom_gdn", "custom", "device", "device", "device", "remat"),
      ("custom_gdn_ctx", "custom", "device", "device", "device", "device"),
      ("save_qkv", "save_qkv_proj", "device", "device", "device", "remat"),
  ]

  # 2 MoE Chunking Options
  chunk_options = [2, 4]

  for mesh_name, fsdp, cp, ep, mbs_list, ga_list in mesh_configs:
    for mbs, ga in zip(mbs_list, ga_list):
      # Pruning rules based on empirical Ghostfish hardware verification:
      # 1. MoE divisibility: MBS must be an integer multiple of (fsdp * ep)
      if mbs % (fsdp * ep) != 0:
        continue
      # 2. SparseCore SPMEM limit: chunks=2 overflows tile_spmem when mbs >= 32
      if chunks == 2 and mbs >= 32:
        continue

      for remat_name, remat_pol, dec_input, gdn_loc, gdn_conv_loc, ctx_loc in remat_options:
        for chunks in chunk_options:
          test_name = f"{mesh_name}_mbs{mbs}_remat_{remat_name}_c{chunks}"
          configs.append((
              test_name,
              fsdp,
              cp,
              ep,
              mbs,
              ga,
              remat_pol,
              dec_input,
              gdn_loc,
              gdn_conv_loc,
              ctx_loc,
              chunks,
          ))

  return configs


def _get_peak_memory_gb(compiled) -> float:
  """Extracts peak HBM memory in GB from compiled memory analysis."""
  if not hasattr(compiled, "memory_analysis"):
    return 0.0
  mem_obj = compiled.memory_analysis()
  if hasattr(mem_obj, "peak_memory_in_bytes"):
    return float(mem_obj.peak_memory_in_bytes) / (1024.0**3)
  if hasattr(mem_obj, "cumulative_size_in_bytes"):
    return float(mem_obj.cumulative_size_in_bytes) / (1024.0**3)
  if isinstance(mem_obj, dict):
    b = mem_obj.get("peak_memory_in_bytes", 0) or mem_obj.get(
        "cumulative_size_in_bytes", 0
    )
    return float(b) / (1024.0**3)
  return 0.0


_SWEEP_CONFIGS = _build_sweep_parameters()


class Qwen397bThroughputSweepTest(parameterized.TestCase):
  """Parallelized sweep evaluating throughput and memory scaling on TPU v7x Ghostfish."""

  @parameterized.named_parameters(*_SWEEP_CONFIGS)
  def test_qwen35_397b_sweep(
      self,
      fsdp: int,
      cp: int,
      ep: int,
      mbs: int,
      ga: int,
      remat_policy: str,
      decoder_layer_input: str,
      gdn: str,
      gdn_conv: str,
      context: str,
      chunks: int,
  ):
    # Resolve configs directory
    configs_dir = os.environ.get("MAXTEXT_CONFIGS_DIR")
    possible_paths = [
        "third_party/py/maxtext/src/maxtext/configs",
        "src/maxtext/configs",
        os.path.join(os.environ.get("TEST_SRCDIR", ""), "google3/third_party/py/maxtext/src/maxtext/configs"),
    ]
    for p in possible_paths:
      if os.path.isdir(p) and os.path.exists(os.path.join(p, "base.yml")):
        configs_dir = p
        break
    if configs_dir is None:
      configs_dir = possible_paths[0]

    os.environ["MAXTEXT_CONFIGS_DIR"] = configs_dir
    base_config_path = os.path.join(configs_dir, "base.yml")

    # 10 verified production SparseCore compiler flags
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

    pdbs = mbs / 256.0

    args = [
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
        f"num_moe_token_chunks={chunks}",
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
        "gdn_cp_mode=auto",
        f"custom_mesh_and_rule={'cp-as-ep' if ep == 1 else ''}",
        "ici_tensor_parallelism=1",
        f"ici_fsdp_parallelism={fsdp}",
        f"ici_context_parallelism={cp}",
        f"ici_expert_parallelism={ep}",
        "context_parallel_strategy=ring",
        "context_parallel_load_balance=False",
        "allow_split_physical_axes=False",
        f"remat_policy={remat_policy}",
        f"decoder_layer_input={decoder_layer_input}",
        f"gdn={gdn}",
        f"gdn_conv={gdn_conv}",
        f"context={context}",
        "num_vocab_tiling=16",
        "use_iota_embed=false",
        f"per_device_batch_size={pdbs}",
        f"gradient_accumulation_steps={ga}",
        "internal_compile=true",
        "internal_compile_num_devices=256",
        "compile_topology=gf=4x4x8",
        "compile_topology_num_slices=1",
        "enable_checkpointing=false",
        f"compile_xla_flags={compile_xla_flags}",
    ]

    print(
        f"\n=== LAUNCHING SWEEP COMBINATION: FSDP={fsdp}, CP={cp}, EP={ep}, MBS={mbs} "
        f"(pdbs={pdbs}), GA={ga}, remat={remat_policy} (gdn={gdn}, ctx={context}), "
        f"chunks={chunks} ===",
        flush=True,
    )

    t0 = time.time()
    try:
      compiled = train_compile_main(tuple(args))
      duration = time.time() - t0
      print(
          f"\n=== SWEEP BASE COMBINATION PASSED in {duration:.2f}s: "
          f"FSDP={fsdp}, CP={cp}, EP={ep}, MBS={mbs}, GA={ga}, "
          f"remat={remat_policy} ===",
          flush=True,
      )

      # Extract memory analysis from JAX compiled object
      peak_gb = _get_peak_memory_gb(compiled)
      leftover_gb = 94.7 - peak_gb if peak_gb > 0 else 0.0
      print(
          f"  [MEMORY ANALYSIS] Peak HBM: {peak_gb:.2f} GB / 94.70 GB | "
          f"Leftover HBM: {leftover_gb:.2f} GB",
          flush=True,
      )

      # Strategy A: Intra-Shard Adaptive Cascade
      # 1. Attention context caching promotion
      if leftover_gb >= 15.0 and context == "remat":
        print(
            "\n>>> [ADAPTIVE CASCADE - LEVEL 2] Leftover"
            f" HBM={leftover_gb:.2f} GB >= 15 GB! Cascading to test"
            " context=device...",
            flush=True,
        )
        args_lvl2 = [
            a if not a.startswith("context=") else "context=device"
            for a in args
        ]
        try:
          compiled_lvl2 = train_compile_main(tuple(args_lvl2))
          peak_lvl2_gb = _get_peak_memory_gb(compiled_lvl2)
          leftover_lvl2_gb = 94.7 - peak_lvl2_gb
          print(
              f"  >>> [LEVEL 2 PASSED] context=device Peak HBM:"
              f" {peak_lvl2_gb:.2f} GB | Leftover: {leftover_lvl2_gb:.2f} GB",
              flush=True,
          )

          # 2. Linear projection caching promotion
          if leftover_lvl2_gb >= 20.0:
            print(
                "\n>>> [ADAPTIVE CASCADE - LEVEL 3] Leftover"
                f" HBM={leftover_lvl2_gb:.2f} GB >= 20 GB! Cascading to test"
                " remat_policy=save_qkv_proj...",
                flush=True,
            )
            args_lvl3 = [
                a if not a.startswith("remat_policy=")
                else "remat_policy=save_qkv_proj"
                for a in args_lvl2
            ]
            compiled_lvl3 = train_compile_main(tuple(args_lvl3))
            peak_lvl3_gb = _get_peak_memory_gb(compiled_lvl3)
            print(
                "  >>> [LEVEL 3 PASSED] save_qkv_proj Peak HBM:"
                f" {peak_lvl3_gb:.2f} GB | Leftover:"
                f" {94.7 - peak_lvl3_gb:.2f} GB",
                flush=True,
            )
        except Exception as e_cascade:  # pylint: disable=broad-exception-caught
          print(
              f"  >>> [REMAT CASCADE REACHED CAPACITY] {e_cascade}", flush=True
          )

      # 3. Batch expansion promotion (halve GA steps)
      if mbs == 32 and leftover_gb >= 35.0:
        print(
            "\n>>> [ADAPTIVE CASCADE - BATCH EXPANSION] Leftover"
            f" HBM={leftover_gb:.2f} GB >= 35 GB! Cascading to test MBS=64,"
            " GA=16 (cuts outer loop by 2x!)...",
            flush=True,
        )
        args_mbs64 = [
            a if not a.startswith("per_device_batch_size=")
            else "per_device_batch_size=0.25"
            for a in args
        ]
        args_mbs64 = [
            a if not a.startswith("gradient_accumulation_steps=")
            else "gradient_accumulation_steps=16"
            for a in args_mbs64
        ]
        try:
          compiled_mbs64 = train_compile_main(tuple(args_mbs64))
          peak_mbs64_gb = _get_peak_memory_gb(compiled_mbs64)
          print(
              f"  >>> [MBS=64 PASSED] Peak HBM: {peak_mbs64_gb:.2f} GB | "
              f"Leftover: {94.7 - peak_mbs64_gb:.2f} GB",
              flush=True,
          )
        except Exception as e_mbs:  # pylint: disable=broad-exception-caught
          print(f"  >>> [MBS=64 CASCADE REACHED CAPACITY] {e_mbs}", flush=True)

    except Exception as e:  # pylint: disable=broad-exception-caught
      duration = time.time() - t0
      print(
          f"\n=== SWEEP COMBINATION FAILED after {duration:.2f}s: FSDP={fsdp}, CP={cp}, EP={ep}, "
          f"MBS={mbs}, remat={remat_policy}: {e} ===",
          flush=True,
      )
      traceback.print_exc()
      sys.stdout.flush()
      sys.stderr.flush()
      err_str = str(e)
      if "RESOURCE_EXHAUSTED" in err_str or "out of memory" in err_str:
        print(
            f"  >>> [CAPACITY BOUNDARY LOGGED] HBM capacity exceeded: {e}",
            flush=True,
        )
      else:
        raise e


if __name__ == "__main__":
  absltest.main()
