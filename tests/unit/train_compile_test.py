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

"""Tests for the Ahead-of-Time (AOT) compilation script.

This module contains unit tests for `train_compile.py`, ensuring that various
model configurations and parallelism strategies can be successfully compiled
for different hardware topologies.

Most tests shrink the model widths, vocabulary, sequence length and topology to the smallest values that still exercise
the code path and mesh axes under test: the AOT compile time is dominated by width and topology size, not by the
decoder block or sharding layout being checked. The few tests whose point is that a full-size model fits in HBM are
marked `integration_test` and `scheduled_only`, so they run only on scheduled CI.
"""

import os
import pickle
import shutil
import tempfile

from absl.testing import parameterized
import jax
from jax.experimental.serialize_executable import serialize
import pytest

from maxtext.configs import pyconfig
from maxtext.trainers.pre_train.train_compile import get_topology_mesh
from maxtext.trainers.pre_train.train_compile import main as train_compile_main
from maxtext.utils.globals import MAXTEXT_ASSETS_ROOT
from tests.utils.test_helpers import get_test_config_path

# The test_save_* tests check that each TPU generation compiles and serializes the train step, not how big the model
# is. base.yml's defaults (batch 12 x seq 2048 x vocab 32k, 16 heads of 128) would otherwise dominate their compile.
_TINY_MODEL = (
    "base_emb_dim=256",
    "base_mlp_dim=256",
    "base_num_decoder_layers=2",
    "base_num_query_heads=4",
    "base_num_kv_heads=4",
    "per_device_batch_size=1",
    "max_target_length=512",
    "vocab_size=1024",
)

# Width overrides for the mixtral-8x7b MoE tests: they check the dropping/dense/megablox/ragged-dot MoE paths, which do
# not depend on the 4096 x 14336 production width.
_SMALL_MIXTRAL = (
    "model_name=mixtral-8x7b",
    "override_model_config=True",
    "base_num_decoder_layers=2",
    "base_emb_dim=1024",
    "base_mlp_dim=2048",
    "base_moe_mlp_dim=2048",
)

# deepseek3-test keeps a 129k vocabulary and an 18432-wide dense MLP; neither matters to the MoE paths tested with it.
_SMALL_DEEPSEEK3_TEST = (
    "model_name=deepseek3-test",
    "override_model_config=True",
    "vocab_size=32768",
    "base_mlp_dim=2048",
)

_LLAMA2_TOKENIZER = (
    "tokenizer_type=tiktoken",
    f"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizers', 'tokenizer.llama2')}",
)

_GEMMA4_TOKENIZER = (
    "tokenizer_type=sentencepiece",
    f"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizers', 'tokenizer_gemma4.model')}",
)


@pytest.mark.tpu_backend
class TrainCompile(parameterized.TestCase):
  """Tests for the Ahead of Time Compilation functionality, train_compile.py"""

  # Default XLA flags for AOT compile tests: skip expensive HLO/LLO optimization passes (latency-hiding scheduler,
  # memory-space assignment, collective pipelining, bf16 propagation, LLO critical-path scheduling, XLU optimization)
  # while still running JAX tracing, explicit-sharding checks, Shardy/SPMD partitioning, layout assignment, fusion,
  # Mosaic/Pallas kernel lowering, buffer assignment, LLO lowering, register allocation, and codegen.
  #
  # Override with `full_opt=True` on `_aot()` (or an explicit `compile_xla_flags=...` override) whenever a test
  # depends on full compiler optimization, including:
  # - Full-size HBM-fit checks, where memory-space assignment or bf16 propagation determines whether the model fits.
  # - Tests asserting on optimized HLO schedules, collective pipelining, latency hiding, or post-optimization layouts.
  # - Tests directly exercising XLA optimization or scheduling flags.
  _FAST_COMPILE_XLA_FLAGS = (
      "--xla_optimization_level=EFFORT_O0"
      " --xla_jf_llo_level=0"
      " --xla_jf_critical_path_scheduler=false"
      " --xla_jf_xlu_optimizer=false"
  )

  def _aot(self, *overrides, save=False, full_opt=False):
    """AOT-compiles the train step for the base test config with `overrides` applied.

    Only `save=True` sets `compiled_trainstep_file`: serializing an executable and writing it out (tens to hundreds of
    MB) costs seconds per test, and nothing reads the file back, so only the tests about saving pay for it. Their file
    goes to a fresh temporary directory that is removed after the test, even if it fails.

    Unless `full_opt=True` or `overrides` sets `compile_xla_flags=`, passes `_FAST_COMPILE_XLA_FLAGS` to skip
    expensive HLO/LLO optimization passes that do not affect compilation smoke or sharding checks.
    """
    argv = ["", get_test_config_path(), *overrides]
    if not full_opt and not any(arg.startswith("compile_xla_flags=") for arg in overrides):
      argv.append(f"compile_xla_flags={self._FAST_COMPILE_XLA_FLAGS}")
    compiled_trainstep_file = None
    if save:
      tmp_dir = tempfile.mkdtemp(prefix="train_compile_test_")
      self.addCleanup(shutil.rmtree, tmp_dir, ignore_errors=True)
      compiled_trainstep_file = os.path.join(tmp_dir, f"{self._testMethodName}.pickle")
      argv.append(f"compiled_trainstep_file={compiled_trainstep_file}")
    train_compile_main(argv)
    if save:
      self.assertGreater(os.path.getsize(compiled_trainstep_file), 0)

  def test_save_compiled_v4(self):
    self._aot("compile_topology=v4-8", "compile_topology_num_slices=1", *_TINY_MODEL, save=True)

  def test_save_compiled_v5e(self):
    self._aot("compile_topology=v5e-16", "compile_topology_num_slices=1", *_TINY_MODEL, save=True)

  # TODO (b/366200617) : This tests fails in AOT, but config works fine on real hardware
  @pytest.mark.skip(reason="Issue w/ kernels_test. Error: The TPU is already in use by process...")
  def test_minimal_offloaded_v5e(self):
    self._aot(
        "compile_topology=v5e-256",
        "compile_topology_num_slices=1",
        "per_device_batch_size=1",
        "ici_fsdp_parallelism=16",
        "ici_tensor_parallelism=16",
        "max_target_length=2048",
        "fused_qkv=true",
        "fused_mlp=true",
        "remat_policy=minimal_offloaded",
        "use_iota_embed=true",
        "global_parameter_scale=128",
        full_opt=True,
    )

  def test_save_flash(self):
    self._aot(
        # The topology and depth do not matter to the custom remat policy with a device-resident context (splash
        # attention output) that this checks, so use 8 chips and 2 layers.
        "compile_topology=v5p-16",
        "compile_topology_num_slices=1",
        "base_num_decoder_layers=2",
        "per_device_batch_size=1",
        "remat_policy=custom",
        "context=device",  # Context is our name for the splash attention output for both TPU and GPU kernels.
        save=True,
    )

  def test_save_compiled_v5p_two_slices(self):
    self._aot("compile_topology=v5p-8", "compile_topology_num_slices=2", *_TINY_MODEL, save=True)

  def test_save_compiled_v6e(self):
    self._aot("compile_topology=v6e-16", "compile_topology_num_slices=1", *_TINY_MODEL, save=True)

  def test_save_compiled_tpu7x(self):
    self._aot(
        "compile_topology=tpu7x-16",
        "compile_topology_num_slices=1",
        "ici_fsdp_parallelism=16",
        *_TINY_MODEL,
        save=True,
    )

  def test_save_compiled_tpu7x_two_slices(self):
    self._aot(
        "compile_topology=tpu7x-8",
        "compile_topology_num_slices=2",
        "ici_fsdp_parallelism=4",
        "ici_tensor_parallelism=2",
        "dcn_data_parallelism=2",
        *_TINY_MODEL,
        save=True,
    )

  # The remat tests check that each remat policy's checkpoint names resolve and compile on a FSDP x TP16 mesh. Those
  # names do not depend on model size, so they use base.yml's 16-head width (one head per tensor shard) on a 128-chip
  # (8x16) FSDP 8 x TP 16 slice instead of global_parameter_scale=128 (8192 emb, 128 heads, 64 layers) on 256 chips.
  _REMAT_MESH = (
      "compile_topology_num_slices=1",
      "ici_fsdp_parallelism=8",
      "ici_tensor_parallelism=16",
      "base_num_decoder_layers=2",
      "fused_qkv=true",
      "fused_mlp=true",
      "use_iota_embed=true",
  )

  def test_remat_save_dot_except_mlpwi(self):
    self._aot(
        "compile_topology=v5e-128",
        *self._REMAT_MESH,
        "per_device_batch_size=0.125",
        "max_target_length=2048",
        "remat_policy=save_dot_except_mlpwi",
    )

  # save_dot_except_mlp and save_qkv_proj are the same `save_only_these_names` policy as save_dot_except_mlpwi, with
  # a subset of its names; test_remat_save_dot_except_mlpwi keeps that family on PRs.
  @pytest.mark.scheduled_only
  def test_remat_save_dot_except_mlp(self):
    self._aot(
        "compile_topology=v5e-128",
        *self._REMAT_MESH,
        "per_device_batch_size=0.25",
        "max_target_length=2048",
        "remat_policy=save_dot_except_mlp",
    )

  @pytest.mark.scheduled_only
  def test_remat_save_qkv_proj(self):
    self._aot(
        "compile_topology=v5e-128",
        *self._REMAT_MESH,
        "per_device_batch_size=0.375",
        "max_target_length=2048",
        "remat_policy=save_qkv_proj",
    )

  def test_remat_full(self):
    self._aot(
        "compile_topology=v6e-128",
        *self._REMAT_MESH,
        "per_device_batch_size=1",
        "max_target_length=1024",
        "remat_policy=full",
    )

  def test_custom_64x4_mesh(self):
    self._aot(
        # The hybrid_ring_64x4 custom mesh needs all 256 chips; the model can stay at base.yml's width, and 8192 tokens
        # still split four ways over the context axis.
        "compile_topology=v6e-256",
        "use_iota_embed=true",
        "compile_topology_num_slices=1",
        "ici_context_parallelism=4",
        "base_num_decoder_layers=2",
        "per_device_batch_size=0.25",
        "max_target_length=8192",
        "allow_split_physical_axes=true",
        "custom_mesh=hybrid_ring_64x4",
    )

  # TODO (b/376470419) : Enable when AOT test work with host offloading.
  @pytest.mark.skip(reason="Enable when AOT test work with host offloading.")
  @pytest.mark.gpu_only
  def test_llama3_1_70b_opt_offload(self):
    self._aot(
        "compile_topology=v6e-256",
        "compile_topology_num_slices=1",
        "model_name=llama3.1-70b",
        "per_device_batch_size=2",
        "optimizer_memory_host_offload=true",
        "gradient_clipping_threshold=0",
        "max_target_length=8192",
        full_opt=True,
    )

  # The custom-mesh code path stays on PRs through test_custom_64x4_mesh.
  @pytest.mark.scheduled_only
  def test_custom_32x8_mesh(self):
    self._aot(
        # The hybrid_ring_32x8 custom mesh needs all 256 chips; the model width does not matter to it.
        "compile_topology=v6e-256",
        "use_iota_embed=true",
        "compile_topology_num_slices=1",
        "ici_expert_parallelism=8",
        *_SMALL_MIXTRAL,
        "megablox=False",
        "sparse_matmul=False",
        "capacity_factor=1",
        "per_device_batch_size=1",
        "max_target_length=1024",
        "allow_split_physical_axes=true",
        "custom_mesh=hybrid_ring_32x8",
        "attention=flash",
    )

  def test_moe_dropping_bf16(self):
    self._aot(
        "compile_topology=v5p-16",
        "use_iota_embed=true",
        "compile_topology_num_slices=1",
        *_SMALL_MIXTRAL,
        "sparse_matmul=False",
        "capacity_factor=1",
        "per_device_batch_size=4",
        "max_target_length=1024",
        "attention=flash",
        "dtype=bfloat16",
    )

  @pytest.mark.skip(reason="b/400476456 Tests are currently flaking / failing due to JAX 0.5.1 upgrade")
  def test_moe_dropping_int8(self):
    self._aot(
        "compile_topology=v5p-128",
        "use_iota_embed=true",
        "compile_topology_num_slices=1",
        "model_name=mixtral-8x7b",
        "sparse_matmul=False",
        "capacity_factor=1",
        "per_device_batch_size=4",
        "max_target_length=128",
        "attention=flash",
        "dtype=bfloat16",
        "quantization=int8",
    )

  # TODO(b/388572320): Add int8 quantization test once this bug is fixed.
  def test_moe_megablox_bf16(self):
    self._aot(
        "compile_topology=v6e-16",
        "use_iota_embed=true",
        "compile_topology_num_slices=1",
        *_SMALL_MIXTRAL,
        "sparse_matmul=True",
        "megablox=True",
        "per_device_batch_size=4",
        "max_target_length=1024",
        "attention=flash",
        "dtype=bfloat16",
    )

  # test_moe_x_sorted_device compiles the same configuration plus remat_policy=custom and moe_x_sorted=device.
  @pytest.mark.scheduled_only
  def test_moe_megablox_ring_ep_random(self):
    self._aot(
        "compile_topology=v5p-16",
        "use_iota_embed=true",
        "compile_topology_num_slices=1",
        *_SMALL_DEEPSEEK3_TEST,
        "base_num_decoder_layers=2",
        "first_num_dense_layers=1",
        "ici_expert_parallelism=4",
        "sparse_matmul=True",
        "megablox=True",
        "per_device_batch_size=4",
        "max_target_length=128",
        "use_ring_of_experts=True",
        "use_random_routing=True",
        "attention=flash",
        "dtype=bfloat16",
    )

  def test_moe_x_sorted_device(self):
    """moe_x_sorted=device under the custom remat policy on the ring-of-experts sparse_matmul path."""
    self._aot(
        "compile_topology=v5p-16",
        "use_iota_embed=true",
        "compile_topology_num_slices=1",
        *_SMALL_DEEPSEEK3_TEST,
        "base_num_decoder_layers=2",
        "first_num_dense_layers=1",
        "ici_expert_parallelism=4",
        "sparse_matmul=True",
        "megablox=True",
        "per_device_batch_size=4",
        "max_target_length=128",
        "use_ring_of_experts=True",
        "use_random_routing=True",
        "attention=flash",
        "dtype=bfloat16",
        "remat_policy=custom",
        "moe_x_sorted=device",
    )

  def test_moe_x_sorted_rejects_emb_chunking(self):
    """moe_x_sorted != remat is rejected with num_moe_emb_chunks > 0, whose path does not tag the routed input."""
    with self.assertRaisesRegex(ValueError, "moe_x_sorted=device is not supported with num_moe_emb_chunks"):
      self._aot(
          "compile_topology=v5p-8",
          "use_iota_embed=true",
          "compile_topology_num_slices=1",
          "model_name=deepseek3-test",
          "ici_expert_parallelism=4",
          "sparse_matmul=True",
          "megablox=False",
          "use_tokamax_gmm=True",
          "use_gmm_v2=True",
          "num_moe_emb_chunks=7",
          "use_ring_of_experts=True",
          "per_device_batch_size=2",
          "max_target_length=1024",
          "attention=flash",
          "dtype=bfloat16",
          "weight_dtype=bfloat16",
          "scan_layers=True",
          "remat_policy=custom",
          "moe_x_sorted=device",
      )

  def test_moe_ragged_dot_bf16(self):
    self._aot(
        "compile_topology=v6e-16",
        "use_iota_embed=true",
        "compile_topology_num_slices=1",
        *_SMALL_MIXTRAL,
        "sparse_matmul=True",
        "megablox=False",
        "per_device_batch_size=4",
        "max_target_length=1024",
        "attention=flash",
        "dtype=bfloat16",
    )

  def test_moe_dense_bf16(self):
    self._aot(
        "compile_topology=v5p-16",
        "use_iota_embed=true",
        "compile_topology_num_slices=1",
        *_SMALL_MIXTRAL,
        "sparse_matmul=False",
        "capacity_factor=-1",
        "per_device_batch_size=4",
        "max_target_length=1024",
        "attention=flash",
        "dtype=bfloat16",
    )

  @pytest.mark.skip(reason="b/400476456 Tests are currently flaking / failing due to JAX 0.5.1 upgrade")
  def test_moe_dense_int8(self):
    self._aot(
        "compile_topology=v5p-128",
        "use_iota_embed=true",
        "compile_topology_num_slices=1",
        "model_name=mixtral-8x7b",
        "sparse_matmul=False",
        "capacity_factor=-1",
        "per_device_batch_size=4",
        "max_target_length=128",
        "attention=flash",
        "dtype=bfloat16",
        "quantization=int8",
    )

  def test_moe_pp_bf16(self):
    pytest.skip("Pipeline parallelism is not supported for the NNX decoder.")
    self._aot(
        "compile_topology=v5p-64",
        "use_iota_embed=true",
        "compile_topology_num_slices=2",
        "model_name=mixtral-8x7b",
        "sparse_matmul=False",
        "capacity_factor=1",
        "per_device_batch_size=4",
        "max_target_length=1024",
        "attention=flash",
        "dtype=bfloat16",
        "dcn_pipeline_parallelism=2",
        "num_layers_per_pipeline_stage=1",
    )

  def test_moe_deepseek_scanned_bf16(self):
    self._aot(
        "compile_topology=v5p-16",
        "use_iota_embed=true",
        "compile_topology_num_slices=1",
        *_SMALL_DEEPSEEK3_TEST,
        # Both the dense and the MoE scan still run more than one layer.
        "base_num_decoder_layers=5",
        "first_num_dense_layers=2",
        "sparse_matmul=True",
        "megablox=False",
        "per_device_batch_size=2",
        "max_target_length=1024",
        "attention=flash",
        "dtype=bfloat16",
        "weight_dtype=bfloat16",
        "scan_layers=True",
    )

  # The FSDP weight-prefetch pipeline is a hand-written custom-vjp scan unrolled by 2, so its cost grows with the 61
  # layers of deepseek3-tiny. Three dense and five MoE layers still give each scan several trips, an odd remainder for
  # the unroll, and avoid its single-trip special case.
  _PREFETCH_LAYERS = ("override_model_config=True", "base_num_decoder_layers=8", "first_num_dense_layers=3")

  def test_moe_deepseek_prefetch_fsdp_weights(self):
    """AOT test for FSDP weight prefetching (prefetch_fsdp_weights) on a scanned DeepSeek MoE model."""
    self._aot(
        "compile_topology=v5p-8",
        "use_iota_embed=true",
        "compile_topology_num_slices=1",
        "model_name=deepseek3-tiny",
        *self._PREFETCH_LAYERS,
        "sparse_matmul=True",
        "megablox=False",
        "per_device_batch_size=1",
        "max_target_length=128",
        "attention=flash",
        "dtype=bfloat16",
        "weight_dtype=bfloat16",
        "scan_layers=True",
        "prefetch_fsdp_weights=True",
    )

  # On PRs, prefetching is covered by test_moe_deepseek_prefetch_fsdp_weights, all-gather context parallelism by
  # test_custom_64x4_mesh, and explicit-sharding MLA + MoE by test_kimi_k2_explicit_sharding.
  @pytest.mark.scheduled_only
  def test_moe_deepseek_prefetch_fsdp_weights_context_parallelism(self):
    """AOT test for FSDP weight prefetching with context parallelism and moe_flat_fsdp_weights."""
    self._aot(
        "compile_topology=v5p-8",
        "use_iota_embed=False",
        "compile_topology_num_slices=1",
        "model_name=deepseek3-tiny",
        *self._PREFETCH_LAYERS,
        "ici_fsdp_parallelism=2",
        "ici_context_parallelism=2",
        "context_parallel_strategy=all_gather",
        "shard_mode=explicit",
        "moe_flat_fsdp_weights=True",
        "sparse_matmul=True",
        "megablox=True",
        "per_device_batch_size=1",
        "max_target_length=1024",
        "attention=flash",
        "dtype=bfloat16",
        "weight_dtype=bfloat16",
        "scan_layers=True",
        "prefetch_fsdp_weights=True",
    )

  def test_moe_emb_chunking_mlp_bias(self):
    """mlp_bias=True runs every chunking op the bias-free case does, plus the per-expert bias branches."""
    self._moe_emb_chunking(mlp_bias=True)

  # Covered on PRs by test_moe_emb_chunking_mlp_bias.
  @pytest.mark.scheduled_only
  def test_moe_emb_chunking_no_mlp_bias(self):
    self._moe_emb_chunking(mlp_bias=False)

  def _moe_emb_chunking(self, mlp_bias):
    """AOT test for MoE embedding chunking with or without per-expert MLP bias."""
    self._aot(
        "compile_topology=v5p-8",
        "use_iota_embed=true",
        "compile_topology_num_slices=1",
        *_SMALL_DEEPSEEK3_TEST,
        "base_num_decoder_layers=2",
        "first_num_dense_layers=1",
        "ici_expert_parallelism=4",
        "sparse_matmul=True",
        "megablox=False",
        "use_tokamax_gmm=True",
        "use_gmm_v2=True",
        # 4 divides deepseek3-test's emb_dim=512 into 128-wide TPU-lane-aligned chunks while still running 3 scan
        # iterations plus the final chunk.
        "num_moe_emb_chunks=4",
        "use_ring_of_experts=True",
        f"mlp_bias={mlp_bias}",
        "per_device_batch_size=2",
        "max_target_length=1024",
        "attention=flash",
        "dtype=bfloat16",
        "weight_dtype=bfloat16",
        "scan_layers=True",
    )

  @pytest.mark.integration_test
  def test_moe_deepseek_unscanned_bf16(self):
    self._aot(
        "compile_topology=v5p-64",
        "use_iota_embed=true",
        "compile_topology_num_slices=1",
        "model_name=deepseek3-test",
        "override_model_config=True",
        "base_num_decoder_layers=2",
        "first_num_dense_layers=1",
        "sparse_matmul=True",
        "megablox=False",
        "per_device_batch_size=1",
        "max_target_length=1024",
        "attention=flash",
        "dtype=bfloat16",
        "weight_dtype=bfloat16",
        "scan_layers=False",
    )

  def test_moe_deepseek_with_device_limit(self):
    self._aot(
        "compile_topology=v5p-16",
        "use_iota_embed=true",
        "compile_topology_num_slices=1",
        # Keeps all 256 experts, which the 8 routing groups below split into groups of 32.
        *_SMALL_DEEPSEEK3_TEST,
        "base_num_decoder_layers=5",
        "first_num_dense_layers=2",
        "sparse_matmul=True",
        "megablox=False",
        "per_device_batch_size=1",
        "max_target_length=1024",
        "attention=flash",
        "dtype=bfloat16",
        "weight_dtype=bfloat16",
        "n_routing_groups=8",
        "topk_routing_group=4",
    )

  def test_moe_deepseek_pipeline_subset(self):
    pytest.skip("Pipeline parallelism is not supported for the NNX decoder.")
    self._aot(
        "compile_topology=v5p-64",
        "compile_topology_num_slices=8",
        "use_iota_embed=true",
        "model_name=deepseek3-test",
        "megablox=True",
        "sparse_matmul=False",
        "capacity_factor=1",
        "per_device_batch_size=1",
        "max_target_length=1024",
        "pipeline_parallel_layers=56",
        "ici_expert_parallelism=16",
        "dcn_pipeline_parallelism=8",
    )

  def test_pipeline_subset(self):
    pytest.skip("Not supported for the NNX decoder.")
    self._aot(
        "compile_topology=v5p-128",
        "compile_topology_num_slices=8",
        "use_iota_embed=true",
        "per_device_batch_size=1",
        "max_target_length=1024",
        "pipeline_parallel_layers=56",
        "base_num_decoder_layers=61",  # Remainder of 5 will fail when sharded incorrectly.
        "ici_expert_parallelism=16",
        "dcn_pipeline_parallelism=8",
    )

  def test_moe_llama4_17b_16e(self):
    self._aot(
        # FSDP 2 x TP 4; the 40 query and 8 KV heads still split four ways.
        "compile_topology=v5p-16",
        "compile_topology_num_slices=1",
        "model_name=llama4-17b-16e",
        "override_model_config=True",
        # Two [RoPE x3, NoPE] cycles, so the scanned block runs more than one iteration.
        "base_num_decoder_layers=8",
        "base_emb_dim=1024",
        "base_mlp_dim=2048",
        "base_moe_mlp_dim=2048",
        "vocab_size=32768",
        "per_device_batch_size=1",
        "max_target_length=1024",
        "dtype=bfloat16",
        "weight_dtype=bfloat16",
        "scan_layers=True",
        "ici_fsdp_parallelism=2",
        "ici_tensor_parallelism=4",
    )

  # gpt-oss-20b's 201k vocabulary does not matter to the MoE and attention-sink paths these tests check. Four layers are
  # two sliding-window/full-attention cycles, so the scanned block runs more than one iteration.
  _SMALL_GPT_OSS = (
      "model_name=gpt-oss-20b",
      "override_model_config=True",
      "base_num_decoder_layers=4",
      "vocab_size=32768",
  )

  # test_moe_gpt_oss_20b_explicit_sharding_expert_parallelism compiles the same model on the same sparse_matmul path
  # under the stricter explicit sharding.
  @pytest.mark.scheduled_only
  def test_moe_gpt_oss_20b_sparse_matmul(self):
    self._aot(
        "compile_topology=v5p-16",
        "compile_topology_num_slices=1",
        *self._SMALL_GPT_OSS,
        "per_device_batch_size=1",
        "max_target_length=1024",
        "dtype=bfloat16",
        "weight_dtype=bfloat16",
        "scan_layers=True",
        "sparse_matmul=True",
        "megablox=True",
        "attention=flash",
    )

  def test_moe_gpt_oss_20b_dense_matmul(self):
    self._aot(
        "compile_topology=v5p-16",
        "compile_topology_num_slices=1",
        *self._SMALL_GPT_OSS,
        "per_device_batch_size=1",
        "max_target_length=1024",
        "dtype=bfloat16",
        "weight_dtype=bfloat16",
        "scan_layers=True",
        "sparse_matmul=False",
        "capacity_factor=-1",
        "attention=flash",
    )

  def test_moe_gpt_oss_20b_explicit_sharding_expert_parallelism(self):
    self._moe_gpt_oss_20b_explicit_sharding("ici_expert_parallelism=8")

  # Explicit sharding for gpt-oss stays on PRs via test_moe_gpt_oss_20b_explicit_sharding_expert_parallelism,
  # and TP explicit sharding via test_gpt3_6b_explicit_sharding.
  @pytest.mark.scheduled_only
  def test_moe_gpt_oss_20b_explicit_sharding_tensor_parallelism(self):
    self._moe_gpt_oss_20b_explicit_sharding("ici_tensor_parallelism=8")

  def _moe_gpt_oss_20b_explicit_sharding(self, parallelism):
    """AOT test for gpt-oss under explicit sharding, on the sparse_matmul path."""
    self._aot(
        "compile_topology=v5p-16",
        "compile_topology_num_slices=1",
        *self._SMALL_GPT_OSS,
        "per_device_batch_size=1",
        "max_target_length=1024",
        "dtype=bfloat16",
        "weight_dtype=bfloat16",
        "scan_layers=True",
        "sparse_matmul=True",
        "megablox=True",
        "attention=flash",
        "shard_mode=explicit",
        "ici_fsdp_parallelism=1",
        parallelism,
    )

  # gpt3-6b defaults to fused_qkv=true, so test_gpt3_6b_explicit_sharding_fused_qkv compiles the same model under
  # explicit sharding.
  @pytest.mark.scheduled_only
  def test_gpt3_6b(self):
    self._aot(
        "compile_topology=v5p-8",
        "compile_topology_num_slices=1",
        "model_name=gpt3-6b",
        "override_model_config=true",
        "base_num_decoder_layers=2",
        "per_device_batch_size=1",
    )

  @parameterized.named_parameters(("fused_qkv", True), ("unfused_qkv", False))
  def test_gpt3_6b_explicit_sharding(self, fused_qkv):
    """AOT test for gpt3 under explicit sharding, with tensor parallelism on the heads axis."""
    self._aot(
        "compile_topology=v5p-8",
        "compile_topology_num_slices=1",
        "model_name=gpt3-6b",
        "per_device_batch_size=1",
        "shard_mode=explicit",
        "override_model_config=true",
        "base_num_decoder_layers=2",
        f"fused_qkv={fused_qkv}",
        "ici_fsdp_parallelism=1",
        "ici_tensor_parallelism=4",
    )

  def test_qwen3_qk_norm(self):
    """AOT test for non-llama qk norm models"""
    self._aot(
        "compile_topology=v5p-8",
        "compile_topology_num_slices=1",
        "model_name=qwen3-0.6b",
        "per_device_batch_size=1",
    )

  # Shrinks the qwen3-next / qwen3.5 GDN, attention and MoE widths: the layout, sharding and kernel-lowering paths
  # only need non-trivial head/expert counts divisible by the mesh axes (up to FSDP 2 x expert 8), not production
  # widths. `head_dim=128` with `partial_rotary_factor=0.5` keeps rotary_dim=64 so qwen3.5's mrope_section [11, 11, 10]
  # still matches rotary_dim / 2.
  _SMALL_QWEN_MOE = (
      "base_emb_dim=1024",
      "base_num_query_heads=8",
      "base_num_kv_heads=2",
      "head_dim=128",
      "partial_rotary_factor=0.5",
      "gdn_num_key_heads=4",
      "gdn_num_value_heads=8",
      "gdn_key_head_dim=64",
      "gdn_value_head_dim=64",
      "base_mlp_dim=256",
      "base_moe_mlp_dim=256",
      "num_experts=16",
      "num_experts_per_tok=2",
      "max_target_length=512",
      "vocab_size=32768",
  )

  def test_qwen3_next(self):
    """AOT test for qwen3-next and GatedDeltaNet implementation"""
    self._aot(
        "compile_topology=v5p-16",
        "compile_topology_num_slices=1",
        "model_name=qwen3-next-80b-a3b",
        "override_model_config=True",
        # Two [GDN x3, full-attention] cycles, so the scanned block runs more than one iteration.
        "base_num_decoder_layers=8",
        *self._SMALL_QWEN_MOE,
        "per_device_batch_size=1",
    )

  def test_deepseek32(self):
    # test deepseek3.2 with sparse attention
    self._aot(
        "compile_topology=v5p-16",
        "use_iota_embed=true",
        "compile_topology_num_slices=1",
        "model_name=deepseek3.2-671b",
        # Shrinks the 671b widths; keeps MLA, the 64-head indexer, and two dense and three MoE layers so that both scans
        # run more than one iteration.
        "override_model_config=True",
        "base_num_decoder_layers=5",
        "first_num_dense_layers=2",
        "base_emb_dim=1024",
        "base_mlp_dim=2048",
        "base_moe_mlp_dim=512",
        "num_experts=16",
        "vocab_size=32768",
        # megablox
        "sparse_matmul=True",
        "megablox=True",
        "per_device_batch_size=1",
        "max_target_length=1024",
        "attention=flash",
        "use_tokamax_splash=True",
        "dtype=bfloat16",
        "weight_dtype=bfloat16",
    )

  # The flash case compiles the same decoder plus the flash-only compressed-attention branches (HCA KV padding, static
  # and CSA indexer masks, splash kernels), whose train step no other PR test compiles. The dot-product-only dense
  # masks stay on PRs through deepseek_v4_remat_test (deepseek4-tiny train step with grads) and
  # deepseek_v4_padded_prefill_decode_test (deepseek4-tiny dot_product forward/decode parity).
  @pytest.mark.scheduled_only
  @pytest.mark.cpu_only
  def test_deepseek4_scanned_dot_product(self):
    self._deepseek4(scan_layers="true", attention="dot_product")

  @pytest.mark.cpu_only
  def test_deepseek4_scanned_flash(self):
    self._deepseek4(scan_layers="true", attention="flash")

  def _deepseek4(self, scan_layers, attention):
    """AOT compile test on deepseek4-tiny (same block, compressed attention, indexer, and mHC as 284b)."""
    # test_deepseek4_full_size keeps the full-width compile.
    self._aot(
        "compile_topology=v5p-8",
        "use_iota_embed=true",
        "compile_topology_num_slices=1",
        "model_name=deepseek4-tiny",
        "per_device_batch_size=1",
        "max_target_length=1024",
        f"scan_layers={scan_layers}",
        f"attention={attention}",
        "use_tokamax_splash=True",
        "sa_block_q=128",
        "sa_block_kv=128",
        "sa_block_kv_compute=128",
        "sa_block_q_dkv=128",
        "sa_block_kv_dkv=128",
        "sa_block_kv_dkv_compute=128",
        "sa_block_q_dq=128",
        "sa_block_kv_dq=128",
        "dtype=bfloat16",
        "weight_dtype=bfloat16",
        "routed_bias=False",
        "override_model_config=True",
        # One hash-routed prefix layer (compress_ratio=0) plus two scanned [128, 4] blocks cover all three compress
        # ratios (0, 128, 4), both hash and standard MoE routing, and a multi-trip scan.
        "first_num_hash_layers=1",
        "base_num_decoder_layers=5",
        # The splash kernels tile the head dim in lanes of 128.
        "head_dim=128",
        "vocab_size=32768",
    )

  # Full-size HBM-fit check: cpu-integration also runs on every PR, so keep it to scheduled CI.
  @pytest.mark.scheduled_only
  @pytest.mark.integration_test
  @pytest.mark.cpu_only
  def test_deepseek4_full_size(self):
    """Compiles five deepseek4-284b layers at full width, which checks that they fit in v5p-256 HBM."""
    self._aot(
        "compile_topology=v5p-256",
        "use_iota_embed=true",
        "compile_topology_num_slices=1",
        "model_name=deepseek4-284b",
        "per_device_batch_size=1",
        "max_target_length=1024",
        "scan_layers=true",
        "attention=flash",
        "use_tokamax_splash=True",
        "sa_block_q=128",
        "sa_block_kv=128",
        "sa_block_kv_compute=128",
        "sa_block_q_dkv=128",
        "sa_block_kv_dkv=128",
        "sa_block_kv_dkv_compute=128",
        "sa_block_q_dq=128",
        "sa_block_kv_dq=128",
        "dtype=bfloat16",
        "weight_dtype=bfloat16",
        "routed_bias=False",
        "override_model_config=True",
        "base_num_decoder_layers=5",
        full_opt=True,
    )

  # deepseek-custom keeps DeepSeek-V3's 129k vocabulary; 32k is enough for the indexer, mHC and engram paths. 512 tokens
  # stay above its indexer_topk of 256.
  _SMALL_DEEPSEEK_CUSTOM = (
      "model_name=deepseek-custom",
      "override_model_config=True",
      "vocab_size=32768",
      "max_target_length=512",
  )

  def test_indexer_dense_warmup(self):
    # test deepseek3.2 with sparse attention
    self._aot(
        "compile_topology=v5p-8",
        "compile_topology_num_slices=1",
        *self._SMALL_DEEPSEEK_CUSTOM,
        "per_device_batch_size=1",
        "scan_layers=True",
        "attention=flash",
        "use_tokamax_splash=True",
        # override
        "base_num_decoder_layers=2",
        "engram_layers=[]",
        # dense warmup
        "indexer_sparse_training=False",
        "indexer_loss_scaling_factor=0.1",
        "trainable_parameters_mask=['.*indexer.*']",
    )

  # test_glm5_compile_glm5_2_scanned keeps the indexer's sparse-training loss (shared MLA code) on PRs, and
  # test_mhc_integration keeps deepseek-custom with tokamax splash.
  @pytest.mark.scheduled_only
  def test_indexer_sparse_training(self):
    # test deepseek3.2 with sparse attention
    self._aot(
        "compile_topology=v5p-8",
        "compile_topology_num_slices=1",
        *self._SMALL_DEEPSEEK_CUSTOM,
        "per_device_batch_size=1",
        "scan_layers=True",
        "attention=flash",
        "use_tokamax_splash=True",
        # override
        "base_num_decoder_layers=2",
        "engram_layers=[]",
        # sparse training
        "indexer_sparse_training=True",
        "indexer_loss_scaling_factor=0.1",
    )

  def test_olmo3_7b(self):
    """AOT test for Olmo3 7B implementation"""
    self._aot(
        "compile_topology=v5p-8",
        "compile_topology_num_slices=1",
        "model_name=olmo3-7b",
        # Two [sliding-window x3, global] cycles, so the scanned block runs more than one iteration.
        "override_model_config=True",
        "base_num_decoder_layers=8",
        "base_emb_dim=1024",
        "vocab_size=32768",
        "per_device_batch_size=1",
        "scan_layers=True",
        "max_target_length=1024",
    )

  def test_mhc_integration(self):
    """AOT test for Manifold-constrained Hyper Connection implementation"""
    self._aot(
        "compile_topology=v5p-8",
        "compile_topology_num_slices=1",
        *self._SMALL_DEEPSEEK_CUSTOM,
        "per_device_batch_size=1",
        "scan_layers=True",
        "attention=flash",
        "use_tokamax_splash=True",
        # override
        "base_num_decoder_layers=2",
        "mhc_expansion_rate=4",
        "engram_layers=[]",
    )

  # Engram stays on PRs through engram_test.py and nnx_decoders_test.test_deepseek_unscanned_engram_layers.
  @pytest.mark.scheduled_only
  def test_engram_integration(self):
    """AOT test for Engram implementation"""
    self._aot(
        "compile_topology=v5p-8",
        "compile_topology_num_slices=1",
        *self._SMALL_DEEPSEEK_CUSTOM,
        "per_device_batch_size=1",
        "scan_layers=True",
        "attention=flash",
        "use_tokamax_splash=True",
        # The engram hash tables are built from a HuggingFace tokenizer; use a vendored one instead of downloading
        # DeepSeek-V3.2's from the Hub.
        "tokenizer_type=huggingface",
        f"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizers', 'qwen3-tokenizer')}",
        # Required by the engram config validation; unused for a local tokenizer.
        "hf_access_token=fake",
        "base_num_decoder_layers=3",
        "engram_layers=[1]",
        # deepseek-custom's 226k-row bases only grow the embedding tables, not the code path.
        "engram_vocab_bases=[4096,4096]",
    )

  def test_circular_pipeline_ag_per_repeat_ep_ds(self):
    pytest.skip("Pipeline parallelism is not supported for the NNX decoder.")
    self._aot(
        "compile_topology=v5p-8",
        "compile_topology_num_slices=1",
        "per_device_batch_size=2",
        "ici_pipeline_parallelism=2",
        "ici_expert_parallelism=2",
        "pipeline_parallel_layers=4",
        "num_pipeline_microbatches=4",
        "model_name=deepseek3-test",
        "override_model_config=true",
        "base_num_decoder_layers=7",
        "use_ring_of_experts=true",
        "use_random_routing=true",
        "max_target_length=128",
        "pipeline_fsdp_ag_per_repeat=true",
    )

  @parameterized.named_parameters(
      {"testcase_name": "dot_product", "attention": "dot_product"},
      {"testcase_name": "tokamax_splash", "attention": "flash"},
  )
  def test_qk_clip(self, attention):
    """AOT test for AdamW optimizer with QK clip for DeepSeek3 Tiny model"""
    self._aot(
        "compile_topology=v5p-8",
        "compile_topology_num_slices=1",
        "model_name=deepseek3-tiny",
        "override_model_config=True",
        "base_num_decoder_layers=2",
        "first_num_dense_layers=1",
        "vocab_size=32768",
        "scan_layers=True",
        "sparse_matmul=True",
        "megablox=True",
        "use_tokamax_gmm=False",
        "max_target_length=128",
        "per_device_batch_size=1",
        "dtype=bfloat16",
        "weight_dtype=float32",
        # attention
        f"attention={attention}",
        "use_tokamax_splash=True",
        # qk clip
        "use_qk_clip=true",
        "qk_clip_threshold=100",
    )

  @parameterized.named_parameters(
      {"testcase_name": "consistent_rms_scaling", "muon_consistent_rms": 0.2},
      {"testcase_name": "width_scaling", "muon_consistent_rms": None},
  )
  def test_muon(self, muon_consistent_rms):
    """AOT test for Muon optimizer for DeepSeek3 Tiny model"""
    self._aot(
        "compile_topology=v5p-8",
        "compile_topology_num_slices=1",
        "model_name=deepseek3-tiny",
        "override_model_config=True",
        "base_num_decoder_layers=2",
        "first_num_dense_layers=1",
        "vocab_size=32768",
        "scan_layers=True",
        "sparse_matmul=True",
        "megablox=True",
        "use_tokamax_gmm=False",
        "max_target_length=128",
        "per_device_batch_size=1",
        "dtype=bfloat16",
        "weight_dtype=float32",
        # tokamax splash attention
        "attention=flash",
        "use_tokamax_splash=True",
        # muon optimizer
        "opt_type=muon",
        "muon_beta=0.95",
        "muon_weight_decay=0.1",
        f"muon_consistent_rms={muon_consistent_rms}",
    )

  # qwen3.5 subclasses the qwen3-next block and layer; test_qwen3_5_explicit_sharding compiles this model with the
  # same widths under explicit sharding, and test_qwen3_next keeps the qwen3-next layer on PRs.
  @pytest.mark.scheduled_only
  def test_qwen3_5(self):
    """AOT test for qwen3-5"""
    self._aot(
        "compile_topology=v5p-16",
        "compile_topology_num_slices=1",
        "model_name=qwen3.5-397b-a17b",
        "override_model_config=True",
        # Two [GDN x3, full-attention] cycles, so the scanned block runs more than one iteration.
        "base_num_decoder_layers=8",
        *self._SMALL_QWEN_MOE,
        "per_device_batch_size=1.0",
        "sparse_matmul=True",
        "megablox=True",
        "attention=flash",
        "use_tokamax_splash=True",
    )

  # Full-size HBM-fit check: cpu-integration also runs on every PR, so keep it to scheduled CI.
  @pytest.mark.scheduled_only
  @pytest.mark.integration_test
  def test_qwen3_8(self):
    """AOT test for qwen3-8 at full width, truncated to two 4-layer [GDN x3, full-attention] cycles.

    Needs expert parallelism: under pure FSDP, gathering one layer's 512 routed experts in fp32 exceeds v5p HBM.
    """
    self._aot(
        "compile_topology=v5p-512",
        "compile_topology_num_slices=1",
        "model_name=qwen3.8-2.4t-a95b",
        "override_model_config=True",
        "base_num_decoder_layers=8",
        "ici_expert_parallelism=8",
        "per_device_batch_size=1.0",
        "max_target_length=1024",
        "sparse_matmul=True",
        "megablox=True",
        "attention=flash",
        "use_tokamax_splash=True",
        full_opt=True,
    )

  # Qwen3_5ScannableBlock subclasses Qwen3NextScannableBlock, so test_qwen3_5_explicit_sharding runs the same nested
  # scan under explicit sharding on PRs; test_qwen3_next keeps the qwen3-next layer itself.
  @pytest.mark.scheduled_only
  def test_qwen3_next_explicit_sharding(self):
    """AOT test for qwen3-next under explicit sharding, at FSDP 2 x expert 8.

    `Qwen3NextScannableBlock` nests two `jax.lax.scan`s, whose carry layout has
    to stay
    invariant across iterations; that only holds if the decoder layer returns
    the
    layout it was handed. Eight layers keep two iterations of the outer scan.
    """
    self._aot(
        "compile_topology=v5p-32",
        "compile_topology_num_slices=1",
        "model_name=qwen3-next-80b-a3b",
        "override_model_config=True",
        "base_num_decoder_layers=8",
        *self._SMALL_QWEN_MOE,
        "per_device_batch_size=1.0",
        "ici_fsdp_parallelism=2",
        "ici_expert_parallelism=8",
        "sparse_matmul=True",
        "megablox=True",
        "attention=flash",
        "use_tokamax_splash=True",
        "shard_mode=explicit",
    )

  # ZeRO-1 with gradient accumulation stays on PRs through test_qwen2_explicit_sharding_zero1 and
  # test_zero1_optimizer_sharding; the GDN decoder's explicit layouts through test_qwen3_5_explicit_sharding.
  @pytest.mark.scheduled_only
  def test_qwen3_next_explicit_sharding_zero1(self):
    """AOT test for qwen3-next under explicit sharding with ZeRO-1 and gradient accumulation."""
    self._aot(
        "compile_topology=v5p-16",
        "compile_topology_num_slices=1",
        "model_name=qwen3-next-80b-a3b",
        "override_model_config=True",
        "base_num_decoder_layers=4",
        *self._SMALL_QWEN_MOE,
        "per_device_batch_size=1.0",
        "sparse_matmul=True",
        "megablox=True",
        "attention=flash",
        "use_tokamax_splash=True",
        "shard_mode=explicit",
        "ici_data_parallelism=-1",
        "ici_fsdp_parallelism=1",
        "gradient_accumulation_steps=2",
        "shard_optimizer_over_data=True",
    )

  def test_qwen3_5_explicit_sharding(self):
    """AOT test for qwen3-5 under explicit sharding, at FSDP 2 x expert 8.

    Explicit sharding type-checks every operation's layout instead of letting
    GSPMD
    infer one, so a missing `out_sharding` fails the trace here rather than
    silently
    costing a collective at a scale a real test cannot reach.
    """
    self._aot(
        "compile_topology=v5p-32",
        "compile_topology_num_slices=1",
        "model_name=qwen3.5-397b-a17b",
        "override_model_config=True",
        "base_num_decoder_layers=8",
        *self._SMALL_QWEN_MOE,
        "per_device_batch_size=1.0",
        "ici_fsdp_parallelism=2",
        "ici_expert_parallelism=8",
        "sparse_matmul=True",
        "megablox=True",
        "attention=flash",
        "use_tokamax_splash=True",
        "shard_mode=explicit",
        # Qwen3.5 defaults to a HuggingFace tokenizer that is not vendored.
        *_LLAMA2_TOKENIZER,
    )

  # See test_qwen3_next_explicit_sharding_zero1 for the PR-time coverage.
  @pytest.mark.scheduled_only
  def test_qwen3_5_explicit_sharding_zero1(self):
    """AOT test for qwen3-5 under explicit sharding with ZeRO-1 and gradient accumulation.

    ZeRO-1 shards the optimizer moments over "data", so the parameters are
    all-gathered
    in bf16 once before the accumulation scan rather than once per microbatch;
    that
    reshard only type-checks if the decoder pins its activation layouts. Four
    layers is
    one full `inhomogeneous_layer_cycle_interval`, which covers both attention
    variants
    while staying small enough to hold data-parallel replicas of the parameters.
    """
    self._aot(
        "compile_topology=v5p-16",
        "compile_topology_num_slices=1",
        "model_name=qwen3.5-35b-a3b",
        "override_model_config=True",
        "base_num_decoder_layers=4",
        *self._SMALL_QWEN_MOE,
        "per_device_batch_size=1.0",
        "sparse_matmul=True",
        "megablox=True",
        "attention=flash",
        "use_tokamax_splash=True",
        "shard_mode=explicit",
        "ici_data_parallelism=-1",
        "ici_fsdp_parallelism=1",
        "gradient_accumulation_steps=2",
        "shard_optimizer_over_data=True",
        *_LLAMA2_TOKENIZER,
    )

  def test_serialization_and_deserialization_formats(self):
    """Tests that our custom binary save/load functions work securely and legacy fallback triggers warning."""

    def load_serialized_compiled_test(save_name):
      with open(save_name, "rb") as f:
        return f.read()

    @jax.jit
    def add_one(x):
      return x + 1

    # Compile simply on CPU
    compiled = add_one.lower(1).compile()
    serialized, _, _ = serialize(compiled)

    # 1. Save and load JAX compiled step using secure raw binary format
    with tempfile.NamedTemporaryFile() as f_secure:
      with open(f_secure.name, "wb") as f:
        f.write(serialized)

      loaded_compiled = load_serialized_compiled_test(f_secure.name)

      # Ensure it loaded the correct JAX serialization bytes
      assert loaded_compiled == serialized

    # 2. Save and load JAX compiled step using legacy pickle format
    with tempfile.NamedTemporaryFile() as f_legacy:
      with open(f_legacy.name, "wb") as f:
        pickle.dump(serialized, f)

      loaded_legacy = load_serialized_compiled_test(f_legacy.name)

      # Ensure it loaded raw pickled bytes (starting with pickle protocol marker)
      # and did NOT unpickle them into JAX serialization bytes.
      assert loaded_legacy.startswith(b"\x80")
      assert loaded_legacy != serialized

  def test_zero1_optimizer_sharding(self):
    """AOT test for Zero-1 optimizer sharding (shard_optimizer_over_data)"""
    self._aot(
        "compile_topology=v5p-8",
        "compile_topology_num_slices=1",
        "base_emb_dim=256",
        "base_mlp_dim=256",
        "base_num_decoder_layers=2",
        "ici_data_parallelism=-1",
        "ici_fsdp_parallelism=1",
        "shard_optimizer_over_data=true",
        "shard_mode=explicit",
    )

  def test_qwen2_explicit_sharding_zero1(self):
    """AOT test for Qwen2 under explicit sharding with ZeRO-1 and gradient accumulation.

    Explicit sharding type-checks every operation's layout rather than letting GSPMD infer
    one, so a missing `out_sharding` in the Qwen2 decoder fails the trace here rather than
    silently costing a collective at a scale we cannot reach in a test.
    """
    self._aot(
        "compile_topology=v5p-16",
        "compile_topology_num_slices=1",
        "model_name=qwen2.5-7b",
        "override_model_config=True",
        "base_num_decoder_layers=2",
        "vocab_size=32768",
        "per_device_batch_size=1",
        "max_target_length=1024",
        "attention=flash",
        "shard_mode=explicit",
        # ZeRO-1 needs a "data" axis to shard the moments over, and MaxTextConfig
        # rejects combining it with FSDP.
        "ici_data_parallelism=-1",
        "ici_fsdp_parallelism=1",
        "gradient_accumulation_steps=4",
        "shard_optimizer_over_data=True",
        # The Qwen2.5 configs default to a HuggingFace tokenizer that is not vendored.
        *_LLAMA2_TOKENIZER,
    )

  # Shrinks the gemma4 widths and its 262k vocabulary for the explicit-sharding tests. 2048 tokens stay above the
  # 1024-token sliding window, so the local layers still mask.
  _SMALL_GEMMA4 = (
      "override_model_config=True",
      "base_emb_dim=1024",
      "vocab_size=32768",
      "per_device_batch_size=1",
      "max_target_length=2048",
  )

  def test_gemma4_dense_explicit_sharding(self):
    """AOT test for gemma4-31b's decoder block under explicit sharding, at FSDP 2 x TP 4.

    The dense Gemma 4 block scans five local-sliding layers and one global layer with a
    wider head, so this checks that both attention variants and the surrounding norms
    agree on a layout.
    """
    self._aot(
        "compile_topology=v5p-16",
        "compile_topology_num_slices=1",
        "model_name=gemma4-31b",
        *self._SMALL_GEMMA4,
        "base_num_decoder_layers=6",
        "base_mlp_dim=2048",
        "ici_fsdp_parallelism=2",
        "ici_tensor_parallelism=4",
        "attention=flash",
        "shard_mode=explicit",
        *_GEMMA4_TOKENIZER,
    )

  # The gemma4 MoE block stays on PRs through router_replay_test's gemma4-26b loss test, and explicit-sharding MoE
  # through the kimi / llama4 / qwen3.5 / gpt-oss explicit tests.
  @pytest.mark.scheduled_only
  def test_gemma4_moe_explicit_sharding(self):
    """AOT test for gemma4-26b's decoder block under explicit sharding.

    Covers the Gemma 4 MoE block -- the scaled router, the unscaled gate norm and the
    shared expert -- at FSDP 2 x expert 8.
    """
    self._aot(
        "compile_topology=v5p-32",
        "compile_topology_num_slices=1",
        "model_name=gemma4-26b",
        *self._SMALL_GEMMA4,
        "base_num_decoder_layers=6",
        "base_mlp_dim=1024",
        "base_moe_mlp_dim=512",
        "num_experts=32",
        "ici_fsdp_parallelism=2",
        "ici_expert_parallelism=8",
        # RoutedMoE.dense_matmul is not onboarded to explicit sharding yet.
        "sparse_matmul=True",
        "megablox=True",
        "attention=flash",
        "shard_mode=explicit",
        *_GEMMA4_TOKENIZER,
    )

  # The E2B/E4B decoder stays on PRs through gemma4_small_test, gemma4_small_layers_test and nnx_decoders_test's
  # gemma4-small sharding checks; the shared gemma4 layers through test_gemma4_dense_explicit_sharding.
  @pytest.mark.scheduled_only
  def test_gemma4_small_explicit_sharding(self):
    """AOT test for gemma4-e4b under explicit sharding.

    The E2B/E4B block adds per-layer embeddings and KV sharing on top of gemma4, and
    builds its stack in a Python loop rather than a scan.
    """
    self._aot(
        "compile_topology=v5p-8",
        "compile_topology_num_slices=1",
        "model_name=gemma4-e4b",
        "override_model_config=True",
        # Twelve layers with the last six KV-shared, so the shared layers span both a local and a global layer.
        "base_num_decoder_layers=12",
        "num_kv_shared_layers=6",
        "base_emb_dim=512",
        "base_mlp_dim=1024",
        "vocab_size=32768",
        "vocab_size_per_layer_input=32768",
        "per_device_batch_size=1",
        # Stays above the 512-token sliding window.
        "max_target_length=1024",
        # E4B has two KV heads, which is as far as the tensor axis can split them.
        "ici_fsdp_parallelism=2",
        "ici_tensor_parallelism=2",
        "scan_layers=False",
        "attention=flash",
        "shard_mode=explicit",
        *_GEMMA4_TOKENIZER,
    )

  def test_kimi_k2_explicit_sharding(self):
    """AOT test for Kimi-K2's decoder block under explicit sharding.

    Kimi-K2 runs on the deepseek decoder block, so this is the explicit-sharding
    check for MLA attention and the shared-expert, sigmoid-routed MoE at FSDP 2 x expert 8.
    """
    self._aot(
        "compile_topology=v5p-32",
        "compile_topology_num_slices=1",
        "model_name=kimi-k2-1t",
        "override_model_config=True",
        "base_num_decoder_layers=2",
        "first_num_dense_layers=1",
        "base_emb_dim=1024",
        "base_mlp_dim=2048",
        "base_moe_mlp_dim=512",
        "num_experts=64",
        "vocab_size=32768",
        "per_device_batch_size=1",
        "max_target_length=1024",
        "ici_fsdp_parallelism=2",
        "ici_expert_parallelism=8",
        "sparse_matmul=True",
        "megablox=True",
        "attention=flash",
        "shard_mode=explicit",
        # Kimi-K2 defaults to a HuggingFace tokenizer that is not vendored.
        *_LLAMA2_TOKENIZER,
    )

  # On PRs, ZeRO-1 with gradient accumulation is covered by test_qwen2_explicit_sharding_zero1, and the llama4 decoder
  # (dense and MoE layers) under explicit sharding by test_llama4_17b_128e_explicit_sharding.
  @pytest.mark.scheduled_only
  def test_llama4_explicit_sharding_zero1(self):
    """AOT test for Llama4 under explicit sharding with ZeRO-1 and gradient accumulation.

    Explicit sharding type-checks every operation's layout rather than letting GSPMD infer
    one, so a missing `out_sharding` in the Llama4 decoder fails the trace here rather than
    silently costing a collective at a scale we cannot reach in a test.
    """
    self._aot(
        "compile_topology=v5p-16",
        "compile_topology_num_slices=1",
        "model_name=llama4-17b-16e",
        "override_model_config=True",
        # ZeRO-1 replicates the parameters over the data axis. Keep four experts and eight layers, which
        # with a MoE step of 2 still covers both the dense and the MoE layer variant.
        "base_num_decoder_layers=8",
        "interleave_moe_layer_step=2",
        "num_experts=4",
        "base_emb_dim=1024",
        "base_mlp_dim=2048",
        "base_moe_mlp_dim=2048",
        "vocab_size=32768",
        "per_device_batch_size=1",
        "max_target_length=1024",
        "attention=flash",
        "remat_policy=full",
        "shard_mode=explicit",
        # ZeRO-1 needs a "data" axis to shard the moments over, and MaxTextConfig
        # rejects combining it with FSDP.
        "ici_data_parallelism=-1",
        "ici_fsdp_parallelism=1",
        "gradient_accumulation_steps=4",
        "shard_optimizer_over_data=True",
        # The Llama4 configs default to a HuggingFace tokenizer that is not vendored.
        *_LLAMA2_TOKENIZER,
    )

  def test_llama4_17b_128e_explicit_sharding(self):
    """AOT test for Llama4 Maverick's decoder block under explicit sharding.

    The 128-expert config interleaves dense and MoE layers, so this covers both layer
    variants with the expert axis sharded, at FSDP 2 x expert 4.
    """
    self._aot(
        "compile_topology=v5p-16",
        "compile_topology_num_slices=1",
        "model_name=llama4-17b-128e",
        # Needed so the layer count, widths and the tokenizer below can override the model config.
        "override_model_config=True",
        "base_num_decoder_layers=4",
        "base_emb_dim=1024",
        "base_mlp_dim=2048",
        "base_moe_mlp_dim=2048",
        "num_experts=16",
        "vocab_size=32768",
        "per_device_batch_size=1",
        "max_target_length=1024",
        "ici_fsdp_parallelism=2",
        "ici_expert_parallelism=4",
        "sparse_matmul=True",
        "megablox=True",
        "attention=flash",
        "shard_mode=explicit",
        # The Llama4 configs default to a HuggingFace tokenizer that is not vendored.
        *_LLAMA2_TOKENIZER,
    )

  def test_vocab_tiling_bf16_nnx(self):
    """AOT compile vocab tiling on the NNX path (vocab_tiling_nnx_loss + custom_vjp) when weight_dtype=bfloat16."""
    self._aot(
        "compile_topology=v5p-8",
        "compile_topology_num_slices=1",
        "base_emb_dim=256",
        "base_mlp_dim=256",
        "base_num_decoder_layers=2",
        "per_device_batch_size=2",
        "max_target_length=1024",
        "num_vocab_tiling=4",
        "weight_dtype=bfloat16",
    )

  @parameterized.named_parameters(
      {"testcase_name": "scanned", "scan_layers": "true"},
  )
  @pytest.mark.cpu_only
  def test_envy(self, scan_layers):
    # test envy compile.
    self._aot(
        "compile_topology=v5p-16",
        "use_iota_embed=true",
        "compile_topology_num_slices=1",
        "model_name=envy-switch-base",
        "per_device_batch_size=1",
        "max_target_length=1024",
        f"scan_layers={scan_layers}",
        "attention=dot_product",
        "dtype=bfloat16",
        "weight_dtype=bfloat16",
        "override_model_config=True",
        # Two dense/MoE cycles (interleave_moe_layer_step=2), so the scanned block runs more than one iteration.
        "base_num_decoder_layers=4",
        "num_experts=16",
    )

  def test_get_topology_mesh_updates_shardy_partitioner(self):
    """Verifies get_topology_mesh updates jax_use_shardy_partitioner from config.shardy."""
    prev_shardy = getattr(jax.config, "jax_use_shardy_partitioner", True)
    self.addCleanup(jax.config.update, "jax_use_shardy_partitioner", prev_shardy)
    jax.config.update("jax_use_shardy_partitioner", False)
    config = pyconfig.initialize(
        (
            "",
            get_test_config_path(),
            "compile_topology=v6e-4",
            "compile_topology_num_slices=1",
            "shardy=True",
        )
    )
    get_topology_mesh(config)
    self.assertTrue(jax.config.jax_use_shardy_partitioner)

  # Shared GLM-5 settings. The shrunk tests keep MLA, the indexer with its loss and IndexShare, at FSDP 4 x expert 8.
  _GLM5_COMMON = (
      "compile_topology_num_slices=1",
      "use_iota_embed=true",
      "override_model_config=true",
      "per_device_batch_size=1",
      "sparse_matmul=true",
      "megablox=true",
      "attention=dot_product",
      "indexer_sparse_training=true",
      "indexer_loss_scaling_factor=0.1",
      "routed_bias_update_rate=0.001",
      "dtype=bfloat16",
      "weight_dtype=bfloat16",
  )

  # glm5.2 is glm5.1 plus IndexShare (configs differ only in RoPE scale and use_index_share), so the glm5_2_scanned
  # case covers the GLM-5 decoder on PRs. The unscanned / pruned IndexShare branches stay on PRs through
  # glm5_model_test.py.
  _GLM5_PR_CASES = (
      {
          "testcase_name": "glm5_2_scanned",
          "model_name": "glm5.2-744b",
          "scan_layers": "true",
          # The first eight layers of GLM-5.2's pattern: shared layers fed by a dense and by a MoE full layer.
          "extra_args": ("base_num_decoder_layers=8", "index_share_pattern=FFFSSSFS"),
      },
  )
  _GLM5_SCHEDULED_CASES = (
      {
          "testcase_name": "glm5_1_scanned",
          "model_name": "glm5.1-744b",
          "scan_layers": "true",
          # Three dense layers and five MoE layers, so both scans run more than one iteration.
          "extra_args": ("base_num_decoder_layers=8",),
      },
      {
          "testcase_name": "glm5_2_unscanned_pruned",
          "model_name": "glm5.2-744b",
          "scan_layers": "false",
          "extra_args": (
              "base_num_decoder_layers=6",
              "first_num_dense_layers=2",
              "index_share_pattern=FFSSFS",
          ),
      },
  )

  @parameterized.named_parameters(*_GLM5_PR_CASES)
  @pytest.mark.cpu_only
  def test_glm5_compile(self, model_name, scan_layers, extra_args):
    """AOT compilation test for GLM-5.2 with scanned IndexShare."""
    self._glm5_compile(model_name, scan_layers, extra_args)

  # The marks must sit below named_parameters: absl copies the generated cases' attributes from the decorated function,
  # so a mark placed above it is dropped and the cases would run on every PR.
  @parameterized.named_parameters(*_GLM5_SCHEDULED_CASES)
  @pytest.mark.scheduled_only
  @pytest.mark.cpu_only
  def test_glm5_compile_scheduled(self, model_name, scan_layers, extra_args):
    """AOT compilation test for GLM-5.1 (scanned) and GLM-5.2 (unscanned pruned IndexShare)."""
    self._glm5_compile(model_name, scan_layers, extra_args)

  def _glm5_compile(self, model_name, scan_layers, extra_args):
    """Shared AOT compilation helper for GLM-5.1 and GLM-5.2 variants."""
    self._aot(
        "compile_topology=v5p-64",
        f"model_name={model_name}",
        *self._GLM5_COMMON,
        "base_emb_dim=1024",
        "base_mlp_dim=2048",
        "base_moe_mlp_dim=512",
        "num_experts=32",
        "vocab_size=32768",
        "max_target_length=1024",
        # Must stay below max_target_length for the indexer loss.
        "indexer_topk=512",
        "ici_fsdp_parallelism=4",
        "ici_expert_parallelism=8",
        f"scan_layers={scan_layers}",
        *extra_args,
    )

  # Full-size HBM-fit check: cpu-integration also runs on every PR, so keep it to scheduled CI.
  @pytest.mark.scheduled_only
  @pytest.mark.integration_test
  @pytest.mark.cpu_only
  def test_glm5_2_full_size(self):
    """Compiles all 78 GLM-5.2 layers at full width, which checks that they fit in v5p-1024 HBM."""
    self._aot(
        "compile_topology=v5p-1024",
        "model_name=glm5.2-744b",
        *self._GLM5_COMMON,
        "max_target_length=4096",
        "ici_fsdp_parallelism=64",
        "ici_expert_parallelism=8",
        "scan_layers=true",
        full_opt=True,
    )
