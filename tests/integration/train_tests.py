# Copyright 2023–2025 Google LLC
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

"""Tests for train.py with various configs"""
import json
import os
import tempfile
import unittest
import numpy as np
import pytest
import jax

from absl.testing import absltest
from maxtext.common.gcloud_stub import is_decoupled
from maxtext.trainers.pre_train.train import main as train_main
from maxtext.utils.globals import MAXTEXT_ASSETS_ROOT
from tests.utils.test_helpers import (
    get_test_config_path,
    get_test_dataset_path,
    get_test_base_output_directory,
    is_rocm_backend,
)


def _small_model_base_emb_dim(device_count):
  """Return a tiny embedding dim divisible by local devices."""
  return ((28 + device_count - 1) // device_count) * device_count


_MOE_OVERRIDES = ["base_moe_mlp_dim=512", "num_experts=8", "num_experts_per_tok=2"]

# One tiny model per Qwen3 decoder block that supports explicit sharding.
_QWEN3_MODELS = {
    "qwen3": ["model_name=qwen3-0.6b"],
    "qwen3_moe": ["model_name=qwen3-30b-a3b"] + _MOE_OVERRIDES,
    "qwen3_custom_moe": [
        "model_name=qwen3-custom-30b-a3b",
        "attention_output_dim=256",
        "moe_expert_input_dim=256",
        # Qwen3CustomMoeDecoderLayer is only registered for the unscanned path.
        "scan_layers=False",
    ]
    + _MOE_OVERRIDES,
}


class TrainTests(unittest.TestCase):
  """Tests train.py with various configs"""

  decoupled = is_decoupled()
  dev_count = jax.device_count()
  _base_output_directory = get_test_base_output_directory()
  dataset_path = get_test_dataset_path()

  _small_model_overrides = [
      f"base_emb_dim={_small_model_base_emb_dim(dev_count)}",
      "base_num_query_heads=4",
      "base_num_kv_heads=4",
      "base_mlp_dim=32",
      "base_num_decoder_layers=2",
      "head_dim=128",
      "max_target_length=128",
      "vocab_size=32",
      # Allow higher unsharded percentage because downscaled models make fixed-size FP8 history tensors relatively larger.
      "sharding_tolerance=0.1",
  ]

  # Routes the MoE layer through dense_matmul, which is what runs wherever the megablox and
  # ragged kernels are unavailable.
  _moe_model_overrides = [
      "decoder_block=mixtral",
      "num_experts=4",
      "num_experts_per_tok=2",
      "base_moe_mlp_dim=32",
      "sparse_matmul=False",
      "megablox=False",
  ]

  _qwen3_overrides = [
      "override_model_config=True",
      "base_num_decoder_layers=2",
      "base_emb_dim=256",
      "base_mlp_dim=512",
      "base_num_query_heads=4",
      "base_num_kv_heads=4",
      "head_dim=128",
      "vocab_size=2048",
      "max_target_length=256",
      # The Qwen3 model configs default to a HuggingFace tokenizer that is not
      # vendored in the repo; use the checked-in tiktoken asset instead.
      "tokenizer_type=tiktoken",
      rf"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizers', 'tokenizer.llama2')}",
  ]

  CONFIGS = {
      "base": [  # short test for train.py with TFDS c4
          None,
          get_test_config_path(),
          f"base_output_directory={_base_output_directory}",
          "run_name=runner_test",
          f"dataset_path={dataset_path}",
          "max_target_length=128",
          "steps=2",
          "enable_checkpointing=False",
          "enable_goodput_recording=False",
          rf"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizers', 'tokenizer.llama2')}",
      ]
      + _small_model_overrides,
      "synthetic": [  # tests base config with synthetic dataset
          None,
          get_test_config_path(),
          f"base_output_directory={_base_output_directory}",
          "run_name=runner_test",
          "steps=2",
          "enable_checkpointing=False",
          "enable_goodput_recording=False",
          "dataset_type=synthetic",
          rf"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizers', 'tokenizer.llama2')}",
      ]
      + _small_model_overrides,
      "pdb_lt_1": [  # tests base config with per_device_batch_size < 1
          None,
          get_test_config_path(),
          f"base_output_directory={_base_output_directory}",
          "run_name=runner_test",
          "dataset_type=synthetic",  # use synthetic dataset_type to decrease training time
          "steps=2",
          "enable_checkpointing=False",
          "enable_goodput_recording=False",
          "per_device_batch_size=0.25",
          "ici_tensor_parallelism=4",
          rf"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizers', 'tokenizer.llama2')}",
      ]
      + _small_model_overrides,
      "int8": [  # tests base config with int8
          None,
          get_test_config_path(),
          f"base_output_directory={_base_output_directory}",
          "run_name=runner_test",
          "dataset_type=synthetic",  # use synthetic dataset_type to decrease training time
          "quantization=int8",
          "steps=2",
          "enable_checkpointing=False",
          "enable_goodput_recording=False",
          rf"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizers', 'tokenizer.llama2')}",
      ]
      + _small_model_overrides,
      "fp8": [  # tests base config with fp8
          None,
          get_test_config_path(),
          f"base_output_directory={_base_output_directory}",
          "run_name=runner_test",
          "dataset_type=synthetic",  # use synthetic dataset_type to decrease training time
          "quantization=fp8",
          "steps=2",
          "enable_checkpointing=False",
          "enable_goodput_recording=False",
          rf"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizers', 'tokenizer.llama2')}",
      ]
      + _small_model_overrides,
      "nanoo_fp8": [  # tests base config with nanoo_fp8
          None,
          get_test_config_path(),
          f"base_output_directory={_base_output_directory}",
          "run_name=runner_test",
          "dataset_type=synthetic",  # use synthetic dataset_type to decrease training time
          "quantization=nanoo_fp8",
          "steps=2",
          "enable_checkpointing=False",
          "enable_goodput_recording=False",
          rf"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizers', 'tokenizer.llama2')}",
      ]
      + _small_model_overrides,
      "moe": [  # tests a MoE model, to be combined with a quantization
          None,
          get_test_config_path(),
          f"base_output_directory={_base_output_directory}",
          "run_name=runner_test",
          "dataset_type=synthetic",  # use synthetic dataset_type to decrease training time
          "steps=2",
          "enable_checkpointing=False",
          "enable_goodput_recording=False",
          rf"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizers', 'tokenizer.llama2')}",
      ]
      + _small_model_overrides
      + _moe_model_overrides,
      "te_fp8_delayedscaling": [  # tests base config with te_fp8_delayedscaling
          None,
          get_test_config_path(),
          f"base_output_directory={_base_output_directory}",
          "run_name=runner_test",
          "dataset_type=synthetic",  # use synthetic dataset_type to decrease training time
          "quantization=te_fp8_delayedscaling",
          "steps=2",
          "enable_checkpointing=False",
          "enable_goodput_recording=False",
          rf"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizers', 'tokenizer.llama2')}",
      ]
      + _small_model_overrides,
      "te_fp8_currentscaling": [  # tests base config with te_fp8_currentscaling
          None,
          get_test_config_path(),
          f"base_output_directory={_base_output_directory}",
          "run_name=runner_test",
          "dataset_type=synthetic",  # use synthetic dataset_type to decrease training time
          "quantization=te_fp8_currentscaling",
          "steps=2",
          "enable_checkpointing=False",
          "enable_goodput_recording=False",
          rf"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizers', 'tokenizer.llama2')}",
      ]
      + _small_model_overrides,
      "te_mxfp8": [  # tests base config with te_mxfp8
          None,
          get_test_config_path(),
          f"base_output_directory={_base_output_directory}",
          "run_name=runner_test",
          "dataset_type=synthetic",  # use synthetic dataset_type to decrease training time
          "quantization=te_mxfp8",
          "steps=2",
          "enable_checkpointing=False",
          "enable_goodput_recording=False",
          rf"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizers', 'tokenizer.llama2')}",
      ]
      + _small_model_overrides,
      "dropout": [  # tests base config with dropout
          None,
          get_test_config_path(),
          f"base_output_directory={_base_output_directory}",
          "run_name=runner_test",
          "dataset_type=synthetic",  # use synthetic dataset_type to decrease training time
          "steps=2",
          "enable_checkpointing=False",
          "enable_goodput_recording=False",
          "max_target_length=128",
          "per_device_batch_size=1",
          "dropout_rate=0.02",
          rf"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizers', 'tokenizer.llama2')}",
      ]
      + _small_model_overrides,
      "hf_input_pipeline": [  # test for train.py with TFDS c4, using HF input pipeline
          None,
          get_test_config_path(),
          f"base_output_directory={_base_output_directory}",
          "run_name=runner_test",
          "steps=2",
          "enable_checkpointing=False",
          "enable_goodput_recording=False",
          "dataset_type=hf",
          "hf_path=parquet",
          f"hf_train_files={dataset_path}/hf/c4/c4-train-00000-of-01637.parquet",
          "tokenizer_path=google-t5/t5-large",
      ]
      + _small_model_overrides,
  }

  @pytest.mark.integration_test
  @pytest.mark.tpu_only
  def test_tpu_base(self):
    train_main(TrainTests.CONFIGS["base"])

  @pytest.mark.integration_test
  @pytest.mark.tpu_only
  def test_tpu_tokamax(self):
    train_main(TrainTests.CONFIGS["synthetic"] + ["use_tokamax_splash=true"])

  @pytest.mark.integration_test
  @pytest.mark.tpu_only
  def test_tpu_ulysses_context_parallelism(self):
    train_main(
        TrainTests.CONFIGS["synthetic"]
        + [
            "attention=flash",
            "use_tokamax_splash=true",
            "ici_context_parallelism=4",
            "context_parallel_strategy=ulysses",
            "context_parallel_load_balance=false",
            "packing=false",
        ]
    )

  @pytest.mark.integration_test
  @pytest.mark.gpu_only
  def test_gpu_base(self):
    train_main(TrainTests.CONFIGS["base"] + ["attention=dot_product"])

  @pytest.mark.integration_test
  @pytest.mark.tpu_only
  def test_tpu_synthetic(self):
    train_main(TrainTests.CONFIGS["synthetic"])

  @pytest.mark.integration_test
  @pytest.mark.gpu_only
  def test_gpu_synthetic(self):
    train_main(TrainTests.CONFIGS["synthetic"] + ["attention=dot_product"])

  @pytest.mark.integration_test
  @pytest.mark.tpu_only
  def test_tpu_pdb_lt_1(self):
    train_main(TrainTests.CONFIGS["pdb_lt_1"])

  @pytest.mark.integration_test
  @pytest.mark.gpu_only
  def test_gpu_pdb_lt_1(self):
    # In decoupled (offline) mode this fractional batch config produces zero TFLOPs and a divide-by-zero in logging.
    if self.decoupled:
      pytest.skip(
          "Skipping pdb_lt_1 in decoupled mode: known divide by zero in TFLOPs logging for per_device_batch_size < 1."
      )
    cfg = TrainTests.CONFIGS["pdb_lt_1"] + ["attention=dot_product"]
    train_main(cfg)

  @pytest.mark.integration_test
  @pytest.mark.tpu_only
  def test_tpu_int8(self):
    train_main(TrainTests.CONFIGS["int8"])

  @pytest.mark.integration_test
  @pytest.mark.gpu_only
  def test_gpu_int8(self):
    train_main(TrainTests.CONFIGS["int8"] + ["attention=dot_product"])

  @pytest.mark.integration_test
  @pytest.mark.tpu_only
  def test_tpu_fp8(self):
    train_main(TrainTests.CONFIGS["fp8"])

  @pytest.mark.external_serving
  @pytest.mark.integration_test
  @pytest.mark.gpu_only
  def test_gpu_fp8(self):
    train_main(TrainTests.CONFIGS["fp8"] + ["attention=dot_product"])

  @pytest.mark.external_serving
  @pytest.mark.integration_test
  @pytest.mark.gpu_only
  def test_gpu_nanoo_fp8(self):
    train_main(TrainTests.CONFIGS["nanoo_fp8"] + ["attention=dot_product"])

  # The quantized MoE tests below carry no hardware marker on purpose. They cover how the
  # quantized einsums are bound to the MoE layer, which breaks on every backend when it breaks,
  # and both fp8 flavors are emulated in XLA rather than needing hardware support.
  @pytest.mark.integration_test
  def test_moe_int8(self):
    train_main(TrainTests.CONFIGS["moe"] + ["quantization=int8"])

  @pytest.mark.integration_test
  def test_moe_fp8(self):
    train_main(TrainTests.CONFIGS["moe"] + ["quantization=fp8"])

  @pytest.mark.integration_test
  def test_moe_nanoo_fp8(self):
    train_main(TrainTests.CONFIGS["moe"] + ["quantization=nanoo_fp8"])

  @pytest.mark.integration_test
  def test_moe_fp8_token_dropping(self):
    # capacity_factor > 0 adds the dispatch and combine einsums to the ones above.
    train_main(TrainTests.CONFIGS["moe"] + ["quantization=fp8", "capacity_factor=1.25"])

  @pytest.mark.skip(reason="No runner with GPU arch >= 89 is available")
  @pytest.mark.integration_test
  @pytest.mark.gpu_only
  def test_gpu_te_fp8_delayedscaling(self):
    train_main(TrainTests.CONFIGS["te_fp8_delayedscaling"] + ["attention=dot_product"])

  @pytest.mark.skip(reason="No runner with GPU arch >= 89 is available")
  @pytest.mark.integration_test
  @pytest.mark.gpu_only
  def test_gpu_te_fp8_delayedscaling_tsp_full_cgemm(self):
    if jax.process_count() <= 1:
      pytest.skip("Requires rank-per-GPU launch (JAX_PROCESS_COUNT > 1)")
    if jax.local_device_count() != 1:
      pytest.skip(f"Requires rank-per-GPU launch (local_device_count==1), " f"got {jax.local_device_count()}")
    if not jax.distributed.is_initialized():
      pytest.skip("Requires jax.distributed.initialize() (hardware=gpu_multiprocess)")

    train_main(
        TrainTests.CONFIGS["te_fp8_delayedscaling"]
        + ["attention=dot_product", "ici_tensor_sequence_parallelism=2", "te_comm_gemm_overlap=full"]
    )

  @pytest.mark.skip(reason="No runner with GPU arch >= 89 is available")
  @pytest.mark.integration_test
  @pytest.mark.gpu_only
  def test_gpu_te_fp8_currentscaling(self):
    train_main(TrainTests.CONFIGS["te_fp8_currentscaling"] + ["attention=dot_product"])

  @pytest.mark.skip(reason="No runner with GPU arch >= 100 is available")
  @pytest.mark.integration_test
  @pytest.mark.gpu_only
  def test_gpu_te_mxfp8(self):
    train_main(TrainTests.CONFIGS["te_mxfp8"] + ["attention=dot_product"])

  @pytest.mark.skip(reason="No runner with GPU arch >= 100 is available")
  @pytest.mark.integration_test
  @pytest.mark.gpu_only
  def test_gpu_te_nvfp4(self):
    train_main(TrainTests.CONFIGS["te_nvfp4"] + ["attention=dot_product"])

  @pytest.mark.integration_test
  @pytest.mark.tpu_only
  def test_tpu_dropout(self):
    train_main(TrainTests.CONFIGS["dropout"])

  @pytest.mark.integration_test
  @pytest.mark.gpu_only
  @pytest.mark.skip(reason="b/454386843. Issue when upgrading to jax=0.8.0")
  def test_gpu_dropout(self):
    train_main(TrainTests.CONFIGS["dropout"] + ["attention=dot_product"])

  @pytest.mark.integration_test
  @pytest.mark.tpu_only
  @unittest.skipIf(is_decoupled(), "Bypassed in offline decoupled runs (no HuggingFace internet)")
  def test_tpu_hf_input_pipeline(self):
    train_main(TrainTests.CONFIGS["hf_input_pipeline"])

  @pytest.mark.external_serving
  @pytest.mark.integration_test
  @pytest.mark.gpu_only
  def test_gpu_hf_input_pipeline(self):
    train_main(TrainTests.CONFIGS["hf_input_pipeline"] + ["attention=dot_product"])

  @pytest.mark.integration_test
  @pytest.mark.gpu_only
  def test_gpu_cudnn_flash_te(self):
    if not jax.local_devices() or jax.local_devices()[0].platform != "cuda":
      pytest.skip("Skipping cudnn_flash_te test: CUDA/cuDNN not available")
    os.environ["NVTE_FUSED_ATTN"] = "1"  # Enable fused attention
    cudnn_flash_te = [  # tests base config on GPU with flash attention
        None,
        get_test_config_path(),
        f"base_output_directory={self._base_output_directory}",
        "run_name=runner_test",
        "dataset_type=synthetic",  # use synthetic dataset_type to decrease training time
        "steps=2",
        "enable_checkpointing=False",
        "enable_goodput_recording=False",
        "attention=cudnn_flash_te",
        "packing=False",
        rf"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizers', 'tokenizer.llama2')}",
    ]
    train_main(cudnn_flash_te)

  @pytest.mark.integration_test
  @pytest.mark.gpu_only
  @pytest.mark.skip(reason="b/489133823. Previously transient in b/462548581.")
  def test_gpu_context_parallelism(self):
    os.environ["NVTE_FUSED_ATTN"] = "1"  # Enable fused attention
    context_parallel = [  # tests base config on GPU with All-Gather based context parallelism
        None,
        get_test_config_path(),
        f"base_output_directory={self._base_output_directory}",
        "run_name=runner_test",
        "dataset_type=synthetic",  # use synthetic dataset_type to decrease training time
        "steps=10",
        "enable_checkpointing=False",
        "enable_goodput_recording=False",
        "attention=cudnn_flash_te",
        "ici_fsdp_parallelism=-1",
        "ici_context_parallelism=2",
        "context_parallel_strategy=all_gather",
        "context_parallel_load_balance=True",
        "packing=False",
        rf"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizers', 'tokenizer.llama2')}",
    ]
    if self.decoupled:
      context_parallel.append("shardy=False")
      axis = next(
          (
              int(a.split("=")[1])
              for a in context_parallel
              if isinstance(a, str) and a.startswith("ici_context_parallelism=")
          ),
          1,
      )
      fsdp = self.dev_count // axis if axis > 0 and self.dev_count % axis == 0 else self.dev_count
      context_parallel.append(f"ici_fsdp_parallelism={fsdp}")
    print("Using dataset_path:", self.dataset_path)
    print("Exists:", os.path.exists(self.dataset_path))
    train_main(context_parallel)

  @pytest.mark.integration_test
  @pytest.mark.gpu_only
  @pytest.mark.skip(reason="b/489133823. Previously transient in b/462548581.")
  def test_gpu_tensor_parallelism(self):
    os.environ["NVTE_FUSED_ATTN"] = "1"  # Enable fused attention
    tensor_parallel = [  # tests base config on GPU with Tensor Parallelism
        None,
        get_test_config_path(),
        f"base_output_directory={self._base_output_directory}",
        "run_name=runner_test",
        f"dataset_path={self.dataset_path}",
        "steps=10",
        "enable_checkpointing=False",
        "enable_goodput_recording=False",
        "attention=cudnn_flash_te",
        "ici_fsdp_parallelism=-1",
        "ici_tensor_parallelism=2",
        "packing=False",
        rf"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizers', 'tokenizer.llama2')}",
    ]
    if self.decoupled:
      tensor_parallel.append("shardy=False")
      axis = next(
          (
              int(a.split("=")[1])
              for a in tensor_parallel
              if isinstance(a, str) and a.startswith("ici_tensor_parallelism=")
          ),
          1,
      )
      fsdp = self.dev_count // axis if axis > 0 and self.dev_count % axis == 0 else self.dev_count
      tensor_parallel.append(f"ici_fsdp_parallelism={fsdp}")
    train_main(tensor_parallel)

  @pytest.mark.integration_test
  @pytest.mark.gpu_only
  def test_gpu_optimizer_offload(self):
    os.environ["NVTE_FUSED_ATTN"] = "1"  # Enable fused attention
    optimizer_offload = [  # tests base config on GPU with optimizer state offload
        None,
        get_test_config_path(),
        f"base_output_directory={self._base_output_directory}",
        "run_name=runner_test",
        f"dataset_path={self.dataset_path}",
        "steps=10",
        "attention=dot_product",
        "optimizer_memory_host_offload=True",  # enable optimizer state offload
        "dataset_type=synthetic",
        "enable_checkpointing=False",
        "enable_goodput_recording=False",
        rf"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizers', 'tokenizer.llama2')}",
    ]
    train_main(optimizer_offload)

  @pytest.mark.integration_test
  @pytest.mark.gpu_only
  def test_gpu_parameter_offload(self):
    if is_rocm_backend():
      # JAX 0.9.1 MSIT enforces memory_space typematch across VJP; MaxText's
      # pinned_host params + device compute mismatch the cotangent at the jit
      # boundary.
      pytest.skip("Parameter memory host offload: JAX MSIT VJP typematch fails for pinned_host params.")
    os.environ["NVTE_FUSED_ATTN"] = "1"  # Enable fused attention
    parameter_offload = [  # tests base config on GPU with parameter offload
        None,
        get_test_config_path(),
        f"base_output_directory={self._base_output_directory}",
        "run_name=runner_test",
        f"dataset_path={self.dataset_path}",
        "steps=10",
        "param_scan_axis=0",  # scan axis 0 is required for parameter offload
        "attention=dot_product",
        "parameter_memory_host_offload=True",  # enable parameter offload
        "dataset_type=synthetic",
        "enable_checkpointing=False",
        "enable_goodput_recording=False",
        rf"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizers', 'tokenizer.llama2')}",
    ]
    train_main(parameter_offload)

  @pytest.mark.gpu_only
  def test_gpu_cudnn_flash_jax(self):
    if not jax.local_devices() or jax.local_devices()[0].platform != "cuda":
      pytest.skip("Skipping cudnn_flash_jax test: CUDA/cuDNN not available")
    cudnn_flash_jax = [  # tests base config on GPU with flash attention
        None,
        get_test_config_path(),
        f"base_output_directory={self._base_output_directory}",
        "run_name=runner_test",
        f"dataset_path={self.dataset_path}",
        "dataset_type=synthetic",  # use synthetic dataset_type to decrease training time
        "steps=2",
        "enable_checkpointing=False",
        "enable_goodput_recording=False",
        "attention=cudnn_flash_jax",
        "packing=False",
        "shardy=False",  # The cudnn kernel is not compatible with shardy, see (b/425746362).
        rf"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizers', 'tokenizer.llama2')}",
    ]
    train_main(cudnn_flash_jax)

  @pytest.mark.integration_test
  def test_base_model_shardy_false(self):
    train_main(TrainTests.CONFIGS["synthetic"] + ["shardy=False"])

  @pytest.mark.integration_test
  @pytest.mark.tpu_only
  # TODO(b/517509898): Skip ZeRo-1 compiler Segfault on TPU7x SparseCore platforms
  @pytest.mark.skip_on_tpu7x
  def test_tpu_zero1_gradient_accumulation(self):
    zero1_ga = [  # tests Zero-1 optimizer sharding with gradient accumulation
        None,
        get_test_config_path(),
        f"base_output_directory={self._base_output_directory}",
        "run_name=runner_test",
        f"dataset_path={self.dataset_path}",
        "steps=3",
        "enable_checkpointing=False",
        "enable_goodput_recording=False",
        "dataset_type=synthetic",
        "remat_policy=minimal",
        "max_target_length=512",
        "per_device_batch_size=2",
        "ici_data_parallelism=-1",
        "dcn_data_parallelism=1",
        "ici_fsdp_parallelism=1",
        "dcn_fsdp_parallelism=1",
        "gradient_accumulation_steps=8",
        "shard_optimizer_over_data=True",
        "shard_mode=explicit",
        "decoder_block=llama2",
        rf"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizers', 'tokenizer.llama2')}",
    ]
    train_main(zero1_ga)

  def _qwen3_losses(self, run_name, extra_args):
    """Trains a tiny Qwen3 model for a few steps and returns its per-step losses."""
    with tempfile.TemporaryDirectory() as tmp_dir:
      metrics_file = os.path.join(tmp_dir, "metrics.txt")
      train_main(
          [
              None,
              get_test_config_path(),
              f"base_output_directory={self._base_output_directory}",
              f"dataset_path={self.dataset_path}",
              f"run_name={run_name}",
              f"metrics_file={metrics_file}",
              "dataset_type=synthetic",
              "steps=3",
              "enable_checkpointing=False",
              "enable_goodput_recording=False",
          ]
          + self._qwen3_overrides
          + list(extra_args)
      )
      with open(metrics_file, "rt", encoding="utf8") as f:
        return [json.loads(line)["learning/loss"] for line in f if line.strip()]

  @pytest.mark.integration_test
  @pytest.mark.tpu_only
  def test_tpu_qwen3_explicit_sharding_matches_auto(self):
    """Explicit sharding only changes how layouts are expressed, so the losses must not move.

    Each decoder block is paired with the parallelism that stresses it most:
    tensor parallelism puts the `heads` axis on the same mesh axis as the
    QK-norm scale, and expert parallelism shards the MoE dispatch.
    """
    parallelism = {
        "qwen3": ["ici_fsdp_parallelism=1", "ici_tensor_parallelism=-1"],
        "qwen3_moe": ["ici_fsdp_parallelism=1", "ici_expert_parallelism=-1"],
        "qwen3_custom_moe": [],
    }
    for decoder_block, model_args in _QWEN3_MODELS.items():
      with self.subTest(decoder_block=decoder_block):
        args = model_args + parallelism[decoder_block]
        auto_losses = self._qwen3_losses(f"{decoder_block}_auto", args + ["shard_mode=auto"])
        explicit_losses = self._qwen3_losses(f"{decoder_block}_explicit", args + ["shard_mode=explicit"])
        print(f"[{decoder_block}] auto losses: {auto_losses}", flush=True)
        print(f"[{decoder_block}] explicit losses: {explicit_losses}", flush=True)
        self.assertTrue(auto_losses, "auto run produced no metrics")
        # The two runs execute the same math, so they match bit-for-bit.
        np.testing.assert_allclose(explicit_losses, auto_losses, rtol=1e-6, atol=0.0)

  @pytest.mark.integration_test
  @pytest.mark.tpu_only
  # TODO(b/517509898): Skip ZeRo-1 compiler Segfault on TPU7x SparseCore platforms
  @pytest.mark.skip_on_tpu7x
  def test_tpu_qwen3_zero1_gradient_accumulation(self):
    """ZeRO-1 only shards the optimizer state, so it must not change the loss trajectory.

    Under explicit sharding this routes the accumulated gradients through the
    `reduced`/`unreduced` PartitionSpec labels applied in
    `maxtext.utils.gradient_accumulation`.
    """
    zero1_ga = [
        "remat_policy=minimal",
        "per_device_batch_size=2",
        "ici_data_parallelism=-1",
        "dcn_data_parallelism=1",
        "ici_fsdp_parallelism=1",
        "dcn_fsdp_parallelism=1",
        "gradient_accumulation_steps=8",
    ]
    for decoder_block in ("qwen3", "qwen3_moe"):
      with self.subTest(decoder_block=decoder_block):
        args = _QWEN3_MODELS[decoder_block] + zero1_ga
        baseline = self._qwen3_losses(
            f"{decoder_block}_ga",
            args + ["shard_mode=auto", "shard_optimizer_over_data=False"],
        )
        sharded = self._qwen3_losses(
            f"{decoder_block}_ga_zero1",
            args + ["shard_mode=explicit", "shard_optimizer_over_data=True"],
        )
        print(f"[{decoder_block}] auto + GA losses: {baseline}", flush=True)
        print(f"[{decoder_block}] explicit + ZeRO-1 + GA losses: {sharded}", flush=True)
        self.assertTrue(baseline, "baseline run produced no metrics")
        # ZeRO-1 reassociates the gradient all-reduce, so allow a little float slack.
        np.testing.assert_allclose(sharded, baseline, rtol=1e-4, atol=0.0)

  @pytest.mark.integration_test
  @pytest.mark.tpu_only
  def test_tpu_gemma_zero1_gradient_accumulation_explicit(self):
    """Gemma under ZeRO-1 + gradient accumulation + explicit sharding.

    This is the combination the explicit annotations exist for: with the optimizer
    moments sharded over the `data` axis, explicit sharding type-checks the gradient
    layout against the moment layout instead of letting GSPMD reconcile them, so any
    missing or wrong `out_sharding` in a Gemma layer fails the step outright.
    """
    # Gemma 3 reads its local/global attention pattern and rope scaling off the named
    # model config, so it cannot run under the placeholder "default" model name.
    families = [
        ("gemma", "tokenizer.gemma", []),
        ("gemma2", "tokenizer.gemma", []),
        ("gemma3", "tokenizer.gemma3", ["model_name=gemma3-4b", "override_model_config=True"]),
    ]
    for decoder_block, tokenizer, extra_args in families:
      with self.subTest(decoder_block=decoder_block):
        gemma_zero1_ga = [
            None,
            get_test_config_path(),
            f"base_output_directory={self._base_output_directory}",
            "run_name=runner_test",
            f"dataset_path={self.dataset_path}",
            "steps=3",
            "enable_checkpointing=False",
            "enable_goodput_recording=False",
            "dataset_type=synthetic",
            "remat_policy=minimal",
            "max_target_length=512",
            "per_device_batch_size=2",
            "base_emb_dim=256",
            "base_mlp_dim=512",
            "base_num_query_heads=4",
            "base_num_kv_heads=4",
            "base_num_decoder_layers=2",
            "head_dim=64",
            # The splash kernel cannot build a mask for a downscaled Gemma (its sliding
            # window leaves empty blocks); dot product attention keeps the focus on sharding.
            "attention=dot_product",
            # Both axes must be > 1 for ZeRO-1 to have a "data" axis to shard the moments
            # over while FSDP still shards the parameters. Filling fsdp with the remaining
            # devices keeps this valid on 4- and 8-device hosts alike. Note that
            # data-parallel-only (ici_fsdp_parallelism=1) segfaults the TPU7x compiler,
            # which is what b/517509898 tracks for the llama2 test above.
            "ici_data_parallelism=2",
            "dcn_data_parallelism=1",
            "ici_fsdp_parallelism=-1",
            "dcn_fsdp_parallelism=1",
            "gradient_accumulation_steps=4",
            "shard_optimizer_over_data=True",
            "shard_mode=explicit",
            f"decoder_block={decoder_block}",
            rf"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizers', tokenizer)}",
        ] + extra_args
        train_main(gemma_zero1_ga)

  @pytest.mark.integration_test
  @pytest.mark.gpu_only
  @pytest.mark.scheduled_only
  @pytest.mark.skip(reason="b/489133823. Previously transient in b/462548581.")
  def test_gpu_zero1_gradient_accumulation(self):
    os.environ["NVTE_FUSED_ATTN"] = "1"  # Enable fused attention
    zero1_ga = [  # tests Zero-1 optimizer sharding with gradient accumulation
        None,
        get_test_config_path(),
        f"base_output_directory={self._base_output_directory}",
        "run_name=runner_test",
        f"dataset_path={self.dataset_path}",
        "steps=10",
        "enable_checkpointing=False",
        "enable_goodput_recording=False",
        "dataset_type=synthetic",
        "attention=cudnn_flash_te",
        "remat_policy=minimal",
        "scan_layers=False",
        "max_target_length=8192",
        "per_device_batch_size=2",
        "ici_data_parallelism=-1",
        "dcn_data_parallelism=1",
        "ici_fsdp_parallelism=1",
        "dcn_fsdp_parallelism=1",
        "gradient_accumulation_steps=8",
        "shard_optimizer_over_data=True",
        "override_model_config=True",
        rf"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizers', 'tokenizer.llama2')}",
    ]
    train_main(zero1_ga)

  @pytest.mark.integration_test
  @pytest.mark.gpu_only
  def test_gpu_packed_attention(self):
    gpu_device = jax.devices("gpu")[0]
    compute_capability = getattr(gpu_device, "compute_capability", None)
    try:
      if float(compute_capability) < 9.0:
        pytest.skip("Packed (THD) attention is only supported on sm90+!")
    except Exception:  # pylint: disable=broad-exception-caught
      # Non-numeric or unknown capability (e.g. ROCm 'gfx942') — skip the test.
      print("checking if Packed THD attention is supported on this host...")
      pytest.skip("Packed (THD) attention is only supported on sm90+!")
    os.environ["NVTE_FUSED_ATTN"] = "1"  # Enable fused attention
    packed_attention = [  # tests base config on GPU with Packed (THD) attention
        None,
        get_test_config_path(),
        f"base_output_directory={self._base_output_directory}",
        "run_name=runner_test",
        "dataset_type=synthetic",  # use synthetic dataset_type to decrease training time
        "steps=10",
        "enable_checkpointing=False",
        "enable_goodput_recording=False",
        "attention=cudnn_flash_te",
        "ici_fsdp_parallelism=-1",
        "packing=True",
        rf"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizers', 'tokenizer.llama2')}",
    ]
    train_main(packed_attention)

  @pytest.mark.integration_test
  @pytest.mark.gpu_only
  @pytest.mark.skip(reason="b/489133823. Previously transient in b/462548581.")
  def test_gpu_ring_attention(self):
    rocm_backend = is_rocm_backend()
    os.environ["NVTE_FUSED_ATTN"] = "1"  # Enable fused attention
    os.environ["NVTE_FUSED_RING_ATTENTION_USE_SCAN"] = "0"  # Disable scan for ring attention
    ring_attention = [  # tests base config on GPU with ring attention
        None,
        get_test_config_path(),
        f"base_output_directory={self._base_output_directory}",
        "run_name=runner_test",
        "dataset_type=synthetic",  # use synthetic dataset_type to decrease training time
        "steps=1" if rocm_backend else "steps=10",
        "enable_checkpointing=False",
        "enable_goodput_recording=False",
        "attention=cudnn_flash_te",
        "ici_fsdp_parallelism=-1",
        "ici_context_parallelism=2",
        "context_parallel_load_balance=True",
        "context_parallel_strategy=ring",
        "packing=False",
        "hardware=gpu",
        rf"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizers', 'tokenizer.llama2')}",
    ]
    if rocm_backend:
      # Keep the ROCm ring-attention smoke test small enough to avoid long TE/XLA compile times.
      ring_attention.extend(
          [
              "max_target_length=512",
              "base_emb_dim=1024",
              "base_mlp_dim=4096",
              "base_num_query_heads=8",
              "base_num_kv_heads=8",
              "base_num_decoder_layers=2",
          ]
      )
    train_main(ring_attention)

  @pytest.mark.integration_test
  @pytest.mark.gpu_only
  def test_gpu_ring_attention_with_packing(self):
    rocm_backend = is_rocm_backend()
    if not rocm_backend:
      gpu_device = jax.devices("gpu")[0]
      compute_capability = getattr(gpu_device, "compute_capability", None)
      try:
        if float(compute_capability) < 9.0:
          pytest.skip("Ring attention with packing is only supported on sm90+!")
      except Exception:  # pylint: disable=broad-exception-caught
        pytest.skip("Ring attention with packing is only supported on sm90+!")
    os.environ["NVTE_FUSED_ATTN"] = "1"  # Enable fused attention
    os.environ["NVTE_FUSED_RING_ATTENTION_USE_SCAN"] = "0"  # Disable scan for ring attention
    thd_ring_attention = [  # tests base config on GPU with ring attention + packing
        None,
        get_test_config_path(),
        f"base_output_directory={self._base_output_directory}",
        "run_name=runner_test",
        f"dataset_path={self.dataset_path}",
        "steps=1" if rocm_backend else "steps=10",
        "enable_checkpointing=False",
        "enable_goodput_recording=False",
        "attention=cudnn_flash_te",
        "ici_fsdp_parallelism=-1",
        "ici_context_parallelism=2",
        "context_parallel_load_balance=True",
        "context_parallel_strategy=ring",
        "packing=True",
        "hardware=gpu",
        rf"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizers', 'tokenizer.llama2')}",
    ]
    if rocm_backend:
      # Keep the ROCm packed-ring smoke test small enough to avoid long TE/XLA compile times.
      thd_ring_attention.extend(
          [
              "max_segments_per_seq=2",
              "max_target_length=512",
              "base_emb_dim=1024",
              "base_mlp_dim=4096",
              "base_num_query_heads=8",
              "base_num_kv_heads=8",
              "base_num_decoder_layers=2",
          ]
      )
    train_main(thd_ring_attention)

  @pytest.mark.integration_test
  @pytest.mark.tpu_only
  def test_fractional_eval_batch_size(self):
    frac_eval = [  # tests Zero-1 optimizer sharding with gradient accumulation
        None,
        get_test_config_path(),
        f"base_output_directory={self._base_output_directory}",
        "run_name=runner_test",
        f"dataset_path={self.dataset_path}",
        "steps=5",
        "enable_checkpointing=False",
        "enable_goodput_recording=False",
        "max_target_length=4096",
        "dataset_type=synthetic",
        "remat_policy=minimal",
        "per_device_batch_size=1",
        "ici_expert_parallelism=-1",
        "ici_fsdp_parallelism=1",
        "use_ring_of_experts=True",
        "model_name=deepseek3-test",
        "eval_per_device_batch_size=0.25",
        "eval_interval=3",
        "eval_steps=1",
        "custom_mesh_and_rule_for_eval=ep-as-cp",
        "override_model_config=true",
        "base_num_decoder_layers=7",
        rf"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizers', 'tokenizer.llama2')}",
    ]
    train_main(frac_eval)


if __name__ == "__main__":
  absltest.main()
