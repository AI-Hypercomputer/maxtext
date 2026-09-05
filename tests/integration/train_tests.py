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

# One tiny model per Mistral-family decoder block that supports explicit sharding.
_MISTRAL_MODELS = {
    "mistral": ["model_name=mistral-7b"],
    "mixtral": [
        "model_name=mixtral-8x7b",
        # RoutedMoE.dense_matmul is not onboarded to explicit sharding yet (a gap it
        # shares with qwen3_moe), so exercise the sparse_matmul path.
        "sparse_matmul=True",
        "megablox=True",
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

  _moe_expert_overrides = [
      "decoder_block=mixtral",
      "num_experts=4",
      "num_experts_per_tok=2",
      "base_moe_mlp_dim=32",
  ]

  # Routes the MoE layer through dense_matmul, which is what runs wherever the megablox and
  # ragged kernels are unavailable.
  _moe_model_overrides = _moe_expert_overrides + [
      "sparse_matmul=False",
      "megablox=False",
  ]

  # The sparse_matmul path, where megablox builds and uses the gmm quantization rule for real
  # rather than falling back to ragged_dot.
  _moe_sparse_model_overrides = _moe_expert_overrides + [
      "sparse_matmul=True",
      "megablox=True",
  ]

  # Every operand has to divide into its tile, and the default tiles are far larger than a
  # downscaled model. The embedding dim is 28 or 32 depending on the device count, so 4 is
  # the largest tile that fits it either way.
  _megablox_tile_overrides = [
      f"{matrix}_tile_{direction}_{dim}={size}"
      for matrix in ("wi", "wo")
      for direction in ("fwd", "dlhs", "drhs")
      for dim, size in (("batch_seq", 16), ("embed_dim", 4), ("mlp_dim", 16))
  ]

  _qwen3_overrides = [
      "override_model_config=True",
      "base_num_decoder_layers=2",
      "base_emb_dim=256",
      "base_mlp_dim=512",
      "base_num_query_heads=8",
      "base_num_kv_heads=8",
      "head_dim=128",
      "vocab_size=2048",
      "max_target_length=256",
      # The Qwen3 model configs default to a HuggingFace tokenizer that is not
      # vendored in the repo; use the checked-in tiktoken asset instead.
      "tokenizer_type=tiktoken",
      rf"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizers', 'tokenizer.llama2')}",
  ]

  _mistral_overrides = [
      "override_model_config=True",
      "base_num_decoder_layers=2",
      "base_emb_dim=256",
      "base_mlp_dim=512",
      "base_num_query_heads=8",
      "base_num_kv_heads=8",
      "head_dim=128",
      "vocab_size=2048",
      "max_target_length=256",
      rf"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizers', 'tokenizer.mistral-v1')}",
  ]

  _qwen2_overrides = [
      "model_name=qwen2.5-7b",
      "override_model_config=True",
      "base_num_decoder_layers=2",
      "base_emb_dim=256",
      "base_mlp_dim=512",
      "base_num_query_heads=8",
      "base_num_kv_heads=8",
      "head_dim=128",
      "vocab_size=2048",
      "max_target_length=256",
      # The Qwen2.5 model configs default to a HuggingFace tokenizer that is not
      # vendored in the repo; use the checked-in tiktoken asset instead.
      "tokenizer_type=tiktoken",
      rf"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizers', 'tokenizer.llama2')}",
  ]

  # Kimi-K2 runs on the deepseek decoder block, so this is the explicit-sharding coverage
  # for MLA attention and for the shared-expert, sigmoid-routed MoE. Keeping
  # first_num_dense_layers=1 below four layers leaves one dense and three MoE layers, so
  # both deepseek sublayers run.
  _kimi_overrides = [
      "model_name=kimi-k2-1t",
      "override_model_config=True",
      "base_num_decoder_layers=4",
      "first_num_dense_layers=1",
      "base_emb_dim=256",
      "base_mlp_dim=512",
      "base_moe_mlp_dim=256",
      "base_num_query_heads=4",
      "base_num_kv_heads=4",
      "q_lora_rank=32",
      "kv_lora_rank=16",
      "num_experts=8",
      "num_experts_per_tok=2",
      "vocab_size=2048",
      "max_target_length=256",
      # RoutedMoE.dense_matmul is not onboarded to explicit sharding yet, so exercise the
      # sparse_matmul path.
      "sparse_matmul=True",
      "megablox=True",
      # Kimi-K2 defaults to a HuggingFace tokenizer that is not vendored in the repo.
      "tokenizer_type=tiktoken",
      rf"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizers', 'tokenizer.llama2')}",
  ]

  # Four layers is one whole `inhomogeneous_layer_cycle_interval`, so both the
  # gated-delta-net and the full-attention layer run. head_dim has to stay at 256
  # because `mrope_section` sums to head_dim * partial_rotary_factor / 2.
  _qwen3_5_overrides = [
      "model_name=qwen3.5-35b-a3b",
      "override_model_config=True",
      "base_num_decoder_layers=4",
      "base_emb_dim=256",
      "base_mlp_dim=256",
      "base_moe_mlp_dim=256",
      "base_num_query_heads=8",
      "base_num_kv_heads=8",
      "head_dim=256",
      "num_experts=8",
      "num_experts_per_tok=2",
      "gdn_num_key_heads=4",
      "gdn_num_value_heads=8",
      "gdn_key_head_dim=64",
      "gdn_value_head_dim=64",
      "vocab_size=2048",
      "max_target_length=256",
      # RoutedMoE.dense_matmul is not onboarded to explicit sharding yet.
      "sparse_matmul=True",
      "megablox=True",
      # The Qwen3.5 model configs default to a HuggingFace tokenizer that is not
      # vendored in the repo; use the checked-in tiktoken asset instead.
      "tokenizer_type=tiktoken",
      (
          rf"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizers', 'tokenizer.llama2')}"
      ),
  ]

  # Same sublayers as Qwen3.5, wired together by Qwen3NextScannableBlock rather than a
  # Python loop. No MRoPE, so head_dim is free.
  _qwen3_next_overrides = [
      "model_name=qwen3-next-80b-a3b",
      "override_model_config=True",
      "base_num_decoder_layers=4",
      "base_emb_dim=256",
      "base_mlp_dim=256",
      "base_moe_mlp_dim=256",
      "base_num_query_heads=8",
      "base_num_kv_heads=8",
      "head_dim=128",
      "num_experts=8",
      "num_experts_per_tok=2",
      "gdn_num_key_heads=4",
      "gdn_num_value_heads=8",
      "gdn_key_head_dim=64",
      "gdn_value_head_dim=64",
      "vocab_size=2048",
      "max_target_length=256",
      "sparse_matmul=True",
      "megablox=True",
  ]

  _QWEN3_HYBRID_MODELS = {
      "qwen3_5": _qwen3_5_overrides,
      "qwen3_next": _qwen3_next_overrides,
  }

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
      "moe_sparse": [  # tests a MoE model on the sparse_matmul path, to be combined with a quantization
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
      + _moe_sparse_model_overrides
      + _megablox_tile_overrides,
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

  # The sparse_matmul tests below carry no hardware marker for the same reasons. What they cover
  # is an attribute read during tracing rather than anything a kernel does, and megablox runs the
  # quantized grouped matmul on CPU through its interpret mode.
  @pytest.mark.integration_test
  def test_moe_fp8_sparse_matmul(self):
    train_main(TrainTests.CONFIGS["moe_sparse"] + ["quantization=fp8"])

  @pytest.mark.integration_test
  def test_moe_nanoo_fp8_sparse_matmul(self):
    train_main(TrainTests.CONFIGS["moe_sparse"] + ["quantization=nanoo_fp8"])

  # int8 takes the `quant_dg` branch of the same read, which the fp8 tests never reach.
  @pytest.mark.integration_test
  def test_moe_int8_sparse_matmul(self):
    train_main(TrainTests.CONFIGS["moe_sparse"] + ["quantization=int8"])

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

  def _losses(self, run_name, model_overrides, extra_args):
    """Trains a tiny model for a few steps and returns its per-step losses."""
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
          + list(model_overrides)
          + list(extra_args)
      )
      with open(metrics_file, "rt", encoding="utf8") as f:
        return [json.loads(line)["learning/loss"] for line in f if line.strip()]

  def _qwen3_losses(self, run_name, extra_args):
    """Trains a tiny Qwen3 model for a few steps and returns its per-step losses."""
    return self._losses(run_name, self._qwen3_overrides, extra_args)

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

  def _mistral_losses(self, run_name, extra_args):
    """Trains a tiny Mistral/Mixtral model for a few steps and returns its per-step losses."""
    return self._losses(run_name, self._mistral_overrides, extra_args)

  @pytest.mark.integration_test
  @pytest.mark.tpu_only
  def test_tpu_mistral_explicit_sharding_matches_auto(self):
    """Explicit sharding only changes how layouts are expressed, so the losses must not move.

    Each decoder block is paired with the parallelism that stresses it most:
    tensor parallelism shards the dense MLP intermediate, and expert parallelism
    shards the MoE dispatch.
    """
    parallelism = {
        "mistral": ["ici_fsdp_parallelism=1", "ici_tensor_parallelism=-1"],
        "mixtral": ["ici_fsdp_parallelism=1", "ici_expert_parallelism=-1"],
    }
    # Under expert parallelism the two modes are bit-for-bit. Under tensor parallelism
    # pinning the MLP intermediate reassociates the backward reduction over the tensor
    # axis, which drifts by a few ULPs by the third step; the runs stay bit-for-bit if
    # the same model is run under FSDP instead.
    rtol = {"mistral": 1e-5, "mixtral": 1e-6}
    for decoder_block, model_args in _MISTRAL_MODELS.items():
      with self.subTest(decoder_block=decoder_block):
        args = model_args + parallelism[decoder_block]
        auto_losses = self._mistral_losses(f"{decoder_block}_auto", args + ["shard_mode=auto"])
        explicit_losses = self._mistral_losses(f"{decoder_block}_explicit", args + ["shard_mode=explicit"])
        print(f"[{decoder_block}] auto losses: {auto_losses}", flush=True)
        print(f"[{decoder_block}] explicit losses: {explicit_losses}", flush=True)
        self.assertTrue(auto_losses, "auto run produced no metrics")
        np.testing.assert_allclose(explicit_losses, auto_losses, rtol=rtol[decoder_block], atol=0.0)

  @pytest.mark.integration_test
  @pytest.mark.tpu_only
  # TODO(b/517509898): Skip ZeRo-1 compiler Segfault on TPU7x SparseCore platforms
  @pytest.mark.skip_on_tpu7x
  def test_tpu_mistral_zero1_gradient_accumulation(self):
    """ZeRO-1 only shards the optimizer state, so it must not change the loss trajectory.

    Under explicit sharding this routes the accumulated gradients through the
    `reduced`/`unreduced` PartitionSpec labels applied in
    `maxtext.utils.gradient_accumulation`, and casts the parameters to bf16 before
    the accumulation scan so the all-gather happens once in low precision.
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
    for decoder_block, model_args in _MISTRAL_MODELS.items():
      with self.subTest(decoder_block=decoder_block):
        args = model_args + zero1_ga
        baseline = self._mistral_losses(
            f"{decoder_block}_ga",
            args + ["shard_mode=auto", "shard_optimizer_over_data=False"],
        )
        sharded = self._mistral_losses(
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
  def test_tpu_qwen2_explicit_sharding_matches_auto(self):
    """Explicit sharding only changes how layouts are expressed, so the losses must not move.

    Tensor parallelism is what stresses the Qwen2 decoder: it shards the MLP intermediate
    and the attention heads, which are the two activations the layer now pins itself.
    """
    args = ["ici_fsdp_parallelism=1", "ici_tensor_parallelism=-1"]
    auto_losses = self._losses("qwen2_auto", self._qwen2_overrides, args + ["shard_mode=auto"])
    explicit_losses = self._losses("qwen2_explicit", self._qwen2_overrides, args + ["shard_mode=explicit"])
    print(f"[qwen2] auto losses: {auto_losses}", flush=True)
    print(f"[qwen2] explicit losses: {explicit_losses}", flush=True)
    self.assertTrue(auto_losses, "auto run produced no metrics")
    # Same as the mistral case: pinning the MLP intermediate reassociates the backward
    # reduction over the tensor axis, so the last step drifts by a few ULPs. Under FSDP
    # instead of tensor parallelism the two runs are bit-for-bit.
    np.testing.assert_allclose(explicit_losses, auto_losses, rtol=1e-5, atol=0.0)

  @pytest.mark.integration_test
  @pytest.mark.tpu_only
  def test_tpu_kimi_explicit_sharding_matches_auto(self):
    """Kimi-K2 under explicit sharding, exercising the deepseek block's MLA and MoE paths.

    Expert parallelism shards the MoE dispatch, which is where the deepseek layer's pinned
    output shardings have to line up with what RoutedMoE returns.
    """
    args = ["ici_fsdp_parallelism=1", "ici_expert_parallelism=-1"]
    auto_losses = self._losses("kimi_auto", self._kimi_overrides, args + ["shard_mode=auto"])
    explicit_losses = self._losses("kimi_explicit", self._kimi_overrides, args + ["shard_mode=explicit"])
    print(f"[kimi-k2] auto losses: {auto_losses}", flush=True)
    print(f"[kimi-k2] explicit losses: {explicit_losses}", flush=True)
    self.assertTrue(auto_losses, "auto run produced no metrics")
    # Expert parallelism does not reassociate any reduction, so the two runs are bit-for-bit.
    np.testing.assert_allclose(explicit_losses, auto_losses, rtol=1e-6, atol=0.0)

  @pytest.mark.integration_test
  @pytest.mark.tpu_only
  # TODO(b/517509898): Skip ZeRo-1 compiler Segfault on TPU7x SparseCore platforms
  @pytest.mark.skip_on_tpu7x
  def test_tpu_qwen2_kimi_zero1_gradient_accumulation(self):
    """ZeRO-1 only shards the optimizer state, so it must not change the loss trajectory.

    Under explicit sharding this routes the accumulated gradients through the
    `reduced`/`unreduced` PartitionSpec labels applied in
    `maxtext.utils.gradient_accumulation`, and casts the parameters to bf16 before the
    accumulation scan so the all-gather happens once in low precision.
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
    for case, model_overrides in (("qwen2", self._qwen2_overrides), ("kimi-k2", self._kimi_overrides)):
      with self.subTest(case=case):
        baseline = self._losses(
            f"{case}_ga",
            model_overrides,
            zero1_ga + ["shard_mode=auto", "shard_optimizer_over_data=False"],
        )
        sharded = self._losses(
            f"{case}_ga_zero1",
            model_overrides,
            zero1_ga + ["shard_mode=explicit", "shard_optimizer_over_data=True"],
        )
        print(f"[{case}] auto + GA losses: {baseline}", flush=True)
        print(f"[{case}] explicit + ZeRO-1 + GA losses: {sharded}", flush=True)
        self.assertTrue(baseline, "baseline run produced no metrics")
        # ZeRO-1 reassociates the gradient all-reduce, so allow a little float slack.
        np.testing.assert_allclose(sharded, baseline, rtol=1e-4, atol=0.0)

  @pytest.mark.integration_test
  @pytest.mark.tpu_only
  def test_tpu_qwen3_hybrid_explicit_sharding_matches_auto(self):
    """Explicit sharding only changes how layouts are expressed, so the losses must not move.

    The two decoders share every sublayer, so each is paired with the
    parallelism that
    stresses a different half of it: expert parallelism shards the MoE dispatch,
    and
    tensor parallelism shards the gated-delta-net head axis, which is the one
    the layer
    has to carry by hand across its reshapes, its head repeat and the
    `jax.shard_map`
    boundary. Qwen3-Next takes the latter because it also nests two
    `jax.lax.scan`s,
    whose carry layout has to stay invariant across iterations.
    """
    parallelism = {
        "qwen3_5": ["ici_fsdp_parallelism=1", "ici_expert_parallelism=-1"],
        # The gated-delta-net weights stay replicated under tensor parallelism, which is
        # a large fraction of a model this small, so relax the unsharded-parameter check.
        "qwen3_next": [
            "ici_fsdp_parallelism=1",
            "ici_tensor_parallelism=-1",
            "sharding_tolerance=0.5",
        ],
    }
    for decoder_block, model_overrides in self._QWEN3_HYBRID_MODELS.items():
      with self.subTest(decoder_block=decoder_block):
        args = parallelism[decoder_block]
        auto_losses = self._losses(
            f"{decoder_block}_auto", model_overrides, args + ["shard_mode=auto"]
        )
        explicit_losses = self._losses(
            f"{decoder_block}_explicit",
            model_overrides,
            args + ["shard_mode=explicit"],
        )
        print(f"[{decoder_block}] auto losses: {auto_losses}", flush=True)
        print(
            f"[{decoder_block}] explicit losses: {explicit_losses}", flush=True
        )
        self.assertTrue(auto_losses, "auto run produced no metrics")
        # `activation_batch` carries the expert axis, so pinning it reassociates the
        # backward reductions: the forward pass is bit-for-bit and the drift only appears
        # once gradients flow. Over 20 steps it stays below 4e-5 relative and changes
        # sign, i.e. it is float noise rather than the two runs pulling apart.
        np.testing.assert_allclose(
            explicit_losses, auto_losses, rtol=1e-4, atol=0.0
        )

  @pytest.mark.integration_test
  @pytest.mark.tpu_only
  # TODO(b/517509898): Skip ZeRo-1 compiler Segfault on TPU7x SparseCore platforms
  @pytest.mark.skip_on_tpu7x
  def test_tpu_qwen3_hybrid_zero1_gradient_accumulation(self):
    """ZeRO-1 only shards the optimizer state, so it must not change the loss trajectory.

    Under explicit sharding this routes the accumulated gradients through the
    `reduced`/`unreduced` PartitionSpec labels applied in
    `maxtext.utils.gradient_accumulation`, and casts the parameters to bf16
    before the
    accumulation scan so the all-gather happens once in low precision rather
    than once
    per microbatch.
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
    for decoder_block, model_overrides in self._QWEN3_HYBRID_MODELS.items():
      with self.subTest(decoder_block=decoder_block):
        baseline = self._losses(
            f"{decoder_block}_ga",
            model_overrides,
            zero1_ga + ["shard_mode=auto", "shard_optimizer_over_data=False"],
        )
        sharded = self._losses(
            f"{decoder_block}_ga_zero1",
            model_overrides,
            zero1_ga
            + ["shard_mode=explicit", "shard_optimizer_over_data=True"],
        )
        print(f"[{decoder_block}] auto + GA losses: {baseline}", flush=True)
        print(
            f"[{decoder_block}] explicit + ZeRO-1 + GA losses: {sharded}",
            flush=True,
        )
        self.assertTrue(baseline, "baseline run produced no metrics")
        # ZeRO-1 reassociates the gradient all-reduce, so allow a little float slack.
        np.testing.assert_allclose(sharded, baseline, rtol=1e-4, atol=0.0)

  @pytest.mark.integration_test
  @pytest.mark.tpu_only
  # TODO(b/517509898): Skip ZeRo-1 compiler Segfault on TPU7x SparseCore platforms
  @pytest.mark.skip_on_tpu7x
  def test_tpu_gemma_zero1_gradient_accumulation_explicit(self):
    """Gemma under ZeRO-1 + gradient accumulation + explicit sharding.

    Explicit sharding type-checks the sharding of every operation rather than letting
    GSPMD infer one, so a missing or wrong `out_sharding` in a Gemma layer fails the
    step outright instead of silently costing a collective. ZeRO-1 and gradient
    accumulation are in the mix because they layer the optimizer-moment and scan-carry
    shardings on top, which is where annotations that look fine in a plain forward pass
    tend to come apart.
    """
    # Gemma 3 reads its local/global attention pattern and rope scaling off the named
    # model config, so it cannot run under the placeholder "default" model name.
    families = [
        ("gemma", "gemma", "tokenizer.gemma", []),
        ("gemma2", "gemma2", "tokenizer.gemma", []),
        ("gemma3", "gemma3", "tokenizer.gemma3", ["model_name=gemma3-4b", "override_model_config=True"]),
        # Host offload keeps the parameters in pinned_host and moves the gradients back to
        # device memory before the optimizer update, which is a second place the layer
        # annotations have to line up with what the trainer asks for.
        ("gemma-host-offload", "gemma", "tokenizer.gemma", ["parameter_memory_host_offload=True", "param_scan_axis=0"]),
    ]
    for case, decoder_block, tokenizer, extra_args in families:
      with self.subTest(case=case):
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
            # Data-parallel only, matching the llama2 test above: ZeRO-1 needs a "data"
            # axis to shard the moments over, and MaxTextConfig rejects combining it with
            # FSDP (the gradients and the moments would end up in different layouts).
            "ici_data_parallelism=-1",
            "dcn_data_parallelism=1",
            "ici_fsdp_parallelism=1",
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
        "ici_expert_parallelism=4",
        "ici_fsdp_parallelism=-1",
        "use_ring_of_experts=True",
        "model_name=deepseek3-test",
        "eval_per_device_batch_size=0.25",
        "eval_interval=3",
        "eval_steps=1",
        "custom_mesh_and_rule_for_eval=ep-as-cp",
        "use_tokamax_splash=true",
        "sa_block_q=1024",
        "sa_block_kv=1024",
        "sa_block_kv_compute=1024",
        "sa_block_q_dkv=1024",
        "sa_block_kv_dkv=1024",
        "sa_block_kv_dkv_compute=1024",
        "sa_block_q_dq=1024",
        "sa_block_kv_dq=1024",
        "override_model_config=true",
        "base_num_decoder_layers=7",
        rf"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizers', 'tokenizer.llama2')}",
    ]
    train_main(frac_eval)


if __name__ == "__main__":
  absltest.main()
