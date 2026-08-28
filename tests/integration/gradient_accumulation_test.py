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

"""Integration tests for gradient accumulation."""

import tempfile

import numpy as np
import json
import unittest
import pytest
import string
import random
import os
import os.path

from flax.linen import partitioning as nn_partitioning

from maxtext.common.gcloud_stub import is_decoupled
from maxtext.configs import pyconfig
from maxtext.trainers.pre_train import train, train_compile
from maxtext.trainers.pre_train.train import main as train_main
from maxtext.utils import maxtext_utils, sharding
from maxtext.utils.globals import MAXTEXT_ASSETS_ROOT
from maxtext.trainers.post_train.sft.train_sft_native import main as sft_main

from tests.utils.hlo_test_utils import cross_replica_all_reduce_sizes, split_by_entry_loop
from tests.utils.test_helpers import get_test_config_path, get_test_dataset_path, get_test_base_output_directory

# Cross-replica reductions that legitimately run once per microbatch — the summed loss and
# the token count — are scalars. The smallest DeepSeek parameter gradient is thousands of
# elements, so this bound separates "a per-microbatch scalar" from "a parameter gradient".
_MAX_PER_MICROBATCH_REDUCTION_ELEMENTS = 8


def generate_random_string(length=10):
  characters = string.ascii_letters  # Include letters, digits, and punctuation
  return "".join(random.choice(characters) for _ in range(length))


def compile_train_step(overrides):
  """AOT-compiles `train.train_step` for a config and returns the compiled executable.

  Mirrors `train_compile.main`, which builds the same executable but does not hand it
  back. Compiling ahead of time against `compile_topology` means the mesh can be larger
  than the host's device count, which is what makes a data-parallel training step
  inspectable from a single-device test runner.
  """
  config = pyconfig.initialize([None, get_test_config_path()] + overrides)
  train_compile.validate_config(config)
  topology_mesh = train_compile.get_topology_mesh(config)
  shaped_args, shaped_kwargs, state_mesh_shardings, _, model = train_compile.get_shaped_inputs(topology_mesh, config)
  # ZeRO-1 moves the params onto the optimizer's layout; gradient accumulation still sees
  # the pre-ZeRO-1 layout through params_shardings.
  params_shardings, state_mesh_shardings = sharding.maybe_update_params_sharding_with_opt(config, state_mesh_shardings)
  input_state_mesh_shardings = sharding.build_zero1_input_state_mesh_shardings(
      config, state_mesh_shardings, params_shardings
  )
  data_sharding = sharding.get_input_data_sharding(config, topology_mesh)
  func, in_shardings, out_shardings, static_argnums, donate_argnums = maxtext_utils.get_functional_train_with_signature(
      train.train_step, data_sharding, input_state_mesh_shardings, model, config, params_shardings
  )
  return train_compile.jit_and_compile(
      func,
      shaped_args,
      shaped_kwargs,
      topology_mesh,
      in_shardings,
      out_shardings,
      static_argnums,
      donate_argnums,
      config,
      nn_partitioning.axis_rules(config.logical_axis_rules),
  )


class GradientAccumulationTest(unittest.TestCase):

  def setUp(self):
    """Set up test fixtures before each test method."""
    decoupled = is_decoupled()
    self.dataset_path = get_test_dataset_path()
    self.base_output_directory = (
        os.environ.get("LOCAL_BASE_OUTPUT", get_test_base_output_directory())
        if decoupled
        else get_test_base_output_directory()
    )

  @pytest.mark.integration_test
  @pytest.mark.tpu_only
  def test_grad_accumulate_same_loss(self):
    random_suffix = generate_random_string()
    temp_dir = tempfile.gettempdir()
    run_accumulate_metrics_file = os.path.join(temp_dir, f"runner_grad_accumulate_{random_suffix}.txt")
    run_regular_metrics_file = os.path.join(temp_dir, f"runner_regular_{random_suffix}.txt")
    shared_maxtext_args = [
        None,
        get_test_config_path(),
        f"base_output_directory={self.base_output_directory}",
        f"dataset_path={self.dataset_path}",
        "dataset_type=synthetic",
        "gradient_clipping_threshold=0",  # Ensures we are testing raw scales of gradients (clipping off)
        "enable_checkpointing=False",
        "enable_goodput_recording=False",
        "decoder_block=simple",
        "base_emb_dim=256",
        "base_num_decoder_layers=4",
        rf"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizers', 'tokenizer.llama2')}",
        "steps=2",
    ]
    # Run with gradient accumulation with accumulate_steps=10, per_device_batch=1 --> simulating per_device_batch=10
    train_main(
        shared_maxtext_args
        + [
            "run_name=runner_grad_accumulate",
            f"metrics_file={run_accumulate_metrics_file}",
            "per_device_batch_size=1",
            "gradient_accumulation_steps=10",
        ]
    )

    # Run without gradient accumulation with per_device_batch=10
    train_main(
        shared_maxtext_args
        + [
            "run_name=runner_grad_accumulate_regular",
            f"metrics_file={run_regular_metrics_file}",
            "per_device_batch_size=10",
            "gradient_accumulation_steps=1",
        ]
    )

    # Assert losses roughly equal
    with (
        open(run_accumulate_metrics_file, "rt", encoding="utf8") as accum_run,
        open(run_regular_metrics_file, "rt", encoding="utf8") as regular_run,
    ):
      accum_run_loss = json.loads(accum_run.readlines()[-1])["learning/loss"]
      regular_run_loss = json.loads(regular_run.readlines()[-1])["learning/loss"]
      print(
          f"[Gradient Accumulation Test] Loss with gradient accumulation: {accum_run_loss}",
          flush=True,
      )
      print(
          f"[Gradient Accumulation Test] Loss without gradient accumulation: {regular_run_loss}",
          flush=True,
      )
      # Not identical due to an epsilon addition in loss denominator.
      np.testing.assert_allclose(accum_run_loss, regular_run_loss, rtol=0.01)

    # Assert grad norms roughly equal
    with (
        open(run_accumulate_metrics_file, "rt", encoding="utf8") as accum_run,
        open(run_regular_metrics_file, "rt", encoding="utf8") as regular_run,
    ):
      accum_run_grad_norm = json.loads(accum_run.readlines()[-1])["learning/raw_grad_norm"]
      regular_run_grad_norm = json.loads(regular_run.readlines()[-1])["learning/raw_grad_norm"]
      print(
          f"[Gradient Accumulation Test] Grad norm with gradient accumulation: {accum_run_grad_norm}",
          flush=True,
      )
      print(
          f"[Gradient Accumulation Test] Grad norm without gradient accumulation: {regular_run_grad_norm}",
          flush=True,
      )
      # Not identical due to an epsilon addition in loss denominator.
      np.testing.assert_allclose(accum_run_grad_norm, regular_run_grad_norm, rtol=0.01)

    # Assert per device tflops are the same (10x smaller microbatch size, but 10x more microbatches)
    with (
        open(run_accumulate_metrics_file, "rt", encoding="utf8") as accum_run,
        open(run_regular_metrics_file, "rt", encoding="utf8") as regular_run,
    ):
      accum_device_tflops = json.loads(accum_run.readlines()[-1])["perf/per_device_tflops"]
      regular_device_tflops = json.loads(regular_run.readlines()[-1])["perf/per_device_tflops"]
      print(
          f"[Gradient Accumulation Test] per_device_tflops with gradient accumulation: {accum_device_tflops}",
          flush=True,
      )
      print(
          f"[Gradient Accumulation Test] per_device_tflops without gradient accumulation: {regular_device_tflops}",
          flush=True,
      )
      np.testing.assert_equal(accum_device_tflops, regular_device_tflops)

  @pytest.mark.integration_test
  @pytest.mark.tpu_only
  def test_deepseek_zero1_reduces_gradients_once_per_step(self):
    """DeepSeek + ZeRO-1 + gradient accumulation must all-reduce gradients once per step.

    Under explicit sharding the accumulator carries an `unreduced` PartitionSpec tag over
    the data axis, so each microbatch's gradient is summed locally and the cross-replica
    reduction happens once, when the accumulated gradient is resharded back after the
    scan. Losing that tag is silent — the gradients stay numerically identical — and only
    shows up as the reduction being emitted inside the accumulation loop, once per
    microbatch. So assert on where the collective sits rather than on the loss.
    """
    compiled = compile_train_step(
        [
            "run_name=deepseek_zero1_ga_all_reduce",
            f"base_output_directory={self.base_output_directory}",
            "enable_checkpointing=False",
            "enable_goodput_recording=False",
            "dataset_type=synthetic",
            "model_name=deepseek3-tiny",
            "compile_topology=v5e-8",
            "compile_topology_num_slices=1",
            "ici_data_parallelism=-1",
            "ici_fsdp_parallelism=1",
            "shard_optimizer_over_data=true",
            "shard_mode=explicit",
            "gradient_accumulation_steps=4",
            "per_device_batch_size=1",
        ]
    )
    inside, outside = split_by_entry_loop(compiled.as_text(), "all-reduce")
    inside_sizes = cross_replica_all_reduce_sizes(inside)
    # Guards the assertion below against passing vacuously: if SPMD partitioning stopped
    # spelling its replica groups over mesh axes, `inside_sizes` would be empty for the
    # wrong reason.
    self.assertTrue(cross_replica_all_reduce_sizes(outside), "no cross-replica all-reduce outside the loop")
    per_microbatch = [size for size in inside_sizes if size > _MAX_PER_MICROBATCH_REDUCTION_ELEMENTS]
    self.assertEqual(
        per_microbatch,
        [],
        f"gradient all-reduce is still inside the accumulation loop: {per_microbatch} elements per buffer",
    )

  @pytest.mark.integration_test
  @pytest.mark.tpu_only
  def test_sft_grad_accumulate_same_loss(self):
    sft_main(
        [
            None,
            get_test_config_path(),
            f"base_output_directory={self.base_output_directory}",
            f"dataset_path={self.dataset_path}",
            "dataset_type=synthetic",
            "max_target_length=128",
            "gradient_clipping_threshold=0",  # Ensures we are testing raw scales of gradients (clipping off).
            "enable_checkpointing=False",
            "enable_goodput_recording=False",
            "base_emb_dim=128",
            "base_num_decoder_layers=1",
            rf"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizers', 'tokenizer.llama2')}",
            "steps=3",
            "gradient_accumulation_steps=2",
            "use_sft=True",
        ]
    )
