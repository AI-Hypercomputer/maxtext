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
import gc
import json
import statistics
import unittest
from unittest import mock
import pytest
import string
import random
import os
import os.path

import jax

from flax.linen import partitioning as nn_partitioning

from maxtext.common.gcloud_stub import is_decoupled
from maxtext.configs import pyconfig
from maxtext.trainers.pre_train import train, train_compile
from maxtext.trainers.pre_train.train import main as train_main
from maxtext.utils import gradient_accumulation, maxtext_utils, sharding
from maxtext.utils.globals import MAXTEXT_ASSETS_ROOT
from maxtext.trainers.post_train.sft.train_sft_native import main as sft_main

from tests.utils.hlo_test_utils import cross_replica_all_reduce_sizes, split_by_entry_loop
from tests.utils.test_helpers import get_test_config_path, get_test_dataset_path, get_test_base_output_directory

# Cross-replica reductions that legitimately run once per microbatch — the summed loss and
# the token count — are scalars. The smallest DeepSeek parameter gradient is thousands of
# elements, so this bound separates "a per-microbatch scalar" from "a parameter gradient".
_MAX_PER_MICROBATCH_REDUCTION_ELEMENTS = 8

# A ~1.3B-parameter DeepSeek MoE, the smallest size at which the per-microbatch expert
# gradient reductions cost enough wall clock to measure over run-to-run noise. Sized to
# fit a single host of four large-HBM chips at roughly a third of the memory limit, so
# the peak comparison is not clamped by rematerialization kicking in on one variant only.
_SCALE_OVERRIDES = [
    "model_name=deepseek3-tiny",
    "override_model_config=True",
    "base_emb_dim=1024",
    "base_mlp_dim=2048",
    "base_moe_mlp_dim=1024",
    "base_num_decoder_layers=24",
    "first_num_dense_layers=1",
    "num_experts=16",
    "num_experts_per_tok=2",
    "max_target_length=1024",
    "per_device_batch_size=2",
    "steps=12",
    "enable_checkpointing=False",
    "enable_goodput_recording=False",
    "dataset_type=synthetic",
    "ici_data_parallelism=-1",
    "ici_fsdp_parallelism=1",
    "shard_optimizer_over_data=true",
    "shard_mode=explicit",
    "gradient_accumulation_steps=4",
]

# Compilation, the first donated-buffer step and the input pipeline reaching steady state
# all land in the opening steps; from the fifth on, step times repeat to a few tenths of
# a percent on a dedicated host.
_SCALE_WARMUP_STEPS = 4
# Measured on a 2x2 v6e host: 0.316 s per step tagged against 0.383 s untagged, and peak
# HBM within 0.01%. The thresholds keep a wide margin on both so the test fails on the
# tag being lost rather than on jitter.
_SCALE_MIN_SPEEDUP = 1.05
_SCALE_MAX_MEMORY_RATIO = 1.02
# Below this the config has been shrunk past the point where the assertions mean anything.
_SCALE_MIN_PEAK_HBM_BYTES = 4 * 2**30
_SCALE_MIN_DEVICES = 4
_SCALE_MIN_HBM_BYTES = 24 * 2**30


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

  def _train_scale_variant(self, variant):
    """Trains the scale config once; returns its median steady-state step time and peak HBM."""
    metrics_file = os.path.join(tempfile.gettempdir(), f"scale_{variant}_{generate_random_string()}.jsonl")
    train_main(
        [
            None,
            get_test_config_path(),
            f"base_output_directory={self.base_output_directory}",
            f"run_name=deepseek_zero1_ga_scale_{variant}",
            f"metrics_file={metrics_file}",
        ]
        + _SCALE_OVERRIDES
    )
    # Drop the train state before reading the peak, so the next variant starts from the
    # same live-memory baseline rather than on top of this one's parameters.
    gc.collect()

    step_times = []
    with open(metrics_file, "rt", encoding="utf8") as metrics:
      for line in metrics:
        record = json.loads(line)
        if "perf/step_time_seconds" in record:
          step_times.append(record["perf/step_time_seconds"])
    steady = step_times[_SCALE_WARMUP_STEPS:]
    self.assertTrue(steady, f"{variant} run logged only {len(step_times)} steps")

    peak_hbm_bytes = max(device.memory_stats()["peak_bytes_in_use"] for device in jax.local_devices())
    step_time = statistics.median(steady)
    print(
        f"[Gradient Accumulation Scale Test] {variant}: {step_time:.6f} s/step over {len(steady)} steps, "
        f"peak HBM so far {peak_hbm_bytes / 2**30:.4f} GiB",
        flush=True,
    )
    return step_time, peak_hbm_bytes

  @pytest.mark.integration_test
  @pytest.mark.tpu_only
  def test_deepseek_zero1_ga_scale_memory_and_step_time(self):
    """A ~1.3B DeepSeek MoE trains faster, and no heavier, once the accumulator is tagged.

    `test_deepseek_zero1_reduces_gradients_once_per_step` reads the collective's position
    out of the HLO, which says nothing about what the deferred reduction is worth. This
    one trains the model twice on real chips — once as shipped, once with the tagging
    suppressed — and compares step time and peak HBM, so a change that keeps the tag but
    reduces gradients per microbatch somewhere downstream still shows up.
    """
    devices = jax.local_devices()
    if len(devices) < _SCALE_MIN_DEVICES:
      self.skipTest(f"needs at least {_SCALE_MIN_DEVICES} local devices, found {len(devices)}")
    hbm_bytes = devices[0].memory_stats()["bytes_limit"]
    if hbm_bytes < _SCALE_MIN_HBM_BYTES:
      self.skipTest(f"needs at least {_SCALE_MIN_HBM_BYTES / 2**30:.0f} GiB per device, found {hbm_bytes / 2**30:.2f}")

    # `peak_bytes_in_use` is a high-water mark the allocator never resets, so order the
    # runs to make that work in the assertion's favour: the untagged variant goes first,
    # its reading is its own peak, and the reading after the tagged run is the larger of
    # the two. "The second reading is no higher than the first" is then exactly "tagging
    # did not raise the peak", with no way for a regression to hide behind the mark.
    # Suppressing the tag reproduces the behaviour before the accumulator carried one.
    with mock.patch.object(gradient_accumulation, "data_is_only_batch_axis", return_value=False):
      untagged_step_time, peak_after_untagged = self._train_scale_variant("untagged")
    tagged_step_time, peak_after_tagged = self._train_scale_variant("tagged")

    # Guards against the config being shrunk to where the reductions stop mattering.
    self.assertGreater(
        peak_after_untagged,
        _SCALE_MIN_PEAK_HBM_BYTES,
        "the scale config no longer reaches the memory footprint these thresholds were calibrated at",
    )
    memory_ratio = peak_after_tagged / peak_after_untagged
    self.assertLess(
        memory_ratio,
        _SCALE_MAX_MEMORY_RATIO,
        f"deferring the reduction cost {memory_ratio:.4f}x peak HBM; accumulating in the parameters' own layout "
        "should not need more live memory than reducing every microbatch",
    )
    speedup = untagged_step_time / tagged_step_time
    self.assertGreater(
        speedup,
        _SCALE_MIN_SPEEDUP,
        f"reducing gradients once per step was only {speedup:.4f}x faster than once per microbatch; the accumulator's "
        "`reduced` tag is being dropped somewhere between gradient accumulation and the layers",
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
