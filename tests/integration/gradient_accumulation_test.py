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
from pathlib import Path

from MaxText.train import main as train_main
from MaxText.sft_trainer import main as sft_main
from MaxText.globals import MAXTEXT_ASSETS_ROOT
from maxtext.common.gcloud_stub import is_decoupled
from maxtext.trainers.post_train.sft.train_sft import main as tunix_sft_train

from tests.utils.test_helpers import get_test_config_path, get_test_dataset_path, get_test_base_output_directory


def generate_random_string(length=10):
  characters = string.ascii_letters  # Include letters, digits, and punctuation
  return "".join(random.choice(characters) for _ in range(length))


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
        "gradient_clipping_threshold=0",  # Ensures we are testing raw scales of gradients (clipping off)
        "enable_checkpointing=False",
        "enable_goodput_recording=False",
        "base_emb_dim=256",
        "base_num_decoder_layers=4",
        rf"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizers', 'tokenizer.llama2')}",
        "steps=20",
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
  def test_sft_grad_accumulate_same_loss(self):
    sft_main(
        [
            None,
            get_test_config_path(),
            f"base_output_directory={self.base_output_directory}",
            f"dataset_path={self.dataset_path}",
            "dataset_path=gs://maxtext-dataset",
            "gradient_clipping_threshold=0",  # Ensures we are testing raw scales of gradients (clipping off).
            "enable_checkpointing=False",
            "enable_goodput_recording=False",
            "base_emb_dim=256",
            "base_num_decoder_layers=4",
            rf"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizers', 'tokenizer.llama2')}",
            "steps=3",
            "gradient_accumulation_steps=2",
            "use_sft=True",
        ]
    )

  @pytest.mark.integration_test
  @pytest.mark.tpu_only
  def test_tunix_sft_grad_accumulate_same_loss(self):
    random_suffix = generate_random_string()
    temp_dir = tempfile.gettempdir()
    run_accumulate_metrics_file = os.path.join(temp_dir, f"runner_sft_grad_accumulate_{random_suffix}.txt")
    run_regular_metrics_file = os.path.join(temp_dir, f"runner_sft_regular_{random_suffix}.txt")

    shared_maxtext_args = [
        None,
        get_test_config_path(),
        f"base_output_directory={self.base_output_directory}",
        f"dataset_path={self.dataset_path}",
        "dataset_path=gs://maxtext-dataset",
        "gradient_clipping_threshold=0",  # Ensures we are testing raw scales of gradients (clipping off).
        "enable_checkpointing=False",
        "enable_goodput_recording=False",
        "base_emb_dim=256",
        "base_num_decoder_layers=4",
        rf"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizer.llama2')}",
        "steps=8",
        "use_sft=True",
        "use_tunix=True",
    ]

    # Run SFT with gradient accumulation (simulating Batch=2 via 1 * 4)
    tunix_sft_train(
        shared_maxtext_args
        + [
            f"run_name=runner_sft_grad_accumulate_{random_suffix}",
            f"metrics_file={run_accumulate_metrics_file}",
            "per_device_batch_size=1",
            "gradient_accumulation_steps=4",
        ]
    )

    # Run SFT regular (Batch=4)
    tunix_sft_train(
        shared_maxtext_args
        + [
            f"run_name=runner_sft_grad_accumulate_regular_{random_suffix}",
            f"metrics_file={run_regular_metrics_file}",
            "per_device_batch_size=4",
            "gradient_accumulation_steps=1",
        ]
    )

    def get_last_metric_from_file(filepath):
      return json.loads(Path(filepath).read_text(encoding="utf8").splitlines()[-1])

    accum_metrics = get_last_metric_from_file(run_accumulate_metrics_file)
    regular_metrics = get_last_metric_from_file(run_regular_metrics_file)

    # Assert losses roughly equal
    accum_loss = accum_metrics["learning/loss"]
    regular_loss = regular_metrics["learning/loss"]

    print(f"[SFT Grad Accum Test] Loss (Accum): {accum_loss}", flush=True)
    print(f"[SFT Grad Accum Test] Loss (Regular): {regular_loss}", flush=True)

    np.testing.assert_allclose(accum_loss, regular_loss, rtol=0.01)

    # Assert per device tflops are the same
    accum_tflops = accum_metrics["perf/per_device_tflops"]
    regular_tflops = regular_metrics["perf/per_device_tflops"]

    print(f"[SFT Grad Accum Test] TFLOPS (Accum): {accum_tflops}", flush=True)
    print(f"[SFT Grad Accum Test] TFLOPS (Regular): {regular_tflops}", flush=True)

    np.testing.assert_equal(accum_tflops, regular_tflops)
