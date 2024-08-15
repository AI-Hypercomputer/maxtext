"""
Copyright 2024 Google LLC
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
     https://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# pylint: disable=missing-module-docstring, missing-function-docstring
import numpy as np
import json
import unittest
import pytest
import string
import random
from train import main as train_main
import sys
import subprocess

def generate_random_string(length=10):
    characters = string.ascii_letters  # Include letters, digits, and punctuation
    return ''.join(random.choice(characters) for _ in range(length))

class GradientAccumulationTest(unittest.TestCase):  

  
  @pytest.mark.tpu
  def test_grad_accumulate_same_loss(self):
    random_suffix = generate_random_string()
    run_accumulate_metrics_file = f"/tmp/runner_grad_accumulate_{random_suffix}.txt"
    run_regular_metrics_file = f"/tmp/runner_regular_{random_suffix}.txt"
    
    cmd = ["bash", "tests/gradient_accumulation_script.sh", run_accumulate_metrics_file, run_regular_metrics_file]
    try:
      result = subprocess.run(
            cmd,                # Command to run the script
            check=True,         # Raise an exception if the script fails
            stdout=sys.stdout,  # Stream to stdout
            stderr=sys.stdout,  # Stream to stdout
            text=True           # Decode output and error as text
      )
      return result
    except subprocess.CalledProcessError as e:
      print(f"Error running script: {e.returncode}")
      print(f"Output: {e.stdout}")
      print(f"Error: {e.stderr}")

    # Assert losses roughly equal
    with open(run_accumulate_metrics_file, 'r', encoding='utf8') as accum_run,\
          open(run_regular_metrics_file, 'r', encoding='utf8') as regular_run:  
        accum_run_loss = json.loads(accum_run.readlines()[-1])["learning/loss"]
        regular_run_loss = json.loads(regular_run.readlines()[-1])["learning/loss"]
        print(f"[Gradient Accumulation Test] Loss with gradient accumulation: {accum_run_loss}", flush=True)
        print(f"[Gradient Accumulation Test] Loss without gradient accumulation: {regular_run_loss}", flush=True)
        # Not identical due to an epsilon addition in loss denominator.
        np.testing.assert_allclose(accum_run_loss, regular_run_loss, rtol=0.01)

    # Assert grad norms roughly equal
    with open(run_accumulate_metrics_file, 'r', encoding='utf8') as accum_run,\
          open(run_regular_metrics_file, 'r', encoding='utf8') as regular_run:  
        accum_run_grad_norm= json.loads(accum_run.readlines()[-1])["learning/raw_grad_norm"]
        regular_run_grad_norm = json.loads(regular_run.readlines()[-1])["learning/raw_grad_norm"]
        print(f"[Gradient Accumulation Test] Grad norm with gradient accumulation: {accum_run_grad_norm}", flush=True)
        print(f"[Gradient Accumulation Test] Grad norm without gradient accumulation: {regular_run_grad_norm}", flush=True)
        # Not identical due to an epsilon addition in loss denominator.
        np.testing.assert_allclose(accum_run_grad_norm, regular_run_grad_norm, rtol=0.01)

    # Assert per device tflops are the same (10x smaller microbatch size, but 10x more microbatches)
    with open(run_accumulate_metrics_file, 'r', encoding='utf8') as accum_run,\
          open(run_regular_metrics_file, 'r', encoding='utf8') as regular_run:    
        accum_device_tflops = json.loads(accum_run.readlines()[-1])["perf/per_device_tflops"]
        regular_device_tflops = json.loads(regular_run.readlines()[-1])["perf/per_device_tflops"]
        print(f"[Gradient Accumulation Test] per_device_tflops with gradient accumulation: {accum_device_tflops}", flush=True)
        print(f"[Gradient Accumulation Test] per_device_tflops without gradient accumulation: {regular_device_tflops}", flush=True)
        np.testing.assert_equal(accum_device_tflops, regular_device_tflops)

    