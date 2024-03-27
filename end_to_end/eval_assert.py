"""
 Copyright 2023 Google LLC

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

# pylint: skip-file
"""Reads and asserts over target values"""
from absl import app
from typing import Sequence
from math import isclose
from google.cloud import storage
import json


def compute_avg_metric(metrics_file, target, start_line=10):
  """ Reads and computes average of target value
  If start_line is negative then uses the last lines, e.g. start from end + 1 - |start_line|"""
  

  avg = 0
  i = 0
  with open(metrics_file, 'r', encoding='utf8') as file:
    lines = file.readlines()
    if start_line < 0:
      start_line = len(lines) + start_line
    for line in lines:
      # skip the first start_line lines for burn in
      if i >= start_line:
        vals = json.loads(line)
        avg += vals[target]
      i+=1
    avg /= (i-start_line)

  return avg


def assert_metric_average(metrics_file, threshold, target):
  avg_value = compute_avg_metric(metrics_file, target)
  # Checks for acceptable performance by asserting that the average metric (e.g. TFLOPs)
  # is greater than the threshold.
  print(f'avg value of target {target} is {avg_value}')
  assert avg_value >= float(threshold)
  print('assert metric average passed.')

def test_final_loss(metrics_file, target_loss):
  target_loss = float(target_loss)
  with open(metrics_file, 'r', encoding='utf8') as metrics:
    use_last_n_data = 10
    avg_final_loss = compute_avg_metric(metrics_file, 'learning/loss', start_line= -1 * use_last_n_data)
    print(f"Mean of last {use_last_n_data} losses is {avg_final_loss}")
    print(f"Target loss is {target_loss}")
    assert avg_final_loss < target_loss
    print('Final loss test passed.')

def test_checkpointing(metrics_file, target, dataset_type):
  """Asserts over loss values from loaded checkpoint"""
  metrics_file_saved = 'saved_' + metrics_file
  metrics_file_restored = 'restored_' + metrics_file

  with open(metrics_file_saved, 'r', encoding='utf8') as saved,\
    open(metrics_file_restored, 'r', encoding='utf8') as restored:
    saved_loss = json.loads(saved.readlines()[-1])[target]
    restored_loss = json.loads(restored.readlines()[0])[target]
    # Checks that checkpoint restore was successful by comparing loss of last
    # step in saved checkpoint to loss of first step in restored checkpoint
    print("saved loss: ", saved_loss)
    print("restored loss: ", restored_loss)
    if dataset_type=='c4':
      assert isclose(saved_loss, restored_loss, rel_tol=0.1)
    elif dataset_type=='c4-array_record':
      assert saved_loss==restored_loss
    else:
      raise ValueError(f"Unknown dataset_type {dataset_type}. dataset_type must be c4, c4-array_record or synthetic")
    print('checkpointing test passed.')

def test_determinism(metrics_file, target):
  """Asserts over loss values from two runs"""
  run_1 = 'run_1_' + metrics_file
  run_2 = 'run_2_' + metrics_file

  with open(run_1, 'r', encoding='utf8') as run_1_file,\
    open(run_2, 'r', encoding='utf8') as run_2_file:
    run_1_loss = json.loads(run_1_file.readlines()[-1])[target]
    run_2_loss = json.loads(run_2_file.readlines()[-1])[target]
    # Check that the two runs have the same loss
    print(f"Run 1 loss:{run_1_loss}", flush=True)
    print(f"Run 2 loss:{run_2_loss}", flush=True)
    assert run_1_loss==run_2_loss
    print('determinism test passed.')

def test_vocab_creation(target):
  bucket_name = target.split("/")[2]
  vocab_path = "/".join(target.split("/")[3:])
  storage_client = storage.Client()
  assert storage.Blob(bucket=storage_client.bucket(bucket_name), name=vocab_path).exists(storage_client)
  print('vocab creation test passed.')

def test_start_step(metrics_file, start_step_target):
  with open(metrics_file, 'r', encoding='utf8') as metrics:
    start_step = json.loads(metrics.readlines()[0])["step"]
  print(f"Start step is {start_step}, start step target is {start_step_target}")
  assert start_step==float(start_step_target)
  print("Start step test passed.")

def main(argv: Sequence[str]) -> None:

  _, test_scenario, *test_vars = argv

  if test_scenario == 'metrics_average':
    assert_metric_average(*test_vars)
  elif test_scenario == 'checkpoint_save_restore':
    test_checkpointing(*test_vars, dataset_type='c4')
  elif test_scenario == 'grain_checkpoint_save_restore':
    test_checkpointing(*test_vars, dataset_type='c4-array_record')
  elif test_scenario == 'determinism':
    test_determinism(*test_vars)
  elif test_scenario == 'vocab_creation':
    test_vocab_creation(*test_vars)
  elif test_scenario == 'final_loss':
    test_final_loss(*test_vars)
  elif test_scenario == 'test_start_step':
    test_start_step(*test_vars)
  else:
     raise ValueError(f"Unrecognized test_scenario {test_scenario}")


if __name__ == "__main__":
  app.run(main)
