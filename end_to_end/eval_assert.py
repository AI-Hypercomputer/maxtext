"""Reads and asserts over target values"""
from absl import app
from typing import Sequence
from math import isclose
from google.cloud import storage
import json


def read(metrics_file, target):
  """Reads and computes average of target value"""
  avg = 0
  i = 0
  with open(metrics_file, 'r', encoding='utf8') as file:
    lines = file.readlines()
    for line in lines:
      # skip the first 10 lines for burn in
      if i >= 10:
        vals = json.loads(line)
        avg += vals[target]
      i+=1
    avg /= (i-10)

  return avg


def assert_metric_average(metrics_file, target, threshold):
  avg_value = read(metrics_file, target)
  # Checks for acceptable performance by asserting that the average metric (e.g. TFLOPs)
  # is greater than the threshold.
  print(f'avg value of target {target} is {avg_value}')
  assert avg_value >= threshold


def test_checkpointing(metrics_file, target):
  """Asserts over loss values from loaded checkpoint"""
  metrics_file_saved = 'saved_' + metrics_file
  metrics_file_restored = 'restored_' + metrics_file

  with open(metrics_file_saved, 'r', encoding='utf8') as saved,\
    open(metrics_file_restored, 'r', encoding='utf8') as restored:
    saved_loss = json.loads(saved.readlines()[-1])[target]
    restored_loss = json.loads(restored.readlines()[0])[target]
    # Checks that checkpoint restore was successful by comparing loss of last
    # step in saved checkpoint to loss of first step in restored checkpoint
    assert isclose(saved_loss, restored_loss, rel_tol=0.1)

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
    assert isclose(run_1_loss, run_2_loss, rel_tol=0.001)

def test_vocab_creation(target):
  bucket_name = target.split("/")[2]
  vocab_path = "/".join(target.split("/")[3:])
  storage_client = storage.Client()
  assert storage.Blob(bucket=storage_client.bucket(bucket_name), name=vocab_path).exists(storage_client)


def main(argv: Sequence[str]) -> None:

  _, metrics_file, threshold, target, test_scenario = argv

  if test_scenario == 'metrics_average':
    assert_metric_average(metrics_file, target, float(threshold))
  elif test_scenario == 'checkpoint_save_restore':
    test_checkpointing(metrics_file, target)
  elif test_scenario == 'determinism':
    test_determinism(metrics_file, target)
  elif test_scenario == 'vocab_creation':
    test_vocab_creation(target)


if __name__ == "__main__":
  app.run(main)
