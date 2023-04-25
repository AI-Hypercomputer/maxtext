"""Reads and asserts over target values"""
from absl import app
from typing import Sequence
from math import isclose
import json


def read(metrics_file, target):
  """Reads and computes average of target value"""
  avg = 0
  i = 0
  with open(metrics_file, 'r', encoding='utf8') as file:
    lines = file.readlines()
    for line in lines:
      if i >= 10:
        vals = json.loads(line)
        avg += vals[target]
      i+=1
    avg /= (i-10)

  return avg


def test_tflops(metrics_file, target, threshold):
  """Asserts over tflops values"""
  avg_tflops = read(metrics_file, target)
  # Checks for acceptable performance by asserting that average tflops value
  # is greater than or equal to threshold
  assert avg_tflops >= threshold


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


def main(argv: Sequence[str]) -> None:

  _, metrics_file, threshold, target, test_scenario = argv

  if test_scenario == 'performance':
    test_tflops(metrics_file, target, float(threshold))
  elif test_scenario == 'checkpoint_save_restore':
    test_checkpointing(metrics_file, target)


if __name__ == "__main__":
  app.run(main)
