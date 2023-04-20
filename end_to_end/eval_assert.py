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
  assert avg_tflops >= threshold


def test_checkpointing(metrics_file, target):
  """Asserts over loss values from loaded checkpoint"""
  metrics_file_saved = 'saved_' + metrics_file
  metrics_file_restored = 'restored_' + metrics_file

  with open(metrics_file_saved, 'r', encoding='utf8') as saved,\
    open(metrics_file_restored, 'r', encoding='utf8') as restored:
    saved_loss = json.loads(saved.readlines()[-1])[target]
    restored_loss = json.loads(restored.readlines()[0])[target]
    assert isclose(saved_loss, restored_loss, rel_tol=0.5)


def main(argv: Sequence[str]) -> None:

  _, metrics_file, threshold, target = argv

  if target == 'perf/per_device_tflops_per_sec':
    test_tflops(metrics_file, target, float(threshold))
  elif target == 'learning/loss':
    test_checkpointing(metrics_file, target)


if __name__ == "__main__":
  app.run(main)
