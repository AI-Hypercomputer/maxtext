"""Reads and asserts over target values"""
from absl import app
from typing import Sequence
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


def main(argv: Sequence[str]) -> None:

  _, metrics_file, threshold, target = argv

  avg_val = read(metrics_file, target)

  assert avg_val >= float(threshold)


if __name__ == "__main__":
  app.run(main)
