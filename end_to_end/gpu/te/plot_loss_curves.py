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
""" Plot loss curves from training logs """

# Usage: python plot_loss_curves.py logdir

import re
import os
import sys
import argparse

import matplotlib.pyplot as plt


def parse_loss_data(file_path):
  """
    Parses a text file for lines matching the pattern:
    completed step: <int>, seconds: <float>, TFLOP/s/device: <float>,
        Tokens/s/device: <float>, total_weights: <int>, loss: <float>
    Returns a list of tuples with the extracted values.
    """
  pattern = re.compile(
    r"completed step: (\d+), seconds: ([\d.]+), TFLOP/s/device: ([\d.]+), Tokens/s/device: ([\d.]+), total_weights: (\d+), loss: ([\d.]+)" # pylint: disable=line-too-long
  )
  results = []
  with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
      match = pattern.search(line)
      if match:
        step = int(match.group(1))
        seconds = float(match.group(2))
        tflops = float(match.group(3))
        tokens_per_sec = float(match.group(4))
        total_weights = int(match.group(5))
        loss = float(match.group(6))
        results.append(
          (step, seconds, tflops, tokens_per_sec, total_weights, loss)
        )
    return results


def main(args):
  parser = argparse.ArgumentParser(
    description="Plot training loss curve from log files."
  )
  parser.add_argument(
    "logdir", type=str, help="Directory containing training log files."
  )
  parsed_args = parser.parse_args(args)

  logdir = parsed_args.logdir
  log_files = [
    os.path.join(logdir, f)
    for f in os.listdir(logdir)
    if os.path.isfile(os.path.join(logdir, f)) and f.endswith(".log")
  ]

  # Extract parallelism configs from filenames
  config_pattern = re.compile(r"dp(\d+)_tpsp(\d+)_fsdp(\d+)")
  configs = {}
  for log_file in log_files:
    fname = os.path.basename(log_file)
    match = config_pattern.search(fname)
    if match:
      dp, tpsp, fsdp = match.groups()
      key = (int(dp), int(tpsp), int(fsdp))
      configs.setdefault(key, []).append(log_file)

  # Plot for each config
  for (dp, tpsp, fsdp), files in configs.items():
    plt.figure(figsize=(8, 5))
    for log_file in files:
      data = parse_loss_data(log_file)
      if not data:
        continue
      steps = [item[0] for item in data]
      losses = [item[5] for item in data]
      plt.plot(
        steps,
        losses,
        marker="",
        linestyle="-",
        label=os.path.basename(log_file),
      )
    plt.legend()
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(f"Loss Curves (dp={dp}, tpsp={tpsp}, fsdp={fsdp})")
    plt.grid(True)
    plt.tight_layout()
    out_image_path = f"loss_curves_dp{dp}_tpsp{tpsp}_fsdp{fsdp}.png"
    plt.savefig(out_image_path)
    print(f"Saved plot to {out_image_path}")
    plt.close()


if __name__ == "__main__":
  main(sys.argv[1:])
