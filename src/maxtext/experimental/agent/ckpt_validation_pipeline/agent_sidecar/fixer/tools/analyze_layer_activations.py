# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helper script for the Fixer agent to inspect layer-by-layer activation similarity."""

import argparse
import json
import sys
from maxtext.experimental.agent.ckpt_validation_pipeline import layer_metrics


def main():
  parser = argparse.ArgumentParser(description="Run layer-by-layer activation divergence analysis.")
  parser.add_argument("--report_json", type=str, required=False, help="Path to forward pass validation report JSON.")
  args = parser.parse_known_args()[0]

  if args.report_json:
    try:
      with open(args.report_json, "r", encoding="utf-8") as f:
        data = json.load(f)
      if "layer_by_layer_metrics" in data:
        print(data["layer_by_layer_metrics"].get("summary_table", "No summary table found."))
        return 0
    except Exception as e:
      print(f"Could not load report JSON: {e}")

  print("Layer metrics module ready. Use layer_metrics.analyze_layer_divergence() in debugging scripts.")
  return 0


if __name__ == "__main__":
  sys.exit(main())
