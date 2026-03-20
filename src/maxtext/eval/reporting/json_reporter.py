# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Write eval results to a structured JSON file."""

from __future__ import annotations

import datetime
import json
import logging
import os

logger = logging.getLogger(__name__)


def write_results(
    benchmark: str,
    model_name: str,
    scores: dict,
    generation_stats: dict,
    config: dict,
    results_path: str = "./eval_results",
) -> dict:
  """Write eval results to a JSON file under *results_path*.

  The output filename is::

      {results_path}/{benchmark}_{model_name}_{timestamp}.json

  Args:
    benchmark: Benchmark name (e.g. "mmlu").
    model_name: Model name (e.g. "llama3.1-8b").
    scores: Dict of metric name to value from the scorer.
    generation_stats: Dict of generation statistics (timing, token counts, etc.).
    config: The full merged configuration dict used for this run.
    results_path: Directory to write the JSON file into.

  Returns:
    Dict with keys:
      - ``results``: The full results dict written to disk.
      - ``local_path``: Absolute path of the written file.
  """
  os.makedirs(results_path, exist_ok=True)

  timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
  # Sanitise model_name for use in a filename.
  safe_model = model_name.replace("/", "_").replace(":", "_")
  filename = f"{benchmark}_{safe_model}_{timestamp}.json"
  local_path = os.path.join(results_path, filename)

  results = {
      "benchmark": benchmark,
      "model_name": model_name,
      "timestamp_utc": timestamp,
      "scores": scores,
      "generation_stats": generation_stats,
      "config": config,
  }

  with open(local_path, "w") as f:
    json.dump(results, f, indent=2)

  logger.info("Results written to %s", local_path)
  return {"results": results, "local_path": os.path.abspath(local_path)}
