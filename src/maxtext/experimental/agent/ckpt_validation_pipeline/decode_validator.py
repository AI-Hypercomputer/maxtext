# Copyright 2023-2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "innovation" basis,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Automated Checkpoint Validation Agent for MaxText."""

import maxtext
import subprocess
import json
import os
import argparse
import absl.logging
from maxtext.utils import gcs_utils
# pylint: disable=no-name-in-module
from maxtext.utils import max_logging as logger

# Initialize logging verbosity to INFO so logger.info is actually printed
absl.logging.set_verbosity(absl.logging.INFO)


def validate_checkpoint(
    run_name, internal_model_name, checkpoint_path, report_gcs_dir, unknown_args
):
  """Validate MaxText checkpoint using passed arguments."""
  logger.info(f"Validating {run_name}...")
  logger.info(f"Reading weights from: {checkpoint_path}")

  # Check mandatory overrides (tokenizer_path, scan_layers)
  overrides_dict = {}
  for arg in unknown_args:
    if "=" in arg:
      k, v = arg.split("=", 1)
      overrides_dict[k] = v

  if "tokenizer_path" not in overrides_dict:
    raise ValueError("REQUIRED: You must provide 'tokenizer_path' as an override.")
  if "scan_layers" not in overrides_dict:
    raise ValueError(
        "REQUIRED: You must provide 'scan_layers' (true/false) as an override."
    )

  # base command
  command = [
      "python3",
      "src/maxtext/inference/decode.py",
      "src/maxtext/configs/base.yml",
      f"run_name={run_name}",
      f"load_parameters_path={checkpoint_path}",
      f"model_name={internal_model_name}",
  ]

  # append additional maxtext configs from unknown args
  if unknown_args:
    logger.info("Applying additional flags from MaxText overrides...")
    for arg in unknown_args:
      command.append(arg)
      logger.info(f"  -> {arg}")

  # find the absolute path to the root of the repository
  maxtext_module_dir = os.path.dirname(maxtext.__file__)
  repo_root = os.path.abspath(os.path.join(maxtext_module_dir, "../../"))
  # run subprocess (from the top level repo directory)
  result = subprocess.run(
      command, text=True, capture_output=True, check=False, cwd=repo_root
  )

  # generate report
  report = {
      "run_name": run_name,
      "model": internal_model_name,
      "success": result.returncode == 0,  # if returncode is 0, command worked
      "stderr": (
          result.stderr if result.returncode != 0 else "Success"
      ),  # store error message if there's a failure
      "checkpoint_used": checkpoint_path,
  }

  # build and save report
  report_dir = os.path.join(os.getcwd(), "reports")
  os.makedirs(report_dir, exist_ok=True)
  output_path = os.path.join(report_dir, f"report_{run_name}.json")

  with open(output_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=4)
  logger.info(f"Report saved locally to {output_path}")

  # upload to GCS if configured
  if report_gcs_dir:
    gcs_dir = report_gcs_dir
    if not gcs_dir.endswith("/"):
      gcs_dir += "/"
    gcs_utils.upload_blob(f"{gcs_dir}report_{run_name}.json", output_path)

  if result.returncode != 0:
    raise RuntimeError(
        f"Subprocess decode.py failed with exit code {result.returncode}. Stderr: {result.stderr}"
    )


import sys

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Validate MaxText Checkpoints")
  parser.add_argument("--run_name", type=str, required=True, help="Validation run name")
  parser.add_argument(
      "--maxtext_model_name",
      type=str,
      required=True,
      help="Internal MaxText model name",
  )
  parser.add_argument(
      "--checkpoint_gcs_path", type=str, required=True, help="GCS path to checkpoint"
  )
  parser.add_argument(
      "--report_gcs_dir", type=str, default="", help="GCS directory for reports"
  )

  args, unknown = parser.parse_known_args()

  try:
    validate_checkpoint(
        args.run_name,
        args.maxtext_model_name,
        args.checkpoint_gcs_path,
        args.report_gcs_dir,
        unknown,
    )
  except (KeyError, ValueError, FileNotFoundError, RuntimeError) as e:
    logger.error(f"FAILED: {e}")
    sys.exit(1)
