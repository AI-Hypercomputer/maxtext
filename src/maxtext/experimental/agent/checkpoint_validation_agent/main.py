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
# pylint: disable=no-name-in-module
from maxtext.utils import max_logging as logger


def upload_to_gcs(local_path, gcs_dir):
  """Uploads a local file to a GCS directory using gsutil."""
  if not gcs_dir.startswith("gs://"):
    logger.error(f"GCS path must start with gs://, got: {gcs_dir}")
    return

  # Ensure the directory path ends with a slash for proper gsutil copying
  if not gcs_dir.endswith("/"):
    gcs_dir += "/"
    
  gcs_dest = f"{gcs_dir}{os.path.basename(local_path)}"
  logger.info(f"Uploading report to {gcs_dest}...")
  
  try:
    subprocess.run(["gsutil", "cp", local_path, gcs_dest], check=True, capture_output=True, text=True)
    logger.info("GCS Upload successful.")
  except subprocess.CalledProcessError as e:
    logger.error(f"Failed to upload report to GCS: {e.stderr}")


def validate_checkpoint(json_config_path):
  """Validate MaxText checkpoint using JSON configuration file."""
  # load data while enforcing json requirement
  if not os.path.exists(json_config_path):
    raise FileNotFoundError(f"Config file not found at: {json_config_path}")

  with open(json_config_path, "r", encoding="utf-8") as f:
    user_config = json.load(f)

  # check json for mandatory fields
  required_keys = [
      "run_name",
      "maxtext_model_name",
      "checkpoint_gcs_path",
      "maxtext_overrides",
  ]
  for key in required_keys:
    if key not in user_config:
      raise KeyError(f"CRITICAL ERROR: JSON config is missing required key '{key}'")

  # extract the mandatory fields from the json config
  run_name = user_config["run_name"]
  internal_model_name = user_config["maxtext_model_name"]
  checkpoint_path = user_config["checkpoint_gcs_path"]
  overrides = user_config.get("maxtext_overrides", {})
  report_gcs_dir = user_config.get("report_gcs_dir")

  # raise error if user doesn't provide required fields
  if "tokenizer_path" not in overrides:
    raise ValueError("REQUIRED: You must provide 'tokenizer_path' in maxtext_overrides in your JSON config.")
  if "scan_layers" not in overrides:
    raise ValueError("REQUIRED: You must provide 'scan_layers' (true/false) in maxtext_overrides in your JSON config.")
  tokenizer = overrides.pop("tokenizer_path")
  scan = overrides.pop("scan_layers")

  logger.info(f"Validating {run_name}...")
  logger.info(f"Reading weights from: {checkpoint_path}")

  # base command
  command = [
      "python3",
      "src/maxtext/inference/decode.py",
      "src/maxtext/configs/base.yml",
      f"run_name={run_name}",
      # from json
      f"load_parameters_path={checkpoint_path}",
      # from registry
      f"model_name={internal_model_name}",
      f"tokenizer_path={tokenizer}",
      f"scan_layers={scan}",
  ]

  # append additional maxtext configs in nested json object
  if overrides:
    logger.info("Applying additional flags from MaxText overrides...")
    for flag_name, flag_value in overrides.items():
      # turns {"max_target_length": 4096} into "max_target_length=4096"
      command.append(f"{flag_name}={flag_value}")
      logger.info(f"  -> {flag_name}={flag_value}")

  # find the absolute path to the root of the repository
  maxtext_module_dir = os.path.dirname(maxtext.__file__)
  repo_root = os.path.abspath(os.path.join(maxtext_module_dir, "../../"))
  # run subprocess (from the top level repo directory)
  result = subprocess.run(command, text=True, capture_output=True, check=False, cwd=repo_root)

  # generate report
  report = {
      "run_name": run_name,
      "model": internal_model_name,
      "success": result.returncode == 0,  # if returncode is 0, command worked
      "stderr": (result.stderr if result.returncode != 0 else "Success"),  # store error message if there's a failure
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
    upload_to_gcs(output_path, report_gcs_dir)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Validate MaxText Checkpoints via JSON config")
  parser.add_argument("--config", type=str, required=True, help="Path to JSON config file")
  args = parser.parse_args()

  try:
    validate_checkpoint(args.config)
  except (KeyError, ValueError, FileNotFoundError) as e:
    logger.error(f"FAILED: {e}")
