"""Automated Checkpoint Validation Agent for MaxText."""

import subprocess
import json
import os
import argparse
# pylint: disable=no-name-in-module
from maxtext import max_logging as logger


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

  # run subprocess
  result = subprocess.run(command, text=True, capture_output=True, check=False)

  # generate report
  report = {
      "run_name": run_name,
      "model": internal_model_name,
      "success": result.returncode == 0,  # if returncode is 0, command worked
      "stderr": (result.stderr if result.returncode != 0 else "Success"),  # store error message if there's a failure
      "checkpoint_used": checkpoint_path,
  }

  # build and save report
  report_dir = os.path.join(os.path.dirname(__file__), "reports")
  os.makedirs(report_dir, exist_ok=True)
  output_path = os.path.join(report_dir, f"report_{run_name}.json")

  with open(output_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=4)
  logger.info(f"Report saved to {output_path}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Validate MaxText Checkpoints via JSON config")
  parser.add_argument("--config", type=str, required=True, help="Path to JSON config file")
  args = parser.parse_args()

  try:
    validate_checkpoint(args.config)
  except (KeyError, ValueError, FileNotFoundError) as e:
    logger.error(f"FAILED: {e}")
