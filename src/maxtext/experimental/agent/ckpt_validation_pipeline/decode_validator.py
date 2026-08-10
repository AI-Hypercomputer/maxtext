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

"""Automated Decode Validation Agent for MaxText."""

import maxtext
import subprocess
import json
import os
import sys
import argparse
import absl.logging
from maxtext.utils import gcs_utils
# pylint: disable=no-name-in-module
from maxtext.utils import max_logging as logger

# Initialize logging verbosity to INFO so logger.info is actually printed
absl.logging.set_verbosity(absl.logging.INFO)


def validate_checkpoint(report_gcs_dir, maxtext_args):
  """Validate MaxText checkpoint using passed arguments."""
  # Check mandatory overrides (tokenizer_path, scan_layers)
  overrides_dict = {}
  for arg in maxtext_args:
    if "=" in arg:
      k, v = arg.split("=", 1)
      overrides_dict[k] = v

  run_name = overrides_dict.get("run_name", "default_run")
  internal_model_name = overrides_dict.get("model_name", "unknown")
  checkpoint_path = overrides_dict.get("load_parameters_path", "unknown")

  logger.info(f"Validating {run_name}...")
  logger.info(f"Reading weights from: {checkpoint_path}")

  if "tokenizer_path" not in overrides_dict:
    raise ValueError("REQUIRED: You must provide 'tokenizer_path' as an override.")
  if "scan_layers" not in overrides_dict:
    raise ValueError("REQUIRED: You must provide 'scan_layers' (true/false) as an override.")

  # base command
  command = [
      "python3",
      "src/maxtext/inference/decode.py",
      "src/maxtext/configs/base.yml",
  ]

  # append additional maxtext configs from maxtext_args
  if maxtext_args:
    logger.info("Applying additional flags from MaxText overrides...")
    for arg in maxtext_args:
      command.append(arg)
      logger.info(f"  -> {arg}")

  # find the absolute path to the root of the repository
  maxtext_module_dir = os.path.dirname(maxtext.__file__)
  repo_root = os.path.abspath(os.path.join(maxtext_module_dir, "../../"))
  # run subprocess with real-time streaming (from the top level repo directory)
  logger.info("=== Subprocess Stdout ===")
  try:
    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=repo_root,
    ) as proc:
      stdout_lines = []
      import threading

      def reader():
        for line in proc.stdout:
          logger.info(line.rstrip())
          stdout_lines.append(line)

      reader_thread = threading.Thread(target=reader)
      reader_thread.daemon = True
      reader_thread.start()

      try:
        proc.wait(timeout=1800)  # 30 minutes timeout
      except subprocess.TimeoutExpired:
        proc.kill()
        stdout_str = "".join(stdout_lines) + "\nSubprocess timed out after 30 minutes"
        stderr_str = "Subprocess timed out after 30 minutes"
        returncode = -1
        logger.error("Subprocess decode.py timed out after 30 minutes!")
      else:
        returncode = proc.returncode
        reader_thread.join(timeout=10)  # Ensure all stdout is read up to EOF
        stdout_str = "".join(stdout_lines)
        stderr_str = "Redirected to stdout"
  except Exception as e:
    returncode = -1
    stdout_str = ""
    stderr_str = str(e)

  # generate report
  report = {
      "run_name": run_name,
      "model": internal_model_name,
      "status": "SUCCESS" if returncode == 0 else "FAILED",
      "success": returncode == 0,  # if returncode is 0, command worked
      "stdout": stdout_str,  # store standard output (contains generated text like "Input ... -> ...")
      "stderr": (stderr_str if returncode != 0 else "Success"),  # store error message if there's a failure
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

  if returncode != 0:
    raise RuntimeError(f"Subprocess decode.py failed with exit code {returncode}. Stderr: {stderr_str}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Validate MaxText Checkpoints")
  parser.add_argument("--report_gcs_dir", type=str, default="", help="GCS directory for reports")

  args, _maxtext_args = parser.parse_known_args()

  try:
    validate_checkpoint(
        args.report_gcs_dir,
        _maxtext_args,
    )
  except Exception as e:
    logger.error(f"FAILED: {e}")
    # Construct and upload a failure report to GCS if GCS dir is provided
    if args.report_gcs_dir:
      overrides_dict = {}
      for arg in _maxtext_args:
        if "=" in arg:
          k, v = arg.split("=", 1)
          overrides_dict[k] = v
      run_name = overrides_dict.get("run_name", "default_run")
      internal_model_name = overrides_dict.get("model_name", "unknown")
      checkpoint_path = overrides_dict.get("load_parameters_path", "unknown")

      report = {
          "run_name": run_name,
          "model": internal_model_name,
          "status": "FAILED",
          "success": False,
          "stdout": "",
          "stderr": str(e) if str(e) else type(e).__name__,
          "checkpoint_used": checkpoint_path,
      }

      report_dir = os.path.join(os.getcwd(), "reports")
      os.makedirs(report_dir, exist_ok=True)
      output_path = os.path.join(report_dir, f"report_{run_name}.json")
      with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

      gcs_dir = args.report_gcs_dir
      if not gcs_dir.endswith("/"):
        gcs_dir += "/"
      gcs_utils.upload_blob(f"{gcs_dir}report_{run_name}.json", output_path)

    if isinstance(e, SystemExit):
      sys.exit(e.code)
    sys.exit(1)
