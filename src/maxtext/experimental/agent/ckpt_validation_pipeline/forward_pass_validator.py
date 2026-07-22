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

"""Automated Forward Pass Logit Validation Wrapper for MaxText.
This script wraps tests/utils/forward_pass_logit_checker.py to standardise
reporting for the Airflow fail-fast pipeline.
"""

import argparse
import json
import os
import subprocess
import absl.logging
import maxtext
from maxtext.utils import gcs_utils
# pylint: disable=no-name-in-module
from maxtext.utils import max_logging as logger

# Initialize logging verbosity to INFO so logger.info is actually printed
absl.logging.set_verbosity(absl.logging.INFO)


def validate_forward_pass(
    run_name, internal_model_name, checkpoint_path, report_gcs_dir, unknown_args
):
  """Run logit checker as a subprocess and generate a standardized JSON report."""
  logger.info(f"Running Forward Pass Logit Verification for {run_name}...")

  # base command
  command = [
      "python3",
      "tests/utils/forward_pass_logit_checker.py",
      "src/maxtext/configs/base.yml",
      f"model_name={internal_model_name}",
      f"load_parameters_path={checkpoint_path}",
      "dtype=float32",
      "activations_in_float32=true",
      "matmul_precision=high",
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

  import sys
  import io
  import runpy
  import inspect

  # applying a monkeypatch to maxtext's model_creation_utils because it has a bug where 
  # it cannot resolve SequenceKey (list indices) to string keys in Linen checkpoints.
  from maxtext.utils import model_creation_utils
  source = inspect.getsource(model_creation_utils._fix_restore_args_for_shape_mismatch)

  target = "if isinstance(node, (list, tuple)) and 0 <= key.idx < len(node):\n          node = node[key.idx]\n          continue\n        return None"
  replacement = "if isinstance(node, (list, tuple)) and 0 <= key.idx < len(node):\n          node = node[key.idx]\n          continue\n        if isinstance(node, dict) and str(key.idx) in node:\n          node = node[str(key.idx)]\n          continue\n        return None"
  patched_source = source.replace(target, replacement)
  
  target2 = "if not isinstance(node, dict):\n        return None\n      name = _key_str(key)\n      if name in node:\n        node = node[name]\n        continue"
  replacement2 = "if isinstance(node, (list, tuple)):\n        name = _key_str(key)\n        if name.isdigit() and 0 <= int(name) < len(node):\n          node = node[int(name)]\n          continue\n        return None\n      " + target2
  patched_source = patched_source.replace(target2, replacement2)
  
  env = dict(model_creation_utils.__dict__)
  exec(patched_source, env)
  model_creation_utils._fix_restore_args_for_shape_mismatch = env["_fix_restore_args_for_shape_mismatch"]

  # run script in same process to apply monkeypatch
  old_stdout = sys.stdout
  old_stderr = sys.stderr
  sys.stdout = stdout_cap = io.StringIO()
  sys.stderr = stderr_cap = io.StringIO()
  
  old_cwd = os.getcwd()
  os.chdir(repo_root)
  
  returncode = 0
  try:
    sys.argv = command[1:]
    runpy.run_path("tests/utils/forward_pass_logit_checker.py", run_name="__main__")
  except SystemExit as e:
    returncode = e.code if e.code is not None else 0
  except Exception as e:
    import traceback
    traceback.print_exc(file=sys.stderr)
    returncode = 1
  finally:
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    os.chdir(old_cwd)
    
  stdout_str = stdout_cap.getvalue()
  stderr_str = stderr_cap.getvalue()

  # generate report
  report = {
      "run_name": run_name,
      "model": internal_model_name,
      "success": returncode == 0,
      "stderr": (stderr_str if returncode != 0 else "Success"),
      "stdout": (stdout_str if returncode != 0 else "Success"),
      "checkpoint_used": checkpoint_path,
      "stage": "forward_pass_validation",
  }

  # build and save report
  report_dir = os.path.join(old_cwd, "reports")
  os.makedirs(report_dir, exist_ok=True)
  output_path = os.path.join(report_dir, f"report_{run_name}_forward_pass.json")

  with open(output_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=4)
  logger.info(f"Report saved locally to {output_path}")

  # upload to GCS using standard MaxText utils
  if report_gcs_dir:
    gcs_dir = report_gcs_dir
    if not gcs_dir.endswith("/"):
      gcs_dir += "/"
    gcs_utils.upload_blob(f"{gcs_dir}report_{run_name}_forward_pass.json", output_path)

  if returncode != 0:
    logger.info(f"Command STDOUT:\n{stdout_str}")
    logger.error(f"Command STDERR:\n{stderr_str}")
    raise ValueError(
        "ERROR: Forward pass logit verification failed! See logs for details."
    )

  logger.info("Forward pass validation successful!")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Validate Forward Pass Logits")
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
    validate_forward_pass(
        args.run_name,
        args.maxtext_model_name,
        args.checkpoint_gcs_path,
        args.report_gcs_dir,
        unknown,
    )
  except (ValueError, KeyError, subprocess.CalledProcessError) as e:
    logger.error(f"FAILED: {e}")
    # Always fail hard to halt the Airflow DAG
    import sys

    sys.exit(1)
