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


def validate_forward_pass(run_name, internal_model_name, checkpoint_path, report_gcs_dir, unknown_args):
    """Run Snehal's logit checker as a subprocess and generate a standardized JSON report."""
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
        "matmul_precision=high"
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
    
    # run subprocess
    result = subprocess.run(command, text=True, capture_output=True, check=False, cwd=repo_root)

    # generate report
    report = {
        "run_name": run_name,
        "model": internal_model_name,
        "success": result.returncode == 0,
        "stderr": (result.stderr if result.returncode != 0 else "Success"),
        "stdout": (result.stdout if result.returncode != 0 else "Success"),
        "checkpoint_used": checkpoint_path,
        "stage": "forward_pass_validation"
    }

    # build and save report
    report_dir = os.path.join(os.getcwd(), "reports")
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

    if result.returncode != 0:
        logger.info(f"Command STDOUT:\n{result.stdout}")
        logger.error(f"Command STDERR:\n{result.stderr}")
        raise ValueError("ERROR: Forward pass logit verification failed! See logs for details.")
    
    logger.info("Forward pass validation successful!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Forward Pass Logits")
    parser.add_argument("--run_name", type=str, required=True, help="Validation run name")
    parser.add_argument("--maxtext_model_name", type=str, required=True, help="Internal MaxText model name")
    parser.add_argument("--checkpoint_gcs_path", type=str, required=True, help="GCS path to checkpoint")
    parser.add_argument("--report_gcs_dir", type=str, default="", help="GCS directory for reports")
    
    args, unknown = parser.parse_known_args()

    try:
        validate_forward_pass(
            args.run_name, 
            args.maxtext_model_name, 
            args.checkpoint_gcs_path, 
            args.report_gcs_dir, 
            unknown
        )
    except (ValueError, KeyError, subprocess.CalledProcessError) as e:
        logger.error(f"FAILED: {e}")
        # Always fail hard to halt the Airflow DAG
        import sys
        sys.exit(1)
