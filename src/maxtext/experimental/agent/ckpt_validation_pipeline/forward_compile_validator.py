# Copyright 2023 Google LLC
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

"""Mock tensor dry-run to validate checkpoint architecture stability."""

import argparse
import json
import traceback
import os
import traceback
import os
import traceback
import os
import traceback
import os
import sys
import time
import jax
import absl.logging
from maxtext.utils import gcs_utils
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from maxtext import pyconfig
from maxtext.models.models import transformer_as_linen
from maxtext.utils.model_creation_utils import create_nnx_abstract_model
from maxtext.utils import max_logging as logger

# Initialize logging verbosity to INFO so logger.info is actually printed
absl.logging.set_verbosity(absl.logging.INFO)


def run_mock_forward(checkpoint_gcs_path, maxtext_model_name, *overrides):
  """Initializes the model abstractly and dry-runs a forward pass."""
  # minimal config required to bypass distributed TPU checks
  config_args = [
      "",
      "src/maxtext/configs/base.yml",
      "skip_jax_distributed_system=true",
  ]

  if checkpoint_gcs_path:
    config_args.append(f"load_parameters_path={checkpoint_gcs_path}")
  if maxtext_model_name:
    config_args.append(f"model_name={maxtext_model_name}")

  # clean and append remaining dynamic overrides (stripping '--' prefixes)
  for override in overrides:
    if override.startswith("--"):
      override = override[2:]
    config_args.append(override)

  # capture returned configuration object
  config = pyconfig.initialize(config_args)

  logger.info(f"Loading model from {config.load_parameters_path}...")
  # create a dummy 1-device hardware mesh using pod's single CPU
  mesh_shape = (1,) * len(config.mesh_axes)
  dummy_device = jax.devices("cpu")[0]
  dummy_mesh = Mesh(np.array([dummy_device]).reshape(mesh_shape), tuple(config.mesh_axes))
  # dynamically generate tensor shapes based on the parsed config
  batch_size = int(config.per_device_batch_size)
  seq_len = int(config.max_target_length)

  logger.info(f"Generating mock tensors with shape: ({batch_size}, {seq_len})")

  # run a single dummy pass
  mock_input = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
  mock_positions = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
  mock_segment_ids = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)

  logger.info("Executing forward pass...")

  if getattr(config, "enable_nnx", False):

    logger.info("Initializing NNX abstract model parameters...")
    _, abstract_model = create_nnx_abstract_model(config, mesh=dummy_mesh)

    logger.info("Tracing forward pass graph with NNX...")

    def forward(m, x, p, s):
      return m(decoder_input_tokens=x, decoder_positions=p, decoder_segment_ids=s)

    out_shape = jax.eval_shape(forward, abstract_model, mock_input, mock_positions, mock_segment_ids)
  else:
    from maxtext.layers import quantizations

    quant = quantizations.configure_quantization(config)
    model = transformer_as_linen(config, mesh=dummy_mesh, quant=quant)

    logger.info("Initializing Linen abstract model parameters...")
    rng = jax.random.PRNGKey(0)
    abstract_variables = jax.eval_shape(
        model.init,
        {"params": rng, "aqt": rng, "dropout": rng},
        mock_input,
        mock_positions,
        mock_segment_ids,
        enable_dropout=False,
    )

    logger.info("Tracing forward pass graph with Linen...")
    out_shape = jax.eval_shape(
        model.apply,
        abstract_variables,
        mock_input,
        mock_positions,
        mock_segment_ids,
        enable_dropout=False,
    )

  logger.info(f"SUCCESS: Model architecture is stable. Output shape: {out_shape}")
  return out_shape


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Mock tensor validation")
  parser.add_argument("--report_gcs_dir", type=str, default="", help="GCS directory for reports")
  parser.add_argument(
      "--checkpoint_gcs_path", type=str, default="", help="GCS directory containing the converted checkpoint"
  )
  parser.add_argument("--maxtext_model_name", type=str, default="", help="MaxText model configuration name")

  args, _overrides = parser.parse_known_args()
  report_gcs_dir = args.report_gcs_dir

  run_name_override = "unknown"
  for override in _overrides:
    if override.startswith("run_name="):
      run_name_override = override.split("=", 1)[1]
      break

  report = {
      "run_name": run_name_override,
      "task": "mock_tensor_validation",
      "timestamp": time.time(),
      "status": "SUCCESS",
  }

  def _save_report(report_data):
    """Saves the mock tensor validation report locally and uploads to GCS."""
    if report_gcs_dir:
      report_name = f"mock_tensor_report_run_name_{run_name_override}_{int(time.time())}.json"
      gcs_dir = report_gcs_dir
      if not gcs_dir.endswith("/"):
        gcs_dir += "/"
      local_report_path = f"/tmp/{report_name}"
      try:
        with open(local_report_path, "w", encoding="utf-8") as f:
          json.dump(report_data, f, indent=2)
        gcs_utils.upload_blob(f"{gcs_dir}{report_name}", local_report_path)
      except Exception as e:
        logger.error(f"Failed to write or upload mock tensor validation report to GCS: {e}")

  try:
    _out_shape = run_mock_forward(args.checkpoint_gcs_path, args.maxtext_model_name, *_overrides)
    report["output_shape"] = str(_out_shape)
  except BaseException as e:  # pylint: disable=broad-exception-caught
    report["status"] = "FAILURE"
    report["error_message"] = traceback.format_exc()
    _save_report(report)
    if isinstance(e, SystemExit):
      sys.exit(e.code)
    raise e

  _save_report(report)
