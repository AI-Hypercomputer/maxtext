"""
A simple Ray trainer script utilizing ray_tpu to run MaxText training on a
multislice TPU.

RayTpuManager ensures that each Ray worker is initialized with multislice
environment settings.

Example usage:

ray job submit \
    --working-dir . \
    --runtime-env-json='{"excludes": [".git", "MaxText/test_assets"]}' \
    -- python MaxText/ray_trainer.py MaxText/configs/base.yml \
        base_output_directory=/tmp/maxtext \
        dataset_type=synthetic \
        per_device_batch_size=2 \
        max_target_length=8192 \
        model_name=default \
        steps=100

"""
import ray
from ray_tpu import RayTpuManager
from ray.job_submission import JobSubmissionClient
from train import main as maxtext_main

import logging
from typing import Sequence
from absl import app


# Default env vars that run on all TPU VMs.
MACHINE_ENV_VARS = {
    "ENABLE_PJRT_COMPATIBILITY": "true",
    "TPU_SLICE_BUILDER_DUMP_CHIP_FORCE": "true",
    "TPU_SLICE_BUILDER_DUMP_ICI": "true",
    "XLA_FLAGS": "--xla_dump_to=/tmp/xla_dump_file --xla_dump_hlo_as_proto",  # Dumps HLOs for debugging
}


def setup_loggers():
  """Sets up loggers for Ray."""
  logging.basicConfig(level=logging.INFO)


def get_job_submission_id() -> str:
  """Returns the Ray job submission ID."""
  c = JobSubmissionClient()
  current_job_id = ray.get_runtime_context().get_job_id()
  jobs = c.list_jobs()
  return [job.submission_id for job in jobs if job.job_id == current_job_id][0]


def main(argv: Sequence[str]):
  ray.init(runtime_env=dict(worker_process_setup_hook=setup_loggers))

  run_name = get_job_submission_id()
  logging.info("Got argv: %s", argv)
  logging.info("Run name: %s", run_name)

  argv.append(f"run_name={run_name}")

  tpu_resources = RayTpuManager.get_available_resources()
  num_detected_tpu_types = len(tpu_resources.keys())
  if num_detected_tpu_types == 0:
    logging.error("Did not detect any TPUs in the cluster, check your Ray cluster setup: %s", ray.available_resources())

  tpu_type = list(tpu_resources.keys())[0]
  if num_detected_tpu_types > 1:
    logging.warning(
        "Detected %d TPUs in the cluster. MaxText does not support clusters with multiple TPU pod slices - falling back to using %s",
        num_detected_tpu_types,
        tpu_type,
    )

  logging.info("Running on pod slice type %s.", tpu_type)

  tasks = RayTpuManager.remote(
      tpus=tpu_resources[tpu_type], actor_or_fn=maxtext_main, multislice=True, env=MACHINE_ENV_VARS, argv=argv)

  try:
    ray.get(tasks)
  except Exception as e:
    logging.error("Caught error during training: %s", e)
    logging.error("Shutting down...")
    ray.shutdown()
    raise e

  logging.info("Training complete!")
  ray.shutdown()


if __name__ == "__main__":
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)
  app.run(main)
