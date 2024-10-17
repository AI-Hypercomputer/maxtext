import ray
from ray_tpu import RayTpuManager
from ray.job_submission import JobSubmissionClient
from trainer import MaxTextTrainer

import logging
import os
import argparse


#### Configurations
# Flags that go into MaxText
MAXTEXT_CONFIG = dict(
    tokenizer_path="assets/tokenizer",
)
# Enables verbose TPU logging.
TPU_VERBOSE_ENV_VARS = {
    "TPU_STDERR_LOG_LEVEL": "0",
    "TPU_MIN_LOG_LEVEL": "0",
    "TF_CPP_MIN_LOG_LEVEL": "0",
}

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


def main(args: argparse.Namespace):
  ray.init(runtime_env=dict(worker_process_setup_hook=setup_loggers))
  run_name = get_job_submission_id()
  logging.info("Got args: %s", args)
  logging.info("This run name: %s", run_name)

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

  logging.info("Creating Ray actors with multislice.")

  config = MAXTEXT_CONFIG
  base_dir = args.base_dir
  # Experiment dir
  output_dir = os.path.join(base_dir, run_name)
  compile_cache_dir = os.path.join(base_dir, "compile_cache")

  if args.data_dir is not None:
    config["dataset_path"] = args.data_dir
  else:
    logging.info("Data dir was not set, defaulting to synthetic data.")
    config["dataset_type"] = "synthetic"

  config["base_output_directory"] = output_dir
  config["jax_cache_dir"] = compile_cache_dir
  config["per_device_batch_size"] = args.per_device_batch_size
  config["max_target_length"] = args.max_target_length
  config["enable_checkpointing"] = args.enable_checkpointing

  env_vars = MACHINE_ENV_VARS
  if args.verbose_tpu:
    env_vars |= TPU_VERBOSE_ENV_VARS

  actors = RayTpuManager.remote(
      tpus=tpu_resources[tpu_type], actor_or_fn=MaxTextTrainer, multislice=True, env=MACHINE_ENV_VARS, config=config
  )

  try:
    # Keep initializations separately so we can catch errors.
    logging.info("Initializing actors.")
    ray.get([actor.initialize.remote(run_name) for actor in actors])
  except Exception as e:
    logging.error("Caught error during initializations: %s", e)
    logging.error("Shutting down...")
    ray.shutdown()
    raise e

  logging.info("Initialization complete. Starting MaxText training...")
  total_steps = int(args.total_steps)
  steps_per_loop = int(args.steps_per_loop)
  steps = 0

  while steps < total_steps:
    logging.info("Training from step %d to %d.", steps, steps_per_loop)

    try:
      ray.get([actor.train.remote(num_steps=steps_per_loop) for actor in actors])
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
  parser = argparse.ArgumentParser(prog="MaxText-Ray-Trainer", description="A Ray trainer for MaxText.")
  parser.add_argument("--base_dir", action="store", required=True, help="Base directory where to store MaxText artifacts.")
  parser.add_argument("--data_dir", action="store", default=None, help="Where MaxText training data is stored.")
  parser.add_argument("--steps_per_loop", action="store", default=50, help="The number of steps to run per loop.")
  parser.add_argument("--total_steps", action="store", default=500, help="The total number of steps to run.")
  parser.add_argument("--per_device_batch_size", action="store", default=2, help="The total number of steps to run.")
  parser.add_argument("--enable_checkpointing", action="store_true", default=False, help="Whether or not to checkpointing.")
  parser.add_argument("--max_target_length", action="store", default=8192, help="The total number of steps to run.")
  parser.add_argument("--verbose_tpu", action="store_true", default=False, help="Whether or not to enable verbose TPU logs.")
  args = parser.parse_args()
  main(args)
