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


# XLA runtime args
XLA_RUNTIME_FLAGS = (
    "TPU_MEGACORE=MEGACORE_DENSE "
    "--xla_enable_async_all_gather=true "
    "--xla_tpu_enable_megascale_barrier=true "
    "--xla_jf_spmd_threshold_for_windowed_einsum_mib=0 "
    "--xla_enable_async_collective_permute=true "
    "--xla_jf_rematerialization_percent_shared_memory_limit=97 "
    "--xla_tpu_decompose_all_gather_einsum=true --xla_tpu_spmd_threshold_for_allgather_cse=10 "
    "--xla_tpu_prefuse_self_attention=false --xla_tpu_rwb_fusion=false "
    "--xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_dcn_max_overlap_estimation=32.0 "
    "--xla_tpu_data_parallel_opt_different_sized_ops=true "
    "--xla_vf_vmem_max_overlap_to_mem_size_async_copy_ratio=10 "
    "--megascale_enable_async_host_commands=true "
    "--xla_tpu_spmd_rng_bit_generator_unsafe=true"
)

# Enables verbose TPU logging.
TPU_VERBOSE_ENV_VARS = {
    "TPU_STDERR_LOG_LEVEL": "0",
    "TPU_MIN_LOG_LEVEL": "0",
    "TF_CPP_MIN_LOG_LEVEL": "0",
}

# Default env vars that run on all TPU VMs.
MACHINE_ENV_VARS = {
    "TPU_PREMAPPED_BUFFER_SIZE": "4294967296",
    "XLA_FLAGS": "--xla_dump_to=/tmp/xla_dump_file --xla_dump_hlo_as_proto", # Dumps HLOs for debugging
    "LIBTPU_INIT_ARGS": XLA_RUNTIME_FLAGS,
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
            num_detected_tpu_types, tpu_type)

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

    env_vars = MACHINE_ENV_VARS
    if args.verbose_tpu:
        env_vars |= TPU_VERBOSE_ENV_VARS

    actors = RayTpuManager.remote(
        tpus=tpu_resources[tpu_type],
        actor_or_fn=MaxTextTrainer,
        multislice=True,
        env=MACHINE_ENV_VARS,
        config=config)

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
    total_steps = args.total_steps
    steps = 0

    while steps < total_steps:
        logging.info("Training from step %d to %d.", steps, args.steps_per_loop)

        try:
            ray.get([actor.train.remote(num_steps=args.steps_per_loop) for actor in actors])
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
    parser = argparse.ArgumentParser(
        prog="MaxText-Ray-Trainer",
        description="A Ray trainer for MaxText.")
    parser.add_argument(
        "--base_dir",
        action="store",
        required=True,
        help="Base directory where to store MaxText artifacts.")
    parser.add_argument(
        "--data_dir",
        action="store",
        default=None,
        help="Where MaxText training data is stored.")
    parser.add_argument(
        "--steps_per_loop",
        action="store",
        default=50,
        help="The number of steps to run per loop.")
    parser.add_argument(
        "--total_steps",
        action="store",
        default=500,
        help="The total number of steps to run.")
    parser.add_argument(
        "--verbose_tpu",
        action="store_true",
        default=False,
        help="Whether or not to enable verbose TPU logs.")
    args = parser.parse_args()
    main(args)
