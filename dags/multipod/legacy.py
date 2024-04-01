# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A DAG to run tests migrated from the legacy XL ML infrastructure"""

import datetime
from airflow import models
from dags import composer_env, gcs_bucket, test_owner
from dags.vm_resource import TpuVersion, Zone, Project, DockerImage, ClusterName
from dags.multipod.configs import legacy_unit_test, gke_config
from dags.multipod.configs.common import SetupMode, Platform

# Run once a day at 9 am UTC (1 am PST)
SCHEDULED_TIME = "0 9 * * *" if composer_env.is_prod_env() else None
DOCKER_IMAGE = {
    SetupMode.STABLE: DockerImage.MAXTEXT_TPU_JAX_STABLE,
    SetupMode.NIGHTLY: DockerImage.MAXTEXT_TPU_JAX_NIGHTLY,
}

# TFLOPS threshold values by configuration
tflop_thresholds = {
    "v4-8": {
        "1": 130,
    },
    "v4-16": {
        "1": 120,
        "2": 100,
    },
    "v5litepod-256": {
        "1": 60,
        "2": 60,
        "4": 60,
        "8": 60,
    },
}

with models.DAG(
    dag_id=f"multipod_legacy_xlml",
    schedule=SCHEDULED_TIME,
    tags=["multipod_team", "xlml", "legacy", "stable", "nightly"],
    start_date=datetime.datetime(2024, 1, 10),
    catchup=False,
    concurrency=2,
) as dag:
  for test_mode in [SetupMode.STABLE, SetupMode.NIGHTLY]:
    # Tests that require scripts from the `jax/unit_tests` folder should follow
    # this pattern.
    # TODO(jonbolin): Example for legacy unit test migration - evaluate whether
    # to remove gpt1-like tests once test migration is complete.

    for n_slice in [1, 2]:
      legacy_unit_test.get_legacy_unit_test_config(
          script_to_copy="gpt1-like.py",
          test_cmd=("python3 gpt1-like.py",),
          tpu_version=TpuVersion.V4,
          tpu_cores=16,
          tpu_zone=Zone.US_CENTRAL2_B.value,
          time_out_in_min=60,
          test_name=f"gpt1-like-{test_mode.value}",
          docker_image=DOCKER_IMAGE[test_mode].value,
          test_owner=test_owner.JON_B,
          num_slices=n_slice,
          cluster_name=ClusterName.V4_16_MULTISLICE_CLUSTER.value,
      ).run()

    # Tests that run MaxText end_to_end tests should follow this pattern.
    gke_config.get_gke_config(
        tpu_version=TpuVersion.V4,
        tpu_cores=8,
        tpu_zone=Zone.US_CENTRAL2_B.value,
        time_out_in_min=60,
        test_name=f"maxtext-decode-{test_mode.value}",
        run_model_cmds=(
            f"bash end_to_end/test_decode.sh 10 gs://maxtext-xlml gs://maxtext-xlml/dataset xlml-decode-v4-8-1slice-{test_mode.value}",
        ),
        docker_image=DOCKER_IMAGE[test_mode].value,
        test_owner=test_owner.JON_B,
    ).run()

    # v4-8 1 slice TFLOPS test
    gke_config.get_gke_config(
        tpu_version=TpuVersion.V4,
        tpu_cores=8,
        tpu_zone=Zone.US_CENTRAL2_B.value,
        time_out_in_min=60,
        test_name=f"maxtext-perf-{test_mode.value}",
        run_model_cmds=(
            f"bash end_to_end/test_tflops.sh xlml {tflop_thresholds['v4-8']['1']} gs://maxtext-xlml gs://maxtext-xlml/dataset xlml-tflops-v4-8-1slice-{test_mode.value}",
        ),
        docker_image=DOCKER_IMAGE[test_mode].value,
        test_owner=test_owner.PRIYANKA_G,
    ).run()

    # v4-16 1 and 2 slice TFLOPS test
    for n_slice in [1, 2]:
      gke_config.get_gke_config(
          tpu_version=TpuVersion.V4,
          tpu_cores=16,
          num_slices=n_slice,
          tpu_zone=Zone.US_CENTRAL2_B.value,
          time_out_in_min=60,
          test_name=f"maxtext-perf-{test_mode.value}",
          run_model_cmds=(
              f"bash end_to_end/test_tflops.sh xlml {tflop_thresholds['v4-16'][str(n_slice)]} gs://maxtext-xlml gs://maxtext-xlml/dataset xlml-tflops-v4-16-{n_slice}slice-{test_mode.value}",
          ),
          cluster_name=ClusterName.V4_16_MULTISLICE_CLUSTER.value,
          docker_image=DOCKER_IMAGE[test_mode].value,
          test_owner=test_owner.PRIYANKA_G,
      ).run()

    # v4-16 two slices determinism test
    slice_num = 2
    accelerator = "v4-16"
    base_output_directory = f"{gcs_bucket.BASE_OUTPUT_DIR}/maxtext_determinism"
    dataset_path = gcs_bucket.MAXTEXT_DIR
    maxtext_v4_configs_test = gke_config.get_gke_config(
        tpu_version=TpuVersion.V4,
        tpu_cores=16,
        num_slices=slice_num,
        cluster_name=ClusterName.V4_16_MULTISLICE_CLUSTER.value,
        tpu_zone=Zone.US_CENTRAL2_B.value,
        time_out_in_min=60,
        test_name=f"maxtext-determinism-{test_mode.value}",
        run_model_cmds=(
            "bash end_to_end/test_determinism.sh"
            f" determinism-{test_mode.value}-{slice_num}x-{accelerator}"
            f" {base_output_directory} {dataset_path}",
        ),
        docker_image=DOCKER_IMAGE[test_mode].value,
        test_owner=test_owner.MATT_D,
    ).run()

    # v4-16 1 slice, v4-8 1 and 2 slices shardings.py test
    for cores in [8, 16]:
      if cores == 8:
        cluster_name = ClusterName.V4_8_MULTISLICE_CLUSTER.value
      elif cores == 16:
        cluster_name = ClusterName.V4_16_MULTISLICE_CLUSTER.value
      for n_slice in [1, 2]:
        if cores == 16 and n_slice == 2:  # Skip test for 2 slice v4-16
          break
        gke_config.get_gke_config(
            tpu_version=TpuVersion.V4,
            tpu_cores=cores,
            num_slices=n_slice,
            tpu_zone=Zone.US_CENTRAL2_B.value,
            time_out_in_min=60,
            test_name=f"maxtext-shardings-{test_mode.value}",
            run_model_cmds=(
                f"python pedagogical_examples/shardings.py {n_slice} {cores//2}",
            ),
            cluster_name=cluster_name,
            docker_image=DOCKER_IMAGE[test_mode].value,
            test_owner=test_owner.MOHIT_K,
        ).run()

  # v4-8 2 slices checkpoint resharding test
  gke_config.get_gke_config(
      tpu_version=TpuVersion.V4,
      tpu_cores=8,
      num_slices=2,
      tpu_zone=Zone.US_CENTRAL2_B.value,
      time_out_in_min=60,
      test_name=f"maxtext-checkpoint-reshard-{SetupMode.STABLE.value}",
      run_model_cmds=(
          f"bash end_to_end/test_checkpoint_resharding.sh xlml-checkpoint-resharding-v4-8-2slice-{SetupMode.STABLE.value} gs://maxtext-xlml gs://maxtext-xlml/dataset",
      ),
      docker_image=DOCKER_IMAGE[SetupMode.STABLE].value,
      test_owner=test_owner.PRIYANKA_G,
  ).run()
