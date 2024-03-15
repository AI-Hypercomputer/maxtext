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
    SetupMode.STABLE: DockerImage.MAXTEXT_JAX_STABLE,
    SetupMode.NIGHTLY: DockerImage.MAXTEXT_JAX_NIGHTLY,
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
