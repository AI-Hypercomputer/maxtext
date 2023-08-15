# Copyright 2023 Google LLC
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

import datetime
from airflow import models
from apis import gcp_config, metric_config, test_config
from apis import task


# TODO(wcromar): Unify these with other tests
TPU_PROD_ENV_ONE_VM = 'tpu-prod-env-one-vm'
EUROPE_WEST4_A = gcp_config.GCPConfig(
    TPU_PROD_ENV_ONE_VM,
    'europe-west4-a',
    metric_config.DatasetOption.XLML_DATASET,
)
US_CENTRAL2_B = gcp_config.GCPConfig(
    TPU_PROD_ENV_ONE_VM,
    'us-central2-b',
    metric_config.DatasetOption.XLML_DATASET,
)


with models.DAG(
    dag_id='jax-integration',
    schedule=None,
    tags=['jax', 'latest'],
    start_date=datetime.datetime(2023, 7, 12),
):
  compilation_cache = task.TpuTask(
      test_config.JSonnetTpuVmTest.from_jax(
          'jax-compilation-cache-test-func-v2-8-1vm'
      ),
      EUROPE_WEST4_A,
  ).run()
  pod = task.TpuTask(
      test_config.JSonnetTpuVmTest.from_jax(
          'jax-pod-latest-tpu-ubuntu2204-base-func-v2-32-1vm'
      ),
      EUROPE_WEST4_A,
  ).run()
  # Tests are currently failing
  # embedding_pjit = task.TPUTask(
  #   test_config.JSonnetTpuVmTest.from_jax('jax-tpu-embedding-pjit-func-v4-8-1vm'),
  #   US_CENTRAL2_B,
  # ).run()
  # embedding_pmap = task.TPUTask(
  #   test_config.JSonnetTpuVmTest.from_jax('jax-tpu-embedding-pmap-func-v3-8-1vm'),
  #   EUROPE_WEST4_A,
  # ).run()
