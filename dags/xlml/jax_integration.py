import datetime
from airflow import models

from apis import gcp_config, test_config
from apis.xlml import task


# TODO(wcromar): Unify these with other tests
TPU_PROD_ENV_ONE_VM = 'tpu-prod-env-one-vm'
EUROPE_WEST4_A = gcp_config.GCPConfig(
  TPU_PROD_ENV_ONE_VM,
  TPU_PROD_ENV_ONE_VM,
  'europe-west4-a',
)
US_CENTRAL2_B = gcp_config.GCPConfig(
  TPU_PROD_ENV_ONE_VM,
  TPU_PROD_ENV_ONE_VM,
  'us-central2-b',
)


with models.DAG(
    dag_id="jax-integration",
    schedule=None,
    tags=["jax", "latest"],
    start_date=datetime.datetime(2023, 7, 12),
):
  compilation_cache = task.TPUTask(
    test_config.JSonnetTpuVmTest.from_jax('jax-compilation-cache-test-func-v2-8-1vm'),
    EUROPE_WEST4_A,
  ).run()
  pod = task.TPUTask(
    test_config.JSonnetTpuVmTest.from_jax('jax-pod-latest-tpu-ubuntu2204-base-func-v2-32-1vm'),
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

