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

"""
A DAG to run PyTorch multislice tests
"""
import datetime
from airflow import models
from dags import composer_env, test_owner
from dags.vm_resource import TpuVersion, Zone, DockerImage, ClusterName
from dags.multipod.configs import pytorch_config
from xlml.apis import metric_config

# Run once a day at 10 am UTC (3 am PST)
SCHEDULED_TIME = "0 10 * * *" if composer_env.is_prod_env() else None

with models.DAG(
    dag_id="pytorch_multislice",
    schedule=SCHEDULED_TIME,
    tags=["multipod_team", "pytorch", "nightly"],
    start_date=datetime.datetime(2024, 3, 1),
    catchup=False,
    concurrency=2,
) as dag:
  v4_8 = ClusterName.V4_8_MULTISLICE_CLUSTER
  v4_16 = ClusterName.V4_16_MULTISLICE_CLUSTER

  for num_slices, cluster in [(1, v4_8), (2, v4_8), (1, v4_16)]:
    ici_chips = 4 if cluster == v4_8 else 8
    run_cmds = (
        (
            "python /pytorch/xla/test/spmd/test_sharding_strategies.py "
            f"--ici_fsdp_parallelism {ici_chips} "
            f"--dcn_data_parallelism {num_slices}"
        ),
    )
    pytorch_config.get_nightly_pytorch_config(
        test_name="shardings",
        test_owner=test_owner.JON_B,
        run_commands=run_cmds,
        cluster=cluster,
        num_slices=num_slices,
    ).run()

  pytorch_config.get_nightly_pytorch_config(
      test_name="checkpoint",
      test_owner=test_owner.JON_B,
      run_commands=(
          f"export CHKPT_PATH={metric_config.SshEnvVars.GCS_OUTPUT.value}",
          "pip install gcsfs",
          (
              "python /pytorch/xla/test/spmd/test_xla_distributed_checkpoint.py "
              "EndToEndCheckpointTest.test_multihost_checkpoint"
          ),
      ),
      cluster=v4_16,
      num_slices=2,
  ).run()
