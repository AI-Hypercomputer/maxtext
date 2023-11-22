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

"""A DAG to run all TorchBench tests with nightly version."""

import datetime
from airflow import models
from configs import composer_env, vm_resource
from configs.benchmark.pytorch import pytorchxla_torchbench_config as config

_MODELS = [
    "BERT_pytorch",
    "Background_Matting",
    "hf_Bert",
    "Super_SloMo",
    "tts_angular",
]

# Schudule the job to run every day at 5:00PM UTC.
SCHEDULED_TIME = "0 17 * * *" if composer_env.is_prod_env() else None

with models.DAG(
    dag_id="pytorchxla-torchbench",
    schedule=SCHEDULED_TIME,
    tags=["pytorchxla", "nightly", "torchbench"],
    start_date=datetime.datetime(2023, 8, 29),
    catchup=False,
) as dag:
  for model in _MODELS:
    torchbench_extra_flags = [f"--filter={model}"]
    config.get_torchbench_config(
        tpu_version=4,
        tpu_cores=8,
        tpu_zone=vm_resource.Zone.US_CENTRAL2_B.value,
        model_name=model,
        time_out_in_min=60,
        extraFlags=" ".join(torchbench_extra_flags),
    ).run()
