# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

export LIBTPU_INIT_ARGS="--xla_enable_async_all_gather=true TPU_MEGACORE=MEGACORE_DENSE"
RUN_NAME=mattdavidow-rs-$(date +%Y-%m-%d-%H-%M)
#python3 MaxText/train.py MaxText/configs/test.yml run_name=${RUN_NAME} enable_profiler=true enable_checkpointing=true dcn_data_parallelism=3 ici_fsdp_parallelism=192 ici_tensor_parallelism=1 scale=1 per_device_batch_size=4 remat_policy=minimal
python3 MaxText/train.py MaxText/configs/test.yml run_name=${RUN_NAME} enable_profiler=true enable_checkpointing=true dcn_data_parallelism=1 ici_fsdp_parallelism=192 ici_tensor_parallelism=1 scale=1 per_device_batch_size=12 remat_policy=full