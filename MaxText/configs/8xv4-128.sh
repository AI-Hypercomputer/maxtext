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
python3 MaxText/train.py MaxText/configs/base.yml run_name=8xv4-128 dcn_data_parallelism=8 ici_fsdp_parallelism=64 steps=10 per_device_batch_size=16  enable_profiler=true remat_policy=full scale=3
# 146 TFLOP/s, 22B
