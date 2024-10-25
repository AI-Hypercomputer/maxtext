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

xla_flags_1 = {
    "xla_jf_auto_cross_replica_sharding": "False",
    "xla_tpu_enable_windowed_einsum_for_reduce_scatter": "False",
    "xla_tpu_enable_windowed_einsum_for_all_gather": "False",
    "xla_tpu_prefer_latch_optimized_rhs_layouts": "True",
}


def dump_flags(flags_set):
  flags_str = [f"--{k}={v}" for k, v in flags_set.items()]
  return " ".join(flags_str)


print(dump_flags(xla_flags_1))
