# Copyright 2024 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Bash script to run AOT and real runs (on a v4-8)
# This needs to run via a bash script so the AOT and real runs each
# initialize jax/XLA separtely (e.g. separate dump directories)
# and we do not contaminate the XLA flags of the second run with the first.

compile_dump_dir=$1
train_dump_dir=$2
custom_args=$3

shared_args="configs/base.yml base_output_directory=gs://runner-maxtext-logs run_name=compile_equivalent_test dataset_path=gs://maxtext-dataset dataset_type=synthetic steps=5 enable_checkpointing=False $custom_args"
aot_args="compile_topology=v4-8 compile_topology_num_slices=1"

export XLA_FLAGS=--xla_dump_to=${compile_dump_dir}
python3 train_compile.py $shared_args $aot_args 

export XLA_FLAGS=--xla_dump_to=${train_dump_dir}
python train.py $shared_args

