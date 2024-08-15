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


# Bash script to kick off a gradient accumulation run and non-accumulation run with the same GBS.

accumulation_metrics_file=$1
regular_metrics_file=$2

shared_args="configs/base.yml base_output_directory=gs://runner-maxtext-logs run_name=compile_equivalent_test dataset_path=gs://maxtext-dataset steps=3 enable_checkpointing=False base_emb_dim=256 base_num_decoder_layers=4 tokenizer_path=../assets/tokenizer.llama2 gradient_clipping_threshold=0"

python train.py $shared_args gradient_accumulation_steps=10 per_device_batch_size=1 metrics_file=$accumulation_metrics_file
python train.py $shared_args gradient_accumulation_steps=1 per_device_batch_size=10 metrics_file=$regular_metrics_file
