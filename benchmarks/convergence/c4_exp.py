# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This file defines specific convergence test configurations for models
pre-trained on the C4 dataset.

It sets up `DatasetHParams` for different C4 variants and `ConvHParams` for
specific models like Llama3-405B and DeepSeek-671B to define their training
and evaluation parameters for convergence analysis.
"""

from benchmarks.benchmark_utils import MaxTextModel, _add_to_model_dictionary
from benchmarks.convergence.convergence_utils import DatasetHParams, ConvHParams, _setup_model_convergence_

from benchmarks.maxtext_v5p_model_configs import deepseek_v3_ep_256_v5p_512

c4_pretrain_model_dict = {}

c4_mlperf_hp = DatasetHParams(
  name="c4mlperf",
  dataset_path="gs://max-datasets-rogue",
  dataset_name="c4/en:3.0.7",
  eval_dataset_name="c4/en:3.0.9",
  dataset_type="c4_mlperf",
  train_split="train2",
  eval_split="validation",
  eval_tokens=47185920,  # 5760*8192 training_tokens, special requirement from mlperf
  add_bos=False,
  add_eos=False,
)

c4_en_hp = DatasetHParams(
  name="c4en",
  dataset_path="gs://maxtext-dataset",
  dataset_name="c4/en:3.0.1",
  dataset_type="tfds",
  train_split="train",
  eval_split="validation",
  eval_tokens=75497472,
  add_bos=False,
  add_eos=False,
)

c4_mutil_hp = DatasetHParams(
  name="c4multi",
  dataset_path="gs://mlperf-llm-public2",
  dataset_name="c4/multilingual:3.1.0",
  dataset_type="tfds",
  train_split="en",
  eval_split="en-validation",
  eval_tokens=824 * 512,  # 824 * 512
  add_bos=False,
  add_eos=False,
)

llama3_405b_hp = ConvHParams(
  global_batch_size=1152,
  warmup_samples=8216000,
  decay_end_samples=1382400000.0,
  total_tokens_to_train=2.64e9,
  training_scaleing_factor=1.0,
  learning_rate=6.944e-8,
  eval_tokens=47185920,
  eval_interval=377487360,
)

# [todo] reuse 405b convergence benchmark hp for now. not tuned yet
deepseek_671b_hp = ConvHParams(
  global_batch_size=1152,
  warmup_samples=8216000,
  decay_end_samples=1382400000.0,
  total_tokens_to_train=2.64e9,
  training_scaleing_factor=1.0,
  learning_rate=6.944e-8,
  eval_tokens=47185920,
  eval_interval=377487360,
)


def load_checkpoint(model: MaxTextModel, checkpoint_path: str):
  model.tuning_params["load_full_state_path"] = checkpoint_path


# Run this for new definitions that should be part of the library.
c4_deepseek_v3_ep_256_v5p_512_gbs_1024 = _add_to_model_dictionary(
  c4_pretrain_model_dict,
  _setup_model_convergence_(
    deepseek_v3_ep_256_v5p_512,
    c4_mlperf_hp,
    deepseek_671b_hp,
    global_batch_size=1024,
    num_devices=256,
  ),
)
