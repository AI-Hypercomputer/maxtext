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

"""  This file contains data classes and convergence setup logic to config convergence testing

new_config = _setup_model_convergence_(
    model_config: MaxTextModel,    # Your existing, performance-tuned model configuration
    dataset_config: DatasetHParams,  # Dataset-specific hyperparameters (e.g., c4_mlperf_hp)
    convergence_config: ConvHParams, # Model-specific hyperparameters for convergence testing (e.g. deepseek_v3_ep_256_v5p_512)
    global_batch_size: int,          # The global batch size to be used
    num_devices: int,                # The number of devices available for training
)
"""
import dataclasses
import math
from benchmarks.benchmark_utils import _add_to_model_dictionary, MaxTextModel

@dataclasses.dataclass
class DatasetHParams:
    name: str
    dataset_path: str
    dataset_name: str
    dataset_type: str
    train_split: str
    eval_split: str
    eval_tokens: int
    add_bos: bool
    add_eos: bool
    eval_dataset_name: str = None

@dataclasses.dataclass
class ConvHParams:
    global_batch_size: int
    warmup_samples: int
    decay_end_samples: int
    total_tokens_to_train: int
    learning_rate: float
    eval_interval: int # in number of tokens
    training_scaleing_factor: float # used in critical_batch_size_scaling
    eval_tokens: int = -1
    seeds: int = 1238


def load_checkpoint(model: MaxTextModel, checkpoint_path: str):
    model.tuning_params["load_parameters_path"] = checkpoint_path
    model.tuning_params["enable_checkpointing"] = True


def setup_dataset(model: MaxTextModel, params: DatasetHParams):
    model.tuning_params["reuse_example_batch"] = 0
    model.tuning_params["dataset_path"] = params.dataset_path
    model.tuning_params["dataset_name"] = params.dataset_name
    model.tuning_params["dataset_type"] = params.dataset_type
    model.tuning_params["dataset_name"] = params.dataset_name
    if params.eval_dataset_name:
        model.tuning_params["eval_dataset_name"] = params.eval_dataset_name
    model.tuning_params["train_split"] = params.train_split
    model.tuning_params["eval_split"] = params.eval_split
    model.tuning_params["add_bos"] = params.add_bos
    model.tuning_params["add_eos"] = params.add_eos
    model.tuning_params["eval_steps"] = params.eval_tokens

def setup_convergence_configs(model, params: ConvHParams, num_devices: int, global_batch_size: int):
    gbs = global_batch_size
    total_steps = params.total_tokens_to_train / gbs

    warmup_steps = math.ceil(params.warmup_samples / gbs - 1e-6)
    decay_end_step = math.ceil(params.decay_end_samples / gbs - 1e-6)

    max_target_length = model.tuning_params["max_target_length"]

    model.tuning_params["per_device_batch_size"] = float(gbs / num_devices)
    model.tuning_params["learning_rate"] =  params.learning_rate * gbs
    model.tuning_params["warmup_steps_fraction"] =  float(warmup_steps / decay_end_step)
    model.tuning_params["learning_rate_schedule_steps"] = decay_end_step
    model.tuning_params["steps"] =  int(total_steps)
    if params.eval_tokens > 0:
        eval_tokens  = params.eval_tokens
    else:
        eval_tokens  =  model.tuning_params["eval_steps"]
    model.tuning_params["eval_steps"] = int(math.ceil(eval_tokens / max_target_length / gbs))
    model.tuning_params["eval_interval"]= int(math.ceil(params.eval_interval / max_target_length / gbs))
    model.tuning_params["data_shuffle_seed"] = params.seeds

def _setup_model_convergence_(
    maxtext_model: MaxTextModel, dataset: DatasetHParams, convergence_configs: ConvHParams, num_devices: int, global_batch_size: int, checkpoint: str=None,
) -> MaxTextModel:
  convergence_model = dataclasses.replace(maxtext_model)
  setup_dataset(convergence_model, dataset)
  setup_convergence_configs(convergence_model, convergence_configs, num_devices, global_batch_size)
  convergence_model.model_name = convergence_model.model_name + "-" + dataset.name
  if not checkpoint is None:
    load_checkpoint(convergence_model, checkpoint)

  return convergence_model