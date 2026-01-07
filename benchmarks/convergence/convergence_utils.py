# Copyright 2023–2026 Google LLC
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
    convergence_config: ConvHParams, # Model-specific hyperparams for convergence tests (ex. deepseek_v3_ep_256_v5p_512)
    global_batch_size: int,          # The global batch size to be used
    num_devices: int,                # The number of devices available for training
)
"""
import dataclasses
import math

from benchmarks.benchmark_utils import MaxTextModel


@dataclasses.dataclass
class DatasetHParams:
  """Hyperparameters for a dataset used in convergence testing.

  Attributes:
    name: A unique name for this dataset configuration.
    dataset_path: The GCS or local path to the dataset.
    dataset_name: The name of the dataset (e.g., 'c4/en-20220301').
    dataset_type: The type of dataset, e.g., 'c4_mlperf'.
    train_split: The name of the training split to use.
    eval_split: The name of the evaluation split to use.
    eval_tokens: The number of tokens to use for evaluation.
    add_bos: Whether to add a beginning-of-sentence token.
    add_eos: Whether to add an end-of-sentence token.
    eval_dataset_name: Optional name for the evaluation dataset if different
      from the training dataset.
  """

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
  """Hyperparameters for a convergence test.

  These parameters define the training dynamics and goals for a specific
  convergence run.

  Attributes:
    global_batch_size: The total batch size across all devices.
    warmup_samples: The number of samples to use for the learning rate warmup
      phase.
    decay_end_samples: The number of samples at which the learning rate decay
      ends.
    total_tokens_to_train: The total number of tokens to train the model on.
    learning_rate: The base learning rate for the optimizer.
    eval_interval: The frequency (in number of tokens) at which to run
      evaluation.
    training_scaleing_factor: A scaling factor used in critical batch size
      calculations.
    eval_tokens: The number of tokens to use for evaluation. If -1, it uses the
      default from the model's tuning parameters.
    seeds: The random seed for data shuffling.
  """

  global_batch_size: int
  warmup_samples: int
  decay_end_samples: int
  total_tokens_to_train: int
  learning_rate: float
  eval_interval: int  # in number of tokens
  training_scaleing_factor: float  # used in critical_batch_size_scaling
  eval_tokens: int = -1
  seeds: int = 1238


def load_checkpoint(model: MaxTextModel, checkpoint_path: str):
  """Sets the full state checkpoint path on a model configuration.

  Args:
    model: The model configuration object to be modified.
    checkpoint_path: The path to the full state checkpoint to load.
  """
  model.tuning_params["load_full_state_path"] = checkpoint_path


def setup_dataset(model: MaxTextModel, params: DatasetHParams):
  """Configures a model's dataset-related parameters.

  This function modifies a `MaxTextModel`'s `tuning_params` in place, setting
  all necessary parameters for data loading and processing based on the
  provided dataset hyperparameters.

  Args:
    model: The model configuration object to be modified.
    params: A `DatasetHParams` object containing the dataset-specific
      hyperparameters.
  """
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
  """Configures a model's training parameters for a convergence test.

  This function modifies a `MaxTextModel`'s `tuning_params` in place. It
  calculates critical training settings like the number of training steps,
  learning rate schedule, and evaluation frequency based on high-level
  convergence goals (e.g., total tokens to train) and the specific hardware
  configuration (number of devices, batch size).

  Args:
    model: The model configuration object to be modified.
    params: A dataclass containing convergence-specific hyperparameters like
      total tokens, learning rate, and warmup/decay samples.
    num_devices: The total number of devices being used for training.
    global_batch_size: The total batch size across all devices.
  """
  gbs = global_batch_size
  total_steps = params.total_tokens_to_train / gbs

  warmup_steps = math.ceil(params.warmup_samples / gbs - 1e-6)
  decay_end_step = math.ceil(params.decay_end_samples / gbs - 1e-6)

  max_target_length = model.tuning_params["max_target_length"]

  model.tuning_params["per_device_batch_size"] = float(gbs / num_devices)
  model.tuning_params["learning_rate"] = params.learning_rate * gbs
  model.tuning_params["warmup_steps_fraction"] = float(warmup_steps / decay_end_step)
  model.tuning_params["learning_rate_schedule_steps"] = decay_end_step
  model.tuning_params["steps"] = int(total_steps)
  if params.eval_tokens > 0:
    eval_tokens = params.eval_tokens
  else:
    eval_tokens = model.tuning_params["eval_steps"]
  model.tuning_params["eval_steps"] = int(math.ceil(eval_tokens / max_target_length / gbs))
  model.tuning_params["eval_interval"] = int(math.ceil(params.eval_interval / max_target_length / gbs))
  model.tuning_params["data_shuffle_seed"] = params.seeds


def _setup_model_convergence_(
    maxtext_model: MaxTextModel,
    dataset: DatasetHParams,
    convergence_configs: ConvHParams,
    num_devices: int,
    global_batch_size: int,
) -> MaxTextModel:
  """Sets up a MaxText model configuration for a convergence test.

  This function takes a base model configuration and applies specific settings
  for the dataset and convergence criteria. It calculates and sets training
  parameters like learning rate, step counts, and batch sizes based on the
  provided convergence hyperparameters and hardware setup.

  Args:
    maxtext_model: The base MaxTextModel configuration, typically tuned for
      performance.
    dataset: A `DatasetHParams` object with dataset-specific hyperparameters.
    convergence_configs: A `ConvHParams` object with model-specific
      hyperparameters for the convergence test.
    num_devices: The number of devices available for training.
    global_batch_size: The global batch size to be used for the run.

  Returns:
    A new `MaxTextModel` instance configured for the specified convergence test.
  """
  convergence_model = dataclasses.replace(maxtext_model)
  setup_dataset(convergence_model, dataset)
  setup_convergence_configs(convergence_model, convergence_configs, num_devices, global_batch_size)
  convergence_model.model_name = convergence_model.model_name + "-" + dataset.name

  return convergence_model
