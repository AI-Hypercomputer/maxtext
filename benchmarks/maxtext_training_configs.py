import dataclasses
import math
import typing
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


@dataclasses.dataclass
class MaxTextModel:
  model_name: str
  model_type: str
  tuning_params: dict[str, typing.Any]
  xla_flags: str

  # Additional pathways tuning params as necessary. Adding
  # enable_single_controller=True to pathways_tuning_params is not necessary.
  pathways_tuning_params: dict[str, typing.Any] = None

  # XLA flags for pathways, if different from the default. Some flags may not
  # be supported by pathways e.g. "--2a886c8_chip_config_name".
  pathways_xla_flag_options: dict[str, typing.Any] = None
    
  def setup_dataset(self, params: DatasetHParams):
    self.tuning_params["reuse_example_batch"] = -1
    self.tuning_params["dataset_path"] = params.dataset_path
    self.tuning_params["dataset_name"] = params.dataset_name
    self.tuning_params["dataset_type"] = params.dataset_type
    self.tuning_params["dataset_name"] = params.dataset_name
    if params.eval_dataset_name:
      self.tuning_params["eval_dataset_name"] = params.eval_dataset_name
    self.tuning_params["train_split"] = params.train_split
    self.tuning_params["eval_split"] = params.eval_split
    self.tuning_params["add_bos"] = params.add_bos
    self.tuning_params["add_eos"] = params.add_eos
    self.tuning_params["eval_steps"] = params.eval_tokens
    self.tuning_params["data_shuffle_seed"] = 1238

  def setup_convergence_configs(self, params: ConvHParams, num_devices: int, global_batch_size: int):
    gbs = global_batch_size
    total_steps = params.total_tokens_to_train / gbs

    warmup_steps = math.ceil(params.warmup_samples / gbs - 1e-6)
    decay_end_step = math.ceil(params.decay_end_samples / gbs - 1e-6)

    max_target_length = self.tuning_params["max_target_length"]
    
    self.tuning_params["per_device_batch_size"] = float(gbs / num_devices)
    self.tuning_params["learning_rate"] =  params.learning_rate * gbs
    self.tuning_params["warmup_steps_fraction"] =  float(warmup_steps / decay_end_step)
    self.tuning_params["learning_rate_schedule_steps"] = decay_end_step
    self.tuning_params["steps"] =  int(total_steps)
    if params.eval_tokens > 0:
      eval_tokens  = params.eval_tokens
    else:
      eval_tokens  =  self.tuning_params["eval_steps"]
    self.tuning_params["eval_steps"] = int(math.ceil(eval_tokens / max_target_length / gbs))
    self.tuning_params["eval_interval"]= int(math.ceil(params.eval_interval / max_target_length / gbs))


# Run this for new definitions that should be part of the library.
def _add_to_model_dictionary(
    model_dictionary: dict[str, MaxTextModel], maxtext_model: MaxTextModel
) -> MaxTextModel:
  print(maxtext_model.model_name.replace("-", "_"))
  model_dictionary[maxtext_model.model_name.replace("-", "_")] = maxtext_model
  return maxtext_model

def _setup_model_convergence_(
    maxtext_model: MaxTextModel, dataset: DatasetHParams, convergence_configs: ConvHParams, num_devices: int, global_batch_size: int,
) -> MaxTextModel:
  convergence_model = dataclasses.replace(maxtext_model)
  convergence_model.setup_dataset(dataset)
  convergence_model.setup_convergence_configs(convergence_configs, num_devices, global_batch_size)
  convergence_model.model_name = convergence_model.model_name + "-" + dataset.name

  return convergence_model