@dataclasses.dataclass
class MaxTextModel:
  model_name: str
  model_type: str
  tuning_params: dict[str, typing.Any]
  xla_flags: str

@dataclasses.dataclass
class DatasetHParams:
    dataset_path: str
    dataset_name: str
    dataset_type: str
    train_split: str
    eval_split: str
    eval_steps: int
    add_bos: bool
    add_eos: bool
    tokenizer_path: str

@dataclasses.dataclass
class ConvHParams:
    global_batch_size: int
    warmup_samples: int
    decay_end_samples: int
    total_tokens_to_train: int
    learning_rate: float
    eval_interval:int