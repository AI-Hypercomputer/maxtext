"""Config for training."""

from typing import Any, Sequence, Tuple
from flax import struct
import jax.numpy as jnp


@struct.dataclass
class T5Config:
  """Global hyperparameters used to minimize obnoxious kwarg plumbing."""

  run_name: str = "my_run_008" 

  # Activation dtypes.
  dtype: Any = jnp.bfloat16
  emb_dim: int = 2048
  num_heads: int = 1
  head_dim: int = 2048
  mlp_dim: int = 4096
  num_decoder_layers: int = 6
  # activation functions are .
  mlp_activations: Sequence[str] = ('relu',)
  dropout_rate: float = 0
  # If `True`, the embedding weights are used in the decoder output layer.
  logits_via_embedding: bool = True  # NOTE: this is True just for testing.
  # minimal, full, or none
  remat_policy: str = 'none'
  scan_layers: bool = False
  param_scan_axis: int = 1

  # Parallelism
  mesh_shape: Tuple[int] = (4,)
  mesh_axes: Tuple[str] = ('data',)
  logical_axis_rules: Sequence = ( ('batch', 'data'), )

  # Dataset
  vocab_size: int = 30000
  vocab_path: str = "gs://cloudtpu_internal_datasets/vocabs/"  # Assumes we're allowed
  dataset_name: str = 'lm1b'
  eval_dataset_name: str = 'lm1b'
  eval_split: str = 'test'
  per_device_batch_size: int = 32
  eval_per_device_batch_size: int = 0
  max_corpus_chars: int = 10**7  # for tokenization

  # Training loop
  steps: int = 20_000
  log_period: int = 100
  save_period: int = 2000
  learning_rate: float = 1e-5
  warmup_steps: int = 2000
  save_checkpoints: bool = False
  restore_checkpoints: bool = False

  # Maximum length cutoff for training examples.
  max_target_length: int = 128
  # Maximum length cutoff for held-out evaluation examples.
  max_eval_target_length: int = 512

  # Maximum length cutoff for predicted tokens.
  max_predict_length: int = 50
  # Sampling temperature for language model inference.
  sampling_temperature: float = 0.6
  # Top k cutoff for logit sampling. If 0 then no top-k cutoff is used.
  sampling_top_k: int = 20
  eos_id: int = 2  # sentencepiece default
  # Prompt for language model sampling.
  prompt: str = "I love to "

