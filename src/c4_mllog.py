"""GPT-3 models with MLPerf Logging.

copied from
google3/third_party/tensorflow_models/mlperf/models/rough/gpt3/c4_mllog.py
"""
import logging as std_logging
from typing import Dict, Optional

import jax
from mlperf_logging.mllog import mllog
from paxml import experiment_registry
from paxml import tasks_lib
from paxml import trainer_lib
from paxml.tasks.lm.params import c4
from praxis import base_hyperparams
from praxis import pax_fiddle

DEFAULT_EVAL_METRICS = 'eval_test_C4Validation/metrics/log_pplx'


class EarlyStoppingFn(base_hyperparams.FiddleBaseParameterizable):
  """Early stopping function to log eval log_pplx and stop when reaching target.

  Attributes:
    target_log_pplx: target log pplx value to stop training when eval log pplx
      reaches this value.
    global_batch_size_tokens: global batch size in terms of tokens.
    eval_frequency_tokens: eval frequency in terms of tokens.
  """

  target_log_pplx: Optional[float] = None
  global_batch_size_tokens: Optional[int] = None
  eval_frequency_tokens: Optional[int] = None

  def __call__(
      self,
      metrics: Dict[str, float],
      running_mode: trainer_lib.RunningMode,
      global_step: int,
      is_last_ckpt: bool,
  ) -> bool:
    """Returns True if run should be stopped early."""
    if DEFAULT_EVAL_METRICS not in metrics.keys():
      return False

    if not hasattr(self, '_mllogger'):
      self._mllogger = mllog.MLLogger(
          logger=std_logging.getLogger('mllog_default')
      )

    current_epoch_num = global_step * self.hparams.global_batch_size_tokens
    first_epoch_num = current_epoch_num - self.hparams.eval_frequency_tokens
    log_pplx = metrics[DEFAULT_EVAL_METRICS]
    self._mllogger.end(
        'block_stop', metadata={'first_epoch_num': first_epoch_num}
    )
    self._mllogger.event(
        'eval_accuracy',
        log_pplx,
        metadata={'epoch_num': current_epoch_num},
    )
    if log_pplx <= self.hparams.target_log_pplx:
      self._mllogger.end('run_stop', metadata={'status': 'success'})
      self._mllogger.event('train_samples', current_epoch_num)
      return True

    self._mllogger.start(
        'block_start',
        metadata={
            'epoch_count': self.hparams.eval_frequency_tokens,
            'first_epoch_num': current_epoch_num,
        },
    )
    return False


def log_mlperf_params(cls, mllogger, task_p) -> None:
  r"""Log MLPerf benchmark configurations."""
  global_batch_size = int(cls.PERCORE_BATCH_SIZE * jax.device_count() + 1e-6)
  assert global_batch_size > 0
  if hasattr(cls, 'NUM_MICROBATCHES'):
    if cls.NUM_MICROBATCHES is not None:
      gradient_accumulation_steps = cls.NUM_MICROBATCHES
    else:
      gradient_accumulation_steps = global_batch_size // cls.MICROBATCH_SIZE
  else:
    gradient_accumulation_steps = 1

  mllogger.event('submission_org', 'Google')
  mllogger.event(
      'submission_platform',
      'tpu-v5l-%d' % (jax.device_count()),
  )
  mllogger.event('submission_status', 'cloud')
  mllogger.event('submission_division', 'closed')
  mllogger.event('submission_benchmark', 'gpt3')

  optimizer_p = task_p.train.learner.optimizer
  mllogger.event('opt_name', 'adam')
  mllogger.event('opt_base_learning_rate', optimizer_p.learning_rate)
  mllogger.event('opt_end_learning_rate', optimizer_p.lr_schedule.min_ratio)
  mllogger.event('opt_weight_decay', optimizer_p.weight_decay)
  mllogger.event(
      'opt_learning_rate_decay_steps',
      optimizer_p.lr_schedule.decay_end
      - optimizer_p.lr_schedule.decay_start
      + 1,
  )
  mllogger.event(
      'opt_learning_rate_warmup_steps', optimizer_p.lr_schedule.decay_start
  )
  if cls.LR_SCHEDULE == 'linear_rampup_cosine_decay':
    mllogger.event(
        'opt_learning_rate_decay_schedule', 'cosine with linear warmup'
    )
  else:
    raise ValueError(f'wrong lr_schedule={cls.LR_SCHEDULE}')

  mllogger.event('init_checkpoint_step', int(1536 * 4000 / global_batch_size))
  mllogger.event('opt_adam_beta_1', optimizer_p.beta1)
  mllogger.event('opt_adam_beta_2', optimizer_p.beta2)
  mllogger.event('opt_adam_epsilon', optimizer_p.epsilon)
  mllogger.event(
      'opt_gradient_clip_norm', optimizer_p.clip_gradient_norm_to_value
  )

  mllogger.event('global_batch_size', global_batch_size)
  mllogger.event('sequence_length', cls.MAX_SEQ_LEN)
  mllogger.event('max_sequence_length', cls.MAX_SEQ_LEN)
  mllogger.event('gradient_accumulation_steps', gradient_accumulation_steps)
  mllogger.event('eval_samples', 24567)


def try_to_init_mlloger(cls):
  r"""Initialize mllogger and log initial logs."""
  if hasattr(cls, 'logged_params'):
    assert cls.logged_params
    return None

  mllogger = mllog.MLLogger()
  mllogger.event('cache_clear')
  mllogger.start('init_start')
  return mllogger


def log_params_and_set_early_stopping_fn(
    cls, task_p, mllogger
) -> pax_fiddle.Config[tasks_lib.SingleTask]:
  r"""Log MLPerf parames and set early stopping fn with logging."""
  if mllogger is not None:
    assert not hasattr(cls, 'logged_params')
    log_mlperf_params(cls, mllogger, task_p)
    cls.logged_params = True

  global_batch_size = int(cls.PERCORE_BATCH_SIZE * jax.device_count() + 1e-6)
  assert global_batch_size > 0
  task_p.early_stopping_fn = pax_fiddle.Config(EarlyStoppingFn)
  task_p.early_stopping_fn.target_log_pplx = cls.TARGET_LOG_PPLX
  task_p.early_stopping_fn.global_batch_size_tokens = (
      global_batch_size * cls.MAX_SEQ_LEN
  )
  task_p.early_stopping_fn.eval_frequency_tokens = (
      cls.EVAL_INTERVAL_STEPS * global_batch_size * cls.MAX_SEQ_LEN
  )
  return task_p


@experiment_registry.register
class C4SpmdPipelineGpt3AdamMLPerfHPBS1p5k768ReplicasMllog(
    c4.C4SpmdPipelineGpt3AdamMLPerfHPBS1p5k768Replicas
):
  r"""GPT-3 config in fp32 for 768 replicas with 1536 global batch size.

  Following MLPerf benchmarking HP requirements, with MLPerf logging.
  """

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    mllogger = try_to_init_mlloger(self)
    task_p = log_params_and_set_early_stopping_fn(
        self, super().task(), mllogger
    )
    return task_p


@experiment_registry.register
class C4SpmdPipelineGpt3AdamMLPerfHPBS2k512ReplicasMllog(
    c4.C4SpmdPipelineGpt3AdamMLPerfHPBS2k512Replicas
):
  r"""GPT-3 config in fp32 for 512 replicas with 2048 global batch size.

  Following MLPerf benchmarking HP requirements, with MLPerf logging.
  """

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    mllogger = try_to_init_mlloger(self)
    task_p = log_params_and_set_early_stopping_fn(
        self, super().task(), mllogger
    )
    return task_p


@experiment_registry.register
class C4SpmdPipelineGpt3AdamMLPerfHPBS3k768ReplicasMllog(
    c4.C4SpmdPipelineGpt3AdamMLPerfHPBS3k768Replicas
):
  r"""GPT-3 config in fp32 for 768 replicas with 3072 global batch size.

  Following MLPerf benchmarking HP requirements, with MLPerf logging.
  """

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    mllogger = try_to_init_mlloger(self)
    task_p = log_params_and_set_early_stopping_fn(
        self, super().task(), mllogger
    )
    return task_p


@experiment_registry.register
class C4SpmdPipelineGpt3AdamMLPerfHPBS4k1024ReplicasMllog(
    c4.C4SpmdPipelineGpt3AdamMLPerfHPBS4k1024Replicas
):
  r"""GPT-3 config in fp32 for 1024 replicas with 4096 global batch size.

  Following MLPerf benchmarking HP requirements, with MLPerf logging.
  """

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    mllogger = try_to_init_mlloger(self)
    task_p = log_params_and_set_early_stopping_fn(
        self, super().task(), mllogger
    )
    return task_p


@experiment_registry.register
class C4SpmdPipelineGpt3AdamMLPerfHPBS8k1024ReplicasMllog(
    c4.C4SpmdPipelineGpt3AdamMLPerfHPBS8k1024Replicas
):
  r"""GPT-3 config in fp32 for 1024 replicas with 8192 global batch size.

  Following MLPerf benchmarking HP requirements, with MLPerf logging.
  """

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    mllogger = try_to_init_mlloger(self)
    task_p = log_params_and_set_early_stopping_fn(
        self, super().task(), mllogger
    )
    return task_p


@experiment_registry.register
class C4SpmdPipelineGpt3SmallAdam8ReplicasMllog(
    c4.C4SpmdPipelineGpt3SmallAdam8Replicas
):
  """Small GPT-3 config in bf16 for 8 replicas with 512 global batch size.

  This was called GPT-3 XL in the GPT-3 paper, with 1.3B parameters,
  with MLPerf logging.
  """

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    mllogger = try_to_init_mlloger(self)
    task_p = log_params_and_set_early_stopping_fn(
        self, super().task(), mllogger
    )
    return task_p