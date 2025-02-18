"""
 Copyright 2023 Google LLC
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      https://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

"""Utils that relevant to mllog for mlperf submission compliance."""
import jax
from mlperf_logging import mllog

mllogger = mllog.get_mllogger()

def init_start():
  if jax.process_index() == 0:
    mllogger.event(mllog.constants.CACHE_CLEAR)
    mllogger.start(mllog.constants.INIT_START)

def init_stop():
  if jax.process_index() == 0:
    mllogger.end(mllog.constants.INIT_STOP)

def run_start():
  if jax.process_index() == 0:
    mllogger.start(mllog.constants.RUN_START)

def block_start(config):
  if jax.process_index() == 0:
    eval_frequency_tokens = config.eval_interval * config.global_batch_size_to_train_on * config.max_target_length
    mllogger.start(
      mllog.constants.BLOCK_START,
      metadata={
        'epoch_count': eval_frequency_tokens,
        'first_epoch_num': 0,
      },
    )

def init_print(config, start_step):
  """an initial mllog for mlperf sumbission compliance check."""
  if jax.process_index() == 0:
    mllogger.event(mllog.constants.SUBMISSION_ORG, 'Google')
    mllogger.event(mllog.constants.SUBMISSION_PLATFORM, 'tpu-v5p')
    mllogger.event(mllog.constants.SUBMISSION_STATUS, mllog.constants.CLOUD)
    mllogger.event(mllog.constants.SUBMISSION_DIVISION, mllog.constants.CLOSED)
    mllogger.event(mllog.constants.SUBMISSION_BENCHMARK, mllog.constants.GPT3)
    mllogger.event(mllog.constants.OPT_NAME, mllog.constants.ADAM)
    mllogger.event(mllog.constants.OPT_BASE_LR, config.learning_rate)
    mllogger.event(mllog.constants.OPT_END_LR, config.cosine_learning_rate_final_fraction)
    mllogger.event(mllog.constants.OPT_WEIGHT_DECAY, config.adam_weight_decay)
    mllogger.event(mllog.constants.OPT_LR_DECAY_STEPS,
                   int(config.learning_rate_schedule_steps * (1 - config.warmup_steps_fraction)))
    mllogger.event(mllog.constants.OPT_LR_WARMUP_STEPS,
                   int(config.learning_rate_schedule_steps * config.warmup_steps_fraction + 1))
    mllogger.event(mllog.constants.OPT_LR_DECAY_SCHEDULE, 'cosine with linear warmup')
    mllogger.event(mllog.constants.INIT_CHECKPOINT_STEP, start_step)
    mllogger.event(mllog.constants.OPT_ADAM_BETA_1, config.adam_b1)
    mllogger.event(mllog.constants.OPT_ADAM_BETA_2, config.adam_b2)
    mllogger.event(mllog.constants.OPT_ADAM_EPSILON, config.adam_eps)
    mllogger.event(mllog.constants.OPT_GRADIENT_CLIP_NORM, config.gradient_clipping_threshold)
    mllogger.event(mllog.constants.GLOBAL_BATCH_SIZE, config.global_batch_size_to_train_on)
    mllogger.event(mllog.constants.MAX_SEQUENCE_LENGTH, config.max_target_length)
    mllogger.event(mllog.constants.GRADIENT_ACCUMULATION_STEPS, 1)
    mllogger.event(mllog.constants.EVAL_SAMPLES, 24567)
    mllogger.event(mllog.constants.SEED, config.data_shuffle_seed)

def early_stop_check(config, step, eval_loss, start_step):
  """an early stop function with mllog for mlperf sumbission compliance check."""
  is_early_stop = eval_loss <= config.target_eval_loss
  if jax.process_index() == 0:
    eval_frequency_tokens = config.eval_interval * config.global_batch_size_to_train_on * config.max_target_length
    current_epoch_num = (step - start_step) * config.global_batch_size_to_train_on * config.max_target_length
    first_epoch_num = current_epoch_num - eval_frequency_tokens
    mllogger.end(
      mllog.constants.BLOCK_STOP,
      metadata={'first_epoch_num': first_epoch_num},
    )
    mllogger.event(
      mllog.constants.EVAL_ACCURACY,
      eval_loss,
      metadata={'epoch_num': current_epoch_num},
    )
    if is_early_stop:
      mllogger.end(mllog.constants.RUN_STOP, metadata={'status': 'success'})
      mllogger.event(mllog.constants.TRAIN_SAMPLES, current_epoch_num)
    else:
      mllogger.start(
        mllog.constants.BLOCK_START,
        metadata={
          'epoch_count': eval_frequency_tokens,
          'first_epoch_num': current_epoch_num,
        },
      )
