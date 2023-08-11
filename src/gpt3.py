"""GPT3 MLPerf configurations on the T5/C4 dataset.

copied from
google3/third_party/tensorflow_models/mlperf/models/rough/gpt3/gpt3.py
"""
from absl import flags
from absl import logging
import jax
from jax import numpy as jnp
from mlperf_logging import mllog
from paxml import base_executor
from paxml import executors
from paxml import experiment_registry
from paxml import programs
from paxml import tasks_lib
from paxml import trainer_lib
from paxml.tasks.lm import input_generator
from paxml.tasks.lm.params import c4
from paxml.tasks.lm.params import c4_mllog
from paxml.tasks.lm.params import quant
from praxis import base_hyperparams
from praxis import base_input
from praxis import layers
from praxis import pax_fiddle
from praxis.layers import transformers

instantiate = base_hyperparams.instantiate
BaseStepFnStaticArgs = trainer_lib.BaseStepFnStaticArgs

# Flags can be removed when Fiddle support is available in Pax (go/pax-rfc-52)
_HYBRIDSIM_PERCORE_BATCH_SIZE = flags.DEFINE_float(
    'hybridsim_percore_batch_size',
    None,
    'If set, override the default PERCORE_BATCH_SIZE in HybridSim experiments',
)
_HYBRIDSIM_ICI_MESH_SHAPE = flags.DEFINE_string(
    'hybridsim_ici_mesh_shape',
    '',
    'If set, override the default ICI_MESH_SHAPE in HybridSim experiments',
)
_HYBRIDSIM_MICROBATCH_SIZE_POWER = flags.DEFINE_integer(
    'hybridsim_microbatch_size_power',
    None,
    'If set, override the default MICROBATCH_SIZE to '
    '2**hybridsim_microbatch_size_power in HybridSim experiments',
)
_HYBRIDSIM_CIRCULAR_REPEAT = flags.DEFINE_integer(
    'hybridsim_circular_repeat',
    None,
    'If set, override the default CIRCULAR_REPEAT in HybridSim experiments',
)
_MLPERF_GPT3_CHECKPOINT_EVERY_N_STEPS = flags.DEFINE_integer(
    'mlperf_gpt3_checkpoint_every_n_steps',
    2000,
    'Steps between saving checkpoints.',
)
_MLPERF_GPT3_CHECKPOINT_MAX_TO_KEEP = flags.DEFINE_integer(
    'mlperf_gpt3_checkpoint_max_to_keep',
    2,
    'Number of checkpoints to keep.',
)
_MLPERF_GPT3_TRAINING_SEED = flags.DEFINE_integer(
    'mlperf_gpt3_training_seed',
    None,
    'If set, override the training input seed.',
)
_MLPERF_GPT3_TRAINING_NUM_BATCHES_TO_SKIP = flags.DEFINE_integer(
    'mlperf_gpt3_training_num_batches_to_skip',
    None,
    'If set, skip given number of training input batches.',
)
_MLPERF_GPT3_SUMMARY_VERBOSITY = flags.DEFINE_integer(
    'mlperf_gpt3_summary_verbosity',
    3,
    'Set to zero to disable summary computation.',
)


class MLPerfExecutor(executors.DefaultExecutor):
  """MLPerf executor."""

  _HAS_DYNAMIC_ATTRIBUTES = True  # silence pytype

  def recreate_partitioned_train_state(self) -> programs.TrainState:
    train_input_for_checkpoint = (
        self._train_input_pipeline
        if self._task.hparams.train.enable_input_checkpointing
        else None
    )
    root_prng_key = jax.random.PRNGKey(self._task.train.random_seed)
    (partitioned_train_state, _, _, _) = self._checkpointer.get_model_states(
        self._partitioner,
        self._partitioner.get_train_state_metadata(),
        root_prng_key,
        train_input_for_checkpoint,
    )
    return partitioned_train_state


class MLPerfTrainProgram(programs.SingleTaskTrainProgram):
  """MLPerf train program with magic to trigger compilation with dummy data.

  The first call to .run() will trigger compilation with dummy data, for both
  train program and eval programs. The eval programs and a bunch of information
  are retrieved from the Executor passed-in during initialization.
  """

  _HAS_DYNAMIC_ATTRIBUTES = True  # silence pytype

  def __init__(
      self,
      dummy_train_input: base_input.BaseInput,
      dummy_eval_input: base_input.BaseInput,
      executor: MLPerfExecutor,
  ):
    super().__init__()
    self._dummy_train_input = dummy_train_input
    self._dummy_eval_input = dummy_eval_input
    self._is_compiled = False
    self._executor = executor

  def run(
      self, state: programs.TrainState, train_step: int
  ) -> programs.TrainProgramOutput:
    # pylint: disable=protected-access
    assert (
        self._executor._train_program is self
    ), 'executor.setup() should have been called already.'
    assert (
        self._partitioner is not None
    ), 'train_program.setup() should have been called already.'
    if not self._is_compiled:
      logging.info('Triggering train compile using dummy data')
      dummy_model_inputs = self._dummy_train_input.get_next_padded()
      dummy_model_inputs = self._partitioner.preprocess_inputs(
          self._dummy_train_input,
          dummy_model_inputs,
          self.train_input_partition_spec(dummy_model_inputs),
      )
      (_, new_state, _) = self.train_step(
          train_step,
          state,
          self._train_prng_seed,
          dummy_model_inputs,
          BaseStepFnStaticArgs(
              unpadded_global_batch_size=self._train_unpadded_global_batch_size
          ),
      )
      logging.info('Done Triggering train compile using dummy data')
      logging.info('Triggering eval compile using dummy data')
      eval_partitioned_state = programs.get_eval_train_state(
          self._task, new_state, self._task.train.eval_use_ema_states
      )
      for eval_program in self._executor._eval_programs:  # pylint: disable=not-an-iterable
        # monkey patch _eval_prng_seed is needed in _run_eval_loop
        eval_program._eval_prng_seed = self._executor._eval_prng_seed
        # monkey patch eval_program._eval_input_pipeline with dummy data
        original_eval_input, eval_program._eval_input_pipeline = (
            eval_program._eval_input_pipeline,
            self._dummy_eval_input,
        )
        eval_program._run_eval_loop(eval_partitioned_state)
        # restore original input
        eval_program._eval_input_pipeline = original_eval_input

      del state, new_state, eval_partitioned_state
      logging.info('Done Triggering eval compile using dummy data')
      # We must recreate the initial state as its buffer is donated.
      logging.info('Recreating initial train state.')
      state = self._executor.recreate_partitioned_train_state()
      self._is_compiled = True
      # FIXME(zqfeng, sgpyc): Use the mllog object
      mllogger = mllog.get_mllogger()
      mllogger.end(key=mllog.constants.INIT_STOP)
      mllogger.start(key=mllog.constants.RUN_START)
    # pylint: enable=protected-access
    return super().run(state, train_step)


@experiment_registry.register
class C4SpmdGpt3AdamOrgHP3K(c4.C4SpmdGpt3AdamOrgHP):
  r"""GPT-3 config with 1.5k batch siz for 3K chips."""
  VOCAB_SIZE = 51200
  SUMMARY_INTERVAL_STEPS = 16
  EVAL_INTERVAL_STEPS = 16

  PERCORE_BATCH_SIZE = 0.5
  ICI_MESH_SHAPE = [1, 256, 12]

  @property
  def CHECKPOINT_EVERY_N_STEPS(self) -> int:
    value = super().CHECKPOINT_EVERY_N_STEPS
    if _MLPERF_GPT3_CHECKPOINT_EVERY_N_STEPS.value:
      value = int(_MLPERF_GPT3_CHECKPOINT_EVERY_N_STEPS.value)
    return value

  @property
  def CHECKPOINT_MAX_TO_KEEP(self) -> int:
    value = super().CHECKPOINT_MAX_TO_KEEP
    if _MLPERF_GPT3_CHECKPOINT_MAX_TO_KEEP.value:
      value = int(_MLPERF_GPT3_CHECKPOINT_MAX_TO_KEEP.value)
    return value

  @property
  def TRAINING_SEED(self) -> int:
    value = super().TRAINING_SEED
    if _MLPERF_GPT3_TRAINING_SEED.value:
      value = int(_MLPERF_GPT3_TRAINING_SEED.value)
    return value

  @property
  def TRAINING_NUM_BATCHES_TO_SKIP(self) -> int:
    value = super().TRAINING_NUM_BATCHES_TO_SKIP
    if _MLPERF_GPT3_TRAINING_NUM_BATCHES_TO_SKIP.value:
      value = int(_MLPERF_GPT3_TRAINING_NUM_BATCHES_TO_SKIP.value)
    return value


@experiment_registry.register
class C4SpmdGpt3AdamOrgHPVLPTest(c4.C4SpmdGpt3AdamOrgHP):
  r"""GPT-3 config to test AOT compilation on a 16x16 VLP pod."""
  VOCAB_SIZE = 51200
  PERCORE_BATCH_SIZE = 1  # 256 global batch size
  NUM_LAYERS = 32
  ICI_MESH_SHAPE = [1, 16, 16]

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    task_p = super().task()
    task_p.train.num_train_steps = 100
    return task_p


class C4SpmdGpt3AdamMLPerfHP(c4.C4SpmdGpt3AdamOrgHP):
  r"""GPT-3 config for MLPerf submission."""
  VOCAB_SIZE = 51200
  FPROP_DTYPE = jnp.bfloat16
  SUMMARY_INTERVAL_STEPS = 1000000
  # subclass must set the eval and the checkpoint intervals
  EVAL_INTERVAL_STEPS = None

  # Let set_adam_and_learning_rate_schedule calculate the following HPs
  # based on global batch size
  LEARNING_RATE = None
  LR_COS_WARMUP = None
  LR_COS_DECAY_START = None
  LR_COS_DECAY_END = None

  PROFILER_CAPTURE_STEP = 2
  PROFILER_MIN_DURATION_SEC = 100
  PROFILER_MAX_NUM_HOSTS = 4

  USE_DUMMY_DATA_FOR_COMPILE = True

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    if self.USE_DUMMY_DATA_FOR_COMPILE:
      # FIXME(sgpyc, zqfeng): Change to actual dummy datasets
      dummy_train_dataset, dummy_eval_dataset = self.datasets()
      self._executor = MLPerfExecutor()
      self._train_program = MLPerfTrainProgram(
          instantiate(dummy_train_dataset),
          instantiate(dummy_eval_dataset),
          self._executor,
      )

  @property
  def CHECKPOINT_EVERY_N_STEPS(self) -> int:
    value = super().CHECKPOINT_EVERY_N_STEPS
    if _MLPERF_GPT3_CHECKPOINT_EVERY_N_STEPS.value:
      value = int(_MLPERF_GPT3_CHECKPOINT_EVERY_N_STEPS.value)
    return value

  @property
  def CHECKPOINT_MAX_TO_KEEP(self) -> int:
    value = super().CHECKPOINT_MAX_TO_KEEP
    if _MLPERF_GPT3_CHECKPOINT_MAX_TO_KEEP.value:
      value = int(_MLPERF_GPT3_CHECKPOINT_MAX_TO_KEEP.value)
    return value

  @property
  def TRAINING_SEED(self) -> int:
    value = super().TRAINING_SEED
    if _MLPERF_GPT3_TRAINING_SEED.value:
      value = int(_MLPERF_GPT3_TRAINING_SEED.value)
    return value

  @property
  def TRAINING_NUM_BATCHES_TO_SKIP(self) -> int:
    value = super().TRAINING_NUM_BATCHES_TO_SKIP
    if _MLPERF_GPT3_TRAINING_NUM_BATCHES_TO_SKIP.value:
      value = int(_MLPERF_GPT3_TRAINING_NUM_BATCHES_TO_SKIP.value)
    return value

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    mllogger = c4_mllog.try_to_init_mlloger(self)
    task_p = c4_mllog.log_params_and_set_early_stopping_fn(
        self, super().task(), mllogger
    )
    task_p.summary_verbosity = int(_MLPERF_GPT3_SUMMARY_VERBOSITY.value)
    if self.TRAINABLE_POSITION_EMB:
      pos_emb_tpl = task_p.model.lm_tpl.position_emb_tpl
      # Avoid all-gathering the position_emb.
      wp = pos_emb_tpl.weight_split_dims_mapping
      if wp.wt is not None:
        wp.wt = [None, wp.wt[1]]
    return task_p

  def executor(self) -> base_executor.BaseExecutor:
    if self.USE_DUMMY_DATA_FOR_COMPILE:
      return self._executor
    else:
      return super().executor()

  def train_program(self) -> programs.BaseTrainProgram:
    if self.USE_DUMMY_DATA_FOR_COMPILE:
      return self._train_program
    else:
      return super().train_program()


@experiment_registry.register
class C4SpmdGpt3AdamDataParallel(c4.C4SpmdGpt3AdamOrgHP):
  r"""Cross-slice data-parallel GPT-3 config."""
  # https://xprof.corp.google.com/overview_page/jlwei-5458704630227267265
  PERCORE_BATCH_SIZE = 0.5  # 512 global batch size
  ICI_MESH_SHAPE = [1, 16, 16]
  DCN_MESH_SHAPE = [4, 1, 1]
  FPROP_DTYPE = jnp.bfloat16
  QUANTIZATION = None

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    task_p = super().task()
    if self.QUANTIZATION is not None:
      model_p = task_p.model
      quant.apply_quantized_layers_sharded(model_p, self.QUANTIZATION)
    return task_p


@experiment_registry.register
class C4SpmdGpt3AdamDataParallelMLPerfHPBS2k(C4SpmdGpt3AdamMLPerfHP):
  # 276 steps to 2.69, and ~35.8 secs / step.
  # https://tensorboard.corp.google.com/experiment/1724103233260043520
  # http://xprof/?session_id=sgpyc-8108798425837854114
  r"""Cross-slice data-parallel GPT-3 config."""
  PERCORE_BATCH_SIZE = 2  # 2048 global batch size
  ICI_MESH_SHAPE = [1, 16, 16]
  DCN_MESH_SHAPE = [4, 1, 1]
  EVAL_INTERVAL_STEPS = 12
  QUANTIZATION = None
  PERCORE_EVAL_BATCH_SIZE = 1.5

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    task_p = super().task()
    if self.QUANTIZATION is not None:
      model_p = task_p.model
      quant.apply_quantized_layers_sharded(model_p, self.QUANTIZATION)
    return task_p


@experiment_registry.register
class C4SpmdGpt3AdamDataParallelMLPerfHPBS2k1x1(C4SpmdGpt3AdamMLPerfHP):
  r"""Single VLC chip GPT-3 config."""
  PERCORE_BATCH_SIZE = 1
  PROFILER_CAPTURE_STEP = None
  ICI_MESH_SHAPE = [1, 1, 1]
  DCN_MESH_SHAPE = [1, 1, 1]
  EVAL_INTERVAL_STEPS = 12
  NUM_LAYERS = 1
  VOCAB_SIZE = 2048
  NUM_HEADS = 6
  MODEL_DIMS = 768
  HIDDEN_DIMS = MODEL_DIMS * 4
  DIMS_PER_HEAD = MODEL_DIMS // NUM_HEADS
  QUANTIZATION = quant.F8B8

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    task_p = super().task()
    task_p.train.num_train_steps = 5
    task_p.train.profiler_capture_step = None
    if self.QUANTIZATION is not None:
      model_p = task_p.model
      quant.apply_quantized_layers_sharded(model_p, self.QUANTIZATION)
    return task_p


@experiment_registry.register
class C4SpmdGpt3AdamDataParallelMLPerfHPBS2p3k(C4SpmdGpt3AdamMLPerfHP):
  r"""Cross-slice data-parallel GPT-3 config."""
  # 260 steps to 2.69, and ~54.8 secs/step.
  # https://tensorboard.corp.google.com/experiment/7111068764987344309
  # http://xprof/?session_id=sgpyc-11233283339303979545
  PERCORE_BATCH_SIZE = 2  # 2304 global batch size
  VOCAB_SIZE = 51168  # made it multiples of 24 for sharding
  ICI_MESH_SHAPE = [1, 16, 24]
  DCN_MESH_SHAPE = [3, 1, 1]
  EVAL_INTERVAL_STEPS = 11


@experiment_registry.register
class C4SpmdGpt3AdamDataParallelMLPerfHPBS1k(C4SpmdGpt3AdamMLPerfHP):
  r"""Cross-slice data-parallel GPT-3 config."""
  # 600 steps to 2.69, and ~18.6 secs/step.
  # https://tensorboard.corp.google.com/experiment/5792413895730692495
  # http://xprof/?session_id=sgpyc-13333100111111040256
  PERCORE_BATCH_SIZE = 1  # 1024 global batch size
  ICI_MESH_SHAPE = [1, 16, 16]
  DCN_MESH_SHAPE = [4, 1, 1]
  EVAL_INTERVAL_STEPS = 24


@experiment_registry.register
class C4SpmdGpt3TinyAdam8Replicas(C4SpmdGpt3AdamMLPerfHP):
  r"""Small GPT-3 config in bf16 for 8 replicas with 256 global batch size.

  This was called GPT-3 Large in the GPT-3 paper, with 760M parameters.
  """

  NUM_LAYERS = 24
  NUM_HEADS = 16
  MODEL_DIMS = 1536
  HIDDEN_DIMS = MODEL_DIMS * 4
  DIMS_PER_HEAD = 96
  VOCAB_SIZE = 51200
  TARGET_LOG_PPLX = 8

  PERCORE_BATCH_SIZE = 32
  FPROP_DTYPE = jnp.bfloat16
  LEARNING_RATE = 2.5e-4
  ICI_MESH_SHAPE = [2, 2, 2]
  DCN_MESH_SHAPE = [1, 1, 1]

  CHECKPOINT_MAX_TO_KEEP = 1000
  EVAL_INTERVAL_STEPS = 10
  SUMMARY_INTERVAL_STEPS = 5
  CHECKPOINT_EVERY_N_STEPS = 200


@experiment_registry.register
class C4SpmdGpt3AdamDataParallel2x16x16(C4SpmdGpt3AdamDataParallel):
  r"""Cross-slice data-parallel GPT-3 config."""
  DCN_MESH_SHAPE = [2, 1, 1]


@experiment_registry.register
class C4SpmdGpt3AdamDataParallel1x16x16(C4SpmdGpt3AdamDataParallel):
  r"""Cross-slice data-parallel GPT-3 config."""
  DCN_MESH_SHAPE = [1, 1, 1]


@experiment_registry.register
class C4SpmdGpt3AdamDataParallel2x2(C4SpmdGpt3AdamDataParallel):
  r"""Data-parallel GPT-3 proxy config to test quantization."""
  DCN_MESH_SHAPE = None
  ICI_MESH_SHAPE = [1, 2, 2]
  NUM_HEADS = 8
  MODEL_DIMS = 512
  HIDDEN_DIMS = MODEL_DIMS * 4
  DIMS_PER_HEAD = MODEL_DIMS // NUM_HEADS
  NUM_LAYERS = 32
  QUANTIZATION = quant.F8B8

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    task_p = super().task()
    task_p.train.num_train_steps = 5
    return task_p


@experiment_registry.register
class C4SpmdGpt3AdamDataParallel2x2FP8(C4SpmdGpt3AdamDataParallel2x2):
  r"""Data-parallel GPT-3 proxy config to test FP8 quantization."""
  QUANTIZATION = quant.FP8


@experiment_registry.register
class C4SpmdGpt3AdamDataParallel2x16x16Int8(C4SpmdGpt3AdamDataParallel2x16x16):
  r"""Cross-slice data-parallel GPT-3 config."""
  QUANTIZATION = quant.F8B8


@experiment_registry.register
class C4SpmdGpt3AdamDataParallelWUS(C4SpmdGpt3AdamDataParallel):
  r"""Cross-slice data-parallel GPT-3 config."""
  # https://xprof.corp.google.com/overview_page/jlwei-8696671320180540264
  DCN_MESH_SHAPE = [1, 4, 1]
  # TODO(b/269676752): find the optimal checkpoint policy. SAVE_QKV_OUT_PROJ
  # runs out of HBM.


@experiment_registry.register
class C4SpmdGpt3AdamDataParallelWUS2x16x16(C4SpmdGpt3AdamDataParallelWUS):
  r"""Cross-slice data-parallel GPT-3 config."""
  DCN_MESH_SHAPE = [1, 2, 1]


class C4SpmdGpt3AdamPipeline(c4.C4SpmdPipelineGpt3AdamMLPerfHP):
  r"""Pipelined GPT-3 config using Adam optimizer with C4 dataset."""
  FPROP_DTYPE = jnp.bfloat16
  # Keep summary & eval at a medium interval
  SUMMARY_INTERVAL_STEPS = 12
  EVAL_INTERVAL_STEPS = 48
  QUANTIZATION = None

  @property
  def CHECKPOINT_EVERY_N_STEPS(self) -> int | None:
    value = super().CHECKPOINT_EVERY_N_STEPS
    if _MLPERF_GPT3_CHECKPOINT_EVERY_N_STEPS.value:
      value = int(_MLPERF_GPT3_CHECKPOINT_EVERY_N_STEPS.value)
    return value

  @property
  def CHECKPOINT_MAX_TO_KEEP(self) -> int:
    value = super().CHECKPOINT_MAX_TO_KEEP
    if _MLPERF_GPT3_CHECKPOINT_MAX_TO_KEEP.value:
      value = int(_MLPERF_GPT3_CHECKPOINT_MAX_TO_KEEP.value)
    return value

  @property
  def TRAINING_SEED(self) -> int:
    value = super().TRAINING_SEED
    if _MLPERF_GPT3_TRAINING_SEED.value:
      value = int(_MLPERF_GPT3_TRAINING_SEED.value)
    return value

  @property
  def TRAINING_NUM_BATCHES_TO_SKIP(self) -> int:
    value = super().TRAINING_NUM_BATCHES_TO_SKIP
    if _MLPERF_GPT3_TRAINING_NUM_BATCHES_TO_SKIP.value:
      value = int(_MLPERF_GPT3_TRAINING_NUM_BATCHES_TO_SKIP.value)
    return value

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    mllogger = c4_mllog.try_to_init_mlloger(self)
    task_p = super().task()
    if self.QUANTIZATION is not None:
      model_p = task_p.model  # pytype: disable=attribute-error  # enable-nested-classes
      quant.apply_quantized_layers_sharded(model_p, self.QUANTIZATION)
    task_p = c4_mllog.log_params_and_set_early_stopping_fn(
        self, task_p, mllogger
    )
    return task_p


@experiment_registry.register
class C4SpmdGpt3AdamPipeline1B(C4SpmdGpt3AdamPipeline):
  r"""Reduced size, pipelined GPT-3 config with 128 batch size on 8 chips.

  https://xprof.corp.google.com/overview_page/shibow-17744565741649282031
  """

  NUM_LAYERS = 16
  NUM_HEADS = 16
  MODEL_DIMS = 2048
  HIDDEN_DIMS = MODEL_DIMS * 4

  PERCORE_BATCH_SIZE = 16

  NUM_STAGES = 8
  ICI_MESH_SHAPE = [8, 1, 1, 1]
  DCN_MESH_SHAPE = [1, 1, 1, 1]

  EMB_W_DATA_DIMS = ('replica', 'data')
  STREAM_IO = True
  USE_REPEATED_LAYER = False

  MICROBATCH_SIZE = 2  # 64 microbatches


@experiment_registry.register
class C4SpmdGpt3AdamPipeline1BTest(C4SpmdGpt3AdamPipeline1B):
  r"""Reduced size, pipelined GPT-3 config with 128 batch size on 8 chips test."""

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    task_p = super().task()

    task_p.train.num_train_steps = 2

    return task_p


@experiment_registry.register
class C4SpmdGpt3AdamPipeline1BTestBroadcastInputs(C4SpmdGpt3AdamPipeline1B):
  r"""Reduced size, pipelined GPT-3 broadcast pipeline inputs test."""

  ICI_MESH_SHAPE = [1, 8, 1, 1]
  NUM_STAGES = 1

  PIPELINE_BROADCAST_INPUTS = True

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    task_p = super().task()

    task_p.train.num_train_steps = 2

    return task_p


@experiment_registry.register
class C4SpmdGpt3AdamPipeline2Slice2Stage2x2x1(C4SpmdGpt3AdamPipeline):
  r"""One GPT-3 single layer per stage; 2 stages; SPMD sharded.

  https://xprof.corp.google.com/overview_page/jlwei-172186628016366400
  https://xprof.corp.google.com/overview_page/jlwei-8899880539227825179

  MODEL=C4SpmdGpt3AdamPipeline2Slice2Stage2x2x1 &&
  CELL=gg &&
  DIR=/cns/${CELL}-d/home/${USER}/xor/${MODEL}/ && \
  gxm learning/multipod/pax/xm_launch.py \
    --xm_resource_pool=search \
    --xm_resource_alloc=group:search/tpu-perf-team-xm-${CELL} \
    --xm_job_name=c4_${MODEL} \
    --platform=pf_2x2x1 \
    --cell=${CELL} \
    --exp=gpt3.${MODEL} \
    --job_log_dir=${DIR} \
    --noxm_monitor_on_launch \
    --num_slices=2 \
    --use_pathways \
    --build_target="//third_party/tensorflow_models/mlperf/models/rough/gpt3:main" \
    --dump_xla=true
  """
  PERCORE_BATCH_SIZE = 1

  NUM_LAYERS = 2
  NUM_STAGES = 2
  ICI_MESH_SHAPE = [1, 1, 1, 4]
  DCN_MESH_SHAPE = [2, 1, 1, 1]

  EMB_W_DATA_DIMS = ('replica', 'data')
  STREAM_IO = False

  MICROBATCH_SIZE = 8  # 1 microbatches


@experiment_registry.register
class Gpt3SpmdPipelineAdam4Slice4Stage2x2x1(C4SpmdGpt3AdamPipeline):
  r"""One GPT-3 single layer per stage; 4 stages; SPMD sharded.

  https://xprof.corp.google.com/trace_viewer/jlwei-10350294222550633972

  MODEL=Gpt3SpmdPipelineAdam4Slice4Stage2x2x1 &&
  CELL=gg &&
  DIR=/cns/${CELL}-d/home/${USER}/xor/${MODEL}/ && \
  gxm learning/multipod/pax/xm_launch.py \
    --xm_resource_pool=search \
    --xm_resource_alloc=group:search/tpu-perf-team-xm-${CELL} \
    --xm_job_name=c4_${MODEL} \
    --platform=pf_2x2x1 \
    --cell=${CELL} \
    --exp=gpt3.${MODEL} \
    --job_log_dir=${DIR} \
    --noxm_monitor_on_launch \
    --num_slices=4 \
    --use_pathways \
    --build_target="//third_party/tensorflow_models/mlperf/models/rough/gpt3:main" \
    --dump_xla=true
  """
  PERCORE_BATCH_SIZE = 4

  NUM_LAYERS = 4
  NUM_STAGES = 4
  ICI_MESH_SHAPE = [1, 1, 1, 4]
  DCN_MESH_SHAPE = [4, 1, 1, 1]

  EMB_W_DATA_DIMS = ('replica', 'data')
  STREAM_IO = False

  MICROBATCH_SIZE = 16  # 4 microbatches


@experiment_registry.register
class C4SpmdGpt3AdamPipeline512(C4SpmdGpt3AdamPipeline):
  r"""Pipelined GPT-3 config on a 8x8x8 slice.

  https://xprof.corp.google.com/overview_page/shibow-7115559665350422175
  52% hardware and 43% model FLOPS utilization.
  """

  PERCORE_BATCH_SIZE = 2  # 1024 global batch size

  NUM_STAGES = 8
  ICI_MESH_SHAPE = [8, 1, 8, 8]
  DCN_MESH_SHAPE = [1, 1, 1, 1]

  EMB_W_DATA_DIMS = ('replica', 'data')
  STREAM_IO = True
  USE_REPEATED_LAYER = False

  MICROBATCH_SIZE = 16  # 64 microbatches
  EVAL_INTERVAL_STEPS = 24

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    task_p = super().task()

    model_p = task_p.model  # pytype: disable=attribute-error  # enable-nested-classes
    if hasattr(model_p, 'lm_tpl'):
      xformer_p = model_p.lm_tpl.stacked_transformer_tpl.pipeline_stage

      if xformer_p.cls == transformers.StackedTransformerRepeated:
        xformer_p = xformer_p.block
      xformer_p = xformer_p.transformer_layer_params_tpl

      for atten_p in (xformer_p.tr_atten_tpl, xformer_p.cross_atten_tpl):
        if atten_p is None:
          continue
        atten_wp = atten_p.weight_split_dims_mapping
        atten_wp.proj = [None, 'mdl', None]

    return task_p


# TODO(b/257267866): Remove class and use Fiddle injections on parent class
#   when Fiddle support is available in Pax
@experiment_registry.register
class C4SpmdGpt3AdamPipeline512HybridSim(C4SpmdGpt3AdamPipeline512):
  """Pipelined GPT-3 config on a 512 slice, modified for HybridSim Search."""

  @property
  def ICI_MESH_SHAPE(self):
    value = super().ICI_MESH_SHAPE
    if _HYBRIDSIM_ICI_MESH_SHAPE.value:
      value = [int(x) for x in _HYBRIDSIM_ICI_MESH_SHAPE.value.split('_')]
    return value

  @property
  def NUM_STAGES(self):
    return self.ICI_MESH_SHAPE[0]

  @property
  def PERCORE_BATCH_SIZE(self):
    value = super().PERCORE_BATCH_SIZE
    if _HYBRIDSIM_PERCORE_BATCH_SIZE.value:
      value = float(_HYBRIDSIM_PERCORE_BATCH_SIZE.value)
    return value

  @property
  def MICROBATCH_SIZE(self):
    value = super().MICROBATCH_SIZE
    if _HYBRIDSIM_MICROBATCH_SIZE_POWER.value is not None:
      value = 2**_HYBRIDSIM_MICROBATCH_SIZE_POWER.value
    return value

  @property
  def CIRCULAR_REPEAT(self):
    value = super().CIRCULAR_REPEAT
    if _HYBRIDSIM_CIRCULAR_REPEAT.value:
      value = _HYBRIDSIM_CIRCULAR_REPEAT.value
    return value

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Sets number of steps low for quick AOT compilation."""
    task_p = super().task()
    task_p.train.num_train_steps = 2
    return task_p

  def datasets(self) -> list[pax_fiddle.Config[base_input.BaseInput]]:
    """Returns only training dataset, to avoid eval in AOT compilation."""
    return [
        self._dataset_common(is_training=True),
    ]


@experiment_registry.register
class C4SpmdGpt3AdamPipeline512CompileTest(C4SpmdGpt3AdamPipeline512):
  r"""Pipelined 8x8x8 GPT-3 for mock tpu testing."""

  # Set these intervals to very large number to turn features off
  SUMMARY_INTERVAL_STEPS = 10000
  CHECKPOINT_EVERY_N_STEPS = 10000
  EVAL_INTERVAL_STEPS = 10000
  CHECKPOINT_MAX_TO_KEEP = 1

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    task_p = super().task()
    task_p.train.num_train_steps = 2
    return task_p

  def datasets(self) -> list[pax_fiddle.Config[base_input.BaseInput]]:
    """Returns only training dataset."""
    return [
        self._dataset_common(is_training=True),
    ]


@experiment_registry.register
class C4SpmdGpt3AdamPipeline3K(C4SpmdGpt3AdamPipeline):
  r"""Pipelined GPT-3 config on a 24x16x8 slice."""

  PERCORE_BATCH_SIZE = 0.5  # 1536 global batch size

  NUM_STAGES = 24
  ICI_MESH_SHAPE = [24, 1, 16, 8]
  DCN_MESH_SHAPE = [1, 1, 1, 1]

  EMB_W_DATA_DIMS = ('replica', 'data')
  STREAM_IO = True
  USE_REPEATED_LAYER = False

  MICROBATCH_SIZE = 16  # 96 microbatches
  EVAL_INTERVAL_STEPS = 16

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    task_p = super().task()
    task_p.train.num_train_steps = 10
    model_p = task_p.model  # pytype: disable=attribute-error  # enable-nested-classes
    if hasattr(model_p, 'lm_tpl'):
      xformer_p = model_p.lm_tpl.stacked_transformer_tpl.pipeline_stage

      if xformer_p.cls == transformers.StackedTransformerRepeated:
        xformer_p = xformer_p.block
      xformer_p = xformer_p.transformer_layer_params_tpl

      for atten_p in (xformer_p.tr_atten_tpl, xformer_p.cross_atten_tpl):
        if atten_p is None:
          continue
        atten_wp = atten_p.weight_split_dims_mapping
        atten_wp.proj = [None, 'mdl', None]

      ff_p = xformer_p.tr_fflayer_tpl
      ff_wp = ff_p.weight_split_dims_mapping
      ff_wp.ffn0 = [None, 'mdl']
      ff_wp.ffn1 = ['mdl', None]

    return task_p


@experiment_registry.register
class C4SpmdGpt3AdamPipelineVFCTest(C4SpmdGpt3AdamPipeline3K):
  r"""Pipelined GPT-3 config to test AOT compilation on VFC."""

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    task_p = super().task()
    task_p.train.num_train_steps = 10
    return task_p


@experiment_registry.register
class C4SpmdGpt3AdamPipeline8kVFCTest(C4SpmdGpt3AdamPipelineVFCTest):
  r"""Pipelined GPT-3 config on a 16x16x32 slice VFC."""

  PERCORE_BATCH_SIZE = 0.1875  # 1536 global batch size

  NUM_STAGES = 32
  ICI_MESH_SHAPE = [32, 1, 16, 16]


@experiment_registry.register
class C4SpmdGpt3AdamPipeline6kVFCTest(C4SpmdGpt3AdamPipelineVFCTest):
  r"""Pipelined GPT-3 config on a 16x16x24 slice VFC."""

  PERCORE_BATCH_SIZE = 0.25  # 1536 global batch size

  NUM_STAGES = 24
  ICI_MESH_SHAPE = [24, 1, 16, 16]


@experiment_registry.register
class C4SpmdGpt3AdamPipeline4kVFCTest(C4SpmdGpt3AdamPipelineVFCTest):
  r"""Pipelined GPT-3 config on a 16x16x16 slice VFC."""

  PERCORE_BATCH_SIZE = 0.375  # 1536 global batch size

  NUM_STAGES = 16
  ICI_MESH_SHAPE = [16, 1, 16, 16]


@experiment_registry.register
class C4SpmdGpt3AdamPipeline3KProxy(C4SpmdGpt3AdamPipeline3K):
  r"""Pipelined GPT-3 config on a 512-chip proxy.

  https://xprof.corp.google.com/overview_page/shibow-17342769417580207371
  56% hardware FLOPS utilization, projected 44% model FLOPS utilization for
  the 3K run.
  Can be improved to 60% model FLOPS utilization with circular pipeline, less
  recomputation and step time improvements.
  """

  NUM_LAYERS = 16
  NUM_STAGES = 4
  ICI_MESH_SHAPE = [4, 1, 16, 8]

  MICROBATCH_SIZE = 16  # 16 microbatches


@experiment_registry.register
class C4SpmdGpt3AdamPipeline6K(C4SpmdGpt3AdamPipeline3K):
  r"""Pipelined GPT-3 config on 8 8x8x12 slices."""

  PERCORE_BATCH_SIZE = 0.25  # 1536 global batch size

  NUM_STAGES = 8
  ICI_MESH_SHAPE = [1, 1, 64, 12]
  DCN_MESH_SHAPE = [8, 1, 1, 1]

  MICROBATCH_SIZE = 64  # 24 microbatches


@experiment_registry.register
class C4SpmdGpt3AdamPipeline6KProxy(C4SpmdGpt3AdamPipeline6K):
  r"""Pipedlined GPT-3 config on 2 8x8x12 slices."""

  NUM_LAYERS = 24
  NUM_STAGES = 2
  DCN_MESH_SHAPE = [2, 1, 1, 1]

  MICROBATCH_SIZE = 64  # 6 microbatches


@experiment_registry.register
class C4SpmdGpt3AdamPipeline8K(C4SpmdGpt3AdamPipeline3K):
  r"""Pipelined GPT-3 config on 16 8x8x8 slices."""

  PERCORE_BATCH_SIZE = 0.1875  # 1536 global batch size

  NUM_STAGES = 16
  ICI_MESH_SHAPE = [1, 1, 64, 8]
  DCN_MESH_SHAPE = [16, 1, 1, 1]

  MICROBATCH_SIZE = 64  # 24 microbatches


@experiment_registry.register
class C4SpmdGpt3AdamPipeline8KProxy(C4SpmdGpt3AdamPipeline8K):
  r"""Pipedlined GPT-3 config on 2 8x8x8 slices."""

  NUM_LAYERS = 12
  NUM_STAGES = 2
  DCN_MESH_SHAPE = [2, 1, 1, 1]

  MICROBATCH_SIZE = 64  # 3 microbatches


@experiment_registry.register
class C4SpmdGpt3AdamPipeline12K(C4SpmdGpt3AdamPipeline3K):
  r"""Pipelined GPT-3 config on 24 8x8x8 slices."""

  PERCORE_BATCH_SIZE = 0.125  # 1536 global batch size

  NUM_STAGES = 24
  ICI_MESH_SHAPE = [1, 1, 64, 8]
  DCN_MESH_SHAPE = [24, 1, 1, 1]

  MICROBATCH_SIZE = 64  # 24 microbatches


@experiment_registry.register
class C4SpmdGpt3AdamPipeline12KProxy(C4SpmdGpt3AdamPipeline12K):
  r"""Pipedlined GPT-3 config on 2 8x8x8 slices.

  https://xprof.corp.google.com/overview_page/shibow-9290764295114455562
  45% hardware FLOPS utilization, projected 28% model FLOPS utilization for
  the 12K run.
  Can be improved to 50% model FLOPS utilization with circular pipeline, less
  recomputation and step time improvements.
  """

  NUM_LAYERS = 8
  NUM_STAGES = 2
  DCN_MESH_SHAPE = [2, 1, 1, 1]

  MICROBATCH_SIZE = 64  # 2 microbatches


@experiment_registry.register
class C4SpmdGpt3AdamPipeline12KV2(C4SpmdGpt3AdamPipeline3K):
  r"""Pipelined GPT-3 config on 16 8x8x12 slices.

  This configuration should have higher utilization than the above one.
  """

  NUM_STAGES = 16
  ICI_MESH_SHAPE = [1, 1, 64, 12]
  DCN_MESH_SHAPE = [16, 1, 1, 1]

  MICROBATCH_SIZE = 64  # 24 microbatches


@experiment_registry.register
class C4SpmdGpt3AdamPipeline12KV2Proxy(C4SpmdGpt3AdamPipeline12KV2):
  r"""Pipedlined GPT-3 config on 2 8x8x12 slices."""

  NUM_LAYERS = 12
  NUM_STAGES = 2
  DCN_MESH_SHAPE = [2, 1, 1, 1]

  MICROBATCH_SIZE = 64  # 3 microbatches


@experiment_registry.register
class C4SpmdGpt3AdamPipeline16K(C4SpmdGpt3AdamPipeline3K):
  r"""Pipelined GPT-3 config on 16 8x8x16 slices."""

  PERCORE_BATCH_SIZE = 0.1875  # 3072 global batch size

  NUM_STAGES = 16
  ICI_MESH_SHAPE = [1, 1, 64, 16]
  DCN_MESH_SHAPE = [16, 1, 1, 1]

  MICROBATCH_SIZE = 64  # 48 microbatches
  EVAL_INTERVAL_STEPS = 8


@experiment_registry.register
class C4SpmdGpt3AdamPipeline16KProxy(C4SpmdGpt3AdamPipeline16K):
  r"""Pipedlined GPT-3 config on 2 8x8x16 slices."""

  NUM_LAYERS = 12
  NUM_STAGES = 2
  DCN_MESH_SHAPE = [2, 1, 1, 1]

  MICROBATCH_SIZE = 64  # 3 microbatches


@experiment_registry.register
class Gpt3Pipeline4Slice4Stage2x2x1SaveDot(C4SpmdGpt3AdamPipeline):
  """One GPT-3 single layer per stage; 4 stages; SPMD sharded."""

  ICI_MESH_SHAPE = [1, 1, 1, 4]
  DCN_MESH_SHAPE = [4, 1, 1, 1]
  EMB_W_DATA_DIMS = ('replica', 'data')
  NUM_HEADS = 24
  MODEL_DIMS = 3072
  HIDDEN_DIMS = MODEL_DIMS * 8
  DIMS_PER_HEAD = MODEL_DIMS // NUM_HEADS
  NUM_LAYERS_PER_STAGE = 2
  NUM_STAGES = 4
  NUM_LAYERS = NUM_LAYERS_PER_STAGE * NUM_STAGES
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_DOT_ONLY
  PERCORE_BATCH_SIZE = 2
  NUM_MICROBATCHES = 4
  STREAM_IO = True


@experiment_registry.register
class Gpt3Pipeline4Slice4Stage2x2x1SaveNothing(
    Gpt3Pipeline4Slice4Stage2x2x1SaveDot
):
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING


@experiment_registry.register
class C4SpmdGpt3AdamPipeline2x2Baseline(C4SpmdGpt3AdamPipeline):
  """Forge baseline test model."""

  PERCORE_BATCH_SIZE = 32  # 128 global batch size

  ICI_MESH_SHAPE = [2, 2, 1, 1]
  DCN_MESH_SHAPE = [1, 1, 1, 1]

  NUM_STAGES = 2
  CIRCULAR_REPEAT = 1
  MICROBATCH_SIZE = 2
  NUM_LAYERS = 8
  NUM_HEADS = 8
  MODEL_DIMS = 2048
  HIDDEN_DIMS = MODEL_DIMS * 4
  DIMS_PER_HEAD = MODEL_DIMS // NUM_HEADS

  def _dataset_common(
      self, is_training
  ) -> pax_fiddle.Config[base_input.BaseInput]:
    """Returns synthetic dataset."""
    num_local_devices = jax.local_device_count()
    batch_size = self.PERCORE_BATCH_SIZE * num_local_devices
    input_p = input_generator.SyntheticLmData.HParams()
    input_p.batch_size = batch_size
    input_p.seq_len = self.MAX_SEQ_LEN
    p = base_input.LingvoInputAdaptor.HParams(
        input=input_p, is_training=is_training
    )
    return p

  def datasets(self) -> list[pax_fiddle.Config[base_input.BaseInput]]:
    """Returns only training dataset."""
    return [
        self._dataset_common(is_training=True),
    ]

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    task_p = super().task()
    task_p.train.num_train_steps = 5
    return task_p


@experiment_registry.register
class C4SpmdGpt3AdamPipeline2x2Quantized(C4SpmdGpt3AdamPipeline2x2Baseline):
  """Forge quantized test model."""

  QUANTIZATION = quant.F8B8


@experiment_registry.register
class C4SpmdGpt3AdamPipeline2x2QuantizedFP8(C4SpmdGpt3AdamPipeline2x2Baseline):
  """Forge quantized test model with FP8."""

  QUANTIZATION = quant.FP8


@experiment_registry.register
class C4SpmdGpt3AdamPipeline4x4xProxyQuantized(C4SpmdGpt3AdamPipeline):
  """Quantized proxy model with model parallelism."""

  # With stochastic rounding:
  # https://xprof.corp.google.com/overview_page/jlwei-569071527917688842
  # Without stochastic rounding:
  # https://xprof.corp.google.com/trace_viewer/jlwei-7140819125744264014
  QUANTIZATION = quant.F8B8

  ICI_MESH_SHAPE = [4, 1, 4, 4]
  DCN_MESH_SHAPE = [1, 1, 1, 1]

  NUM_HEADS = 24
  MODEL_DIMS = 3072
  HIDDEN_DIMS = MODEL_DIMS * 4
  DIMS_PER_HEAD = MODEL_DIMS // NUM_HEADS

  PERCORE_BATCH_SIZE = 2
  MICROBATCH_SIZE = 8
  NUM_STAGES = 4