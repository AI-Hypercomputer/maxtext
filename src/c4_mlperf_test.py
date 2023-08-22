"""referred to google3/platforms/deepsea/ffds/xor_bmk/bmk.py."""
import jax
from jax import numpy as jnp
from jax.experimental import multihost_utils
from paxml import experiment_registry
from paxml import tasks_lib
from paxml.tasks.lm.params.gpt3 import C4SpmdGpt3AdamMLPerfHP
from praxis import pax_fiddle
import portpicker
import socket


@experiment_registry.register
class C4SpmdGpt3Adam2x8x4Test(C4SpmdGpt3AdamMLPerfHP):
  r"""Cross-slice data-parallel GPT-3 config."""
  NUM_LAYERS = 10
  NUM_HEADS = 24
  MODEL_DIMS = 6144
  HIDDEN_DIMS = MODEL_DIMS * 4

  PERCORE_BATCH_SIZE = 0.5
  ICI_MESH_SHAPE = [1, 8, 4]
  DCN_MESH_SHAPE = [2, 1, 1]
  FPROP_DTYPE = jnp.bfloat16

  # for mlperf_gpt3_checkpoint_every_n_steps=2000
  CHECKPOINT_EVERY_N_STEPS = 100

  # mlperf_gpt3_checkpoint_max_to_keep=20
  CHECKPOINT_SAVE_MAX_TO_KEEP = 20

  EVAL_INTERVAL_STEPS = 100
  EVAL_SKIP_TRAIN = True

  # for jax_softmax_custom_jvp
  jax.config.update("jax_softmax_custom_jvp", False)
  TARGET_LOG_PPLX = 7.5

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    task_p = super().task()
    task_p.train.num_train_steps = 800

    # mlperf_gpt3_summary_verbosity=0
    task_p.summary_verbosity = 0

    task_p.train.eval_skip_train = self.EVAL_SKIP_TRAIN

    return task_p

def gen_local_ip():
  hostname = socket.gethostname()
  return socket.gethostbyname(hostname)

def gen_local_ip_nums():
  return [int(num) for num in gen_local_ip().split(':')[-1].split('.')]

def get_coordinator_ip():
  local_ip_nums = jax.numpy.array(gen_local_ip_nums())
  coordinator_ip_nums = multihost_utils.broadcast_one_to_all(local_ip_nums)
  coordinator_ip_strings = [str(num) for num in list(coordinator_ip_nums)]
  return '.'.join(coordinator_ip_strings)

@experiment_registry.register
class C4SpmdGpt3AdamDataParallel2x4x4(C4SpmdGpt3AdamMLPerfHP):
  r"""Cross-slice data-parallel GPT-3 config."""

  port = multihost_utils.broadcast_one_to_all(jax.numpy.array(portpicker.pick_unused_port()))
  coordinator_address = get_coordinator_ip() + ':' + str(port)
  jax.distributed.initialize(coordinator_address=coordinator_address,
                             num_processes=jax.process_count(),
                             process_id=jax.process_index())

  print("JAX INFO")
  print("port: ",port)
  print("coordinator_address: ",coordinator_address)
  print("jax.process_count: ",jax.process_count())
  print("jax.process_index: ",jax.process_index())

  # 2 x v5litepod-16

  NUM_LAYERS = 10
  NUM_HEADS = 24
  MODEL_DIMS = 6144
  HIDDEN_DIMS = MODEL_DIMS * 4

  PERCORE_BATCH_SIZE = 0.5
  ICI_MESH_SHAPE = [1, 4, 4]
  DCN_MESH_SHAPE = [2, 1, 1]
  FPROP_DTYPE = jnp.bfloat16

  EVAL_INTERVAL_STEPS = 40
  EVAL_SKIP_TRAIN = True

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    task_p = super().task()
    task_p.train.num_train_steps = 100  # Run for 80 steps
    task_p.summary_verbosity = 0

    task_p.train.eval_skip_train = self.EVAL_SKIP_TRAIN
    return task_p
  
@experiment_registry.register
class C4SpmdGpt3AdamDataParallel2x2x4(C4SpmdGpt3AdamMLPerfHP):
  r"""Cross-slice data-parallel GPT-3 config."""

  # 2 x v4-16

  NUM_LAYERS = 10
  NUM_HEADS = 24
  MODEL_DIMS = 6144
  HIDDEN_DIMS = MODEL_DIMS * 4

  PERCORE_BATCH_SIZE = 0.5
  ICI_MESH_SHAPE = [2, 2, 2]
  DCN_MESH_SHAPE = [2, 1, 1]
  FPROP_DTYPE = jnp.bfloat16

  EVAL_INTERVAL_STEPS = 40
  EVAL_SKIP_TRAIN = True

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    task_p = super().task()
    task_p.train.num_train_steps = 100  # Run for 80 steps
    task_p.summary_verbosity = 0

    task_p.train.eval_skip_train = self.EVAL_SKIP_TRAIN
    return task_p


@experiment_registry.register
class C4SpmdGpt3AdamDataParallel1x16x16(C4SpmdGpt3AdamMLPerfHP):
  r"""Cross-slice data-parallel GPT-3 config."""

  # 2 x v5litepod-256

  PERCORE_BATCH_SIZE = 0.5  # 512 global batch size
  ICI_MESH_SHAPE = [1, 16, 16]
  DCN_MESH_SHAPE = [1, 1, 1]
  FPROP_DTYPE = jnp.bfloat16

  # for mlperf_gpt3_checkpoint_every_n_steps=2000
  CHECKPOINT_EVERY_N_STEPS = 2000

  # mlperf_gpt3_checkpoint_max_to_keep=20
  CHECKPOINT_SAVE_MAX_TO_KEEP = 20
  PERCORE_EVAL_BATCH_SIZE = 1.5
  EVAL_INTERVAL_STEPS = 12
  EVAL_SKIP_TRAIN = True
  TARGET_LOG_PPLX = 7.5

  # for jax_softmax_custom_jvp
  jax.config.update("jax_softmax_custom_jvp", False)

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    task_p = super().task()
    task_p.train.num_train_steps = 1000

    task_p.train.eval_skip_train = self.EVAL_SKIP_TRAIN
    # mlperf_gpt3_summary_verbosity=0
    task_p.summary_verbosity = 0
    return task_p


@experiment_registry.register
class C4SpmdGpt3AdamDataParallel2x16x16(C4SpmdGpt3AdamMLPerfHP):
  r"""Cross-slice data-parallel GPT-3 config."""

  # 2 x v5litepod-256

  PERCORE_BATCH_SIZE = 0.5  # 512 global batch size
  ICI_MESH_SHAPE = [1, 16, 16]
  DCN_MESH_SHAPE = [2, 1, 1]
  FPROP_DTYPE = jnp.bfloat16

  # for mlperf_gpt3_checkpoint_every_n_steps=2000
  CHECKPOINT_EVERY_N_STEPS = 2000

  # mlperf_gpt3_checkpoint_max_to_keep=20
  CHECKPOINT_SAVE_MAX_TO_KEEP = 20

  # for jax_softmax_custom_jvp
  jax.config.update("jax_softmax_custom_jvp", False)

  PERCORE_EVAL_BATCH_SIZE = 1.5
  EVAL_INTERVAL_STEPS = 12
  EVAL_SKIP_TRAIN = True

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    task_p = super().task()
    task_p.summary_verbosity = 0

    task_p.train.eval_skip_train = self.EVAL_SKIP_TRAIN

    task_p.train.num_train_steps = 1000  # Run for 80 steps
    return task_p


@experiment_registry.register
class C4SpmdGpt3AdamDataParallel4x16x16(C4SpmdGpt3AdamMLPerfHP):
  r"""Cross-slice data-parallel GPT-3 config."""

  # 2 x v5litepod-256

  PERCORE_BATCH_SIZE = 0.5  # 512 global batch size
  ICI_MESH_SHAPE = [1, 16, 16]
  DCN_MESH_SHAPE = [4, 1, 1]
  FPROP_DTYPE = jnp.bfloat16

  # for mlperf_gpt3_checkpoint_every_n_steps=2000
  CHECKPOINT_EVERY_N_STEPS = 2000

  # mlperf_gpt3_checkpoint_max_to_keep=20
  CHECKPOINT_SAVE_MAX_TO_KEEP = 20

  # for jax_softmax_custom_jvp
  jax.config.update("jax_softmax_custom_jvp", False)

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    task_p = super().task()
    task_p.train.num_train_steps = 1000

    task_p.train.eval_skip_train = self.EVAL_SKIP_TRAIN
    # mlperf_gpt3_summary_verbosity=0
    task_p.summary_verbosity = 0

    return task_p


@experiment_registry.register
class C4SpmdGpt3AdamDataParallel8x16x16(C4SpmdGpt3AdamMLPerfHP):
  r"""Cross-slice data-parallel GPT-3 config."""

  PERCORE_BATCH_SIZE = 0.5  # 512 global batch size
  ICI_MESH_SHAPE = [1, 16, 16]
  DCN_MESH_SHAPE = [8, 1, 1]
  FPROP_DTYPE = jnp.bfloat16

  # for mlperf_gpt3_checkpoint_every_n_steps=2000
  CHECKPOINT_EVERY_N_STEPS = 2000

  # mlperf_gpt3_checkpoint_max_to_keep=20
  CHECKPOINT_SAVE_MAX_TO_KEEP = 20

  # for jax_softmax_custom_jvp
  jax.config.update("jax_softmax_custom_jvp", False)

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    task_p = super().task()
    task_p.train.num_train_steps = 1000

    task_p.train.eval_skip_train = self.EVAL_SKIP_TRAIN
    # mlperf_gpt3_summary_verbosity=0
    task_p.summary_verbosity = 0

    return task_p
