# Copyright 2026 Google LLC
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

"""Production pure-NNX `M3Trainer` and `< 90s` CPU offline `compile_aot` driver."""

from collections.abc import Iterator, Mapping
import dataclasses
import math
import os
import time
from typing import Any

from flax import nnx
import jax
from jax.experimental.serialize_executable import serialize
from jax.experimental.topologies import get_topology_desc
import jax.numpy as jnp
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
import optax

from maxtext.common import profiler
from maxtext.configs.types import LearningRateScheduleType, OptimizerType
from maxtext.m3 import models as m3_models
from maxtext.m3.configs.pyconfig_bridge import from_pyconfig
from maxtext.m3.configs.training_config import TrainingConfig
from maxtext.m3.core.checkpoint import (
    latest_checkpoint_step,
    restore_m3_checkpoint,
    save_m3_checkpoint,
)
from maxtext.m3.core.data import (
    M3Batch,
    M3SyntheticDataIterator,
    adapt_dict_to_m3_batch,
    adapt_iterator_to_m3_batch,
)
from maxtext.m3.core.interfaces import verify_m3_model_invariants
from maxtext.m3.core.sharding import create_mesh
from maxtext.m3.models.registry import (
    MODEL_GARDEN_REGISTRY,
    ModelConfig,
    _ensure_builtin_presets_registered,
)
from maxtext.m3.train import loop as m3_loop
from maxtext.m3.train.loss import LossBreakdown
from maxtext.m3.train.step import (
    accumulated_loss_and_grads,
    apply_router_bias_update,
    make_eval_step,
    make_train_step,
    reshape_batch_to_microbatches,
)
from maxtext.utils import accelerator_to_spec_map, max_logging


@dataclasses.dataclass(frozen=True)
class AOTCompileResult:
  """Compiled XLA executable and static HBM/FLOP analyses returned by `M3Trainer.compile_aot`."""

  compiled: Any
  memory_analysis: Any
  cost_analysis: Any
  mesh: Mesh


@dataclasses.dataclass(frozen=True)
class TrainResult:
  """Summary metrics returned by `M3Trainer.train()`."""

  final_step: int
  final_loss: float
  step_losses: tuple[float, ...]
  eval_metrics: dict[int, dict[str, float]]


class _ModelCreationAdapter:
  """Adapts `(ModelConfig, TrainingConfig)` to the constructor interface of `m3_models.create_model`."""

  def __init__(self, arch: ModelConfig, train_cfg: TrainingConfig):
    self._arch = arch
    self.transforms = train_cfg.transforms
    self.seed = train_cfg.seed
    self.init_weights_seed = train_cfg.seed
    self.scan_layers = train_cfg.transforms.scan_layers
    self.param_scan_axis = train_cfg.transforms.param_scan_axis
    self.remat_policy = train_cfg.transforms.remat_policy

  def __getattr__(self, name: str) -> Any:
    return getattr(self._arch, name)


class _ProfilerConfigAdapter:
  """Minimal adapter exposing profiling/tensorboard attributes expected by `maxtext.common.profiler.Profiler`."""

  def __init__(self, args: TrainingConfig):
    self._args = args
    p = args.profiling
    self.steps = args.steps
    self.profiler = p.profiler
    self.skip_first_n_steps_for_profiler = p.skip_first_n_steps_for_profiler
    self.profiler_steps = p.profiler_steps
    self.profile_cleanly = p.profile_cleanly
    self.profile_periodically_period = p.profile_periodically_period
    self.upload_all_profiler_results = p.upload_all_profiler_results
    self.managed_mldiagnostics = args.diagnostics.managed_mldiagnostics
    base_dir = args.base_output_directory or "/tmp/m3"
    self.tensorboard_dir = getattr(args.tensorboard, "tensorboard_dir", "") or os.path.join(
        base_dir, args.run_name, "tensorboard"
    )

  def __getattr__(self, name: str) -> Any:
    for sub in (self._args.profiling, self._args.diagnostics, self._args):
      if hasattr(sub, name):
        return getattr(sub, name)
    raise AttributeError(f"'{type(self).__name__}' object has no attribute {name!r}")


def _resolve_jnp_dtype(dtype_enum: Any) -> jnp.dtype:
  name = str(getattr(dtype_enum, "value", dtype_enum))
  return jnp.dtype(name)


def _is_nnx_var(x: Any) -> bool:
  return isinstance(x, nnx.Variable)


def build_lr_schedule(args: TrainingConfig) -> optax.Schedule:
  """Builds an Optax learning-rate schedule from `TrainingConfig`."""
  sched_steps = int(args.steps) if args.learning_rate_schedule_steps <= 0 else int(args.learning_rate_schedule_steps)
  total_steps = max(1, sched_steps)
  warmup_steps = min(total_steps - 1, max(0, int(round(total_steps * args.warmup_steps_fraction))))
  peak_lr = float(args.learning_rate)
  end_lr = peak_lr * float(args.cosine_learning_rate_final_fraction)

  sched_type = args.lr_schedule_type
  sched_name = str(getattr(sched_type, "value", sched_type)).strip().lower()
  if sched_name == "constant":
    if warmup_steps > 0:
      return optax.join_schedules(
          schedules=[
              optax.linear_schedule(0.0, peak_lr, warmup_steps),
              optax.constant_schedule(peak_lr),
          ],
          boundaries=[warmup_steps],
      )
    return optax.constant_schedule(peak_lr)

  if sched_name == "linear":
    decay_steps = max(1, total_steps - warmup_steps)
    linear_decay = optax.linear_schedule(peak_lr, end_lr, decay_steps)
    if warmup_steps > 0:
      return optax.join_schedules(
          schedules=[optax.linear_schedule(0.0, peak_lr, warmup_steps), linear_decay],
          boundaries=[warmup_steps],
      )
    return linear_decay

  if sched_type == LearningRateScheduleType.WSD or sched_name == "wsd":
    stable_end = max(warmup_steps, int(round(total_steps * 0.8)))
    decay_steps = max(1, total_steps - stable_end)
    schedules = []
    boundaries = []
    if warmup_steps > 0:
      schedules.append(optax.linear_schedule(0.0, peak_lr, warmup_steps))
      boundaries.append(warmup_steps)
    schedules.append(optax.constant_schedule(peak_lr))
    boundaries.append(stable_end)
    schedules.append(optax.linear_schedule(peak_lr, end_lr, decay_steps))
    return optax.join_schedules(schedules=schedules, boundaries=boundaries)

  # Default: COSINE
  if warmup_steps > 0:
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=peak_lr,
        warmup_steps=warmup_steps,
        decay_steps=max(warmup_steps + 1, total_steps),
        end_value=end_lr,
    )
  return optax.cosine_decay_schedule(
      init_value=peak_lr,
      decay_steps=max(1, total_steps),
      alpha=float(args.cosine_learning_rate_final_fraction),
  )


def _path_tokens(key_path: tuple[Any, ...]) -> tuple[str, ...]:
  tokens = []
  for part in key_path:
    key = getattr(part, "key", getattr(part, "name", part))
    tokens.append(str(key).lower())
  return tuple(tokens)


def _weight_decay_mask(params: Any) -> Any:
  """Excludes norms, scales, biases, and 1D vectors from decoupled weight decay."""

  def _include(key_path: tuple[Any, ...], leaf: Any) -> bool:
    tokens = _path_tokens(key_path)
    if any("norm" in tok or tok in ("bias", "scale", "embedding") for tok in tokens):
      return False
    arr = leaf.get_value() if isinstance(leaf, nnx.Variable) else leaf
    ndim = getattr(arr, "ndim", 2)
    return int(ndim) >= 2

  return jax.tree_util.tree_map_with_path(_include, params)


def _vision_freeze_mask(params: Any) -> Any:
  """Marks vision encoder parameters `True` so `optax.masked(optax.set_to_zero(), ...)` freezes them."""

  def _is_vision(key_path: tuple[Any, ...], _: Any) -> bool:
    tokens = _path_tokens(key_path)
    return any("vision" in tok or "visual" in tok or tok == "vit" for tok in tokens)

  return jax.tree_util.tree_map_with_path(_is_vision, params)


def build_m3_optimizer(args: TrainingConfig) -> optax.GradientTransformation:
  """Builds the Optax gradient transformation chain (clipping, AdamW/SGD, schedule, and masks)."""
  lr_schedule = build_lr_schedule(args)
  chain_parts: list[optax.GradientTransformation] = []

  if args.gradient_clipping_threshold > 0.0:
    chain_parts.append(optax.clip_by_global_norm(float(args.gradient_clipping_threshold)))

  if args.optimizer_type == OptimizerType.SGD:
    core_tx = optax.sgd(learning_rate=lr_schedule)
  else:
    core_tx = optax.adamw(
        learning_rate=lr_schedule,
        b1=float(args.adam_b1),
        b2=float(args.adam_b2),
        eps=float(args.adam_eps),
        weight_decay=float(args.adam_weight_decay),
        mask=_weight_decay_mask,
    )
  chain_parts.append(core_tx)

  if args.freeze_vision_encoder:
    chain_parts.append(optax.masked(optax.set_to_zero(), _vision_freeze_mask))

  return optax.chain(*chain_parts)


def _variable_named_sharding(var: Any, mesh: Mesh) -> NamedSharding:
  """Extracts `NamedSharding(mesh, spec)` from an `nnx.Variable` (or replicated `P()` fallback)."""
  if isinstance(var, nnx.Variable):
    spec = var.get_metadata().get("out_sharding")
    if isinstance(spec, P):
      return NamedSharding(mesh, spec)
    raw = var.get_value()
    sharding = getattr(raw, "sharding", None)
    if isinstance(sharding, NamedSharding) and sharding.mesh == mesh:
      return sharding
  return NamedSharding(mesh, P())


def _build_topology_mesh(args: TrainingConfig, topology: str, num_slices: int) -> Mesh:
  """Builds an abstract multi-slice or single-slice `Mesh` over `topology` without touching accelerators."""
  if topology.lower() == "cpu":
    devices = list(jax.devices())
  else:
    spec = accelerator_to_spec_map.get_system_characteristics(topology)
    devices = list(
        get_topology_desc(
            platform=spec.platform,
            topology_name=spec.topology_name,
            chip_config_name=spec.chip_config_name,
            chips_per_host_bounds=spec.chips_per_host_bounds,
            num_slices=num_slices,
            wrap=spec.wrap,
        ).devices
    )
  mesh = create_mesh(ici=args.mesh.ici, dcn=args.mesh.dcn, devices=devices)
  if args.mesh.shard_mode == "explicit":
    mesh = Mesh(
        mesh.devices,
        mesh.axis_names,
        axis_types=(AxisType.Explicit,) * len(mesh.axis_names),
    )
  return mesh


class M3Trainer:
  """Pure-NNX pre-training and continued-pre-training driver for m3 models.

  Supports both live single/multi-slice execution (`train()`, `evaluate()`) and `< 90s` CPU-host
  offline Ahead-of-Time compilation (`compile_aot(topology, num_slices)`) with zero weight allocation.
  """

  def __init__(
      self,
      model: ModelConfig | str | nnx.Module,
      args: TrainingConfig | None = None,
      *,
      train_iterator: Any = None,
      eval_iterator: Any = None,
      mesh: Mesh | None = None,
      check_invariants: bool = True,
  ):
    self.args = TrainingConfig() if args is None else args
    self._mesh = mesh
    self._train_iterator = train_iterator
    self._eval_iterator = eval_iterator
    self._check_invariants = check_invariants
    self.tx = build_m3_optimizer(self.args)

    if isinstance(model, nnx.Module):
      self.model_config: ModelConfig | None = None
      self._model: nnx.Module | None = model
      self._optimizer: nnx.Optimizer | None = None
    elif isinstance(model, str):
      self.model_config = ModelConfig.from_preset(
          model,
          param_dtype=self.args.param_dtype,
          compute_dtype=self.args.compute_dtype,
      )
      self._model = None
      self._optimizer = None
    elif isinstance(model, ModelConfig):
      self.model_config = model
      self._model = None
      self._optimizer = None
    else:
      raise TypeError(f"`model` must be a `ModelConfig`, preset name `str`, or `nnx.Module`; got {type(model).__name__}.")

    grad_dtype = _resolve_jnp_dtype(self.args.grad_dtype)
    self._train_step_fn = make_train_step(
        num_vocab_tiles=self.args.num_vocab_tiles,
        z_loss_multiplier=self.args.z_loss_multiplier,
        aux_loss_weight=self.args.aux_loss_weight,
        router_bias_update_rate=self.args.router_bias_update_rate,
        grad_dtype=grad_dtype,
    )
    self._eval_step_fn = make_eval_step(
        num_vocab_tiles=self.args.num_vocab_tiles,
        z_loss_multiplier=self.args.z_loss_multiplier,
        aux_loss_weight=self.args.aux_loss_weight,
    )
    self._start_step = 0

  @property
  def mesh(self) -> Mesh:
    if self._mesh is None:
      base = create_mesh(ici=self.args.mesh.ici, dcn=self.args.mesh.dcn)
      if self.args.mesh.shard_mode == "explicit":
        base = Mesh(base.devices, base.axis_names, axis_types=(AxisType.Explicit,) * len(base.axis_names))
      self._mesh = base
    return self._mesh

  @property
  def is_initialized(self) -> bool:
    """Returns `True` once host/device model parameters have been materialized by `setup()`."""
    return self._model is not None

  @property
  def model(self) -> nnx.Module:
    self.setup()
    assert self._model is not None
    return self._model

  @property
  def optimizer(self) -> nnx.Optimizer:
    self.setup()
    assert self._optimizer is not None
    return self._optimizer

  def _checkpoint_root(self) -> str:
    if not self.args.base_output_directory:
      return ""
    return os.path.join(self.args.base_output_directory, self.args.run_name, "checkpoints")

  def _global_batch_size(self, mesh: Mesh) -> int:
    """Computes global token batch size across `mesh` and `gradient_accumulation_steps`."""
    micro_batch = int(round(self.args.per_device_batch_size * mesh.size))
    if micro_batch < 1:
      raise ValueError(
          f"per_device_batch_size={self.args.per_device_batch_size} on {mesh.size} device(s) yields microbatch < 1."
      )
    batch_shards = math.prod(mesh.shape[ax] for ax in ("data", "fsdp", "expert") if ax in mesh.axis_names)
    if micro_batch % batch_shards != 0:
      raise ValueError(
          f"Microbatch size {micro_batch} does not divide evenly across {batch_shards} data/fsdp/expert shards."
      )
    return micro_batch * int(self.args.gradient_accumulation_steps)

  def _ensure_train_iterator(self) -> Iterator[M3Batch]:
    """Lazily constructs the default `M3SyntheticDataIterator` if no train iterator was injected."""
    if self._train_iterator is None:
      vocab_size = 256
      if self.model_config is not None:
        vocab_size = self.model_config.vocab_size
      elif self._model is not None and hasattr(self._model, "config"):
        vocab_size = int(getattr(self._model.config, "vocab_size", 256))
      self._train_iterator = M3SyntheticDataIterator(
          batch_size=self._global_batch_size(self.mesh),
          seq_len=self.args.max_target_length,
          vocab_size=vocab_size,
          seed=self.args.data.data_shuffle_seed,
          mesh=self.mesh,
      )
    return self._train_iterator

  def setup(self) -> None:
    """Initializes mesh, verifies model invariants, constructs `(model, optimizer)`, and restores checkpoints."""
    if self._model is not None and self._optimizer is not None:
      return

    mesh = self.mesh
    with jax.set_mesh(mesh):
      if self._model is None:
        assert self.model_config is not None
        adapter_cfg = _ModelCreationAdapter(self.model_config, self.args)
        if self._check_invariants:
          micro_b = max(1, int(round(self.args.per_device_batch_size * mesh.size)))
          verify_m3_model_invariants(
              lambda: m3_models.create_model(adapter_cfg, mesh, rngs=nnx.Rngs(self.args.seed)),
              mesh,
              batch_size=micro_b,
              seq_len=min(16, self.args.max_target_length),
          )
        self._model, self._optimizer = m3_loop.setup_model(adapter_cfg, mesh, tx=self.tx)
      else:
        from maxtext.m3.infra import remat as m3_remat  # pylint: disable=import-outside-toplevel
        from maxtext.m3.infra import scan as m3_scan  # pylint: disable=import-outside-toplevel

        self._model = m3_remat.maybe_apply_remat(self._model, self.args)
        self._model = m3_scan.maybe_apply_scan(self._model, self.args)
        self._optimizer = nnx.Optimizer(self._model, self.tx, wrt=nnx.Param)

      train_iter = self._ensure_train_iterator()
      ckpt_root = self._checkpoint_root()
      if self.args.load_checkpoint_path:
        restored = restore_m3_checkpoint(
            self.args.load_checkpoint_path,
            self._model,
            self._optimizer,
            data_iterator=train_iter,
        )
        self._start_step = restored + 1
      elif self.args.enable_checkpointing and ckpt_root and latest_checkpoint_step(ckpt_root) is not None:
        restored = restore_m3_checkpoint(
            ckpt_root,
            self._model,
            self._optimizer,
            data_iterator=train_iter,
        )
        self._start_step = restored + 1

  def _prepare_step_batch(self, raw_batch: Mapping[str, Any]) -> M3Batch:
    if "weights" not in raw_batch or "positions" not in raw_batch or "segment_ids" not in raw_batch:
      batch = adapt_dict_to_m3_batch(raw_batch)
    else:
      batch = raw_batch  # type: ignore[assignment]
    if self.args.gradient_accumulation_steps > 1:
      return reshape_batch_to_microbatches(batch, self.args.gradient_accumulation_steps)
    return batch

  def evaluate(self) -> dict[str, float]:
    """Runs `args.eval_steps` pure evaluation steps and returns aggregated token-weighted metrics."""
    self.setup()
    assert self._model is not None
    if self._eval_iterator is None:
      return {}

    with jax.set_mesh(self.mesh), jax.default_matmul_precision(self.args.matmul_precision.value):
      eval_fn = nnx.cached_partial(self._eval_step_fn, self._model)
      total_xent = 0.0
      total_z = 0.0
      total_aux = 0.0
      total_tokens = 0.0
      steps_run = 0
      for _ in range(self.args.eval_steps):
        try:
          raw = next(self._eval_iterator)
        except StopIteration:
          break
        batch = adapt_dict_to_m3_batch(raw) if "weights" not in raw else raw
        bd = eval_fn(batch)
        total_xent += float(bd.xent_sum)
        total_z += float(bd.z_loss_sum)
        total_aux += float(bd.aux_loss)
        total_tokens += float(bd.valid_tokens)
        steps_run += 1

    if steps_run == 0:
      return {}
    denom = max(total_tokens, 1.0)
    return {
        "eval/loss": (total_xent + total_z) / denom + total_aux / steps_run,
        "eval/lm_loss": total_xent / denom,
        "eval/valid_tokens": total_tokens,
    }

  def train(self) -> TrainResult:
    """Executes the production training loop with zero per-step NNX graph traversal and 1-step async logging."""
    self.setup()
    assert self._model is not None and self._optimizer is not None
    train_iter = self._ensure_train_iterator()
    ckpt_root = self._checkpoint_root()
    tokens_per_step = self._global_batch_size(self.mesh) * self.args.max_target_length

    prof = profiler.Profiler(_ProfilerConfigAdapter(self.args), offset_step=self._start_step)
    recorded_losses: list[float] = []
    eval_history: dict[int, dict[str, float]] = {}

    with (
        jax.set_mesh(self.mesh),
        jax.default_matmul_precision(self.args.matmul_precision.value),
        profiler.nnx_named_scope_context(enabled=bool(self.args.profiling.profiler)),
    ):
      # Cache the split graph structure once before the loop so per-step dispatch performs zero NNX graph walks.
      step_fn = nnx.cached_partial(self._train_step_fn, self._model, self._optimizer)
      pending: tuple[int, LossBreakdown] | None = None
      last_done = time.perf_counter()

      for step in range(self._start_step, self.args.steps):
        raw_batch = next(train_iter)
        step_batch = self._prepare_step_batch(raw_batch)
        prof.maybe_activate_profiler(step, pending)
        with jax.profiler.StepTraceAnnotation("train", step_num=step):
          step_bd = step_fn(step_batch)

        if pending is not None:
          prev_step, prev_bd = pending
          prev_loss = float(prev_bd.total_loss)
          recorded_losses.append(prev_loss)
          now = time.perf_counter()
          dt = max(now - last_done, 1e-9)
          last_done = now
          if prev_step % self.args.log_period == 0:
            max_logging.log(
                f"m3 train step {prev_step}: loss={prev_loss:.4f} lm_loss={float(prev_bd.lm_loss):.4f} "
                f"({dt:.3f}s, {tokens_per_step / dt:,.0f} tok/s)"
            )

        pending = (step, step_bd)
        prof.maybe_deactivate_profiler(step, pending)

        is_last = (step + 1) == self.args.steps
        if self.args.eval_interval > 0 and ((step + 1) % self.args.eval_interval == 0 or is_last):
          metrics = self.evaluate()
          if metrics:
            eval_history[step] = metrics

        if self.args.enable_checkpointing and ckpt_root and ((step + 1) % self.args.checkpoint_period == 0 or is_last):
          # Flush pending device work before serializing model, optimizer, and iterator state.
          prev_step, prev_bd = pending
          prev_loss = float(prev_bd.total_loss)
          recorded_losses.append(prev_loss)
          pending = None
          last_done = time.perf_counter()
          save_m3_checkpoint(
              ckpt_root,
              step,
              self._model,
              self._optimizer,
              data_iterator=train_iter,
          )

      if pending is not None:
        prev_step, prev_bd = pending
        prev_loss = float(prev_bd.total_loss)
        recorded_losses.append(prev_loss)
        now = time.perf_counter()
        dt = max(now - last_done, 1e-9)
        if prev_step % self.args.log_period == 0 or prev_step == self.args.steps - 1:
          max_logging.log(
              f"m3 train step {prev_step}: loss={prev_loss:.4f} lm_loss={float(prev_bd.lm_loss):.4f} "
              f"({dt:.3f}s, {tokens_per_step / dt:,.0f} tok/s)"
          )

    final_loss = recorded_losses[-1] if recorded_losses else float("nan")
    final_step = max(self._start_step, self.args.steps) - 1
    return TrainResult(
        final_step=final_step,
        final_loss=final_loss,
        step_losses=tuple(recorded_losses),
        eval_metrics=eval_history,
    )

  def compile_aot(
      self,
      topology: str | None = None,
      num_slices: int | None = None,
  ) -> AOTCompileResult:
    """Compiles the production train step offline on CPU against `topology` with zero weight allocation."""
    if self.model_config is None:
      raise ValueError("`compile_aot` requires `M3Trainer` to be constructed with a `ModelConfig` or preset name.")
    target_topology = topology or self.args.aot.compile_topology
    if not target_topology:
      raise ValueError("`topology` (or `args.aot.compile_topology`) must be non-empty for `compile_aot`.")
    target_slices = int(num_slices if num_slices is not None else self.args.aot.compile_topology_num_slices)
    if target_slices < 1:
      raise ValueError(f"`num_slices` must be >= 1; got {target_slices}.")

    topology_mesh = _build_topology_mesh(self.args, target_topology, target_slices)
    adapter_cfg = _ModelCreationAdapter(self.model_config, self.args)
    micro_b = max(1, int(round(self.args.per_device_batch_size * topology_mesh.size)))
    seq_len = self.args.max_target_length
    grad_accum = self.args.gradient_accumulation_steps
    grad_dtype = _resolve_jnp_dtype(self.args.grad_dtype)

    if self._check_invariants:
      verify_m3_model_invariants(
          lambda: m3_models.create_model(adapter_cfg, topology_mesh, rngs=nnx.Rngs(self.args.seed)),
          topology_mesh,
          batch_size=micro_b,
          seq_len=min(16, seq_len),
      )

    def _init_abstract_pair() -> tuple[nnx.Module, nnx.Optimizer]:
      return m3_loop.setup_model(adapter_cfg, topology_mesh, tx=self.tx)

    with jax.set_mesh(topology_mesh), jax.default_matmul_precision(self.args.matmul_precision.value):
      abs_model, abs_opt = nnx.eval_shape(_init_abstract_pair)
      graphdef_m, abs_state_m = nnx.split(abs_model)
      graphdef_o, abs_state_o = nnx.split(abs_opt)

      model_shardings = jax.tree.map(
          lambda v: _variable_named_sharding(v, topology_mesh), abs_state_m, is_leaf=_is_nnx_var
      )
      opt_shardings = jax.tree.map(lambda v: _variable_named_sharding(v, topology_mesh), abs_state_o, is_leaf=_is_nnx_var)

      batch_axes = tuple(ax for ax in ("data", "fsdp", "expert") if ax in topology_mesh.axis_names)
      ctx_axis = "context" if "context" in topology_mesh.axis_names else None
      if grad_accum > 1:
        tok_shape = (grad_accum, micro_b, seq_len)
        batch_spec = P(None, batch_axes, ctx_axis)
      else:
        tok_shape = (micro_b, seq_len)
        batch_spec = P(batch_axes, ctx_axis)
      batch_sharding = NamedSharding(topology_mesh, batch_spec)

      shaped_batch: M3Batch = {
          "inputs": jax.ShapeDtypeStruct(tok_shape, jnp.int32, sharding=batch_sharding),
          "targets": jax.ShapeDtypeStruct(tok_shape, jnp.int32, sharding=batch_sharding),
          "positions": jax.ShapeDtypeStruct(tok_shape, jnp.int32, sharding=batch_sharding),
          "segment_ids": jax.ShapeDtypeStruct(tok_shape, jnp.int32, sharding=batch_sharding),
          "weights": jax.ShapeDtypeStruct(tok_shape, jnp.float32, sharding=batch_sharding),
      }
      batch_in_shardings = jax.tree.map(lambda _: batch_sharding, shaped_batch)
      scalar_rep = NamedSharding(topology_mesh, P())

      def _functional_step(state_m: Any, state_o: Any, mb: M3Batch) -> tuple[Any, Any, jax.Array]:
        m = nnx.merge(graphdef_m, state_m)
        o = nnx.merge(graphdef_o, state_o)
        bd, grads = accumulated_loss_and_grads(
            m,
            mb,
            num_vocab_tiles=self.args.num_vocab_tiles,
            z_loss_multiplier=self.args.z_loss_multiplier,
            aux_loss_weight=self.args.aux_loss_weight,
            grad_dtype=grad_dtype,
        )
        o.update(m, grads)
        if self.args.router_bias_update_rate > 0.0:
          apply_router_bias_update(m, bd.router_token_counts, self.args.router_bias_update_rate)
        _, next_state_m = nnx.split(m)
        _, next_state_o = nnx.split(o)
        return next_state_m, next_state_o, bd.total_loss

      jitted = jax.jit(
          _functional_step,
          in_shardings=(model_shardings, opt_shardings, batch_in_shardings),
          out_shardings=(model_shardings, opt_shardings, scalar_rep),
          donate_argnums=(0, 1),
      )
      lowered = jitted.lower(abs_state_m, abs_state_o, shaped_batch)
      compiled = lowered.compile()

    out_file = self.args.aot.compiled_trainstep_file
    if out_file:
      os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
      serialized, _, _ = serialize(compiled)
      with open(out_file, "wb") as f:
        f.write(serialized)

    return AOTCompileResult(
        compiled=compiled,
        memory_analysis=compiled.memory_analysis(),
        cost_analysis=compiled.cost_analysis(),
        mesh=topology_mesh,
    )


def train_from_pyconfig(config: Any, recorder: Any = None) -> TrainResult:
  """Translates `config` via `from_pyconfig` and runs `M3Trainer.train()`."""
  del recorder  # Reserved for optional Goodput event integration.
  model_cfg, train_cfg = from_pyconfig(config)
  trainer = M3Trainer(model=model_cfg, args=train_cfg)
  return trainer.train()


def compile_from_pyconfig(config: Any) -> AOTCompileResult:
  """Translates `config` via `from_pyconfig` and runs `M3Trainer.compile_aot()`."""
  model_cfg, train_cfg = from_pyconfig(config)
  trainer = M3Trainer(model=model_cfg, args=train_cfg)
  result = trainer.compile_aot()
  peak_bytes = getattr(result.memory_analysis, "peak_memory_in_bytes", None)
  if peak_bytes:
    max_logging.log(f"m3 compile_aot peak HBM: {peak_bytes / 2**30:.2f} GiB")
  return result


def _is_m3_trainer_model(model_name: str) -> bool:
  """Returns `True` when `model_name` belongs to `MODEL_GARDEN_REGISTRY` (or is unregistered everywhere)."""
  _ensure_builtin_presets_registered()
  return model_name in MODEL_GARDEN_REGISTRY or model_name not in m3_models.MODEL_REGISTRY


def maybe_train_from_pyconfig(config: Any, recorder: Any = None) -> bool:
  """Dispatches `config` to `M3Trainer.train()` when `config.model_name` targets `M3Trainer`."""
  model_name = str(getattr(config, "model_name", ""))
  if not _is_m3_trainer_model(model_name):
    return False
  train_from_pyconfig(config, recorder=recorder)
  return True


def maybe_compile_from_pyconfig(config: Any) -> bool:
  """Dispatches `config` to `M3Trainer.compile_aot()` when `config.model_name` targets `M3Trainer`."""
  model_name = str(getattr(config, "model_name", ""))
  if not _is_m3_trainer_model(model_name):
    return False
  compile_from_pyconfig(config)
  return True


__all__ = [
    "AOTCompileResult",
    "M3Trainer",
    "TrainResult",
    "adapt_iterator_to_m3_batch",
    "build_lr_schedule",
    "build_m3_optimizer",
    "compile_from_pyconfig",
    "maybe_compile_from_pyconfig",
    "maybe_train_from_pyconfig",
    "train_from_pyconfig",
]
