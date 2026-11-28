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

"""Side-by-side performance and memory parity harness for m3 vs. legacy MaxText.

Runs the same MaxText training configuration twice -- once with `use_m3_model=false`
(legacy baseline) and once with `use_m3_model=true` (m3 candidate) -- and compares:
  * Steady-state step time (mean & median, pipelined 1 step ahead after warmup)
  * Throughput (tokens/s/device), TFLOP/s/device, and optional MFU
  * Compiled HBM breakdown (`compiled.memory_analysis()`)
  * Compiled HLO collective counts (`all-gather`, `reduce-scatter`, `all-reduce`, `all-to-all`)
  * Parameter count and finite-loss sanity

Evaluates the m3 cutover parity gates (`>= 98%` throughput and `<= 102%` peak HBM by default).

Two m3 drivers are supported via `--m3_driver`:
  * `loop` (default): compares the m3 training loop (`maxtext.m3.train.loop`: `setup_model` +
    `train_step`) against legacy `maxtext.trainers.pre_train.train`.
  * `train`: runs both legacy (`use_m3_model=false`) and m3 (`use_m3_model=true`) through
    `maxtext.trainers.pre_train.train` to isolate model and sharding differences inside `train.py`.

Example (CPU or TPU)::

    python3 -m maxtext.m3.train.perf_parity \\
      --warmup_steps=2 --measure_steps=5 \\
      -- src/maxtext/configs/base.yml model_name=qwen3-0.6b \\
         per_device_batch_size=2 max_target_length=2048 \\
         ici_fsdp_parallelism=-1 scan_layers=true

Pass `--compile_only` to skip step execution and compare only parameter counts, compiled HBM,
and HLO collectives. Pass `--enforce_gates` to exit non-zero when any parity gate fails.
"""

from collections.abc import Sequence
from dataclasses import asdict, dataclass
import gc
import json
import math
import os
import re
import statistics
import sys
import time
from typing import Any

from absl import app
from absl import flags
from flax import nnx
from flax.core.spmd import logical_axis_rules
import jax
import jax.numpy as jnp

from maxtext.common.goodput import create_goodput_recorder
from maxtext.configs import pyconfig
from maxtext.m3 import models as m3_models
from maxtext.m3.core import kernels as m3_kernels
from maxtext.m3.core.sharding import mesh_from_config, sharding_from_config
from maxtext.m3.train import loop as m3_loop
from maxtext.trainers.pre_train import train as legacy_train
from maxtext.utils import max_logging
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils
from maxtext.utils import sharding as legacy_sharding
from maxtext.utils import train_utils

_WARMUP_STEPS = flags.DEFINE_integer("warmup_steps", 2, "Number of unmeasured warmup steps per backend.")
_MEASURE_STEPS = flags.DEFINE_integer("measure_steps", 5, "Number of steady-state timed steps per backend.")
_M3_DRIVER = flags.DEFINE_enum(
    "m3_driver",
    "loop",
    ["loop", "train"],
    "Driver for the m3 run: 'loop' (m3/train/loop.py, default) or 'train' (legacy train.py with use_m3_model=true).",
)
_COMPILE_ONLY = flags.DEFINE_boolean(
    "compile_only",
    False,
    "Only compile both train steps and compare parameter counts, compiled HBM, and HLO collectives.",
)
_MIN_THROUGHPUT_RATIO = flags.DEFINE_float(
    "min_throughput_ratio",
    0.98,
    "Minimum allowed m3/legacy tokens-per-second ratio for the throughput parity gate.",
)
_MAX_HBM_RATIO = flags.DEFINE_float(
    "max_hbm_ratio",
    1.02,
    "Maximum allowed m3/legacy HBM ratio for the memory parity gate.",
)
_PEAK_TFLOPS = flags.DEFINE_float(
    "peak_tflops",
    None,
    "Optional per-device peak BF16/compute TFLOP/s for MFU calculation (e.g., 459 for v5p, 918 for v6e).",
)
_OUTPUT_JSON = flags.DEFINE_string(
    "output_json",
    "",
    "Optional path to write the machine-readable parity report JSON (process 0 only).",
)
_ENFORCE_GATES = flags.DEFINE_boolean(
    "enforce_gates",
    False,
    "Exit with status 1 if any parity gate fails.",
)

# Match both synchronous collectives (`all-gather(`) and async starts (`all-gather-start(`),
# without double-counting `-done(` instructions.
_COLLECTIVE_RE = re.compile(r"\b(all-gather|reduce-scatter|all-reduce|all-to-all)(?:-start)?\(")

# Overrides forced onto both legacy and m3 runs to keep the harness hermetic and side-effect free.
_BENCHMARK_CONFIG_OVERRIDES: dict[str, str] = {
    "dataset_type": "synthetic",
    "enable_checkpointing": "false",
    "enable_tensorboard": "false",
    "enable_goodput_recording": "false",
    "eval_interval": "0",
}


@dataclass(frozen=True)
class CompiledMemoryStats:
  """Per-device memory breakdown from `compiled.memory_analysis()`."""

  argument_bytes: int
  output_bytes: int
  temp_bytes: int
  alias_bytes: int
  host_temp_bytes: int
  compiled_peak_bytes: int | None

  @property
  def total_estimated_bytes(self) -> int:
    """Total estimated bytes using the same formula as `max_utils.print_compiled_memory_stats`."""
    return self.output_bytes + self.temp_bytes + self.argument_bytes - self.alias_bytes

  @property
  def effective_peak_bytes(self) -> int:
    """Compiler-reported peak HBM when available, otherwise `total_estimated_bytes`."""
    return self.compiled_peak_bytes if self.compiled_peak_bytes is not None else self.total_estimated_bytes


@dataclass(frozen=True)
class HloCollectives:
  """Collective op counts extracted from compiled HLO text."""

  all_gather: int = 0
  reduce_scatter: int = 0
  all_reduce: int = 0
  all_to_all: int = 0

  @property
  def total(self) -> int:
    return self.all_gather + self.reduce_scatter + self.all_reduce + self.all_to_all


@dataclass(frozen=True)
class BackendResult:
  """Measured compilation, memory, collective, and steady-state throughput metrics for one backend."""

  label: str
  driver: str
  model_name: str
  num_devices: int
  platform: str
  num_params: int
  compile_time_s: float
  compiled_memory: CompiledMemoryStats | None
  hlo_collectives: HloCollectives
  warmup_steps: int
  measure_steps: int
  mean_step_time_s: float | None
  median_step_time_s: float | None
  tokens_per_sec_per_device: float | None
  tflops_per_sec_per_device: float | None
  mfu: float | None
  first_loss: float | None
  last_loss: float | None

  @property
  def hbm_for_gate(self) -> tuple[str, int] | None:
    """Returns `(metric_name, bytes)` using per-executable compiled HBM.

    Prefers XLA's `compiled_peak_bytes` when it accounts for temporaries (`>= temp_bytes`, true on TPU XLA),
    otherwise `total_estimated_bytes` (`output + temp + argument - alias`, used on CPU XLA where
    `peak_memory_in_bytes` omits temporaries).
    """
    if self.compiled_memory is not None:
      peak = self.compiled_memory.compiled_peak_bytes
      if peak is not None and peak > 0 and peak >= self.compiled_memory.temp_bytes:
        return ("compiled_peak_hbm", peak)
      if self.compiled_memory.total_estimated_bytes > 0:
        return ("compiled_total_hbm", self.compiled_memory.total_estimated_bytes)
    return None


@dataclass(frozen=True)
class ParityGates:
  """Pass/fail thresholds for m3 vs. legacy parity."""

  min_throughput_ratio: float = 0.98
  max_hbm_ratio: float = 1.02
  require_finite_loss: bool = True
  require_param_count_match: bool = True


@dataclass(frozen=True)
class GateOutcome:
  """Result of evaluating one parity gate."""

  name: str
  passed: bool
  detail: str


@dataclass(frozen=True)
class ParityReport:
  """Complete comparison between legacy and m3 runs."""

  legacy: BackendResult
  m3: BackendResult
  gates: ParityGates
  throughput_ratio: float | None
  step_time_ratio: float | None
  hbm_metric_name: str | None
  hbm_ratio: float | None
  outcomes: tuple[GateOutcome, ...]

  @property
  def passed(self) -> bool:
    return all(outcome.passed for outcome in self.outcomes)

  def to_dict(self) -> dict[str, Any]:
    """Serializes the full report, including computed summary properties."""
    data = asdict(self)
    data["passed"] = self.passed
    if self.legacy.compiled_memory is not None:
      data["legacy"]["compiled_memory"]["total_estimated_bytes"] = self.legacy.compiled_memory.total_estimated_bytes
    if self.m3.compiled_memory is not None:
      data["m3"]["compiled_memory"]["total_estimated_bytes"] = self.m3.compiled_memory.total_estimated_bytes
    return data


def override_argv(base_argv: Sequence[str], overrides: dict[str, str]) -> list[str]:
  """Returns a copy of `base_argv` with `overrides` replacing any existing `key=...` entries."""
  if len(base_argv) < 2:
    raise ValueError(
        "Expected at least a program name and config YAML path in argv "
        f"(e.g. ['perf_parity', 'src/maxtext/configs/base.yml', ...]), got {list(base_argv)!r}."
    )
  override_keys = set(overrides)
  filtered = [base_argv[0], base_argv[1]]
  for arg in base_argv[2:]:
    key, sep, _ = arg.partition("=")
    if sep and key in override_keys:
      continue
    filtered.append(arg)
  for key, value in overrides.items():
    filtered.append(f"{key}={value}")
  return filtered


def extract_compiled_memory(compiled: Any) -> CompiledMemoryStats | None:
  """Extracts per-device memory breakdown from a compiled JAX executable."""
  stats = compiled.memory_analysis()
  if stats is None:
    return None
  peak_raw = getattr(stats, "peak_memory_in_bytes", None)
  peak_bytes = int(peak_raw) if peak_raw is not None and int(peak_raw) > 0 else None
  return CompiledMemoryStats(
      argument_bytes=int(stats.argument_size_in_bytes),
      output_bytes=int(stats.output_size_in_bytes),
      temp_bytes=int(stats.temp_size_in_bytes),
      alias_bytes=int(stats.alias_size_in_bytes),
      host_temp_bytes=int(stats.host_temp_size_in_bytes),
      compiled_peak_bytes=peak_bytes,
  )


def count_hlo_collectives(hlo_text: str) -> HloCollectives:
  """Counts collective communication ops in compiled HLO text."""
  counts = {"all-gather": 0, "reduce-scatter": 0, "all-reduce": 0, "all-to-all": 0}
  for match in _COLLECTIVE_RE.finditer(hlo_text):
    counts[match.group(1)] += 1
  return HloCollectives(
      all_gather=counts["all-gather"],
      reduce_scatter=counts["reduce-scatter"],
      all_reduce=counts["all-reduce"],
      all_to_all=counts["all-to-all"],
  )


def _make_train_batch_fn(config: Any, shaped_batch: dict[str, jax.ShapeDtypeStruct]):
  """Builds a once-compiled synthetic batch generator matching `shaped_batch` shardings."""
  names = sorted(shaped_batch)
  seq_len = int(config.max_target_length)
  vocab_size = max(int(config.vocab_size), 1)

  def _seq_axis(shape: tuple[int, ...]) -> int:
    axes = [i for i, n in enumerate(shape) if n == seq_len and i > 0]
    return axes[0] if axes else len(shape) - 1

  def _gen(key: jax.Array) -> dict[str, jax.Array]:
    out: dict[str, jax.Array] = {}
    keys = jax.random.split(key, len(names))
    for sub_key, name in zip(keys, names):
      shape, dtype = shaped_batch[name].shape, shaped_batch[name].dtype
      if name in ("inputs", "targets"):
        out[name] = jax.random.randint(sub_key, shape, 0, vocab_size, dtype=dtype)
      elif name.endswith("_position"):
        out[name] = jax.lax.broadcasted_iota(dtype, shape, _seq_axis(shape))
      elif jnp.issubdtype(dtype, jnp.floating):
        out[name] = jnp.zeros(shape, dtype)
      else:
        out[name] = jnp.ones(shape, dtype)
    return out

  out_shardings = {name: shaped_batch[name].sharding for name in names}
  return jax.jit(_gen, out_shardings=out_shardings)


def _compute_throughput_metrics(
    config: Any,
    mean_step_time_s: float,
    num_devices: int,
    peak_tflops: float | None,
) -> tuple[float, float, float | None]:
  """Computes `(tokens_per_sec_per_device, tflops_per_sec_per_device, mfu)`."""
  tokens_per_step = config.global_batch_size_to_train_on * config.max_target_length
  tokens_per_sec_per_device = tokens_per_step / (num_devices * mean_step_time_s)
  per_device_tflops, _, _ = maxtext_utils.calculate_tflops_training_per_device(config, log=False)
  tflops_per_sec_per_device = per_device_tflops / mean_step_time_s
  mfu = tflops_per_sec_per_device / peak_tflops if peak_tflops is not None else None
  return tokens_per_sec_per_device, tflops_per_sec_per_device, mfu


def _validate_run_params(
    warmup_steps: int,
    measure_steps: int,
    compile_only: bool,
    peak_tflops: float | None,
) -> None:
  """Validates step-count and hardware-peak arguments before compiling."""
  if warmup_steps < 0:
    raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}.")
  if not compile_only and measure_steps < 1:
    raise ValueError(f"measure_steps must be >= 1 when compile_only is False, got {measure_steps}.")
  if peak_tflops is not None and peak_tflops <= 0:
    raise ValueError(f"peak_tflops must be > 0, got {peak_tflops}.")


def _run_pipelined_steps(
    dispatch_step: Any,
    *,
    warmup_steps: int,
    measure_steps: int,
    sync_state: Any = None,
) -> tuple[float, float, float, float]:
  """Runs warmup then steady-state steps pipelined 1 step ahead.

  Args:
    dispatch_step: Callable `(step_idx: int) -> jax.Array` dispatching one step and returning its scalar loss.
    warmup_steps: Unmeasured warmup iterations (compilation + allocator warmup).
    measure_steps: Measured steady-state iterations.
    sync_state: Optional zero-arg callable that blocks on mutated state at warmup/measure boundaries.

  Returns:
    `(mean_step_time_s, median_step_time_s, first_loss, last_loss)`.
  """
  step_idx = 0
  last_warm_loss: jax.Array | None = None
  for _ in range(warmup_steps):
    last_warm_loss = dispatch_step(step_idx)
    step_idx += 1
  if sync_state is not None:
    sync_state()
  elif last_warm_loss is not None:
    jax.block_until_ready(last_warm_loss)

  step_times: list[float] = []
  pending_loss: jax.Array | None = None
  first_loss: float | None = None
  last_loss: float | None = None
  t_start = time.perf_counter()
  last_done = t_start
  for _ in range(measure_steps):
    step_loss = dispatch_step(step_idx)
    step_idx += 1
    if pending_loss is not None:
      loss_val = float(pending_loss)
      now = time.perf_counter()
      step_times.append(now - last_done)
      last_done = now
      if first_loss is None:
        first_loss = loss_val
      last_loss = loss_val
    pending_loss = step_loss
  if pending_loss is not None:
    loss_val = float(pending_loss)
    if sync_state is not None:
      sync_state()
    now = time.perf_counter()
    step_times.append(now - last_done)
    if first_loss is None:
      first_loss = loss_val
    last_loss = loss_val
  total_elapsed = time.perf_counter() - t_start
  assert first_loss is not None and last_loss is not None
  return total_elapsed / measure_steps, statistics.median(step_times), first_loss, last_loss


def run_train_backend(
    config: Any,
    *,
    label: str,
    warmup_steps: int = 2,
    measure_steps: int = 5,
    compile_only: bool = False,
    peak_tflops: float | None = None,
) -> BackendResult:
  """Compiles and benchmarks `maxtext.trainers.pre_train.train.train_step` for `config`."""
  _validate_run_params(warmup_steps, measure_steps, compile_only, peak_tflops)

  state = None
  compiled = None
  p_train_step = None
  model = None
  data_iterator = None
  data_loader = None
  checkpoint_manager = None
  try:
    recorder = create_goodput_recorder(config)
    (
        _,
        checkpoint_manager,
        state_mesh_shardings,
        model,
        mesh,
        _,
        data_iterator,
        data_loader,
        _,
        _,
        state,
    ) = train_utils.setup_train_loop(config, recorder)

    _, setup_params, _ = nnx.split(state.model, nnx.Param, ...)
    num_params = int(max_utils.calculate_num_params_from_pytree(setup_params))
    del setup_params

    jit_model, state = nnx.split(state)
    params_shardings, state_mesh_shardings = legacy_sharding.maybe_update_params_sharding_with_opt(
        config, state_mesh_shardings
    )
    p_train_step, _ = train_utils.jit_train_and_eval_step(
        config,
        jit_model,
        mesh,
        state,
        state_mesh_shardings,
        legacy_train.train_step,
        eval_step=None,
        eval_data_iterator=None,
        params_shardings=params_shardings,
    )

    with jax.set_mesh(mesh), logical_axis_rules(config.logical_axis_rules):
      data_sharding = legacy_sharding.get_input_data_sharding(config, mesh)
      shaped_batch = maxtext_utils.get_shaped_batch(config, batch_sharding=data_sharding)
      if config.shard_optimizer_over_data:
        state = legacy_sharding.reshard_train_state(state, state_mesh_shardings, config.enable_single_controller)
      compiler_options = max_utils.parse_libtpu_flags_to_dict(config.compile_xla_flags)
      t_compile_start = time.perf_counter()
      compiled = p_train_step.lower(state, shaped_batch).compile(compiler_options=compiler_options or None)
      compile_time_s = time.perf_counter() - t_compile_start

    compiled_memory = extract_compiled_memory(compiled)
    hlo_collectives = count_hlo_collectives(compiled.as_text())
    num_devices = len(jax.devices())
    platform = jax.devices()[0].platform

    mean_step_s = median_step_s = tokens_per_sec = tflops_per_sec = mfu = first_loss = last_loss = None

    if not compile_only:
      batch_fn = _make_train_batch_fn(config, shaped_batch)
      data_key = jax.random.key(config.init_weights_seed)

      def _dispatch(step_idx: int) -> jax.Array:
        nonlocal state
        batch = batch_fn(jax.random.fold_in(data_key, step_idx))
        state, metrics = compiled(state, batch)
        return metrics["scalar"]["learning/loss"]

      with jax.set_mesh(mesh), logical_axis_rules(config.logical_axis_rules):
        mean_step_s, median_step_s, first_loss, last_loss = _run_pipelined_steps(
            _dispatch,
            warmup_steps=warmup_steps,
            measure_steps=measure_steps,
            sync_state=lambda: jax.block_until_ready(state),
        )
      tokens_per_sec, tflops_per_sec, mfu = _compute_throughput_metrics(config, mean_step_s, num_devices, peak_tflops)

    return BackendResult(
        label=label,
        driver="train",
        model_name=config.model_name,
        num_devices=num_devices,
        platform=platform,
        num_params=num_params,
        compile_time_s=compile_time_s,
        compiled_memory=compiled_memory,
        hlo_collectives=hlo_collectives,
        warmup_steps=warmup_steps,
        measure_steps=0 if compile_only else measure_steps,
        mean_step_time_s=mean_step_s,
        median_step_time_s=median_step_s,
        tokens_per_sec_per_device=tokens_per_sec,
        tflops_per_sec_per_device=tflops_per_sec,
        mfu=mfu,
        first_loss=first_loss,
        last_loss=last_loss,
    )
  finally:
    del state, compiled, p_train_step, model, data_iterator, data_loader, checkpoint_manager
    gc.collect()


def _validate_loop_config(config: Any) -> None:
  """Rejects config options that `maxtext.m3.train.loop` does not implement."""
  if config.gradient_accumulation_steps > 1:
    raise ValueError(
        "m3 loop driver does not support gradient accumulation; "
        f"got gradient_accumulation_steps={config.gradient_accumulation_steps}."
    )
  if jnp.dtype(config.grad_dtype) != jnp.float32:
    raise ValueError(f"m3 loop driver keeps gradients in float32; got grad_dtype={jnp.dtype(config.grad_dtype).name}.")


def run_m3_loop_backend(
    config: Any,
    *,
    label: str = "m3",
    warmup_steps: int = 2,
    measure_steps: int = 5,
    compile_only: bool = False,
    peak_tflops: float | None = None,
) -> BackendResult:
  """Compiles and benchmarks `maxtext.m3.train.loop` (`setup_model` + `train_step`) for `config`."""
  _validate_run_params(warmup_steps, measure_steps, compile_only, peak_tflops)
  _validate_loop_config(config)

  model = None
  optimizer = None
  compiled = None
  try:
    mesh = mesh_from_config(config)
    m3_kernels.validate_attention_config(config, mesh=mesh)
    sharding = sharding_from_config(config)
    spec = m3_loop.batch_spec(sharding)
    matmul_precision = getattr(config.matmul_precision, "value", config.matmul_precision)
    num_devices = len(jax.devices())
    platform = jax.devices()[0].platform

    with jax.set_mesh(mesh), jax.default_matmul_precision(matmul_precision):
      model, optimizer = m3_loop.setup_model(config, mesh)
      num_params = int(sum(x.size for x in jax.tree.leaves(nnx.state(model, nnx.Param))))
      data_key = jax.random.key(config.init_weights_seed)
      sample_batch = m3_loop.make_synthetic_batch(config, mesh, spec, data_key)
      compiler_options = max_utils.parse_libtpu_flags_to_dict(config.compile_xla_flags)

      t_compile_start = time.perf_counter()
      compiled = m3_loop.train_step.lower(model, optimizer, sample_batch).compile(
          compiler_options=compiler_options or None
      )
      compile_time_s = time.perf_counter() - t_compile_start

      compiled_memory = extract_compiled_memory(compiled)
      hlo_collectives = count_hlo_collectives(compiled.as_text())

      mean_step_s = median_step_s = tokens_per_sec = tflops_per_sec = mfu = first_loss = last_loss = None

      if not compile_only:
        step_fn = nnx.cached_partial(compiled, model, optimizer)

        def _dispatch(step_idx: int) -> jax.Array:
          batch = m3_loop.make_synthetic_batch(config, mesh, spec, jax.random.fold_in(data_key, step_idx))
          return step_fn(batch)

        mean_step_s, median_step_s, first_loss, last_loss = _run_pipelined_steps(
            _dispatch,
            warmup_steps=warmup_steps,
            measure_steps=measure_steps,
            sync_state=lambda: jax.block_until_ready(nnx.state((model, optimizer))),
        )
        tokens_per_sec, tflops_per_sec, mfu = _compute_throughput_metrics(config, mean_step_s, num_devices, peak_tflops)

    return BackendResult(
        label=label,
        driver="loop",
        model_name=config.model_name,
        num_devices=num_devices,
        platform=platform,
        num_params=num_params,
        compile_time_s=compile_time_s,
        compiled_memory=compiled_memory,
        hlo_collectives=hlo_collectives,
        warmup_steps=warmup_steps,
        measure_steps=0 if compile_only else measure_steps,
        mean_step_time_s=mean_step_s,
        median_step_time_s=median_step_s,
        tokens_per_sec_per_device=tokens_per_sec,
        tflops_per_sec_per_device=tflops_per_sec,
        mfu=mfu,
        first_loss=first_loss,
        last_loss=last_loss,
    )
  finally:
    del model, optimizer, compiled
    gc.collect()


def evaluate_parity(
    legacy: BackendResult,
    m3: BackendResult,
    gates: ParityGates = ParityGates(),
) -> ParityReport:
  """Computes ratios and evaluates parity gates between `legacy` and `m3` results."""
  outcomes: list[GateOutcome] = []

  if gates.require_param_count_match:
    matched = legacy.num_params == m3.num_params
    outcomes.append(
        GateOutcome(
            name="param_count",
            passed=matched,
            detail=f"m3={m3.num_params:,} vs legacy={legacy.num_params:,}",
        )
    )

  throughput_ratio: float | None = None
  step_time_ratio: float | None = None
  if legacy.tokens_per_sec_per_device is not None and m3.tokens_per_sec_per_device is not None:
    throughput_ratio = m3.tokens_per_sec_per_device / legacy.tokens_per_sec_per_device
    step_time_ratio = (
        m3.mean_step_time_s / legacy.mean_step_time_s
        if legacy.mean_step_time_s and m3.mean_step_time_s is not None
        else None
    )
    outcomes.append(
        GateOutcome(
            name="throughput",
            passed=throughput_ratio >= gates.min_throughput_ratio,
            detail=(
                f"ratio={throughput_ratio:.4f} "
                f"(gate >= {gates.min_throughput_ratio:.2f}; "
                f"m3={m3.tokens_per_sec_per_device:,.1f} vs legacy={legacy.tokens_per_sec_per_device:,.1f} tok/s/dev)"
            ),
        )
    )

  if gates.require_finite_loss and (legacy.last_loss is not None or m3.last_loss is not None):
    losses = [legacy.first_loss, legacy.last_loss, m3.first_loss, m3.last_loss]
    all_finite = all(val is not None and math.isfinite(val) for val in losses)
    outcomes.append(
        GateOutcome(
            name="finite_loss",
            passed=all_finite,
            detail=f"legacy={legacy.last_loss}, m3={m3.last_loss}",
        )
    )

  hbm_metric_name: str | None = None
  hbm_ratio: float | None = None
  hbm_pair: tuple[str, int, int] | None = None
  legacy_hbm = legacy.hbm_for_gate
  m3_hbm = m3.hbm_for_gate
  if legacy_hbm is not None and m3_hbm is not None and legacy_hbm[0] == m3_hbm[0]:
    hbm_pair = (legacy_hbm[0], legacy_hbm[1], m3_hbm[1])
  elif legacy.compiled_memory is not None and m3.compiled_memory is not None:
    if legacy.compiled_memory.total_estimated_bytes > 0:
      hbm_pair = (
          "compiled_total_hbm",
          legacy.compiled_memory.total_estimated_bytes,
          m3.compiled_memory.total_estimated_bytes,
      )
  if hbm_pair is not None:
    hbm_metric_name, legacy_bytes, m3_bytes = hbm_pair
    hbm_ratio = m3_bytes / legacy_bytes
    outcomes.append(
        GateOutcome(
            name="hbm",
            passed=hbm_ratio <= gates.max_hbm_ratio,
            detail=(
                f"{hbm_metric_name} ratio={hbm_ratio:.4f} "
                f"(gate <= {gates.max_hbm_ratio:.2f}; "
                f"m3={_fmt_bytes(m3_bytes)} vs legacy={_fmt_bytes(legacy_bytes)})"
            ),
        )
    )

  return ParityReport(
      legacy=legacy,
      m3=m3,
      gates=gates,
      throughput_ratio=throughput_ratio,
      step_time_ratio=step_time_ratio,
      hbm_metric_name=hbm_metric_name,
      hbm_ratio=hbm_ratio,
      outcomes=tuple(outcomes),
  )


def _fmt_bytes(num_bytes: int | None) -> str:
  if num_bytes is None:
    return "n/a"
  gib = num_bytes / 2**30
  if gib >= 0.1:
    return f"{gib:.3f} GiB"
  return f"{num_bytes / 2**20:.2f} MiB"


def _fmt_ratio(m3_val: float | None, legacy_val: float | None) -> str:
  if m3_val is None or legacy_val is None or legacy_val == 0:
    return "n/a"
  return f"{m3_val / legacy_val:.3f}x"


def format_report(report: ParityReport) -> str:
  """Formats a human-readable side-by-side parity table and gate summary."""
  leg = report.legacy
  m3 = report.m3
  rows: list[tuple[str, str, str, str]] = [
      ("Driver", leg.driver, m3.driver, "-"),
      ("Parameters", f"{leg.num_params:,}", f"{m3.num_params:,}", _fmt_ratio(m3.num_params, leg.num_params)),
      (
          "Compile time",
          f"{leg.compile_time_s:.2f}s",
          f"{m3.compile_time_s:.2f}s",
          _fmt_ratio(m3.compile_time_s, leg.compile_time_s),
      ),
  ]
  if leg.mean_step_time_s is not None and m3.mean_step_time_s is not None:
    rows.extend(
        [
            (
                "Step time (mean)",
                f"{leg.mean_step_time_s * 1000:.2f} ms",
                f"{m3.mean_step_time_s * 1000:.2f} ms",
                _fmt_ratio(m3.mean_step_time_s, leg.mean_step_time_s),
            ),
            (
                "Step time (median)",
                f"{leg.median_step_time_s * 1000:.2f} ms",
                f"{m3.median_step_time_s * 1000:.2f} ms",
                _fmt_ratio(m3.median_step_time_s, leg.median_step_time_s),
            ),
            (
                "Throughput (tok/s/dev)",
                f"{leg.tokens_per_sec_per_device:,.1f}",
                f"{m3.tokens_per_sec_per_device:,.1f}",
                _fmt_ratio(m3.tokens_per_sec_per_device, leg.tokens_per_sec_per_device),
            ),
            (
                "Compute (TFLOP/s/dev)",
                f"{leg.tflops_per_sec_per_device:,.2f}",
                f"{m3.tflops_per_sec_per_device:,.2f}",
                _fmt_ratio(m3.tflops_per_sec_per_device, leg.tflops_per_sec_per_device),
            ),
        ]
    )
    if leg.mfu is not None and m3.mfu is not None:
      rows.append(("MFU", f"{leg.mfu:.2%}", f"{m3.mfu:.2%}", _fmt_ratio(m3.mfu, leg.mfu)))
    rows.append(("Last loss", f"{leg.last_loss:.4f}", f"{m3.last_loss:.4f}", "-"))

  leg_mem = leg.compiled_memory
  m3_mem = m3.compiled_memory
  rows.extend(
      [
          (
              "Compiled total HBM",
              _fmt_bytes(leg_mem.total_estimated_bytes if leg_mem else None),
              _fmt_bytes(m3_mem.total_estimated_bytes if m3_mem else None),
              _fmt_ratio(
                  m3_mem.total_estimated_bytes if m3_mem else None,
                  leg_mem.total_estimated_bytes if leg_mem else None,
              ),
          ),
          (
              "Compiled temp HBM",
              _fmt_bytes(leg_mem.temp_bytes if leg_mem else None),
              _fmt_bytes(m3_mem.temp_bytes if m3_mem else None),
              _fmt_ratio(m3_mem.temp_bytes if m3_mem else None, leg_mem.temp_bytes if leg_mem else None),
          ),
          (
              "Compiled peak HBM",
              _fmt_bytes(leg_mem.compiled_peak_bytes if leg_mem else None),
              _fmt_bytes(m3_mem.compiled_peak_bytes if m3_mem else None),
              _fmt_ratio(
                  m3_mem.compiled_peak_bytes if m3_mem else None,
                  leg_mem.compiled_peak_bytes if leg_mem else None,
              ),
          ),
          (
              "HLO all-gather",
              str(leg.hlo_collectives.all_gather),
              str(m3.hlo_collectives.all_gather),
              f"{m3.hlo_collectives.all_gather - leg.hlo_collectives.all_gather:+d}",
          ),
          (
              "HLO reduce-scatter",
              str(leg.hlo_collectives.reduce_scatter),
              str(m3.hlo_collectives.reduce_scatter),
              f"{m3.hlo_collectives.reduce_scatter - leg.hlo_collectives.reduce_scatter:+d}",
          ),
          (
              "HLO all-reduce",
              str(leg.hlo_collectives.all_reduce),
              str(m3.hlo_collectives.all_reduce),
              f"{m3.hlo_collectives.all_reduce - leg.hlo_collectives.all_reduce:+d}",
          ),
          (
              "HLO all-to-all",
              str(leg.hlo_collectives.all_to_all),
              str(m3.hlo_collectives.all_to_all),
              f"{m3.hlo_collectives.all_to_all - leg.hlo_collectives.all_to_all:+d}",
          ),
      ]
  )

  header = f"m3 vs. legacy parity: {m3.model_name} on {m3.num_devices}x {m3.platform} (m3_driver={m3.driver})"
  lines = [header, "=" * len(header)]
  col_widths = [max(len(r[i]) for r in [("Metric", "Legacy", "m3", "m3 / Legacy"), *rows]) for i in range(4)]
  fmt = f"{{:<{col_widths[0]}}}  {{:>{col_widths[1]}}}  {{:>{col_widths[2]}}}  {{:>{col_widths[3]}}}"
  lines.append(fmt.format("Metric", "Legacy", "m3", "m3 / Legacy"))
  lines.append("-" * (sum(col_widths) + 6))
  for row in rows:
    lines.append(fmt.format(*row))
  lines.append("")
  lines.append(f"Gates overall: {'PASS' if report.passed else 'FAIL'}")
  for outcome in report.outcomes:
    status = "PASS" if outcome.passed else "FAIL"
    lines.append(f"  [{status}] {outcome.name}: {outcome.detail}")
  return "\n".join(lines)


def run_parity_benchmark(
    base_argv: Sequence[str],
    *,
    m3_driver: str = "loop",
    warmup_steps: int = 2,
    measure_steps: int = 5,
    compile_only: bool = False,
    gates: ParityGates = ParityGates(),
    peak_tflops: float | None = None,
) -> ParityReport:
  """Runs legacy (`train.py`, `use_m3_model=false`) and m3 (`use_m3_model=true`) back-to-back and compares them."""
  if m3_driver not in ("loop", "train"):
    raise ValueError(f"m3_driver must be 'loop' or 'train', got {m3_driver!r}.")
  _validate_run_params(warmup_steps, measure_steps, compile_only, peak_tflops)

  legacy_argv = override_argv(base_argv, {**_BENCHMARK_CONFIG_OVERRIDES, "use_m3_model": "false"})
  m3_argv = override_argv(base_argv, {**_BENCHMARK_CONFIG_OVERRIDES, "use_m3_model": "true"})

  m3_config = pyconfig.initialize(m3_argv)
  if m3_config.model_name not in m3_models.MODEL_REGISTRY:
    raise ValueError(
        f"Model {m3_config.model_name!r} is not registered in maxtext.m3.models.MODEL_REGISTRY "
        f"(registered: {', '.join(sorted(m3_models.MODEL_REGISTRY))})."
    )
  if m3_driver == "loop":
    _validate_loop_config(m3_config)
  legacy_config = pyconfig.initialize(legacy_argv)

  max_logging.log(f"Running legacy baseline for {legacy_config.model_name} (driver=train)...")
  legacy_result = run_train_backend(
      legacy_config,
      label="legacy",
      warmup_steps=warmup_steps,
      measure_steps=measure_steps,
      compile_only=compile_only,
      peak_tflops=peak_tflops,
  )
  # Clear compilation caches between runs so legacy executables do not hold device memory during m3.
  jax.clear_caches()
  gc.collect()

  max_logging.log(f"Running m3 candidate for {m3_config.model_name} (driver={m3_driver})...")
  if m3_driver == "train":
    m3_result = run_train_backend(
        m3_config,
        label="m3",
        warmup_steps=warmup_steps,
        measure_steps=measure_steps,
        compile_only=compile_only,
        peak_tflops=peak_tflops,
    )
  else:
    m3_result = run_m3_loop_backend(
        m3_config,
        label="m3",
        warmup_steps=warmup_steps,
        measure_steps=measure_steps,
        compile_only=compile_only,
        peak_tflops=peak_tflops,
    )

  return evaluate_parity(legacy_result, m3_result, gates=gates)


def write_report_json(report: ParityReport, output_path: str) -> None:
  """Writes `report.to_dict()` as formatted JSON, creating parent directories if needed."""
  out_dir = os.path.dirname(output_path)
  if out_dir:
    os.makedirs(out_dir, exist_ok=True)
  with open(output_path, "w", encoding="utf-8") as f:
    json.dump(report.to_dict(), f, indent=2)


def main(argv: Sequence[str]) -> None:
  gates = ParityGates(
      min_throughput_ratio=_MIN_THROUGHPUT_RATIO.value,
      max_hbm_ratio=_MAX_HBM_RATIO.value,
  )
  report = run_parity_benchmark(
      argv,
      m3_driver=_M3_DRIVER.value,
      warmup_steps=_WARMUP_STEPS.value,
      measure_steps=_MEASURE_STEPS.value,
      compile_only=_COMPILE_ONLY.value,
      gates=gates,
      peak_tflops=_PEAK_TFLOPS.value,
  )
  max_logging.log("\n" + format_report(report))

  if _OUTPUT_JSON.value and jax.process_index() == 0:
    write_report_json(report, _OUTPUT_JSON.value)
    max_logging.log(f"Wrote parity report JSON to {_OUTPUT_JSON.value}")

  if _ENFORCE_GATES.value and not report.passed:
    sys.exit(1)


if __name__ == "__main__":
  app.run(main)
