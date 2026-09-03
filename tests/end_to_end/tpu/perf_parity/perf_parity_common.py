# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared rig for the trainer/model parity arms. Model-agnostic; `--model` picks the model.

Nothing here is specific to one model, which is why the file is not named after one. Three
of the four arms take `--model` and have been run at both sizes on this rig:

  * `qwen3-0.6b` (the default) -- the shape all four arms can run, and the only one the
    tunix arm can, since tunix implements that architecture and not the others. Runs
    unscanned at `--tp 8` on 8 devices.
  * `qwen3.5-35b-a3b` -- MaxText-side arms only, and always `--scan` (unscanned it OOMs).
    Two mesh shapes have been measured: `--tp 2`, and `--ep 8 --ring-of-experts
    --ragged-sort` over its 256 experts. `--tp 8` is rejected by the sharding checks
    because the model has 2 KV heads. See `RESULTS-qwen35-35b-20260902.md`.

`--ep` is the third mesh axis and it is what gates the two MoE kernel flags: MaxText reads
the EP rank off `logical_axis_rules` (`exp` -> the `expert` physical axis) rather than off
an int, and refuses `use_ring_of_experts` / `use_ragged_sort` when that rank is 1.

`qwen3_0p6b_tunix_profile.py` is the one arm named after a model, because it is the one arm
locked to one.

The gemma4-e2b pair before this had to truncate the model to 12 layers and guess at a
matching KV-sharing split, because the two repos parameterise that architecture
differently. qwen3-0.6b needs none of that: `tunix.models.qwen3.ModelConfig.qwen3_0p6b`
and MaxText's `configs/models/qwen3-0.6b.yml` describe the same 28-layer network field
for field -- vocab 151936, embed 1024, hidden 3072, 16 query heads, 8 KV heads, head_dim
128, norm eps 1e-6, rope theta 1e6, tied embeddings -- so both arms run the real model
with nothing cut.

Everything that is not the model lives here rather than being duplicated per arm, so
"only the model differs" is enforced by construction instead of by eyeballing a diff.

`SEQ_LEN` is 1024, not the 256 the gemma4 pair used. At 256 the step was far too small
to see past host and compile overhead -- the gemma4 traces showed a 64 ms device step
inside a 2.6 s wall step. 1024 puts roughly 30 TFLOPs of real work in each step, which
is enough for the device time to be the thing being measured.

Gradient accumulation means the same thing in both trainers, but neither is told it the
same way. Tunix reads `gradient_accumulation_steps` off its `TrainingConfig`, consumes
one dataset item per *micro* step and applies an update every `ga`-th one, and counts
`max_steps` in optimizer steps -- so a `ga`-way run needs `ga * steps` dataset items. The
MaxText engine reads no such config: `_micro_step_count` is driven purely by how many
`fwd_bwd` calls the caller makes before each `update()`. Both therefore run a global
batch of `batch * ga` per optimizer step, with the micro-batch held at `batch`.
"""

import argparse
import contextlib
import os
import statistics
import time
from typing import Any, List

import jax
import numpy as np
import optax
from tunix.rl import common
from tunix.sft import hooks
from tunix.sft import peft_trainer

LEARNING_RATE = 1e-5
BATCH_SIZE = 8
SEQ_LEN = 1024
# base.yml's adamw settings, restated so `--opt adamw` means the same thing on both sides.
# optax's own defaults differ on two of these (`b2=0.999`, `weight_decay=1e-4`), so leaving
# either side to its default would compare two different optimizers. `sgd` stays the rig
# default: it is what the tunix arms were written against and what every figure recorded
# before `--opt` existed was taken with.
ADAM_B1 = 0.9
ADAM_B2 = 0.95
ADAM_EPS = 1e-8
ADAM_WEIGHT_DECAY = 0.1
# Steps dropped from the steady-state figure: compilation lands in the first, and the
# next two cover the throttler filling its two-deep inflight queue.
WARMUP_STEPS = 3
BENCHMARK_STEPS = 20
MAX_STEPS = WARMUP_STEPS + BENCHMARK_STEPS
ACCUM_STEPS = 1
# qwen3-0.6b's full vocab, identical on both sides.
VOCAB_SIZE = 151936

TOKENIZER_ID = "Qwen/Qwen3-0.6B"
# The model both trainers can run. `--model` moves the MaxText-side arms off it; the tunix
# arm cannot follow, since tunix implements this one architecture and not the others.
MODEL_NAME = "qwen3-0.6b"

# Where each arm writes its xprof trace. Set `PERF_PARITY_PROFILE_ROOT` to a GCS bucket to
# collect them somewhere durable; the destination has no bearing on the measurement, and
# `StepTimer` below reports the step time without the profile being opened at all.
PROFILE_ROOT = os.environ.get("PERF_PARITY_PROFILE_ROOT", "perf_parity_traces")


def slug(model: str) -> str:
  """`qwen3.5-35b-a3b` -> `qwen3_5_35b_a3b`, for use in a MaxText `run_name`.

  `run_name` becomes a path component under `base_output_directory`, so the two models
  need to produce different ones -- otherwise a 35b run overwrites the 0.6b run's output.
  """
  return model.replace(".", "_").replace("-", "_")


def profile_dir(arm: str) -> str:
  """Trace destination for one arm, e.g. `qwen3-0.6b-tunix` or `qwen3.5-35b-a3b-engine-scan`."""
  return os.path.join(PROFILE_ROOT, arm)


# base.yml's `enable_tpu_profiling_options: true` block, verbatim -- see
# `maxtext.common.profiler.Profiler.__init__`. This rig calls `jax.profiler.trace` directly
# rather than going through `Profiler`, so the same advanced configuration has to be built
# here to get the same trace.
TPU_PROFILING_OPTIONS = {
    "tpu_num_chips_to_profile_per_task": 1,
    "tpu_num_sparse_core_tiles_to_trace": 1,
    "tpu_num_sparse_cores_to_trace": 2,
}


def maybe_trace(log_dir: str, spec: "RunSpec"):
  """The profiler around the timed loop, unless `--no-trace` asked for it to be left off.

  Tracing is not free and it is not free *evenly*: the profiler charges per dispatch, so an
  arm that issues dozens of tiny eager ops per step pays far more for being watched than one
  that issues two. MaxText's `update` reads 27.0 ms traced against 14.0 ms untraced where
  tunix goes 8.2 -> 18.5 ms, so a traced A/B flatters whichever side dispatches less. Quote
  `--no-trace` numbers for wall clock and traced ones for where the time went.

  `TPU_PROFILING_OPTIONS` is on by default because the default profiler *silently drops
  events* under `--ragged-sort`: the sort kernels emit on the order of a million SparseCore
  events per step, the device trace fills, and what lands is a clipped subset -- one
  `jit_accum_kernel` execution reading 225.99 ms against a real 724 ms, with no warning
  anywhere. Capping the SparseCore tiles and cores traced, and profiling one chip per task
  rather than all four, keeps the XLA-module timeline intact. Pass `--no-tpu-profiling-options`
  to go back to the unrestricted capture; the cap is a *coverage* restriction, so a trace
  taken with it holds one chip's device plane, which is the one these arms read anyway.
  """
  if not spec.trace:
    return contextlib.nullcontext()
  options = None
  if spec.tpu_profiling_options:
    options = jax.profiler.ProfileOptions()
    options.advanced_configuration = dict(TPU_PROFILING_OPTIONS)
  print(f"tracing to {log_dir}  (advanced tpu options: {spec.tpu_profiling_options})", flush=True)
  return jax.profiler.trace(log_dir=log_dir, profiler_options=options)


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
  """Adds the flags every arm honours. Defaults reproduce the original 4-device run."""
  parser.add_argument("--ga", type=int, default=ACCUM_STEPS, help="micro-batches per optimizer step")
  parser.add_argument("--devices", type=int, default=None, help="use only the first N local devices")
  parser.add_argument("--dp", type=int, default=1, help="pure data-parallel mesh axis; Zero-1 shards over this one")
  parser.add_argument("--fsdp", type=int, default=None, help="fsdp mesh axis (default: fills the devices)")
  parser.add_argument("--tp", type=int, default=1, help="tensor-parallel mesh axis")
  parser.add_argument("--ep", type=int, default=1, help="expert-parallel mesh axis; MoE models only")
  parser.add_argument(
      "--ring-of-experts",
      dest="ring_of_experts",
      action="store_true",
      help="MoE dispatch as a ring over the expert axis instead of all-to-all; needs --ep > 1",
  )
  parser.add_argument(
      "--ragged-sort",
      dest="ragged_sort",
      action="store_true",
      help="Pallas ragged-sort kernels in the MoE permute path; needs --ep > 1",
  )
  parser.add_argument("--batch", type=int, default=BATCH_SIZE, help="examples per micro-batch")
  parser.add_argument("--seq", type=int, default=SEQ_LEN, help="tokens per example")
  parser.add_argument("--steps", type=int, default=MAX_STEPS, help="optimizer steps, warmup included")
  parser.add_argument("--no-trace", dest="trace", action="store_false", help="skip xprof; wall clock only")
  parser.add_argument(
      "--no-tpu-profiling-options",
      dest="tpu_profiling_options",
      action="store_false",
      help="drop base.yml's advanced TPU profiling caps; the raw capture drops events under --ragged-sort",
  )
  parser.add_argument("--model", default=MODEL_NAME, help="MaxText model_name; the tunix arm only has qwen3-0.6b")
  parser.add_argument("--scan", action="store_true", help="use MaxText's scanned decoder")
  parser.add_argument("--opt", default="sgd", choices=("sgd", "adamw"), help="optimizer, matched across arms")
  parser.add_argument(
      "--explicit",
      action="store_true",
      help="shard_mode=explicit, so the mesh axes are Explicit rather than Auto; MaxText arms only",
  )
  parser.add_argument(
      "--zero1",
      action="store_true",
      help="shard_optimizer_over_data: shard the optimizer moments over --dp. Implies --explicit; engine arm only",
  )
  return parser


class RunSpec:
  """Resolved shape of one run: which devices, what mesh, how much data.

  Kept as one object so the three arms cannot drift apart on the parts that are supposed
  to be held fixed. `devices` is a prefix of `jax.devices()` rather than the whole list,
  which is how a 2-device mesh is run on a 4-chip host without restricting the runtime.
  """

  def __init__(self, args: argparse.Namespace):
    all_devices = jax.devices()
    if args.devices is not None and not 0 < args.devices <= len(all_devices):
      raise ValueError(f"--devices={args.devices} but only {len(all_devices)} are visible")
    self.devices = all_devices[: args.devices] if args.devices else all_devices
    self.dp = args.dp
    self.tp = args.tp
    self.ep = args.ep
    self.fsdp = args.fsdp if args.fsdp else len(self.devices) // (self.dp * self.tp * self.ep)
    if self.dp * self.fsdp * self.tp * self.ep != len(self.devices):
      raise ValueError(
          f"mesh dp={self.dp} x fsdp={self.fsdp} x tp={self.tp} x ep={self.ep} = "
          f"{self.dp * self.fsdp * self.tp * self.ep} does not cover {len(self.devices)} devices"
      )
    self.ring_of_experts = args.ring_of_experts
    self.ragged_sort = args.ragged_sort
    # MaxText infers the EP rank from `logical_axis_rules` (the `exp` rule maps to the
    # `expert` physical axis), and rejects both flags outright when that rank is 1 --
    # `types.py` raises "When EP rank is 1, use_ring_of_experts must be False". Catching it
    # here names the flag that is missing rather than the invariant that broke.
    if (self.ring_of_experts or self.ragged_sort) and self.ep == 1:
      raise ValueError("--ring-of-experts / --ragged-sort need an expert axis; pass --ep > 1")
    self.ga = args.ga
    self.batch = args.batch
    self.seq = args.seq
    self.steps = args.steps
    self.trace = args.trace
    self.tpu_profiling_options = args.tpu_profiling_options
    self.model = args.model
    self.scan = args.scan
    self.opt = args.opt
    # Zero-1 needs `Explicit` mesh axes, so asking for one without the other is always a
    # mistake rather than a shape: `_zero1_active` would decline and the run would silently
    # be the baseline it was meant to be compared against. Imply it instead of rejecting it.
    self.zero1 = args.zero1
    self.explicit = args.explicit or args.zero1
    # `types.py` raises this itself, but only after the model has been built. Zero-1 shards
    # the moments over "data" on top of the parameters' own layout, and FSDP has already
    # sharded those over "fsdp", so the add inside the update is a type error under explicit
    # sharding and an extra collective under auto.
    if self.zero1 and self.fsdp != 1:
      raise ValueError(f"--zero1 cannot be combined with FSDP (fsdp={self.fsdp}); pass --dp {len(self.devices)}")
    if self.zero1 and self.dp <= 1:
      raise ValueError("--zero1 has nothing to shard over without a data axis; pass --dp > 1")
    # SGD carries no parameter-shaped state, so Zero-1 would have nothing to move and the
    # arm would measure the all-gather without the saving that pays for it.
    if self.zero1 and self.opt == "sgd":
      raise ValueError("--zero1 with --opt sgd is vacuous: sgd has no optimizer moments to shard. Pass --opt adamw")

  @property
  def micro_steps(self) -> int:
    """Dataset items consumed: tunix takes one per micro step, and so does the engine."""
    return self.steps * self.ga

  @property
  def global_batch(self) -> int:
    return self.batch * self.ga

  @property
  def per_device_batch(self) -> float:
    """MaxText's `per_device_batch_size`, which describes the *micro*-batch."""
    return self.batch / len(self.devices)

  def tag(self, arm: str) -> str:
    """Profile subdirectory. Only non-default shapes are suffixed, so the original
    4-device ga=1 traces keep the paths they were written under."""
    parts = []
    if self.ga != ACCUM_STEPS:
      parts.append(f"ga{self.ga}")
    if len(self.devices) != len(jax.devices()):
      parts.append(f"d{len(self.devices)}")
    if self.dp != 1:
      parts.append(f"dp{self.dp}" if self.fsdp == 1 else f"dp{self.dp}fsdp{self.fsdp}")
    if self.tp != 1:
      parts.append(f"fsdp{self.fsdp}tp{self.tp}")
    if self.ep != 1:
      parts.append(f"fsdp{self.fsdp}ep{self.ep}" if self.tp == 1 else f"ep{self.ep}")
    if self.ring_of_experts:
      parts.append("roe")
    if self.ragged_sort:
      parts.append("rsort")
    if self.batch != BATCH_SIZE:
      parts.append(f"b{self.batch}")
    if self.seq != SEQ_LEN:
      parts.append(f"s{self.seq}")
    if self.opt != "sgd":
      parts.append(self.opt)
    # Only one of these is ever tagged: `--zero1` implies `--explicit`, and the pair
    # `explicit` alone is the control arm that isolates the mesh mode from the feature.
    if self.zero1:
      parts.append("zero1")
    elif self.explicit:
      parts.append("explicit")
    return "-".join([arm] + parts)

  def describe(self) -> str:
    moe = [name for name, on in (("ring_of_experts", self.ring_of_experts), ("ragged_sort", self.ragged_sort)) if on]
    return (
        f"devices={len(self.devices)} ({self.devices[0].device_kind})  "
        f"mesh dp={self.dp} fsdp={self.fsdp} tp={self.tp} ep={self.ep}  "
        + (f"moe={'+'.join(moe)}  " if moe else "")
        + f"opt={self.opt}  shard_mode={'explicit' if self.explicit else 'auto'}  zero1={self.zero1}  "
        + f"ga={self.ga}  "
        f"micro-batch={self.batch}x{self.seq}  global-batch={self.global_batch}  "
        f"steps={self.steps} ({self.micro_steps} micro)"
    )


def report_peak_hbm(spec: RunSpec) -> None:
  """Peak HBM per device over the whole process, straight off the TPU allocator.

  `peak_bytes_in_use` is a high-water mark since process start, so this has to be read after
  the loop and it covers compilation as well as steady state. That is the number that decides
  whether a shape fits, which is what Zero-1 is bought with -- and unlike
  `Compiled.memory_analysis()` it is one call that means the same thing on both trainers,
  rather than a per-kernel figure only the engine arm can enumerate.

  Reported for every device in the mesh, not just the first: Zero-1 shards the moments over
  `data`, and a layout that is lopsided across replicas would show up here and nowhere else.
  """
  peaks, limits = [], []
  for device in spec.devices:
    stats = device.memory_stats() or {}
    peaks.append(stats.get("peak_bytes_in_use", 0) / 1e9)
    limits.append(stats.get("bytes_limit", 0) / 1e9)
  if not any(peaks):
    return
  spread = f"  (min {min(peaks):.2f} max {max(peaks):.2f})" if max(peaks) - min(peaks) > 0.01 else ""
  print(f"  peak HBM / device: {max(peaks):.2f} G of {max(limits):.2f} G{spread}")


def optimizer_overrides(spec: RunSpec) -> dict:
  """MaxText config keys that make `create_training_optimizer` build `optax_optimizer(spec)`.

  Paired with `optax_optimizer` below so the two sides cannot drift: the engine takes its
  optimizer from the config and the PeftTrainer arms take an `optax` object, and matching
  those by eye is how `adam_b2` ends up at 0.95 on one side and 0.999 on the other.

  `gradient_clipping_threshold=0.0` and the flattened schedule are not optimizer choices so
  much as removals: base.yml clips at 1.0 and runs warmup+cosine, neither of which the
  `optax` side does, and the clip is a full-tree l2 norm on every step.
  """
  overrides = {
      "opt_type": spec.opt,
      "learning_rate": LEARNING_RATE,
      "gradient_clipping_threshold": 0.0,
      "warmup_steps_fraction": 0.0,
      "learning_rate_final_fraction": 1.0,
  }
  if spec.opt == "adamw":
    overrides |= {
        "adam_b1": ADAM_B1,
        "adam_b2": ADAM_B2,
        "adam_eps": ADAM_EPS,
        "adam_weight_decay": ADAM_WEIGHT_DECAY,
    }
  return overrides


def optax_optimizer(spec: RunSpec):
  """The optimizer the PeftTrainer arms pass in, matched to `optimizer_overrides`.

  `inject_hyperparams` is what lets tunix read the learning rate back out of the optimizer
  state; it is not a performance choice and both `--opt` values keep it.
  """
  schedule = optax.constant_schedule(LEARNING_RATE)
  if spec.opt == "adamw":
    return optax.inject_hyperparams(optax.adamw)(
        learning_rate=schedule, b1=ADAM_B1, b2=ADAM_B2, eps=ADAM_EPS, weight_decay=ADAM_WEIGHT_DECAY
    )
  return optax.inject_hyperparams(optax.sgd)(learning_rate=schedule)


def make_gen_model_input_fn(pad_id: int):
  """Builds the `gen_model_input_fn` both arms pass to the trainer."""

  def gen_model_input_fn(x: peft_trainer.TrainingInput):
    pad_mask = x.input_tokens != pad_id
    positions = common.build_positions_from_mask(pad_mask)
    attention_mask = common.make_causal_attn_mask(pad_mask)
    return {
        "input_tokens": x.input_tokens,
        "input_mask": x.input_mask,
        "positions": positions,
        "attention_mask": attention_mask,
    }

  return gen_model_input_fn


def make_dataset(
    num_steps: int = MAX_STEPS,
    batch_size: int = BATCH_SIZE,
    seq_len: int = SEQ_LEN,
    vocab_size: int = VOCAB_SIZE,
) -> List[Any]:
  """Creates `num_steps` deterministic single-batch training inputs."""
  rng = np.random.default_rng(0)
  dataset = []
  for _ in range(num_steps):
    tokens = rng.integers(0, vocab_size, size=(batch_size, seq_len)).astype(np.int32)
    dataset.append(
        peft_trainer.TrainingInput(
            input_tokens=tokens,
            input_mask=np.ones((batch_size, seq_len), dtype=np.int32),
        )
    )
  return dataset


def make_dataset_for(spec: RunSpec) -> List[Any]:
  """`make_dataset` sized for one run -- `ga * steps` micro-batches."""
  return make_dataset(num_steps=spec.micro_steps, batch_size=spec.batch, seq_len=spec.seq)


class StepTimer(hooks.TrainingHooks):
  """Records the wall-clock cadence of the training loop, step by step.

  The trainer dispatches asynchronously, so a timestamp taken at the top of a step is
  not that step's device time. It is still the right thing to measure: `wait_for_next`
  blocks immediately before `on_train_step_start` whenever the inflight queue is full,
  so once the queue saturates the loop advances exactly as fast as the device retires
  steps. Taking the median over the post-warmup steps then gives the steady-state step
  time without needing the profile at all -- which matters, because a 5M-event cap makes
  `trace.json.gz` unreadable for runs this size.

  Under gradient accumulation the hook fires once per *micro* step, so `report(group=ga)`
  sums each run of `ga` consecutive gaps back into one optimizer step. Group boundaries
  line up because tunix applies its update on the last micro step of each group, and the
  engine arm calls the hook before each `fwd_bwd` for the same reason.
  """

  def __init__(self):
    self.starts: List[float] = []
    self.train_start: float | None = None
    self.train_end: float | None = None

  def on_train_start(self, train_ctx):
    self.train_start = time.perf_counter()

  def on_train_end(self, train_ctx):
    self.train_end = time.perf_counter()

  def on_train_step_start(self, train_ctx):
    self.starts.append(time.perf_counter())

  def on_train_step_end(self, train_ctx, train_step, train_loss):
    pass

  def on_eval_step_start(self, train_ctx):
    pass

  def on_eval_step_end(self, train_ctx, eval_loss):
    pass

  def report(self, label: str, warmup: int = WARMUP_STEPS, group: int = 1) -> None:
    """Prints the total, the first-step (compile) cost and the steady-state step time."""
    gaps = [b - a for a, b in zip(self.starts, self.starts[1:])]
    if group > 1:
      # Sum whole groups only; a trailing partial group has no update in it.
      deltas = [sum(gaps[i : i + group]) for i in range(0, len(gaps) - group + 1, group)]
    else:
      deltas = gaps
    total = (self.train_end or time.perf_counter()) - (self.train_start or 0.0)
    print(f"\n===== {label}")
    print(f"  steps dispatched : {len(self.starts)}" + (f" micro ({group} per step)" if group > 1 else ""))
    print(f"  train() total    : {total:.1f}s (includes compilation)")
    if deltas:
      print(f"  step 0 -> 1      : {deltas[0]:.3f}s (compile lands here)")
    # Printed in full because the aggregate hid a real difference: the MaxText arm's
    # train() total exceeded step 0 plus 22 steady steps by ~46s, which only the
    # per-step list can attribute to a specific step (a later recompilation) rather
    # than to teardown.
    print("  per-step (ms)    : " + " ".join(f"{d * 1e3:.0f}" for d in deltas))
    steady = deltas[warmup:]
    if steady:
      print(
          f"  steady state     : median {statistics.median(steady) * 1e3:8.1f}ms  "
          f"mean {statistics.mean(steady) * 1e3:8.1f}ms  "
          f"min {min(steady) * 1e3:8.1f}ms  max {max(steady) * 1e3:8.1f}ms  "
          f"(n={len(steady)}, first {warmup} dropped)"
      )
    else:
      print("  steady state     : not enough steps to measure")
    if self.starts and self.train_end:
      print(f"  drain after last : {self.train_end - self.starts[-1]:.3f}s")
