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

"""Shared rig for the qwen3-0.6b model comparison under tunix `peft_trainer_v2`.

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
from tunix.rl import common
from tunix.sft import hooks
from tunix.sft import peft_trainer

LEARNING_RATE = 1e-5
BATCH_SIZE = 8
SEQ_LEN = 1024
# Steps dropped from the steady-state figure: compilation lands in the first, and the
# next two cover the throttler filling its two-deep inflight queue.
WARMUP_STEPS = 3
BENCHMARK_STEPS = 20
MAX_STEPS = WARMUP_STEPS + BENCHMARK_STEPS
ACCUM_STEPS = 1
# qwen3-0.6b's full vocab, identical on both sides.
VOCAB_SIZE = 151936

TOKENIZER_ID = "Qwen/Qwen3-0.6B"

# Where each arm writes its xprof trace. Set `PERF_PARITY_PROFILE_ROOT` to a GCS bucket to
# collect them somewhere durable; the destination has no bearing on the measurement, and
# `StepTimer` below reports the step time without the profile being opened at all.
PROFILE_ROOT = os.environ.get("PERF_PARITY_PROFILE_ROOT", "perf_parity_traces")


def profile_dir(arm: str) -> str:
  """Trace destination for one arm, e.g. `qwen3-0.6b-tunix`."""
  return os.path.join(PROFILE_ROOT, arm)


def maybe_trace(log_dir: str, spec: "RunSpec"):
  """The profiler around the timed loop, unless `--no-trace` asked for it to be left off.

  Tracing is not free and it is not free *evenly*: the profiler charges per dispatch, so an
  arm that issues dozens of tiny eager ops per step pays far more for being watched than one
  that issues two. MaxText's `update` reads 27.0 ms traced against 14.0 ms untraced where
  tunix goes 8.2 -> 18.5 ms, so a traced A/B flatters whichever side dispatches less. Quote
  `--no-trace` numbers for wall clock and traced ones for where the time went.
  """
  if not spec.trace:
    return contextlib.nullcontext()
  print(f"tracing to {log_dir}", flush=True)
  return jax.profiler.trace(log_dir=log_dir)


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
  """Adds the flags every arm honours. Defaults reproduce the original 4-device run."""
  parser.add_argument("--ga", type=int, default=ACCUM_STEPS, help="micro-batches per optimizer step")
  parser.add_argument("--devices", type=int, default=None, help="use only the first N local devices")
  parser.add_argument("--fsdp", type=int, default=None, help="fsdp mesh axis (default: fills the devices)")
  parser.add_argument("--tp", type=int, default=1, help="tensor-parallel mesh axis")
  parser.add_argument("--batch", type=int, default=BATCH_SIZE, help="examples per micro-batch")
  parser.add_argument("--seq", type=int, default=SEQ_LEN, help="tokens per example")
  parser.add_argument("--steps", type=int, default=MAX_STEPS, help="optimizer steps, warmup included")
  parser.add_argument("--no-trace", dest="trace", action="store_false", help="skip xprof; wall clock only")
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
    self.tp = args.tp
    self.fsdp = args.fsdp if args.fsdp else len(self.devices) // self.tp
    if self.fsdp * self.tp != len(self.devices):
      raise ValueError(
          f"mesh fsdp={self.fsdp} x tp={self.tp} = {self.fsdp * self.tp} does not cover " f"{len(self.devices)} devices"
      )
    self.ga = args.ga
    self.batch = args.batch
    self.seq = args.seq
    self.steps = args.steps
    self.trace = args.trace

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
    if self.tp != 1:
      parts.append(f"fsdp{self.fsdp}tp{self.tp}")
    if self.batch != BATCH_SIZE:
      parts.append(f"b{self.batch}")
    if self.seq != SEQ_LEN:
      parts.append(f"s{self.seq}")
    return "-".join([arm] + parts)

  def describe(self) -> str:
    return (
        f"devices={len(self.devices)} ({self.devices[0].device_kind})  "
        f"mesh fsdp={self.fsdp} tp={self.tp}  ga={self.ga}  "
        f"micro-batch={self.batch}x{self.seq}  global-batch={self.global_batch}  "
        f"steps={self.steps} ({self.micro_steps} micro)"
    )


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
