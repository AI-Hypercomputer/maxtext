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

"""Standalone validation and benchmarking harness for `MaxTextTrainingEngine`.

Drives the engine the way Tunix's `TrainerWorker` does in the Trellis RL stack -- `fwd_bwd` once
per micro-batch, then `update` -- on synthetic payloads and without a sampler, so the trainer half
of an RL step can be sized, timed and checked on its own before a sampler is attached:

  * `--mode=aot` compiles every kernel the engine runs (`fwd_bwd`, `fwd_bwd_accum`, `update`, and
    the forward-only eval kernel) for `compile_topology` on this host and reports the memory of
    each. Nothing is allocated, so a CPU host can say whether a TPU cell fits.
  * `--mode=run` runs `--steps` optimizer steps on the local devices and reports wall-clock per
    step, tokens/s per device and per chip, TFLOP/s per device and MFU -- on the same accounting
    `train.py` logs, so the two are directly comparable -- then exercises the read-only entry points
    at the same cell: `run_eval`, forward-only log-prob scoring through `fwd_only`, and
    `prepare_weight_sync`.

`--loss_type=sft` is MaxText's own cross-entropy loss, the loss `train.py` benchmarks.
`--loss_type=grpo` wraps the model in `TunixMaxTextAdapter` and trains with Tunix's
`algo_core.grpo_loss_fn`, through the same four-key model-input function Tunix's orchestrator ships
to the trainer, on `RLTrainerPayload` micro-batches shaped as `PaddedBatchAssembler` emits them.
`--router_replay` adds the `[batch, seq, layers, top_k]` `routed_experts` a rollout captures.

Everything after the flags is MaxText's own config, parsed by `pyconfig.initialize`, so the
arguments of a `train.py` run carry over unchanged. The micro-batch is
`per_device_batch_size * num_devices` and an optimizer step is `gradient_accumulation_steps`
micro-batches: the split `train.py` makes, driven from outside instead of inside one `lax.scan`.

Qwen3.5-397B on 128 v7x chips (GBS 1024 = micro-batch 64 x 16), live:

  python3 -m maxtext.training_engine.engine_benchmark --mode=run --loss_type=grpo --steps=5 \
    --eval_batches=2 --logprob_batches=2 \
    src/maxtext/configs/base.yml model_name=qwen3.5-397b-a17b max_target_length=65536 \
    custom_mesh_and_rule=cp-as-ep ici_fsdp_parallelism=32 ici_context_parallelism=4 \
    ici_expert_parallelism=2 per_device_batch_size=0.25 gradient_accumulation_steps=16 ...

and ahead of time on a CPU host, with `--mode=aot compile_topology=tpu7x-256
compile_topology_num_slices=1` added.
"""

from collections.abc import Callable, Sequence
import contextlib
import dataclasses
import functools
import json
import math
import os
import re
import resource
import statistics
import time
from typing import Any

from absl import app
from absl import flags
from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.configs import pyconfig
from maxtext.trainers.pre_train import train_compile as pre_train_compile
from maxtext.training_engine import abstract_engine
from maxtext.training_engine import maxtext_engine
from maxtext.training_engine import maxtext_engine_compile
from maxtext.utils import gcs_utils
from maxtext.utils import max_logging
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils
import numpy as np
import pathwaysutils
from tunix.rl import algo_core
from tunix.rl import common as tunix_common

_MODE = flags.DEFINE_enum("mode", "run", ["run", "aot"], "run: execute on local devices. aot: compile only.")
_LOSS_TYPE = flags.DEFINE_enum("loss_type", "grpo", ["sft", "grpo"], "Which loss the engine trains with.")
_STEPS = flags.DEFINE_integer("steps", 5, "run: optimizer steps to execute.")
_WARMUP_STEPS = flags.DEFINE_integer(
    "warmup_steps", 1, "run: leading steps left out of the reported statistics. Compilation happens before step 0."
)
_DISTINCT_MICRO_BATCHES = flags.DEFINE_integer(
    "distinct_micro_batches",
    2,
    "run: micro-batches built up front and cycled. Building them inside the timed loop would put host-side "
    "generation on the clock; a router-replay micro-batch at 64K is several GB.",
)
_PROMPT_LENGTH = flags.DEFINE_integer(
    "prompt_length", 4096, "grpo: prompt tokens per row; the completion is the rest of max_target_length."
)
_ROUTER_REPLAY = flags.DEFINE_bool("router_replay", False, "grpo: attach routed_experts, as a rollout captures them.")
_OLD_LOGPS = flags.DEFINE_bool(
    "old_per_token_logps", True, "grpo: attach sampler log-probs, as off-policy training receives them."
)
_GRPO_BETA = flags.DEFINE_float("grpo_beta", 0.0, "grpo: KL coefficient. Non-zero also attaches ref_per_token_logps.")
_GRPO_EPSILON = flags.DEFINE_float("grpo_epsilon", 0.2, "grpo: lower PPO clip.")
_GRPO_EPSILON_HIGH = flags.DEFINE_float("grpo_epsilon_high", 0.28, "grpo: upper PPO clip.")
_GRPO_LOSS_AGG_MODE = flags.DEFINE_string("grpo_loss_agg_mode", "token-mean", "grpo: loss aggregation mode.")
_LOGPS_CHUNK_SIZE = flags.DEFINE_integer(
    "compute_logps_chunk_size",
    0,
    "grpo: compute log-probs this many tokens at a time (Tunix's `compute_logps_chunk_size`), so the full "
    "[tokens, vocab] logits are never materialized. 0 is unchunked, which is what Tunix's orchestrator sends today.",
)
_PAD_ID = flags.DEFINE_integer("pad_id", 248044, "grpo: tokenizer pad id. Qwen3.5's by default.")
_EOS_ID = flags.DEFINE_integer("eos_id", 248044, "grpo: tokenizer eos id. Qwen3.5's by default.")
_EVAL_BATCHES = flags.DEFINE_integer("eval_batches", 0, "run: micro-batches in one run_eval pass; 0 skips it.")
_LOGPROB_BATCHES = flags.DEFINE_integer(
    "logprob_batches", 0, "run, grpo: micro-batches scored forward-only through fwd_only; 0 skips it."
)
_WEIGHT_SYNC = flags.DEFINE_bool("weight_sync", False, "run: stage and release a weight sync after the last update.")
_PROFILE_STEPS = flags.DEFINE_integer(
    "profile_steps",
    0,
    "run: optimizer steps to trace with XProf, after the timed ones and outside their statistics. Every "
    "micro-step and the update are annotated. Use this instead of MaxText's profiler_steps, which the engine "
    "counts in micro-steps.",
)
_PROFILE_DIR = flags.DEFINE_string("profile_dir", "", "run: where the trace goes; the config's tensorboard_dir if unset.")
_DEVICES_PER_CHIP = flags.DEFINE_integer(
    "devices_per_chip", 0, "JAX devices per physical chip (v7x 2, v5p 1). 0 reads it off the devices."
)
_PEAK_TFLOPS_PER_CHIP = flags.DEFINE_float(
    "peak_tflops_per_chip", 0.0, "bf16 dense peak per chip, for MFU. 0 looks it up from the device kind."
)
_TARGET_TOKENS_PER_SEC_PER_CHIP = flags.DEFINE_float(
    "target_tokens_per_sec_per_chip", 0.0, "run: an acceptance target to report the measured number against."
)
_HBM_GIB_PER_DEVICE = flags.DEFINE_float(
    "hbm_gib_per_device", 0.0, "aot: HBM per device, to report headroom against. 0 reports none."
)
_REPORT_PATH = flags.DEFINE_string("report_path", "", "Write the JSON report here: a local path or gs://.")

_GIB = 1024.0**3

# bf16 dense peak TFLOP/s per chip, from the public Cloud TPU specs. Matched against `device_kind`
# in order, so a more specific name has to come before any name it contains.
_PEAK_BF16_TFLOPS_PER_CHIP = (("7x", 2307.0), ("v6", 918.0), ("v5p", 459.0), ("v5", 197.0), ("v4", 275.0))

PASS, FAIL = "PASS", "FAIL"

# The event JAX records once per XLA backend compile, with the jitted function's name.
_BACKEND_COMPILE_EVENT = "/jax/core/compile/backend_compile_duration"


class CompileLog:
  """Records every XLA backend compile and the harness phase it happened in.

  A step time is only a steady-state number if nothing recompiled inside it, and a retrace that
  lands in every step looks exactly like a slow kernel. JAX names the jitted function on each
  compile, so a kernel compiled up front can be told apart from the one-off helper compiles an
  eager metric read triggers -- the first is the failure, the second is noise.
  """

  def __init__(self) -> None:
    self.phase = "setup"
    self.events: list[tuple[str, str, float]] = []
    jax.monitoring.register_event_duration_secs_listener(self._record)

  def _record(self, event: str, duration_secs: float, **kwargs: Any) -> None:
    if event == _BACKEND_COMPILE_EVENT:
      self.events.append((self.phase, str(kwargs.get("fun_name", "?")), duration_secs))

  def names(self, phase: str) -> list[str]:
    return [name for p, name, _ in self.events if p == phase]

  def summary(self) -> dict[str, dict[str, Any]]:
    """Per phase: how many compiles, their total seconds, and the functions compiled."""
    out: dict[str, dict[str, Any]] = {}
    for phase, name, secs in self.events:
      entry = out.setdefault(phase, {"count": 0, "seconds": 0.0, "functions": []})
      entry["count"] += 1
      entry["seconds"] += secs
      if name not in entry["functions"]:
        entry["functions"].append(name)
    return out


def recompiled_kernels(kernel_names: Sequence[str], later_names: Sequence[str]) -> list[str]:
  """The functions compiled again later that were first compiled as engine kernels."""
  kernels = set(kernel_names)
  return sorted({name for name in later_names if name in kernels})


@dataclasses.dataclass(frozen=True)
class GrpoConfig:
  """The fields `algo_core.grpo_loss_fn` reads off its `algo_config`.

  Frozen, so it compares by value: the engine closes it over as a static input and recompiles
  when it changes, which it decides by comparing the old one to the new.
  """

  beta: float = 0.0
  epsilon: float = 0.2
  epsilon_high: float = 0.28
  loss_algo: str = "grpo"
  loss_agg_mode: str = "token-mean"
  temperature: float = 1.0
  kl_loss_mode: str = "low_var_kl"
  kl_clamp_value: float | None = None


def grpo_model_input(
    payload: Any, *, algo_config: GrpoConfig, pad_id: int, eos_id: int, logps_chunk_size: int = 0
) -> dict[str, Any]:
  """Maps an `RLTrainerPayload` micro-batch to `grpo_loss_fn`'s keyword arguments.

  The same four keys as Tunix's `algorithm_adapter._algo_model_input`, the function its orchestrator
  hands the trainer through `with_gen_model_input_fn`. Kept here rather than imported because it is
  private to Tunix and its module has changed shape across releases; the contract is only these keys.
  A non-zero `logps_chunk_size` adds `compute_logps_chunk_size`, which `grpo_loss_fn` reads from its
  keyword arguments.
  """
  inputs = {"train_example": payload, "algo_config": algo_config, "pad_id": pad_id, "eos_id": eos_id}
  if logps_chunk_size > 0:
    inputs["compute_logps_chunk_size"] = logps_chunk_size
  return inputs


def devices_per_chip(devices: Sequence[Any]) -> int:
  """Returns how many JAX devices share a physical chip: 2 on v7x, 1 on a v5p megacore.

  Read off the devices' chip coordinates, which both cores of a v7x chip report identically. Devices
  with no coordinates (CPU) count as one per chip.
  """
  chips = {(getattr(d, "slice_index", 0), tuple(d.coords)) for d in devices if getattr(d, "coords", None) is not None}
  if len(chips) == 0 or len(devices) % len(chips):
    return 1
  return len(devices) // len(chips)


def peak_tflops_per_chip(device_kind: str) -> float:
  """Returns the bf16 dense peak for `device_kind`, or 0.0 when it is not a known TPU."""
  kind = device_kind.lower()
  for key, peak in _PEAK_BF16_TFLOPS_PER_CHIP:
    if key in kind:
      return peak
  return 0.0


def throughput(
    config: pyconfig.HyperParameters, step_time_s: float, chip_devices: int, peak_per_chip: float = 0.0
) -> dict[str, float]:
  """Turns one optimizer step's wall-clock into the throughput `train.py` reports.

  Tokens and TFLOPs come from the same `maxtext_utils` helpers `train.py`'s metric logger divides by
  its step time -- both already count all `gradient_accumulation_steps` micro-batches -- so the two
  numbers are comparable. Per-chip figures scale the per-device ones by `chip_devices`.

  Args:
    config: The run's config.
    step_time_s: Wall-clock of one optimizer step.
    chip_devices: JAX devices per physical chip.
    peak_per_chip: bf16 dense peak TFLOP/s per chip; MFU is omitted when 0.

  Returns:
    Tokens and TFLOP/s per device and per chip, tokens/s over all devices, and `mfu` when a peak
    was given.
  """
  tokens_per_device = maxtext_utils.calculate_tokens_training_per_device(config)
  tflops_per_device, _, _ = maxtext_utils.calculate_tflops_training_per_device(config, log=False)
  out = {
      "tokens_per_step": tokens_per_device * jax.device_count(),
      "tokens_per_sec_per_device": tokens_per_device / step_time_s,
      "tokens_per_sec_per_chip": tokens_per_device * chip_devices / step_time_s,
      "tokens_per_sec_total": tokens_per_device * jax.device_count() / step_time_s,
      "tflops_per_sec_per_device": tflops_per_device / step_time_s,
      "tflops_per_sec_per_chip": tflops_per_device * chip_devices / step_time_s,
  }
  if peak_per_chip > 0:
    out["mfu"] = out["tflops_per_sec_per_chip"] / peak_per_chip
  return out


def _token_range(vocab_size: int, pad_id: int, eos_id: int) -> tuple[int, int]:
  """Returns a `[low, high)` id range holding neither the pad nor the eos token.

  A synthetic token equal to eos would cut the completion mask short -- Tunix masks everything
  after the first eos -- and one equal to pad would be masked out of attention, so either would
  silently shrink the work being timed.
  """
  specials = sorted({pad_id, eos_id})
  low, high = (0, min(specials[0], vocab_size)) if specials[0] >= 2 else (specials[-1] + 1, vocab_size)
  if high - low < 2:
    raise ValueError(f"No token range avoids pad_id={pad_id} and eos_id={eos_id} in a vocabulary of {vocab_size}.")
  return low, high


def _normalize(index: tuple[Any, ...], shape: tuple[int, ...]) -> tuple[slice, ...]:
  """Turns `make_array_from_callback`'s index, whose slices may be open, into closed slices."""
  return tuple(slice(*s.indices(n)[:2]) if isinstance(s, slice) else slice(s, s + 1) for s, n in zip(index, shape))


def global_array(
    shape: tuple[int, ...],
    sharding: jax.sharding.Sharding | None,
    fill: Callable[[tuple[slice, ...], np.random.Generator], np.ndarray],
    seed: int,
) -> jax.Array:
  """Builds a global array one shard at a time, without ever holding the whole of it on a host.

  `jax.jit` will not take a host array for a sharding that spans other processes, and a
  router-replay micro-batch at 64K is gigabytes, so each process fills only the shards it owns. The
  random stream is seeded by the shard's global coordinates, not by the process, so two devices
  holding the same (replicated) shard always receive identical data -- which a replicated
  dimension requires -- and the array does not depend on how many hosts built it.

  Args:
    shape: The global shape.
    sharding: Where it lives; None builds it on the default device.
    fill: `fill(slices, rng)` returns the block of the global array at `slices`.
    seed: Distinguishes one array from another of the same shape.

  Returns:
    The array.
  """
  full = tuple(slice(0, n) for n in shape)
  if sharding is None:
    return jnp.asarray(fill(full, np.random.default_rng([seed])))

  def callback(index):
    slices = _normalize(index, shape)
    return fill(slices, np.random.default_rng([seed, *[s.start for s in slices]]))

  return jax.make_array_from_callback(shape, sharding, callback)


def _block_shape(slices: tuple[slice, ...]) -> tuple[int, ...]:
  return tuple(s.stop - s.start for s in slices)


def _tokens(low: int, high: int) -> Callable[..., np.ndarray]:
  return lambda slices, rng: rng.integers(low, high, size=_block_shape(slices), dtype=np.int32)


def _constant(value: float, dtype: Any) -> Callable[..., np.ndarray]:
  return lambda slices, rng: np.full(_block_shape(slices), value, dtype=dtype)


def _positions(slices: tuple[slice, ...], rng: np.random.Generator) -> np.ndarray:
  """Positions along the sequence dim (dim 1), broadcast over every other dim."""
  del rng
  shape = _block_shape(slices)
  arange = np.arange(slices[1].start, slices[1].stop, dtype=np.int32).reshape((1, -1) + (1,) * (len(shape) - 2))
  return np.broadcast_to(arange, shape).copy()


def _normal(slices: tuple[slice, ...], rng: np.random.Generator) -> np.ndarray:
  return rng.standard_normal(size=_block_shape(slices)).astype(np.float32)


def _routed_experts(num_experts: int) -> Callable[..., np.ndarray]:
  """Distinct experts per token and layer, spread uniformly the way random routing spreads them.

  `top_k` evenly spaced experts from a random start: distinct by construction and cheap enough to
  generate for a 64K x 60-layer micro-batch, where an argsort over all experts per token is not.
  """

  def fill(slices, rng):
    *lead, top_k = _block_shape(slices)
    start = rng.integers(0, num_experts, size=(*lead, 1), dtype=np.int32)
    stride = max(num_experts // top_k, 1)
    return ((start + np.arange(top_k, dtype=np.int32) * stride) % num_experts).astype(np.int32)

  return fill


class PayloadFactory:
  """Builds synthetic micro-batches for the engine, on its shardings or as shapes only.

  Every array is created where the engine's compiled step expects it (`engine.input_sharding`),
  so what reaches the kernel is a payload the step would receive from a real batch assembler, with
  no reshard in front of it.
  """

  def __init__(self, config: pyconfig.HyperParameters, engine: maxtext_engine.MaxTextTrainingEngine, loss_type: str):
    self._config = config
    self._engine = engine
    self._loss_type = loss_type
    self.batch = int(config.micro_batch_size_to_train_on)
    self.seq = int(config.max_target_length)
    if loss_type == "grpo":
      self.prompt = _PROMPT_LENGTH.value
      if not 0 < self.prompt < self.seq:
        raise ValueError(f"--prompt_length={self.prompt} must be in (0, max_target_length={self.seq}).")
      self.completion = self.seq - self.prompt
      self._low, self._high = _token_range(config.vocab_size, _PAD_ID.value, _EOS_ID.value)
      # A randomly initialised model assigns every token about 1/V, so a sampler log-prob of
      # -log(V) keeps the importance ratio near 1, as it is for an on-policy micro-batch. An
      # arbitrary constant would put the ratio against the clip bounds on every token.
      self._logp = -math.log(config.vocab_size)

  def _grpo_fields(self) -> dict[str, tuple[tuple[int, ...], Any, Callable[..., np.ndarray]]]:
    """`RLTrainerPayload` field -> (global shape, dtype, fill), for the fields this run attaches."""
    b, p, c = self.batch, self.prompt, self.completion
    fields = {
        "prompt_ids": ((b, p), np.int32, _tokens(self._low, self._high)),
        "prompt_mask": ((b, p), np.float32, _constant(1.0, np.float32)),
        "completion_ids": ((b, c), np.int32, _tokens(self._low, self._high)),
        "completion_mask": ((b, c), np.float32, _constant(1.0, np.float32)),
        "advantages": ((b,), np.float32, _normal),
    }
    if _OLD_LOGPS.value:
      fields["old_per_token_logps"] = ((b, c), np.float32, _constant(self._logp, np.float32))
    if _GRPO_BETA.value != 0.0:
      fields["ref_per_token_logps"] = ((b, c), np.float32, _constant(self._logp, np.float32))
    if _ROUTER_REPLAY.value:
      shape = (b, p + c, self._config.num_decoder_layers, self._config.num_experts_per_tok)
      fields["routed_experts"] = (shape, np.int32, _routed_experts(self._config.num_experts))
    return fields

  def _sft_fields(self) -> dict[str, tuple[tuple[int, ...], Any, Callable[..., np.ndarray]]]:
    """MaxText batch key -> (global shape, dtype, fill), for one micro-batch."""
    fields = {}
    for key, aval in maxtext_engine_compile.get_shaped_micro_batch(self._config).items():
      if key.endswith("_position"):
        fill = _positions
      elif key.endswith("_segmentation") or key in ("corruption_mask", "targets_loss_mask"):
        fill = _constant(1, np.int32)
      else:
        fill = _tokens(0, self._config.vocab_size)
      fields[key] = (tuple(aval.shape), aval.dtype, fill)
    return fields

  def _fields(self):
    return self._grpo_fields() if self._loss_type == "grpo" else self._sft_fields()

  def _wrap(self, arrays: dict[str, Any]) -> Any:
    return abstract_engine.RLTrainerPayload(**arrays) if self._loss_type == "grpo" else arrays

  def micro_batch(self, index: int) -> Any:
    """Returns micro-batch `index`, built on device; distinct indices give distinct data."""
    arrays = {}
    for n, (name, (shape, dtype, fill)) in enumerate(sorted(self._fields().items())):
      sharding = self._engine.input_sharding(jax.ShapeDtypeStruct(shape, dtype))
      arrays[name] = global_array(shape, sharding, fill, seed=index * 1000 + n)
    return self._wrap(arrays)

  def placed_avals(self) -> dict[str, jax.ShapeDtypeStruct]:
    """Every array of one micro-batch as an aval on the sharding it is built on, for sizing it."""
    return {
        name: jax.ShapeDtypeStruct(shape, dtype, sharding=self._engine.input_sharding(jax.ShapeDtypeStruct(shape, dtype)))
        for name, (shape, dtype, _) in self._fields().items()
    }

  def abstract_micro_batch(self) -> Any:
    """Returns a micro-batch of `jax.ShapeDtypeStruct`s, for compiling without allocating."""
    return self._wrap({name: jax.ShapeDtypeStruct(shape, dtype) for name, (shape, dtype, _) in self._fields().items()})


def build_engine(config: pyconfig.HyperParameters, mesh: jax.sharding.Mesh, loss_type: str, abstract: bool = False):
  """Returns the engine for `loss_type`, configured as the RL trainer configures it.

  GRPO wraps the model in `TunixMaxTextAdapter` -- `grpo_loss_fn` calls it with Tunix's signature --
  and installs the loss and model-input function the orchestrator installs over RPC.
  """
  if loss_type == "grpo" and config.num_vocab_tiling > 1 and _LOGPS_CHUNK_SIZE.value <= 0:
    # Under vocab tiling the decoder returns no logits in train mode (`nnx_decoders.py`, it sows
    # hidden states for MaxText's tiled cross-entropy instead), and unchunked `grpo_loss_fn` needs
    # the logits. Chunked log-probs ask for the hidden states instead (`skip_lm_head`), which the
    # model returns under vocab tiling too. Without this the failure surfaces deep inside Tunix.
    raise ValueError(
        f"--loss_type=grpo needs num_vocab_tiling=1 or --compute_logps_chunk_size > 0, got num_vocab_tiling="
        f"{config.num_vocab_tiling}: unchunked, the GRPO loss reads the full logits, which vocab tiling never "
        "materializes."
    )
  kwargs = {"wrap_with_tunix_adapter": True, "tokenizer_pad_id": _PAD_ID.value} if loss_type == "grpo" else {}
  engine_class = maxtext_engine_compile.AbstractMaxTextEngine if abstract else maxtext_engine.MaxTextTrainingEngine
  engine = engine_class(config, mesh=mesh, **kwargs)
  if loss_type == "grpo":
    algo_config = GrpoConfig(
        beta=_GRPO_BETA.value,
        epsilon=_GRPO_EPSILON.value,
        epsilon_high=_GRPO_EPSILON_HIGH.value,
        loss_agg_mode=_GRPO_LOSS_AGG_MODE.value,
    )
    engine.with_loss_fn(algo_core.grpo_loss_fn, has_aux=True).with_gen_model_input_fn(
        functools.partial(
            grpo_model_input,
            algo_config=algo_config,
            pad_id=_PAD_ID.value,
            eos_id=_EOS_ID.value,
            logps_chunk_size=_LOGPS_CHUNK_SIZE.value,
        )
    )
  return engine


def _describe_cell(config: pyconfig.HyperParameters, mesh: jax.sharding.Mesh, chip_devices: int) -> dict[str, Any]:
  num_devices = int(np.prod(list(mesh.shape.values())))
  return {
      "model_name": config.model_name,
      "loss_type": _LOSS_TYPE.value,
      "router_replay": _ROUTER_REPLAY.value if _LOSS_TYPE.value == "grpo" else None,
      "compute_logps_chunk_size": _LOGPS_CHUNK_SIZE.value if _LOSS_TYPE.value == "grpo" else None,
      "mesh": {k: int(v) for k, v in mesh.shape.items()},
      "devices": num_devices,
      "devices_per_chip": chip_devices,
      "chips": num_devices // chip_devices,
      "max_target_length": int(config.max_target_length),
      "micro_batch_size": int(config.micro_batch_size_to_train_on),
      "gradient_accumulation_steps": int(config.gradient_accumulation_steps),
      "global_batch_size": int(config.micro_batch_size_to_train_on * config.gradient_accumulation_steps),
  }


def _memory(compiled: jax.stages.Compiled) -> dict[str, float]:
  """The compiler's memory analysis in GiB, with `resident = argument + output - alias + temp`."""
  analysis = compiled.memory_analysis()
  if analysis is None:
    return {}
  fields = ("argument", "output", "alias", "temp", "generated_code", "host_temp")
  out = {f"{name}_gib": getattr(analysis, f"{name}_size_in_bytes", 0) / _GIB for name in fields}
  out["resident_gib"] = out["argument_gib"] + out["output_gib"] - out["alias_gib"] + out["temp_gib"]
  return out


_UNIT_GIB = {"K": 1 / 1024**2, "M": 1 / 1024, "G": 1.0, "T": 1024.0}
_OOM_PATTERNS = (
    re.compile(r"HLO temporaries \(([0-9.]+)([KMGT])\) exceeds available HBM \(([0-9.]+)([KMGT])\)"),
    re.compile(r"Used ([0-9.]+)([KMGT]) of ([0-9.]+)([KMGT]) hbm"),
)


def parse_oom(message: str) -> dict[str, float]:
  """What an XLA out-of-memory error says it needed and had, in GiB; empty if it says neither.

  XLA gates the compile on its temporaries alone, so the first figure is a temp size, not the
  kernel's resident total.
  """
  for pattern in _OOM_PATTERNS:
    match = pattern.search(message)
    if match:
      needed, needed_unit, available, available_unit = match.groups()
      return {
          "temp_gib": float(needed) * _UNIT_GIB[needed_unit],
          "hbm_gib_reported": float(available) * _UNIT_GIB[available_unit],
      }
  return {}


def device_bytes(tree: Any) -> int:
  """Bytes one device holds of `tree`'s arrays or avals: each leaf's shard, not its global size."""
  total = 0
  for leaf in jax.tree.leaves(tree):
    if not hasattr(leaf, "shape") or not hasattr(leaf, "dtype"):
      continue
    sharding = getattr(leaf, "sharding", None)
    shape = sharding.shard_shape(leaf.shape) if sharding is not None else leaf.shape
    total += int(np.prod(shape, dtype=np.int64)) * np.dtype(leaf.dtype).itemsize
  return total


def device_peaks(
    kernels: dict[str, dict[str, Any]], optimizer_gib: float, accumulator_gib: float, other_payloads_gib: float
) -> dict[str, float]:
  """Each kernel's resident plus what stays in HBM beside it without being one of its arguments.

  The compiler's memory analysis sees a kernel's own arguments, outputs and temporaries, not the
  rest of the engine's state that stays resident while the kernel runs: the optimizer moments are
  arguments of `update` only, the gradient accumulator is carried between `fwd_bwd` calls and so
  sits beside an eval that runs mid-step, and every pooled micro-batch but the one being consumed
  sits beside every kernel. Taking a kernel's resident alone as the device peak understates it by
  exactly those.
  """
  beside = {
      "fwd_bwd": optimizer_gib + other_payloads_gib,
      "fwd_bwd_accum": optimizer_gib + other_payloads_gib,
      "update": other_payloads_gib,
      "eval": optimizer_gib + accumulator_gib + other_payloads_gib,
  }
  return {name: k["resident_gib"] + beside.get(name, 0.0) for name, k in kernels.items() if "resident_gib" in k}


def run_aot(config: pyconfig.HyperParameters) -> dict[str, Any]:
  """Compiles every kernel for `compile_topology` and reports what each needs."""
  pre_train_compile.validate_config(config)
  if config.enable_diloco:
    raise NotImplementedError("enable_diloco: the engine has no DiLoCo outer step, so this would compile something else.")
  topology_mesh = pre_train_compile.get_topology_mesh(config)
  chip_devices = _DEVICES_PER_CHIP.value or devices_per_chip(topology_mesh.devices.flatten().tolist())
  report = {"mode": "aot", **_describe_cell(config, topology_mesh, chip_devices)}
  max_logging.log(f"engine_benchmark: compiling for {config.compile_topology}: {report}")

  engine = build_engine(config, topology_mesh, _LOSS_TYPE.value, abstract=True)
  dummy = PayloadFactory(config, engine, _LOSS_TYPE.value).abstract_micro_batch()
  report["input_sharding"] = str(engine.input_sharding(jax.ShapeDtypeStruct((report["micro_batch_size"], 1), np.int32)))

  # Each kernel is compiled on its own, so one that does not fit still leaves the others'
  # numbers -- and the compiler's out-of-memory message, which states what it needed -- in the
  # report, instead of one RESOURCE_EXHAUSTED hiding all four. `compile_kernels` is the same
  # lowering compiled in one go.
  start = time.perf_counter()
  # pylint: disable-next=protected-access
  lowered = engine._lower_kernels(dummy)
  options = engine._xla_options(None)  # pylint: disable=protected-access
  kernels: dict[str, dict[str, Any]] = {}
  for name in (*maxtext_engine.KERNEL_NAMES, "eval"):
    try:
      if name == "eval":
        executable = engine.compile_eval_kernel(dummy)
      else:
        with engine._sharding_ctx():  # pylint: disable=protected-access
          executable = lowered[name].compile(compiler_options=options)
    except Exception as exc:  # pylint: disable=broad-exception-caught
      message = str(exc)
      if "RESOURCE_EXHAUSTED" not in message and "exceeds" not in message:
        raise
      kernels[name] = {"oom": True, **parse_oom(message), "error": message[:2000]}
      max_logging.log(f"engine_benchmark: kernel {name} does not fit: {message[:500]}")
      continue
    kernels[name] = _memory(executable)
    cost = executable.cost_analysis()
    cost = cost[0] if isinstance(cost, (list, tuple)) and cost else cost
    if isinstance(cost, dict) and "flops" in cost:
      kernels[name]["tflops"] = cost["flops"] / 1e12
    max_logging.log(f"engine_benchmark: kernel {name}: {kernels[name]}")
  report["compile_s"] = time.perf_counter() - start
  report["kernels"] = kernels
  report["any_oom"] = any(k.get("oom") for k in kernels.values())

  # What stays resident beside the kernels -- see `device_peaks`.
  optimizer_gib = device_bytes(nnx.state(engine.optimizer, nnx.OptState)) / _GIB
  _, _, _, grads_aval, _ = lowered["fwd_bwd"].out_info
  accumulator_gib = device_bytes(grads_aval) / _GIB
  payload_gib = device_bytes(PayloadFactory(config, engine, _LOSS_TYPE.value).placed_avals()) / _GIB
  other_payloads_gib = (max(_DISTINCT_MICRO_BATCHES.value, 1) - 1) * payload_gib
  report["parameters_gib"] = device_bytes(nnx.state(engine.model, nnx.Param)) / _GIB
  report["resident_outside_kernels_gib"] = {
      "optimizer_state": optimizer_gib,
      "gradient_accumulator": accumulator_gib,
      "micro_batch": payload_gib,
      "other_pooled_micro_batches": other_payloads_gib,
  }
  peaks = device_peaks(kernels, optimizer_gib, accumulator_gib, other_payloads_gib)
  for name, peak in peaks.items():
    kernels[name]["device_peak_gib"] = peak
  report["peak_resident_gib"] = max((k.get("resident_gib", 0.0) for k in kernels.values()), default=0.0)
  report["device_peak_gib"] = max(peaks.values(), default=0.0)
  # Only kernels that compiled have a peak. With one out of memory, the figure above is a lower bound
  # on the device's peak, not the peak, and `fits` is already False.
  report["device_peak_kernels"] = sorted(peaks)
  if _HBM_GIB_PER_DEVICE.value > 0:
    report["hbm_gib_per_device"] = _HBM_GIB_PER_DEVICE.value
    report["headroom_gib"] = _HBM_GIB_PER_DEVICE.value - report["device_peak_gib"]
  if report["any_oom"]:
    report["fits"] = False
  elif _HBM_GIB_PER_DEVICE.value > 0:
    report["fits"] = report["headroom_gib"] >= 0
  return report


def _reduce(metric: Any) -> float | None:
  """One number for a buffered metric, reduced as the engine's logger reduces it."""
  if metric is None:
    return None
  value = metric.compute() if isinstance(metric, abstract_engine.WeightedMetric) else metric
  return float(np.mean(np.asarray(value)))


def _check(checks: dict[str, Any], name: str, ok: bool, detail: Any) -> None:
  checks[name] = {"status": PASS if ok else FAIL, "detail": detail}
  max_logging.log(f"engine_benchmark: check {name}: {checks[name]['status']} ({detail})")


def _time_step(
    engine, pool: list[Any], step: int, accumulation: int, annotate: bool = False
) -> tuple[float, dict[str, float | None]]:
  """Runs one optimizer step and returns its wall-clock and its loss and gradient norm.

  The clock stops when the step's own outputs are ready -- the metrics buffer holds the loss of
  every micro-batch and the gradient norm the update kernel returned -- so the time is the step's,
  not the time to dispatch it. `annotate` labels each micro-step and the update in a trace.
  """
  label = jax.profiler.TraceAnnotation if annotate else lambda name: contextlib.nullcontext()
  start = time.perf_counter()
  for micro in range(accumulation):
    with label(f"fwd_bwd_microstep_{micro}"):
      engine.fwd_bwd(pool[(step * accumulation + micro) % len(pool)])
  with label("engine_update"):
    engine.update()
  buffer = engine.get_metrics(clear_cache=True)
  if "gradient_norm" not in buffer.scalar_metrics:
    # No norm to wait on, so wait on what the update wrote instead.
    jax.block_until_ready(nnx.state(engine.model, nnx.Param))
  jax.block_until_ready(buffer)
  elapsed = time.perf_counter() - start
  return elapsed, {
      "loss": _reduce(buffer.weighted_metrics.get("loss")),
      "grad_norm": _reduce(buffer.scalar_metrics.get("gradient_norm")),
      # 1.0 when the update kernel refused the step (`skip_step_on_nan`, `skip_step_on_spikes`).
      "step_skipped": _reduce(buffer.scalar_metrics.get("step_skipped")),
  }


def _score_logprobs(model: Any, prompt_ids: jax.Array, completion_ids: jax.Array, *, pad_id: int, eos_id: int, **kw):
  """Forward-only per-token log-probs of the completion: the reference-KL / agreement scoring pass."""
  graphdef, state = nnx.split(model)
  return tunix_common.compute_per_token_logps(
      graphdef, state, prompt_tokens=prompt_ids, completion_tokens=completion_ids, pad_id=pad_id, eos_id=eos_id, **kw
  )


def run_live(config: pyconfig.HyperParameters) -> dict[str, Any]:
  """Runs the benchmark on the local devices and exercises the read-only entry points."""
  # Flag conflicts first, before the mesh and the model are built.
  # The engine's own profiler runs only with `profiler` set; `profiler_steps` alone (default 5) is inert.
  if config.profiler and config.profiler_steps > 0:
    first = config.skip_first_n_steps_for_profiler
    if _PROFILE_STEPS.value > 0:
      raise ValueError(
          "Set --profile_steps or MaxText's profiler (with profiler_steps), not both: the engine would start a "
          "second trace inside the first."
      )
    max_logging.log(
        "engine_benchmark: WARNING the engine's profiler counts MICRO-steps, not optimizer steps as train.py "
        f"does: profiler_steps={config.profiler_steps} traces micro-steps [{first}, "
        f"{first + config.profiler_steps - 1}], {config.gradient_accumulation_steps} to a step. "
        "--profile_steps traces whole optimizer steps instead."
    )
  mesh = maxtext_utils.get_mesh_from_config(config)
  chip_devices = _DEVICES_PER_CHIP.value or devices_per_chip(jax.devices())
  peak = _PEAK_TFLOPS_PER_CHIP.value or peak_tflops_per_chip(jax.devices()[0].device_kind)
  report = {"mode": "run", "device_kind": jax.devices()[0].device_kind, **_describe_cell(config, mesh, chip_devices)}
  max_logging.log(f"engine_benchmark: {report}")

  compiles = CompileLog()
  engine = build_engine(config, mesh, _LOSS_TYPE.value)
  factory = PayloadFactory(config, engine, _LOSS_TYPE.value)
  checks: dict[str, Any] = {}
  report["checks"] = checks
  try:
    pool = [factory.micro_batch(i) for i in range(max(_DISTINCT_MICRO_BATCHES.value, 1))]
    report["input_sharding"] = str(engine.input_sharding(jax.ShapeDtypeStruct((factory.batch, factory.seq), np.int32)))

    compiles.phase = "compile"
    start = time.perf_counter()
    engine.compile(pool[0])
    report["compile_s"] = time.perf_counter() - start
    max_logging.log(f"engine_benchmark: compiled fwd_bwd, fwd_bwd_accum, update and eval in {report['compile_s']:.1f}s")

    accumulation = int(config.gradient_accumulation_steps)
    step_times, losses, norms, skipped = [], [], [], []
    for step in range(_STEPS.value):
      compiles.phase = "warmup" if step < _WARMUP_STEPS.value else "timed"
      elapsed, values = _time_step(engine, pool, step, accumulation)
      step_times.append(elapsed)
      losses.append(values["loss"])
      norms.append(values["grad_norm"])
      skipped.append(values["step_skipped"])
      rates = throughput(config, elapsed, chip_devices, peak)
      max_logging.log(
          f"engine_benchmark: step {step}: {elapsed:.3f}s, {rates['tokens_per_sec_per_chip']:.1f} tokens/s/chip, "
          f"{rates['tflops_per_sec_per_device']:.1f} TFLOP/s/device, loss {values['loss']}, grad_norm {values['grad_norm']}"
      )
    report.update(step_times_s=step_times, loss=losses, grad_norm=norms, step_skipped=skipped)

    retraced = recompiled_kernels(compiles.names("compile"), compiles.names("timed"))
    _check(
        checks,
        "no_kernel_recompile_in_timed_steps",
        not retraced,
        f"recompiled: {retraced}" if retraced else f"{len(compiles.names('timed'))} helper compile(s), no kernel",
    )

    if _PROFILE_STEPS.value > 0:
      compiles.phase = "profile"
      # Its own window, after the timed one: a trace slows the steps it covers, and tracing many
      # long steps overflows the TPU trace buffers and silently drops the later ones.
      profile_dir = _PROFILE_DIR.value or config.tensorboard_dir
      jax.profiler.start_trace(profile_dir)
      try:
        for step in range(_STEPS.value, _STEPS.value + _PROFILE_STEPS.value):
          with jax.profiler.StepTraceAnnotation("engine_step", step_num=step):
            _time_step(engine, pool, step, accumulation, annotate=True)
      finally:
        jax.profiler.stop_trace()
      report["profile_dir"] = profile_dir
      max_logging.log(f"engine_benchmark: traced {_PROFILE_STEPS.value} step(s) to {profile_dir}")

    timed = step_times[_WARMUP_STEPS.value :] or step_times
    report["timed_steps"] = len(timed)
    report["median_step_s"] = statistics.median(timed)
    report["mean_step_s"] = statistics.fmean(timed)
    report["min_step_s"], report["max_step_s"] = min(timed), max(timed)
    report.update(throughput(config, report["median_step_s"], chip_devices, peak))
    report["peak_tflops_per_chip"] = peak
    if _TARGET_TOKENS_PER_SEC_PER_CHIP.value > 0:
      report["target_tokens_per_sec_per_chip"] = _TARGET_TOKENS_PER_SEC_PER_CHIP.value
      report["fraction_of_target"] = report["tokens_per_sec_per_chip"] / _TARGET_TOKENS_PER_SEC_PER_CHIP.value

    finite = [v for v in losses if v is not None and math.isfinite(v)]
    _check(checks, "loss_finite", len(finite) == len(losses) > 0, f"{len(finite)} of {len(losses)} steps")
    _check(checks, "grad_norm_finite", all(n is not None and math.isfinite(n) for n in norms), norms)
    # A refused update still costs a step and reads like one; a benchmark that trained nothing is not
    # a benchmark. No `step_skipped` at all means the engine does not report it, which is not a skip.
    _check(checks, "updates_applied", not any(k for k in skipped if k), f"step_skipped per step: {skipped}")

    if _EVAL_BATCHES.value > 0:
      compiles.phase = "eval"
      before = (engine.train_step, engine.micro_step_count, engine.has_accumulated_grads)
      start = time.perf_counter()
      metrics = engine.run_eval(pool[i % len(pool)] for i in range(_EVAL_BATCHES.value))
      report["eval_s"] = time.perf_counter() - start
      report["eval_metrics"] = metrics
      report["eval_tokens_per_sec_per_chip"] = (
          _EVAL_BATCHES.value * factory.batch * factory.seq / report["eval_s"] / (jax.device_count() / chip_devices)
      )
      after = (engine.train_step, engine.micro_step_count, engine.has_accumulated_grads)
      eval_loss = metrics.get("loss")
      _check(
          checks,
          "run_eval",
          after == before
          and metrics.get("eval_batches") == _EVAL_BATCHES.value
          and eval_loss is not None
          and math.isfinite(eval_loss),
          f"loss {eval_loss}, batches {metrics.get('eval_batches')}, training state before {before} after {after}",
      )

    if _LOGPROB_BATCHES.value > 0 and _LOSS_TYPE.value == "grpo":
      compiles.phase = "logprob"
      start, shapes, bad = time.perf_counter(), [], 0
      for i in range(_LOGPROB_BATCHES.value):
        payload = pool[i % len(pool)]
        extra = {"routed_experts": payload.routed_experts} if payload.routed_experts is not None else {}
        if _LOGPS_CHUNK_SIZE.value > 0:
          extra["chunk_size"] = _LOGPS_CHUNK_SIZE.value
        logps = engine.fwd_only(
            _score_logprobs,
            payload.prompt_ids,
            payload.completion_ids,
            pad_id=_PAD_ID.value,
            eos_id=_EOS_ID.value,
            **extra,
        )
        logps = logps[0] if isinstance(logps, tuple) else logps
        shapes.append(tuple(logps.shape))
        bad += int(not bool(jnp.all(jnp.isfinite(logps))))
      report["logprob_s"] = time.perf_counter() - start
      expected = (factory.batch, factory.completion)
      _check(
          checks,
          "logprob_scoring",
          bad == 0 and all(s == expected for s in shapes),
          f"shapes {sorted(set(shapes))} (expected {expected}), {bad} non-finite",
      )

    if _WEIGHT_SYNC.value:
      compiles.phase = "weight_sync"
      rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
      start = time.perf_counter()
      try:
        staged = engine.prepare_weight_sync()
        report["weight_sync_s"] = time.perf_counter() - start
        engine.release_weight_sync()
        variables = sum(len(m.variables) for m in staged)
        report["weight_sync_peak_rss_gib_delta"] = (
            (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - rss_before) * 1024 / _GIB
        )
        _check(checks, "weight_sync", variables > 0, f"{len(staged)} work unit(s), {variables} variables")
      except Exception as exc:  # pylint: disable=broad-exception-caught
        # Recorded rather than raised: the timing above is still worth reporting, and the
        # check's FAIL fails the run on exit.
        _check(checks, "weight_sync", False, f"{type(exc).__name__}: {exc}")
  finally:
    report["compiles"] = compiles.summary()
    engine.close()
  return report


def _write_report(report: dict[str, Any], path: str) -> None:
  if not path or jax.process_index() != 0:
    return
  if path.startswith("gs://"):
    gcs_utils.write_dict_to_gcs_json(report, path)
  else:
    with open(path, "w", encoding="utf-8") as f:
      json.dump(report, f, indent=2, default=str)
  max_logging.log(f"engine_benchmark: report written to {path}")


def main(argv: Sequence[str]) -> None:
  if _MODE.value == "aot":
    # libtpu takes a machine-wide lockfile when it loads, so a second compile on the same host --
    # or any other process that loaded libtpu -- would abort here. Compiling touches no device,
    # so sharing is safe; a live run never sets this.
    os.environ.setdefault("ALLOW_MULTIPLE_LIBTPU_LOAD", "1")
    # As `maxtext_engine_compile`: the compiled RNG must match the one a run uses.
    jax.config.update("jax_default_prng_impl", "unsafe_rbg")
    os.environ["LIBTPU_INIT_ARGS"] = (
        os.environ.get("LIBTPU_INIT_ARGS", "") + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"
    )
    config = pyconfig.initialize(maxtext_engine_compile.with_engine_hlo_dump_defaults(argv))
    report = run_aot(config)
  else:
    pathwaysutils.initialize()
    jax.config.update("jax_default_prng_impl", "unsafe_rbg")
    config = pyconfig.initialize(argv)
    max_utils.print_system_information()
    report = run_live(config)

  max_logging.log("engine_benchmark: REPORT " + json.dumps(report, default=str))
  _write_report(report, _REPORT_PATH.value)
  failed = [name for name, check in report.get("checks", {}).items() if check["status"] == FAIL]
  if failed:
    raise SystemExit(f"engine_benchmark: failed checks: {failed}")


if __name__ == "__main__":
  app.run(main)
