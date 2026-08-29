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

"""Head-to-head: MaxTextTrainingEngine vs Tunix `peft_trainer_v2.PeftTrainer`.

Same model (Qwen3-0.6B from the public GCS checkpoint, wrapped in TunixMaxTextAdapter),
same loss (`tunix.rl.algo_core.grpo_loss_fn`), same TrainExample micro-batches, same optax
transformation -- driven once through each trainer, plus an independently computed
sum-of-grads / sum-of-denoms reference.

Setup is lifted from tests/post_training/integration/maxtext_engine_grpo_loss_test.py.
The comparison method is the one tests/end_to_end/tpu/compare_training_engine.py uses for
the engine-vs-native check: build micro-batches with *unequal* valid-token counts, so
mean-of-means and sum/sum normalization visibly disagree.

Run one trainer per process (`--trainer`) so the HBM number belongs to that trainer alone.
`--trainer=ref` computes only the reference gradient.
"""

import argparse
import contextlib
import dataclasses
import functools
import json
import os
import time
from typing import Any

from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.common import common_types
from maxtext.configs import pyconfig
from maxtext.training_engine import maxtext_engine
from maxtext.utils import maxtext_utils
from maxtext.utils import model_creation_utils
from maxtext.utils import train_utils
from tests.utils.test_helpers import get_test_config_path

from tunix.experimental.train import peft_trainer_v2
from tunix.sft import sharding_utils
from tunix.rl import algo_core
from tunix.rl import common as tunix_common

_PAD_ID = 151643
_EOS_ID = 151645
_CHECKPOINT = "gs://maxtext-model-checkpoints/qwen3-0.6b/2025-10-27/scanned/0/items"


def _config(**overrides) -> pyconfig.HyperParameters:
  """The GRPO integration test's config, plus whatever this run overrides."""
  argv = [
      "compare_tunix_trainer.py",
      get_test_config_path("base.yml"),
      "model_name=qwen3-0.6b",
      "run_name=compare_tunix_trainer",
      "enable_checkpointing=True",
      f"load_parameters_path={_CHECKPOINT}",
      "scan_layers=True",
      "convert_checkpoint_if_possible=False",
      "init_weights_seed=42",
      "dtype=float32",
      "weight_dtype=float32",
      "grad_dtype=float32",
      "enable_tensorboard=False",
      "record_internal_nn_metrics=False",
      "skip_jax_distributed_system=True",
      "warmup_steps_fraction=0.0",
      "learning_rate=1e-4",
      "max_target_length=64",
      # MaxText clips inside its update kernel; Tunix does not clip at all. Clipping on
      # only one side would mask exactly the normalization difference being measured.
      "gradient_clipping_threshold=0.0",
  ]
  argv.extend(f"{k}={v}" for k, v in overrides.items())
  return pyconfig.initialize(argv)


@dataclasses.dataclass(frozen=True)
class _GrpoConfig:
  beta: float = 0.04
  epsilon: float = 0.2
  epsilon_high: float = 0.2
  loss_algo: str = "grpo"
  loss_agg_mode: str = "token-mean"
  temperature: float = 1.0
  kl_loss_mode: str = "low_var_kl"
  kl_clamp_value: float | None = None


def _build_model(cfg, mesh):
  return model_creation_utils.from_pretrained(
      config=cfg,
      mesh=mesh,
      model_mode=common_types.MODEL_MODE_TRAIN,
      rng_key=jax.random.PRNGKey(cfg.init_weights_seed),
      wrap_with_tunix_adapter=True,
      tokenizer_pad_id=_PAD_ID,
  )


def _train_example(model, algo_config, seed, batch, prompt_len, completion_len, valid_len):
  """One micro-batch whose reference log-probs are the model's own.

  `valid_len` truncates completion_mask, which is what makes the per-micro-batch
  denominator (`Σ completion_mask`) differ across the accumulation window.
  """
  rng = np.random.default_rng(seed)
  prompt_ids = jnp.asarray(rng.integers(1000, 2000, size=(batch, prompt_len)), dtype=jnp.int32)
  completion_ids = jnp.asarray(rng.integers(1000, 2000, size=(batch, completion_len)), dtype=jnp.int32)

  graphdef, state = nnx.split(model)
  ref_logps = tunix_common.compute_per_token_logps(
      graphdef,
      state,
      prompt_tokens=prompt_ids,
      completion_tokens=completion_ids,
      pad_id=_PAD_ID,
      eos_id=_EOS_ID,
      stop_gradient=True,
      temperature=algo_config.temperature,
  )
  if isinstance(ref_logps, tuple):
    ref_logps = ref_logps[0]

  mask = (jnp.arange(completion_len)[None, :] < valid_len).astype(jnp.int32)
  mask = jnp.broadcast_to(mask, (batch, completion_len))
  return tunix_common.TrainExample(
      prompt_ids=prompt_ids,
      prompt_mask=jnp.ones((batch, prompt_len), dtype=jnp.int32),
      completion_ids=completion_ids,
      completion_mask=mask,
      advantages=jnp.asarray(rng.normal(size=(batch,)), dtype=jnp.float32),
      ref_per_token_logps=ref_logps,
      old_per_token_logps=None,
  )


def _shard_example(example, mesh):
  """Places every leaf batch-sharded over `fsdp`, once, before any trainer sees it.

  Both trainers want this and neither does it for you in a way the other would match:
  MaxText bakes `P('fsdp', None)` into the compiled kernel's `in_shardings` and raises if
  the argument arrives replicated, while Tunix re-shards inside `_prepare_payload`. Doing
  it up front means the reference gradient and both trainers read byte-identical, identically
  placed inputs, so nothing below is a placement artifact.
  """

  def place(x):
    if not isinstance(x, jax.Array):
      return x
    spec = jax.sharding.PartitionSpec(*(("fsdp",) + (None,) * (x.ndim - 1)))
    return jax.device_put(x, jax.sharding.NamedSharding(mesh, spec))

  return jax.tree.map(place, example)


def _model_inputs(payload, algo_config):
  return {"train_example": payload, "algo_config": algo_config, "pad_id": _PAD_ID, "eos_id": _EOS_ID}


def _params(model):
  return nnx.to_pure_dict(nnx.state(model, nnx.Param))


def _digest(tree) -> dict[str, float]:
  """A few order-insensitive scalars that pin down a whole gradient/param tree."""
  leaves = [jnp.asarray(x).astype(jnp.float32) for x in jax.tree.leaves(tree)]
  sq = sum(float(jnp.sum(x * x)) for x in leaves)
  return {
      "l2_norm": float(np.sqrt(sq)),
      "abs_sum": sum(float(jnp.sum(jnp.abs(x))) for x in leaves),
      "max_abs": max(float(jnp.max(jnp.abs(x))) for x in leaves),
      "num_leaves": len(leaves),
  }


def _rel_delta(a, b) -> dict[str, float]:
  la = [jnp.asarray(x).astype(jnp.float32) for x in jax.tree.leaves(a)]
  lb = [jnp.asarray(x).astype(jnp.float32) for x in jax.tree.leaves(b)]
  diff_sq = sum(float(jnp.sum((x - y) ** 2)) for x, y in zip(la, lb))
  base_sq = sum(float(jnp.sum(y * y)) for y in lb)
  max_abs = max(float(jnp.max(jnp.abs(x - y))) for x, y in zip(la, lb))
  return {"rel_l2": float(np.sqrt(diff_sq) / (np.sqrt(base_sq) + 1e-30)), "max_abs_diff": max_abs}


def _spec_report(tree, label, k=3):
  """PartitionSpec of the `k` biggest leaves -- enough to tell sharded from replicated.

  Replication of a parameter-sized tree is the difference between ~0.3 GiB and ~2.4 GiB per
  device here, so this is the first thing to check when two trainers disagree on HBM.
  """
  leaves = [x for x in jax.tree.leaves(tree) if isinstance(x, jax.Array)]
  leaves.sort(key=lambda x: -x.size)
  out = {}
  for x in leaves[:k]:
    spec = getattr(getattr(x, "sharding", None), "spec", None)
    out[f"{label}{list(x.shape)}"] = str(spec)
  out["_global_gib"] = round(sum(x.size * x.dtype.itemsize for x in leaves) / 2**30, 3)
  out["_per_device_gib"] = round(
      sum(int(np.prod(x.sharding.shard_shape(x.shape))) * x.dtype.itemsize for x in leaves) / 2**30, 3
  )
  return out


def _peak_hbm_gib() -> float:
  peak = 0
  for d in jax.local_devices():
    stats = d.memory_stats() or {}
    peak = max(peak, stats.get("peak_bytes_in_use", 0))
  return peak / 2**30


@functools.wraps(algo_core.grpo_loss_fn)
def _diff_fn(model, *args, **kwargs):
  """Exactly Tunix v2's `diff_fn`: differentiate the *unreduced sum*, keep the LossOutput.

  `functools.wraps` is load-bearing, not cosmetic -- `nnx.value_and_grad` resolves kwargs
  to positions off the wrapped signature and rejects a bare `**kwargs` function.
  """
  out = algo_core.grpo_loss_fn(model, *args, **kwargs)
  return out.primary_loss.unreduced_sum, out


def _reference_grads(model, examples, algo_config):
  """The exact sum/sum gradient: Σ_i ∇(unreduced_sum_i) / Σ_i denom_i.

  Computed with the same `nnx.value_and_grad(diff_fn)` Tunix differentiates, so any
  disagreement below is normalization, not a different derivative.
  """

  grad_fn = nnx.value_and_grad(_diff_fn, argnums=0, has_aux=True)

  total_grads = None
  total_denom = jnp.float32(0.0)
  total_sum = jnp.float32(0.0)
  per_micro = []
  for ex in examples:
    (unreduced_sum, out), grads = grad_fn(model, **_model_inputs(ex, algo_config))
    grads = nnx.to_pure_dict(grads)
    denom = jnp.asarray(out.primary_loss.denominator, jnp.float32)
    total_grads = grads if total_grads is None else jax.tree.map(jnp.add, total_grads, grads)
    total_denom = total_denom + denom
    total_sum = total_sum + jnp.asarray(unreduced_sum, jnp.float32)
    per_micro.append(
        {"denominator": float(denom), "unreduced_sum": float(unreduced_sum), "loss": float(out.primary_loss.compute())}
    )

  sum_over_sum = jax.tree.map(lambda g: g / total_denom, total_grads)
  # What mean-of-means produces instead: each micro-batch pre-scaled by its own denominator.
  # Recomputing per-micro grads to build it would double the cost, so it is derived in the
  # comparison step from the per-trainer numbers rather than here.
  return {
      "grads": sum_over_sum,
      "total_denominator": float(total_denom),
      "total_unreduced_sum": float(total_sum),
      "loss": float(total_sum / total_denom),
      "per_micro": per_micro,
  }


def _reference_mean_of_means(model, examples, algo_config):
  """Tunix v2's normalization, computed independently: (1/N) Σ_i ∇(sum_i)/denom_i."""

  grad_fn = nnx.value_and_grad(_diff_fn, argnums=0, has_aux=True)
  acc = None
  for ex in examples:
    (_, out), grads = grad_fn(model, **_model_inputs(ex, algo_config))
    scale = out.primary_loss.compute_scale()
    scaled = jax.tree.map(lambda g: g * scale, nnx.to_pure_dict(grads))  # pylint: disable=cell-var-from-loop
    acc = scaled if acc is None else jax.tree.map(jnp.add, acc, scaled)
  return jax.tree.map(lambda g: g / float(len(examples)), acc)


def run_maxtext(cfg, mesh, examples, algo_config, iters, compile_engine=True, kernel_bench=False):
  """Drives one accumulation window plus `iters` timed updates through MaxText's engine."""
  engine = maxtext_engine.MaxTextTrainingEngine(cfg, mesh=mesh, wrap_with_tunix_adapter=True, tokenizer_pad_id=_PAD_ID)
  engine.with_loss_fn(algo_core.grpo_loss_fn, has_aux=True).with_gen_model_input_fn(
      lambda p: _model_inputs(p, algo_config)
  )
  # Without this the engine stays on the eager path -- every optimizer primitive dispatched
  # and XLA-compiled one at a time. Tunix v2 has no equivalent switch: its `compile()` is
  # `pass` and `fwd_bwd`/`update` always go through `nnx.jit`.
  if compile_engine:
    engine.compile(examples[0])
  # Numerics come from step 0 and only step 0. The examples carry `ref_per_token_logps`
  # taken from the initial weights, so any warm-up update first would move the policy away
  # from its own reference and blow the KL term up -- the loss stops being comparable to the
  # reference computed at step 0. Timing runs afterwards, where drift does not matter.
  for ex in examples:
    engine.fwd_bwd(ex)
  eff = jax.tree.map(
      lambda g: jnp.asarray(g, jnp.float32) / jnp.asarray(engine._accumulated_denominator, jnp.float32),  # pylint: disable=protected-access
      nnx.to_pure_dict(engine._accumulated_grads),  # pylint: disable=protected-access
  )
  eff = jax.tree.map(jnp.copy, eff)
  denom = float(jnp.asarray(engine._accumulated_denominator))  # pylint: disable=protected-access
  before = jax.tree.map(jnp.copy, _params(engine.model))
  engine.update()
  after = _params(engine.model)
  buf = engine.get_metrics(clear_cache=True)
  # The recorder appends one entry per micro-step, so `compute()` returns a *vector* of
  # per-micro losses, not a step scalar. MetricsLogger._process_metrics reduces it with
  # np.mean (no aggregation_fn is registered for "loss"), i.e. mean-of-means -- even though
  # the gradient the engine actually applies is sum/sum. Report all three so the gap is visible.
  wm = buf.weighted_metrics["loss"]
  per_micro = np.asarray(jnp.asarray(wm.compute()).reshape(-1))
  loss = float(per_micro[0])
  loss_logged_mean_of_means = float(np.mean(per_micro))
  loss_sum_over_sum = float(jnp.sum(jnp.asarray(wm.unreduced_sum)) / jnp.sum(jnp.asarray(wm.denominator)))
  kl = float(jnp.asarray(buf.weighted_metrics["kl"].compute()).reshape(-1)[0])
  grad_norm = float(jnp.asarray(buf.scalar_metrics.get("gradient_norm", jnp.nan)).reshape(-1)[0])
  aux_recorded = sorted(set(buf.weighted_metrics) | set(buf.scalar_metrics))
  # Reduced to scalars now, not after the timing loop: `update()` donates the parameter
  # buffers, so the next loop iteration deletes the arrays `after` points at.
  opt_state = nnx.state(engine._optimizer, nnx.optimizer.OptState)  # pylint: disable=protected-access
  numerics = {
      "sharding": {
          **_spec_report(_params(engine.model), "param"),
          **{f"opt.{k}": v for k, v in _spec_report(jax.tree.leaves(nnx.to_pure_dict(opt_state)), "slot").items()},
          **{f"grad.{k}": v for k, v in _spec_report(eff, "grad").items()},
      },
      "loss": loss,
      "loss_per_micro": [round(float(x), 6) for x in per_micro],
      "loss_logged_mean_of_means": loss_logged_mean_of_means,
      "loss_sum_over_sum": loss_sum_over_sum,
      "kl": kl,
      "grad_norm": grad_norm,
      "metrics_recorded": aux_recorded,
      "accumulated_denominator": denom,
      "effective_grads": eff,
      "weight_delta": _rel_delta(after, before),
      "delta_digest": _digest(jax.tree.map(jnp.subtract, after, before)),
  }
  del after, before

  def sync():
    # Params alone are not enough: `fwd_bwd` leaves them untouched, so blocking on them
    # returns immediately and the micro-step's real work would be billed to `update`.
    grads = engine._accumulated_grads  # pylint: disable=protected-access
    leaves = jax.tree.leaves(_params(engine.model))
    if grads is not None:
      leaves += jax.tree.leaves(nnx.to_pure_dict(grads))
    return leaves

  timings = _time_loop(engine.fwd_bwd, engine.update, examples, iters, sync)

  if kernel_bench:
    # Strip the Python wrapper off and call the compiled executable directly, so the
    # fwd_bwd number splits into "XLA" and "everything the trainer does around it".
    model = engine._state.model if engine._state is not None else engine.model  # pylint: disable=protected-access
    dynamic_batch, _ = maxtext_engine._split_static_and_dynamic(  # pylint: disable=protected-access
        engine._prepare_batch(examples[0])  # pylint: disable=protected-access
    )  # pylint: disable=protected-access

    def split():
      return nnx.split(model, nnx.Param, ...)

    _, params, rest = split()
    with engine._sharding_ctx():  # pylint: disable=protected-access
      timings["kernel_only_ms_median"] = _bench(
          lambda: engine._compiled_fwd_bwd(params, rest, dynamic_batch), iters  # pylint: disable=protected-access
      )
    timings["nnx_split_ms_median"] = _bench(split, iters, block=False)
    with engine._sharding_ctx():  # pylint: disable=protected-access
      f = engine._compiled_fwd_bwd  # pylint: disable=protected-access
      c = f.lower(params, rest, dynamic_batch).compile() if hasattr(f, "lower") else f
      timings["compiled"] = _compiled_stats(c)
  return {**numerics, **timings}


def run_tunix(cfg, mesh, examples, algo_config, iters, with_axis_rules, kernel_bench=False):
  """The same window and timed updates through Tunix `peft_trainer_v2.PeftTrainer`."""
  model = _build_model(cfg, mesh)
  _, tx = train_utils.create_training_optimizer(cfg, model)
  tcfg = peft_trainer_v2.TrainingConfig(
      eval_every_n_steps=10**9,
      max_steps=None,
      gradient_accumulation_steps=len(examples) if len(examples) > 1 else None,
      data_sharding_axis=("fsdp",),
  )
  trainer = peft_trainer_v2.PeftTrainer(model, tx, tcfg)
  trainer.with_loss_fn(algo_core.grpo_loss_fn, has_aux=True).with_gen_model_input_fn(
      lambda p: _model_inputs(p, algo_config)
  )

  ctx = nn_partitioning.axis_rules(cfg.logical_axis_rules) if with_axis_rules else _null_ctx()
  with ctx:
    # Re-place the payloads through Tunix's own helper once, up front. `_prepare_payload`
    # compares `x.sharding.spec` against `PartitionSpec("fsdp")` exactly, and the rank-aware
    # `P('fsdp', None)` the harness (and MaxText's compiled `in_shardings`) uses is not `==`
    # to it -- so without this every micro-step re-materializes the whole payload through
    # `jax.make_array_from_process_local_data`, a host round trip, and the measurement would
    # be of that rather than of the trainer. The resulting device layout is the same.
    examples = [sharding_utils.shard_input(ex, tcfg.data_sharding_axis) for ex in examples]
    # Step 0 only, for the same reason as the MaxText side: the reference log-probs in
    # `examples` belong to the initial weights.
    for ex in examples:
      trainer.fwd_bwd(ex)
    acc = trainer.grad_accumulator
    eff = jax.tree.map(jnp.copy, nnx.to_pure_dict(acc.get()))
    denom = float(jnp.asarray(acc.denom[...]))
    before = jax.tree.map(jnp.copy, _params(model))
    trainer.update()
    after = _params(model)
    loss, grad_norm = _tunix_metrics(trainer)
    kl = float("nan")  # see _tunix_metrics: v2 drops the loss aux dict
    opt_state = nnx.state(trainer.optimizer, nnx.optimizer.OptState)
    numerics = {
        "sharding": {
            **_spec_report(_params(model), "param"),
            **{f"opt.{k}": v for k, v in _spec_report(jax.tree.leaves(nnx.to_pure_dict(opt_state)), "slot").items()},
            **{f"grad.{k}": v for k, v in _spec_report(eff, "grad").items()},
        },
        "loss": loss,
        "kl": kl,
        "grad_norm": grad_norm,
        "accumulated_denominator": denom,
        "effective_grads": eff,
        "weight_delta": _rel_delta(after, before),
        "delta_digest": _digest(jax.tree.map(jnp.subtract, after, before)),
    }
    del after, before

    def sync():
      return jax.tree.leaves(_params(model)) + jax.tree.leaves(trainer.grad_accumulator.grads)

    timings = _time_loop(trainer.fwd_bwd, trainer.update, examples, iters, sync)

    if kernel_bench:
      fwd_bwd_step, _, _ = trainer.jit_fwd_bwd_update_and_eval_step()
      payload = trainer._prepare_payload(examples[0])  # pylint: disable=protected-access
      timings["kernel_only_ms_median"] = _bench(
          lambda: fwd_bwd_step(inputs=payload, model=trainer.model, grad_accumulator=trainer.grad_accumulator), iters
      )
      timings["nnx_split_ms_median"] = _bench(lambda: nnx.split(trainer.model), iters, block=False)
      base, bound = fwd_bwd_step, {}
      while isinstance(base, functools.partial):  # v2 wraps the jit fn in maybe_cache_and_partial
        bound = {**base.keywords, **bound}
        base = base.func
      timings["compiled"] = _compiled_stats(
          base.lower(inputs=payload, model=trainer.model, grad_accumulator=trainer.grad_accumulator, **bound).compile()
      )
  return {**numerics, **timings}


def _tunix_metrics(trainer):
  """Reads step 0's loss and grad norm off Tunix's own buffer.

  Not via `get_metrics()`: `_write_train_metrics` deliberately skips the first step so
  metric I/O overlaps the next one, so after a single update `get_metrics()` still returns
  the empty `MetricsBuffer(id=-1)` and every field reads as NaN. The step-0 buffer is
  sitting in `_prev_buffered_train_metrics`.

  There is no `kl` to read. `_post_process_train_step` is `pass` in the base PeftTrainer,
  so grpo_loss_fn's whole aux dict -- kl, kl_loss, entropy, clip fractions -- is discarded.
  MaxText's engine records all of it.
  """
  buf = trainer._prev_buffered_train_metrics or trainer._buffered_train_metrics  # pylint: disable=protected-access
  if buf is None:
    return float("nan"), float("nan")
  grad_norm = float("nan")
  entry = buf.additional_metrics.get("grad_norm")
  if entry is not None:
    values, op = entry
    grad_norm = float(np.asarray(op([np.asarray(v, dtype=np.float32) for v in values])))
  return float(buf.loss), grad_norm


class _null_ctx:  # pylint: disable=invalid-name

  def __enter__(self):
    return None

  def __exit__(self, *exc):
    return False


def _compiled_stats(compiled):
  """FLOPs / bytes-accessed / HBM breakdown for an AOT-compiled executable."""
  out = {}
  try:
    ca = compiled.cost_analysis()
    ca = ca[0] if isinstance(ca, list) else ca
    out["gflops"] = round(float(ca.get("flops", 0)) / 1e9, 1)
    out["bytes_accessed_gib"] = round(float(ca.get("bytes accessed", 0)) / 2**30, 2)
  except Exception as e:  # pylint: disable=broad-except
    out["cost_error"] = repr(e)[:200]
  try:
    ma = compiled.memory_analysis()
    for f in ("argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes", "alias_size_in_bytes"):
      out[f.replace("_in_bytes", "_gib")] = round(getattr(ma, f, 0) / 2**30, 3)
  except Exception as e:  # pylint: disable=broad-except
    out["mem_error"] = repr(e)[:200]
  try:
    txt = compiled.as_text()
    for op in ("all-gather", "all-reduce", "reduce-scatter", "all-to-all", "collective-permute"):
      n = txt.count(op + "(") + txt.count(op + "-start(")
      if n:
        out["hlo_" + op] = n
  except Exception as e:  # pylint: disable=broad-except
    out["hlo_error"] = repr(e)[:200]
  return out


def _bench(fn, iters, block=True):
  """Median wall clock of `fn`, warmed up once."""
  out = fn()
  if block:
    jax.block_until_ready(out)
  times = []
  for _ in range(iters):
    t0 = time.perf_counter()
    out = fn()
    if block:
      jax.block_until_ready(out)
    times.append((time.perf_counter() - t0) * 1e3)
  return float(np.median(times))


# Set from --xprof; `tag` is rebound per trainer so each gets its own xplane directory.
_XPROF = {"dir": None, "tag": None}


@contextlib.contextmanager
def _maybe_trace():
  """Capture an xplane trace of the steady-state loop only, never warmup/compile."""
  if not _XPROF["dir"]:
    yield None
    return
  path = os.path.join(_XPROF["dir"], _XPROF["tag"])
  os.makedirs(path, exist_ok=True)
  jax.profiler.start_trace(path)
  try:
    yield path
  finally:
    jax.profiler.stop_trace()


def _time_loop(fwd_bwd, update, examples, iters, leaves_fn):
  """Median wall clock of one full update, plus the peak HBM the whole run touched."""
  fwd_times, update_times = [], []
  with _maybe_trace() as trace_path:
    for step in range(iters):
      # StepTraceAnnotation gives xprof real step boundaries, so the trace viewer can
      # report a per-step breakdown instead of one undifferentiated blob.
      with jax.profiler.StepTraceAnnotation("train", step_num=step):
        for ex in examples:
          t0 = time.perf_counter()
          with jax.profiler.TraceAnnotation("fwd_bwd"):
            fwd_bwd(ex)
            jax.block_until_ready(leaves_fn())
          fwd_times.append((time.perf_counter() - t0) * 1e3)
        t0 = time.perf_counter()
        with jax.profiler.TraceAnnotation("update"):
          update()
          jax.block_until_ready(leaves_fn())
        update_times.append((time.perf_counter() - t0) * 1e3)
  n = len(examples)
  per_update = [sum(fwd_times[i * n : (i + 1) * n]) + update_times[i] for i in range(iters)]
  return {
      "fwd_bwd_ms_median": float(np.median(fwd_times)),
      "update_ms_median": float(np.median(update_times)),
      "total_ms_median": float(np.median(per_update)),
      "fwd_bwd_ms_all": [round(t, 2) for t in fwd_times],
      "update_ms_all": [round(t, 2) for t in update_times],
      "peak_hbm_gib": _peak_hbm_gib(),
      "xprof_local_dir": trace_path,
  }


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument(
      "--trainer",
      choices=["maxtext", "tunix", "tunix_no_axis_rules", "ref", "both"],
      required=True,
      help="'both' runs the two trainers back to back in one process, which is the "
      "only way to diff their effective gradients directly; take HBM from the "
      "single-trainer runs instead.",
  )
  ap.add_argument("--ga", type=int, default=1)
  ap.add_argument("--batch", type=int, default=8)
  ap.add_argument("--iters", type=int, default=5)
  ap.add_argument("--ragged", action="store_true", help="unequal valid-token counts per micro-batch")
  ap.add_argument(
      "--skip-ref",
      action="store_true",
      help="perf runs only: the reference pass allocates its own grad tree and would\n"
      "contaminate this process's peak-HBM reading",
  )
  ap.add_argument(
      "--no-compile",
      action="store_true",
      help="maxtext only: skip engine.compile(), i.e. the eager path the GRPO integration test exercises",
  )
  ap.add_argument(
      "--kernel-bench", action="store_true", help="also time the bare compiled executable and the per-call nnx.split"
  )
  ap.add_argument("--xprof", default=None, help="local directory to write one xplane trace per trainer into")
  ap.add_argument("--profile", action="store_true")
  ap.add_argument("--out", default=None)
  args = ap.parse_args()

  prompt_len, completion_len = 8, 8
  if args.ragged and args.ga > 1:
    valid = [completion_len, 2, 5, 1, 8, 3, 7, 4][: args.ga]
  else:
    valid = [completion_len] * args.ga

  cfg = _config(
      gradient_accumulation_steps=args.ga,
      micro_batch_size_to_train_on=args.batch,
      per_device_batch_size=max(1, args.batch // jax.device_count()),
  )
  mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
  algo_config = _GrpoConfig()

  # One model builds the shared micro-batches so every trainer sees byte-identical inputs.
  ref_model = _build_model(cfg, mesh)
  examples = [
      _shard_example(
          _train_example(
              ref_model,
              algo_config,
              seed=i,
              batch=args.batch,
              prompt_len=prompt_len,
              completion_len=completion_len,
              valid_len=v,
          ),
          mesh,
      )
      for i, v in enumerate(valid)
  ]

  one_param = jax.tree.leaves(_params(ref_model))[0]
  result: dict[str, Any] = {
      "mesh": {k: int(v) for k, v in mesh.shape.items() if v > 1},
      "param_sharding": str(getattr(one_param, "sharding", None)),
      "param_shape": list(one_param.shape),
      "trainer": args.trainer + ("+eager" if args.no_compile and args.trainer == "maxtext" else ""),
      "ga": args.ga,
      "batch": args.batch,
      "valid_lens": valid,
      "devices": jax.device_count(),
  }

  ref_grads = None
  if args.skip_ref:
    del ref_model
  else:
    with mesh, nn_partitioning.axis_rules(cfg.logical_axis_rules):
      ref = _reference_grads(ref_model, examples, algo_config)
      result["reference"] = {k: v for k, v in ref.items() if k != "grads"}
      result["reference"]["digest"] = _digest(ref["grads"])
      if args.ga > 1:
        mom = _reference_mean_of_means(ref_model, examples, algo_config)
        result["reference"]["mean_of_means_digest"] = _digest(mom)
        result["reference"]["mean_of_means_vs_sum_over_sum"] = _rel_delta(mom, ref["grads"])
      ref_grads = ref["grads"]
    del ref_model

  if args.trainer == "ref":
    _emit(result, args.out)
    return

  def _go(which):
    if which == "maxtext":
      return run_maxtext(
          cfg, mesh, examples, algo_config, args.iters, compile_engine=not args.no_compile, kernel_bench=args.kernel_bench
      )
    return run_tunix(
        cfg, mesh, examples, algo_config, args.iters, with_axis_rules=which == "tunix", kernel_bench=args.kernel_bench
    )

  which = ["maxtext", "tunix"] if args.trainer == "both" else [args.trainer]
  effs = {}
  _XPROF["dir"] = args.xprof
  with mesh:
    for name in which:
      _XPROF["tag"] = f"{name}_ga{args.ga}_bs{args.batch}"
      if args.profile:
        import cProfile  # pylint: disable=import-outside-toplevel
        import pstats  # pylint: disable=import-outside-toplevel

        pr = cProfile.Profile()
        pr.enable()
        out = _go(name)
        pr.disable()
        pstats.Stats(pr).sort_stats("cumulative").print_stats(35)
      else:
        out = _go(name)
      effs[name] = out.pop("effective_grads")
      out["effective_grad_digest"] = _digest(effs[name])
      if ref_grads is not None:
        out["effective_grad_vs_eager_reference"] = _rel_delta(effs[name], ref_grads)
      if args.trainer == "both":
        result[name] = out
      else:
        result.update(out)

  if len(effs) == 2:
    # The number that actually answers "do the two trainers agree": trainer-vs-trainer,
    # not trainer-vs-eager-reference. Both run jitted and batch-sharded; the reference runs
    # eagerly, and that alone moves f32 reductions by ~5e-3 relative.
    result["maxtext_vs_tunix"] = _rel_delta(effs["maxtext"], effs["tunix"])
  _emit(result, args.out)


def _emit(result, out_path):
  text = json.dumps(result, indent=2, sort_keys=True)
  print("=== RESULT ===")
  print(text)
  if out_path:
    with open(out_path, "w", encoding="utf-8") as f:
      f.write(text)


if __name__ == "__main__":
  os.environ.setdefault("JAX_PLATFORMS", "")
  main()
