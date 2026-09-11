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

"""Compares MaxTextTrainingEngine against Tunix `peft_trainer_v2.PeftTrainer`.

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
import math
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


def _as_lens(valid_len, batch):
  """Broadcasts `valid_len` to `[batch]` scored-token counts, one per sequence.

  Uniform counts are the degenerate case; see `_packed_train_example`.
  """
  lens = np.asarray(valid_len, np.int64).reshape(-1)
  if lens.size == 1:
    lens = np.repeat(lens, batch)
  if lens.size != batch:
    raise ValueError(f"valid_len has {lens.size} entries; expected 1 or --batch {batch}.")
  return lens


# Cycled over a micro-batch's sequences so adjacent segments differ. Floored at half the
# ceiling: any inequality breaks the degeneracy, since the collapse into `token-mean` needs the
# counts *equal* rather than close, so a wider spread buys no coverage and measurably costs.
#
# TODO(mazumdera): explain the ragged step-1 loss spike. Measured on
# `--batch 24 --seq 128 --layout-steps 3`: a 0.125 floor puts step-1 loss at 4.99e+05, this 0.5
# floor at 2.26e+03, uniform segments at 0.66. It is confined to the corroborating loss column --
# step 2 recovers to 0.41, the arms agree to ~15%, weights move a normal 2.8e-03, and the verdict
# metric is the step-0 gradient -- so it does not affect any recorded verdict. Until it is
# understood, do not read a step-1 loss under packing as meaningful.
_SEGMENT_LEN_FRACTIONS = (1.0, 0.5625, 0.875, 0.625, 0.9375, 0.5, 0.8125, 0.75)


def _segment_valid_lens(ceiling, batch, offset):
  """Per-sequence scored-token counts spread over `[1, ceiling]`, rotated by `offset`.

  `ceiling` is the micro-batch's own cap, which `--ragged` varies across the accumulation
  window; this varies the segments underneath it, so the two compose.
  """
  f = _SEGMENT_LEN_FRACTIONS
  return np.array([max(1, round(ceiling * f[(offset + i) % len(f)])) for i in range(batch)], np.int64)


def _train_example(model, algo_config, seed, batch, prompt_len, completion_len, valid_len):
  """One micro-batch whose reference log-probs are the model's own.

  `valid_len` truncates completion_mask, which is what makes the per-micro-batch
  denominator (`Σ completion_mask`) differ across the accumulation window. Per sequence: a
  scalar applies to all, a `[batch]` array gives each its own, which is what keeps this arm
  comparable to a packed one built with ragged segments.
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

  lens = _as_lens(valid_len, batch)
  mask = jnp.asarray((np.arange(completion_len)[None, :] < lens[:, None]).astype(np.int32))
  return tunix_common.TrainExample(
      prompt_ids=prompt_ids,
      prompt_mask=jnp.ones((batch, prompt_len), dtype=jnp.int32),
      completion_ids=completion_ids,
      completion_mask=mask,
      advantages=jnp.asarray(rng.normal(size=(batch,)), dtype=jnp.float32),
      ref_per_token_logps=ref_logps,
      old_per_token_logps=None,
  )


def _packed_train_example(
    model, algo_config, seed, batch, prompt_len, completion_len, valid_len, segments_per_row, pad_to=None, leak=False
):
  """The same micro-batch as `_train_example`, with `segments_per_row` sequences per row.

  Draws in the same order and with the same shapes, so for a given seed the per-sequence tokens
  and advantages are identical and the two builders differ only in layout.

  Packing forces three shape changes: a row is one stream, so `prompt_ids` is `[rows, 0]` and
  every token lives in `completion_ids` (`compute_per_token_logps` front-pads its output back to
  the row width, keeping `ref_per_token_logps` aligned index-for-index); `advantages` becomes
  per-token, one scalar per row being unable to distinguish that row's segments; and
  `completion_mask` masks each segment's prompt span as well as its tail past `valid_len`,
  leaving the unpacked builder's active token count and so its loss denominator.

  `valid_len` is per sequence, ordered as the unpacked builder draws them: sequence `b` is
  segment `b % segments_per_row` of row `b // segments_per_row`. Uniform counts are degenerate,
  a mean of equal-length per-segment means being the mean over their union.

  `pad_to` widens the row past the tokens the segments need, leaving slack in segment 0 -- the
  padding bucket `num_segments` counts but that nothing here used to fill. Slack is the only
  condition under which treating that bucket as a real segment changes the result, which is why
  two of the four budgets tunix's `test_sequence_packing` sweeps are padded.

  `leak` puts every real token in segment 1 so attention crosses the boundaries: the negative
  control for `--compare-layouts`, calibrating what a broken run scores. It is valid only under
  uniform `valid_len`, since `segment_ids` drives the segmented loss too and collapsing
  equal-count segments leaves that reduction unchanged; ragged counts would break attention and
  loss at once. `segment_positions` stays correct either way.

  Compare gradients, not loss. Against the adapter as it was before it accepted `segment_ids`,
  with attention leaking across every boundary, packed and unpacked agreed on the loss to seven
  decimals while their gradients were 70% apart in relative L2: `ref_per_token_logps` comes from
  the same forward pass the loss then scores, so contamination cancels out of the loss and
  survives only in the derivative.
  """
  if batch % segments_per_row:
    raise ValueError(f"--batch {batch} must be divisible by --segments-per-row {segments_per_row}.")
  rows = batch // segments_per_row
  seq_len = prompt_len + completion_len
  content_len = segments_per_row * seq_len
  packed_len = content_len if pad_to is None else pad_to
  if packed_len < content_len:
    raise ValueError(f"pad_to {packed_len} cannot hold {segments_per_row} x {seq_len} = {content_len} tokens.")
  slack = packed_len - content_len

  # Same draws, same order, same shapes as `_train_example`.
  rng = np.random.default_rng(seed)
  prompt_ids = rng.integers(1000, 2000, size=(batch, prompt_len))
  completion_ids = rng.integers(1000, 2000, size=(batch, completion_len))
  advantages = rng.normal(size=(batch,))

  def _pad(row, fill):
    """Right-pads one `[content_len]` row out to `packed_len`, broadcast over rows later."""
    return row if not slack else np.concatenate([row, np.full(slack, fill, dtype=row.dtype)])

  ids = np.concatenate([prompt_ids, completion_ids], axis=1).reshape(rows, content_len)
  if slack:
    ids = np.concatenate([ids, np.full((rows, slack), _PAD_ID, dtype=ids.dtype)], axis=1)
  packed_ids = jnp.asarray(ids, jnp.int32)
  # Segment ids are 1-based; 0 is the padding bucket, and `num_segments` counts it.
  numbering = np.ones(content_len, np.int64) if leak else np.repeat(np.arange(1, segments_per_row + 1), seq_len)
  seg_ids = jnp.asarray(np.broadcast_to(_pad(numbering, 0), (rows, packed_len)), jnp.int32)
  # Positions restart at every boundary -- otherwise RoPE would place segment 2 as though it
  # continued segment 1, which is a distinct bug from attention crossing the boundary.
  seg_pos = jnp.asarray(
      np.broadcast_to(_pad(np.tile(np.arange(seq_len), segments_per_row), 0), (rows, packed_len)), jnp.int32
  )

  # Per sequence, not per row: with ragged `valid_len` the segments of one row are masked to
  # different lengths, so this cannot be tiled from a single segment the way the ids can.
  lens = _as_lens(valid_len, batch)
  within = np.arange(seq_len)[None, :]
  per_seq_mask = ((within >= prompt_len) & (within < prompt_len + lens[:, None])).astype(np.float32)
  mask_rows = per_seq_mask.reshape(rows, content_len)
  if slack:
    mask_rows = np.concatenate([mask_rows, np.zeros((rows, slack), np.float32)], axis=1)
  mask = jnp.asarray(mask_rows, jnp.float32)

  graphdef, state = nnx.split(model)
  ref_logps = tunix_common.compute_per_token_logps(
      graphdef,
      state,
      prompt_tokens=jnp.zeros((rows, 0), jnp.int32),
      completion_tokens=packed_ids,
      pad_id=_PAD_ID,
      eos_id=_EOS_ID,
      stop_gradient=True,
      temperature=algo_config.temperature,
      segment_ids=seg_ids,
      segment_positions=seg_pos,
  )
  if isinstance(ref_logps, tuple):
    ref_logps = ref_logps[0]

  per_token_adv = np.repeat(advantages.reshape(rows, segments_per_row), seq_len, axis=1)
  if slack:
    per_token_adv = np.concatenate([per_token_adv, np.zeros((rows, slack))], axis=1)

  return tunix_common.TrainExample(
      prompt_ids=jnp.zeros((rows, 0), jnp.int32),
      prompt_mask=jnp.zeros((rows, 0), jnp.int32),
      completion_ids=packed_ids,
      completion_mask=mask,
      advantages=jnp.asarray(per_token_adv, jnp.float32),
      ref_per_token_logps=ref_logps,
      old_per_token_logps=None,
      segment_ids=seg_ids,
      segment_positions=seg_pos,
      num_segments=segments_per_row + 1,
  )


def _shard_example(example, mesh):
  """Places every leaf batch-sharded over `fsdp`, once, before any trainer sees it.

  Both trainers require it and each shards differently: MaxText bakes `P('fsdp', None)` into
  the compiled kernel's `in_shardings` and raises if the argument arrives replicated, while
  Tunix re-shards inside `_prepare_payload`. Sharding up front means the reference gradient
  and both trainers read byte-identical, identically placed inputs, so no result below is a
  placement artifact.
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


def _engine_grads(engine):
  """MaxText's accumulated gradients as a total, not a per-replica partial sum.

  While the data-parallel all-reduce is deferred the accumulator is tagged `unreduced`;
  `_reduced_accumulated_grads` (`maxtext_engine.py:1782-1795`) runs the reduction `update()`
  still owes, so what comes back is the total the pending step will apply.

  Reading `_accumulated_grads` raw would not report a fraction -- it reports nothing, every op
  here rejecting it. Measured: `div` and `_rel_delta`'s `sub`/`abs` raise
  `NotImplementedError: unreduced rule for ... is not implemented`, `mul` raises
  `ShardingTypeError` for being bilinear, and `jnp.sum` traces but fails to materialize. So the
  exposure was a crash on the first gradient read, never a wrong number.

  TODO(mazumdera): exercise this under explicit sharding. It is a no-op in every run to date,
  because `_plain_grad_shardings` is `None` unless `shard_mode: explicit` on an all-Explicit
  pure-data-parallel mesh (see `_deferred_all_reduce_shardings`), and the accessor then returns
  the raw value unchanged. It starts mattering the first time this harness runs on such a mesh.
  """
  return engine._reduced_accumulated_grads()  # pylint: disable=protected-access


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

  `functools.wraps` is required, not cosmetic -- `nnx.value_and_grad` resolves kwargs to
  positions off the wrapped signature and rejects a bare `**kwargs` function.
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


def _train_to_final_params(cfg, mesh, examples, algo_config, steps, *, compile_engine=True):
  """Runs `steps` real optimizer steps. Returns the step-0 gradient, final params and losses.

  Deliberately not `run_maxtext`: that function measures *one* step in isolation. This one runs
  a window, then keeps going, because the two arms answer different questions.

  `grads0` is read at the initial weights, before any `update()`, so the arms differentiate the
  same function at the same point and cannot have diverged for any other reason. A difference
  there is a packing difference.

  `final` is the weights after `steps` updates, and it is a much blunter instrument. Adam's
  first update is essentially `lr * sign(g)`, so a gradient component near zero that flips sign
  between the arms moves that weight by a full `2 * lr` no matter how tiny the underlying
  difference was, and from step 1 on the two arms are simply on different trajectories. Read it
  for whether the arms stay together, not for how far apart they are.
  """
  engine = maxtext_engine.MaxTextTrainingEngine(cfg, mesh=mesh, wrap_with_tunix_adapter=True, tokenizer_pad_id=_PAD_ID)
  engine.with_loss_fn(algo_core.grpo_loss_fn, has_aux=True).with_gen_model_input_fn(
      lambda p: _model_inputs(p, algo_config)
  )
  if compile_engine:
    engine.compile(examples[0])
  initial = jax.tree.map(jnp.copy, _params(engine.model))
  losses, grads0 = [], None
  for step in range(steps):
    for ex in examples:
      engine.fwd_bwd(ex)
    if step == 0:
      # Unlike Tunix's accumulator, MaxText's holds the raw sum; `run_maxtext` divides too.
      denom = jnp.asarray(engine._accumulated_denominator, jnp.float32)  # pylint: disable=protected-access
      grads0 = jax.tree.map(
          lambda g, d=denom: jnp.copy(jnp.asarray(g, jnp.float32) / d),
          nnx.to_pure_dict(_engine_grads(engine)),
      )
    engine.update()
    wm = engine.get_metrics(clear_cache=True).weighted_metrics["loss"]
    losses.append(float(jnp.sum(jnp.asarray(wm.unreduced_sum)) / jnp.sum(jnp.asarray(wm.denominator))))
  return {
      "final": jax.tree.map(jnp.copy, _params(engine.model)),
      "initial": initial,
      "losses": losses,
      "grads0": grads0,
  }


def _tunix_train_to_final_params(cfg, mesh, examples, algo_config, steps, *, with_axis_rules=True):
  """`_train_to_final_params` against Tunix `PeftTrainer` instead of the MaxText engine.

  Gives `--compare-layouts` the same packed-vs-unpacked weight comparison on Tunix's trainer
  that it runs on MaxText's. Tunix's own `test_sequence_packing` cannot make that comparison:
  its toy model ignores segment ids in the attention mask, which is why that test carries a
  `TODO` and runs at `atol=5e-2, rtol=1e-1`. Here the model is real and honours them.

  The losses lag one step behind MaxText's: `_write_train_metrics` skips the first step, so
  `_tunix_metrics` reads the previous step's buffer. They are provenance either way -- the
  negative control shows the loss agreeing to seven decimals while the weights are 93% apart.
  """
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
    # Same reason as `run_tunix`: re-place once through Tunix's helper so `_prepare_payload`
    # does not host-round-trip the payload on every micro-step.
    examples = [sharding_utils.shard_input(ex, tcfg.data_sharding_axis) for ex in examples]
    initial = jax.tree.map(jnp.copy, _params(model))
    losses, grads0 = [], None
    for step in range(steps):
      for ex in examples:
        trainer.fwd_bwd(ex)
      if step == 0:
        # `get()` already divides by the accumulated denominator, unlike MaxText's.
        grads0 = jax.tree.map(jnp.copy, nnx.to_pure_dict(trainer.grad_accumulator.get()))
      trainer.update()
      losses.append(_tunix_metrics(trainer)[0])
    return {
        "final": jax.tree.map(jnp.copy, _params(model)),
        "initial": initial,
        "losses": losses,
        "grads0": grads0,
    }


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
      nnx.to_pure_dict(_engine_grads(engine)),
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
    #
    # Raw, deliberately, where the two reads above take `_engine_grads`: this is a timing
    # barrier, not a measurement. Reducing here would move the all-reduce that `update()` owes
    # into the window being timed and bill it to `fwd_bwd`. Nothing reads a value, so the
    # unreduced tag never has to survive an op. Do not "fix" this one to match.
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

# Untimed, untraced updates run before the measurement window opens.
_WARMUP = 2


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


def _time_loop(fwd_bwd, update, examples, iters, leaves_fn, warmup=_WARMUP):
  """Median wall clock of one full update, plus the peak HBM the whole run touched.

  The warmup runs outside `_maybe_trace`. It is not just noise reduction: MaxText's update
  kernel is donated, and its first post-donation invocation re-lowers against the new
  parameter layout, so tracing from a cold loop drops a ~570 ms `CompileAndLoad` in the
  middle of the window and makes the trace unreadable next to a warm one.
  """
  fwd_times, update_times = [], []
  for _ in range(warmup):
    for ex in examples:
      fwd_bwd(ex)
    update()
  jax.block_until_ready(leaves_fn())
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


# The budget sweep from tunix's `grpo_learner_test.py::test_sequence_packing`, expressed as
# multiples of one sequence instead of the absolute token counts tunix hardcodes (266 / 300 /
# 532 / 1000, against its own 256+10 sequence). Keeping the *shape* of the sweep rather than the
# numbers is what makes it portable to a different sequence length: what the four cases test is
# one-vs-many sequences per row, crossed with exactly-full vs padded.
_LAYOUT_BUDGETS = (
    ("single_sequence", 1, 1.0),  # tunix 266 -- exactly one sequence, no slack
    ("single_sequence_with_padding", 1, 300 / 266),  # tunix 300
    ("multiple_sequences", 2, 1.0),  # tunix 532 -- exactly two
    ("large_budget", 3, 1000 / (3 * 266)),  # tunix 1000 -- three fit, remainder is slack
)


def _round_up(n, multiple):
  return n if not multiple else -(-n // multiple) * multiple


def compare_layouts(args, mesh, algo_config, *, prompt_len, completion_len, valid):
  """7d: trains one trainer on packed and unpacked layouts of identical data, diffs step-0 grads.

  This is a different question from what `--packed` asks. `--packed` gives both trainers the
  same packed examples and compares MaxText against Tunix, which cannot see a packing defect
  they share -- they share `TunixMaxTextAdapter`. Here there is one trainer and two layouts of
  byte-identical data, so the only thing that differs is packing itself.

  `valid` is a scored-token count per sequence, spent as one row's completion mask when
  unpacked and one segment's when packed. Unequal counts are the point: they stop
  `sequence-mean-token-mean` collapsing into `token-mean`.

  Tunix runs its version at `atol=5e-2, rtol=1e-1` with a TODO explaining why: its toy model
  ignores segment ids in the attention mask, so its two arms are not actually computing the
  same thing. MaxText's attention honours them, so this asserts a real tolerance.
  """
  seq_len = prompt_len + completion_len
  # Round each budget up to `--layout-align`. TPU splash attention rejects a key-block size that
  # is not a multiple of 128, and tunix's ratios land on widths like 144 that are not -- but a
  # real packer's token budget is block-aligned anyway, so aligning is closer to production than
  # tunix's raw numbers are. What the sweep tests survives it: rounding only ever *adds* slack,
  # and the exactly-full cases keep zero slack as long as `seq_len` is aligned too.
  align = args.layout_align
  if align and seq_len % align:
    raise SystemExit(
        f"--seq {args.seq} gives a {seq_len}-token sequence, which is not a multiple of "
        f"--layout-align {align}; the exactly-full budgets would silently acquire slack. Use a "
        f"multiple of {2 * align} for --seq, or --layout-align 0 with a non-splash attention kernel."
    )
  budgets = [(name, spr, _round_up(int(round(spr * seq_len * ratio)), align)) for name, spr, ratio in _LAYOUT_BUDGETS]
  # Each budget's row count is `batch // spr`, and `_arm` derives `per_device_batch_size` from
  # it by floor division -- so a row count that does not divide by the device count does not
  # fail, it silently rounds the arm down to a smaller global batch than the one requested and
  # the layouts stop being comparable. Require the product up front instead.
  devices = jax.device_count()
  every = [spr * devices for _, spr, _ in budgets]
  if any(args.batch % n for n in every):
    need = math.lcm(*every)
    raise SystemExit(
        f"--batch {args.batch} must be divisible by segments-per-row x {devices} devices for every "
        f"budget ({sorted(set(every))}); use {need} or a multiple."
    )

  widest = max(width for _, _, width in budgets)
  base = functools.partial(_config, gradient_accumulation_steps=args.ga)
  # The sweep runs ragged; the negative control is the one arm that still requires uniform
  # lengths (see below).
  control_valid = [completion_len] * len(valid)
  # Whether the sweep's unpacked arm can serve as the control's baseline: the test is equality with
  # the counts the control runs on, NOT "each micro-batch internally uniform" -- those differ,
  # and the gap is reachable: `--ragged --uniform-segments --ga 4` gives e.g. [64, 16, 40, 8],
  # each internally uniform but none matching the control's [64, 64, 64, 64], so the control's
  # delta would be the attention leak plus a data difference, labelled "shared".
  shares_control_baseline = all(
      np.asarray(v).reshape(-1).tolist() == [completion_len] * np.asarray(v).size for v in valid
  )

  def _arm(build, rows, row_len, lens):
    cfg = base(
        micro_batch_size_to_train_on=rows,
        per_device_batch_size=max(1, rows // jax.device_count()),
        max_target_length=max(64, row_len),
    )
    ref_model = _build_model(cfg, mesh)
    examples = [
        _shard_example(
            build(
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
        for i, v in enumerate(lens)
    ]
    del ref_model
    if args.trainer == "maxtext":
      return _train_to_final_params(
          cfg, mesh, examples, algo_config, args.layout_steps, compile_engine=not args.no_compile
      )
    return _tunix_train_to_final_params(
        cfg, mesh, examples, algo_config, args.layout_steps, with_axis_rules=args.trainer == "tunix"
    )

  # The unpacked arm does not depend on the budget, so it is trained once and reused. Its
  # `max_target_length` is the widest packed row rather than its own, since that config value
  # is the one thing the two arms would otherwise not share.
  with mesh:
    base_arm = _arm(_train_example, args.batch, max(widest, seq_len), valid)
    unpacked, unpacked_grads = base_arm["final"], base_arm["grads0"]
    moved = _rel_delta(unpacked, base_arm.pop("initial"))
    if moved["rel_l2"] < 1e-9:
      raise SystemExit(
          f"training moved the weights by rel_l2={moved['rel_l2']:.3e} -- every layout comparison "
          "below would pass vacuously. Check the learning rate and warmup."
      )

    out = {
        "mode": "compare_layouts",
        "trainer": args.trainer,
        "layout_steps": args.layout_steps,
        "layout_align": align,
        "seq_len": seq_len,
        "batch": args.batch,
        "ga": args.ga,
        "loss_agg_mode": algo_config.loss_agg_mode,
        "segment_valid_lens": [np.asarray(v).reshape(-1).tolist() for v in valid],
        "uniform_segments": all(len(set(np.asarray(v).reshape(-1).tolist())) == 1 for v in valid),
        "unpacked_weights_moved": moved,
        "unpacked_losses": [round(x, 9) for x in base_arm["losses"]],
        "budgets": {},
    }
    for name, spr, width in budgets:
      arm = _arm(
          functools.partial(_packed_train_example, segments_per_row=spr, pad_to=width),
          args.batch // spr,
          width,
          valid,
      )
      out["budgets"][name] = {
          "segments_per_row": spr,
          "row_len": width,
          "slack_tokens": width - spr * seq_len,
          "rows": args.batch // spr,
          "losses": [round(x, 9) for x in arm["losses"]],
          "grads0_vs_unpacked": _rel_delta(arm["grads0"], unpacked_grads),
          "final_weights_vs_unpacked": _rel_delta(arm["final"], unpacked),
      }
      del arm

    # The negative control. Re-runs the first multi-segment budget with every real token in one
    # segment, so attention crosses the boundaries -- the defect the four arms above are meant
    # to be able to see. Its delta is the scale the others are read against: they are only
    # "small" relative to something known-broken measured the same way, and if it lands among
    # them then this mode is blind and its verdict is worthless whatever the numbers say.
    #
    # Runs UNIFORM even when the budgets above ran ragged, hence its own baseline: collapsing
    # ragged segments would change the segmented reduction as well as the attention mask, and a
    # control that breaks two things at once establishes neither.
    control_spr, control_width = next(((s, w) for _, s, w in budgets if s > 1), (0, 0))
    if control_spr:
      control_base = (
          base_arm if shares_control_baseline else _arm(_train_example, args.batch, max(widest, seq_len), control_valid)
      )
      leaked = _arm(
          functools.partial(_packed_train_example, segments_per_row=control_spr, pad_to=control_width, leak=True),
          args.batch // control_spr,
          control_width,
          control_valid,
      )
      out["negative_control"] = {
          "segments_per_row": control_spr,
          "row_len": control_width,
          "valid_len": completion_len,
          "baseline": "shared" if shares_control_baseline else "own_uniform_unpacked_arm",
          "losses": [round(x, 9) for x in leaked["losses"]],
          "grads0_vs_unpacked": _rel_delta(leaked["grads0"], control_base["grads0"]),
          "final_weights_vs_unpacked": _rel_delta(leaked["final"], control_base["final"]),
      }
      del leaked, control_base
      # Both metrics have to separate broken from correct, and they fail differently if they do
      # not: `grads0` would mean packing genuinely changes nothing measurable, `final_weights`
      # that Adam has buried the difference under its own sign noise.
      for metric, floor in (("grads0_vs_unpacked", 3.0), ("final_weights_vs_unpacked", 3.0)):
        worst = max(b[metric]["rel_l2"] for b in out["budgets"].values())
        margin = out["negative_control"][metric]["rel_l2"] / (worst + 1e-30)
        out["negative_control"][f"margin.{metric}"] = margin
        if margin < floor:
          raise SystemExit(
              f"negative control is only {margin:.2f}x the worst real budget on {metric} "
              f"({out['negative_control'][metric]['rel_l2']:.3e} vs {worst:.3e}) -- that metric "
              "cannot see attention leaking across segment boundaries, so it cannot support a "
              "verdict on packing."
          )
  return out


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
  ap.add_argument(
      "--seq",
      type=int,
      default=16,
      help="tokens per example, split evenly between prompt and completion. The default is "
      "deliberately tiny so the numerics run stays cheap; raise it to compare step time at "
      "a size where the matmuls, not the dispatch overhead, dominate.",
  )
  ap.add_argument("--iters", type=int, default=5)
  ap.add_argument("--ragged", action="store_true", help="unequal valid-token counts per micro-batch")
  ap.add_argument(
      "--uniform-segments",
      action="store_true",
      help="give every sequence in a micro-batch the same valid-token count. Off by default: "
      "uniform counts are the one regime in which sequence-mean-token-mean equals token-mean, "
      "so a uniform packed run cannot see the segmented reduction at all. Kept for reproducing "
      "runs recorded before this was fixed, and for the --compare-layouts negative control.",
  )
  ap.add_argument(
      "--packed",
      action="store_true",
      help="pack --segments-per-row sequences into each row, with segment_ids/segment_positions/"
      "num_segments set. Token count is unchanged; the row count drops by that factor. Both "
      "trainers see the same packed examples, so this compares packing *composition* -- segmented "
      "aggregation, the summed denominator, accumulation -- between them. It does NOT prove the "
      "attention mask is honoured: both arms share TunixMaxTextAdapter, so a gate regression would "
      "corrupt them identically and this comparison would still agree. "
      "maxtext_engine_packing_test.py is the oracle for that.",
  )
  ap.add_argument("--segments-per-row", type=int, default=2)
  ap.add_argument(
      "--compare-layouts",
      action="store_true",
      help="7d: train one trainer on packed and unpacked layouts of identical data and diff the "
      "STEP-0 GRADIENTS, over tunix's four packing budgets (one/many sequences per row, crossed "
      "with exactly-full/padded). Final weights are reported too but only corroborate: Adam buries "
      "the difference under its own sign noise, scoring 6.4x separation against a known-broken arm "
      "where the gradient scores 44.8x. Orthogonal to --packed, which fixes the layout and varies "
      "the trainer; this fixes the trainer and varies the layout, which is the only arrangement "
      "where a defect shared by both trainers is visible. Runs under --trainer maxtext or tunix: "
      "tunix's own test_sequence_packing cannot substitute, its toy model ignoring segment ids in "
      "the attention mask. Ignores --packed, --segments-per-row and --iters.",
  )
  ap.add_argument(
      "--layout-steps",
      type=int,
      default=3,
      help="optimizer steps per arm under --compare-layouts. The step-0 gradient does not depend "
      "on this; it buys the corroborating final-weight comparison somewhere to accumulate. 1 is a "
      "valid and much cheaper run if the gradient verdict is all that is wanted.",
  )
  ap.add_argument(
      "--layout-align",
      type=int,
      default=128,
      help="round every --compare-layouts row width up to a multiple of this. TPU splash attention "
      "requires 128; 0 disables it, which only works on a kernel that accepts arbitrary lengths.",
  )
  ap.add_argument(
      "--loss-agg-mode",
      default=None,
      help="default token-mean, or sequence-mean-token-mean under --packed. token-mean is "
      "segment-agnostic by construction -- its segmented branch reduces to the same expression as "
      "the row branch -- so a packed run under it would agree even with segmentation entirely "
      "broken. Overriding it back to token-mean is legitimate to request, and a bad default.",
  )
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

  prompt_len = completion_len = args.seq // 2
  if args.ragged and args.ga > 1:
    valid = [round(completion_len * f) for f in (1.0, 0.25, 0.625, 0.125, 1.0, 0.375, 0.875, 0.5)][: args.ga]
  else:
    valid = [completion_len] * args.ga
  # `--ragged` sets each micro-batch's ceiling; this spreads the sequences underneath it, so a
  # packed row holds unequal segments and the segmented reduction stops restating the row
  # branch. Only the packed modes need it: an unpacked run reduces per row regardless.
  if (args.packed or args.compare_layouts) and not args.uniform_segments:
    valid = [_segment_valid_lens(v, args.batch, i) for i, v in enumerate(valid)]
    # Every micro-batch, not just the first: under `--ragged` the first carries the 1.0
    # fraction and so is likeliest to survive rounding, while a later one scaled to 0.125 of it
    # can collapse to all-ones unnoticed. `--seq 16 --ragged --ga 4` is exactly that.
    collapsed = [i for i, v in enumerate(valid) if len(set(v.tolist())) == 1]
    if collapsed:
      raise SystemExit(
          f"--seq {args.seq} rounds every segment of micro-batch(es) {collapsed} to the same "
          f"length ({[int(valid[i][0]) for i in collapsed]}), which makes sequence-mean-token-mean "
          "identical to token-mean and leaves the segmented reduction untested there. Raise --seq "
          f"(needs {'16' if not args.ragged else 'enough that the smallest --ragged ceiling still spreads'}), "
          "or pass --uniform-segments to mean it."
      )

  if args.compare_layouts:
    if args.trainer not in ("maxtext", "tunix", "tunix_no_axis_rules"):
      raise SystemExit(f"--compare-layouts needs a trainer that trains; got --trainer {args.trainer}.")
    probe = _config()
    mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(probe), probe.mesh_axes)
    algo_config = _GrpoConfig(loss_agg_mode=args.loss_agg_mode or "sequence-mean-token-mean")
    _emit(
        compare_layouts(args, mesh, algo_config, prompt_len=prompt_len, completion_len=completion_len, valid=valid),
        args.out,
    )
    return

  # Packing keeps the token count and shrinks the row count, so the model sees the same work
  # laid out differently -- which is the only way the two runs are comparable.
  rows = args.batch // args.segments_per_row if args.packed else args.batch
  row_len = args.seq * args.segments_per_row if args.packed else args.seq
  if args.packed and rows < 1:
    raise SystemExit(f"--batch {args.batch} is too small to pack {args.segments_per_row} per row.")

  cfg = _config(
      gradient_accumulation_steps=args.ga,
      micro_batch_size_to_train_on=rows,
      per_device_batch_size=max(1, rows // jax.device_count()),
      max_target_length=max(64, row_len),
  )
  mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
  algo_config = _GrpoConfig(
      loss_agg_mode=args.loss_agg_mode or ("sequence-mean-token-mean" if args.packed else _GrpoConfig.loss_agg_mode)
  )

  # One model builds the shared micro-batches so every trainer sees byte-identical inputs.
  ref_model = _build_model(cfg, mesh)
  build = (
      functools.partial(_packed_train_example, segments_per_row=args.segments_per_row) if args.packed else _train_example
  )
  examples = [
      _shard_example(
          build(
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
      "seq": args.seq,
      "valid_lens": [np.asarray(v).reshape(-1).tolist() for v in valid],
      "uniform_segments": bool(args.uniform_segments),
      "devices": jax.device_count(),
      "packed": args.packed,
      "segments_per_row": args.segments_per_row if args.packed else None,
      "rows": rows,
      "row_len": row_len,
      "loss_agg_mode": algo_config.loss_agg_mode,
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
      _XPROF["tag"] = f"{name}_ga{args.ga}_bs{args.batch}_seq{args.seq}"
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
