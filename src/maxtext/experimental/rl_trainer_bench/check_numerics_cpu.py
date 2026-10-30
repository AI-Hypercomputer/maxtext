#!/usr/bin/env python3
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

"""CPU check of the numerics knobs of bench_driver.py (no TPU, no cluster): why gradient_norm was 0, and the fix.

Builds the TRAINER_WARMUP_COMPILE payload with Tunix's own builder from the e2e orchestrator argv of a DRY_RUN
render (--launch-render; exactly as the driver does: `run_deepswe_dist._parse_args`, `_build_algo`,
`BatchConfig`, routed_experts left out to keep it light), then the realistic payload, then runs Tunix's GRPO
loss (`algo.loss_fn()`, the function `configure_worker` hands the trainer) on it with a one-parameter stand-in for
the model: `compute_per_token_logps` is replaced by `theta + base_logp` per token, so everything after the forward
pass (gates, masks, aggregation, aux) is the real code and d loss / d theta is the gradient signal.

Checks, per configuration:
  1. advantages of the warm-up and realistic payloads (ADVANTAGES=inherit|random);
  2. gates on (production argv), trainer logps far from the payload's synthetic rollout logps (-0.7): every
     sequence dropped, loss denominator 0, gradient 0 (the engine's `_update_kernel` turns a 0 denominator into
     zero gradients, hence gradient_norm=0 with step_skipped=0);
  3. gates on, trainer logps equal to the rollout logps: sequences kept, gradient non-zero (the gate is the cause);
  4. LOGP_GATES=off (`strip_logp_gates`), far logps: sequences kept, gradient non-zero;
  5. `summarize_metrics` / `metric_log_lines` on a hand-built MaxText MetricsBuffer with a zero-denominator entry.
With --fsdp 64 it also reports the packed rows per micro-batch of the 256-chip variant.

Run with a Tunix and MaxText that match the e2e run, e.g. inside the trainer image:
  docker run --rm --entrypoint python -v $PWD:/tb:ro -v <dir of render>:/r:ro -e JAX_PLATFORMS=cpu <image> \
    /tb/check_numerics_cpu.py --launch-render /r/<launch-dryrun>.yaml
"""

# The JAX, Tunix and MaxText imports sit inside the functions that need them, so the CPU tests and --help run
# without a TPU runtime or the Tunix package installed.
# pylint: disable=import-outside-toplevel

import argparse
import dataclasses
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench_driver as d  # pylint: disable=g-import-not-at-top


def orch_argv_from_render(path, fsdp=None):
  from render_bench import _proc_main, load_documents, select_jobsets, split_command  # pylint: disable=g-import-not-at-top

  _, orch = select_jobsets(load_documents(path))
  _, flags = split_command(_proc_main(orch)[1]["command"][-1])
  if fsdp is not None:
    flags = [t for t in flags if not t.startswith("--trainer_fsdp=")] + [f"--trainer_fsdp={fsdp}"]
  return flags


def build(orch_argv):
  """Builds the trainer config, algo config and warmup payload from an orchestrator argv, as the e2e launch does."""
  from tunix.experimental.examples.deepswe_dist import run_deepswe_dist  # pylint: disable=g-import-not-at-top
  from tunix.experimental.examples.deepswe_dist import trainer_warmup  # pylint: disable=g-import-not-at-top
  from tunix.experimental.orchestrator import batch_assembly  # pylint: disable=g-import-not-at-top

  o = run_deepswe_dist._parse_args(orch_argv)  # pylint: disable=protected-access
  algo = run_deepswe_dist._build_algo(o)  # pylint: disable=protected-access
  bc = batch_assembly.BatchConfig(
      pad_id=0,
      max_prompt_length=o.max_prompt_length,
      max_response_length=o.max_response_length,
      max_seq_token_per_tpu=o.max_seq_token_per_tpu,
      max_segments_per_packed_row=o.max_segments_per_packed_row,
      trainer_fsdp=o.trainer_fsdp,
      trainer_dp=o.trainer_dp,
      trainer_expert=o.trainer_expert,
  )
  warm = trainer_warmup.build_warmup_payload(
      algo=algo,
      batch_config=bc,
      num_generations=o.num_generations,
      mini_batch_size=o.mini_batch_size,
      train_micro_batch_size=algo.train_micro_batch_size,
      routed_experts_shape=None,
  )
  return o, algo, warm


def adv_stats(p):
  a = np.asarray(p.advantages)[np.asarray(p.completion_mask) > 0]
  return {
      "scored": int(a.size),
      "nonzero_frac": round(float((a != 0).mean()), 4),
      "uniq": np.unique(a)[:6].tolist() if np.unique(a).size <= 6 else f"{np.unique(a).size} distinct",
      "mean": round(float(a.mean()), 5),
      "std": round(float(a.std()), 5),
  }


def run_loss(algo, payload, base_logp):
  """(loss value, loss denominator, d loss/d theta, kept_frac) of Tunix's GRPO loss with logps = theta + base."""
  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  from flax import nnx  # pylint: disable=g-import-not-at-top
  from tunix.rl import algo_core  # pylint: disable=g-import-not-at-top

  class Stand(nnx.Module):

    def __init__(self):
      self.theta = nnx.Param(jnp.zeros((), jnp.float32))

  def fake_logps(graphdef, state, *, completion_tokens, return_moe_audit=False, **_):
    theta = nnx.merge(graphdef, state).theta.value
    lp = theta + jnp.full(completion_tokens.shape, base_logp, jnp.float32)
    out = (lp, jnp.zeros_like(lp), jnp.array(False))
    return out + (None,) if return_moe_audit else out

  orig = algo_core.common.compute_per_token_logps
  algo_core.common.compute_per_token_logps = fake_logps
  try:
    ex = jax.tree.map(lambda x: jnp.asarray(x) if isinstance(x, np.ndarray) else x, payload)
    gin = algo.build_gen_model_input_fn(pad_id=0, eos_id=0)(ex)
    loss_fn = algo.loss_fn()
    gdef, st = nnx.split(Stand())

    def f(s):
      out = loss_fn(nnx.merge(gdef, s), **gin)
      return out.primary_loss.unreduced_sum, out

    (_, out), g = jax.value_and_grad(f, has_aux=True)(st)
  finally:
    algo_core.common.compute_per_token_logps = orig
  kept = out.aux_metrics.get("sample_mask/kept_frac")
  return (
      float(out.primary_loss.compute()),
      float(out.primary_loss.denominator),
      float(g["theta"].value),
      float(kept.compute()) if kept is not None else None,
  )


def check_summary():
  """Checks summarize_metrics and the STEP_LOSS/STEP_METRICS lines on a hand-built metrics buffer."""
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  from maxtext.training_engine import abstract_engine  # pylint: disable=g-import-not-at-top

  wm = abstract_engine.WeightedMetric
  buf = abstract_engine.MetricsBuffer(
      id=0,
      weighted_metrics={
          "loss": wm(jnp.array([2.0, 0.0, -3.0]), jnp.array([4.0, 0.0, 6.0])),
          "sample_mask/kept_frac": wm(jnp.array([1.0, 0.0, 2.0]), jnp.array([1.0, 0.0, 2.0]), min_denom=1.0),
      },
      scalar_metrics={"gradient_norm": jnp.array([0.125]), "is_ratio/mean": jnp.array([1.0, 1.0, 1.0])},
  )
  s = d.summarize_metrics(buf)
  assert abs(s["loss"] - (-1.0 / 10.0)) < 1e-12, s
  assert s["micro_denominator"] == [4.0, 0.0, 6.0] and np.isnan(s["micro_loss"][1]), s
  assert abs(s["micro_loss"][0] - 0.5) < 1e-12 and abs(s["micro_loss"][2] + 0.5) < 1e-12, s
  assert s["scalar"]["gradient_norm"] == 0.125 and s["scalar"]["is_ratio/mean"] == [1.0, 1.0, 1.0], s
  for line in d.metric_log_lines(0, s):
    print(line)
  return True


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--launch-render", required=True, help="DRY_RUN output of the e2e launcher (see README)")
  ap.add_argument("--fsdp", type=int, default=None, help="override --trainer_fsdp (64 for the 256-chip variant)")
  ap.add_argument("--rows", type=int, default=4, help="rows of the micro-batch fed to the loss (CPU time)")
  ap.add_argument(
      "--far-logp",
      type=float,
      default=-12.0,
      help="stand-in trainer logp per token; ~ -ln(vocab) for a near-uniform random-init model",
  )
  a = ap.parse_args()

  argv = orch_argv_from_render(a.launch_render, a.fsdp)
  gates = [t for t in argv if t.split("=", 1)[0] in d.LOGP_GATE_FLAGS]
  print("orch gate flags:", gates)
  o, algo_on, warm = build(argv)
  rows, budget = np.asarray(warm.completion_ids).shape
  print(
      f"trainer_fsdp={o.trainer_fsdp} trainer_expert={o.trainer_expert}: warm-up micro-batch rows={rows} "
      f"budget={budget} segments/row={[len(np.unique(r[r > 0])) for r in np.asarray(warm.segment_ids)[:4]]}"
  )
  print("warm-up advantages:", adv_stats(warm))
  print(
      "warm-up overlong:",
      None if warm.overlong is None else np.unique(np.asarray(warm.overlong)).tolist(),
      "| old_per_token_logps is None:",
      warm.old_per_token_logps is None,
      "| rollout_per_token_logps unique:",
      np.unique(np.asarray(warm.rollout_per_token_logps)).tolist()[:4],
  )

  kw = {"num_experts": 512, "ep": 8, "pad_id": 0, "vocab_size": 248_000, "pad_fraction": 0.07, "seed": 0}
  inherited = d.realistic_payload(warm, **kw)
  rnd = d.realistic_payload(warm, advantages="random", **kw)
  for f in ("completion_ids", "segment_ids", "segment_positions", "completion_mask", "rollout_per_token_logps"):
    assert np.array_equal(np.asarray(getattr(inherited, f)), np.asarray(getattr(rnd, f))), f
  print("realistic ADVANTAGES=inherit:", adv_stats(inherited))
  print("realistic ADVANTAGES=random: ", adv_stats(rnd), "(ids/segments/masks/rollout logps identical to inherit)")
  print(
      "realistic rollout_per_token_logps on scored tokens:",
      np.unique(np.asarray(rnd.rollout_per_token_logps)[np.asarray(rnd.completion_mask) > 0]).tolist(),
  )

  _, algo_off, _ = build(d.strip_logp_gates(argv))
  print(
      "gates off: algo_config"
      f" seq_logprob_error_threshold={algo_off.algo_config.seq_logprob_error_threshold}"
      f" truncated_importance_sampling_type={algo_off.algo_config.truncated_importance_sampling_type}"
  )

  small = dataclasses.replace(
      rnd,
      **{
          f.name: np.asarray(getattr(rnd, f.name))[: a.rows]
          for f in dataclasses.fields(rnd)
          if isinstance(getattr(rnd, f.name), np.ndarray) and np.asarray(getattr(rnd, f.name)).shape[:1] == (rows,)
      },
  )
  results = {}
  for label, algo, base in (
      (f"gates on, trainer logp {a.far_logp:.1f}", algo_on, a.far_logp),
      ("gates on, trainer logp -0.7 (= rollout)", algo_on, -0.7),
      (f"gates off, trainer logp {a.far_logp:.1f}", algo_off, a.far_logp),
  ):
    loss, den, grad, kept = run_loss(algo, small, base)
    results[label] = (loss, den, grad, kept)
    print(
        f"GRPO loss [{label}] rows={a.rows}: loss={loss:.6g} denominator={den:.6g} dloss/dtheta={grad:.6g} "
        f"kept_frac={kept}"
    )
  vals = list(results.values())
  on_far, on_eq, off_far = vals[0], vals[1], vals[2]
  assert on_far[1] == 0.0 and on_far[2] == 0.0, on_far
  assert on_eq[1] > 0 and on_eq[2] != 0.0, on_eq
  assert off_far[1] > 0 and off_far[2] != 0.0, off_far
  check_summary()
  print("NUMERICS_CPU_CHECK_PASSED")


if __name__ == "__main__":
  main()
