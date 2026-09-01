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

"""Third arm: MaxText's qwen3-0.6b driven by `MaxTextTrainingEngine`, not by PeftTrainer.

`qwen3_maxtext_profile.py` held the trainer fixed and varied the model. This varies the
trainer and holds the model fixed, so the three runs together separate the two costs:

  tunix model + PeftTrainer  ->  qwen3_tunix_profile.py
  MaxText model + PeftTrainer ->  qwen3_maxtext_profile.py   (isolates the model)
  MaxText model + engine      ->  this file                  (isolates the trainer)

The model config here is byte-identical to `qwen3_maxtext_profile.py`'s, and the loss is
literally tunix's own `peft_trainer_v2._default_loss_fn` -- the engine accepts it
unchanged because it returns a `tunix.sft.utils.LossOutput`, which is the same class
object `abstract_engine` re-exports. So the arithmetic is the same on both sides and the
difference is the trainer.

Optimizer parity took four config overrides. The engine builds its optimizer from the
MaxText config via `train_utils.create_training_optimizer`, and base.yml's defaults are
nothing like `optax.sgd(constant_schedule(1e-5))`:

  * `opt_type="sgd"`      -- base.yml defaults to adamw, which carries two extra
                             parameter-sized moment buffers through every update.
  * `gradient_clipping_threshold=0.0` -- base.yml clips at 1.0, an extra full-tree l2 norm
                             per step. tunix's bare `optax.sgd` does not clip.
  * `learning_rate=1e-5`, `warmup_steps_fraction=0.0`,
    `learning_rate_final_fraction=1.0` -- flattens MaxText's warmup+cosine into the
                             constant schedule the tunix arms use. Schedule shape does not
                             affect step cost, but it does affect what the weights do.

Two structural differences remain, and they are the point of the comparison rather than
something to equalise away:

  * PeftTrainer at `gradient_accumulation_steps=1` runs one fused jitted train step. The
    engine always splits fwd/bwd from the optimizer update into two separately jitted
    programs (`_compiled_fwd_bwd` and `_compiled_update`), so it dispatches twice per step
    and XLA never sees the two halves together.
  * That also means the engine calls `InflightThrottler.wait_for_next()` twice per step
    against the same 2-deep queue, so its effective pipelining depth is half PeftTrainer's.

`compile()` is called with a real payload. Skipping it is not a neutral choice: without
`_compile_requested`, `fwd_bwd` takes the eager path and runs `jax.value_and_grad`
op-by-op.

`--model` and `--scan` move this arm off the shared shape. The tunix arm cannot follow --
it implements one architecture -- so past qwen3-0.6b the comparison is engine against
`PeftTrainer` over the *same* MaxText model, which is the pairing that isolates the trainer
anyway. A model with few KV heads constrains `--tp`: qwen3.5-35b-a3b has 2, so `--tp 8` is
rejected by the sharding checks and `--tp 2 --scan` is the shape that runs on 8 devices.

Run from this directory -- the arms import `qwen3_common` as a sibling, and a working
directory that contains a `tunix/` checkout will shadow the installed package:

  cd tests/end_to_end/tpu/perf_parity && python qwen3_engine_profile.py
"""

import argparse
import os
import statistics
import time

from flax import nnx
import jax
from maxtext.configs import pyconfig
from maxtext.training_engine import maxtext_engine
from maxtext.utils import maxtext_utils
from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR
import qwen3_common as qc
from transformers import AutoTokenizer
from tunix.experimental.train import peft_trainer_v2


def _build_config(spec: qc.RunSpec):
  """The qwen3_maxtext_profile.py config, plus the optimizer overrides listed above.

  `gradient_accumulation_steps` stays at 1 no matter what `--ga` says. The engine reads
  no accumulation setting at all -- `_micro_step_count` counts `fwd_bwd` calls since the
  last `update()` -- so putting the real value here would only mislead base.yml's batch
  arithmetic into splitting `per_device_batch_size` a second time.
  """
  return pyconfig.initialize(
      [None, os.path.join(MAXTEXT_CONFIGS_DIR, "base.yml")],
      model_name=spec.model,
      run_name="perf_parity_qwen3_0p6b_engine",
      base_output_directory=os.path.join(os.getcwd(), "maxtext_out"),
      max_target_length=spec.seq,
      per_device_batch_size=spec.per_device_batch,
      ici_fsdp_parallelism=spec.fsdp,
      ici_tensor_parallelism=spec.tp,
      dtype="float32",
      weight_dtype="float32",
      remat_policy="none",
      scan_layers=spec.scan,
      enable_dropout=False,
      enable_checkpointing=False,
      convert_checkpoint_if_possible=False,
      skip_jax_distributed_system=True,
      use_multimodal=False,
      init_weights_seed=0,
      # Optimizer parity with `optax.inject_hyperparams(optax.sgd)(constant_schedule(1e-5))`.
      opt_type="sgd",
      learning_rate=qc.LEARNING_RATE,
      gradient_clipping_threshold=0.0,
      warmup_steps_fraction=0.0,
      learning_rate_final_fraction=1.0,
      learning_rate_schedule_steps=spec.steps,
      steps=spec.steps,
      gradient_accumulation_steps=1,
  )


def _report_nnx_graph_cost(engine) -> None:
  """Times the host-side NNX graph traversals: the two the engine still pays, and the two it
  no longer does.

  The engine used to call `nnx.split(model, nnx.Param, ...)` then `nnx.update(model, ...)` in
  `fwd_bwd`, and `nnx.split(state)` then `nnx.update(state, ...)` in `update` -- four full
  traversals of an unrolled 28-layer graph per step, none jitted and none dependent on the
  device, where PeftTrainer v2 pays for them once. Both `split`s are now served from the pure
  state the engine carries across steps, so the figures below split into what a step costs
  (the two `update`s) and what the cache saves (the two `split`s).

  The `update`s are deliberately kept. They are the publish barrier: they are what keeps
  `engine.model`, `save_checkpoint` and `prepare_weight_sync` reading the same weights the
  kernels just produced, rather than a snapshot the engine happens to be training on.

  Measured after the loop with the device idle, so what is timed is pure Python.
  """
  model = engine.model
  state = engine.state
  reps = 5

  def timeit(fn):
    fn()  # warm any caching
    start = time.perf_counter()
    for _ in range(reps):
      fn()
    return (time.perf_counter() - start) / reps * 1e3

  _, params, rest = nnx.split(model, nnx.Param, ...)
  _, state_pure = nnx.split(state)
  print(
      "  host nnx graph   : paid per step  "
      f"update(model)={timeit(lambda: nnx.update(model, rest)):.1f}ms  "
      f"update(state)={timeit(lambda: nnx.update(state, state_pure)):.1f}ms"
  )
  print(
      "                   : saved by cache  "
      f"split(model)={timeit(lambda: nnx.split(model, nnx.Param, ...)):.1f}ms  "
      f"split(state)={timeit(lambda: nnx.split(state)):.1f}ms  "
      f"[{len(jax.tree.leaves(params))} param leaves]"
  )


def main() -> None:
  spec = qc.RunSpec(qc.add_common_args(argparse.ArgumentParser()).parse_args())
  profile_dir = qc.profile_dir(spec.tag(f"{spec.model}-engine" + ("-scan" if spec.scan else "")))
  print(spec.describe(), flush=True)

  tokenizer = AutoTokenizer.from_pretrained(qc.TOKENIZER_ID)
  dataset = qc.make_dataset_for(spec)

  config = _build_config(spec)
  print(
      f"scan_layers={config.scan_layers}  attention={config.attention}  "
      f"remat={config.remat_policy}  opt={config.opt_type}  clip={config.gradient_clipping_threshold}",
      flush=True,
  )

  # Built here rather than left to the engine: `wrap_with_tunix_adapter=True` requires a
  # mesh up front, since the adapter is constructed under it.
  mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(config, devices=spec.devices), config.mesh_axes)

  build_start = time.perf_counter()
  engine = maxtext_engine.MaxTextTrainingEngine(
      config,
      mesh=mesh,
      wrap_with_tunix_adapter=True,
      tokenizer_pad_id=tokenizer.pad_token_id,
  )
  print(f"engine built in {time.perf_counter() - build_start:.1f}s", flush=True)
  print(f"mesh: {mesh}", flush=True)

  # `_default_loss_fn` returns a LossOutput, so its aux travels inside the structured
  # return rather than alongside it, and `has_aux` is not consulted for this shape.
  engine.with_loss_fn(peft_trainer_v2._default_loss_fn)  # pylint: disable=protected-access
  engine.with_gen_model_input_fn(qc.make_gen_model_input_fn(tokenizer.pad_token_id))

  # The engine owns its own mesh and axis-rule context (`_sharding_ctx`), so unlike the
  # PeftTrainer arms nothing is entered around the loop here. This is how the orchestrator
  # drives it.
  timer = qc.StepTimer()
  fwd_bwd_s, update_s = [], []
  with qc.maybe_trace(profile_dir, spec):
    timer.on_train_start(engine)
    # Compiled inside the trace so its cost lands in the same place PeftTrainer's does:
    # in the first step of the profile rather than before it.
    engine.compile(dataset[0])
    # `spec.ga` micro-batches per optimizer step. The engine takes accumulation from the
    # call pattern rather than from config, so this loop *is* the GA setting: the first
    # `fwd_bwd` after an `update()` starts a fresh accumulator and the rest fold into it.
    # The hook fires before every `fwd_bwd`, matching where tunix fires it, so
    # `report(group=...)` sums each run of `ga` gaps back into one optimizer step.
    for step in range(spec.steps):
      for micro in range(spec.ga):
        timer.on_train_step_start(engine)
        t0 = time.perf_counter()
        engine.fwd_bwd(dataset[step * spec.ga + micro])
        fwd_bwd_s.append(time.perf_counter() - t0)
      t1 = time.perf_counter()
      engine.update()
      update_s.append(time.perf_counter() - t1)
    engine.close()
    timer.on_train_end(engine)
    jax.effects_barrier()

  timer.report(f"maxtext {spec.model} + MaxTextTrainingEngine (scan_layers={spec.scan})", group=spec.ga)
  # Split the step in two. Both halves include a blocking `wait_for_next`, so this does not
  # separate host work from waiting -- but it does say which of the engine's two dispatches
  # the time sits behind. Under GA the fwd_bwd figure is per micro-batch and the update
  # figure is per optimizer step, so the two only add up after scaling by `ga`.
  print(
      "  fwd_bwd / update : "
      f"{statistics.median(fwd_bwd_s[qc.WARMUP_STEPS * spec.ga:]) * 1e3:.1f}ms per micro / "
      f"{statistics.median(update_s[qc.WARMUP_STEPS:]) * 1e3:.1f}ms per step (medians)"
  )
  _report_nnx_graph_cost(engine)
  print(f"train steps completed: {engine.train_step}", flush=True)
  if spec.trace:
    print(f"trace written to {profile_dir}", flush=True)


if __name__ == "__main__":
  main()
