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

Unlike `qwen3_maxtext_profile.py` this arm has no `--scan` switch: only the unscanned
variant has actually been run through the engine, and an unexercised code path here would
be worth less than the line it saves.

Run from this directory -- the arms import `qwen3_common` as a sibling, and a working
directory that contains a `tunix/` checkout will shadow the installed package:

  cd tests/end_to_end/tpu/perf_parity && python qwen3_engine_profile.py
"""

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

_PROFILE_DIR = qc.profile_dir("qwen3-0.6b-engine")


def _build_config(num_devices: int):
  """The qwen3_maxtext_profile.py config, plus the optimizer overrides listed above."""
  return pyconfig.initialize(
      [None, os.path.join(MAXTEXT_CONFIGS_DIR, "base.yml")],
      model_name="qwen3-0.6b",
      run_name="perf_parity_qwen3_0p6b_engine",
      base_output_directory=os.path.join(os.getcwd(), "maxtext_out"),
      max_target_length=qc.SEQ_LEN,
      per_device_batch_size=qc.BATCH_SIZE / num_devices,
      ici_fsdp_parallelism=num_devices,
      dtype="float32",
      weight_dtype="float32",
      remat_policy="none",
      scan_layers=False,
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
      learning_rate_schedule_steps=qc.MAX_STEPS,
      steps=qc.MAX_STEPS,
      gradient_accumulation_steps=qc.ACCUM_STEPS,
  )


def _report_nnx_graph_cost(engine) -> None:
  """Times the host-side NNX graph traversals the engine repeats on every step.

  `fwd_bwd` calls `nnx.split(model, nnx.Param, ...)` then `nnx.update(model, ...)`, and
  `update` calls `nnx.split(state)` then `nnx.update(state, ...)` -- four full traversals
  of an unrolled 28-layer graph per step, none of them jitted and none of them dependent
  on the device. PeftTrainer v2 pays this once. Measured after the loop, with the device
  idle, so what is timed is pure Python.
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
      "  host nnx graph   : "
      f"split(model)={timeit(lambda: nnx.split(model, nnx.Param, ...)):.1f}ms  "
      f"update(model)={timeit(lambda: nnx.update(model, rest)):.1f}ms  "
      f"split(state)={timeit(lambda: nnx.split(state)):.1f}ms  "
      f"update(state)={timeit(lambda: nnx.update(state, state_pure)):.1f}ms  "
      f"[{len(jax.tree.leaves(params))} param leaves]"
  )


def main() -> None:
  devices = jax.devices()
  print(f"devices: {len(devices)} x {devices[0].device_kind}", flush=True)

  tokenizer = AutoTokenizer.from_pretrained(qc.TOKENIZER_ID)
  dataset = qc.make_dataset()

  config = _build_config(len(devices))
  print(
      f"scan_layers={config.scan_layers}  attention={config.attention}  "
      f"remat={config.remat_policy}  opt={config.opt_type}  clip={config.gradient_clipping_threshold}",
      flush=True,
  )

  # Built here rather than left to the engine: `wrap_with_tunix_adapter=True` requires a
  # mesh up front, since the adapter is constructed under it.
  mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(config), config.mesh_axes)

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
  print(f"tracing to {_PROFILE_DIR}", flush=True)
  with jax.profiler.trace(log_dir=_PROFILE_DIR):
    timer.on_train_start(engine)
    # Compiled inside the trace so its cost lands in the same place PeftTrainer's does:
    # in the first step of the profile rather than before it.
    engine.compile(dataset[0])
    for payload in dataset:
      timer.on_train_step_start(engine)
      t0 = time.perf_counter()
      engine.fwd_bwd(payload)
      t1 = time.perf_counter()
      engine.update()
      fwd_bwd_s.append(t1 - t0)
      update_s.append(time.perf_counter() - t1)
    engine.close()
    timer.on_train_end(engine)
    jax.effects_barrier()

  timer.report("maxtext qwen3-0.6b + MaxTextTrainingEngine")
  # Split the step in two. Both halves include a blocking `wait_for_next`, so this does not
  # separate host work from waiting -- but it does say which of the engine's two dispatches
  # the time sits behind, which is the first thing to know when the step is 4x PeftTrainer's.
  print(
      "  fwd_bwd / update : "
      f"{statistics.median(fwd_bwd_s[qc.WARMUP_STEPS:]) * 1e3:.1f}ms / "
      f"{statistics.median(update_s[qc.WARMUP_STEPS:]) * 1e3:.1f}ms (medians)"
  )
  _report_nnx_graph_cost(engine)
  print(f"train steps completed: {engine.train_step}", flush=True)
  print(f"trace written to {_PROFILE_DIR}", flush=True)


if __name__ == "__main__":
  main()
