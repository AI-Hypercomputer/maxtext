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

"""Baseline arm: tunix `peft_trainer_v2.PeftTrainer` driving tunix's own qwen3-0.6b.

Pair this with `qwen3_maxtext_profile.py`, which holds the trainer, optimizer, dataset,
input fn, mesh shape and step count fixed and swaps only the model. See
`qwen3_common.py` for what is shared and why the shape changed from the gemma4 pair.

Run from this directory -- the arms import `qwen3_common` as a sibling, and a working
directory that contains a `tunix/` checkout will shadow the installed package:

  cd tests/end_to_end/tpu/perf_parity && python qwen3_tunix_profile.py
"""

import argparse
import time

from flax import nnx
import jax
import optax
import qwen3_common as qc
from transformers import AutoTokenizer
from tunix.experimental.train import peft_trainer_v2
from tunix.models.qwen3 import model as qwen3_model


def create_sharded_model(config, rngs, mesh):
  """Initialises Qwen3 straight into its sharded layout.

  Building under `nnx.jit` with the partition spec applied inside means the full
  replicated parameter tree is never materialised on one device, which is what MaxText's
  `from_pretrained` does on the other arms.
  """

  @nnx.jit
  def _init(rngs):
    model = qwen3_model.Qwen3(config, rngs=rngs)
    state = nnx.state(model)
    pspecs = nnx.get_partition_spec(state)
    sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    nnx.update(model, sharded_state)
    return model

  with mesh:
    return _init(rngs)


def main() -> None:
  spec = qc.RunSpec(qc.add_common_args(argparse.ArgumentParser()).parse_args())
  profile_dir = qc.profile_dir(spec.tag("qwen3-0.6b-tunix"))
  print(spec.describe(), flush=True)

  tokenizer = AutoTokenizer.from_pretrained(qc.TOKENIZER_ID)
  dataset = qc.make_dataset_for(spec)

  # Sharded on `fsdp`. `ShardingConfig.get_default_sharding()` names both `fsdp` and
  # `tp`, so the mesh has to carry both axes even at tp=1. Passing an explicit device
  # list is what lets a 2-device mesh run on a 4-chip host.
  mesh = jax.make_mesh(
      (spec.fsdp, spec.tp),
      ("fsdp", "tp"),
      devices=spec.devices,
      axis_types=(jax.sharding.AxisType.Auto,) * 2,
  )
  print(f"mesh: {mesh}", flush=True)

  # Unmodified: 28 layers, the real qwen3-0.6b.
  config = qwen3_model.ModelConfig.qwen3_0p6b()

  build_start = time.perf_counter()
  model = create_sharded_model(config, nnx.Rngs(0), mesh)
  jax.block_until_ready(nnx.state(model))
  print(f"model built in {time.perf_counter() - build_start:.1f}s", flush=True)

  with mesh:
    optimizer = optax.inject_hyperparams(optax.sgd)(learning_rate=optax.constant_schedule(qc.LEARNING_RATE))
    # `max_steps` counts optimizer steps, not micro steps: the loop breaks on
    # `_train_steps >= max_steps`, and `_train_steps` only advances on an update.
    trainer_config = peft_trainer_v2.TrainingConfig(
        eval_every_n_steps=20000,
        max_steps=spec.steps,
        gradient_accumulation_steps=spec.ga,
    )
    trainer = peft_trainer_v2.PeftTrainer(model, optimizer, trainer_config)
    trainer = trainer.with_gen_model_input_fn(qc.make_gen_model_input_fn(tokenizer.pad_token_id))
    timer = qc.StepTimer()
    trainer.with_training_hooks(timer)

    print(f"tracing to {profile_dir}", flush=True)
    with jax.profiler.trace(log_dir=profile_dir):
      trainer.train(dataset, skip_jit=False)
      jax.effects_barrier()

  timer.report("tunix qwen3-0.6b", group=spec.ga)
  print(f"trace written to {profile_dir}", flush=True)


if __name__ == "__main__":
  main()
