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

"""Comparison arm: the same tunix trainer driving MaxText's qwen3-0.6b.

Only the model changes from `qwen3_tunix_profile.py`; everything else comes out of
`qwen3_common.py`. The swap is a straight drop-in because
`peft_trainer_v2._default_loss_fn` calls `model(input_tokens, positions, None,
attention_mask)` positionally, which is exactly `TunixMaxTextAdapter.__call__`'s
signature -- and also exactly `tunix.models.qwen3.Qwen3.__call__`'s.

Three MaxText defaults are overridden, each to match a tunix default rather than to
flatter either side:

  * `dtype="float32"`. `tunix.models.qwen3.ModelConfig` defaults `dtype` and
    `param_dtype` to float32; MaxText's base.yml runs bfloat16 compute over float32
    weights. Left alone MaxText would win on numerics rather than on implementation.
    (`weight_dtype` is already float32 on both sides.)
  * `remat_policy="none"`. base.yml defaults to `full`, which recomputes the whole layer
    in the backward pass -- roughly a third more FLOPs. tunix's `RematConfig.NONE` does
    not remat at all.
  * `scan_layers=False`. tunix builds its 28 decoder layers as a `ModuleList` and runs a
    Python loop over them, so MaxText's default `nn.scan` would be comparing compilation
    strategies, not implementations. Pass `--scan` to run the scanned variant instead;
    it is MaxText's production default and worth a third trace.

The logical axis rules have to be live at trace time. MaxText layers place their
sharding constraints through `nn_partitioning.get_axis_rules()`, a context variable;
traced outside `nn_partitioning.axis_rules(...)` every one of those constraints becomes
a silent no-op and XLA guesses the partitioning for activations and gradients.
`MaxTextTrainingEngine._sharding_ctx` records 1012 ms against 581 ms on
llama3.1-8b/fsdp=8 for that mistake, with identical numerics. PeftTrainer knows nothing
about the rules, so this script enters them around `train()` itself.

Deliberately not equalised, because it is part of what is being compared: MaxText's
`attention: autoselected` picks its own TPU kernel, while tunix's qwen3 leaves
`use_flash_attention` at its default of False. The chosen kernel is logged.

Run from this directory -- the arms import `qwen3_common` as a sibling, and a working
directory that contains a `tunix/` checkout will shadow the installed package:

  cd tests/end_to_end/tpu/perf_parity && python qwen3_maxtext_profile.py [--scan]
"""

import argparse
import os
import time

from flax.linen import partitioning as nn_partitioning
import jax
from maxtext.configs import pyconfig
from maxtext.utils import model_creation_utils
from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR
import optax
import qwen3_common as qc
from transformers import AutoTokenizer
from tunix.experimental.train import peft_trainer_v2


def _build_config(spec: qc.RunSpec, scan_layers: bool):
  """Builds the MaxText config for a qwen3-0.6b matched to the tunix run.

  `per_device_batch_size` describes the *micro*-batch. MaxText's own
  `gradient_accumulation_steps` is left at 1 deliberately: it would have base.yml split
  the loaded batch into micro-batches itself, whereas here the trainer is handed one
  micro-batch at a time and does the accumulating.
  """
  return pyconfig.initialize(
      [None, os.path.join(MAXTEXT_CONFIGS_DIR, "base.yml")],
      model_name=spec.model,
      run_name="perf_parity_qwen3_0p6b",
      base_output_directory=os.path.join(os.getcwd(), "maxtext_out"),
      max_target_length=spec.seq,
      per_device_batch_size=spec.per_device_batch,
      ici_fsdp_parallelism=spec.fsdp,
      ici_tensor_parallelism=spec.tp,
      # The three overrides that match tunix's defaults; see the module docstring.
      dtype="float32",
      weight_dtype="float32",
      remat_policy="none",
      scan_layers=scan_layers,
      enable_dropout=False,
      enable_checkpointing=False,
      # Random init, as on the tunix side. Without this MaxText tries to pull the real
      # checkpoint from HF and convert it.
      convert_checkpoint_if_possible=False,
      skip_jax_distributed_system=True,
      use_multimodal=False,
      init_weights_seed=0,
  )


def main() -> None:
  spec = qc.RunSpec(qc.add_common_args(argparse.ArgumentParser()).parse_args())
  scan_layers = spec.scan
  profile_dir = qc.profile_dir(spec.tag(f"{spec.model}-maxtext" + ("-scan" if scan_layers else "")))
  print(spec.describe(), flush=True)

  tokenizer = AutoTokenizer.from_pretrained(qc.TOKENIZER_ID)
  dataset = qc.make_dataset_for(spec)

  config = _build_config(spec, scan_layers)
  print(f"scan_layers={config.scan_layers}  attention={config.attention}  remat={config.remat_policy}", flush=True)

  build_start = time.perf_counter()
  # mesh=None, so MaxText builds its own mesh from the ici_* settings -- but over the
  # devices this run was given, which may be a subset of the host's.
  model, mesh = model_creation_utils.from_pretrained(
      config,
      devices=spec.devices,
      wrap_with_tunix_adapter=True,
      tokenizer_pad_id=tokenizer.pad_token_id,
  )
  print(f"model built in {time.perf_counter() - build_start:.1f}s", flush=True)
  print(f"mesh: {mesh}", flush=True)

  optimizer = optax.inject_hyperparams(optax.sgd)(learning_rate=optax.constant_schedule(qc.LEARNING_RATE))
  trainer_config = peft_trainer_v2.TrainingConfig(
      eval_every_n_steps=20000,
      max_steps=spec.steps,
      gradient_accumulation_steps=spec.ga,
  )
  trainer = peft_trainer_v2.PeftTrainer(model, optimizer, trainer_config)
  trainer = trainer.with_gen_model_input_fn(qc.make_gen_model_input_fn(tokenizer.pad_token_id))
  timer = qc.StepTimer()
  trainer.with_training_hooks(timer)

  # The axis rules have to be live when the first call triggers tracing, not merely when
  # the jit is constructed -- see the module docstring.
  with jax.set_mesh(mesh), nn_partitioning.axis_rules(config.logical_axis_rules):
    with qc.maybe_trace(profile_dir, spec):
      trainer.train(dataset, skip_jit=False)
      jax.effects_barrier()

  timer.report(f"maxtext {spec.model} (scan_layers={scan_layers})", group=spec.ga)
  if spec.trace:
    print(f"trace written to {profile_dir}", flush=True)


if __name__ == "__main__":
  main()
