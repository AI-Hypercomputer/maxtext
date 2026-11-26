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

"""Evaluation entrypoint for checkpoints with Multi-Pass Precision Emulation.

Supports two evaluation modes:
1. **Benchmark Harness Mode (default)**:
   Delegates to `maxtext.eval.runner.run` (`lm_eval`, `evalchemy`, `simple_evals`,
   `eval`) with multi-pass emulation hooks active so `VllmServerManager` and
   `MaxTextForCausalLM` execute with all configured multi-pass rules.
2. **Native MaxText Inspection & Forward-Pass Mode (`--native_eval` or `emulate_native_eval=True`)**:
   Loads the model/checkpoint directly via `model_creation_utils.from_pretrained`,
   evaluates per-pass reconstruction error / SNR across targeted weights, and
   computes forward-pass loss/perplexity.

Example usage (Harness mode):
  python3 eval.py src/maxtext/eval/configs/eval.yml \\
    model_name=qwen3-0.6b \\
    load_parameters_path=/path/to/ckpt \\
    emulate_multipass_enabled=True \\
    emulate_multipass_preset=F8E4M3_F8E4M3_F32_X3 \\
    --harness lm_eval --tasks gsm8k

Example usage (Native inspection & forward pass):
  python3 eval.py src/maxtext/configs/base.yml \\
    model_name=qwen3-0.6b \\
    load_parameters_path=/path/to/ckpt \\
    emulate_multipass_enabled=True \\
    emulate_multipass_preset=F8E4M3_F8E4M3_F32_X3 \\
    --native_eval
"""

from __future__ import annotations

import json
import logging
import math
import sys
from typing import Any, Sequence

try:
  import numpy as np
except ImportError:  # pragma: no cover
  np = None  # type: ignore

from maxtext.codesign.hooks import (
    extract_and_attach_multipass_config,
    install_emulation_hooks,
)
from maxtext.codesign.interception import annotate_model_paths
from maxtext.codesign.multi_pass.checkpoint_utils import (
    apply_multipass_to_model_weights,
    inspect_model_or_pytree_multipass,
)


def run_native_emulated_eval(maxtext_args: Sequence[str]) -> dict[str, Any]:
  """Loads a MaxText model/checkpoint, inspects multi-pass error, and runs an emulated forward pass."""
  install_emulation_hooks(workflow="eval")
  import jax  # pylint: disable=import-outside-toplevel
  import jax.numpy as jnp  # pylint: disable=import-outside-toplevel
  from maxtext.common.common_types import MODEL_MODE_TRAIN  # pylint: disable=import-outside-toplevel
  from maxtext.configs import pyconfig  # pylint: disable=import-outside-toplevel
  from maxtext.utils import model_creation_utils  # pylint: disable=import-outside-toplevel

  config = pyconfig.initialize(list(maxtext_args))
  emulation_cfg = getattr(config, "multipass_emulation", None)
  if emulation_cfg is None:
    emulation_cfg = extract_and_attach_multipass_config(config)

  model, _ = model_creation_utils.from_pretrained(config)
  if emulation_cfg.enabled:
    annotate_model_paths(model, emulation_cfg)
    if emulation_cfg.apply_weight_multipass_on_load:
      apply_multipass_to_model_weights(model, emulation_cfg)

  report = inspect_model_or_pytree_multipass(model, emulation_cfg)
  aux_reports = {
      feat.feature_name: feat.inspect_model_or_pytree(model) for feat in emulation_cfg.active_auxiliary_features
  }
  logging.info(
      "[HardwareEmulation][Eval] Total params: %d | Mean targeted rel L2 err: %.6e | Summary: %s",
      report["total_parameters"],
      report["mean_targeted_relative_l2_error"],
      emulation_cfg.summary(),
  )

  batch_size = max(1, int(getattr(config, "per_device_batch_size", 1)))
  seq_len = min(128, int(getattr(config, "max_target_length", 128)))
  vocab_size = int(getattr(config, "vocab_size", 32000))

  rng = np.random.default_rng(42)
  input_tokens = jnp.asarray(rng.integers(1, min(1000, vocab_size), size=(batch_size, seq_len), dtype=np.int32))
  positions = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32)[None, :], (batch_size, seq_len))
  segment_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)

  logits = model(
      input_tokens,
      positions,
      decoder_segment_ids=segment_ids,
      enable_dropout=False,
      model_mode=MODEL_MODE_TRAIN,
  )
  if isinstance(logits, tuple):
    logits = logits[0]

  shift_logits = logits[:, :-1, :].astype(jnp.float32)
  shift_labels = input_tokens[:, 1:]
  log_probs = jax.nn.log_softmax(shift_logits, axis=-1)
  token_log_probs = jnp.take_along_axis(log_probs, shift_labels[..., None], axis=-1).squeeze(-1)
  nll = float(-jnp.mean(token_log_probs))
  ppl = float(math.exp(min(nll, 50.0)))

  summary = {
      "multipass_emulation": emulation_cfg.to_dict(),
      "auxiliary_features": aux_reports,
      "total_parameters": report["total_parameters"],
      "targeted_parameters": report["targeted_parameters"],
      "mean_targeted_relative_l2_error": report["mean_targeted_relative_l2_error"],
      "forward_pass_nll": nll,
      "forward_pass_perplexity": ppl,
      "logits_shape": list(logits.shape),
  }
  print(json.dumps(summary, indent=2))
  return summary


def main(argv: Sequence[str] | None = None) -> None:
  """Runs evaluation with multi-pass emulation hooks (either native inspection or benchmark harness)."""
  install_emulation_hooks(workflow="eval")
  args = list(argv if argv is not None else sys.argv)
  native_flag = False
  cleaned_args: list[str] = []
  for a in args:
    if a in (
        "--native_eval",
        "--inspect_multipass",
        "emulate_native_eval=True",
        "emulate_native_eval=true",
    ):
      native_flag = True
    else:
      cleaned_args.append(a)

  if native_flag:
    run_native_emulated_eval(cleaned_args)
    return

  from maxtext.eval.runner import run as eval_run  # pylint: disable=import-outside-toplevel

  sys.argv = cleaned_args
  eval_run.main()


if __name__ == "__main__":
  main()
