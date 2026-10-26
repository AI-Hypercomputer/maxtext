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

"""CLI utility to inspect per-pass reconstruction error or materialize multi-pass weights in Orbax checkpoints.

Supports:
1. Inspecting any checkpoint for per-tensor and per-pass relative L2 error and SNR (dB).
2. Materializing/baking a `MultipassEmulationConfig`'s reconstructed weight sum (`sum_k p_k`)
   into an Orbax checkpoint and writing `multipass_emulation_metadata.json`.

Example usage:
  python3 convert_checkpoint.py src/maxtext/configs/base.yml \\
    model_name=qwen3-0.6b \\
    load_parameters_path=/path/to/input_ckpt \\
    output_checkpoint_dir=/path/to/output_ckpt \\
    emulate_multipass_enabled=True \\
    emulate_multipass_preset=F8E4M3_F8E4M3_F32_X3
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, Sequence

try:
  from etils import epath
except ImportError:  # pragma: no cover
  import pathlib as epath  # type: ignore

from maxtext.codesign.multi_pass.checkpoint_utils import (
    apply_multipass_to_model_weights,
    inspect_model_or_pytree_multipass,
    save_multipass_metadata,
)
from maxtext.codesign.multi_pass.multipass_config import MultipassEmulationConfig


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
  """Runs the checkpoint multi-pass inspection and optional materialization utility."""
  try:
    from maxtext.codesign.hooks import (  # pylint: disable=import-outside-toplevel,no-name-in-module
        extract_and_attach_multipass_config,
        install_emulation_hooks,
    )

    install_emulation_hooks()
  except ImportError:
    extract_and_attach_multipass_config = None  # type: ignore

  args = list(argv if argv is not None else sys.argv)
  inspect_only = False
  explicit_out_dir = ""
  cleaned_args: list[str] = []
  for a in args:
    if a in ("--inspect_only", "inspect_only=True", "inspect_only=true"):
      inspect_only = True
    elif a.startswith("--output_checkpoint_dir="):
      explicit_out_dir = a.split("=", 1)[1]
    elif a.startswith("output_checkpoint_dir="):
      explicit_out_dir = a.split("=", 1)[1]
    else:
      cleaned_args.append(a)

  from flax import nnx  # pylint: disable=import-outside-toplevel
  import orbax.checkpoint as ocp  # pylint: disable=import-outside-toplevel
  from maxtext.configs import pyconfig  # pylint: disable=import-outside-toplevel
  from maxtext.utils import model_creation_utils  # pylint: disable=import-outside-toplevel

  config = pyconfig.initialize(cleaned_args)
  mp_cfg = getattr(config, "multipass_emulation", None)
  if mp_cfg is None:
    if extract_and_attach_multipass_config is not None:
      mp_cfg = extract_and_attach_multipass_config(config)
    else:
      mp_cfg = MultipassEmulationConfig.from_maxtext_config(config)

  model, _ = model_creation_utils.from_pretrained(config)
  if not inspect_only and mp_cfg.enabled:
    num_modified = apply_multipass_to_model_weights(model, mp_cfg)
    logging.info(
        "[MultipassEmulation] Materialized multi-pass weight reconstruction onto %d tensors.",
        num_modified,
    )

  report = inspect_model_or_pytree_multipass(model, mp_cfg)

  out_dir = explicit_out_dir or getattr(config, "checkpoint_dir", "")
  if not inspect_only and out_dir:
    out_path = epath.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    state = nnx.state(model, nnx.Param)
    checkpointer = ocp.StandardCheckpointer()
    step_dir = str(out_path / "0" / "items")
    checkpointer.save(step_dir, state)
    checkpointer.wait_until_finished()
    save_multipass_metadata(
        str(out_path),
        mp_cfg,
        {
            "mean_targeted_relative_l2_error": report["mean_targeted_relative_l2_error"],
            "targeted_parameters": report["targeted_parameters"],
        },
    )
    logging.info("[MultipassEmulation] Saved checkpoint to %s", out_dir)

  summary = {
      "multipass_emulation": mp_cfg.to_dict(),
      "total_parameters": report["total_parameters"],
      "targeted_parameters": report["targeted_parameters"],
      "mean_targeted_relative_l2_error": report["mean_targeted_relative_l2_error"],
      "num_tensors_inspected": len(report["tensors"]),
  }
  print(json.dumps(summary, indent=2))
  return report


if __name__ == "__main__":
  main()
