# Copyright 2023–2026 Google LLC
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

r"""Captures golden reference activations with PyTorch forward hooks and records sha256.

Run by the implementer in the oracle/golden step, on the pinned reference only:

  python3 capture_golden.py --factory pkg.module:make --hooks 'model.layers.*' 'lm_head' \
      --seed 0 --out goldens/forward.npz [--wrappers pkg.module:wrap]

`--factory` returns (torch.nn.Module, dict of inputs). Hooks record the inputs
and outputs of matching modules as `<module>#<call>.in.<arg>` and
`<module>#<call>.out[.<field>]`; the final output is `__output__[.<field>]`.
A glob such as `model.layers.*` also matches every submodule below each layer;
use `model.layers.[0-9]` style globs to hook layer boundaries only. bfloat16
tensors (numpy has no bfloat16) are stored as float32 and listed under
`upcast_from_bfloat16`. Nothing is added to the reference code. Writes `<out>` and
`<out stem>_manifest.json` with the npz sha256, the reference versions and the
exact command, for the golden manifest and the provenance test.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import importlib
import json
import pathlib
import sys
from typing import Any, Callable

import numpy as np


def _require_torch() -> Any:
  """Imports torch lazily so --help and pure-Python helpers stay torch-free."""
  try:
    import torch  # pylint: disable=import-outside-toplevel
  except ImportError as exc:
    raise ImportError("torch is required to capture PyTorch golden activations") from exc
  return torch


def load_callable(target: str) -> Callable[..., Any]:
  """Imports `pkg.module:attr`; raises if it is missing or not callable."""
  module_name, sep, attr = target.partition(":")
  if not sep:
    raise ValueError(f"Expected 'pkg.module:attr', got {target!r}")
  fn = getattr(importlib.import_module(module_name), attr)
  if not callable(fn):
    raise TypeError(f"{target} is not callable")
  return fn


class Collected:
  """Tensors captured so far, plus non-tensor leaves and bfloat16 upcasts, by key."""

  def __init__(self):
    self.tensors: dict[str, np.ndarray] = {}
    self.skipped: dict[str, str] = {}
    self.upcast: list[str] = []

  def add(self, obj: Any, prefix: str) -> None:
    """Collects tensors from tensors, sequences and mappings; records other leaves as skipped."""
    torch = _require_torch()
    if isinstance(obj, torch.Tensor):
      tensor = obj.detach().cpu()
      if tensor.dtype == torch.bfloat16:
        tensor = tensor.to(torch.float32)
        self.upcast.append(prefix)
      self.tensors[prefix] = tensor.numpy()
    elif isinstance(obj, (tuple, list)):
      for i, item in enumerate(obj):
        self.add(item, f"{prefix}.{i}")
    elif isinstance(obj, dict):  # Includes transformers ModelOutput.
      for k, v in obj.items():
        self.add(v, f"{prefix}.{k}")
    else:
      self.skipped[prefix] = type(obj).__name__


def sha256_file(path: pathlib.Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as f:
    for block in iter(lambda: f.read(1 << 20), b""):
      digest.update(block)
  return digest.hexdigest()


def capture(
    factory: str, hooks: list[str], seed: int, out: pathlib.Path, *, wrappers: str | None = None, command: str = ""
) -> dict[str, Any]:
  """Captures goldens; `command` is the exact command recorded in the manifest."""
  torch = _require_torch()
  torch.manual_seed(seed)
  np.random.seed(seed)
  model, inputs = load_callable(factory)()
  if not isinstance(model, torch.nn.Module) or not isinstance(inputs, dict):
    raise TypeError(f"Factory must return (torch.nn.Module, dict), got ({type(model)}, {type(inputs)})")
  model.eval()

  if wrappers is not None:
    clean, wrapped = Collected(), Collected()
    with torch.no_grad():
      clean.add(model(**inputs), "__output__")
      model = load_callable(wrappers)(model)
      if not isinstance(model, torch.nn.Module):
        raise TypeError(f"Wrapper {wrappers} must return a torch.nn.Module, got {type(model)}")
      wrapped.add(model(**inputs), "__output__")
    if clean.tensors.keys() != wrapped.tensors.keys() or any(
        not np.array_equal(v, wrapped.tensors[k]) for k, v in clean.tensors.items()
    ):
      raise AssertionError("Capture wrappers changed the model outputs; wrappers must be pure pass-through")

  collected = Collected()
  calls: dict[str, int] = {}
  handles = []
  matched = [name for name, _ in model.named_modules() if name and any(fnmatch.fnmatch(name, p) for p in hooks)]
  if not matched:
    raise ValueError(f"No module matches hooks {hooks}")
  modules = dict(model.named_modules())
  for name in matched:

    def hook(_mod, args, kwargs, output, name=name):
      prefix = f"{name}#{calls.setdefault(name, 0)}"
      calls[name] += 1
      for key, val in list(enumerate(args)) + list(kwargs.items()):
        collected.add(val, f"{prefix}.in.{key}")
      collected.add(output, f"{prefix}.out")

    handles.append(modules[name].register_forward_hook(hook, with_kwargs=True))
  with torch.no_grad():
    collected.add(model(**inputs), "__output__")
  for h in handles:
    h.remove()

  out.parent.mkdir(parents=True, exist_ok=True)
  np.savez(str(out), **collected.tensors)
  manifest = {
      "file": out.name,
      "sha256": sha256_file(out),
      "command": command,
      "seed": seed,
      "torch_version": torch.__version__,
      "factory": factory,
      "hooks": hooks,
      "modules_matched": len(matched),
      "tensors": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in sorted(collected.tensors.items())},
      "upcast_from_bfloat16": sorted(collected.upcast),
      "skipped_non_tensor_leaves": collected.skipped,
  }
  out.with_name(f"{out.stem}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
  return manifest


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--factory", required=True, help="'pkg.module:fn' returning (model, inputs dict)")
  parser.add_argument("--hooks", nargs="+", required=True, help="Module-name globs to hook")
  parser.add_argument("--seed", type=int, required=True)
  parser.add_argument("--out", required=True, help="Output .npz path")
  parser.add_argument("--wrappers", help="'pkg.module:fn' pass-through wrapper; outputs must stay bitwise identical")
  args = parser.parse_args()
  manifest = capture(
      args.factory, args.hooks, args.seed, pathlib.Path(args.out), wrappers=args.wrappers, command=" ".join(sys.argv)
  )
  print(
      f"GOLDEN file={args.out} sha256={manifest['sha256']} tensors={len(manifest['tensors'])} "
      f"skipped={len(manifest['skipped_non_tensor_leaves'])}"
  )
  return 0


if __name__ == "__main__":
  sys.exit(main())
