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

r"""Cuts a real Hugging Face checkpoint down to a mini-twin with renumbered layers.

Phase 0 / oracle tool. Keeps every non-layer tensor (embeddings, final norm,
output head) and the chosen layers of each stack, at real hidden sizes:

  python3 slice_hf_mini.py --hf-dir <ckpt> --out-dir <mini> \
      --slice 'num_hidden_layers=\.layers\.(\d+)\.:0,3' \
      [--slice 'vision_config.depth=\.blocks\.(\d+)\.:0,1']

Each `--slice` names the config key holding the stack's layer count (dotted for
nested configs), a regex whose group 1 is the layer index, and the kept ids.
Every list in the same config dict whose length equals the original layer count
(for example `layer_types`) is re-indexed to the kept ids. Any other key there
that looks layer-indexed (`first_k_dense_replace`, `*_freq`, `mlp_only_layers`,
...) stops the run unless `--set key=<json>` gives its new value explicitly.
Writes config.json, model.safetensors, copied tokenizer/processor files, and
mini_manifest.json with sha256 of every output file and the exact command.
"""

import argparse
import hashlib
import json
import pathlib
import re
import shutil
import sys
from typing import Any

AUXILIARY_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
)


def parse_slice(spec: str) -> tuple[str, re.Pattern[str], dict[int, int]]:
  """Parses 'config.key=regex:id,id' into (key, pattern, old id -> new id)."""
  key, sep, rest = spec.partition("=")
  regex, sep2, ids = rest.rpartition(":")
  if not (sep and sep2 and key and regex and ids):
    raise ValueError(f"Invalid --slice {spec!r}; expected 'config.key=regex:id0,id1'")
  kept = sorted(int(x) for x in ids.split(","))
  if len(set(kept)) != len(kept):
    raise ValueError(f"Duplicate layer ids in {spec!r}")
  return key, re.compile(regex), {old: new for new, old in enumerate(kept)}


# Keys in a sliced stack's config dict that may index layers without being a
# list of the layer count's length; each needs an explicit --set.
LAYER_INDEXED_KEY_RE = re.compile(r"layer|dense|freq|interval|pattern|every|replace|step", re.IGNORECASE)


def resolve_parent(config: dict[str, Any], dotted: str) -> tuple[dict[str, Any], str]:
  *parents, leaf = dotted.split(".")
  node = config
  for part in parents:
    node = node[part]
  if leaf not in node:
    raise KeyError(f"Config key {dotted!r} not found")
  return node, leaf


def slice_config(config: dict[str, Any], dotted: str, id_map: dict[int, int]) -> tuple[list[str], list[str]]:
  """Sets the layer count, re-indexes per-layer lists; returns (changes, keys needing --set)."""
  parent, leaf = resolve_parent(config, dotted)
  original = parent[leaf]
  kept = sorted(id_map)
  if not isinstance(original, int) or kept[-1] >= original:
    raise ValueError(f"{dotted}={original!r} cannot keep layer ids {kept}")
  prefix = dotted.rpartition(".")[0]

  def qualify(key: str) -> str:
    return f"{prefix}.{key}" if prefix else key

  parent[leaf] = len(kept)
  changes, unresolved = [f"{dotted}: {original} -> {len(kept)}"], []
  for key, value in parent.items():
    if key == leaf:
      continue
    if isinstance(value, list) and len(value) == original:
      parent[key] = [value[i] for i in kept]
      changes.append(f"{qualify(key)}: re-indexed to kept ids {kept}")
    elif isinstance(value, (int, list)) and not isinstance(value, bool) and LAYER_INDEXED_KEY_RE.search(key):
      unresolved.append(qualify(key))
  return changes, unresolved


def sha256_file(path: pathlib.Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as f:
    for block in iter(lambda: f.read(1 << 20), b""):
      digest.update(block)
  return digest.hexdigest()


def slice_checkpoint(
    hf_dir: pathlib.Path, out_dir: pathlib.Path, specs: list[str], sets: dict[str, Any]
) -> dict[str, Any]:
  """Slices a HuggingFace checkpoint directory down to selected layer indices."""
  try:
    from safetensors import safe_open  # pylint: disable=import-outside-toplevel
    from safetensors.torch import save_file  # pylint: disable=import-outside-toplevel
  except ImportError as exc:
    raise ImportError("safetensors and torch are required for slice_checkpoint") from exc
  if out_dir.exists() and any(out_dir.iterdir()):
    raise FileExistsError(f"Output dir {out_dir} is not empty; slice into a fresh directory")
  config = json.loads((hf_dir / "config.json").read_text(encoding="utf-8"))
  rules = [parse_slice(s) for s in specs]
  config_changes, unresolved = [], []
  for key, _, id_map in rules:
    changes, needs_set = slice_config(config, key, id_map)
    config_changes += changes
    unresolved += needs_set
  missing = sorted(set(unresolved) - set(sets))
  if missing:
    raise ValueError(f"Layer-indexed config keys need an explicit --set key=<json>: {missing}")
  for dotted, value in sets.items():
    parent, leaf = resolve_parent(config, dotted)
    config_changes.append(f"{dotted}: {parent[leaf]!r} -> {value!r} (--set)")
    parent[leaf] = value
  shards = sorted(hf_dir.glob("*.safetensors"))
  if not shards:
    raise FileNotFoundError(f"No .safetensors files in {hf_dir}")

  kept_tensors, kept_ids, dropped = {}, [set() for _ in rules], 0
  for shard in shards:
    with safe_open(str(shard), framework="pt") as f:
      for name in f.keys():
        matches = [(i, m) for i, (_, pattern, _) in enumerate(rules) if (m := pattern.search(name))]
        if len(matches) > 1:
          raise ValueError(f"Tensor {name!r} matches more than one --slice regex")
        if not matches:
          kept_tensors[name] = f.get_tensor(name)
          continue
        i, m = matches[0]
        old = int(m.group(1))
        id_map = rules[i][2]
        if old not in id_map:
          dropped += 1
          continue
        kept_ids[i].add(old)
        kept_tensors[name[: m.start(1)] + str(id_map[old]) + name[m.end(1) :]] = f.get_tensor(name)
  for (key, pattern, id_map), found in zip(rules, kept_ids):
    if found != set(id_map):
      raise ValueError(f"{key} ({pattern.pattern}): requested ids {sorted(id_map)}, found {sorted(found)}")

  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
  save_file(kept_tensors, str(out_dir / "model.safetensors"))
  copied = [n for n in AUXILIARY_FILES if (hf_dir / n).is_file()]
  for name in copied:
    shutil.copy2(hf_dir / name, out_dir / name)
  manifest = {
      "source": str(hf_dir),
      "command": " ".join(sys.argv),
      "slices": specs,
      "config_changes": config_changes,
      "kept_tensors": len(kept_tensors),
      "dropped_tensors": dropped,
      "sha256": {p.name: sha256_file(p) for p in sorted(out_dir.iterdir()) if p.name != "mini_manifest.json"},
  }
  (out_dir / "mini_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
  return manifest


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--hf-dir", required=True)
  parser.add_argument("--out-dir", required=True)
  parser.add_argument("--slice", action="append", required=True, dest="slices")
  parser.add_argument("--set", action="append", default=[], dest="sets", help="dotted.key=<json value>")
  args = parser.parse_args()
  sets = {}
  for item in args.sets:
    key, sep, raw = item.partition("=")
    if not sep or key in sets:
      parser.error(f"--set {item!r}: expected one 'dotted.key=<json>' per key")
    sets[key] = json.loads(raw)
  manifest = slice_checkpoint(pathlib.Path(args.hf_dir), pathlib.Path(args.out_dir), args.slices, sets)
  print(f"MINI_TWIN out={args.out_dir} kept={manifest['kept_tensors']} dropped={manifest['dropped_tensors']}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
