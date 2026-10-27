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

"""Creates a randomly initialised, structurally faithful mini DeepSeek-V4.1-Flash checkpoint.

The output mirrors https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/main (same tensor names, shapes per
layer and per expert, sharded `model-XXXXX-of-YYYYY.safetensors` + `model.safetensors.index.json`, `config.json`,
`inference/config.json`, tokenizer files), but text-only and unquantized:

  * Backbone reduced from 40 to 14 layers, keeping every distinct layer type and repeating each repeating group
    twice (so scanned MaxText stacks are exercised):
        mini layer : 0  1   2  3   4  5   6   7   8   9   10  11  12  13
        orig layer : 0  1   2  3   8  9   14  15  20  21  24  25  28  29
        role       : SWA SWA+E | r2 src,c | r2 src,c | r2 src+E,c | r1 kv/idx/cand src,c | idx src,c | idx src,c
  * Routed experts reduced from 384 to 8 (6 still activated per token).
  * Engram tables shrunk by lowering `engram_vocab_size` (the prime-sized hash buckets derive from it);
    `engram_num_embeddings` is recomputed from the resulting primes. Hashing logic is unchanged.
  * `index_topk` 512 -> 4 and `candidate_topk_blocks` 2048 -> 8 so a ~500 token prompt exercises both top-k levels.
  * No vision tower / aligner and no DSpark (`mtp.*`) layers. No `quantization_config`: every weight upstream stores
    as FP8/FP4 (+ `.scale`) is stored as plain bf16 without a scale; everything else keeps its upstream dtype.

Usage (from the MaxText root; needs the upstream config.json and tokenizer files):
  python -m tests.utils.deepseek_v41_reference.create_mini_checkpoint \
      --upstream_dir ~/hf_downloads/dsv41flash --output_dir /tmp/dsv41_mini \
      [--gcs_output gs://snehalv-data/deepseek-v4.1-flash/hf/mini]
"""

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess

import torch

from . import runner
from .engram import EngramLayout

# mini layer index -> upstream layer index
MINI_TO_ORIG_LAYERS = [0, 1, 2, 3, 8, 9, 14, 15, 20, 21, 24, 25, 28, 29]
NUM_EXPERTS = 8
ENGRAM_VOCAB_SIZE = 10000
INDEX_TOPK = 4
CANDIDATE_TOPK_BLOCKS = 8
MAX_SHARD_BYTES = 4 * 1024**3

# Weights upstream stores quantized (FP8 / FP4 + scale); stored as bf16 here. Everything else keeps its dtype.
_F32_SUFFIXES = (
    "attn.attn_sink",
    "ffn.gate.bias",
    "ffn.gate.bias_vl",
    "hc_attn_base",
    "hc_attn_fn",
    "hc_attn_scale",
    "hc_ffn_base",
    "hc_ffn_fn",
    "hc_ffn_scale",
)


def _remap_layer_ids(ids, num_orig_layers):
  """Maps upstream layer ids onto mini ids, dropping ids whose layer was removed."""
  orig_to_mini = {o: m for m, o in enumerate(MINI_TO_ORIG_LAYERS)}
  del num_orig_layers
  return [orig_to_mini[i] for i in ids if i in orig_to_mini]


def engram_num_embeddings(layer_ids, max_ngram_size, n_heads, vocab_size):
  """Rows of each layer's engram table: the sum of its prime-sized buckets (see `EngramLayout`)."""

  class _Args:
    pass

  a = _Args()
  a.engram_layer_ids = layer_ids
  a.engram_max_ngram_size = max_ngram_size
  a.engram_n_heads = n_heads
  a.engram_vocab_size = vocab_size
  a.engram_num_embeddings = [0] * len(layer_ids)
  a.engram_head_dim = 1
  layout = EngramLayout.from_args(a)
  return [sum(sum(per_ngram) for per_ngram in layer) for layer in layout.primes]


def make_mini_config(hf_config: dict) -> dict:
  """Applies the mini-model changes to the upstream HF-layout config.json."""
  cfg = copy.deepcopy(hf_config)
  cfg.pop("quantization_config", None)
  cfg.pop("vision_config", None)
  t = cfg["text_config"]
  n_orig = t["num_hidden_layers"]

  # Upstream: one engram table row count per engram layer, computed from engram_vocab_size the same way.
  orig_rows = engram_num_embeddings(
      t["engram_layer_ids"], t["engram_max_ngram_size"], t["engram_n_heads"], t["engram_vocab_size"]
  )
  assert orig_rows == t["engram_num_embeddings"], (orig_rows, t["engram_num_embeddings"])

  t["num_hidden_layers"] = len(MINI_TO_ORIG_LAYERS)
  t["compress_ratios"] = [t["compress_ratios"][o] for o in MINI_TO_ORIG_LAYERS]
  t["kv_source_layer_ids"] = _remap_layer_ids(t["kv_source_layer_ids"], n_orig)
  t["index_source_layer_ids"] = _remap_layer_ids(t["index_source_layer_ids"], n_orig)
  (t["candidate_source_layer_id"],) = _remap_layer_ids([t["candidate_source_layer_id"]], n_orig)
  t["engram_layer_ids"] = _remap_layer_ids(t["engram_layer_ids"], n_orig)
  # DSpark reads the last len(dspark_target_layer_ids) backbone layers.
  n_targets = len(t["dspark_target_layer_ids"])
  t["dspark_target_layer_ids"] = list(range(t["num_hidden_layers"] - n_targets, t["num_hidden_layers"]))
  t["num_nextn_predict_layers"] = 0  # no mtp.* layers in the mini checkpoint
  t["n_routed_experts"] = NUM_EXPERTS
  t["dspark_n_routed_experts"] = NUM_EXPERTS
  assert t["num_experts_per_tok"] < NUM_EXPERTS
  t["index_topk"] = INDEX_TOPK
  t["candidate_topk_blocks"] = CANDIDATE_TOPK_BLOCKS
  t["engram_vocab_size"] = ENGRAM_VOCAB_SIZE
  t["engram_num_embeddings"] = engram_num_embeddings(
      t["engram_layer_ids"], t["engram_max_ngram_size"], t["engram_n_heads"], ENGRAM_VOCAB_SIZE
  )
  return cfg


def make_mini_inference_config(inference_config: dict, mini_hf_config: dict) -> dict:
  """Same changes on upstream's inference/config.json (ModelArgs field names)."""
  c = copy.deepcopy(inference_config)
  t = mini_hf_config["text_config"]
  c.update(
      n_layers=t["num_hidden_layers"],
      n_mtp_layers=0,
      compress_ratios=t["compress_ratios"],
      kv_source_layers=t["kv_source_layer_ids"],
      index_source_layers=t["index_source_layer_ids"],
      candidate_source_layer=t["candidate_source_layer_id"],
      engram_layer_ids=t["engram_layer_ids"],
      dspark_target_layer_ids=t["dspark_target_layer_ids"],
      n_routed_experts=t["n_routed_experts"],
      dspark_n_routed_experts=t["dspark_n_routed_experts"],
      index_topk=t["index_topk"],
      candidate_topk_blocks=t["candidate_topk_blocks"],
      engram_vocab_size=t["engram_vocab_size"],
      engram_num_embeddings=t["engram_num_embeddings"],
      vision_n_layers=0,
      dtype="bf16",
      expert_dtype=None,
  )
  return c


def _generator(name: str, seed: int) -> torch.Generator:
  digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
  return torch.Generator().manual_seed(int.from_bytes(digest[:8], "little"))


def init_tensor(name: str, shape, seed: int) -> torch.Tensor:
  """Deterministic random init that keeps activations / logits in a realistic range."""
  g = _generator(name, seed)
  randn = lambda: torch.randn(shape, generator=g, dtype=torch.float32)  # pylint: disable=unnecessary-lambda-assignment
  leaf = name.rsplit(".", 1)[-1]
  if name.endswith("norm.weight") or leaf in ("q_weight", "k_weight"):
    t = 1.0 + 0.1 * randn()
  elif leaf == "attn_sink":
    t = 0.5 * randn()
  elif name.endswith("gate.bias") or name.endswith("gate.bias_vl"):
    t = 0.1 * randn()
  elif leaf.endswith("_fn") and leaf.startswith("hc_"):
    t = 0.02 * randn()
  elif leaf.endswith("_base") and leaf.startswith("hc_"):
    t = 0.1 * randn()
  elif leaf.endswith("_scale") and leaf.startswith("hc_"):
    t = 0.5 + 0.1 * randn()
  else:
    t = 0.02 * randn()
  dtype = torch.float32 if name.endswith(_F32_SUFFIXES) else torch.bfloat16
  return t.to(dtype)


def checkpoint_tensor_specs(mini_hf_config: dict, tokenizer) -> list[tuple[str, tuple]]:
  """Names and shapes of every tensor in the mini checkpoint, in upstream order (model built on `meta`)."""
  args = runner.model_args_from_hf_config(mini_hf_config, dtype="bf16", max_seq_len=128, max_batch_size=1)
  with torch.device("meta"):
    model = runner.build_model(args, tokenizer)
  specs = [(n, tuple(p.shape)) for n, p in model.named_parameters()]
  # Upstream keeps a vision-span routing bias on every gate; the text-only reference only builds it with vision.
  out = []
  for n, shape in specs:
    out.append((n, shape))
    if n.endswith("ffn.gate.bias"):
      out.append((n + "_vl", shape))
  return out


def _layer_sort_key(name: str):
  parts = name.split(".")
  if parts[0] == "layers":
    return (1, int(parts[1]), name)
  return (0 if name.startswith("embed") else 2, 0, name)


def write_checkpoint(specs, output_dir: str, seed: int):
  """Writes deterministic synthetic safetensors shards for the mini checkpoint."""
  from safetensors.torch import save_file  # pylint: disable=import-outside-toplevel

  specs = sorted(specs, key=lambda s: _layer_sort_key(s[0]))
  # Pack into shards of at most MAX_SHARD_BYTES (estimated from shapes; bf16 = 2 bytes, f32 = 4 bytes).
  shards, current, current_bytes = [], [], 0
  for name, shape in specs:
    nbytes = (4 if name.endswith(_F32_SUFFIXES) else 2) * int(torch.Size(shape).numel())
    if current and current_bytes + nbytes > MAX_SHARD_BYTES:
      shards.append(current)
      current, current_bytes = [], 0
    current.append((name, shape))
    current_bytes += nbytes
  shards.append(current)

  weight_map, total = {}, 0
  for i, shard in enumerate(shards):
    fname = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
    tensors = {name: init_tensor(name, shape, seed) for name, shape in shard}
    total += sum(t.numel() * t.element_size() for t in tensors.values())
    save_file(tensors, os.path.join(output_dir, fname), metadata={"format": "pt"})
    weight_map.update({name: fname for name in tensors})
    print(f"wrote {fname} ({len(tensors)} tensors)", flush=True)
  with open(os.path.join(output_dir, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
    json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, f, indent=2)


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument(
      "--upstream_dir", required=True, help="dir with upstream config.json, inference/config.json, tokenizer"
  )
  parser.add_argument("--output_dir", required=True)
  parser.add_argument("--gcs_output", default="", help="optional gs:// destination to copy the checkpoint to")
  parser.add_argument("--seed", type=int, default=0)
  args = parser.parse_args()

  from transformers import AutoTokenizer  # pylint: disable=import-outside-toplevel

  os.makedirs(os.path.join(args.output_dir, "inference"), exist_ok=True)
  hf_config = runner.load_hf_config(args.upstream_dir)
  mini = make_mini_config(hf_config)
  with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
    json.dump(mini, f, indent=2)
  inference_cfg_path = os.path.join(args.upstream_dir, "inference", "config.json")
  if os.path.exists(inference_cfg_path):
    with open(inference_cfg_path, encoding="utf-8") as f:
      inference_cfg = json.load(f)
    with open(os.path.join(args.output_dir, "inference", "config.json"), "w", encoding="utf-8") as f:
      json.dump(make_mini_inference_config(inference_cfg, mini), f, indent=2)
  for fname in ("tokenizer.json", "tokenizer_config.json"):
    shutil.copyfile(os.path.join(args.upstream_dir, fname), os.path.join(args.output_dir, fname))

  tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
  specs = checkpoint_tensor_specs(mini, tokenizer)
  write_checkpoint(specs, args.output_dir, args.seed)

  # Sanity check: the reference loads the checkpoint with no missing / unexpected tensors.
  model, _ = runner.load_reference_model(args.output_dir, dtype="bf16", max_seq_len=128, tokenizer=tokenizer)
  del model
  print(f"mini checkpoint written to {args.output_dir}")

  if args.gcs_output:
    subprocess.run(["gcloud", "storage", "cp", "-r", os.path.join(args.output_dir, "*"), args.gcs_output], check=True)
    print(f"copied to {args.gcs_output}")


if __name__ == "__main__":
  main()
