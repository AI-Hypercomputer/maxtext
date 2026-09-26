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

"""Manual TPU check: MaxText Qwen3.5 -> tpu-inference (vLLM/torchax) weight sync.

Two runs on the same 8-chip host:

  # 1. Reference: load Qwen/Qwen3.5-35B-A3B from the HF checkpoint on the torchax path,
  #    dump the runner state + greedy generations.
  python qwen3_5_vllm_weight_sync_check.py dump --out /some/dir

  # 2. Load the MaxText checkpoint, start Tunix's VllmSampler with dummy weights, run
  #    update_params() (MaxText mapping -> canonical vLLM layout -> tpu-inference
  #    load_canonical_weights), and compare every tensor + generations against the dump.
  python qwen3_5_vllm_weight_sync_check.py sync --out /some/dir --rl-yml src/maxtext/configs/post_train/rl.yml

Requires tpu-inference with `VllmModelWrapper.load_canonical_weights`, Tunix with the
torchax branch of `VllmSampler.update_params`, and `MODEL_IMPL_TYPE=vllm`,
`VLLM_ENABLE_V1_MULTIPROCESSING=0` in the environment.
"""

import tpu_inference  # noqa: F401  pylint: disable=unused-import  # must precede jax

import argparse
import json
import os
import sys
import time

import numpy as np
import jax
from flax import nnx
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from maxtext.configs import pyconfig, types
from maxtext.utils import model_creation_utils
from tunix.generate import mappings, vllm_sampler

MODEL = "Qwen/Qwen3.5-35B-A3B"
CKPT = "gs://maxtext-model-checkpoints/qwen3.5-35b-a3b/unscanned/0/items"
PROMPTS = [
    "The capital of France is",
    "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her "
    "friends every day with four. She sells the remainder at the farmers market daily for $2 per fresh duck egg. "
    "How much in dollars does she make every day at the farmers market? Answer:",
    "def fibonacci(n):",
    "In 1492, Columbus",
]
ENGINE_KWARGS = dict(
    model=MODEL,
    dtype="bfloat16",
    max_model_len=2048,
    max_num_seqs=8,
    max_num_batched_tokens=2048,
    language_model_only=True,
    limit_mm_per_prompt={"image": 0, "video": 0},
    enable_prefix_caching=False,
    disable_log_stats=True,
)
SAMPLING = SamplingParams(temperature=0.0, max_tokens=32, logprobs=1)


def log(*a):
  print("[sync-check]", *a, flush=True)


def _to_numpy(arr):
  return np.asarray(jax.device_get(arr))


def _load_npy(path):
  import ml_dtypes  # pylint: disable=import-outside-toplevel

  arr = np.load(path)
  if arr.dtype.kind == "V" and arr.dtype.itemsize == 2:
    arr = arr.view(ml_dtypes.bfloat16)
  return arr


def _generate(llm):
  outs = llm.generate(PROMPTS, SAMPLING)
  gens = []
  for p, o in zip(PROMPTS, outs):
    c = o.outputs[0]
    gens.append({"prompt": p, "text": c.text, "token_ids": list(c.token_ids)})
    log(repr(p[:40]), "->", repr(c.text[:100]))
  return gens


def dump(out):
  os.makedirs(os.path.join(out, "arrays"), exist_ok=True)
  llm = LLM(tensor_parallel_size=8, gpu_memory_utilization=0.45, **ENGINE_KWARGS)
  state = llm.llm_engine.model_executor.driver_worker.model_runner.state
  meta = []
  for name in sorted(state):
    arr = state[name]
    meta.append({"name": name, "shape": [int(s) for s in arr.shape], "dtype": str(arr.dtype)})
    np.save(os.path.join(out, "arrays", name + ".npy"), _to_numpy(arr))
  json.dump(meta, open(os.path.join(out, "vllm_state_meta.json"), "w"), indent=1)
  json.dump(_generate(llm), open(os.path.join(out, "vllm_ref_generations.json"), "w"), indent=1)
  log(f"dumped {len(meta)} tensors to {out}")


def sync(out, rl_yml, random_init=False):
  t0 = time.time()
  argv = [
      "",
      rl_yml,
      "model_name=qwen3.5-35b-a3b",
      f"load_parameters_path={CKPT}",
      f"tokenizer_path={MODEL}",
      "run_name=qwen35_vllm_sync_check",
      "scan_layers=false",
      "use_pathways=false",
      "checkpoint_storage_use_ocdbt=true",
      "checkpoint_storage_use_zarr3=true",
      "convert_checkpoint_if_possible=false",
      "dtype=bfloat16",
      "weight_dtype=bfloat16",
      "max_target_length=512",
      "per_device_batch_size=1",
      "log_config=false",
  ]
  config = pyconfig.initialize(argv, config_class=types.RLConfig)
  adapter, mt_mesh = model_creation_utils.from_pretrained(config, devices=jax.devices(), wrap_with_tunix_adapter=True)
  log(f"MaxText model loaded in {time.time() - t0:.0f}s")

  # Like train_rl: the rollout mesh keeps the trainer's device order so single-host
  # jax.device_put can reshard between the MaxText and vLLM meshes.
  rollout_mesh = jax.sharding.Mesh(mt_mesh.devices.flatten().reshape(1, -1), ("data", "model"))
  cfg = vllm_sampler.VllmConfig(
      mesh=rollout_mesh,
      tensor_parallel_size=8,
      data_parallel_size=1,
      mapping_config=mappings.MappingConfig.build(mapping_obj=adapter),
      init_with_random_weights=random_init,
      hbm_utilization=0.3,
      engine_kwargs=dict(ENGINE_KWARGS),
  )
  sampler = vllm_sampler.VllmSampler(AutoTokenizer.from_pretrained(MODEL), cfg)
  runner = sampler._model_runner  # pylint: disable=protected-access

  inproc_ref = None
  if not random_init:
    # Same process, same engine configuration (KV-cache size, buckets): generations with the
    # HF weights are the exact reference for the post-sync generations. Cross-process
    # references (the dump) can legitimately differ by a greedy tie-break because the engine
    # configuration differs.
    log("in-process reference generations with the HF weights:")
    inproc_ref = [g["token_ids"] for g in _generate(sampler.llm)]
    key = jax.random.PRNGKey(0)
    for name in sorted(runner.state):
      old = runner.state[name]
      if name.rsplit(".", 1)[-1].startswith("_") or "rotary_emb" in name:
        continue
      key, sub = jax.random.split(key)
      new = jax.device_put((jax.random.normal(sub, old.shape, jax.numpy.float32) * 0.02).astype(old.dtype), old.sharding)
      new.block_until_ready()
      runner.state[name] = new
      old.delete()
    runner.state_leaves = runner.state
    log("weights overwritten with on-device random values; generations now:")
    _generate(sampler.llm)

  t0 = time.time()
  sampler.update_params(nnx.state(adapter))
  jax.block_until_ready(list(runner.state.values()))
  log(f"update_params done in {time.time() - t0:.1f}s")

  ok = True
  meta_path = os.path.join(out, "vllm_state_meta.json")
  if os.path.exists(meta_path):
    diffs, worst = [], (0.0, None)
    for m in json.load(open(meta_path)):
      name = m["name"]
      ref = _load_npy(os.path.join(out, "arrays", name + ".npy"))
      got = _to_numpy(runner.state[name])
      if got.shape != ref.shape or got.dtype != ref.dtype:
        diffs.append((name, f"{got.shape}/{got.dtype} vs {ref.shape}/{ref.dtype}"))
        continue
      d = float(np.abs(got.astype(np.float32) - ref.astype(np.float32)).max())
      worst = max(worst, (d, name))
      # Attention scale scalars and the rotary cos/sin cache are computed by the sampler, not synced.
      if d != 0.0 and not name.endswith("_scale") and "rotary_emb" not in name:
        diffs.append((name, f"maxabs {d:.3g}"))
    log(f"worst tensor diff vs HF-loaded model: {worst[0]:.3g} at {worst[1]}; {len(diffs)} tensors differ")
    for n, why in diffs[:20]:
      log("   ", n, why)
    ok &= not diffs
  log("generations after the sync:")
  gens = _generate(sampler.llm)
  ref_path = os.path.join(out, "vllm_ref_generations.json")
  if os.path.exists(ref_path):
    ref = json.load(open(ref_path))
    same = sum(g["token_ids"] == r["token_ids"] for g, r in zip(gens, ref))
    log(f"greedy generations identical to the dumped (separate-process) reference: {same}/{len(ref)}")
    if inproc_ref is None:
      ok &= same == len(ref)
  if inproc_ref is not None:
    same = sum(g["token_ids"] == r for g, r in zip(gens, inproc_ref))
    log(f"greedy generations identical to the in-process HF-weight reference: {same}/{len(inproc_ref)}")
    ok &= same == len(inproc_ref)
  log("RESULT", "PASS" if ok else "FAIL")
  return ok


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("mode", choices=["dump", "sync"])
  parser.add_argument("--out", required=True)
  parser.add_argument("--rl-yml", default="src/maxtext/configs/post_train/rl.yml")
  parser.add_argument(
      "--random-init",
      action="store_true",
      help="start the sampler from vLLM's dummy loader (slow CPU init, ~17 min for 35B) instead of "
      "loading the HF weights, capturing in-process reference generations and overwriting the "
      "weights with on-device random values",
  )
  args = parser.parse_args()
  if args.mode == "dump":
    dump(args.out)
  else:
    sys.exit(0 if sync(args.out, args.rl_yml, random_init=args.random_init) else 1)


if __name__ == "__main__":
  main()
