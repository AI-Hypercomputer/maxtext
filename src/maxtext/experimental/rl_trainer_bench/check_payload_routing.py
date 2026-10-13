#!/usr/bin/env python3
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

"""CPU check of bench_driver.realistic_payload at the 397B production shapes (no TPU, no cluster; needs Tunix).

Builds the TRAINER_WARMUP_COMPILE payload with Tunix's own builder at the 397B e2e run's BatchConfig (64 rows x 65,536
tokens, 16 segments per row at most, trainer_fsdp=32, trainer_expert=2, routed_experts [60 layers, top-10]), then
the realistic regular and final micro-batches, and checks them under the cp-as-ep EP=8 layout
(`payload_shard_load_ratio`: EP group = 2 rows x 4 context slices, 4 token chunks per slice, 512 experts in 8
shards of 64):

  * shapes, dtypes and fields equal the warm-up payload's; padding has pad ids and UNSET routing; every real
    token has top-k distinct experts in range (so MaxText replays it rather than falling back to the router);
  * regular, routing=probe: each layer's max shard-load ratio is within 0.10 of its target in the default
    calibration table, the max over layers is in that run's regular range [1.8, 2.45], and the hottest layer is
    the table's (model layer 59);
  * ROUTING_SCALE stretches it: scale 1.46 puts the max at about 2.45, the highest regular micro-batch of that run;
  * routing=uniform: max ratio below 1.2;
  * final: the pad fraction is FINAL_PAD_FRACTION of whole rows plus PAD_FRACTION of the rest.

Run with a Tunix that matches the e2e run, e.g. inside the trainer image:
  docker run --rm --entrypoint python -v $PWD:/tb:ro -e JAX_PLATFORMS=cpu <image> /tb/check_payload_routing.py
The shapes and thresholds are the 397B recipe's; edit the constants below for another model.
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_driver as d  # pylint: disable=g-import-not-at-top

from tunix.experimental.common import datatypes  # pylint: disable=g-import-not-at-top
from tunix.experimental.examples.deepswe_dist import trainer_warmup  # pylint: disable=g-import-not-at-top
from tunix.experimental.orchestrator import algorithm_adapter  # pylint: disable=g-import-not-at-top
from tunix.experimental.orchestrator import batch_assembly  # pylint: disable=g-import-not-at-top
from tunix.rl import algorithm_config  # pylint: disable=g-import-not-at-top

NUM_EXPERTS, EP, LAYERS, TOP_K = 512, 8, 60, 10


def main():
  assert datatypes.UNSET_ROUTED_EXPERT == d.UNSET_ROUTED_EXPERT
  algo = algorithm_adapter.GRPOAdapter(
      algo_config=algorithm_config.GRPOConfig(num_generations=16, beta=0.0, temperature=1.0),
      mini_batch_size=72, train_micro_batch_size=64, max_packed_len=65536, max_response_length=57344)
  bc = batch_assembly.BatchConfig(pad_id=0, max_prompt_length=8192, max_response_length=57344,
                                  max_seq_token_per_tpu=65536, max_segments_per_packed_row=16,
                                  trainer_fsdp=32, trainer_dp=1, trainer_expert=2)
  t = time.monotonic()
  warm = trainer_warmup.build_warmup_payload(algo=algo, batch_config=bc, num_generations=16, mini_batch_size=72,
                                             train_micro_batch_size=64, routed_experts_shape=(LAYERS, TOP_K))
  print(f"warm-up payload {time.monotonic() - t:.1f}s: {batch_assembly.describe_payload(warm)}", flush=True)
  w_ratio, w_pad = d.payload_shard_load_ratio(warm, num_experts=NUM_EXPERTS, ep=EP)
  print(f"warm-up: pad {w_pad:.3f}, replayed max ratio {w_ratio.max():.3f} (all routing UNSET: the router decides)")

  kw = dict(num_experts=NUM_EXPERTS, ep=EP, pad_id=0, vocab_size=248_000)
  targets = d.layer_targets(LAYERS)
  results = {}
  for name, extra in (("regular", {}), ("scale1.46", {"scale": 1.46}), ("uniform", {"routing": "uniform"}),
                      ("final", {"pad_rows_fraction": 0.25, "seed": 1})):
    t = time.monotonic()
    p = d.realistic_payload(warm, **kw, **extra)
    gen = time.monotonic() - t
    ratio, pad = d.payload_shard_load_ratio(p, num_experts=NUM_EXPERTS, ep=EP)
    top = np.argsort(-ratio)[:5]
    print(f"{name}: built {gen:.1f}s, pad {pad:.4f}, max ratio {ratio.max():.3f}, hottest "
          f"{[(int(l), round(float(ratio[l]), 3)) for l in top]}, min {ratio.min():.3f}", flush=True)
    results[name] = (p, ratio, pad)

  p, ratio, pad = results["regular"]
  for f in ("completion_ids", "completion_mask", "advantages", "segment_ids", "segment_positions",
            "old_per_token_logps", "rollout_per_token_logps", "overlong", "routed_experts"):
    a, b = np.asarray(getattr(p, f)), np.asarray(getattr(warm, f))
    assert a.shape == b.shape and a.dtype == b.dtype, (f, a.shape, b.shape, a.dtype, b.dtype)
  assert p.num_segments == warm.num_segments
  routed = np.asarray(p.routed_experts)
  real = np.asarray(p.segment_ids) > 0
  assert (routed[~real] == d.UNSET_ROUTED_EXPERT).all() and (np.asarray(p.completion_ids)[~real] == 0).all()
  r = routed[real]  # [n, L, K]
  assert r.min() >= 0 and r.max() < NUM_EXPERTS
  srt = np.sort(r[:200_000], axis=-1)
  assert not (srt[..., 1:] == srt[..., :-1]).any(), "duplicate experts would fall back to the router"
  assert len(np.unique(np.asarray(p.completion_ids)[real][:100_000])) > 50_000, "token ids not varied"
  assert abs(pad - 0.07) < 0.005, pad
  dev = np.abs(ratio - targets)
  print(f"regular: |ratio - target| max {dev.max():.3f}, mean {dev.mean():.3f}")
  assert dev.max() < 0.10, dev.max()
  assert 1.8 <= ratio.max() <= 2.45 and int(np.argmax(ratio)) == 59, (ratio.max(), np.argmax(ratio))
  assert 2.35 <= results["scale1.46"][1].max() <= 2.55, results["scale1.46"][1].max()
  assert results["uniform"][1].max() < 1.2, results["uniform"][1].max()
  fpad = results["final"][2]
  assert abs(fpad - (0.25 + 0.75 * 0.07)) < 0.01, fpad
  print("PAYLOAD_ROUTING_CHECK_PASSED", flush=True)


if __name__ == "__main__":
  main()
