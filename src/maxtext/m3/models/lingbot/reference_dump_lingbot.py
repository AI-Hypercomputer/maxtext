# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dump LingBot-World-Fast reference intermediates for MaxText parity checks (`compare_reference.py`).

Standalone script for a GPU machine with the reference repo (https://github.com/Robbyant/lingbot-world) and its
environment (torch + CUDA, flash-attn); it does not import MaxText. `--repo` points at the reference checkout.

Runs the official `WanI2VFast` pipeline (same args as generate_fast.py) and records, on rank 0:
  context (T5 output), noise (initial), step_noise (per chunk / per re-noise step), y (I2V conditioning),
  cam (Plucker map), flow / t / start (every DiT call, incl. the t=0 KV-cache commits), video (decoded, in [-1, 1]).
`--ckpt_dir` is the prepared lingbot-world-base-cam directory with `lingbot_world_fast/` inside (the reference infers
the camera control type from 'cam' in that path).

torchrun --standalone --nproc_per_node=4 reference_dump_lingbot.py --repo <lingbot-world> --ckpt_dir ... \
  --image ... --action_path ... \
  --frame_num 81 --prompt "..." --out lingbot_ref.npz
"""

import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

RECORD = {"noise": [], "flow": [], "y": [], "cam": [], "t": [], "start": []}


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--repo", default=os.environ.get("LINGBOT_WORLD_REPO", "lingbot-world"))
  p.add_argument("--ckpt_dir", required=True)
  p.add_argument("--image", required=True)
  p.add_argument("--action_path", required=True)
  p.add_argument("--prompt", required=True)
  p.add_argument("--frame_num", type=int, default=81)
  p.add_argument("--size", default="480*832")
  p.add_argument("--seed", type=int, default=42)
  p.add_argument("--out", default="lingbot_ref.npz")
  args = p.parse_args()
  sys.path.insert(0, os.path.abspath(args.repo))
  import wan  # pylint: disable=import-outside-toplevel,import-error,no-name-in-module
  from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS  # pylint: disable=import-outside-toplevel,import-error,no-name-in-module
  from wan.distributed.util import init_distributed_group  # pylint: disable=import-outside-toplevel,import-error,no-name-in-module

  rank = int(os.getenv("RANK", "0"))
  world_size = int(os.getenv("WORLD_SIZE", "1"))
  local_rank = int(os.getenv("LOCAL_RANK", "0"))
  logging.basicConfig(level=logging.INFO if rank == 0 else logging.ERROR)
  torch.cuda.set_device(local_rank)
  if world_size > 1:
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    init_distributed_group()
  cfg = WAN_CONFIGS["i2v-A14B"]
  pipe = wan.WanI2VFast(
      config=cfg,
      checkpoint_dir=args.ckpt_dir,
      device_id=local_rank,
      rank=rank,
      t5_fsdp=world_size > 1,
      dit_fsdp=world_size > 1,
      use_sp=world_size > 1,
  )

  # Record the seeded Gaussian draws (initial noise, then one per re-noise step).
  orig_randn = torch.randn

  def randn(*a, **k):
    out = orig_randn(*a, **k)
    if k.get("generator") is not None:
      RECORD["noise"].append(out.float().cpu().numpy())
    return out

  torch.randn = randn

  orig_t5 = pipe.text_encoder.__call__

  class T5Rec:
    """Records the T5 prompt embedding."""

    def __init__(self, enc):
      self.enc = enc

    def __getattr__(self, name):
      return getattr(self.enc, name)

    def __call__(self, *a, **k):
      out = self.enc(*a, **k)
      RECORD["context"] = out[0].float().cpu().numpy()
      return out

  pipe.text_encoder = T5Rec(pipe.text_encoder)
  del orig_t5

  class ModelRec:
    """Records every DiT call (output, timestep, chunk start, conditioning)."""

    def __init__(self, model):
      self.__dict__["model"] = model

    def __getattr__(self, name):
      return getattr(self.__dict__["model"], name)

    def __call__(self, *a, **k):
      out = self.__dict__["model"](*a, **k)
      if rank == 0:
        RECORD["flow"].append(out[0].float().cpu().numpy())
        RECORD["t"].append(float(k["t"].float().cpu()[0]))
        RECORD["start"].append(int(k["current_start"]))
        if not RECORD["y"] or RECORD["start"][-1] != RECORD["start"][-2]:
          RECORD["y"].append(k["y"][0].float().cpu().numpy())
          RECORD["cam"].append(k["dit_cond_dict"]["c2ws_plucker_emb"][0].float().cpu().numpy())
      return out

  pipe.model = ModelRec(pipe.model)

  img = Image.open(args.image).convert("RGB")
  video = pipe.generate(
      args.prompt,
      img,
      action_path=args.action_path,
      chunk_size=3,
      max_area=MAX_AREA_CONFIGS[args.size],
      frame_num=args.frame_num,
      shift=cfg.sample_shift,
      seed=args.seed,
      offload_model=False,
  )
  if rank == 0:
    noise = RECORD.pop("noise")
    flows = RECORD.pop("flow")
    np.savez(
        args.out,
        context=RECORD["context"],
        noise=noise[0],
        step_noise=np.stack(noise[1:]),
        flow=np.stack(flows),
        t=np.array(RECORD["t"]),
        start=np.array(RECORD["start"]),
        y=np.concatenate(RECORD["y"], axis=1),
        cam=np.concatenate(RECORD["cam"], axis=2),
        video=video.float().cpu().numpy(),
    )
    print("saved", args.out, flush=True)
  if dist.is_initialized():
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
  main()
