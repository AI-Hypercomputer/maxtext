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

"""End-to-end parity of the MaxText LingBot-World-Fast pipeline against the reference implementation.

The reference (`WanI2VFast` from https://github.com/Robbyant/lingbot-world, run on GPU) dumps its intermediates and
Gaussian draws to an `.npz` (keys: context, noise, step_noise, y, cam, flow, t, start, video); see the model README.
This script replays the same noise through `LingBotWorldPipeline` and reports per-stage errors:

  python -m maxtext.m3.models.lingbot.compare_reference src/maxtext/m3/models/lingbot/configs/lingbot-world-fast.yml \
      checkpoint_directory=<converted ckpt> image_path=<image> action_path=<dir> reference_npz=<ref.npz> \
      prompt="..." num_frames=81 output_dir=<dir>
"""

import json
import os
from typing import Sequence

from absl import logging as absl_logging

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.m3.configs import config as config_lib
from maxtext.m3.models.lingbot.pipeline_lingbot import (
    LingBotWorldPipeline,
    commit_kv_cache,
    cross_kv_pass,
    transformer_chunk_pass,
)
from maxtext.m3.models.lingbot.transformer_lingbot import rope_cos_sin
from maxtext.multimodal.processor_vae import export_to_video
from maxtext.utils import max_logging


def _stats(name: str, actual, expected) -> dict:
  actual = np.asarray(actual, np.float64).reshape(-1)
  expected = np.asarray(expected, np.float64).reshape(-1)
  diff = actual - expected
  cos = float(np.dot(actual, expected) / (np.linalg.norm(actual) * np.linalg.norm(expected) + 1e-30))
  rel = float(np.linalg.norm(diff) / (np.linalg.norm(expected) + 1e-30))
  stats = {"max_abs": float(np.abs(diff).max()), "mean_abs": float(np.abs(diff).mean()), "rel_l2": rel, "cosine": cos}
  max_logging.log(f"[parity] {name:<28} " + "  ".join(f"{k}={v:.3e}" for k, v in stats.items()))
  return stats


def to_uint8(video: np.ndarray) -> np.ndarray:
  """Reference video [3, F, H, W] in [-1, 1] -> [F, H, W, 3] uint8, quantized like `vae_decode_pass`."""
  video = np.clip(video / 2.0 + 0.5, 0.0, 1.0)
  return (np.transpose(video, (1, 2, 3, 0)) * 255.0).astype(np.uint8)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
  mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
  return float(10 * np.log10(255.0**2 / mse)) if mse > 0 else float("inf")


def teacher_forced_flows(pipeline: LingBotWorldPipeline, ref: dict, config) -> np.ndarray:
  """DiT outputs for every reference call, fed the reference inputs (T5 context, I2V condition, camera, noisy latents
  and KV cache rebuilt from the reference x0s), so per-call errors do not compound across chunks / steps."""
  transformer = pipeline.transformer
  scheduler = pipeline.scheduler
  chunk = config.chunk_size
  _, lat_f, lat_h, lat_w = ref["noise"].shape
  frame_seqlen = (lat_h // 2) * (lat_w // 2)
  steps = len(config.timesteps_index)
  num_chunks = lat_f // chunk
  step_noise = ref["step_noise"].reshape(num_chunks, steps - 1, *ref["step_noise"].shape[1:])
  ref_flows = ref["flow"].reshape(num_chunks, steps + 1, *ref["flow"].shape[1:])  # 4 denoise + 1 commit per chunk
  context = np.zeros((1, config.max_sequence_length, ref["context"].shape[-1]), np.float32)
  context[0, : ref["context"].shape[0]] = ref["context"]
  graphdef, state, rest = nnx.split(transformer, nnx.Param, ...)
  flows = []
  with jax.set_mesh(pipeline.mesh):
    cross_kv = cross_kv_pass(graphdef, state, rest, jnp.asarray(context, config.activations_dtype))
    shape = transformer.kv_cache_shape(1, lat_f * frame_seqlen)
    kv_cache = (jnp.zeros(shape, config.activations_dtype), jnp.zeros(shape, config.activations_dtype))
    for c in range(num_chunks):
      sl = slice(c * chunk, (c + 1) * chunk)
      start = jnp.int32(c * chunk * frame_seqlen)
      cos, sin = rope_cos_sin(transformer.head_dim, c * chunk, chunk, lat_h // 2, lat_w // 2)
      cond = jnp.asarray(ref["y"][None, :, sl])
      ref_cam = ref["cam"].reshape(ref["cam"].shape[-4:])  # recorded as [1, 384, F, H, W]
      cam = jnp.asarray(ref_cam[None, :, sl], config.activations_dtype)
      xt = ref["noise"][:, sl]
      for s, t in enumerate(scheduler.timesteps):
        flow, _ = transformer_chunk_pass(
            graphdef,
            state,
            rest,
            jnp.asarray(xt[None]),
            jnp.full((1,), t, jnp.float32),
            cond,
            cam,
            cross_kv,
            cos,
            sin,
            kv_cache,
            start,
        )
        flows.append(np.asarray(flow)[0])
        x0 = (xt.astype(np.float64) - scheduler.x0_sigma(t) * ref_flows[c, s].astype(np.float64)).astype(np.float32)
        if s < steps - 1:
          sigma = scheduler.noise_sigma(scheduler.timesteps[s + 1])
          xt = (1 - sigma) * x0 + sigma * step_noise[c, s]
      _, chunk_kv = transformer_chunk_pass(
          graphdef,
          state,
          rest,
          jnp.asarray(x0[None]),
          jnp.zeros((1,), jnp.float32),
          cond,
          cam,
          cross_kv,
          cos,
          sin,
          kv_cache,
          start,
      )
      kv_cache = commit_kv_cache(kv_cache, chunk_kv, start)
  return np.stack(flows)


def main(argv: Sequence[str] | None = None) -> None:
  absl_logging.set_verbosity(absl_logging.INFO)
  config = config_lib.initialize(argv)
  ref = dict(np.load(config.reference_npz))
  num_chunks = ref["noise"].shape[1] // config.chunk_size
  steps = len(config.timesteps_index)
  noise = {
      "noise": ref["noise"],
      "step_noise": ref["step_noise"].reshape(num_chunks, steps - 1, *ref["step_noise"].shape[1:]),
  }
  pipeline = LingBotWorldPipeline.from_checkpoint(config)
  intermediates = {}
  video, trace = pipeline(
      prompt=config.prompt,
      height=config.height,
      width=config.width,
      num_frames=config.num_frames,
      noise=noise,
      intermediates=intermediates,
  )
  max_logging.log(f"[parity] timing: {trace}")

  results = {}
  seq_len = ref["context"].shape[0]
  prompt_embeds = np.asarray(jax.device_get(intermediates["prompt_embeds"]), np.float32)[0]
  results["t5_context"] = _stats("T5 context", prompt_embeds[:seq_len], ref["context"])
  results["t5_padding_zero"] = float(np.abs(prompt_embeds[seq_len:]).max())
  results["i2v_condition"] = _stats("I2V condition (mask+VAE)", np.asarray(intermediates["condition"])[0], ref["y"])
  results["camera"] = _stats(
      "camera Plucker", np.asarray(intermediates["cam"], np.float32)[0], ref["cam"].reshape(ref["cam"].shape[-4:])
  )
  ref_flows = ref["flow"][ref["t"] != 0]  # drop the KV-cache commit calls (t = 0)
  flows = np.stack(intermediates["flow"])[:, 0]
  for i, (flow, ref_flow) in enumerate(zip(flows, ref_flows)):
    results[f"flow_{i}"] = _stats(f"DiT flow chunk {i // steps} step {i % steps}", flow, ref_flow)
  results["flow_all"] = _stats("DiT flow (all calls)", flows, ref_flows)
  if getattr(config, "teacher_forcing", True):
    forced = teacher_forced_flows(pipeline, ref, config)
    for i, (flow, ref_flow) in enumerate(zip(forced, ref_flows)):
      results[f"forced_flow_{i}"] = _stats(f"forced flow chunk {i // steps} step {i % steps}", flow, ref_flow)
    results["forced_flow_all"] = _stats("forced flow (all calls)", forced, ref_flows)

  ours = np.asarray(video)[0]
  theirs = to_uint8(ref["video"])
  results["video"] = _stats("video (uint8)", ours, theirs)
  results["video_psnr_db"] = psnr(ours, theirs)
  per_frame = [psnr(a, b) for a, b in zip(ours, theirs)]
  max_logging.log(
      f"[parity] video PSNR {results['video_psnr_db']:.2f} dB (per frame min {min(per_frame):.2f}, "
      f"max {max(per_frame):.2f}); shape {ours.shape}"
  )

  output_dir = getattr(config, "output_dir", "") or "."
  os.makedirs(output_dir, exist_ok=True)
  export_to_video(ours, os.path.join(output_dir, "lingbot_maxtext.mp4"), fps=config.fps)
  export_to_video(theirs, os.path.join(output_dir, "lingbot_reference.mp4"), fps=config.fps)
  np.save(os.path.join(output_dir, "lingbot_maxtext_video.npy"), ours)
  with open(os.path.join(output_dir, "parity_metrics.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, indent=1)
  max_logging.log(f"[parity] wrote videos and parity_metrics.json to {output_dir}")


if __name__ == "__main__":
  main()
