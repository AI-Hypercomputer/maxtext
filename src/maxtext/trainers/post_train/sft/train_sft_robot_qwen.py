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

"""Robot SFT of a Qwen3.5 VLM on the same Bridge task/data/labels as OpenVLA (`train_sft_robot.py`).

Same recipe as the OpenVLA shim (data cache, reached-state relabeled actions, q01/q99 normalization, 256 uniform bins,
loss only on the 7 action tokens + end token); only the sequence format and the image pre-processing change:

  inputs : <|im_start|>user\\n<|vision_start|> <|image_pad|> x N <|vision_end|>
           What action should the robot take to {instruction}?<|im_end|>\\n<|im_start|>assistant\\n a1..a7 <|im_end|>

  - Action tokens reuse the last 256 regular BPE ids of the Qwen3.5 vocab (247788..248043, rare CJK words), exactly as
    OpenVLA reuses the last 256 Llama ids: token = 248044 - digitize(norm_action). They have pretrained embeddings and
    lm_head rows, so the format also works with frozen embeddings (LoRA).
  - Image: the 224x224 cache frame -> Qwen patchify (patch 16, temporal 2, merge 2) -> 14x14 grid -> N = 49 soft tokens.
  - Positions: 3D MRoPE ids from `processor_qwen3_omni.get_rope_index` (inputs_position has shape [L, 3]).

Usage (MaxText config overrides first, then `--`, then robot data args, same as train_sft_robot):
  python -m maxtext.trainers.post_train.sft.train_sft_robot_qwen src/maxtext/configs/base.yml model_name=qwen3.5-9b \\
      use_multimodal=true load_parameters_path=<ckpt>/0/items action_norm_stats_path=<openvla config dir> ... \\
      -- --data_dir=/dev/shm/hengtaoguo/openvla/bridge224 --split=train --unnorm_key=bridge_orig
"""

import glob
import collections
import copy
import functools
import io
import os
import random
import sys
from concurrent import futures
from typing import Iterator

import jax
import numpy as np
from absl import app
from absl import flags
from PIL import Image

from maxtext.common.goodput import RECORD_JOB_START_TIME, maybe_monitor_goodput, record_goodput
from maxtext.input_pipeline import input_pipeline_interface, multihost_dataloading
from maxtext.multimodal import processor_openvla as pov
from maxtext.multimodal import processor_qwen3_omni as pq
from maxtext.trainers.post_train.sft import train_sft_robot as tsr
from maxtext.trainers.pre_train.train import get_train_func, initialize
from maxtext.utils import max_logging

QWEN_ACTION_VOCAB_END = 248044  # first Qwen3.5 special token; action ids are [248044 - 256, 248044)
IM_END_ID = 248046  # <|im_end|>
NUM_BINS = tsr.NUM_BINS


# ----------------------------------------------------------------------------------------------------------------------
# Storage helpers: shards and small asset dirs can live on local disk or on GCS (gs://). On a multi-host slice every
# host streams only its own disjoint subset of shards straight from GCS (no dataset staging on pod disk).
# ----------------------------------------------------------------------------------------------------------------------
def _is_gcs(path: str) -> bool:
  return path.startswith("gs://")


@functools.lru_cache(maxsize=1)
def _gcs_client():
  from google.cloud import storage  # pylint: disable=import-outside-toplevel

  return storage.Client()


def _split_gcs(path: str) -> tuple[str, str]:
  bucket, _, prefix = path[len("gs://") :].partition("/")
  return bucket, prefix


def list_shards(data_dir: str, split: str, max_shards: int = 0) -> list[str]:
  """Sorted `<split>-*.npz` shard paths in `data_dir` (local dir or gs:// prefix)."""
  if _is_gcs(data_dir):
    bucket, prefix = _split_gcs(data_dir.rstrip("/"))
    blobs = _gcs_client().list_blobs(bucket, prefix=f"{prefix}/{split}-")
    found = sorted(f"gs://{bucket}/{b.name}" for b in blobs if b.name.endswith(".npz"))
  else:
    found = sorted(glob.glob(os.path.join(data_dir, f"{split}-*.npz")))
  found = found[:max_shards] if max_shards else found
  if not found:
    raise FileNotFoundError(f"no {split}-*.npz shards in {data_dir}")
  return found


def read_shard(path: str):
  """Loads one cache shard (~20 MB npz); GCS shards are fetched whole into memory."""
  if _is_gcs(path):
    bucket, name = _split_gcs(path)
    return np.load(io.BytesIO(_gcs_client().bucket(bucket).blob(name).download_as_bytes()))
  return np.load(path)


def localize_dir(path: str, dest_root: str = "/tmp/robot_assets") -> str:
  """Copies a small gs:// directory (tokenizer, norm stats) to local disk; local paths are returned unchanged."""
  if not path or not _is_gcs(path):
    return path
  bucket, prefix = _split_gcs(path.rstrip("/"))
  dest = os.path.join(dest_root, os.path.basename(prefix))
  os.makedirs(dest, exist_ok=True)
  for blob in _gcs_client().list_blobs(bucket, prefix=prefix + "/"):
    rel = blob.name[len(prefix) + 1 :]
    if rel and not rel.endswith("/"):
      out = os.path.join(dest, rel)
      os.makedirs(os.path.dirname(out), exist_ok=True)
      blob.download_to_filename(out)
  max_logging.log(f"[qwen robot data] localized {path} -> {dest}")
  return dest


def host_shards(files: list[str]) -> list[str]:
  """This host's disjoint slice of the shard list (all shards on a single-host run)."""
  count, index = jax.process_count(), jax.process_index()
  mine = files[index::count]
  if not mine:
    raise ValueError(f"{len(files)} shards cannot feed {count} hosts; use more shards")
  return mine


def host_batch_size(global_batch: int) -> int:
  count = jax.process_count()
  if global_batch % count:
    raise ValueError(f"global batch {global_batch} is not divisible by {count} hosts")
  return global_batch // count


class QwenActionTokenizer(tsr.ActionTokenizer):
  """OpenVLA ActionTokenizer with the action ids moved to the tail of the Qwen3.5 BPE vocab."""

  def __init__(self, norm_stats: dict, label_transform: str = "none", vocab_end: int = QWEN_ACTION_VOCAB_END):
    super().__init__(norm_stats, label_transform)
    self.vocab_end = vocab_end
    self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

  def __call__(self, action: np.ndarray) -> np.ndarray:
    a = self.transform(action)
    norm = 2 * (a - self.q01) / np.maximum(self.q99 - self.q01, 1e-8) - 1
    norm = np.where(self.mask, np.clip(norm, -1, 1), a)  # gripper (mask=False) stays in {0, 1}
    return (self.vocab_end - np.digitize(norm, self.bins)).astype(np.int32)

  def decode(self, tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Action token ids [..., 7] -> (normalized [-1, 1], un-normalized) actions (OpenVLA `predict_action` math)."""
    idx = np.clip(self.vocab_end - np.asarray(tokens) - 1, 0, NUM_BINS - 2)
    norm = self.bin_centers[idx]
    unnorm = np.where(self.mask, 0.5 * (norm + 1) * (self.q99 - self.q01) + self.q01, norm)
    return norm, unnorm


def reformat_prompt_qwen_vla(instruction: str) -> str:
  instr = instruction.strip().lower().rstrip(".")
  return (
      "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
      f"What action should the robot take to {instr}?<|im_end|>\n<|im_start|>assistant\n"
  )


# ----------------------------------------------------------------------------------------------------------------------
# Image augmentation, matching OpenVLA's RLDS config (prismatic/vla/datasets/datasets.py, image_aug=True):
#   random_resized_crop(scale=[0.9, 0.9], ratio=[1.0, 1.0]) -> random_brightness(0.2) -> random_contrast(0.8, 1.2)
#   -> random_saturation(0.8, 1.2) -> random_hue(0.05), applied per frame on floats in [0, 1] (tf.image semantics).
# At test time OpenVLA center-crops the same area (experiments/robot/openvla_utils.py: crop_scale=0.9) to avoid a shift.
# ----------------------------------------------------------------------------------------------------------------------
CROP_AREA = 0.9


# Float reference HSV conversions (exact tf.image semantics); the training path uses PIL's faster 8-bit HSV and is
# validated against these.
def _rgb_to_hsv(x):
  r, g, b = x[..., 0], x[..., 1], x[..., 2]
  mx, mn = x.max(-1), x.min(-1)
  d = mx - mn
  safe = np.where(d > 0, d, 1.0)
  h = np.where(mx == r, (g - b) / safe % 6.0, np.where(mx == g, (b - r) / safe + 2.0, (r - g) / safe + 4.0))
  h = np.where(d > 0, h / 6.0, 0.0)
  s = np.where(mx > 0, d / np.where(mx > 0, mx, 1.0), 0.0)
  return np.stack([h, s, mx], -1)


def _hsv_to_rgb(x):
  h, s, v = x[..., 0] * 6.0, x[..., 1], x[..., 2]
  i = np.floor(h).astype(np.int32) % 6
  f = h - np.floor(h)
  p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
  choices = [np.stack(c, -1) for c in ((v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q))]
  return np.choose(i[..., None], choices)


def _crop_resize(image: np.ndarray, top: float, left: float, frac: float) -> np.ndarray:
  h, w = image.shape[:2]
  ch, cw = frac * h, frac * w
  box = (left * w, top * h, left * w + cw, top * h + ch)
  return np.asarray(Image.fromarray(image).resize((w, h), resample=Image.Resampling.BILINEAR, box=box))


def center_crop_resize(image: np.ndarray, area: float = CROP_AREA) -> np.ndarray:
  """OpenVLA test-time crop: center square covering `area` of the frame, resized back to the input size."""
  frac = float(np.sqrt(area))
  return _crop_resize(image, (1 - frac) / 2, (1 - frac) / 2, frac)


def openvla_train_augment(image: np.ndarray, rng: np.random.Generator, area: float = CROP_AREA) -> np.ndarray:
  """OpenVLA train-time image augmentation of one RGB uint8 frame (same size and dtype out).

  Random square crop covering `area` of the frame (resized back), then brightness +-0.2, contrast x[0.8, 1.2],
  saturation x[0.8, 1.2] and hue +-0.05, in OpenVLA's order and with tf.image semantics on [0, 1] values.
  """
  frac = float(np.sqrt(area))  # square crop of `area` of the image (ratio 1.0), random position
  x = _crop_resize(image, rng.uniform(0, 1 - frac), rng.uniform(0, 1 - frac), frac).astype(np.float32)
  x += rng.uniform(-0.2, 0.2) * 255.0  # brightness (additive, as tf.image.adjust_brightness on [0, 1])
  m = x.mean(axis=(0, 1), keepdims=True)
  x = (x - m) * rng.uniform(0.8, 1.2) + m  # contrast (per-channel mean, as tf.image.adjust_contrast)
  rgb = np.clip(x + 0.5, 0, 255).astype(np.uint8)
  # saturation / hue in HSV via PIL's C conversion (8-bit HSV; ~10x faster than float numpy HSV)
  hsv = np.asarray(Image.fromarray(rgb).convert("HSV")).astype(np.float32)
  hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.8, 1.2), 0, 255)
  hsv[..., 0] = (hsv[..., 0] + rng.uniform(-0.05, 0.05) * 256.0) % 256.0
  out = Image.fromarray(np.clip(hsv + 0.5, 0, 255).astype(np.uint8), mode="HSV").convert("RGB")
  return np.asarray(out)


class QwenRobotExampleBuilder:
  """(jpeg frame, instruction, action) -> fixed-length MaxText training example for a Qwen3.5 VLM."""

  def __init__(
      self,
      config,
      tokenizer,
      action_tokenizer: QwenActionTokenizer,
      image_size: int = 224,
      *,
      image_mode: str = "none",
      seed: int = 0,
  ):
    self.config = config
    self.max_len = config.max_target_length
    self.tokenizer = tokenizer
    self.action_tokenizer = action_tokenizer
    self.image_size = image_size
    self.image_mode = image_mode  # "none" | "train" (OpenVLA random aug) | "center" (OpenVLA test-time center crop)
    self.rng = np.random.default_rng(seed)
    self.qwen_tokens = pq.QwenTokens(config)
    merge = config.spatial_merge_size_for_vit
    grid = image_size // config.patch_size_for_vit
    assert grid % merge == 0, f"image_size {image_size} must be a multiple of patch*merge"
    self.grid_thw = np.asarray([[1, grid, grid]], np.int32)
    self.num_image_tokens = grid * grid // (merge * merge)
    self._prompt_cache = {}
    self.num_too_long = 0

  def _prompt(self, instruction: str):
    """-> (prompt ids with expanded image placeholders, 3D positions for the full padded sequence)."""
    if instruction not in self._prompt_cache:
      ids = self.tokenizer.encode(reformat_prompt_qwen_vla(instruction), add_special_tokens=False)
      pad = self.qwen_tokens.image_pad
      out = []
      for t in ids:
        out.extend([pad] * self.num_image_tokens if t == pad else [t])
      prompt = np.asarray(out, np.int32)
      n = len(prompt) + 8  # 7 actions + <|im_end|>
      if n > self.max_len:
        self._prompt_cache[instruction] = (prompt, None)
      else:
        seq = np.zeros((1, self.max_len), np.int32)
        seq[0, : len(prompt)] = prompt
        seq[0, len(prompt) : n] = self.action_tokenizer.vocab_end - 1  # any non-special token
        seg = np.zeros((1, self.max_len), np.int32)
        seg[0, :n] = 1
        pos, _ = pq.get_rope_index(
            input_ids=seq,
            image_grid_thw=self.grid_thw,
            attention_mask=seg,
            spatial_merge_size=self.config.spatial_merge_size_for_vit,
            config=self.config,
        )
        self._prompt_cache[instruction] = (prompt, pos[0].astype(np.int32))  # [L, 3]
    return self._prompt_cache[instruction]

  def pixels(self, image: np.ndarray) -> np.ndarray:
    """RGB HxWx3 uint8 -> Qwen patchified pixels [C, temporal_patch, H, W] (layout of the MaxText training path)."""
    cfg = self.config
    pv, grid = pq.pre_process_qwen3_image([image], cfg, force_resize=(self.image_size, self.image_size))
    return np.reshape(
        pv,
        (
            cfg.num_channels_for_vit,
            cfg.temporal_patch_size_for_vit * grid[0, 0],
            cfg.patch_size_for_vit * grid[0, 1],
            cfg.patch_size_for_vit * grid[0, 2],
        ),
    )

  def __call__(self, jpeg: bytes, instruction: str, action: np.ndarray) -> dict | None:
    prompt, pos3 = self._prompt(instruction)
    if pos3 is None:
      self.num_too_long += 1
      return None
    image = np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB"))
    if self.image_mode == "train":
      image = openvla_train_augment(image, self.rng)
    elif self.image_mode == "center":
      image = center_crop_resize(image)
    act = self.action_tokenizer(action)
    seq = np.concatenate([prompt, act, [IM_END_ID]]).astype(np.int32)
    n = len(seq)
    inputs = np.zeros(self.max_len, np.int32)
    targets = np.zeros(self.max_len, np.int32)
    inputs[:n] = seq
    targets[: n - 1] = seq[1:]
    seg = np.zeros(self.max_len, np.int32)
    seg[:n] = 1
    loss_mask = np.zeros(self.max_len, np.int32)
    loss_mask[len(prompt) - 1 : n - 1] = 1  # positions whose label is a1..a7 or <|im_end|>
    return {
        "inputs": inputs,
        "inputs_position": pos3,
        "inputs_segmentation": seg,
        "targets": targets,
        "targets_position": np.arange(self.max_len, dtype=np.int32),
        "targets_segmentation": loss_mask,
        "images": self.pixels(image).astype(np.float32),  # [C, T, H, W]
    }


def raw_robot_batches(files, batch_size, seed, buffer_shards, *, label_source="openvla", prefetch_groups=2, io_threads=8):
  """Same shuffling as `train_sft_robot.robot_batches`, but yields lists of raw (jpeg, instruction, action) rows.

  Shard reads run in a background thread pool `prefetch_groups` shuffle buffers ahead, so GCS latency of the next
  buffer refill is hidden behind the batches of the current one.
  """
  order_rng = random.Random(seed)
  pool_rng = random.Random(seed + 1)
  state = {"epoch": 0}

  def shard_groups():
    while True:
      order = files[:]
      order_rng.shuffle(order)
      for i in range(0, len(order), buffer_shards):
        yield order[i : i + buffer_shards], i + buffer_shards >= len(order)

  executor = futures.ThreadPoolExecutor(max_workers=io_threads, thread_name_prefix="robot_shard_io")
  groups = shard_groups()
  pending = collections.deque()

  def submit_next():
    group, last_in_epoch = next(groups)
    pending.append(([executor.submit(read_shard, f) for f in group], last_in_epoch))

  try:
    for _ in range(prefetch_groups + 1):
      submit_next()
    while True:
      shard_futures, last_in_epoch = pending.popleft()
      submit_next()
      pool = []
      for fut in shard_futures:
        d = fut.result()
        jpeg, off = d["jpeg"], d["offsets"]
        if label_source == "openvla":
          actions, rows = d["actions_openvla"], np.nonzero(d["valid_openvla"])[0]
        else:
          actions, rows = d["actions"], np.arange(len(d["actions"]))
        instrs = [str(s) for s in d["instructions"]]
        instr_id = d["instr_id"]
        for r in rows:
          pool.append((jpeg[off[r] : off[r + 1]].tobytes(), instrs[instr_id[r]], actions[r]))
      pool_rng.shuffle(pool)
      for j in range(0, len(pool) - batch_size + 1, batch_size):
        yield pool[j : j + batch_size]
      if last_in_epoch:
        state["epoch"] += 1
        max_logging.log(f"[qwen robot data] host {jax.process_index()} finished epoch {state['epoch']}")
  finally:
    try:
      executor.shutdown(wait=False, cancel_futures=True)
    except Exception:  # pylint: disable=broad-exception-caught
      pass  # interpreter shutdown: module globals may already be gone


def qwen_batches(builder, files, batch_size, seed, buffer_shards, *, label_source="openvla", num_workers=8):
  """Stacked training batches; (rare) over-length examples are skipped.

  Examples of each raw batch are built by `num_workers` threads (JPEG decode, augmentation and Qwen patchify release
  the GIL in PIL / numpy). Each thread owns a builder clone with its own RNG, and prompts (tokenizer + MRoPE ids) are
  resolved in the calling thread first, so the HF tokenizer is never used concurrently. Output order is deterministic.
  """
  workers = []
  for k in range(max(1, num_workers)):
    b = copy.copy(builder)  # shares tokenizer / prompt cache; own RNG
    b.rng = np.random.default_rng([seed, k])
    workers.append(b)
  executor = futures.ThreadPoolExecutor(max_workers=len(workers), thread_name_prefix="robot_example")

  def build_chunk(k, rows):
    return [workers[k](*x) for x in rows]

  pending = []
  try:
    for raw in raw_robot_batches(files, batch_size, seed, buffer_shards, label_source=label_source):
      for x in raw:
        builder._prompt(x[1])  # pylint: disable=protected-access
      step = -(-len(raw) // len(workers))
      chunks = [raw[i : i + step] for i in range(0, len(raw), step)]
      for exs in executor.map(build_chunk, range(len(chunks)), chunks):
        for ex in exs:
          if ex is not None:
            pending.append(ex)
          if len(pending) == batch_size:
            yield {k: np.stack([e[k] for e in pending]) for k in pending[0]}
            pending = []
  finally:
    try:
      executor.shutdown(wait=False, cancel_futures=True)
    except Exception:  # pylint: disable=broad-exception-caught
      pass


def make_qwen_robot_data_iterator(robot_args):
  """Returns a `create_data_iterator(config, mesh)` replacement bound to the robot data args."""

  def create_data_iterator(config, mesh):
    from transformers import AutoTokenizer  # pylint: disable=import-outside-toplevel

    all_files = list_shards(robot_args.data_dir, robot_args.split, robot_args.max_shards)
    files = host_shards(all_files)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
    norm_stats = pov.load_openvla_norm_stats(config.action_norm_stats_path)[robot_args.unnorm_key]
    seed = robot_args.seed + 1000 * jax.process_index()
    action_tokenizer = QwenActionTokenizer(norm_stats, robot_args.label_transform)
    builder = QwenRobotExampleBuilder(
        config,
        tokenizer,
        action_tokenizer,
        robot_args.image_size,
        image_mode="train" if robot_args.image_aug else "none",
        seed=seed,
    )
    # MultiHostDataLoadIterator stacks the per-host batches along axis 0, so each host loads global / num_hosts.
    batch_size = host_batch_size(config.global_batch_size_to_load)
    max_logging.log(
        f"[qwen robot data] host {jax.process_index()}/{jax.process_count()}: {len(files)} of {len(all_files)} "
        f"{robot_args.split} shards from {robot_args.data_dir} image={robot_args.image_size} "
        f"({builder.num_image_tokens} image tokens) label_source={robot_args.label_source} host_batch={batch_size} "
        f"global_batch={config.global_batch_size_to_load} image_aug={robot_args.image_aug}"
    )
    host_iter: Iterator = qwen_batches(
        builder, files, batch_size, seed, robot_args.buffer_shards, label_source=robot_args.label_source
    )
    train_iterator = multihost_dataloading.MultiHostDataLoadIterator(host_iter, mesh)
    eval_iterator = None
    if config.eval_interval > 0:
      eval_files = host_shards(list_shards(robot_args.data_dir, robot_args.eval_split, robot_args.eval_max_shards))
      eval_batch = host_batch_size(config.global_batch_size_to_load_eval)
      eval_buffer = min(4, len(eval_files))
      eval_seed = 1234 + 1000 * jax.process_index()
      # Eval sees test-time preprocessing: OpenVLA's center crop when training with augmentation, else the raw frame.
      eval_builder = QwenRobotExampleBuilder(
          config,
          tokenizer,
          action_tokenizer,
          robot_args.image_size,
          image_mode="center" if robot_args.image_aug else "none",
      )
      eval_source = tsr._Reiterable(  # pylint: disable=protected-access
          lambda: qwen_batches(
              eval_builder, eval_files, eval_batch, eval_seed, eval_buffer, label_source=robot_args.label_source
          )
      )
      eval_iterator = multihost_dataloading.MultiHostDataLoadIterator(eval_source, mesh)
      max_logging.log(
          f"[qwen robot data] eval: {len(eval_files)} {robot_args.eval_split} shards on this host, host_batch={eval_batch}"
          f" image_mode={eval_builder.image_mode}"
      )
    return train_iterator, eval_iterator

  return create_data_iterator


def parse_robot_args(argv):
  """Parses the robot data args after `--`: the OpenVLA shim's args plus `--image_size` (default 224) and
  `--image_aug` (OpenVLA train-time augmentation + center-crop eval; default off)."""
  image_size, image_aug = 224, False
  rest = []
  for a in argv:
    if a.startswith("--image_size="):
      image_size = int(a.split("=", 1)[1])
    elif a == "--image_aug" or a.startswith("--image_aug="):
      image_aug = a == "--image_aug" or a.split("=", 1)[1].lower() in ("1", "true", "yes")
    else:
      rest.append(a)
  args = tsr.parse_robot_args(rest)
  args.image_size = image_size
  args.image_aug = image_aug
  return args


def main(argv) -> None:
  argv = list(argv)
  if "--" in argv:
    split = argv.index("--")
    maxtext_argv, robot_argv = argv[:split], argv[split + 1 :]
  else:
    maxtext_argv, robot_argv = argv, []
  robot_args = parse_robot_args(robot_argv)
  # Small asset dirs given as gs:// (tokenizer, OpenVLA norm stats) are copied to local disk before config parsing.
  for i, arg in enumerate(maxtext_argv):
    key, sep, value = arg.partition("=")
    if sep and key in ("tokenizer_path", "action_norm_stats_path") and _is_gcs(value):
      maxtext_argv[i] = f"{key}={localize_dir(value)}"
  input_pipeline_interface.create_data_iterator = make_qwen_robot_data_iterator(robot_args)
  config, recorder = initialize(maxtext_argv)
  record_goodput(recorder, RECORD_JOB_START_TIME)
  train_func = get_train_func(config, recorder, maxtext_argv)
  with maybe_monitor_goodput(config):
    train_func()


if __name__ == "__main__":

  def _parse_flags(argv):
    flags.FLAGS(argv[:1])
    return argv

  app.run(main, flags_parser=_parse_flags)
  sys.exit(0)
