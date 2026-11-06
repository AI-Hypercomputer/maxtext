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
import io
import os
import random
import sys
from typing import Iterator

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


class QwenRobotExampleBuilder:
  """(jpeg frame, instruction, action) -> fixed-length MaxText training example for a Qwen3.5 VLM."""

  def __init__(self, config, tokenizer, action_tokenizer: QwenActionTokenizer, image_size: int = 224):
    self.config = config
    self.max_len = config.max_target_length
    self.tokenizer = tokenizer
    self.action_tokenizer = action_tokenizer
    self.image_size = image_size
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


def raw_robot_batches(files, batch_size, seed, buffer_shards, label_source="openvla"):
  """Same shuffling as `train_sft_robot.robot_batches`, but yields lists of raw (jpeg, instruction, action) rows."""
  rng = random.Random(seed)
  epoch = 0
  while True:
    order = files[:]
    rng.shuffle(order)
    for i in range(0, len(order), buffer_shards):
      pool = []
      for f in order[i : i + buffer_shards]:
        d = np.load(f)
        jpeg, off = d["jpeg"], d["offsets"]
        if label_source == "openvla":
          actions, rows = d["actions_openvla"], np.nonzero(d["valid_openvla"])[0]
        else:
          actions, rows = d["actions"], np.arange(len(d["actions"]))
        instrs = [str(s) for s in d["instructions"]]
        for r in rows:
          pool.append((jpeg[off[r] : off[r + 1]].tobytes(), instrs[d["instr_id"][r]], actions[r]))
      rng.shuffle(pool)
      for j in range(0, len(pool) - batch_size + 1, batch_size):
        yield pool[j : j + batch_size]
    epoch += 1
    max_logging.log(f"[qwen robot data] finished epoch {epoch}")


def qwen_batches(builder, files, batch_size, seed, buffer_shards, label_source="openvla"):
  """Stacked training batches; (rare) over-length examples are skipped."""
  pending = []
  for raw in raw_robot_batches(files, batch_size, seed, buffer_shards, label_source):
    for x in raw:
      ex = builder(*x)
      if ex is not None:
        pending.append(ex)
      if len(pending) == batch_size:
        yield {k: np.stack([e[k] for e in pending]) for k in pending[0]}
        pending = []


def make_qwen_robot_data_iterator(robot_args):
  """Returns a `create_data_iterator(config, mesh)` replacement bound to the robot data args."""

  def create_data_iterator(config, mesh):
    from transformers import AutoTokenizer  # pylint: disable=import-outside-toplevel

    def shard_files(split, max_shards):
      found = sorted(glob.glob(os.path.join(robot_args.data_dir, f"{split}-*.npz")))
      found = found[:max_shards] if max_shards else found
      if not found:
        raise FileNotFoundError(f"no {split}-*.npz shards in {robot_args.data_dir}")
      return found

    files = shard_files(robot_args.split, robot_args.max_shards)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
    norm_stats = pov.load_openvla_norm_stats(config.action_norm_stats_path)[robot_args.unnorm_key]
    builder = QwenRobotExampleBuilder(
        config, tokenizer, QwenActionTokenizer(norm_stats, robot_args.label_transform), robot_args.image_size
    )
    batch_size = config.global_batch_size_to_load
    max_logging.log(
        f"[qwen robot data] {len(files)} shards split={robot_args.split} image={robot_args.image_size} "
        f"({builder.num_image_tokens} image tokens) label_source={robot_args.label_source} batch={batch_size}"
    )
    host_iter: Iterator = qwen_batches(
        builder, files, batch_size, robot_args.seed, robot_args.buffer_shards, robot_args.label_source
    )
    train_iterator = multihost_dataloading.MultiHostDataLoadIterator(host_iter, mesh)
    eval_iterator = None
    if config.eval_interval > 0:
      eval_files = shard_files(robot_args.eval_split, robot_args.eval_max_shards)
      eval_batch = config.global_batch_size_to_load_eval
      eval_source = tsr._Reiterable(  # pylint: disable=protected-access
          lambda: qwen_batches(builder, eval_files, eval_batch, 1234, 4, robot_args.label_source)
      )
      eval_iterator = multihost_dataloading.MultiHostDataLoadIterator(eval_source, mesh)
      max_logging.log(f"[qwen robot data] eval: {len(eval_files)} {robot_args.eval_split} shards, batch={eval_batch}")
    return train_iterator, eval_iterator

  return create_data_iterator


def parse_robot_args(argv):
  """Parses the robot data args after `--`: the OpenVLA shim's args plus `--image_size` (default 224)."""
  image_size = 224
  rest = []
  for a in argv:
    if a.startswith("--image_size="):
      image_size = int(a.split("=", 1)[1])
    else:
      rest.append(a)
  args = tsr.parse_robot_args(rest)
  args.image_size = image_size
  return args


def main(argv) -> None:
  argv = list(argv)
  if "--" in argv:
    split = argv.index("--")
    maxtext_argv, robot_argv = argv[:split], argv[split + 1 :]
  else:
    maxtext_argv, robot_argv = argv, []
  robot_args = parse_robot_args(robot_argv)
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
