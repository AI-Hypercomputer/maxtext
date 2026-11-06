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

"""Supervised fine-tuning of a vision-language-action model (OpenVLA) on robot trajectories.

Thin shim over the standard MaxText trainer (`maxtext.trainers.pre_train.train`): optimizer, sharding, checkpoint
loading/saving, metrics and the multimodal loss path are all reused. Only the data iterator is replaced by a robot
iterator that turns (camera frame, instruction, 7-DoF action) timesteps into OpenVLA training sequences:

  inputs : [BOS] [IMG x256] "In: What action should the robot take to {instruction}?\\nOut:" [29871] [a1..a7] [EOS]
  targets: next-token labels; loss mask (targets_segmentation) = 1 only where the label is an action token or EOS.

Labels follow OpenVLA's Bridge transform (`bridge_oxe_dataset_transform` + `relabel_bridge_actions`): the movement
action is the *reached* proprio delta state[t+1, :6] - state[t, :6] (not the commanded world_vector/rotation_delta),
the gripper is open_gripper[t], and the first/last step of every episode are dropped. The cache stores these as
`actions_openvla` / `valid_openvla` (added by `bridge_labels.py apply`); `--label_source=raw` uses the commanded actions.
On 256 frames the released OpenVLA scores action CE 2.7 / token acc 71% on the relabeled targets vs 11.5 / 19% on raw.

Action tokens follow OpenVLA's ActionTokenizer: normalize with the dataset q01/q99 (gripper not normalized), clip to
[-1, 1], `np.digitize` into 256 uniform bins, token = 32000 - bin (exact inverse of `processor_openvla` decoding).

Data: the 224x224 BridgeData cache produced by `build_bridge_cache.py` (`<split>-<shard>.npz`).

Usage (MaxText config overrides first, then `--`, then robot data args):
  python -m maxtext.trainers.post_train.sft.train_sft_robot src/maxtext/configs/base.yml model_name=openvla-7b \\
      use_multimodal=true load_parameters_path=<ckpt>/0/items ... steps=10 \\
      -- --data_dir=/dev/shm/hengtaoguo/openvla/bridge224 --split=train --unnorm_key=bridge_orig
"""

import argparse
import glob
import io
import os
import random
from typing import Iterator

import numpy as np
from absl import app
from absl import flags
from PIL import Image

from maxtext.common.goodput import RECORD_JOB_START_TIME, maybe_monitor_goodput, record_goodput
from maxtext.input_pipeline import input_pipeline_interface, multihost_dataloading
from maxtext.multimodal import processor_openvla as pov
from maxtext.trainers.pre_train.train import get_train_func, initialize
from maxtext.utils import max_logging

EOS_ID = 2
NUM_BINS = 256
ACTION_VOCAB_END = 32000  # OpenVLA: token = 32000 - digitize(action)


# ----------------------------------------------------------------------------------------------------------------------
# Example construction
# ----------------------------------------------------------------------------------------------------------------------
class ActionTokenizer:
  """OpenVLA-compatible continuous action -> token ids."""

  def __init__(self, norm_stats: dict, label_transform: str = "none"):
    stats = norm_stats["action"]
    self.q01 = np.asarray(stats["q01"], np.float32)
    self.q99 = np.asarray(stats["q99"], np.float32)
    self.mask = np.asarray(stats.get("mask", np.ones_like(self.q01, dtype=bool)), bool)
    self.bins = np.linspace(-1, 1, NUM_BINS)
    self.label_transform = label_transform

  def transform(self, action: np.ndarray) -> np.ndarray:
    """Optional label transform, e.g. to simulate a new embodiment for adaptation demos."""
    a = action.astype(np.float32).copy()
    if self.label_transform == "flip_y_yaw":  # mirrored lateral axis: y and yaw inverted
      a[1] *= -1
      a[5] *= -1
    elif self.label_transform != "none":
      raise ValueError(f"unknown label_transform {self.label_transform!r}")
    return a

  def __call__(self, action: np.ndarray) -> np.ndarray:
    a = self.transform(action)
    norm = 2 * (a - self.q01) / np.maximum(self.q99 - self.q01, 1e-8) - 1
    norm = np.where(self.mask, np.clip(norm, -1, 1), a)  # gripper (mask=False) stays in {0, 1}
    return (ACTION_VOCAB_END - np.digitize(norm, self.bins)).astype(np.int32)


class RobotExampleBuilder:
  """(jpeg frame, instruction, action) -> fixed-length MaxText training example."""

  def __init__(self, config, tokenizer, action_tokenizer: ActionTokenizer):
    self.max_len = config.max_target_length
    self.tokenizer = tokenizer
    self.action_tokenizer = action_tokenizer
    self._prompt_cache = {}

  def _prompt_ids(self, instruction: str) -> np.ndarray:
    if instruction not in self._prompt_cache:
      prompt = pov.reformat_prompt_openvla(instruction)
      ids = [self.tokenizer.bos_token_id] + self.tokenizer.encode(prompt, add_special_tokens=False)
      # -> [BOS, IMG x256, prompt..., 29871]
      self._prompt_cache[instruction] = pov.add_extra_tokens_for_images_openvla(np.asarray(ids, np.int32))
    return self._prompt_cache[instruction]

  def __call__(self, jpeg: bytes, instruction: str, action: np.ndarray) -> dict:
    image = np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB"))
    pixels = pov.preprocess_mm_data_openvla([image]).pixel_values  # [1, 224, 224, 6]
    prompt = self._prompt_ids(instruction)
    act = self.action_tokenizer(action)
    seq = np.concatenate([prompt, act, [EOS_ID]]).astype(np.int32)
    n = len(seq)
    if n > self.max_len:
      raise ValueError(f"sequence length {n} > max_target_length {self.max_len}")
    inputs = np.zeros(self.max_len, np.int32)
    targets = np.zeros(self.max_len, np.int32)
    inputs[:n] = seq
    targets[: n - 1] = seq[1:]
    seg = np.zeros(self.max_len, np.int32)
    seg[:n] = 1
    loss_mask = np.zeros(self.max_len, np.int32)
    loss_mask[len(prompt) - 1 : n - 1] = 1  # positions whose label is a1..a7 or EOS
    pos = np.arange(self.max_len, dtype=np.int32)
    return {
        "inputs": inputs,
        "inputs_position": pos,
        "inputs_segmentation": seg,
        "targets": targets,
        "targets_position": pos,
        "targets_segmentation": loss_mask,
        "images": pixels.astype(np.float32),  # [N=1, 224, 224, 6]
    }


# ----------------------------------------------------------------------------------------------------------------------
# Data iterator over the cached shards
# ----------------------------------------------------------------------------------------------------------------------
def robot_batches(
    config,
    builder: RobotExampleBuilder,
    files: list[str],
    batch_size: int,
    seed: int,
    buffer_shards: int,
    label_source: str = "openvla",
):
  """Infinite shuffled iterator: mixes `buffer_shards` shards at a time, shuffles all their timesteps."""
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
          if "actions_openvla" not in d.files:
            raise KeyError(f"{f} has no actions_openvla; run `bridge_labels.py apply` on the cache first")
          actions, rows = d["actions_openvla"], np.nonzero(d["valid_openvla"])[0]
        else:
          actions, rows = d["actions"], np.arange(len(d["actions"]))
        instrs = [str(s) for s in d["instructions"]]
        for r in rows:
          pool.append((jpeg[off[r] : off[r + 1]].tobytes(), instrs[d["instr_id"][r]], actions[r]))
      rng.shuffle(pool)
      for j in range(0, len(pool) - batch_size + 1, batch_size):
        examples = [builder(*x) for x in pool[j : j + batch_size]]
        yield {k: np.stack([e[k] for e in examples]) for k in examples[0]}
    epoch += 1
    max_logging.log(f"[robot data] finished epoch {epoch}")


class _Reiterable:
  """Iterable whose `iter()` restarts the underlying generator (MultiHostDataLoadIterator.reset calls iter())."""

  def __init__(self, make_gen):
    self.make_gen = make_gen

  def __iter__(self):
    return self.make_gen()


def make_robot_data_iterator(robot_args):
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
    stats_path = config.action_norm_stats_path or config.tokenizer_path
    norm_stats = pov.load_openvla_norm_stats(stats_path)[robot_args.unnorm_key]
    builder = RobotExampleBuilder(config, tokenizer, ActionTokenizer(norm_stats, robot_args.label_transform))
    batch_size = config.global_batch_size_to_load
    max_logging.log(
        f"[robot data] {len(files)} shards from {robot_args.data_dir} split={robot_args.split} "
        f"unnorm_key={robot_args.unnorm_key} label_source={robot_args.label_source} "
        f"label_transform={robot_args.label_transform} batch={batch_size}"
    )
    host_iter: Iterator = robot_batches(
        config,
        builder,
        files,
        batch_size,
        seed=robot_args.seed,
        buffer_shards=robot_args.buffer_shards,
        label_source=robot_args.label_source,
    )
    train_iterator = multihost_dataloading.MultiHostDataLoadIterator(host_iter, mesh)
    eval_iterator = None
    if config.eval_interval > 0:
      eval_files = shard_files(robot_args.eval_split, robot_args.eval_max_shards)
      eval_batch = config.global_batch_size_to_load_eval
      # Re-iterable with a fixed seed: every eval pass (iterator reset) sees the same held-out batches.
      eval_source = _Reiterable(
          lambda: robot_batches(
              config, builder, eval_files, eval_batch, seed=1234, buffer_shards=4, label_source=robot_args.label_source
          )
      )
      eval_iterator = multihost_dataloading.MultiHostDataLoadIterator(eval_source, mesh)
      max_logging.log(f"[robot data] eval: {len(eval_files)} {robot_args.eval_split} shards, batch={eval_batch}")
    return train_iterator, eval_iterator

  return create_data_iterator


def parse_robot_args(argv):
  """Parses the robot data args that follow `--` on the command line (data dir, split, labels, shuffling)."""
  ap = argparse.ArgumentParser("train_sft_robot")
  ap.add_argument("--data_dir", default="/dev/shm/hengtaoguo/openvla/bridge224")
  ap.add_argument("--split", default="train")
  ap.add_argument("--unnorm_key", default="bridge_orig")
  ap.add_argument(
      "--label_source",
      default="openvla",
      choices=["openvla", "raw"],
      help="openvla: reached-state-delta labels as in OpenVLA training (default); raw: commanded actions",
  )
  ap.add_argument("--label_transform", default="none", choices=["none", "flip_y_yaw"])
  ap.add_argument("--max_shards", type=int, default=0, help="0 = all shards")
  ap.add_argument("--eval_split", default="test", help="held-out split used when eval_interval > 0")
  ap.add_argument("--eval_max_shards", type=int, default=32, help="held-out shards used for eval (0 = all)")
  ap.add_argument("--buffer_shards", type=int, default=8, help="shards mixed per shuffle buffer")
  ap.add_argument("--seed", type=int, default=0)
  return ap.parse_args(argv)


def main(argv) -> None:
  argv = list(argv)
  if "--" in argv:
    split = argv.index("--")
    maxtext_argv, robot_argv = argv[:split], argv[split + 1 :]
  else:
    maxtext_argv, robot_argv = argv, []
  robot_args = parse_robot_args(robot_argv)
  # `train_utils.setup_train_loop` imports `create_data_iterator` from this module at call time.
  input_pipeline_interface.create_data_iterator = make_robot_data_iterator(robot_args)
  config, recorder = initialize(maxtext_argv)
  record_goodput(recorder, RECORD_JOB_START_TIME)
  train_func = get_train_func(config, recorder, maxtext_argv)
  with maybe_monitor_goodput(config):
    train_func()


if __name__ == "__main__":
  # Pass-through flags parser: absl would otherwise reject the robot args after `--`; app.run still sets up logging.
  def _parse_flags(argv):
    flags.FLAGS(argv[:1])  # mark absl flags as parsed without consuming any arguments
    return argv

  app.run(main, flags_parser=_parse_flags)
