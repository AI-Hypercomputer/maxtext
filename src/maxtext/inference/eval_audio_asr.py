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

"""Batched ASR evaluation (audio + prompt -> transcript) for audio-input models such as the Ming thinker.

Streams a parquet dataset (columns `audio` {bytes,path} / `text`, e.g. the shards written by
`tools/data_generation/prepare_gigaspeech_parquet.py`), runs greedy decoding through `MaxEngine`
(one prefill per clip, batched generate across all decode slots) and reports:

  * WER  - corpus word error rate after text normalization (standard ASR metric),
  * CER  - corpus character error rate after normalization,
  * EM   - exact match rate after normalization,
  * raw_EM - exact match of the raw strings (casing/punctuation included: the "style" the SFT teaches),
  * format_ok - fraction of outputs that stopped on an end-of-turn token (no runaway generation).

Normalization: lower-case, strip punctuation (keep in-word apostrophes), collapse whitespace, so casing and
punctuation differences do not count as recognition errors.

Usage (MaxText config args as for `maxtext.inference.decode`, then script flags after `--`):

  python -m maxtext.inference.eval_audio_asr src/maxtext/configs/base.yml model_name=ming-3.0-audio-tiny \
      load_parameters_path=gs://.../0/items tokenizer_type=huggingface tokenizer_path=<hf dir> \
      per_device_batch_size=8 max_prefill_predict_length=256 max_target_length=384 \
      decode_sampling_strategy=greedy prompt="Transcribe the speech into text." ... \
      -- --eval_files=gs://.../gigaspeech_test/test-*.parquet --output=/path/eval_base.jsonl --max_examples=5000

Audio is padded to the smallest bucket in `--audio_buckets` (mel frames) that fits the clip, so prefill compiles
once per bucket; the number of audio placeholder tokens follows the clip's true length (as in training/inference).
"""

import argparse
import json
import os
import re
import sys
import time
from types import SimpleNamespace
from typing import Sequence

import fsspec
import jax
import numpy as np
import pyarrow.parquet as pq

from maxtext.configs import pyconfig
from maxtext.inference.maxengine import maxengine
from maxtext.multimodal import processor as mm_processor
from maxtext.multimodal import processor_ming
from maxtext.utils import max_logging

END_OF_TURN_IDS = set(processor_ming.EOS_IDS)  # <|endoftext|>, <|role_end|>


# ----------------------------------------------------------------------------- metrics


def normalize_text(text: str) -> str:
  text = re.sub(r"<\|[^|]*\|>", " ", text)  # special tokens
  text = text.lower().replace("-", " ")
  text = re.sub(r"[^a-z0-9' ]+", " ", text)
  text = re.sub(r"(?<![a-z])'|'(?![a-z])", " ", text)  # quotes that are not in-word apostrophes
  return " ".join(text.split())


def edit_distance(ref: Sequence, hyp: Sequence) -> int:
  """Levenshtein distance between two sequences."""
  prev = list(range(len(hyp) + 1))
  for i, r in enumerate(ref, 1):
    cur = [i]
    for j, h in enumerate(hyp, 1):
      cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
    prev = cur
  return prev[-1]


class Metrics:
  """Corpus-level accumulators."""

  def __init__(self):
    self.n = self.word_errors = self.words = self.char_errors = self.chars = 0
    self.exact = self.raw_exact = self.format_ok = 0

  def add(self, ref_raw: str, hyp_raw: str, stopped: bool):
    ref, hyp = normalize_text(ref_raw), normalize_text(hyp_raw)
    self.n += 1
    self.word_errors += edit_distance(ref.split(), hyp.split())
    self.words += len(ref.split())
    self.char_errors += edit_distance(ref, hyp)
    self.chars += len(ref)
    self.exact += int(ref == hyp)
    self.raw_exact += int(ref_raw.strip() == hyp_raw.strip())
    self.format_ok += int(stopped)

  def summary(self) -> dict:
    return {
        "num_examples": self.n,
        "wer": self.word_errors / max(self.words, 1),
        "cer": self.char_errors / max(self.chars, 1),
        "exact_match": self.exact / max(self.n, 1),
        "raw_exact_match": self.raw_exact / max(self.n, 1),
        "format_ok": self.format_ok / max(self.n, 1),
    }


# ----------------------------------------------------------------------------- data


def iter_examples(files_glob: str, max_examples: int):
  """Yields rows (segment_id, audio, text, duration) from the parquet shards matching , in file order."""
  fs, _, paths = fsspec.get_fs_token_paths(files_glob)
  count = 0
  for path in sorted(paths):
    with fs.open(path, "rb") as f:
      pf = pq.ParquetFile(f)
      for rg in range(pf.metadata.num_row_groups):
        for row in pf.read_row_group(rg, columns=["segment_id", "audio", "text", "duration"]).to_pylist():
          yield row
          count += 1
          if max_examples and count >= max_examples:
            return


def pick_bucket(duration_s: float, buckets: list[int]) -> int:
  frames = int(np.ceil(duration_s * 100)) + 1
  for b in buckets:
    if frames <= b:
      return b
  return buckets[-1]


# ----------------------------------------------------------------------------- main


def main(argv: Sequence[str]) -> None:
  if "--" in argv:
    split = argv.index("--")
    config_argv, script_argv = argv[:split], argv[split + 1 :]
  else:
    config_argv, script_argv = argv, []
  parser = argparse.ArgumentParser()
  parser.add_argument("--eval_files", required=True, help="Parquet glob (local or gs://) with audio/text columns.")
  parser.add_argument("--output", required=True, help="Per-example JSONL; `<output>.summary.json` holds the metrics.")
  parser.add_argument("--max_examples", type=int, default=0)
  parser.add_argument("--audio_buckets", default="500,1000,1500,2000", help="Mel-frame buckets (100 frames/s).")
  parser.add_argument("--log_every", type=int, default=10, help="Log running metrics every N batches.")
  args = parser.parse_args(script_argv)
  buckets = sorted(int(b) for b in args.audio_buckets.split(","))

  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  config = pyconfig.initialize(config_argv)
  assert config.use_audio and config.use_multimodal, "eval_audio_asr needs use_audio=true use_multimodal=true"
  jax.config.update("jax_use_shardy_partitioner", config.shardy)

  engine = maxengine.MaxEngine(config)
  rng = jax.random.PRNGKey(1234)
  rng, rng_load = jax.random.split(rng)
  params = engine.load_params(rng_load)
  tokenizer_model = engine.build_tokenizer(engine.get_tokenizer())
  batch_size = int(config.per_device_batch_size * jax.device_count())
  max_gen_steps = config.max_target_length - config.max_prefill_predict_length
  prompt_text = mm_processor.reformat_prompt(
      prompt=config.prompt,
      image_placeholder=config.image_placeholder,
      model_name=config.model_name,
      num_images=0,
      audio_placeholder=config.audio_placeholder,
      num_audios=1,
  )
  max_logging.log(f"batch_size={batch_size} max_gen_steps={max_gen_steps} buckets={buckets}\nprompt: {prompt_text!r}")

  def prepare(row):
    """Featurize one clip: (audio_values [1, T, n_mels], padded tokens [prefill_len], true_length)."""
    bucket = pick_bucket(float(row["duration"]), buckets)
    # Same featurizer as training, with the per-bucket static length instead of config.audio_max_mel_frames.
    bucket_config = SimpleNamespace(
        audio_max_mel_frames=bucket, model_name=config.model_name, decoder_block=config.decoder_block
    )
    feats = mm_processor.preprocess_audio_for_training(row["audio"], bucket_config)
    offsets = mm_processor.get_image_offsets(config=config, processor_output=feats)
    tokens, true_length = tokenizer_model.encode(
        prompt_text, is_bos=False, prefill_lengths=[config.max_prefill_predict_length - offsets]
    )
    tokens = mm_processor.prepare_text_for_image_fusion(tokens=tokens, config=config, processor_output=feats)
    true_length += offsets
    assert true_length <= config.max_prefill_predict_length, f"{true_length=} > {config.max_prefill_predict_length=}"
    return feats.audio_values, np.asarray(tokens), true_length

  metrics = Metrics()
  fs_out, _, _ = fsspec.get_fs_token_paths(args.output)
  if os.path.dirname(args.output):
    fs_out.makedirs(os.path.dirname(args.output), exist_ok=True)
  out = fsspec.open(args.output, "w").open()
  start = time.time()
  batch, num_batches = [], 0

  def run_batch(rows, num_real):
    """Prefills `rows` into slots 0..len(rows)-1, decodes, scores the first `num_real` (the rest are padding)."""
    nonlocal rng
    prepared = [prepare(r) for r in rows]
    first_tokens = []
    prefill_results = []
    for slot, (audio_values, tokens, true_length) in enumerate(prepared):
      rng, rng_prefill = jax.random.split(rng)
      prefill_result, first_token = engine.prefill(
          params=params,
          padded_tokens=tokens,
          audio_values=audio_values,
          audio_masks=None,
          true_length=true_length,
          rng=rng_prefill,
          slot=slot,
      )
      prefill_results.append(prefill_result)
      first_tokens.append(int(first_token.get_result_at_slot(0).tokens.item()))
    rng, rng_init = jax.random.split(rng)
    decode_state = engine.init_decode_state(rng_init)
    for slot, prefill_result in enumerate(prefill_results):
      decode_state = engine.insert(prefill_result, decode_state, slot=slot)

    outputs = [[t] for t in first_tokens]
    done = [t in END_OF_TURN_IDS for t in first_tokens]
    for _ in range(max_gen_steps - 1):
      if all(done):
        break
      rng, rng_gen = jax.random.split(rng)
      decode_state, sampled = engine.generate(params, decode_state, rng=rng_gen)
      sampled_np = np.asarray(sampled.data)  # [batch, ...]; column tokens_idx[0] holds the token
      for slot in range(len(rows)):
        if done[slot]:
          continue
        tok = int(sampled_np[slot, sampled.tokens_idx[0]])
        outputs[slot].append(tok)
        if tok in END_OF_TURN_IDS:
          done[slot] = True
    for slot, row in enumerate(rows[:num_real]):
      ids = outputs[slot]
      stopped = ids[-1] in END_OF_TURN_IDS
      text_ids = [t for t in ids if t not in END_OF_TURN_IDS]
      hyp = tokenizer_model.decode(text_ids).strip()
      metrics.add(row["text"], hyp, stopped)
      out.write(
          json.dumps(
              {
                  "segment_id": row["segment_id"],
                  "duration": float(row["duration"]),
                  "ref": row["text"],
                  "hyp": hyp,
                  "stopped": stopped,
                  "num_tokens": len(ids),
              },
              ensure_ascii=False,
          )
          + "\n"
      )

  for row in iter_examples(args.eval_files, args.max_examples):
    batch.append(row)
    if len(batch) == batch_size:
      run_batch(batch, len(batch))
      batch, num_batches = [], num_batches + 1
      if num_batches % args.log_every == 0:
        s = metrics.summary()
        max_logging.log(
            f"[{s['num_examples']} ex, {time.time() - start:.0f}s] WER {s['wer']:.4f} CER {s['cer']:.4f} "
            f"EM {s['exact_match']:.3f} rawEM {s['raw_exact_match']:.3f} format_ok {s['format_ok']:.3f}"
        )
        out.flush()
  if batch:  # final partial batch: fill the unused slots with copies of the last row, score only the real ones
    run_batch(batch + [batch[-1]] * (batch_size - len(batch)), len(batch))
  out.close()

  summary = metrics.summary()
  summary.update(
      {
          "eval_files": args.eval_files,
          "load_parameters_path": config.load_parameters_path,
          "prompt": config.prompt,
          "seconds": time.time() - start,
      }
  )
  with fsspec.open(args.output + ".summary.json", "w") as f:
    json.dump(summary, f, indent=2)
  max_logging.log("SUMMARY " + json.dumps(summary))


if __name__ == "__main__":
  main(sys.argv)
