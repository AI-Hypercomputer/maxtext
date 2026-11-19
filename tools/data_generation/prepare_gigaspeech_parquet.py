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

"""Re-shard a GigaSpeech (speechcolab/gigaspeech) parquet split into audio-SFT parquet files on GCS.

The HF repo already provides auto-converted parquet files (`parquet-data/<subset>/<split>-*.parquet`)
whose `audio` column holds 16 kHz WAV bytes. This tool streams them one input file at a time:

  1. downloads (or reads) the file,
  2. normalizes the transcript (`<COMMA>` -> `,`, garbage-only utterances dropped, lower-cased with
     sentence capitalization),
  3. drops segments longer than `--max_duration` seconds (the training pipeline pads audio to a static
     length, see `audio_max_mel_frames`),
  4. appends rows to output shards of `--rows_per_shard` rows with columns `audio` (struct<bytes, path>),
     `text`, `duration`, `segment_id` under `--output_dir` (local or `gs://`),
  5. optionally deletes the downloaded input (`--delete_inputs`) so only one ~4 GB file is on disk at a time.

Example (s train split, 2000-row shards, ~115 files):

  HF_HOME=/dev/shm/$USER python tools/data_generation/prepare_gigaspeech_parquet.py \
      --subset s --split train --max_duration 20 --rows_per_shard 2000 --delete_inputs \
      --cache_dir /dev/shm/gigaspeech_raw --output_dir gs://my-bucket/data/gigaspeech_s

The resulting files are consumed with `hf_path=parquet hf_train_files=gs://my-bucket/data/gigaspeech_s/train-*.parquet`
(see `src/maxtext/configs/post_train/sft-audio-gigaspeech.yml`).
"""

import argparse
import os
import re

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq

_PUNCT_TAGS = {
    "<COMMA>": ",",
    "<PERIOD>": ".",
    "<QUESTIONMARK>": "?",
    "<EXCLAMATIONPOINT>": "!",
}
_GARBAGE_TAGS = ("<SIL>", "<MUSIC>", "<NOISE>", "<OTHER>")

_SCHEMA = pa.schema(
    [
        ("segment_id", pa.string()),
        ("audio", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
        ("text", pa.string()),
        ("duration", pa.float32()),
    ]
)


def normalize_text(text: str) -> str:
  """GigaSpeech transcript -> plain sentence. Returns "" for garbage-only utterances."""
  for tag in _GARBAGE_TAGS:
    text = text.replace(tag, " ")
  for tag, punct in _PUNCT_TAGS.items():
    text = text.replace(f" {tag}", punct).replace(tag, punct)
  text = " ".join(text.split()).lower()
  if not re.search(r"[a-z0-9]", text):
    return ""
  # Capitalize the first letter of the utterance and of every sentence.
  text = re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)
  return re.sub(r"\bi\b", "I", text)


def _input_files(args):
  """Yields (local_path, downloaded) for each input parquet file, downloading lazily."""
  if args.input_files:
    for path in args.input_files:
      yield path, False
    return
  from huggingface_hub import HfApi, hf_hub_download  # pylint: disable=import-outside-toplevel

  prefix = f"parquet-data/{args.subset}/{args.split}-"
  names = sorted(s.rfilename for s in HfApi().dataset_info(args.hf_repo).siblings if s.rfilename.startswith(prefix))
  if not names:
    raise ValueError(f"No files matching {prefix}* in {args.hf_repo}")
  print(f"{len(names)} input files for {args.subset}/{args.split}", flush=True)
  for name in names:
    yield hf_hub_download(args.hf_repo, name, repo_type="dataset", local_dir=args.cache_dir), True


class _ShardWriter:
  """Buffers rows and writes `rows_per_shard`-row parquet files `<split>-<idx:05d>.parquet`."""

  def __init__(self, output_dir, split, rows_per_shard):
    self.output_dir, self.split, self.rows_per_shard = output_dir, split, rows_per_shard
    self.rows = {"segment_id": [], "audio": [], "text": [], "duration": []}
    self.num_shards = 0
    self.num_rows = 0
    fs, _, _ = fsspec.get_fs_token_paths(output_dir)
    fs.makedirs(output_dir, exist_ok=True)

  def add(self, segment_id, audio, text, duration):
    self.rows["segment_id"].append(segment_id)
    self.rows["audio"].append(audio)
    self.rows["text"].append(text)
    self.rows["duration"].append(duration)
    if len(self.rows["text"]) >= self.rows_per_shard:
      self.flush()

  def flush(self):
    """Writes the buffered rows (if any) as the next shard."""
    n = len(self.rows["text"])
    if n == 0:
      return
    out = os.path.join(self.output_dir, f"{self.split}-{self.num_shards:05d}.parquet")
    with fsspec.open(out, "wb") as f:
      pq.write_table(pa.Table.from_pydict(self.rows, schema=_SCHEMA), f, row_group_size=256)
    self.num_shards += 1
    self.num_rows += n
    self.rows = {k: [] for k in self.rows}
    print(f"wrote {out} ({n} rows, {self.num_rows} total)", flush=True)


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--hf_repo", default="speechcolab/gigaspeech")
  parser.add_argument("--subset", default="xs", help="GigaSpeech subset: xs, s, m, l, xl.")
  parser.add_argument("--split", default="train", help="train, validation or test.")
  parser.add_argument("--input_files", nargs="*", default=None, help="Local parquet files (skips the HF download).")
  parser.add_argument("--cache_dir", default="/dev/shm/gigaspeech_raw", help="Where the HF parquet files are downloaded.")
  parser.add_argument("--delete_inputs", action="store_true", help="Delete each downloaded input file after processing.")
  parser.add_argument("--output_dir", required=True, help="Local or gs:// directory for the output shards.")
  parser.add_argument("--rows_per_shard", type=int, default=2000)
  parser.add_argument("--max_duration", type=float, default=20.0, help="Drop segments longer than this (seconds).")
  parser.add_argument("--max_rows", type=int, default=0, help="Optional cap on the number of rows (0 = all).")
  args = parser.parse_args()

  writer = _ShardWriter(args.output_dir, args.split, args.rows_per_shard)
  num_dropped_len, num_dropped_text, num_seen = 0, 0, 0
  done = False
  for path, downloaded in _input_files(args):
    pf = pq.ParquetFile(path)
    for rg in range(pf.metadata.num_row_groups):
      table = pf.read_row_group(rg, columns=["segment_id", "text", "audio", "begin_time", "end_time"])
      for row in table.to_pylist():
        num_seen += 1
        duration = float(row["end_time"]) - float(row["begin_time"])
        if duration > args.max_duration:
          num_dropped_len += 1
          continue
        text = normalize_text(row["text"])
        if not text:
          num_dropped_text += 1
          continue
        writer.add(row["segment_id"], row["audio"], text, duration)
        if args.max_rows and writer.num_rows + len(writer.rows["text"]) >= args.max_rows:
          done = True
          break
      print(f"{os.path.basename(path)} row group {rg}: {num_seen} rows seen", flush=True)
      if done:
        break
    if downloaded and args.delete_inputs:
      os.remove(path)
    if done:
      break
  writer.flush()
  print(
      f"kept {writer.num_rows} rows in {writer.num_shards} shards, dropped {num_dropped_len} (> {args.max_duration}s)"
      f" + {num_dropped_text} (empty text)"
  )


if __name__ == "__main__":
  main()
