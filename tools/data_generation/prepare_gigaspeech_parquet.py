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
whose `audio` column holds 16 kHz WAV bytes. This tool:

  1. downloads (or reads) those files,
  2. normalizes the transcript (`<COMMA>` -> `,`, garbage-only utterances dropped, lower-cased with
     sentence capitalization),
  3. drops segments longer than `--max_duration` seconds (the training pipeline pads audio to a static
     length, see `audio_max_mel_frames`),
  4. writes `--num_shards` parquet files with columns `audio` (struct<bytes, path>), `text`, `duration`,
     `segment_id` to `--output_dir` (local or `gs://`).

Example (xs train split, 8 shards):

  HF_HOME=/dev/shm/$USER python tools/data_generation/prepare_gigaspeech_parquet.py \
      --subset xs --split train --max_duration 20 --num_shards 8 \
      --output_dir gs://my-bucket/data/gigaspeech_xs

The resulting files are consumed with `hf_path=parquet hf_train_files=gs://my-bucket/data/gigaspeech_xs/train-*.parquet`
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


def _list_input_files(args) -> list[str]:
  if args.input_files:
    return args.input_files
  from huggingface_hub import HfApi, hf_hub_download  # pylint: disable=import-outside-toplevel

  prefix = f"parquet-data/{args.subset}/{args.split}-"
  names = sorted(s.rfilename for s in HfApi().dataset_info(args.hf_repo).siblings if s.rfilename.startswith(prefix))
  if not names:
    raise ValueError(f"No files matching {prefix}* in {args.hf_repo}")
  return [hf_hub_download(args.hf_repo, n, repo_type="dataset", local_dir=args.cache_dir) for n in names]


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--hf_repo", default="speechcolab/gigaspeech")
  parser.add_argument("--subset", default="xs", help="GigaSpeech subset: xs, s, m, l, xl.")
  parser.add_argument("--split", default="train", help="train, validation or test.")
  parser.add_argument("--input_files", nargs="*", default=None, help="Local parquet files (skips the HF download).")
  parser.add_argument("--cache_dir", default="/dev/shm/gigaspeech_raw", help="Where the HF parquet files are downloaded.")
  parser.add_argument("--output_dir", required=True, help="Local or gs:// directory for the output shards.")
  parser.add_argument("--num_shards", type=int, default=8)
  parser.add_argument("--max_duration", type=float, default=20.0, help="Drop segments longer than this (seconds).")
  parser.add_argument("--max_rows", type=int, default=0, help="Optional cap on the number of rows (0 = all).")
  args = parser.parse_args()

  segment_ids, audios, texts, durations = [], [], [], []
  num_dropped_len, num_dropped_text = 0, 0
  for path in _list_input_files(args):
    pf = pq.ParquetFile(path)
    for rg in range(pf.metadata.num_row_groups):
      table = pf.read_row_group(rg, columns=["segment_id", "text", "audio", "begin_time", "end_time"])
      for row in table.to_pylist():
        duration = float(row["end_time"]) - float(row["begin_time"])
        if duration > args.max_duration:
          num_dropped_len += 1
          continue
        text = normalize_text(row["text"])
        if not text:
          num_dropped_text += 1
          continue
        segment_ids.append(row["segment_id"])
        audios.append(row["audio"])
        texts.append(text)
        durations.append(duration)
      print(f"{os.path.basename(path)} row group {rg}: kept {len(texts)} rows so far", flush=True)
      if args.max_rows and len(texts) >= args.max_rows:
        break
    if args.max_rows and len(texts) >= args.max_rows:
      break
  if args.max_rows:
    segment_ids, audios, texts, durations = (x[: args.max_rows] for x in (segment_ids, audios, texts, durations))

  print(f"kept {len(texts)} rows, dropped {num_dropped_len} (> {args.max_duration}s) + {num_dropped_text} (empty text)")
  schema = pa.schema(
      [
          ("segment_id", pa.string()),
          ("audio", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
          ("text", pa.string()),
          ("duration", pa.float32()),
      ]
  )
  n = len(texts)
  fs, _, _ = fsspec.get_fs_token_paths(args.output_dir)
  fs.makedirs(args.output_dir, exist_ok=True)
  for shard in range(args.num_shards):
    lo, hi = shard * n // args.num_shards, (shard + 1) * n // args.num_shards
    table = pa.Table.from_pydict(
        {
            "segment_id": segment_ids[lo:hi],
            "audio": audios[lo:hi],
            "text": texts[lo:hi],
            "duration": durations[lo:hi],
        },
        schema=schema,
    )
    out = os.path.join(args.output_dir, f"{args.split}-{shard:05d}-of-{args.num_shards:05d}.parquet")
    with fsspec.open(out, "wb") as f:
      pq.write_table(table, f, row_group_size=256)
    print(f"wrote {out} ({hi - lo} rows)", flush=True)


if __name__ == "__main__":
  main()
