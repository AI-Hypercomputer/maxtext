# Copyright 2023–2025 Google LLC
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

""" Train tokenizer
Example usage (parquet):
  python3 -m MaxText.train_tokenizer \
    --grain_train_files=gs://my-bucket/data/*.parquet \
    --grain_file_type=parquet

Example usage (arrayrecord):
  python3 -m MaxText.train_tokenizer \
    --grain_train_files=gs://my-bucket/data/*.arrayrecord \
    --grain_file_type=arrayrecord \
    --data_column=text
"""

import glob
import os
import shutil
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path

from absl import app
from absl import flags
from absl import logging

from sentencepiece import SentencePieceTrainer

import jax
import grain.python as grain

from maxtext.input_pipeline import input_pipeline_utils
from maxtext.utils.globals import MAXTEXT_ASSETS_ROOT
from maxtext.utils import gcs_utils


_GRAIN_TRAIN_FILES = flags.DEFINE_string(
    "grain_train_files", None, "File pattern for training data (local or gs://)", required=True
)
_GRAIN_FILE_TYPE = flags.DEFINE_string(
    "grain_file_type", "parquet", "Type of data files. Supported: 'parquet', 'arrayrecord', 'tfrecord'."
)
_DATA_COLUMN = flags.DEFINE_string("data_column", "text", "Column name to extract text from (used for arrayrecord).")
_VOCAB_SIZE = flags.DEFINE_integer("vocab_size", 32_768, "Vocab size")
_MAX_CORPUS_CHARS = flags.DEFINE_integer("max_corpus_chars", 10_000_000, "Max corpus chars")
_ASSETS_PATH = flags.DEFINE_string("assets_path", MAXTEXT_ASSETS_ROOT, "Path to assets directory")
_VOCAB_MODEL_NAME = flags.DEFINE_string("vocab_model_name", "tokenizer", "Output tokenizer model name")


def build_grain_iterator(data_file_pattern: str, data_file_type: str, data_keys: tuple[str, ...] = ("text",)) -> Iterator:
  """Build a grain iterator from a file pattern for tokenizer training.

  Args:
    data_file_pattern: Glob pattern for data files (local path or gs://).
    data_file_type: One of 'arrayrecord' or 'parquet'.
    data_keys: Column names to extract from each example (used for arrayrecord).

  Returns:
    A Python iterator yielding examples as dicts.
  """
  if data_file_pattern.startswith("gs://"):
    data_files = gcs_utils.gcs_glob_pattern(data_file_pattern)
  else:
    data_files = glob.glob(str(Path(data_file_pattern).expanduser().resolve()))
  if not data_files:
    raise FileNotFoundError(f"No files found matching pattern: {data_file_pattern}")
  logging.info("Found %d files for tokenizer training.", len(data_files))

  if data_file_type == "parquet":
    dataset = grain.MapDataset.source(data_files)
    dataset = dataset.map(grain.experimental.ParquetIterDataset)
    dataset = grain.experimental.InterleaveIterDataset(dataset, cycle_length=len(data_files))
    dataset = dataset.map(input_pipeline_utils.KeepFeatures(feature_names=list(data_keys)))
    return iter(dataset)
  elif data_file_type == "arrayrecord":
    source = grain.ArrayRecordDataSource(data_files)
    dataset = grain.MapDataset.source(source)
    dataset = dataset.map(input_pipeline_utils.ParseFeatures(list(data_keys), tokenize=True))
    dataset = dataset.map(input_pipeline_utils.NormalizeFeatures(list(data_keys), tokenize=True))
    return iter(dataset)
  elif data_file_type == "tfrecord":
    dataset = grain.MapDataset.source(data_files)
    dataset = dataset.map(input_pipeline_utils.make_tfrecord_iter_dataset)
    dataset = grain.experimental.InterleaveIterDataset(dataset, cycle_length=len(data_files))
    dataset = dataset.map(input_pipeline_utils.ParseFeatures(list(data_keys), tokenize=True))
    dataset = dataset.map(input_pipeline_utils.NormalizeFeatures(list(data_keys), tokenize=True))
    return iter(dataset)
  else:
    raise ValueError(f"Unsupported grain_file_type: {data_file_type!r}. Use 'parquet', 'arrayrecord', or 'tfrecord'.")


def _dump_chars_to_textfile(dataset_iter: Iterator, maxchars: int = int(1e7), data_keys=("text",)) -> tuple[str, int]:
  """Write part of a grain dataset to lines in a text file.

  Args:
    dataset_iter: Iterator yielding examples as dicts.
    maxchars: Approximate number of characters to save from dataset.
    data_keys: Keys in each example to dump.

  Returns:
    Name of temp file with dataset bytes, exact number of characters dumped.
  """
  char_count = 0
  temp_dir = tempfile.gettempdir()
  with tempfile.NamedTemporaryFile(
      delete=False, prefix=os.path.join(temp_dir, "ds_chars"), mode="w", encoding="utf-8"
  ) as outfp:
    while char_count < maxchars:
      example = next(dataset_iter)
      for k in data_keys:
        val = example[k]
        if isinstance(val, bytes):
          val = val.decode("utf-8")
        line = val + "\n"
        char_count += len(line)
        outfp.write(line)
  return outfp.name, char_count


def _train_sentencepiece(
    dataset_iter: Iterator,
    *,
    vocab_size: int,
    maxchars: int = int(1e7),
    model_path: str,
    model_type: str = "unigram",
    character_coverage: float = 1.0,
    data_keys=("text",),
):
  """Train SentencePiece tokenizer from subset of a grain dataset.

  Args:
    dataset_iter: Iterator yielding examples as dicts.
    vocab_size: Size of vocab tokens to train.
    maxchars: Number of characters to use for sentencepiece training.
    model_path: Path to save vocab model to (local or gs://).
    model_type: Type of sentencepiece vocab to train.
    character_coverage: Amount of characters covered by the model.
    data_keys: Keys of dataset to use for training.

  Returns:
    Path to the trained sentencepiece vocabulary model.
  """
  if model_path.startswith("gs://"):
    abs_model_path = model_path
  else:
    abs_model_path = os.path.abspath(os.path.expanduser(model_path))
  fname, _ = _dump_chars_to_textfile(dataset_iter, maxchars=maxchars, data_keys=data_keys)
  temp_dir = tempfile.gettempdir()
  with tempfile.NamedTemporaryFile(delete=False, prefix=os.path.join(temp_dir, "sp_tmp")) as model_fp:
    pass  # we just want a prefix'd tmp-filename
  argstr = " ".join(
      [
          f"--input={fname}",
          f"--vocab_size={vocab_size}",
          f"--character_coverage={character_coverage}",
          f"--model_prefix={model_fp.name}",
          f"--model_type={model_type}",
      ]
  )
  SentencePieceTrainer.Train(argstr)
  if jax.process_index() == 0:
    if abs_model_path.startswith("gs://"):
      gcs_utils.upload_blob(abs_model_path, model_fp.name + ".model")
      logging.info("Uploaded %s to %s", model_fp.name + ".model", abs_model_path)
    else:
      parent = os.path.dirname(abs_model_path)
      if parent:
        os.makedirs(parent, exist_ok=True)
      shutil.copy(model_fp.name + ".model", abs_model_path)
      logging.info("Copied %s to %s", model_fp.name + ".model", abs_model_path)
  else:
    if abs_model_path.startswith("gs://"):
      while not gcs_utils.gcs_path_exists(abs_model_path):
        time.sleep(1)
    else:
      while not os.path.exists(abs_model_path):
        time.sleep(1)
    time.sleep(1)
  return abs_model_path


def train_tokenizer(
    dataset_iter: Iterator,
    *,
    vocab_path: str,
    vocab_size: int,
    max_corpus_chars: int,
    data_keys: tuple[str] = ("text",),
):
  """Tokenizer training function."""
  logging.info("SentencePiece vocab not found, building one from data.")
  vocab_path = _train_sentencepiece(
      dataset_iter,
      vocab_size=vocab_size,
      maxchars=max_corpus_chars,
      model_path=vocab_path,
      data_keys=data_keys,
  )
  logging.info("Model saved at %s", vocab_path)


def main(argv):
  del argv
  data_keys = (_DATA_COLUMN.value,)
  dataset_iter = build_grain_iterator(_GRAIN_TRAIN_FILES.value, _GRAIN_FILE_TYPE.value, data_keys=data_keys)
  train_tokenizer(
      dataset_iter,
      vocab_path=os.path.join(_ASSETS_PATH.value, _VOCAB_MODEL_NAME.value),
      vocab_size=_VOCAB_SIZE.value,
      max_corpus_chars=_MAX_CORPUS_CHARS.value,
      data_keys=data_keys,
  )


if __name__ == "__main__":
  app.run(main)
