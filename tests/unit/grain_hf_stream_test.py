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

"""Tests for `grain_file_type=hf`: Hugging Face streaming datasets in the Grain pipeline."""

import collections
import csv
import gzip
import json
import os
import shutil
import sys
import tempfile
import unittest
from types import SimpleNamespace

import grain.python as grain
import jax
import numpy as np
import pydantic
import pytest
from jax.experimental import mesh_utils
from jax.sharding import Mesh

from maxtext.configs import pyconfig
from maxtext.input_pipeline import data_processing_utils
from maxtext.input_pipeline import grain_data_processing
from maxtext.input_pipeline import hf_data_processing
from maxtext.input_pipeline import input_pipeline_interface
from maxtext.input_pipeline import input_pipeline_utils
from maxtext.utils.globals import MAXTEXT_ASSETS_ROOT
from tests.utils.test_helpers import get_test_config_path

pytestmark = pytest.mark.skipif(
    not hasattr(grain.experimental, "HFIterDataset"), reason="grain_file_type=hf requires grain>=0.2.19"
)

_ROWS_PER_FILE = 10
_NUM_FILES = 8
_QWEN3_TOKENIZER = os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizers", "qwen3-tokenizer")


def _get_id(x):
  return int(x["id"])


def _write_jsonl(path, rows, compress=False):
  opener = gzip.open if compress else open
  with opener(path, "wt", encoding="utf-8") as f:
    for row in rows:
      f.write(json.dumps(row) + "\n")


def _write_id_files(directory, suffixes=(".jsonl",)):
  """Writes `_NUM_FILES` JSON-lines files with globally unique `id`s, cycling through `suffixes`."""
  files = []
  for f in range(_NUM_FILES):
    suffix = suffixes[f % len(suffixes)]
    path = os.path.join(directory, f"part-{f:02d}{suffix}")
    rows = [{"id": f * _ROWS_PER_FILE + i, "text": f"document {f} row {i} " * 8} for i in range(_ROWS_PER_FILE)]
    _write_jsonl(path, rows, compress=suffix.endswith(".gz"))
    files.append(path)
  return files


def _ids(ds):
  return [_get_id(x) for x in ds]


class MakeHfStreamingIterDatasetTest(unittest.TestCase):
  """Source-level tests for `input_pipeline_utils.make_hf_streaming_iter_dataset`."""

  def setUp(self):
    super().setUp()
    self.tmp = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self.tmp)
    # .json files containing JSON lines are common on the Hub; HF's json builder handles them.
    self.files = _write_id_files(self.tmp, suffixes=(".jsonl", ".jsonl.gz", ".json"))
    self.all_ids = list(range(_NUM_FILES * _ROWS_PER_FILE))

  def _make(self, **kwargs):
    kwargs.setdefault("data_files", self.files)
    return input_pipeline_utils.make_hf_streaming_iter_dataset("json", **kwargs)

  def test_reads_jsonl_gzip_and_json_lines_files(self):
    self.assertEqual(sorted(_ids(self._make())), self.all_ids)

  def test_glob_data_files(self):
    ds = self._make(data_files=os.path.join(self.tmp, "part-*"))
    self.assertEqual(sorted(_ids(ds)), self.all_ids)

  def test_csv_and_text_builders(self):
    csv_path = os.path.join(self.tmp, "data.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
      writer = csv.DictWriter(f, fieldnames=["id", "text"])
      writer.writeheader()
      writer.writerows({"id": i, "text": f"row {i}"} for i in range(5))
    txt_path = os.path.join(self.tmp, "data.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
      f.write("\n".join(f"line {i}" for i in range(5)) + "\n")

    csv_rows = list(input_pipeline_utils.make_hf_streaming_iter_dataset("csv", data_files=csv_path))
    self.assertEqual([int(r["id"]) for r in csv_rows], list(range(5)))
    txt_rows = list(input_pipeline_utils.make_hf_streaming_iter_dataset("text", data_files=txt_path))
    self.assertEqual([r["text"] for r in txt_rows], [f"line {i}" for i in range(5)])

  def test_num_epoch_repeats(self):
    ids = _ids(self._make(num_epoch=2))
    self.assertEqual(collections.Counter(ids), collections.Counter(self.all_ids * 2))

  def test_host_sharding_is_disjoint_and_complete(self):
    # 8 files over 2 or 4 hosts splits by file; over 3 hosts splits by example.
    for host_count in (2, 3, 4):
      with self.subTest(host_count=host_count):
        per_host = [
            _ids(self._make(dataloading_host_index=h, dataloading_host_count=host_count)) for h in range(host_count)
        ]
        flat = [i for ids in per_host for i in ids]
        self.assertEqual(sorted(flat), self.all_ids)
        self.assertEqual(len(flat), len(set(flat)))

  def test_mp_prefetch_reads_each_element_once(self):
    for workers in (1, 2, 4, 8):
      with self.subTest(workers=workers):
        ds = self._make(grain_worker_count=workers).map(_get_id)
        ds = ds.mp_prefetch(grain.MultiprocessingOptions(num_workers=workers))
        ids = list(ds)
        self.assertEqual(sorted(ids), self.all_ids)

  def test_mp_prefetch_with_host_sharding(self):
    per_host = []
    for h in range(2):
      ds = self._make(dataloading_host_index=h, dataloading_host_count=2, grain_worker_count=2).map(_get_id)
      per_host.append(list(ds.mp_prefetch(grain.MultiprocessingOptions(num_workers=2))))
    flat = [i for ids in per_host for i in ids]
    self.assertEqual(sorted(flat), self.all_ids)

  def test_worker_count_exceeding_shards_raises(self):
    with self.assertRaisesRegex(ValueError, "grain_worker_count"):
      self._make(dataloading_host_index=0, dataloading_host_count=2, grain_worker_count=_NUM_FILES)

  def test_checkpoint_round_trip(self):
    it = iter(self._make())
    consumed = [_get_id(next(it)) for _ in range(13)]
    state = it.get_state()
    json.dumps(state)  # Grain checkpoints iterator state as JSON.
    expected_rest = [_get_id(x) for x in it]

    restored = iter(self._make())
    restored.set_state(state)
    self.assertEqual([_get_id(x) for x in restored], expected_rest)
    self.assertEqual(sorted(consumed + expected_rest), self.all_ids)

  def test_checkpoint_round_trip_with_mp_prefetch(self):
    def make_it():
      ds = self._make(grain_worker_count=2).map(_get_id)
      return iter(ds.mp_prefetch(grain.MultiprocessingOptions(num_workers=2)))

    it = make_it()
    consumed = [next(it) for _ in range(21)]
    state = it.get_state()
    expected_rest = list(it)

    restored = make_it()
    restored.set_state(state)
    self.assertEqual(list(restored), expected_rest)
    self.assertEqual(sorted(consumed + expected_rest), self.all_ids)


class GetDatasetsHfTest(unittest.TestCase):
  """`grain_data_processing.get_datasets` with `grain_file_type=hf`."""

  def setUp(self):
    super().setUp()
    self.tmp = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self.tmp)
    self.files = _write_id_files(self.tmp)

  def _get(self, shuffle):
    return grain_data_processing.get_datasets(
        data_file_pattern="",
        data_file_type="hf",
        shuffle=shuffle,
        shuffle_seed=0,
        shuffle_buffer_size=16,
        num_epoch=1,
        dataloading_host_index=0,
        dataloading_host_count=1,
        grain_worker_count=1,
        grain_num_threads=1,
        grain_prefetch_buffer_size=1,
        grain_data_source_max_workers=1,
        hf_dataset_kwargs={"path": "json", "data_files": self.files, "split": "train"},
    )

  def test_shuffle_is_deterministic_permutation(self):
    unshuffled = _ids(self._get(shuffle=False))
    shuffled = _ids(self._get(shuffle=True))
    self.assertEqual(sorted(shuffled), sorted(unshuffled))
    self.assertNotEqual(shuffled, unshuffled)
    self.assertEqual(shuffled, _ids(self._get(shuffle=True)))

  def test_hf_dataset_kwargs_from_config(self):
    config = SimpleNamespace(
        grain_file_type="hf",
        hf_path="json",
        hf_name="",
        hf_data_dir="",
        hf_train_files="train/*.jsonl",
        hf_eval_files="eval/*.jsonl",
        hf_eval_split="",
        train_split="train",
    )
    train = grain_data_processing._get_hf_dataset_kwargs(config, "train")  # pylint: disable=protected-access
    self.assertEqual(train["data_files"], "train/*.jsonl")
    self.assertEqual(train["split"], "train")
    eval_kwargs = grain_data_processing._get_hf_dataset_kwargs(config, "eval")  # pylint: disable=protected-access
    self.assertEqual(eval_kwargs["data_files"], "eval/*.jsonl")
    self.assertEqual(eval_kwargs["split"], "train")
    config.hf_eval_files, config.hf_eval_split = "", "validation"
    eval_kwargs = grain_data_processing._get_hf_dataset_kwargs(config, "eval")  # pylint: disable=protected-access
    self.assertEqual(eval_kwargs["split"], "validation")
    config.grain_file_type = "parquet"
    self.assertIsNone(grain_data_processing._get_hf_dataset_kwargs(config, "train"))  # pylint: disable=protected-access


class _GrainHfConfigMixin:
  """Builds pyconfig + mesh for a `dataset_type=grain grain_file_type=hf` run on local JSON files."""

  def _init(self, use_pydantic=False, **overrides):
    """Returns (config, mesh, process_indices) for a local `grain_file_type=hf` run with `overrides`."""
    kwargs = {
        "per_device_batch_size": 1,
        "run_name": "test",
        "mesh_axes": ["data"],
        "logical_axis_rules": [["batch", "data"]],
        "data_sharding": ["data"],
        "base_output_directory": self.tmp,
        "dataset_type": "grain",
        "grain_file_type": "hf",
        "hf_path": "json",
        "grain_worker_count": 1,
        "grain_per_worker_buffer_size": 1,
        "tokenizer_type": "huggingface",
        "tokenizer_path": _QWEN3_TOKENIZER,
        "enable_checkpointing": False,
        "max_target_length": 128,
        "eval_interval": -1,
        **overrides,
    }
    # The DPO trainer uses the pydantic config, whose nested `dpo` section supports attribute access.
    init_fn = pyconfig.initialize_pydantic if use_pydantic else pyconfig.initialize
    config = init_fn([sys.argv[0], get_test_config_path()], **kwargs)
    mesh = Mesh(mesh_utils.create_device_mesh((len(jax.devices()),)), config.mesh_axes)
    process_indices = input_pipeline_interface.get_process_loading_real_data(
        config.data_sharding,
        config.global_batch_size_to_load,
        config.global_batch_size_to_train_on,
        config.max_target_length,
        mesh,
    )
    return config, mesh, process_indices

  def _first_batch(self, **overrides):
    config, mesh, process_indices = self._init(**overrides)
    return next(grain_data_processing.make_grain_train_iterator(config, mesh, process_indices)), config


_PACKED_KEYS = {"inputs", "inputs_position", "inputs_segmentation", "targets", "targets_position", "targets_segmentation"}


class GrainHfPretrainTest(_GrainHfConfigMixin, unittest.TestCase):
  """End-to-end pretraining batches from `grain_file_type=hf`."""

  def setUp(self):
    super().setUp()
    self.tmp = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self.tmp)
    _write_id_files(self.tmp, suffixes=(".jsonl", ".jsonl.gz"))
    self.train_files = os.path.join(self.tmp, "part-*")

  def test_train_batch_shapes_and_determinism(self):
    batch1, config = self._first_batch(hf_train_files=self.train_files)
    expected_shape = [jax.device_count(), config.max_target_length]
    self.assertEqual({k: list(v.shape) for k, v in batch1.items()}, {k: expected_shape for k in _PACKED_KEYS})
    batch2, _ = self._first_batch(hf_train_files=self.train_files)
    for k in _PACKED_KEYS:
      np.testing.assert_array_equal(batch1[k], batch2[k])

  def test_multiple_workers(self):
    batch, config = self._first_batch(hf_train_files=self.train_files, grain_worker_count=2)
    self.assertEqual(list(batch["inputs"].shape), [jax.device_count(), config.max_target_length])

  def test_eval_iterator(self):
    config, mesh, process_indices = self._init(
        hf_train_files=self.train_files, hf_eval_files=self.train_files, eval_interval=10, eval_steps=1
    )
    batch = next(grain_data_processing.make_grain_eval_iterator(config, mesh, process_indices))
    self.assertEqual(list(batch["inputs"].shape), [jax.device_count(), config.max_target_length])


class GrainHfSftTest(_GrainHfConfigMixin, unittest.TestCase):
  """SFT from `grain_file_type=hf`, including formatting functions and parity with `dataset_type=hf`."""

  def setUp(self):
    super().setUp()
    self.tmp = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self.tmp)
    self.messages_file = os.path.join(self.tmp, "messages.jsonl")
    _write_jsonl(
        self.messages_file,
        [
            {
                "messages": [
                    {"role": "user", "content": f"What is {i} plus {i}?"},
                    {"role": "assistant", "content": f"{i} plus {i} is {2 * i}."},
                ],
                "extra": i,
            }
            for i in range(32)
        ],
    )
    self.qa_file = os.path.join(self.tmp, "qa.jsonl")
    _write_jsonl(self.qa_file, [{"question": f"Q{i}?", "answer": f"A{i}."} for i in range(32)])

  def _sft_overrides(self, **overrides):
    return {
        "use_sft": True,
        "sft_train_on_completion_only": True,
        "train_data_columns": ["messages"],
        "hf_train_files": self.messages_file,
        "enable_data_shuffling": False,
        **overrides,
    }

  def test_messages_with_prompt_masking(self):
    batch, config = self._first_batch(**self._sft_overrides())
    expected_shape = [jax.device_count(), config.max_target_length]
    self.assertEqual({k: list(v.shape) for k, v in batch.items()}, {k: expected_shape for k in _PACKED_KEYS})
    self.assertTrue(np.any(batch["inputs"] != batch["targets"]), "Prompt tokens should be masked in targets.")

  def test_formatting_func(self):
    batch, _ = self._first_batch(
        **self._sft_overrides(
            train_data_columns=["question", "answer"],
            hf_train_files=self.qa_file,
            formatting_func_path="maxtext.input_pipeline.instruction_data_processing.math_qa_formatting",
        )
    )
    self.assertEqual(set(batch), _PACKED_KEYS)

  def test_inputs_match_dataset_type_hf(self):
    """Without packing or shuffling, the first SFT batch matches the existing HF pipeline."""
    overrides = self._sft_overrides(packing=False)
    grain_batch, _ = self._first_batch(**overrides)

    config, mesh, process_indices = self._init(
        **{**overrides, "dataset_type": "hf", "hf_train_files": self.messages_file}
    )
    hf_batch = next(hf_data_processing.make_hf_train_iterator(config, mesh, process_indices))
    np.testing.assert_array_equal(grain_batch["inputs"], hf_batch["inputs"])
    np.testing.assert_array_equal(grain_batch["inputs_segmentation"], hf_batch["inputs_segmentation"])


class GrainHfDpoTest(_GrainHfConfigMixin, unittest.TestCase):
  """DPO from `grain_file_type=hf` using the checked-in DPO JSON fixtures."""

  def setUp(self):
    super().setUp()
    self.tmp = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self.tmp)

  def test_three_column_dpo_from_json_array_file(self):
    # A regular JSON array file (not JSON lines); HF's json builder handles both.
    data_file = os.path.join("tests", "assets", "local_datasets", "dpo", "dpo_3_column_dataset.json")
    batch, config = self._first_batch(
        use_pydantic=True,
        use_dpo=True,
        train_data_columns=["prompt", "chosen", "rejected"],
        hf_train_files=data_file,
        enable_data_shuffling=False,
        packing=False,
        max_target_length=64,
    )
    keys = {"prompt_ids", "prompt_mask", "chosen_ids", "chosen_mask", "rejected_ids", "rejected_mask"}
    self.assertEqual(set(batch), keys)
    prompt_len, response_len = config.dpo.max_prompt_length, config.max_target_length - config.dpo.max_prompt_length
    for k in keys:
      expected_len = prompt_len if k.startswith("prompt") else response_len
      self.assertEqual(list(batch[k].shape), [jax.device_count(), expected_len], k)
    self.assertTrue(np.all(batch["chosen_mask"].sum(axis=1) > 0))


class VisionSftElementHelpersTest(unittest.TestCase):
  """Element-wise ShareGPT and default-prompt helpers used by Grain vision SFT."""

  def test_maybe_format_sharegpt(self):
    element = {"conversations": [{"from": "human", "value": "describe"}, {"from": "gpt", "value": "a cat"}], "video": 1}
    out = input_pipeline_utils.maybe_format_sharegpt(dict(element), text_columns=["query", "response"])
    self.assertEqual(out["query"], "describe")
    self.assertEqual(out["response"], ["a cat"])
    already = {"query": "q", "conversations": []}
    self.assertIs(input_pipeline_utils.maybe_format_sharegpt(already, text_columns=["query", "response"]), already)

  def test_add_default_prompt_to_element(self):
    self.assertEqual(
        input_pipeline_utils.add_default_prompt_to_element({"captions": "c"}, "prompt", "Describe.")["prompt"],
        "Describe.",
    )
    self.assertEqual(
        input_pipeline_utils.add_default_prompt_to_element({"prompt": "p"}, "prompt", "Describe.")["prompt"], "p"
    )


class FormatAndBatchTargetTransformTest(unittest.TestCase):
  """`format_and_batch(target_transform=...)` replaces the causal shift (used for block diffusion)."""

  def test_target_transform_replaces_shift(self):
    calls = []

    def transform(batch):
      calls.append(True)
      return batch

    config = SimpleNamespace(packing=False, max_target_length=4, grain_use_elastic_iterator=False)
    rows = [{"inputs": np.arange(1, 4, dtype=np.int32), "targets": np.arange(1, 4, dtype=np.int32)}] * 2
    ds = grain.MapDataset.source(rows).to_iter_dataset()
    ds = data_processing_utils.format_and_batch(
        ds,
        config,
        batch_size=2,
        pad_id=0,
        data_columns=("inputs", "targets"),
        tokenizer_model=None,
        target_transform=transform,
    )
    batch = next(iter(ds))
    self.assertTrue(calls)
    # Unshifted: targets still equal inputs.
    np.testing.assert_array_equal(batch["inputs"], batch["targets"])

  def test_elastic_rejects_target_transform(self):
    config = SimpleNamespace(packing=False, max_target_length=4, grain_use_elastic_iterator=True)
    ds = grain.MapDataset.source([{"inputs": np.ones(2, np.int32), "targets": np.ones(2, np.int32)}])
    with self.assertRaisesRegex(ValueError, "ElasticIterator"):
      data_processing_utils.format_and_batch(ds, config, 1, 0, ("inputs", "targets"), None, target_transform=lambda x: x)


class GrainHfConfigTest(unittest.TestCase):
  """Config validation for `grain_file_type=hf`."""

  def _init(self, **overrides):
    kwargs = {"run_name": "test", "dataset_type": "grain", "grain_file_type": "hf", "eval_interval": -1, **overrides}
    return pyconfig.initialize([sys.argv[0], get_test_config_path()], **kwargs)

  def test_hf_path_is_a_valid_train_source(self):
    config = self._init(hf_path="json", hf_train_files="/tmp/*.jsonl")
    self.assertEqual(config.grain_file_type, "hf")

  def test_requires_hf_path(self):
    with self.assertRaisesRegex((ValueError, pydantic.ValidationError), "hf_path"):
      self._init(hf_path="")

  def test_eval_files_set_eval_split(self):
    config = self._init(hf_path="json", hf_train_files="/tmp/*.jsonl", hf_eval_files="/tmp/e.jsonl", eval_interval=10)
    self.assertEqual(config.hf_eval_split, "train")

  def test_eval_requires_split_or_files(self):
    with self.assertRaisesRegex((ValueError, pydantic.ValidationError), "hf_eval_split"):
      self._init(hf_path="json", eval_interval=10)


if __name__ == "__main__":
  unittest.main()
