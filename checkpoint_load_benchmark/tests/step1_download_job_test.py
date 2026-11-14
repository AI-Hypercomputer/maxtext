# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests checkpoint_load_benchmark/bench/step1_download_job.py with fake HF and fake GCS."""

import io
import json
import os
import shutil
import struct
import sys
import tempfile
import unittest

_BENCHMARK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BENCHMARK_DIR not in sys.path:
  sys.path.insert(0, _BENCHMARK_DIR)

from bench import metadata  # pylint: disable=wrong-import-position
from bench import step1_download_job  # pylint: disable=wrong-import-position


def _make_safetensors_bytes(tensor_name: str = "w", dtype: str = "BF16") -> bytes:
  hdr = json.dumps({tensor_name: {"dtype": dtype, "shape": [2, 2], "data_offsets": [0, 8]}}).encode("utf-8")
  return struct.pack("<Q", len(hdr)) + hdr + (b"\x01" * 8)


class _FakeBlob:
  """In-memory GCS blob implementing the subset used by step1_download_job."""

  def __init__(self, bucket: "_FakeBucket", name: str):
    self._bucket = bucket
    self.name = name

  @property
  def size(self) -> int:
    return len(self._bucket.objects.get(self.name, b""))

  def open(self, mode: str, **_kwargs):
    """Opens an in-memory read or write stream on the fake blob."""
    bucket = self._bucket
    name = self.name
    if "w" in mode:
      buf = io.BytesIO()

      class _WriteCtx:

        def __enter__(self):
          return buf

        def __exit__(self, *_exc):
          bucket.objects[name] = buf.getvalue()

      return _WriteCtx()
    return io.BytesIO(bucket.objects[name])

  def upload_from_string(self, data: str | bytes, content_type: str = "") -> None:
    del content_type
    self._bucket.objects[self.name] = data.encode("utf-8") if isinstance(data, str) else bytes(data)

  def delete(self) -> None:
    self._bucket.objects.pop(self.name, None)


class _FakeBucket:
  """In-memory fake GCS bucket."""

  def __init__(self, name: str):
    self.name = name
    self.objects: dict[str, bytes] = {}

  def blob(self, blob_name: str) -> _FakeBlob:
    return _FakeBlob(self, blob_name)

  def copy_blob(self, blob: _FakeBlob, dst_bucket: "_FakeBucket", new_name: str) -> _FakeBlob:
    dst_bucket.objects[new_name] = blob._bucket.objects[blob.name]  # pylint: disable=protected-access
    return dst_bucket.blob(new_name)


class _FakeStorageClient:
  """In-memory fake google.cloud.storage.Client."""

  def __init__(self):
    self.buckets: dict[str, _FakeBucket] = {}

  def bucket(self, name: str) -> _FakeBucket:
    if name not in self.buckets:
      self.buckets[name] = _FakeBucket(name)
    return self.buckets[name]

  def list_blobs(self, bucket: _FakeBucket | str, prefix: str = ""):
    b = self.bucket(bucket) if isinstance(bucket, str) else bucket
    return [b.blob(k) for k in sorted(b.objects) if k.startswith(prefix)]


class _FakeResponse:
  """Fake streaming requests.Response for HF download tests."""

  def __init__(self, data: bytes, status_code: int = 200):
    self.content = data
    self.raw = io.BytesIO(data)
    self.status_code = status_code
    self.headers = {"Content-Length": str(len(data))}

  def raise_for_status(self) -> None:
    if self.status_code >= 400:
      raise RuntimeError(f"HTTP {self.status_code}")

  def __enter__(self):
    return self

  def __exit__(self, *_exc):
    return None


class Step1DownloadJobTest(unittest.TestCase):
  """Verifies cold Step 1 safety guards, metrics, DeepSeek Step 1b, and regional copy."""

  def setUp(self):
    super().setUp()
    self.tmp_dir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self.tmp_dir)
    self.gcs = _FakeStorageClient()

  def _fake_hf_setup(self, files_map: dict[str, bytes]):
    """Builds fake HF Hub callbacks serving files from `files_map`."""
    repo_files = list(files_map)

    class FakeApi:
      """Fake HuggingFace Hub API client."""

      def list_repo_files(self, repo_id: str):
        """Returns preconfigured fake repository filenames."""
        del repo_id
        return repo_files

    def fake_url(repo_id: str, filename: str) -> str:
      return f"hf://{repo_id}/{filename}"

    def fake_get_bytes(url: str, hdrs: dict[str, str]) -> bytes:
      del hdrs
      fn = url.split("/", 4)[-1]
      return files_map[fn]

    def fake_requests_get(url: str, **_kwargs):
      fn = url.split("/", 4)[-1]
      if fn not in files_map:
        return _FakeResponse(b"", status_code=404)
      return _FakeResponse(files_map[fn])

    return lambda _tok: FakeApi(), fake_get_bytes, fake_url, fake_requests_get

  def test_cold_download_refuses_nonempty_target_and_records_cold_metrics(self):
    st_bytes = _make_safetensors_bytes("w", "BF16")
    idx_bytes = json.dumps({"weight_map": {"w": "model-00001-of-00001.safetensors"}}).encode("utf-8")
    files_map = {
        "config.json": b'{"hidden_size": 128}',
        "model.safetensors.index.json": idx_bytes,
        "model-00001-of-00001.safetensors": st_bytes,
        "original/consolidated.00.safetensors": b"should_be_ignored",
    }
    api_fn, get_bytes_fn, url_fn, req_get = self._fake_hf_setup(files_map)

    gcs_dir = "gs://mesa-maxtext-europe-west4/mesa-bench-v2/hf_cache/Qwen_Test"
    # Pre-populate target folder to verify non-empty refusal.
    self.gcs.bucket("mesa-maxtext-europe-west4").blob(
        "mesa-bench-v2/hf_cache/Qwen_Test/stale.safetensors"
    ).upload_from_string(b"stale")
    local_res = os.path.join(self.tmp_dir, "step1_fail.json")
    with self.assertRaises(RuntimeError):
      step1_download_job.run_cold_hf_to_gcs_step1(
          platform="v5e",
          model_name="qwen3-0.6b",
          hf_repo_id="Qwen/Test",
          gcs_cache_dir=gcs_dir,
          run_id="r1",
          local_result_path=local_res,
          storage_client=self.gcs,
          hf_api_factory=api_fn,
          http_get_bytes=get_bytes_fn,
          hf_url_fn=url_fn,
          requests_get=req_get,
      )
    with open(local_res, encoding="utf-8") as f:
      self.assertEqual(json.load(f)["run_status"], "FAILED")

    # With clean_target=True, stale blob is deleted first and cold download succeeds from 0 files.
    local_ok = os.path.join(self.tmp_dir, "step1_ok.json")
    rec = step1_download_job.run_cold_hf_to_gcs_step1(
        platform="v5e",
        model_name="qwen3-0.6b",
        hf_repo_id="Qwen/Test",
        gcs_cache_dir=gcs_dir,
        run_id="r2",
        clean_target=True,
        local_result_path=local_ok,
        result_gcs_uri="gs://mesa-maxtext-europe-west4/mesa-bench-v2/logs/step1_result.json",
        storage_client=self.gcs,
        hf_api_factory=api_fn,
        http_get_bytes=get_bytes_fn,
        hf_url_fn=url_fn,
        requests_get=req_get,
        header_inspector=lambda _paths: {
            "hf_checkpoint_dtype": "BF16",
            "checkpoint_number_of_files": 1,
            "checkpoint_number_of_tensors": 1,
            "checkpoint_size_gb": round(len(st_bytes) / 1e9, 6),
        },
    )
    self.assertEqual(rec["run_status"], "OK")
    self.assertEqual(rec["step1_files_already_in_gcs_count"], 0)
    self.assertEqual(rec["step1_number_of_files"], 1)
    self.assertEqual(rec["step1_value_source"], "cold_download")
    uploaded_keys = set(self.gcs.bucket("mesa-maxtext-europe-west4").objects)
    self.assertIn("mesa-bench-v2/hf_cache/Qwen_Test/model-00001-of-00001.safetensors", uploaded_keys)
    self.assertNotIn("mesa-bench-v2/hf_cache/Qwen_Test/stale.safetensors", uploaded_keys)
    self.assertIn("mesa-bench-v2/logs/step1_result.json", uploaded_keys)

  def test_deepseek_step1b_and_cross_region_copy(self):
    src_bucket = self.gcs.bucket("mesa-maxtext-us-central1")
    src_bucket.blob("ds/fp8/config.json").upload_from_string("{}")
    src_bucket.blob("ds/fp8/model.safetensors").upload_from_string(_make_safetensors_bytes("w", "F8_E4M3"))

    def fake_dequant(in_dir: str, out_dir: str, _cache_size: int) -> None:
      self.assertTrue(os.path.exists(os.path.join(in_dir, "model.safetensors")))
      with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        f.write("{}")
      with open(os.path.join(out_dir, "model.safetensors"), "wb") as f:
        f.write(_make_safetensors_bytes("w", "BF16"))

    rec_1b = step1_download_job.run_deepseek_step1b_dequantize(
        platform="v7x",
        model_name="deepseek3-671b",
        hf_repo_id="deepseek-ai/DeepSeek-V3",
        source_gcs_dir="gs://mesa-maxtext-us-central1/ds/fp8",
        dest_gcs_dir="gs://mesa-maxtext-us-central1/ds/bf16",
        scratch_dir=os.path.join(self.tmp_dir, "scratch_1b"),
        run_id="r-ds-1b",
        storage_client=self.gcs,
        dequantize_fn=fake_dequant,
    )
    self.assertEqual(rec_1b["run_status"], "OK")
    self.assertEqual(rec_1b["loaded_checkpoint_dtype"], "BF16")
    self.assertFalse(os.path.exists(os.path.join(self.tmp_dir, "scratch_1b", "input_fp8")))

    self.gcs.bucket("mesa-maxtext-southamerica-west1").blob("ds/bf16/stale.safetensors").upload_from_string(b"old")
    rec_cp = step1_download_job.copy_bf16_gcs_to_region(
        platform="v6e",
        dest_region="southamerica-west1",
        model_name="deepseek3-671b",
        hf_repo_id="deepseek-ai/DeepSeek-V3",
        source_gcs_dir="gs://mesa-maxtext-us-central1/ds/bf16",
        dest_gcs_dir="gs://mesa-maxtext-southamerica-west1/ds/bf16",
        run_id="r-ds-copy",
        clean_target=True,
        storage_client=self.gcs,
    )
    self.assertEqual(rec_cp["run_status"], "OK")
    self.assertEqual(rec_cp["checkpoint_number_of_files"], 1)
    self.assertIn("ds/bf16/model.safetensors", self.gcs.bucket("mesa-maxtext-southamerica-west1").objects)
    self.assertNotIn("ds/bf16/stale.safetensors", self.gcs.bucket("mesa-maxtext-southamerica-west1").objects)
    # Silence unused import warning cleanly while cross-checking metadata helper on local path if needed.
    self.assertTrue(callable(metadata.inspect_safetensors_headers))


if __name__ == "__main__":
  unittest.main()
