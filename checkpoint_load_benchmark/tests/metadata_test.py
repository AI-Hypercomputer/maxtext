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

"""Tests checkpoint_load_benchmark/bench/metadata.py."""

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


def _write_fake_safetensors(path: str, header_dict: dict, payload_bytes: int = 16) -> None:
  hdr_bytes = json.dumps(header_dict).encode("utf-8")
  with open(path, "wb") as f:
    f.write(struct.pack("<Q", len(hdr_bytes)))
    f.write(hdr_bytes)
    f.write(b"\x00" * payload_bytes)


class MetadataTest(unittest.TestCase):
  """Verifies header-only safetensors inspection, env metadata, and CODE_VERSION loading."""

  def setUp(self):
    super().setUp()
    self.tmp_dir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self.tmp_dir)

  def test_inspect_safetensors_headers_detects_bf16_and_deepseek_scales(self):
    bf16_path = os.path.join(self.tmp_dir, "bf16.safetensors")
    _write_fake_safetensors(
        bf16_path,
        {"w": {"dtype": "BF16", "shape": [4, 4], "data_offsets": [0, 32]}},
        payload_bytes=32,
    )
    info = metadata.inspect_safetensors_headers([bf16_path])
    self.assertEqual(info["hf_checkpoint_dtype"], "BF16")
    self.assertEqual(info["checkpoint_number_of_files"], 1)
    self.assertEqual(info["checkpoint_number_of_tensors"], 1)
    self.assertGreater(info["checkpoint_bytes"], 32)

    fp4_path = os.path.join(self.tmp_dir, "ds_v4.safetensors")
    _write_fake_safetensors(
        fp4_path,
        {
            "layers.0.attn.w.weight": {"dtype": "I8", "shape": [8], "data_offsets": [0, 8]},
            "layers.0.attn.w.scale": {"dtype": "F8_E8M0", "shape": [2], "data_offsets": [8, 10]},
        },
        payload_bytes=10,
    )
    ds_info = metadata.inspect_safetensors_headers([fp4_path])
    self.assertEqual(ds_info["hf_checkpoint_dtype"], "FP4+scales")
    self.assertEqual(ds_info["checkpoint_number_of_tensors"], 2)

  def test_collect_env_cluster_and_code_version_metadata(self):
    env_meta = metadata.collect_env_cluster_metadata(
        {
            "POD_NAME": "worker-0",
            "NODE_NAME": "node-abc",
            "BENCH_PLATFORM": "v6e",
            "BENCH_KUEUE_QUEUE": "default",
            "BENCH_IMAGE": "gcr.io/img:v1",
            "BENCH_HOST_MACHINE_TYPE": "n2-standard-64",
        }
    )
    self.assertEqual(env_meta["pod_name"], "worker-0")
    self.assertEqual(env_meta["node_name"], "node-abc")
    self.assertEqual(env_meta["tpu_platform"], "v6e")
    self.assertEqual(env_meta["kueue_local_queue_name"], "default")
    self.assertEqual(env_meta["container_image"], "gcr.io/img:v1")
    self.assertEqual(env_meta["host_machine_type"], "n2-standard-64")

    cv_path = os.path.join(self.tmp_dir, "CODE_VERSION.json")
    with open(cv_path, "w", encoding="utf-8") as f:
      json.dump(
          {
              "git_branch_name": "docs/checkpoint-load-benchmark",
              "git_commit": "abc123",
              "git_working_tree_clean": True,
              "code_tarball_sha256": "deadbeef",
          },
          f,
      )
    loaded = metadata.load_code_version_metadata(self.tmp_dir)
    self.assertEqual(loaded["git_commit"], "abc123")
    self.assertTrue(loaded["git_working_tree_clean"])
    self.assertEqual(loaded["code_tarball_sha256"], "deadbeef")


if __name__ == "__main__":
  unittest.main()
