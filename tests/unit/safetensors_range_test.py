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

"""Bounded remote tensor reads must never accept a full-file response."""
import io
import json
import struct
from unittest.mock import patch
import numpy as np
import pytest
from maxtext.checkpoint_conversion.utils.safetensors_range import RemoteSafetensors, read_range


class Response(io.BytesIO):

  def __init__(self, data, start, end, total, status=206):
    super().__init__(data)
    self.status = status
    self.headers = {"Content-Range": f"bytes {start}-{end}/{total}"}


def test_extract_rows_in_requested_order_without_full_tensor_read():

  payload = bytes(range(24))
  header = json.dumps({"table": {"dtype": "BF16", "shape": [6, 2], "data_offsets": [0, 24]}}).encode()
  source = struct.pack("<Q", len(header)) + header + payload
  requests = []

  def open_range(request, timeout):
    start, end = map(int, request.get_header("Range").removeprefix("bytes=").split("-"))
    requests.append((start, end))
    return Response(source[start : end + 1], start, end, len(source))

  with patch("urllib.request.urlopen", open_range):
    reader = RemoteSafetensors("https://example.test/shard")
    assert reader.read_rows("table", [5, 1, 5]) == payload[20:24] + payload[4:8] + payload[20:24]
  assert all(end - start + 1 < len(source) for start, end in requests)


def test_reject_server_ignoring_range():

  with patch("urllib.request.urlopen", return_value=Response(b"abcdefgh", 0, 7, 8, status=200)):
    with pytest.raises(ValueError, match="206"):
      read_range("https://example.test/shard", 0, 8)


def test_reject_wrong_content_range_and_short_read():

  with patch("urllib.request.urlopen", return_value=Response(b"abcd", 4, 7, 8)):
    with pytest.raises(ValueError, match="Content-Range"):
      read_range("https://example.test/shard", 0, 4)
  with patch("urllib.request.urlopen", return_value=Response(b"ab", 0, 3, 8)):
    with pytest.raises(ValueError, match="Truncated"):
      read_range("https://example.test/shard", 0, 4)


def test_stream_tensor_retains_bfloat16_and_bounds_reads():

  payload = np.array([0x3F80, 0x4000, 0x4040, 0x4080], dtype=np.uint16).tobytes()
  header = json.dumps({"table": {"dtype": "BF16", "shape": [2, 2], "data_offsets": [0, 8]}}).encode()
  source = struct.pack("<Q", len(header)) + header + payload

  def open_range(request, timeout):
    start, end = map(int, request.get_header("Range").removeprefix("bytes=").split("-"))
    return Response(source[start : end + 1], start, end, len(source))

  with patch("urllib.request.urlopen", open_range):
    reader = RemoteSafetensors("https://example.test/shard")
    result = reader.read_tensor_numpy("table", chunk_bytes=4)
  assert str(result.dtype) == "bfloat16"
  np.testing.assert_array_equal(result.astype(np.float32), [[1, 2], [3, 4]])
