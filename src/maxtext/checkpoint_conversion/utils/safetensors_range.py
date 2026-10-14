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

"""Strict bounded HTTP reads for safetensors checkpoint slicing."""

import json
import math
import re
import struct
import urllib.request

ITEM_SIZES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}


def read_range(url, start, length):
  """Read exactly the requested bytes, rejecting servers that ignore ranges."""
  if start < 0 or length <= 0:
    raise ValueError("Invalid byte range")
  end = start + length - 1
  # Query identifies the range to avoid caches reusing a different range's response.
  separator = "&" if "?" in url else "?"
  request = urllib.request.Request(f"{url}{separator}byte_range={start}-{end}", headers={"Range": f"bytes={start}-{end}"})
  with urllib.request.urlopen(request, timeout=120) as response:
    if response.status != 206:
      raise ValueError(f"Expected HTTP 206, received {response.status}; refusing full-file download")
    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
    if match is None or (int(match[1]), int(match[2])) != (start, end):
      raise ValueError("Incorrect Content-Range")
    data = response.read(length + 1)
    if len(data) != length:
      raise ValueError("Truncated or oversized range response")
    return data


class RemoteSafetensors:
  """Read remote safetensors metadata, selected rows, or bounded tensor chunks."""

  def __init__(self, url):
    self.url = url
    header_size = struct.unpack("<Q", read_range(url, 0, 8))[0]
    if not 0 < header_size <= 100_000_000:
      raise ValueError("Invalid safetensors header size")
    self.header = json.loads(read_range(url, 8, header_size))
    self.data_start = 8 + header_size

  def iter_tensor_bytes(self, name, chunk_bytes=8 * 1024 * 1024):
    """Yield bounded byte chunks after checking the tensor layout."""
    spec = self.header[name]
    start, end = spec["data_offsets"]
    expected = math.prod(spec["shape"]) * ITEM_SIZES[spec["dtype"]]
    if expected != end - start:
      raise ValueError(f"Invalid tensor offsets for {name}")
    for offset in range(start, end, chunk_bytes):
      yield read_range(self.url, self.data_start + offset, min(chunk_bytes, end - offset))

  def read_rows(self, name, rows):
    """Extract exact matrix rows while preserving the caller order."""
    spec = self.header[name]
    if len(spec["shape"]) != 2:
      raise ValueError("Row extraction requires a matrix")
    row_bytes = spec["shape"][1] * ITEM_SIZES[spec["dtype"]]
    start, end = spec["data_offsets"]
    if end - start != math.prod(spec["shape"]) * ITEM_SIZES[spec["dtype"]]:
      raise ValueError("Invalid tensor offsets")
    cache = {}
    for row in sorted(set(rows)):
      if row < 0 or row >= spec["shape"][0]:
        raise IndexError(f"Row {row} outside tensor {name}")
      cache[row] = read_range(self.url, self.data_start + start + row * row_bytes, row_bytes)
    return b"".join(cache[row] for row in rows)

  def read_tensor_numpy(self, name, chunk_bytes=8 * 1024 * 1024):
    """Fill one native-dtype array without caching the containing source shard."""
    import numpy as np  # pylint: disable=import-outside-toplevel
    import ml_dtypes  # pylint: disable=import-outside-toplevel

    spec = self.header[name]
    dtypes = {
        "BF16": ml_dtypes.bfloat16,
        "F16": np.float16,
        "F32": np.float32,
        "F64": np.float64,
        "I64": np.int64,
        "I32": np.int32,
        "I16": np.int16,
        "I8": np.int8,
        "U8": np.uint8,
        "BOOL": np.bool_,
    }
    array = np.empty(spec["shape"], dtype=dtypes[spec["dtype"]])
    raw = array.view(np.uint8).reshape(-1)
    offset = 0
    for chunk in self.iter_tensor_bytes(name, chunk_bytes):
      raw[offset : offset + len(chunk)] = np.frombuffer(chunk, dtype=np.uint8)
      offset += len(chunk)
    return array
