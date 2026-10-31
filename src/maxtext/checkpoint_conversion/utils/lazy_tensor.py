# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Lazy tensor proxy used by Hugging Face to MaxText checkpoint conversion.

Defined here rather than in `to_maxtext` so that running `python -m
maxtext.checkpoint_conversion.to_maxtext` (where `to_maxtext` executes as
`__main__`) produces the same `LazyTensor` class object registered by
`lazy_save`.
"""

from typing import Callable

import ml_dtypes
from maxtext.utils import max_logging
import numpy as np


class LazyTensor:
  """
  A proxy object that looks like a NumPy array but delays actual loading
  and transformation until __array__ is called (e.g., by Orbax during save).
  """

  def __init__(
      self,
      load_fn: Callable[[], np.ndarray],
      shape: tuple,
      dtype,
      name: str = "unknown",
  ):
    self._load_fn = load_fn
    self.shape = shape
    try:
      self.dtype = np.dtype(dtype)
    except (TypeError, ValueError):
      dtype_str = str(dtype)
      if "float8_e4m3fn" in dtype_str:
        self.dtype = np.dtype(ml_dtypes.float8_e4m3fn)
      elif "bfloat16" in dtype_str:
        self.dtype = np.dtype(ml_dtypes.bfloat16)
      else:
        self.dtype = np.dtype(np.float32)
    self.ndim = len(shape)
    self.name = name

  @property
  def size(self):
    """Total number of elements in the tensor."""
    return np.prod(self.shape)

  @property
  def nbytes(self):
    """Return estimated nbytes so Orbax doesn't need to load the real array to find out."""
    return self.size * self.dtype.itemsize

  @property
  def itemsize(self):
    return self.dtype.itemsize

  def __array__(self, dtype=None):
    """
    Materializes the tensor data.

    When this method is invoked, it finally calls the `_load_fn` that was
    provided during initialization. This function executes the actual loading
    and transformation of the tensor from the Hugging Face checkpoint. The
    resulting NumPy array is then returned to the caller.
    """
    # This method is called just-in-time by Orbax when saving this specific leaf.
    try:
      arr = self._load_fn()
    except Exception as e:
      max_logging.log(f"FATAL ERROR: Failed to load tensor '{self.name}' (shape {self.shape}). Error: {e}")
      # Re-raise the original exception so it doesn't get masked by "object __array__..."
      raise

    if not isinstance(self.shape, list) and arr.shape != self.shape:
      raise ValueError(f"Shape mismatch for tensor '{self.name}'. Expected {self.shape}, but got {arr.shape}.")

    # Ensure it's a standard numpy array (converts JAX arrays if necessary)
    if not isinstance(arr, np.ndarray):
      arr = np.array(arr)

    if dtype is not None and arr.dtype != dtype:
      return arr.astype(dtype)
    return arr

  def __repr__(self):
    return f"LazyTensor(name={self.name}, shape={self.shape}, dtype={self.dtype})"
