
from contextlib import AbstractContextManager, ExitStack
from typing import List


class ContextManagers:
  """
  Wrapper for `contextlib.ExitStack` which enters a collection of context managers. Adaptation of `ContextManagers`
  in the `fastcore` library.
  """

  def __init__(self, context_managers: List[AbstractContextManager]):
    self.context_managers = context_managers
    self.stack = ExitStack()

  def __enter__(self):
    for context_manager in self.context_managers:
      self.stack.enter_context(context_manager)

  def __exit__(self, *args, **kwargs):
    self.stack.__exit__(*args, **kwargs)

from functools import lru_cache
import jax

@lru_cache
def is_jax_xpu_available(check_device: bool = False) -> bool:
  """Checks if XPU acceleration is available for JAX.

  Args:
    check_device: This parameter is kept for API consistency with the original
      PyTorch function but is less meaningful in JAX, as backend availability
      implies device presence.

  Returns:
    A boolean indicating whether XPU devices are available for JAX.
  """
  # The check_device parameter is ignored in the JAX implementation.
  try:
    # This will raise a RuntimeError if the 'xpu' backend is not found
    # or no devices of that type are available.
    return jax.device_count(backend="xpu") > 0
  except RuntimeError:
    return False
# Reused from Qwen3ForCausalLM.utils.is_jax_xpu_available
from MaxText.utils import is_jax_xpu_available


_is_jax_xpu_available = is_jax_xpu_available()
# Copyright 2024 Google LLC
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

"""
{0} requires the torch ccl library but it was not found in your environment. You can install it with pip:
`pip install oneccl_bind_pt -f https://developer.intel.com/ipex-whl-stable`
Please note that you may need to restart your runtime after installation.
"""
CCL_IMPORT_ERROR = """
{0} requires the torch ccl library but it was not found in your environment. You can install it with pip:
`pip install oneccl_bind_pt -f https://developer.intel.com/ipex-whl-stable`
Please note that you may need to restart your runtime after installation.
"""
DETECTRON2_IMPORT_ERROR = """
{0} requires the detectron2 library but it was not found in your environment. Check out the instructions on the
installation page: https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md and follow the ones
that match your environment. Please note that you may need to restart your runtime after installation.
"""JINJA_IMPORT_ERROR = """
{0} requires the jinja library but it was not found in your environment. You can install it with pip: `pip install
jinja2`. Please note that you may need to restart your runtime after installation.
"""
LEVENSHTEIN_IMPORT_ERROR = """
{0} requires the python-Levenshtein library but it was not found in your environment. You can install it with pip: `pip
install python-Levenshtein`. Please note that you may need to restart your runtime after installation.
"""
PEFT_IMPORT_ERROR = """
{0} requires the peft library but it was not found in your environment. You can install it with pip: `pip install
peft`. Please note that you may need to restart your runtime after installation.
"""PRETTY_MIDI_IMPORT_ERROR = """
{0} requires the pretty_midi library. But that was not found in your environment. You can install them with pip:
`pip install pretty_midi`
Please note that you may need to restart your runtime after installation.
"""PROTOBUF_IMPORT_ERROR = """
{0} requires the protobuf library but it was not found in your environment. Check out the instructions on the
installation page of its repo: https://github.com/protocolbuffers/protobuf/tree/master/python#installation and follow the ones
that match your environment. Please note that you may need to restart your runtime after installation.
"""PYCTCDECODE_IMPORT_ERROR = """
{0} requires the pyctcdecode library but it was not found in your environment. You can install it with pip:
`pip install pyctcdecode`. Please note that you may need to restart your runtime after installation.
"""PYDANTIC_IMPORT_ERROR = """
{0} requires the pydantic library but it was not found in your environment. You can install it with pip:
`pip install pydantic`. Please note that you may need to restart your runtime after installation.
"""PYTESSERACT_IMPORT_ERROR = """
{0} requires the PyTesseract library but it was not found in your environment. You can install it with pip:
`pip install pytesseract`. Please note that you may need to restart your runtime after installation.
"""SACREMOSES_IMPORT_ERROR = """
{0} requires the sacremoses library but it was not found in your environment. You can install it with pip:
`pip install sacremoses`. Please note that you may need to restart your runtime after installation.
"""TENSORFLOW_IMPORT_ERROR = """
{0} requires the TensorFlow library but it was not found in your environment. Check out the instructions on the
installation page: https://www.tensorflow.org/install and follow the ones that match your environment.
Please note that you may need to restart your runtime after installation.
"""TENSORFLOW_PROBABILITY_IMPORT_ERROR = """
{0} requires the tensorflow_probability library but it was not found in your environment. You can install it with pip as
explained here: https://github.com/tensorflow/probability. Please note that you may need to restart your runtime after installation.
"""
TENSORFLOW_TEXT_IMPORT_ERROR = """
{0} requires the tensorflow_text library but it was not found in your environment. You can install it with pip as
explained here: https://www.tensorflow.org/text/guide/tf_text_intro.
Please note that you may need to restart your runtime after installation.
"""

import importlib.util

# `importlib.metadata.util` doesn't work with `opencv-python-headless`.
_cv2_available = importlib.util.find_spec("cv2") is not None


def is_cv2_available():
  return _cv2_available
def is_faiss_available():
  """Checks if the faiss library is available."""
  return _faiss_available
def is_nltk_available() -> bool:
  return _nltk_available
def is_pandas_available() -> bool:
  """Returns True if pandas is available."""
  return _pandas_available
import importlib.util


def is_protobuf_available():
  """Checks if protobuf is available."""
  if importlib.util.find_spec("google") is None:
    return False
  return importlib.util.find_spec("google.protobuf") is not None
def is_sklearn_available() -> bool:
  """Returns True if scikit-learn is available."""
  return _sklearn_available
# Copyright 2024 The HuggingFace Team. All rights reserved.
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

from .. import _uroman_available


def is_uroman_available() -> bool:
  """Checks if uroman is available."""
  return _uroman_available
def is_uvicorn_available() -> bool:
  """Checks if uvicorn is available."""
  return _uvicorn_available
# The variable `_yt_dlp_available` is assumed to be defined at the module level.
# It is set by `importlib.util.find_spec("yt_dlp") is not None`
def is_yt_dlp_available():
  return _yt_dlp_available

from enum import Enum
import operator
from typing import Callable


class VersionComparison(Enum):
  """An enumeration of version comparison operators."""

  EQUAL = operator.eq
  NOT_EQUAL = operator.ne
  GREATER_THAN = operator.gt
  LESS_THAN = operator.lt
  GREATER_THAN_OR_EQUAL = operator.ge
  LESS_THAN_OR_EQUAL = operator.le

  @staticmethod
  def from_string(version_string: str) -> Callable:
    """Converts a string representation to a comparison operator.

    Args:
      version_string: The string representation of the operator (e.g., '==', '>=')

    Returns:
      The corresponding operator function from the `operator` module.
    """
    string_to_operator = {
        "=": VersionComparison.EQUAL.value,
        "==": VersionComparison.EQUAL.value,
        "!=": VersionComparison.NOT_EQUAL.value,
        ">": VersionComparison.GREATER_THAN.value,
        "<": VersionComparison.LESS_THAN.value,
        ">=": VersionComparison.GREATER_THAN_OR_EQUAL.value,
        "<=": VersionComparison.LESS_THAN_OR_EQUAL.value,
    }

    return string_to_operator[version_string]

import re
from functools import lru_cache
from typing import Tuple


@lru_cache
def split_package_version(package_version_str: str) -> Tuple[str, str, str]:
  """Splits a package version string into its name, operator, and version."""
  pattern = r"([a-zA-Z0-9_-]+)([!<>=~]+)([0-9.]+)"
  match = re.match(pattern, package_version_str)
  if match:
    return (match.group(1), match.group(2), match.group(3))
  else:
    raise ValueError(f"Invalid package version string: {package_version_str}")

import importlib.util
import importlib.metadata

try:
  _mlx_available = (
      importlib.util.find_spec("mlx") is not None
      and importlib.metadata.version("mlx") is not None
  )
except importlib.metadata.PackageNotFoundError:
  _mlx_available = False

from typing import Any

from ..utils.import_utils import is_mlx_available


def _is_mlx(x: Any) -> bool:
  """Helper to check for mlx array type."""
  import mlx.core as mx

  return isinstance(x, mx.array)


def is_mlx_array(x: Any) -> bool:
  """Tests if `x` is a mlx array or not. Safe to call even when mlx is not installed."""
  return False if not is_mlx_available() else _is_mlx(x)

from typing import List
import jax.numpy as jnp


def _max_by_axis(the_list: List[List[int]]) -> jnp.ndarray:
  """Computes the element-wise maximum across a list of lists.

  Args:
    the_list: A list of lists of integers, e.g., [[1, 5, 3], [4, 2, 6]].

  Returns:
    A JAX array containing the element-wise maximums.
  """
  return jnp.max(jnp.array(the_list), axis=0)

import importlib.util


_av_available = importlib.util.find_spec("av") is not None

import importlib.util


_av_available: bool = importlib.util.find_spec("av") is not None


def is_av_available() -> bool:
  """Returns whether the PyAv library is available."""
  return _av_available

import importlib.metadata
import importlib.util
from functools import lru_cache


@lru_cache
def is_jinja_available() -> bool:
  """Checks if jinja2 is available."""
  if importlib.util.find_spec("jinja2") is None:
    return False
  try:
    importlib.metadata.version("jinja2")
    return True
  except importlib.metadata.PackageNotFoundError:
    return False

import importlib.util
import importlib.metadata
from typing import Union, Tuple

def _is_package_available(pkg_name: str, return_version: bool = False) -> Union[Tuple[bool, str], bool]:
  """
  Checks if a package is available and optionally returns its version.
  This is a simplified, framework-agnostic version of the original utility.
  """
  package_exists = importlib.util.find_spec(pkg_name) is not None
  package_version = "N/A"
  if package_exists:
    try:
      package_version = importlib.metadata.version(pkg_name)
    except importlib.metadata.PackageNotFoundError:
      package_exists = False

  if return_version:
    return package_exists, package_version
  return package_exists

_keras_nlp_available = _is_package_available("keras_nlp")

import importlib.util

_pretty_midi_available = importlib.util.find_spec("pretty_midi") is not None

import importlib.util

_torchcodec_available = importlib.util.find_spec("torchcodec") is not None

from enum import Enum


class ExplicitEnum(str, Enum):
  """
  Enum with more explicit error message for missing values.
  """

  @classmethod
  def _missing_(cls, value):
    raise ValueError(
        f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
    )

# Reused from Qwen3ForCausalLM.utils.ExplicitEnum
from Qwen3ForCausalLM.utils import ExplicitEnum


class TensorType(ExplicitEnum):
  """
  Possible values for the `return_tensors` argument in [`PreTrainedTokenizerBase.__call__`]. Useful for
  tab-completion in an IDE.
  """

  PYTORCH = "pt"
  TENSORFLOW = "tf"
  NUMPY = "np"
  JAX = "jax"
  MLX = "mlx"

from typing import Any

import jax

# This function is assumed to be available in the same file, as in the original source.
from . import _get_frameworks_and_test_func


def is_tensor(x: Any) -> bool:
  """
    Tests if `x` is a `jax.Array`, `np.ndarray`, `torch.Tensor`, `tf.Tensor`, or `mlx.array`
    in the order defined by `infer_framework_from_repr`.
    """
  # This gives us a smart order to test the frameworks with the corresponding tests.
  framework_to_test_func = _get_frameworks_and_test_func(x)
  for test_func in framework_to_test_func.values():
    if test_func(x):
      return True

  # Check for JAX tracers, which are the equivalent of PyTorch's fx.Proxy
  if isinstance(x, jax.core.Tracer):
    return True

  return False

from typing import Any, List, Tuple, Union

# The following dependencies are defined in the same file:
# Backend, is_torch_available, is_tf_available,
# PYTORCH_IMPORT_ERROR_WITH_TF, TF_IMPORT_ERROR_WITH_PYTORCH, BACKENDS_MAPPING


def requires_backends(obj: Any, backends: Union[str, List[str], Tuple[str, ...]]):
  """
  Decorator function to check for backend availability and raise informative errors.
  """
  if not isinstance(backends, (list, tuple)):
    backends = [backends]

  name = obj.__name__ if hasattr(obj, "__name__") else obj.__class__.__name__

  # Raise an error for users who might not realize that classes without "TF" are torch-only
  if "torch" in backends and "tf" not in backends and not is_torch_available() and is_tf_available():
    raise ImportError(PYTORCH_IMPORT_ERROR_WITH_TF.format(name))

  # Raise the inverse error for PyTorch users trying to load TF classes
  if "tf" in backends and "torch" not in backends and is_torch_available() and not is_tf_available():
    raise ImportError(TF_IMPORT_ERROR_WITH_PYTORCH.format(name))

  failed = []
  for backend in backends:
    if isinstance(backend, Backend):
      available, msg = backend.is_satisfied, backend.error_message
    else:
      available, msg = BACKENDS_MAPPING[backend]

    if not available():
      failed.append(msg.format(name))

  if failed:
    raise ImportError("".join(failed))
