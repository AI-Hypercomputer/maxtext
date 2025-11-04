
from functools import lru_cache
import importlib.metadata
from packaging import version

# No matching JAX module found for `_is_package_available`.
# The logic is refactored to be self-contained by using a try-except block.

@lru_cache
def is_huggingface_hub_greater_or_equal(library_version: str, accept_dev: bool = False) -> bool:
  """Checks if the installed `huggingface_hub` version is greater than or equal to a given version."""
  try:
    installed_version_str = importlib.metadata.version("huggingface_hub")
    installed_version = version.parse(installed_version_str)
    required_version = version.parse(library_version)

    if accept_dev:
      return version.parse(installed_version.base_version) >= required_version
    else:
      return installed_version >= required_version
  except importlib.metadata.PackageNotFoundError:
    return False

import importlib.metadata
from functools import lru_cache

from packaging import version


@lru_cache
def is_jax_greater_or_equal(library_version: str, accept_dev: bool = False):
  """
  Accepts a library version and returns True if the current version of the library is greater than or equal to the
  given version. If `accept_dev` is True, it will also accept development versions (e.g. 0.4.23.dev20240101 matches
  0.4.23).
  """
  if not _is_package_available("jax"):
    return False

  if accept_dev:
    return version.parse(version.parse(importlib.metadata.version("jax")).base_version) >= version.parse(
        library_version
    )
  else:
    return version.parse(importlib.metadata.version("jax")) >= version.parse(library_version)
# JAX does not have a direct equivalent for torch.xpu, so this is set to False.
_is_torch_xpu_available = FalseBS4_IMPORT_ERROR = """
{0} requires the Beautiful Soup library but it was not found in your environment. You can install it with pip:
`pip install beautifulsoup4`. Please note that you may need to restart your runtime after installation.
"""
CV2_IMPORT_ERROR = """
{0} requires the OpenCV library but it was not found in your environment. You can install it with:

pip install decord

FASTAPI_IMPORT_ERROR = """
{0} requires the fastapi library but it was not found in your environment. You can install it with pip:
`pip install fastapi`. Please note that you may need to restart your runtime after installation.
"""

FTFY_IMPORT_ERROR = """
{0} requires the ftfy library but it was not found in your environment. Check out the instructions on the
installation section: https://github.com/rspeer/python-ftfy/tree/master#installing and follow the ones
that match your environment. Please note that you may need to restart your runtime after installation.
"""
NATTEN_IMPORT_ERROR = """
{0} requires the natten library but it was not found in your environment. You can install it by referring to:
shi-labs.com/natten . You can also install it with pip (may take longer to build):
`pip install natten`. Please note that you may need to restart your runtime after installation.
"""
# docstyle-ignore
FLAX_IMPORT_ERROR = """
{0} requires the FLAX library but it was not found in your environment. Check out the instructions on the
installation page: https://github.com/google/flax and follow the ones that match your environment.
Please note that you may need to restart your runtime after installation.
"""
RICH_IMPORT_ERROR = """
{0} requires the rich library but it was not found in your environment. You can install it with pip: `pip install
rich`. Please note that you may need to restart your runtime after installation.
"""
import importlib.metadata
import importlib.util
import os

from . import logging


_flax_available = False
if USE_JAX in ENV_VARS_TRUE_AND_AUTO_VALUES:
  _flax_available, _flax_version = _is_package_available("flax", return_version=True)
  if _flax_available:
    _jax_available, _jax_version = _is_package_available("jax", return_version=True)
    if _jax_available:
      logger.info(f"JAX version {_jax_version}, Flax version {_flax_version} available.")
    else:
      _flax_available = _jax_available = False
      _jax_version = _flax_version = "N/A"

# docstyle-ignore
SENTENCEPIECE_IMPORT_ERROR = """
{0} requires the SentencePiece library but it was not found in your environment. Check out the instructions on the
installation page of its repo: https://github.com/google/sentencepiece#installation and follow the ones
that match your environment. Please note that you may need to restart your runtime after installation.
"""

pip install torchcodec
def is_datasets_available() -> bool:
  """Check if datasets is available."""
  return _datasets_available
import importlib.util
from typing import Union

# Rewriting _is_package_available for use by other functions.
# This is a simplified version of the original, as the full complexity
# for torch, quark, etc., is not needed for most packages.
def _is_package_available(pkg_name: str, return_version: bool = False) -> Union[tuple[bool, str], bool]:
  """Check if a package is available and optionally return its version."""
  package_exists = importlib.util.find_spec(pkg_name) is not None
  if return_version:
    package_version = "N/A"
    if package_exists:
      try:
        package_version = importlib.metadata.version(pkg_name)
      except importlib.metadata.PackageNotFoundError:
        # Fallback for packages that don't expose version metadata correctly.
        pass
    return package_exists, package_version
  else:
    return package_exists

_detectron2_available = _is_package_available("detectron2")

def is_detectron2_available() -> bool:
  """Check if detectron2 is available."""
  return _detectron2_available

def is_ftfy_available() -> bool:
  return _ftfy_available
def is_g2p_en_available() -> bool:
  """Returns True if g2p_en is available."""
  return _g2p_en_available
def is_natten_available() -> bool:
  """Checks if the natten library is available."""
  return _natten_available

from enum import Enum
import operator
from typing import Any, Callable


class VersionComparison(Enum):
  """Represents a comparison operator for versions."""

  EQUAL = operator.eq
  NOT_EQUAL = operator.ne
  GREATER_THAN = operator.gt
  LESS_THAN = operator.lt
  GREATER_THAN_OR_EQUAL = operator.ge
  LESS_THAN_OR_EQUAL = operator.le

  @staticmethod
  def from_string(version_string: str) -> Callable[[Any, Any], bool]:
    """
    Converts a string representation of a comparison operator to its function.

    Args:
      version_string: The string operator (e.g., '==', '>=', '<').

    Returns:
      The corresponding operator function (e.g., operator.eq, operator.ge).
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
  """Splits a package version string like 'torch>=2.0.0' into its components.

  Args:
    package_version_str: The string to parse.

  Returns:
    A tuple containing the package name, the comparison operator, and the
    version number.

  Raises:
    ValueError: If the string is not in a valid format.
  """
  pattern = r"([a-zA-Z0-9_-]+)([!<>=~]+)([0-9.]+)"
  match = re.match(pattern, package_version_str)
  if match:
    return (match.group(1), match.group(2), match.group(3))
  else:
    raise ValueError(f"Invalid package version string: {package_version_str}")
