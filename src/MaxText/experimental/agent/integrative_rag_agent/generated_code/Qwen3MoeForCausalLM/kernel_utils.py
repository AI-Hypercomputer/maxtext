
# Copyright 2025 The HuggingFace Team. All rights reserved.
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

# The original PyTorch `kernels` library provides a mechanism for dynamically replacing
# nn.Module forward methods with optimized, pre-compiled kernels from an external source.
# This functionality is specific to the PyTorch ecosystem and does not have a direct
# equivalent in JAX/MaxText. In JAX, optimizations are typically achieved through
# JIT compilation (XLA), custom Pallas kernels, or selecting different JAX-native
# implementations via configuration flags.
#
# This file provides stub implementations to maintain API compatibility for any code
# that might reference these functions, while making it clear that the underlying
# dynamic kernel swapping feature is not supported.

_hub_kernels_available = False


def is_hub_kernels_available():
  """
  Returns whether the `kernels` library is available. In this JAX conversion,
  it always returns False as the library is PyTorch-specific.
  """
  return _hub_kernels_available


# Stub to make decorators in transformers work when `kernels` is not installed.
def use_kernel_forward_from_hub(*args, **kwargs):
  """
  A stub for the `use_kernel_forward_from_hub` decorator.

  This is a no-op that returns the original class unmodified.
  """

  def decorator(cls):
    return cls

  return decorator


class LayerRepository:
  """
  A stub for the `LayerRepository` class.

  Raises a RuntimeError upon instantiation, indicating that this feature
  is not available in the JAX environment.
  """

  def __init__(self, *args, **kwargs):
    raise RuntimeError(
        "LayerRepository requires the PyTorch `kernels` library, which is not supported in this JAX environment."
    )


def replace_kernel_forward_from_hub(*args, **kwargs):
  """
  A stub for the `replace_kernel_forward_from_hub` function.

  Raises a RuntimeError, indicating that this feature is not available.
  """
  raise RuntimeError(
      "replace_kernel_forward_from_hub requires the PyTorch `kernels` library, which is not supported in this JAX"
      " environment."
  )


def register_kernel_mapping(*args, **kwargs):
  """
  A stub for the `register_kernel_mapping` function.

  Raises a RuntimeError, indicating that this feature is not available.
  """
  raise RuntimeError(
      "register_kernel_mapping requires the PyTorch `kernels` library, which is not supported in this JAX environment."
  )


__all__ = [
    "LayerRepository",
    "is_hub_kernels_available",
    "use_kernel_forward_from_hub",
    "register_kernel_mapping",
    "replace_kernel_forward_from_hub",
]

# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law of a an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# The original PyTorch `kernels` library allows for dynamic replacement of `nn.Module`
# layers with optimized, pre-compiled kernels from the Hugging Face Hub. This is a
# PyTorch-specific feature. JAX/MaxText uses a different paradigm where optimized
# kernels (e.g., Pallas, Triton) are implemented and selected within the JAX code
# itself. Therefore, a direct functional conversion is not applicable.
#
# This module provides API-compatible stubs that indicate this functionality is not
# available in the JAX/MaxText environment, mirroring the behavior of the original
# code when the `kernels` library is not installed.


def use_kernel_forward_from_hub(*args, **kwargs):
  """
  No-op decorator. In PyTorch, this would flag a class for potential kernel
  replacement. In JAX, this functionality is not supported and this decorator
  does nothing.
  """

  def decorator(cls):
    return cls

  return decorator


class LayerRepository:
  """
  Stub for LayerRepository. Raises a RuntimeError on instantiation as dynamic
  kernel loading is not a feature in JAX/MaxText.
  """

  def __init__(self, *args, **kwargs):
    raise RuntimeError(
        "LayerRepository and dynamic kernel loading from the Hub are PyTorch-specific "
        "features and are not supported in the JAX/MaxText environment."
    )


def replace_kernel_forward_from_hub(*args, **kwargs):
  """
  Stub for replace_kernel_forward_from_hub. Raises a RuntimeError as this
  is a PyTorch-specific feature.
  """
  raise RuntimeError(
      "replace_kernel_forward_from_hub is a PyTorch-specific feature and is not "
      "supported in the JAX/MaxText environment."
  )


def register_kernel_mapping(*args, **kwargs):
  """
  Stub for register_kernel_mapping. Raises a RuntimeError as this is a
  PyTorch-specific feature.
  """
  raise RuntimeError(
      "register_kernel_mapping is a PyTorch-specific feature and is not "
      "supported in the JAX/MaxText environment."
  )


_hub_kernels_available = False


def is_hub_kernels_available():
  """
  Returns False, as the PyTorch `kernels` library for dynamic kernel loading
  is not available in the JAX/MaxText environment.
  """
  return _hub_kernels_available


__all__ = [
    "LayerRepository",
    "is_hub_kernels_available",
    "use_kernel_forward_from_hub",
    "register_kernel_mapping",
    "replace_kernel_forward_from_hub",
]
