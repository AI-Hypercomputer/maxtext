
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
"""Flax activation functions."""

import math
from flax.linen import Module
import jax
from jax import numpy as jnp

from MaxText.common_types import Array


class LaplaceActivation(Module):
  """
  Applies elementwise activation based on Laplace function, introduced in MEGA as an attention activation. See
  https://huggingface.co/papers/2209.10655

  Inspired by squared relu, but with bounded range and gradient for better stability
  """

  def __call__(
      self, inputs: Array, mu: float = 0.707107, sigma: float = 0.282095
  ) -> Array:
    inputs = (inputs - mu) / (sigma * math.sqrt(2.0))
    return 0.5 * (1.0 + jax.lax.erf(inputs))

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

"""Flax activation functions."""

import math
from collections import OrderedDict
from functools import partial
from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import Array


class PytorchGELUTanh(nn.Module):
  """
  A fast C implementation of the tanh approximation of the GeLU activation
  function. See https://huggingface.co/papers/1606.08415.

  This implementation is equivalent to NewGELU and FastGELU but much faster.
  However, it is not an exact numerical match due to rounding errors.
  """

  @nn.compact
  def __call__(self, input: Array) -> Array:
    return nn.gelu(input, approximate=True)


class NewGELUActivation(nn.Module):
  """
  Implementation of the GELU activation function currently in Google BERT repo
  (identical to OpenAI GPT). Also see the Gaussian Error Linear Units paper:
  https://huggingface.co/papers/1606.08415
  """

  @nn.compact
  def __call__(self, input: Array) -> Array:
    return (
        0.5
        * input
        * (
            1.0
            + jnp.tanh(
                math.sqrt(2.0 / math.pi)
                * (input + 0.044715 * jnp.power(input, 3.0))
            )
        )
    )


class GELUActivation(nn.Module):
  """
  Original Implementation of the GELU activation function in Google BERT repo
  when initially created. For information: OpenAI GPT's GELU is slightly
  different (and gives slightly different results): 0.5 * x * (1 +
  jnp.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * jnp.power(x, 3)))) This is
  now written in C in nn.functional Also see the Gaussian Error Linear Units
  paper: https://huggingface.co/papers/1606.08415
  """

  use_gelu_python: bool = False

  def _gelu_python(self, input: Array) -> Array:
    return input * 0.5 * (1.0 + jax.lax.erf(input / math.sqrt(2.0)))

  @nn.compact
  def __call__(self, input: Array) -> Array:
    if self.use_gelu_python:
      return self._gelu_python(input)
    else:
      return nn.gelu(input, approximate=False)


class FastGELUActivation(nn.Module):
  """
  Applies GELU approximation that is slower than QuickGELU but more accurate.
  See: https://github.com/hendrycks/GELUs
  """

  @nn.compact
  def __call__(self, input: Array) -> Array:
    return 0.5 * input * (
        1.0 + jnp.tanh(input * 0.7978845608 * (1.0 + 0.044715 * input * input))
    )


class QuickGELUActivation(nn.Module):
  """
  Applies GELU approximation that is fast but somewhat inaccurate. See:
  https://github.com/hendrycks/GELUs
  """

  @nn.compact
  def __call__(self, input: Array) -> Array:
    return input * nn.sigmoid(1.702 * input)


class ClippedGELUActivation(nn.Module):
  """
  Clip the range of possible GeLU outputs between [min, max]. This is
  especially useful for quantization purpose, as it allows mapping negatives
  values in the GeLU spectrum. For more information on this trick, please refer
  to https://huggingface.co/papers/2004.09602.

  Gaussian Error Linear Unit. Original Implementation of the gelu activation
  function in Google Bert repo when initially created.

  For information: OpenAI GPT's gelu is slightly different (and gives slightly
  different results): 0.5 * x * (1 + jnp.tanh(math.sqrt(2 / math.pi) * (x +
  0.044715 * jnp.power(x, 3)))). See
  https://huggingface.co/papers/1606.08415
  """

  min_val: float
  max_val: float

  def setup(self):
    if self.min_val > self.max_val:
      raise ValueError(
          f"min should be < max (got min: {self.min_val}, max:"
          f" {self.max_val})"
      )
    self.gelu = GELUActivation()

  def __call__(self, x: Array) -> Array:
    return jnp.clip(self.gelu(x), self.min_val, self.max_val)


class AccurateGELUActivation(nn.Module):
  """
  Applies GELU approximation that is faster than default and more accurate than
  QuickGELU. See: https://github.com/hendrycks/GELUs

  Implemented along with MEGA (Moving Average Equipped Gated Attention)
  """

  precomputed_constant: float

  def setup(self):
    self.precomputed_constant = math.sqrt(2 / math.pi)

  @nn.compact
  def __call__(self, input: Array) -> Array:
    return 0.5 * input * (
        1
        + jnp.tanh(
            self.precomputed_constant
            * (input + 0.044715 * jnp.power(input, 3))
        )
    )


class MishActivation(nn.Module):
  """
  See Mish: A Self-Regularized Non-Monotonic Activation Function (Misra.,
  https://huggingface.co/papers/1908.08681). Also visit the official repository
  for the paper: https://github.com/digantamisra98/Mish
  """

  @nn.compact
  def __call__(self, input: Array) -> Array:
    return nn.mish(input)


class LinearActivation(nn.Module):
  """
  Applies the linear activation function, i.e. forwarding input directly to
  output.
  """

  @nn.compact
  def __call__(self, input: Array) -> Array:
    return input


class LaplaceActivation(nn.Module):
  """
  Applies elementwise activation based on Laplace function, introduced in MEGA
  as an attention activation. See https://huggingface.co/papers/2209.10655

  Inspired by squared relu, but with bounded range and gradient for better
  stability
  """

  @nn.compact
  def __call__(
      self, input: Array, mu: float = 0.707107, sigma: float = 0.282095
  ) -> Array:
    input = (input - mu) / (sigma * math.sqrt(2.0))
    return 0.5 * (1.0 + jax.lax.erf(input))


class ReLUSquaredActivation(nn.Module):
  """
  Applies the relu^2 activation introduced in
  https://huggingface.co/papers/2109.08668v2
  """

  @nn.compact
  def __call__(self, input: Array) -> Array:
    relu_applied = nn.relu(input)
    squared = jnp.square(relu_applied)
    return squared


# Wrapper classes for functional activations to achieve consistency with PyTorch's nn.Module approach
class LeakyReLUActivation(nn.Module):
  negative_slope: float = 0.01

  @nn.compact
  def __call__(self, x: Array) -> Array:
    return nn.leaky_relu(x, self.negative_slope)


class ReLUActivation(nn.Module):
  @nn.compact
  def __call__(self, x: Array) -> Array:
    return nn.relu(x)


class ReLU6Activation(nn.Module):
  @nn.compact
  def __call__(self, x: Array) -> Array:
    return nn.relu6(x)


class SigmoidActivation(nn.Module):
  @nn.compact
  def __call__(self, x: Array) -> Array:
    return nn.sigmoid(x)


class SiLUActivation(nn.Module):
  @nn.compact
  def __call__(self, x: Array) -> Array:
    return nn.silu(x)


class TanhActivation(nn.Module):
  @nn.compact
  def __call__(self, x: Array) -> Array:
    return jnp.tanh(x)


class ClassInstantier(OrderedDict):
  def __getitem__(self, key):
    content = super().__getitem__(key)
    cls, kwargs = content if isinstance(content, tuple) else (content, {})
    return cls(**kwargs)


ACT2CLS = {
    "gelu": GELUActivation,
    "gelu_10": (ClippedGELUActivation, {"min_val": -10, "max_val": 10}),
    "gelu_fast": FastGELUActivation,
    "gelu_new": NewGELUActivation,
    "gelu_python": (GELUActivation, {"use_gelu_python": True}),
    "gelu_pytorch_tanh": PytorchGELUTanh,
    "gelu_accurate": AccurateGELUActivation,
    "laplace": LaplaceActivation,
    "leaky_relu": LeakyReLUActivation,
    "linear": LinearActivation,
    "mish": MishActivation,
    "quick_gelu": QuickGELUActivation,
    "relu": ReLUActivation,
    "relu2": ReLUSquaredActivation,
    "relu6": ReLU6Activation,
    "sigmoid": SigmoidActivation,
    "silu": SiLUActivation,
    "swish": SiLUActivation,
    "tanh": TanhActivation,
    "prelu": nn.PReLU,
}
ACT2FN = ClassInstantier(ACT2CLS)


def get_activation(activation_string: str) -> Callable:
  if activation_string in ACT2FN:
    return ACT2FN[activation_string]
  else:
    raise KeyError(
        f"function {activation_string} not found in ACT2FN mapping"
        f" {list(ACT2FN.keys())}"
    )


# For backwards compatibility
gelu_python = get_activation("gelu_python")
gelu_new = get_activation("gelu_new")
gelu = get_activation("gelu")
gelu_fast = get_activation("gelu_fast")
quick_gelu = get_activation("quick_gelu")
silu = get_activation("silu")
mish = get_activation("mish")
linear_act = get_activation("linear")

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

"""Activation functions, adapted from HuggingFace."""

import math
from collections import OrderedDict
from typing import Any

import jax
import jax.numpy as jnp
import flax.linen as nn

from MaxText.common_types import Array


class PytorchGELUTanh(nn.Module):
  """
  A fast approximation of the GeLU activation function using tanh.
  This is equivalent to `nn.gelu(x, approximate=True)`.
  """

  @nn.compact
  def __call__(self, x: Array) -> Array:
    return nn.gelu(x, approximate=True)


class NewGELUActivation(nn.Module):
  """
  Implementation of the GELU activation function currently in Google BERT repo
  (identical to OpenAI GPT). Also see the Gaussian Error Linear Units paper:
  https://huggingface.co/papers/1606.08415
  """

  @nn.compact
  def __call__(self, x: Array) -> Array:
    return 0.5 * x * (1.0 + jnp.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * jnp.power(x, 3.0))))


class GELUActivation(nn.Module):
  """
  Original Implementation of the GELU activation function in Google BERT repo
  when initially created. For information: OpenAI GPT's GELU is slightly
  different (and gives slightly different results).
  This is now written in C in nn.functional Also see the Gaussian Error Linear
  Units paper: https://huggingface.co/papers/1606.08415
  """

  use_gelu_python: bool = False

  def _gelu_python(self, x: Array) -> Array:
    return x * 0.5 * (1.0 + jax.lax.erf(x / math.sqrt(2.0)))

  @nn.compact
  def __call__(self, x: Array) -> Array:
    if self.use_gelu_python:
      return self._gelu_python(x)
    return nn.gelu(x)


class FastGELUActivation(nn.Module):
  """
  Applies GELU approximation that is slower than QuickGELU but more accurate.
  See: https://github.com/hendrycks/GELUs
  """

  @nn.compact
  def __call__(self, x: Array) -> Array:
    return 0.5 * x * (1.0 + jnp.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))


class QuickGELUActivation(nn.Module):
  """
  Applies GELU approximation that is fast but somewhat inaccurate.
  See: https://github.com/hendrycks/GELUs
  """

  @nn.compact
  def __call__(self, x: Array) -> Array:
    return x * jax.nn.sigmoid(1.702 * x)


class ClippedGELUActivation(nn.Module):
  """
  Clip the range of possible GeLU outputs between [min, max]. This is
  especially useful for quantization purpose.
  """

  min_val: float
  max_val: float

  def setup(self):
    if self.min_val > self.max_val:
      raise ValueError(f"min should be < max (got min: {self.min_val}, max: {self.max_val})")

  @nn.compact
  def __call__(self, x: Array) -> Array:
    return jnp.clip(nn.gelu(x), self.min_val, self.max_val)


class AccurateGELUActivation(nn.Module):
  """
  Applies GELU approximation that is faster than default and more accurate
  than QuickGELU. See: https://github.com/hendrycks/GELUs
  """

  precomputed_constant: float = math.sqrt(2 / math.pi)

  @nn.compact
  def __call__(self, x: Array) -> Array:
    return 0.5 * x * (1 + jnp.tanh(self.precomputed_constant * (x + 0.044715 * jnp.power(x, 3))))


class MishActivation(nn.Module):
  """
  Mish: A Self-Regularized Non-Monotonic Activation Function.
  See: https://huggingface.co/papers/1908.08681
  """

  @nn.compact
  def __call__(self, x: Array) -> Array:
    return nn.mish(x)


class LinearActivation(nn.Module):
  """
  Applies the linear activation function, i.e. forwarding input directly to
  output.
  """

  @nn.compact
  def __call__(self, x: Array) -> Array:
    return x


class LaplaceActivation(nn.Module):
  """
  Applies elementwise activation based on Laplace function, introduced in
  MEGA as an attention activation. See https://huggingface.co/papers/2209.10655
  """

  mu: float = 0.707107
  sigma: float = 0.282095

  @nn.compact
  def __call__(self, x: Array) -> Array:
    x = (x - self.mu) / (self.sigma * math.sqrt(2.0))
    return 0.5 * (1.0 + jax.lax.erf(x))


class ReLUSquaredActivation(nn.Module):
  """
  Applies the relu^2 activation introduced in
  https://huggingface.co/papers/2109.08668v2
  """

  @nn.compact
  def __call__(self, x: Array) -> Array:
    relu_applied = nn.relu(x)
    squared = jnp.square(relu_applied)
    return squared


class ClassInstantier(OrderedDict):
  """A dictionary that instantiates a class when a key is accessed."""

  def __getitem__(self, key: str) -> Any:
    content = super().__getitem__(key)
    cls, kwargs = content if isinstance(content, tuple) else (content, {})
    return cls(**kwargs)


ACT2CLS = {
    "gelu": GELUActivation,
    "gelu_10": (ClippedGELUActivation, {"min_val": -10, "max_val": 10}),
    "gelu_fast": FastGELUActivation,
    "gelu_new": NewGELUActivation,
    "gelu_python": (GELUActivation, {"use_gelu_python": True}),
    "gelu_pytorch_tanh": PytorchGELUTanh,
    "gelu_accurate": AccurateGELUActivation,
    "laplace": LaplaceActivation,
    "leaky_relu": nn.LeakyReLU,
    "linear": LinearActivation,
    "mish": MishActivation,
    "quick_gelu": QuickGELUActivation,
    "relu": nn.relu,
    "relu2": ReLUSquaredActivation,
    "relu6": nn.relu6,
    "sigmoid": nn.sigmoid,
    "silu": nn.silu,
    "swish": nn.silu,
    "tanh": nn.tanh,
    "prelu": nn.PReLU,
}
ACT2FN = ClassInstantier(ACT2CLS)
