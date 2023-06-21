# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Configuration dataclasses."""

import dataclasses
from typing import Any, Callable, Optional, Union
import jax
import jax.numpy as jnp

DType = Any
Context = Any  # TODO(lew): We could put Context in a separate file.

FreshScaleFn = Callable[[jnp.ndarray], jnp.ndarray]
ClipAndRoundFn = Callable[[jnp.ndarray, Context], jnp.ndarray]
NoiseFn = Callable[[tuple, jax.random.KeyArray], jnp.ndarray]


@dataclasses.dataclass
class NoNumerics:
  """No quantization, use a native type such as bf16."""

  pass


@dataclasses.dataclass
class IntNumerics:
  bits: int
  preserve_zero: bool


Numerics = Union[NoNumerics, IntNumerics]


@dataclasses.dataclass
class Tensor:
  """Configuration of quantization of one tensor or one side of tensor op."""

  numerics: Numerics
  calib_shared_axes: Optional[list]
  bound: Optional[float]
  bound_stop_grad: bool
  # false = map max val on the end of the last bucket
  # true = map max val on the middle of the last
  preserve_max_val: bool
  clip: bool
  round: bool
  # noise+clip+round
  # We apply gradient of clip_and_round in bwd pass.
  clip_and_round: Optional[ClipAndRoundFn]
  fresh_scale: Optional[FreshScaleFn]
  noise_fn: Optional[NoiseFn]
  # Round up the calibration to power of 2 (po2).
  po2_scale: bool
  use_fake_quant: bool
  dtype: DType

  @classmethod
  def make(cls, bits: Optional[int]) -> 'Tensor':
    """Makes."""
    if bits is None:
      numerics = NoNumerics()
    else:
      pz = False if bits == 1 else True
      numerics = IntNumerics(bits=bits, preserve_zero=pz)

    if (
        bits is not None
        and bits <= 8
        and bits != 1  # we currently round to -0.5 and 0.5 for 1 bit (pz)
    ):
      dtype = jnp.int8
    else:
      dtype = jnp.bfloat16

    return Tensor(
        numerics=numerics,
        calib_shared_axes=None,
        bound=None,
        bound_stop_grad=True,
        preserve_max_val=False,
        clip=True,
        round=True,
        clip_and_round=None,
        fresh_scale=None,
        noise_fn=None,
        po2_scale=False,
        use_fake_quant=False,
        dtype=dtype,
    )


@dataclasses.dataclass
class DotGeneralRaw:
  """Configuration of quantization of one dot_general without gradient."""

  lhs: Tensor
  rhs: Tensor
  # use_fwd_quant is observed when this dot_general is used in gradient.
  # use_fwd_quant is ignored in forward pass.
  # Whether the gradient should be taken at unquantized wgt/act or quantized.
  use_fwd_quant: bool

  @classmethod
  def make(cls, lhs_bits=None, rhs_bits=None) -> 'DotGeneralRaw':
    """Create quantization configs for input matrices to a matmul."""
    return DotGeneralRaw(
        lhs=Tensor.make(lhs_bits),
        rhs=Tensor.make(rhs_bits),
        use_fwd_quant=True,
    )

  @classmethod
  def make_conv_general_dilated(
      cls,
      spatial_dimensions=2,
      lhs_bits: Optional[int] = None,
      rhs_bits: Optional[int] = None,
  ) -> 'DotGeneralRaw':
    """Create quantization config conv_general_dilated."""
    config = cls.make(lhs_bits, rhs_bits)
    # Hardcoding flax assumptions.
    if config.lhs:
      config.lhs.calib_shared_axes = list(range(1, spatial_dimensions + 2))
    if config.rhs:
      config.rhs.calib_shared_axes = list(range(0, spatial_dimensions + 2 - 1))
    return config


@dataclasses.dataclass
class DotGeneral:
  """Configuration of quantization of dot_general and its gradients."""
  fwd: DotGeneralRaw
  dlhs: DotGeneralRaw
  drhs: DotGeneralRaw

  @classmethod
  def make(
      cls,
      lhs_bits: Optional[int] = None,
      rhs_bits: Optional[int] = None,
  ) -> 'DotGeneral':
    """Create quantization configs for input matrices to a matmul."""
    return cls(
        fwd=DotGeneralRaw.make(lhs_bits, rhs_bits),
        dlhs=DotGeneralRaw.make(),
        drhs=DotGeneralRaw.make(),
    )


def fully_quantized(
    *,
    fwd_bits: int = 8,
    bwd_bits: int = 8,
    use_fwd_quant: bool = True,
    use_stochastic_rounding: bool = True,
    use_dummy_static_bound: bool = False,
) -> DotGeneral:
  """Fully Quantized Training."""
  cfg = DotGeneral(
      fwd=DotGeneralRaw.make(fwd_bits, fwd_bits),
      dlhs=DotGeneralRaw.make(bwd_bits, bwd_bits),
      drhs=DotGeneralRaw.make(bwd_bits, bwd_bits),
  )
  cfg.fwd.use_fwd_quant = use_fwd_quant
  cfg.dlhs.use_fwd_quant = use_fwd_quant
  cfg.drhs.use_fwd_quant = use_fwd_quant

  if use_stochastic_rounding:

    def noise_fn(shape, key):
      return jax.random.uniform(key, shape) - 0.5

    cfg.dlhs.lhs.noise_fn = noise_fn
    cfg.dlhs.rhs.noise_fn = noise_fn
    cfg.drhs.lhs.noise_fn = noise_fn
    cfg.drhs.rhs.noise_fn = noise_fn

  if use_dummy_static_bound:
    cfg.fwd.lhs.bound = 1.0
    cfg.fwd.rhs.bound = 1.0
    cfg.drhs.lhs.bound = 1.0
    cfg.drhs.rhs.bound = 1.0
    cfg.dlhs.lhs.bound = 1.0
    cfg.dlhs.rhs.bound = 1.0

  return cfg
