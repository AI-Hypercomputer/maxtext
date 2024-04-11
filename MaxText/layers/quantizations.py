#  Copyright 2024 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Quantization library."""

import functools

from aqt.jax.v2 import config as aqt_config
from aqt.jax.v2.flax import aqt_flax
from common_types import Array, Config
from dataclasses import dataclass
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten_with_path, tree_unflatten

MAX_INT8 = 127.5

@dataclass
class Quantization:
    """Base class for quantization configurations"""

    def dot_general_cls(self):
        """ Placeholder for dot_general implementation in subclasses. """
        pass


@dataclass
class AqtQuantization:
  """ Configures AQT quantization github.com/google/aqt. """
  quant_dg: aqt_config.DotGeneral
  quant_mode: aqt_flax.QuantMode = aqt_flax.QuantMode.TRAIN

  def dot_general_cls(self):
    """ Returns dot_general configured with aqt params. """
    aqt_dg_cls = functools.partial(
      aqt_flax.AqtDotGeneral,
      self.quant_dg,
      rhs_quant_mode=self.quant_mode
      )
    return aqt_dg_cls

  def einsum(self):
    """ Returns einsum configured with aqt params """
    aqt_einsum = functools.partial(aqt_flax.AqtEinsum(
      cfg=self.quant_dg,
      lhs_quant_mode=self.quant_mode
      )
    )
    return aqt_einsum

@dataclass
class Fp8Quantization(Quantization):
  """ Configures Fp8 quantization for NVIDIA GPUs"""
  quant_mode = "train"

  def dot_general_cls(self):
    """ Returns dot_general configured with aqt params. """
    return nn.Fp8DotGeneralOp

def _get_quant_config(config):
  """Set quantization params based on user configuration."""
  if not config.quantization or config.quantization == '':
    return None
  elif config.quantization == "int8":
    if config.quantization_local_shard_count == 0:
      drhs_bits = None
      drhs_accumulator_dtype = None
      drhs_local_aqt=None
    else:
      drhs_bits = 8
      drhs_accumulator_dtype = jnp.int32
      drhs_local_aqt = aqt_config.LocalAqt(config.quantization_local_shard_count)
    return aqt_config.config_v3(
      fwd_bits=8,
      dlhs_bits=8,
      drhs_bits=drhs_bits,
      rng_type='jax.uniform',
      dlhs_local_aqt=None,
      drhs_local_aqt=drhs_local_aqt,
      fwd_accumulator_dtype=jnp.int32,
      dlhs_accumulator_dtype=jnp.int32,
      drhs_accumulator_dtype=drhs_accumulator_dtype,
    )
  elif config.quantization == "fp8":
    return "fp8"
  else:
    raise ValueError(f'Invalid value configured for quantization {config.quantization}.')

def in_convert_mode(quant):
  return quant and (quant.quant_mode == aqt_flax.QuantMode.CONVERT)

def in_serve_mode(quant):
  return quant and (quant.quant_mode == aqt_flax.QuantMode.SERVE)

def get_quant_mode(quant_mode_str: str = 'train'):
  """ Set quant mode."""
  if quant_mode_str == 'train':
    return aqt_flax.QuantMode.TRAIN
  elif quant_mode_str == 'serve':
    return aqt_flax.QuantMode.SERVE
  elif quant_mode_str == 'convert':
    return aqt_flax.QuantMode.CONVERT
  else:
    raise ValueError(f'Invalid quantization mode {quant_mode_str}.')
  return None

def configure_quantization(config: Config, quant_mode_str: str = 'train'):
  """ Configure quantization based on user config and quant mode."""
  quant_cfg = _get_quant_config(config)
  if quant_cfg:
    if quant_cfg == "fp8":
        return Fp8Quantization()
    quant_mode = get_quant_mode(quant_mode_str)
    return AqtQuantization(quant_dg=quant_cfg, quant_mode=quant_mode)
  return None

def _get_aqt_key_paths(aqt_vars):
  """ Generate a list of paths which have aqt state """
  aqt_tree_flat, _ = jax.tree_util.tree_flatten_with_path(aqt_vars)
  aqt_key_paths = []
  for k, _ in aqt_tree_flat:
    pruned_keys = []
    for d in list(k):
      if 'AqtDotGeneral' in d.key:
        pruned_keys.append(jax.tree_util.DictKey(key='kernel'))
        break
      else:
        assert 'Aqt' not in d.key, f"Unexpected Aqt op {d.key} in {k}."
        pruned_keys.append(d)
    aqt_key_paths.append(tuple(pruned_keys))
  return aqt_key_paths


def remove_quantized_params(params, aqt_vars):
  """Remove param values with aqt tensors to Null to optimize memory."""
  aqt_paths = _get_aqt_key_paths(aqt_vars)
  tree_flat, tree_struct = tree_flatten_with_path(params)
  for i, (k, v) in enumerate(tree_flat):
    if k in aqt_paths:
      v = {}
    tree_flat[i] = v
  return tree_unflatten(tree_struct, tree_flat)

def configure_kv_quantization(config: Config):
  """ Configure kv quantization based on user config."""
  return False if not config.quantize_kvcache else True

def quantize_kv(kv: Array):
  """Quantize key/values stored in kvcache."""
  scale = jnp.max(jnp.abs(kv), axis=-1, keepdims=True)
  value = jnp.int8(jnp.rint(kv * (MAX_INT8 / scale)))
  return value, scale

def unquantize_kv(value: Array, scale:Array, dtype:jnp.dtype):
  """Unquantize key/values stored in kvcache."""
  return value.astype(dtype) * scale / MAX_INT8
