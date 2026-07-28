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

"""Tests for the mHC-lite Pallas TPU kernel."""

import itertools
import math

import jax
import jax.numpy as jnp
from maxtext.kernels.residual import mhc_fwd_kernels as mhc
import numpy as np
import pytest


_NAMES = (
    "output",
    "x",
    "norm_scale",
    "pre_alpha",
    "pre_bias",
    "pre_scale",
    "post_alpha",
    "post_bias",
    "post_scale",
    "res_alpha",
    "res_bias",
    "res_scale",
    "branch_weight",
)
_BLOCK_SIZE = 8


def _make_inputs():
  """Builds deterministic production-dtype inputs for the kernel tests."""
  batch, sequence, streams, embedding = 1, 16, 4, 128
  flattened_size = streams * embedding
  permutation_count = math.factorial(streams)
  keys = jax.random.split(jax.random.key(0), 14)

  def random_bf16(index, shape, scale=1.0):
    return (jax.random.normal(keys[index], shape, jnp.float32) * scale).astype(jnp.bfloat16)

  fan_in = 1.0 / math.sqrt(flattened_size)
  x = random_bf16(0, (batch, sequence, streams, embedding))
  norm_scale = (1.0 + random_bf16(1, (flattened_size,), 0.1)).astype(jnp.bfloat16)
  pre_alpha = random_bf16(2, (flattened_size, streams), fan_in)
  pre_bias = random_bf16(3, (streams,), 0.1)
  pre_scale = (1.0 + random_bf16(4, (1,), 0.1)).astype(jnp.bfloat16)
  post_alpha = random_bf16(5, (flattened_size, streams), fan_in)
  post_bias = random_bf16(6, (streams,), 0.1)
  post_scale = (1.0 + random_bf16(7, (1,), 0.1)).astype(jnp.bfloat16)
  res_alpha = random_bf16(8, (flattened_size, permutation_count), fan_in)
  res_bias = random_bf16(9, (permutation_count,), 0.1)
  res_scale = (1.0 + random_bf16(10, (1,), 0.1)).astype(jnp.bfloat16)
  branch_weight = random_bf16(11, (embedding, embedding), 1.0 / math.sqrt(embedding))
  cotangent = random_bf16(12, (batch, sequence, streams, embedding))
  permutations = jnp.eye(streams, dtype=jnp.bfloat16)[jnp.array(list(itertools.permutations(range(streams))))]
  differentiable = (
      x,
      norm_scale,
      pre_alpha,
      pre_bias,
      pre_scale,
      post_alpha,
      post_bias,
      post_scale,
      res_alpha,
      res_bias,
      res_scale,
      branch_weight,
  )
  return differentiable, permutations, cotangent


def _kernel_pipeline(permutations, *args):
  """Runs the interpreted Pallas pipeline around a representative branch."""
  (
      x,
      norm_scale,
      pre_alpha,
      pre_bias,
      pre_scale,
      post_alpha,
      post_bias,
      post_scale,
      res_alpha,
      res_bias,
      res_scale,
      branch_weight,
  ) = args
  layer_input, context = mhc.pre(
      x,
      norm_scale,
      pre_alpha,
      pre_bias,
      pre_scale,
      post_alpha,
      post_bias,
      post_scale,
      res_alpha,
      res_bias,
      res_scale,
      permutations,
      rms_epsilon=1e-6,
      block_size=_BLOCK_SIZE,
      interpret=True,
  )
  layer_output = jnp.dot(layer_input, branch_weight, preferred_element_type=jnp.float32)
  return mhc.post(layer_output, context, block_size=_BLOCK_SIZE, interpret=True)


def _reference_pipeline(permutations, *args):
  """Runs the independent normalize-early mHC-lite reference."""
  (
      x,
      norm_scale,
      pre_alpha,
      pre_bias,
      pre_scale,
      post_alpha,
      post_bias,
      post_scale,
      res_alpha,
      res_bias,
      res_scale,
      branch_weight,
  ) = args
  batch, sequence, streams, embedding = x.shape
  tokens = batch * sequence
  flattened_size = streams * embedding
  x_flat = x.reshape(tokens, streams, embedding)
  flattened = x_flat.reshape(tokens, flattened_size)
  flattened_f32 = flattened.astype(jnp.float32)
  normalized = (
      flattened_f32
      * jax.lax.rsqrt(jnp.mean(flattened_f32 * flattened_f32, axis=-1, keepdims=True) + 1e-6)
      * norm_scale.astype(jnp.float32)
  ).astype(jnp.bfloat16)

  h_pre = (
      jax.nn.sigmoid(
          pre_scale.astype(jnp.float32) * jnp.dot(normalized, pre_alpha, preferred_element_type=jnp.float32)
          + pre_bias.astype(jnp.float32)
      )
      + 1e-6
  )
  layer_input = sum(
      h_pre[:, stream : stream + 1] * x_flat[:, stream, :].astype(jnp.float32) for stream in range(streams)
  ).astype(jnp.bfloat16)
  layer_output = jnp.dot(layer_input, branch_weight, preferred_element_type=jnp.float32)

  h_post = 2.0 * jax.nn.sigmoid(
      post_scale.astype(jnp.float32) * jnp.dot(normalized, post_alpha, preferred_element_type=jnp.float32)
      + post_bias.astype(jnp.float32)
  )
  weights = jax.nn.softmax(
      res_scale.astype(jnp.float32) * jnp.dot(normalized, res_alpha, preferred_element_type=jnp.float32)
      + res_bias.astype(jnp.float32),
      axis=-1,
  )
  residual = jnp.dot(
      weights,
      permutations.reshape(permutations.shape[0], streams * streams).astype(jnp.float32),
  ).reshape(tokens, streams, streams)
  residual_mix = jnp.einsum(
      "tkj,tkd->tjd",
      residual.astype(jnp.bfloat16),
      x_flat,
      preferred_element_type=jnp.float32,
  )
  post_mix = h_post[:, :, None] * layer_output[:, None, :].astype(jnp.float32)
  return (residual_mix + post_mix).astype(jnp.bfloat16).reshape(batch, sequence, streams, embedding)


def test_forward_and_backward_match_normalize_early_reference():
  differentiable, permutations, cotangent = _make_inputs()
  output, kernel_vjp = jax.vjp(lambda *args: _kernel_pipeline(permutations, *args), *differentiable)
  expected, reference_vjp = jax.vjp(
      lambda *args: _reference_pipeline(permutations, *args),
      *differentiable,
  )
  actual_values = (output,) + kernel_vjp(cotangent)
  expected_values = (expected,) + reference_vjp(cotangent)

  for name, actual, expected_value in zip(_NAMES, actual_values, expected_values, strict=True):
    actual_f32 = np.asarray(actual, np.float32)
    expected_f32 = np.asarray(expected_value, np.float32)
    scale = max(float(np.max(np.abs(expected_f32))), 1e-7)
    relative_error = float(np.max(np.abs(actual_f32 - expected_f32))) / scale
    assert relative_error <= 0.02, f"{name} relative error {relative_error:.4e} exceeded 2%"


@pytest.mark.parametrize(
    ("shape", "match"),
    (
        ((1, 15, 4, 128), "token count"),
        ((1, 16, 3, 128), "expansion rate 4"),
        ((1, 16, 4, 96), "divisible by 128"),
    ),
)
def test_pre_rejects_unsupported_shapes(shape, match):
  x = jnp.zeros(shape, jnp.bfloat16)
  streams, embedding = shape[-2:]
  permutations = jnp.zeros((math.factorial(streams), streams, streams), jnp.bfloat16)
  flattened_size = streams * embedding
  with pytest.raises(ValueError, match=match):
    mhc.pre(
        x,
        jnp.zeros((flattened_size,), jnp.bfloat16),
        jnp.zeros((flattened_size, streams), jnp.bfloat16),
        jnp.zeros((streams,), jnp.bfloat16),
        jnp.zeros((1,), jnp.bfloat16),
        jnp.zeros((flattened_size, streams), jnp.bfloat16),
        jnp.zeros((streams,), jnp.bfloat16),
        jnp.zeros((1,), jnp.bfloat16),
        jnp.zeros((flattened_size, math.factorial(streams)), jnp.bfloat16),
        jnp.zeros((math.factorial(streams),), jnp.bfloat16),
        jnp.zeros((1,), jnp.bfloat16),
        permutations,
        rms_epsilon=1e-6,
        interpret=True,
    )
