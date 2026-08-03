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
import numpy as np
import pytest
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from maxtext.kernels.residual import mhc_fwd_kernels as mhc

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
_BWD_BLOCK_SIZE = 16
_BWD_FEATURE_BLOCK_SIZE = 128
_REDUCED_NAMES = frozenset(("pre_bias", "pre_scale", "post_bias", "post_scale", "res_bias", "res_scale"))


def _make_inputs(*, batch=1, sequence=16, streams=4, embedding=256):
  """Builds deterministic production-dtype inputs for the kernel tests."""
  flattened_size = streams * embedding
  permutation_count = math.factorial(streams)
  keys = jax.random.split(jax.random.key(0), 14)

  def random_bf16(index, shape, scale=1.0):
    return (jax.random.normal(keys[index], shape, jnp.float32) * scale).astype(jnp.bfloat16)

  def random_f32(index, shape, scale=1.0):
    return jax.random.normal(keys[index], shape, jnp.float32) * scale

  fan_in = 1.0 / math.sqrt(flattened_size)
  x = random_bf16(0, (batch, sequence, streams, embedding))
  norm_scale = 1.0 + random_f32(1, (flattened_size,), 0.1)
  pre_alpha = random_f32(2, (flattened_size, streams), fan_in)
  pre_bias = random_f32(3, (streams,), 0.1)
  pre_scale = 1.0 + random_f32(4, (1,), 0.1)
  post_alpha = random_f32(5, (flattened_size, streams), fan_in)
  post_bias = random_f32(6, (streams,), 0.1)
  post_scale = 1.0 + random_f32(7, (1,), 0.1)
  res_alpha = random_f32(8, (flattened_size, permutation_count), fan_in)
  res_bias = random_f32(9, (permutation_count,), 0.1)
  res_scale = 1.0 + random_f32(10, (1,), 0.1)
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


def _kernel_pipeline(
    permutations,
    *args,
    block_size=_BLOCK_SIZE,
    bwd_block_size=_BWD_BLOCK_SIZE,
    post_bwd_block_size=_BWD_BLOCK_SIZE,
    bwd_feature_block_size=_BWD_FEATURE_BLOCK_SIZE,
    interpret=True,
):
  """Runs the Pallas pipeline around a representative branch."""
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
      block_size=block_size,
      bwd_block_size=bwd_block_size,
      interpret=interpret,
  )
  # MaxText's attention and MLP branches emit config.dtype, so exercise a
  # bfloat16 branch output here rather than the float32 the raw dot produces.
  layer_output = jnp.dot(layer_input, branch_weight, preferred_element_type=jnp.float32).astype(jnp.bfloat16)
  return mhc.post(
      layer_output,
      context,
      block_size=block_size,
      bwd_block_size=post_bwd_block_size,
      bwd_feature_block_size=bwd_feature_block_size,
      interpret=interpret,
  )


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
  layer_output = jnp.dot(layer_input, branch_weight, preferred_element_type=jnp.float32).astype(jnp.bfloat16)

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


def _assert_allclose(name, actual, expected):
  """Checks an array using the hardware-tested, scale-aware tolerances."""
  tolerance = 0.05 if name in _REDUCED_NAMES else 0.02
  actual_f32 = np.asarray(actual, np.float32)
  expected_f32 = np.asarray(expected, np.float32)
  scale = max(float(np.max(np.abs(expected_f32))), 1e-7)
  relative_error = float(np.max(np.abs(actual_f32 - expected_f32))) / scale
  assert relative_error <= tolerance, f"{name} relative error {relative_error:.4e} exceeded {tolerance:.0%}"


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
    _assert_allclose(name, actual, expected_value)


@pytest.mark.tpu_only
@pytest.mark.skipif(jax.default_backend() != "tpu", reason="requires TPU")
def test_production_tiles_match_reference_on_tpu():
  """Compiles the production tiles on a compact representative shape."""
  # This compact shape exercises the same 64/32/256/1024 tiles as the
  # DeepSeek-V4 production shape without its multi-gigabyte working set.
  differentiable, permutations, cotangent = _make_inputs(sequence=256, embedding=1024)

  @jax.jit
  def run(*args):
    output, kernel_vjp = jax.vjp(
        lambda *primals: _kernel_pipeline(
            permutations,
            *primals,
            block_size=mhc.DEFAULT_BLOCK_SIZE,
            bwd_block_size=mhc.DEFAULT_BWD_BLOCK_SIZE,
            post_bwd_block_size=mhc.DEFAULT_POST_BWD_BLOCK_SIZE,
            bwd_feature_block_size=mhc.DEFAULT_POST_BWD_FEATURE_BLOCK_SIZE,
            interpret=False,
        ),
        *args,
    )
    return (output,) + kernel_vjp(cotangent)

  actual_values = run(*differentiable)
  expected, reference_vjp = jax.vjp(
      lambda *args: _reference_pipeline(permutations, *args),
      *differentiable,
  )
  expected_values = (expected,) + reference_vjp(cotangent)
  for name, actual, expected_value in zip(_NAMES, actual_values, expected_values, strict=True):
    _assert_allclose(name, actual, expected_value)


@pytest.mark.skipif(jax.device_count() < 2, reason="requires at least two devices")
@pytest.mark.parametrize(
    ("axis_name", "x_spec"), (("batch", P("shard", None, None, None)), ("sequence", P(None, "shard", None, None)))
)
def test_shard_map_runs_on_local_token_shards(axis_name, x_spec):
  """Checks separate pre/post shard_maps, including context and gradient assembly."""
  differentiable, permutations, cotangent = _make_inputs()
  repeat_axis = 0 if axis_name == "batch" else 1
  differentiable = (
      jnp.repeat(differentiable[0], 2, axis=repeat_axis),
      *differentiable[1:],
  )
  cotangent = jnp.repeat(cotangent, 2, axis=repeat_axis)
  mesh = Mesh(np.asarray(jax.devices()[:2]), ("shard",))
  # Virtual CPU devices use the interpreter; real TPU CI compiles Mosaic.
  interpret = jax.default_backend() != "tpu"
  layer_spec = P("shard", None, None) if axis_name == "batch" else P(None, "shard", None)
  coefficient_spec = layer_spec
  residual_spec = P("shard", None, None, None) if axis_name == "batch" else P(None, "shard", None, None)
  context_spec = (x_spec, coefficient_spec, residual_spec)

  def sharded_pipeline(*args):
    def local_pre(*local_args):
      local_x = local_args[0]
      expected_local_size = 1 if axis_name == "batch" else 16
      assert local_x.shape[repeat_axis] == expected_local_size
      return mhc.pre(
          *local_args,
          permutations,
          rms_epsilon=1e-6,
          block_size=_BLOCK_SIZE,
          bwd_block_size=_BWD_BLOCK_SIZE,
          interpret=interpret,
      )

    layer_input, context = jax.shard_map(
        local_pre,
        mesh=mesh,
        in_specs=(x_spec,) + (P(),) * 10,
        out_specs=(layer_spec, context_spec),
        check_vma=False,
    )(*args[:-1])
    branch_weight = args[-1]
    layer_output = jnp.dot(layer_input, branch_weight, preferred_element_type=jnp.float32).astype(jnp.bfloat16)

    def local_post(local_layer_output, local_context):
      return mhc.post(
          local_layer_output,
          local_context,
          block_size=_BLOCK_SIZE,
          bwd_block_size=_BWD_BLOCK_SIZE,
          bwd_feature_block_size=_BWD_FEATURE_BLOCK_SIZE,
          interpret=interpret,
      )

    return jax.shard_map(
        local_post,
        mesh=mesh,
        in_specs=(layer_spec, context_spec),
        out_specs=x_spec,
        check_vma=False,
    )(layer_output, context)

  output, kernel_vjp = jax.vjp(sharded_pipeline, *differentiable)
  expected, reference_vjp = jax.vjp(
      lambda *args: _reference_pipeline(permutations, *args),
      *differentiable,
  )
  actual_values = (output,) + kernel_vjp(cotangent)
  expected_values = (expected,) + reference_vjp(cotangent)

  for name, actual, expected_value in zip(_NAMES, actual_values, expected_values, strict=True):
    _assert_allclose(name, actual, expected_value)


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


@pytest.mark.parametrize(
    ("kwargs", "match"),
    (
        ({"bwd_block_size": 10, "bwd_feature_block_size": 128}, "positive multiple of 8"),
        ({"bwd_block_size": 16, "bwd_feature_block_size": 192}, "positive multiple of 128"),
        ({"bwd_block_size": 16, "bwd_feature_block_size": 384}, "embedding dimension"),
    ),
)
def test_post_rejects_unsupported_backward_blocks(kwargs, match):
  x = jnp.zeros((1, 16, 4, 256), jnp.bfloat16)
  context = (
      x,
      jnp.zeros((1, 16, 4), jnp.float32),
      jnp.zeros((1, 16, 4, 4), jnp.float32),
  )
  with pytest.raises(ValueError, match=match):
    mhc.post(
        jnp.zeros((1, 16, 256), jnp.bfloat16),
        context,
        block_size=8,
        interpret=True,
        **kwargs,
    )


def _reduced_parameter_grads(parameter_dtype, sequence=2048):
  """Returns pipeline gradients with the mHC parameters cast to a given dtype."""
  batch, streams, embedding = 1, 4, 256
  flattened_size = streams * embedding
  permutation_count = math.factorial(streams)
  keys = jax.random.split(jax.random.key(1), 13)
  fan_in = 1.0 / math.sqrt(flattened_size)

  def parameter(index, shape, scale=1.0, offset=0.0):
    value = offset + jax.random.normal(keys[index], shape, jnp.float32) * scale
    return value.astype(parameter_dtype)

  permutations = jnp.eye(streams, dtype=jnp.bfloat16)[jnp.array(list(itertools.permutations(range(streams))))]
  x = (jax.random.normal(keys[0], (batch, sequence, streams, embedding), jnp.float32)).astype(jnp.bfloat16)
  branch_weight = (jax.random.normal(keys[11], (embedding, embedding), jnp.float32) / math.sqrt(embedding)).astype(
      jnp.bfloat16
  )
  differentiable = (
      x,
      parameter(1, (flattened_size,), 0.1, 1.0),
      parameter(2, (flattened_size, streams), fan_in),
      parameter(3, (streams,), 0.1),
      parameter(4, (1,), 0.1, 1.0),
      parameter(5, (flattened_size, streams), fan_in),
      parameter(6, (streams,), 0.1),
      parameter(7, (1,), 0.1, 1.0),
      parameter(8, (flattened_size, permutation_count), fan_in),
      parameter(9, (permutation_count,), 0.1),
      parameter(10, (1,), 0.1, 1.0),
      branch_weight,
  )

  def pipeline(*args):
    x, norm_scale, pre_a, pre_b, pre_s, post_a, post_b, post_s, res_a, res_b, res_s, weight = args
    layer_input, context = mhc.pre(
        x,
        norm_scale,
        pre_a,
        pre_b,
        pre_s,
        post_a,
        post_b,
        post_s,
        res_a,
        res_b,
        res_s,
        permutations,
        rms_epsilon=1e-6,
        block_size=8,
        bwd_block_size=8,
        interpret=True,
    )
    layer_output = jnp.dot(layer_input, weight, preferred_element_type=jnp.float32).astype(jnp.bfloat16)
    return mhc.post(
        layer_output,
        context,
        block_size=8,
        bwd_block_size=256,
        bwd_feature_block_size=_BWD_FEATURE_BLOCK_SIZE,
        interpret=True,
    )

  output, vjp = jax.vjp(pipeline, *differentiable)
  cotangent = jax.random.normal(keys[12 % len(keys)], output.shape, jnp.float32).astype(output.dtype)
  return vjp(cotangent)


def test_parameter_grads_survive_low_precision_parameters():
  """Pins float32 accumulation of the token-reduced parameter gradients.

  These six gradients are summed once per backward grid program. An accumulator
  that inherited a bfloat16 parameter dtype would round the running total once
  per program -- 256 times here -- instead of once at the end, which XLA never
  does. The error that introduces grows with the token count and reached 9% at
  the DeepSeek-V4 shape, so it is invisible in a few-token test.
  """
  golden = _reduced_parameter_grads(jnp.float32)
  low_precision = _reduced_parameter_grads(jnp.bfloat16)

  for name, expected, actual in zip(_NAMES[1:], golden, low_precision, strict=True):
    if name not in _REDUCED_NAMES:
      continue
    expected_f32 = np.asarray(expected, np.float32)
    actual_f32 = np.asarray(actual, np.float32)
    scale = max(float(np.max(np.abs(expected_f32))), 1e-7)
    relative_error = float(np.max(np.abs(actual_f32 - expected_f32))) / scale
    assert relative_error <= 0.02, f"{name} relative error {relative_error:.4e} exceeded 2%"
