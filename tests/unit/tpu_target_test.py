# Copyright 2026 Google LLC
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

"""Tests that kernels choose their TPU implementation by compile target, not by host backend.

An ahead-of-time compile for a TPU topology runs on a CPU host. These tests trace on the CPU backend
with and without a TPU target, as `train_compile` and `maxtext_engine_compile` do, and check which
implementation each kernel lowers to.
"""

import contextlib
import re
import unittest

from absl.testing import parameterized
import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
import pytest

from maxtext.configs import pyconfig
from maxtext.kernels.gdn.gdn_bwd import api as gdn_api
from maxtext.kernels.gdn.gdn_bwd import pallas_mosaic_tpu_bwd as gdn_bwd
from maxtext.kernels.ragged import ragged_gather
from maxtext.kernels.ragged import ragged_gather_reduce
from maxtext.kernels.ragged import ragged_gather_reduce_v2
from maxtext.trainers.pre_train import train_compile
from tests.utils.test_helpers import get_test_config_path

# GDN shapes the Pallas kernels accept: head dims a multiple of 128 and a sequence of whole 64-token chunks.
_GDN_BATCH, _GDN_SEQ, _GDN_CHUNK, _GDN_HEAD_DIM, _GDN_CONV = 1, 64, 64, 128, 4
_GDN_WIDTH = 3 * _GDN_HEAD_DIM  # One query, one key and one value head.


def _tpu_target(device_kind="TPU7x"):
  """Returns a context in which traces target a TPU of `device_kind`, as they do in an AOT compile."""
  device = jax.sharding.AbstractDevice(device_kind=device_kind, num_cores=1, platform="tpu")
  return jax.sharding.use_abstract_mesh(jax.sharding.AbstractMesh((), (), abstract_device=device))


def _tpu_kernel_names(fn, *args):
  """Returns the names of the TPU kernels `fn` compiles to when lowered for a TPU."""
  lowered = jax.jit(fn).trace(*args).lower(lowering_platforms=("tpu",))
  return re.findall(r'kernel_name = "([^"]+)"', lowered.as_text())


def _ragged_gather(x, indices):
  return ragged_gather.ragged_gather(x, indices, jnp.array(0), jnp.array(x.shape[0]))


def _ragged_gather_reduce(x, indices, weights, mask):
  return ragged_gather_reduce.ragged_gather_reduce(x, indices, weights, mask, reduce_group_size=4)


def _ragged_gather_reduce_v2(x, indices, weights, mask):
  return ragged_gather_reduce_v2.ragged_gather_reduce(x, indices, weights, mask, reduce_group_size=4)


def _gdn_forward(qkv, b, a):
  return gdn_api._run_local_gdn_decoupled_fwd(  # pylint: disable=protected-access
      qkv,
      b,
      a,
      jnp.zeros((_GDN_CONV, 1, _GDN_WIDTH)),
      jnp.zeros((_GDN_WIDTH,)),
      jnp.zeros((1,)),
      jnp.zeros((1,)),
      None,
      None,
      num_k_heads=1,
      num_v_heads=1,
      head_k_dim=_GDN_HEAD_DIM,
      head_v_dim=_GDN_HEAD_DIM,
      conv_kernel_size=_GDN_CONV,
      chunk_size=_GDN_CHUNK,
      use_qk_norm_in_gdn=False,
      compute_dtype=jnp.float32,
  )


def _gdn_backward(qkv, b, a):
  num_chunks = _GDN_SEQ // _GDN_CHUNK
  return gdn_bwd.pallas_gdn_bwd_kernel(
      qkv,
      b,
      a,
      jnp.zeros((1,)),
      jnp.zeros((1,)),
      jnp.zeros((_GDN_BATCH, _GDN_SEQ, 1, _GDN_HEAD_DIM)),
      jnp.zeros((_GDN_BATCH, num_chunks, 1, _GDN_HEAD_DIM, _GDN_HEAD_DIM)),
      jnp.zeros((_GDN_BATCH, num_chunks, 1, _GDN_CHUNK, _GDN_CHUNK)),
      num_v_heads=1,
      kq_head_dim=_GDN_HEAD_DIM,
      v_head_dim=_GDN_HEAD_DIM,
      chunk_size=_GDN_CHUNK,
  )


_ROWS, _HIDDEN, _GATHERED = 256, 4096, 1024
_GATHER_ARGS = (jax.ShapeDtypeStruct((_ROWS, _HIDDEN), jnp.bfloat16), jax.ShapeDtypeStruct((_GATHERED,), jnp.int32))
_GATHER_REDUCE_ARGS = _GATHER_ARGS + (
    jax.ShapeDtypeStruct((_GATHERED,), jnp.float32),
    jax.ShapeDtypeStruct((_GATHERED,), jnp.bool_),
)
_GDN_ARGS = (
    jax.ShapeDtypeStruct((_GDN_BATCH, _GDN_SEQ, _GDN_WIDTH), jnp.float32),
    jax.ShapeDtypeStruct((_GDN_BATCH, _GDN_SEQ, 1), jnp.float32),
    jax.ShapeDtypeStruct((_GDN_BATCH, _GDN_SEQ, 1), jnp.float32),
)

# The ragged gathers choose by `pltpu.is_tpu_device`, the GDN forward and backward kernels by
# `gdn_bwd.runtime_utils.target_platform`; both must follow the compile target.
_KERNELS = (
    ("ragged_gather", _ragged_gather, _GATHER_ARGS, "sc_ragged_gather"),
    ("ragged_gather_reduce", _ragged_gather_reduce, _GATHER_REDUCE_ARGS, "sc_ragged_gather_reduce"),
    ("ragged_gather_reduce_v2", _ragged_gather_reduce_v2, _GATHER_REDUCE_ARGS, "sc_ragged_gather_reduce_v2"),
    ("gdn_forward", _gdn_forward, _GDN_ARGS, "fused_conv1d_gdn"),
    ("gdn_backward", _gdn_backward, _GDN_ARGS, "gdn_bwd_kernel"),
)


class KernelSelectionTest(parameterized.TestCase):
  """Each kernel lowers to its TPU kernel under a TPU target and to its fallback without one."""

  @parameterized.named_parameters(*_KERNELS)
  def test_tpu_target_gets_the_tpu_kernel(self, fn, args, kernel_name):
    with _tpu_target():
      names = _tpu_kernel_names(fn, *args)
    self.assertLen(names, 1)
    self.assertTrue(names[0].startswith(kernel_name), names)

  @parameterized.named_parameters(*_KERNELS)
  def test_cpu_host_without_a_target_gets_the_fallback(self, fn, args, kernel_name):
    del kernel_name
    self.assertEqual(_tpu_kernel_names(fn, *args), [])

  @parameterized.named_parameters(*_KERNELS[:3])
  def test_cpu_mesh_gets_the_fallback(self, fn, args, kernel_name):
    """A mesh of CPU devices is not a TPU target, so the ragged gathers keep their fallback under it."""
    del kernel_name
    with jax.set_mesh(jax.make_mesh((1,), ("x",))):
      self.assertEqual(_tpu_kernel_names(fn, *args), [])

  def test_kernel_tiles_for_the_target_chip(self):
    """A v7x SparseCore has twice a v6e's SIMD lanes, so the kernel pads 200 gathered rows to 512, not 256."""
    indices = jax.ShapeDtypeStruct((200,), jnp.int32)

    def gathered_rows(device_kind):
      with _tpu_target(device_kind):
        lowered = jax.jit(_ragged_gather).trace(_GATHER_ARGS[0], indices).lower(lowering_platforms=("tpu",))
      (call,) = [line for line in lowered.as_text().splitlines() if "tpu_custom_call" in line]
      return int(re.search(r"-> tensor<(\d+)x", call).group(1))

    self.assertEqual(gathered_rows("TPU7x"), 512)
    self.assertEqual(gathered_rows("TPU v6 lite"), 256)


@pytest.mark.tpu_backend
class TrainCompileKernelTest(unittest.TestCase):
  """`train_compile` compiles the TPU kernel for a TPU topology on this CPU host."""

  def setUp(self):
    super().setUp()
    self.config = pyconfig.initialize(
        ["", get_test_config_path(), "compile_topology=tpu7x-8", "compile_topology_num_slices=1"]
    )
    self.mesh = train_compile.get_topology_mesh(self.config)

  def test_topology_mesh_is_the_target(self):
    with jax.set_mesh(self.mesh):
      self.assertTrue(pltpu.is_tpu_device())
      self.assertEqual(pltpu.get_tpu_info().generation, 7)

  def test_compiled_program_has_the_sparsecore_kernel(self):
    axes = tuple(name for name in self.mesh.axis_names if self.mesh.shape[name] > 1)
    gather = jax.shard_map(_ragged_gather, mesh=self.mesh, in_specs=(P(), P(axes)), out_specs=P(axes), check_vma=False)
    args = (
        jax.ShapeDtypeStruct((_ROWS, _HIDDEN), jnp.bfloat16, sharding=NamedSharding(self.mesh, P())),
        jax.ShapeDtypeStruct((self.mesh.size * 512,), jnp.int32, sharding=NamedSharding(self.mesh, P(axes))),
    )
    compiled = train_compile.jit_and_compile(
        gather, args, {}, self.mesh, None, None, (), (), self.config, contextlib.nullcontext()
    )
    self.assertIn("sc_ragged_gather", compiled.as_text())


if __name__ == "__main__":
  unittest.main()
