# Copyright 2023–2025 Google LLC
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

"""Tests for the vocab-parallel LM head (`lm_head_vocab_parallel`).

The head's two orientations differ only in where the FSDP axes sit, so anything
they disagree about numerically is a bug. These tests pin that down end to end:
the same kernel, the same targets, and the same loss, computed once in each
layout on a multi-device mesh.
"""

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

from flax import linen as nn  # pylint: disable=wrong-import-position
import jax  # pylint: disable=wrong-import-position
import jax.numpy as jnp  # pylint: disable=wrong-import-position
from jax.sharding import AxisType, Mesh  # pylint: disable=wrong-import-position
import numpy as np  # pylint: disable=wrong-import-position
import pytest  # pylint: disable=wrong-import-position

from maxtext.common.common_types import ShardMode  # pylint: disable=wrong-import-position
from maxtext.configs import pyconfig  # pylint: disable=wrong-import-position
from maxtext.utils import max_utils  # pylint: disable=wrong-import-position
from maxtext.utils import sharding  # pylint: disable=wrong-import-position
from maxtext.utils.globals import MAXTEXT_PKG_DIR  # pylint: disable=wrong-import-position

from tests.utils.test_helpers import get_test_config_path  # pylint: disable=wrong-import-position

pytestmark = pytest.mark.cpu_only

_FSDP = 4
_BATCH, _LENGTH, _EMBED, _VOCAB = 4, 8, 16, 32
# base.yml's mesh, collapsed to the one axis that matters here. `create_sharding` drops
# the size-one axes, so the shipped rules resolve exactly as they do in a real run.
_MESH_AXES = (
    "diloco",
    "data",
    "stage",
    "fsdp",
    "fsdp_transpose",
    "context",
    "context_usp_ulysses",
    "context_autoregressive",
    "tensor",
    "tensor_sequence",
    "expert",
    "autoregressive",
)


def _rules():
  """The shipped `logical_axis_rules`, including the four the flag adds."""
  return pyconfig.initialize(
      [os.path.join(MAXTEXT_PKG_DIR, "train.py"), get_test_config_path()],
      skip_jax_distributed_system=True,
      shard_mode="explicit",
  ).logical_axis_rules


def _mesh(axis_type):
  devices = np.array(jax.devices()).reshape([_FSDP if name == "fsdp" else 1 for name in _MESH_AXES])
  return Mesh(devices, _MESH_AXES, axis_types=tuple(axis_type for _ in _MESH_AXES))


def _skip_unless_multi_device():
  if jax.device_count() < _FSDP:
    pytest.skip(f"needs {_FSDP} devices; set XLA_FLAGS=--xla_force_host_platform_device_count={_FSDP}")


@pytest.mark.parametrize("shard_mode", (ShardMode.EXPLICIT, ShardMode.AUTO))
def test_vocab_parallel_one_hot_matches_jax_nn_one_hot(shard_mode):
  """The shard-local iota comparison builds exactly the matrix `jax.nn.one_hot` would."""
  _skip_unless_multi_device()
  axis_type = AxisType.Explicit if shard_mode == ShardMode.EXPLICIT else AxisType.Auto
  mesh = _mesh(axis_type)
  logical_axes = sharding.lm_head_logical_axes(vocab_parallel=True)[2]
  targets = jnp.asarray(np.random.default_rng(0).integers(0, _VOCAB, (_BATCH, _LENGTH)), jnp.int32)

  def build(t):
    return sharding.vocab_parallel_one_hot(t, _VOCAB, mesh, logical_axes, shard_mode)

  with jax.set_mesh(mesh), nn.logical_axis_rules(_rules()):
    got = jax.jit(build)(targets)

  np.testing.assert_array_equal(np.asarray(got), np.asarray(jax.nn.one_hot(targets, _VOCAB)))
  # And it is laid out for a vocab-parallel head: FSDP on the vocab axis, not on batch.
  spec = got.sharding.spec
  assert "fsdp" in jax.tree.leaves(spec[-1] or ())
  assert "fsdp" not in jax.tree.leaves(spec[0] or ())


def test_vocab_parallel_loss_matches_default_orientation():
  """Turning the head sideways changes the layout of the loss, not its value.

  Runs the head and cross entropy twice over one mesh -- once with the kernel
  sharded on `embed` and batch-sharded logits, once with the kernel sharded on
  `vocab` and vocab-sharded logits -- from the same weights and the same targets.
  """
  _skip_unless_multi_device()
  mesh = _mesh(AxisType.Explicit)
  rng = np.random.default_rng(0)
  hidden = jnp.asarray(rng.normal(size=(_BATCH, _LENGTH, _EMBED)), jnp.float32)
  kernel = jnp.asarray(rng.normal(size=(_EMBED, _VOCAB)), jnp.float32)
  targets = jnp.asarray(rng.integers(0, _VOCAB, (_BATCH, _LENGTH)), jnp.int32)

  def loss(y, w, t, vocab_parallel):
    in_axes, kernel_axes, out_axes = sharding.lm_head_logical_axes(vocab_parallel)
    w = sharding.maybe_shard_with_logical(w, kernel_axes, mesh, ShardMode.EXPLICIT)
    if in_axes is not None:
      y = sharding.maybe_shard_with_logical(y, in_axes, mesh, ShardMode.EXPLICIT)
    logits = jnp.einsum("ble,ev->blv", y, w, out_sharding=sharding.create_sharding(mesh, out_axes))
    if vocab_parallel:
      one_hot = sharding.vocab_parallel_one_hot(t, _VOCAB, mesh, out_axes, ShardMode.EXPLICIT)
    else:
      one_hot = jax.nn.one_hot(t, _VOCAB)
    xent, _ = max_utils.cross_entropy_with_logits(logits, one_hot, z_loss=0.0)
    return jnp.sum(xent)

  with jax.set_mesh(mesh), nn.logical_axis_rules(_rules()):
    grad = jax.jit(jax.value_and_grad(loss, argnums=1), static_argnums=3)
    default_loss, default_grad = grad(hidden, kernel, targets, False)
    parallel_loss, parallel_grad = grad(hidden, kernel, targets, True)
    # The kernel gradient comes back in each orientation's own layout: sharded on
    # `embed` by default, on `vocab` when the head is turned sideways.
    assert "fsdp" in jax.tree.leaves(default_grad.sharding.spec[0] or ())
    assert "fsdp" in jax.tree.leaves(parallel_grad.sharding.spec[1] or ())

  np.testing.assert_allclose(float(parallel_loss), float(default_loss), rtol=1e-6)
  np.testing.assert_allclose(np.asarray(parallel_grad), np.asarray(default_grad), rtol=1e-5, atol=1e-6)


def test_lm_head_logical_axes_orientations_differ_only_in_fsdp_placement():
  """The sideways kernel takes the FSDP axes off `embed` and puts them on `vocab`."""
  rules = _rules()
  _, default_kernel, default_out = sharding.lm_head_logical_axes(vocab_parallel=False)
  parallel_in, parallel_kernel, parallel_out = sharding.lm_head_logical_axes(vocab_parallel=True)

  def axes(logical_name):
    return set(nn.logical_to_mesh_axes((logical_name,), rules=rules)[0] or ())

  assert "fsdp" in axes(default_kernel[0]) and "fsdp" not in axes(default_kernel[1])
  assert "fsdp" not in axes(parallel_kernel[0]) and "fsdp" in axes(parallel_kernel[1])
  # ... and off the batch axis of everything the sideways head touches, so that the
  # hidden state is gathered once instead of the kernel.
  assert "fsdp" in axes(default_out[0])
  assert "fsdp" not in axes(parallel_out[0])
  assert parallel_in[0] == parallel_out[0]
