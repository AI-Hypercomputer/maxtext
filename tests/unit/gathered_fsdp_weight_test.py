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

"""Tests for the gathered-FSDP-weight remat residual (`save_fsdp_gathered_weights`).

The transform moves an all-gather that XLA would have emitted anyway, so the only
things it may change are where the gather happens, what dtype it happens in, and
whether remat can keep the result. It must not change which values come out. These
tests pin all four, in both shard modes, on a four-device mesh.
"""

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

from flax import linen as nn  # pylint: disable=wrong-import-position
from flax import nnx  # pylint: disable=wrong-import-position
import jax  # pylint: disable=wrong-import-position
from jax.ad_checkpoint import checkpoint_name  # pylint: disable=wrong-import-position
import jax.numpy as jnp  # pylint: disable=wrong-import-position
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec  # pylint: disable=wrong-import-position
import numpy as np  # pylint: disable=wrong-import-position
import pytest  # pylint: disable=wrong-import-position

from maxtext.configs import pyconfig  # pylint: disable=wrong-import-position
from maxtext.layers import nnx_decoders  # pylint: disable=wrong-import-position
from maxtext.utils.globals import MAXTEXT_PKG_DIR  # pylint: disable=wrong-import-position

from tests.utils.test_helpers import get_test_config_path  # pylint: disable=wrong-import-position

pytestmark = pytest.mark.cpu_only

_FSDP = 4
_EMBED, _MLP = 8, 16
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


def _config(shard_mode, enabled=True):
  return pyconfig.initialize(
      [os.path.join(MAXTEXT_PKG_DIR, "train.py"), get_test_config_path()],
      skip_jax_distributed_system=True,
      shard_mode=shard_mode,
      save_fsdp_gathered_weights=enabled,
  )


def _mesh(shard_mode):
  axis_type = AxisType.Explicit if shard_mode == "explicit" else AxisType.Auto
  devices = np.array(jax.devices()).reshape([_FSDP if name == "fsdp" else 1 for name in _MESH_AXES])
  return Mesh(devices, _MESH_AXES, axis_types=tuple(axis_type for _ in _MESH_AXES))


def _skip_unless_multi_device():
  if jax.device_count() < _FSDP:
    pytest.skip(f"needs {_FSDP} devices; set XLA_FLAGS=--xla_force_host_platform_device_count={_FSDP}")


def _kernel():
  """A wi-shaped kernel: FSDP on `embed`, i.e. on the major dimension."""
  return jnp.asarray(np.random.default_rng(0).normal(size=(_EMBED, _MLP)), jnp.float32)


def _hoist(params, mesh, config):
  """Run the transform inside jit, under the mesh and the shipped logical rules."""
  with jax.set_mesh(mesh), nn.logical_axis_rules(config.logical_axis_rules):
    return jax.jit(lambda p: nnx_decoders.hoist_fsdp_weight_gather(p, mesh, config))(params)


def _explicit_params(kernel, mesh):
  return {"wi": jax.device_put(kernel, NamedSharding(mesh, PartitionSpec("fsdp", None)))}


def _auto_params(kernel, mesh, names=("embed", "mlp")):
  """nnx params carrying the logical axis names, the way scanned layers do."""
  sharded = jax.device_put(kernel, NamedSharding(mesh, PartitionSpec("fsdp", None)))
  return {"wi": nnx.Param(sharded, sharding_names=names)}


def _value(params):
  leaf = params["wi"]
  return leaf.get_value() if isinstance(leaf, nnx.Variable) else leaf


@pytest.mark.parametrize("shard_mode", ("explicit", "auto"))
def test_gathered_kernel_is_unsharded_and_in_compute_dtype(shard_mode):
  """The FSDP axis comes off the kernel and the value arrives in config.dtype."""
  _skip_unless_multi_device()
  config, mesh = _config(shard_mode), _mesh(shard_mode)
  kernel = _kernel()
  params = _explicit_params(kernel, mesh) if shard_mode == "explicit" else _auto_params(kernel, mesh)

  got = _value(_hoist(params, mesh, config))

  assert got.dtype == config.dtype
  assert "fsdp" not in jax.tree.leaves(got.sharding.spec or ())
  # The gather is the only thing that happened: the values still match, to bf16.
  np.testing.assert_allclose(np.asarray(got, np.float32), np.asarray(kernel), rtol=1e-2)


@pytest.mark.parametrize("shard_mode", ("explicit", "auto"))
def test_disabled_flag_is_the_identity(shard_mode):
  """With the flag off nothing is rewritten, not even the dtype."""
  _skip_unless_multi_device()
  config, mesh = _config(shard_mode, enabled=False), _mesh(shard_mode)
  kernel = _kernel()
  params = _explicit_params(kernel, mesh) if shard_mode == "explicit" else _auto_params(kernel, mesh)

  got = _value(_hoist(params, mesh, config))

  assert got.dtype == jnp.float32
  np.testing.assert_array_equal(np.asarray(got), np.asarray(kernel))


@pytest.mark.parametrize("shard_mode", ("explicit", "auto"))
def test_rank_one_scales_keep_their_float32_accumulation(shard_mode):
  """Norm scales are rank 1 and never FSDP-gathered, so they must come out untouched."""
  _skip_unless_multi_device()
  config, mesh = _config(shard_mode), _mesh(shard_mode)
  scale = jnp.asarray(np.random.default_rng(1).normal(size=(_EMBED,)), jnp.float32)
  if shard_mode == "explicit":
    params = {"wi": jax.device_put(scale, NamedSharding(mesh, PartitionSpec("fsdp")))}
  else:
    params = {
        "wi": nnx.Param(jax.device_put(scale, NamedSharding(mesh, PartitionSpec("fsdp"))), sharding_names=("embed",))
    }

  got = _value(_hoist(params, mesh, config))

  assert got.dtype == jnp.float32
  np.testing.assert_array_equal(np.asarray(got), np.asarray(scale))


@pytest.mark.parametrize("shard_mode", ("explicit", "auto"))
def test_gathered_kernel_is_named_for_the_remat_policy(shard_mode):
  """The gather is tagged, otherwise the policy below has nothing to keep."""
  _skip_unless_multi_device()
  config, mesh = _config(shard_mode), _mesh(shard_mode)
  params = _explicit_params(_kernel(), mesh) if shard_mode == "explicit" else _auto_params(_kernel(), mesh)

  with jax.set_mesh(mesh), nn.logical_axis_rules(config.logical_axis_rules):
    jaxpr = jax.make_jaxpr(lambda p: nnx_decoders.hoist_fsdp_weight_gather(p, mesh, config))(params)

  assert nnx_decoders._GATHERED_FSDP_WEIGHT in str(jaxpr)  # pylint: disable=protected-access


def test_policy_keeps_the_gathered_weight_alongside_the_configured_one():
  """Extending a policy adds the gathered weights without dropping what it already saved."""
  base = jax.checkpoint_policies.save_only_these_names("query_proj")
  extended = nnx_decoders.save_gathered_fsdp_weights_policy(base)
  # The primitive `checkpoint_name` emits, which is what a policy is asked about.
  name_p = jax.make_jaxpr(lambda x: checkpoint_name(x, "probe"))(1.0).eqns[0].primitive

  def saves(policy, name):
    return policy(name_p, name=name)

  assert saves(extended, nnx_decoders._GATHERED_FSDP_WEIGHT)  # pylint: disable=protected-access
  assert saves(extended, "query_proj")
  assert not saves(extended, "mlpwi")
  # And it is still well defined when no policy was configured at all.
  assert saves(nnx_decoders.save_gathered_fsdp_weights_policy(None), nnx_decoders._GATHERED_FSDP_WEIGHT)  # pylint: disable=protected-access


def test_gathering_does_not_change_what_the_layer_computes():
  """A dot against the hoisted kernel gives what a dot against the sharded one gives.

  This is the property the flag rests on: the consumer casts to config.dtype anyway,
  so pulling that cast in front of the gather is arithmetically a no-op.
  """
  _skip_unless_multi_device()
  config, mesh = _config("explicit"), _mesh("explicit")
  kernel = _kernel()
  inputs = jnp.asarray(np.random.default_rng(2).normal(size=(2, _EMBED)), jnp.float32)

  def apply(params, x):
    kern = nnx_decoders.hoist_fsdp_weight_gather(params, mesh, config)["wi"]
    return jnp.dot(jnp.asarray(x, config.dtype), jnp.asarray(kern, config.dtype))

  with jax.set_mesh(mesh), nn.logical_axis_rules(config.logical_axis_rules):
    sharded = _explicit_params(kernel, mesh)
    hoisted = jax.jit(apply)(sharded, inputs)
    plain = jax.jit(lambda x, k: jnp.dot(jnp.asarray(x, config.dtype), jnp.asarray(k, config.dtype)))(inputs, kernel)

  np.testing.assert_array_equal(np.asarray(hoisted), np.asarray(plain))
