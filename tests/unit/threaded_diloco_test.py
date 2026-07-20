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

"""Unit tests for threaded DiLoCo components."""

import sys
import unittest
import threading
import time
import os
import subprocess
from maxtext.configs import pyconfig
from maxtext.trainers.diloco.threaded_diloco import make_learner_config
from maxtext.trainers.diloco.decomposed_transport import ThreadedTransportManager

class ThreadedDilocoUnitTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    # Need to add src to path if not already there, but maxtext imports usually assume src is in path.
    # We will initialize config with base.yml
    self.config = pyconfig.initialize(
        [sys.argv[0], "src/maxtext/configs/base.yml"],
        run_name="test",
        enable_diloco=True,
        enable_streaming_diloco=True,
        num_diloco_replicas=2,
    )

  def test_make_learner_config(self):
    learner_config = make_learner_config(self.config, learner_idx=1, num_learners=2)

    # Check that diloco is removed from mesh_axes
    self.assertNotIn("diloco", learner_config.mesh_axes)

    # Check logical_axis_rules
    for _, physical_axes in learner_config.logical_axis_rules:
      if isinstance(physical_axes, str):
        self.assertNotEqual(physical_axes, "diloco")
      elif isinstance(physical_axes, (list, tuple)):
        self.assertNotIn("diloco", physical_axes)

    # Check other flags
    self.assertTrue(learner_config.enable_local_data_loading)
    self.assertEqual(learner_config.learner_idx, 1)
    self.assertEqual(learner_config.num_learners, 2)
    self.assertFalse(learner_config.enable_streaming_diloco)
    self.assertFalse(learner_config.enable_diloco)

  def test_transport_manager_basic(self):
    manager = ThreadedTransportManager(num_learners=2)

    # Test learner to syncer
    manager.send_to_syncer(learner_idx=0, step=1, fragment_id=1, data="l0_s1_f1")
    manager.send_to_syncer(learner_idx=1, step=1, fragment_id=1, data="l1_s1_f1")

    self.assertEqual(manager.recv_from_learner(learner_idx=0, step=1, fragment_id=1), "l0_s1_f1")
    self.assertEqual(manager.recv_from_learner(learner_idx=1, step=1, fragment_id=1), "l1_s1_f1")

    # Test syncer to learner
    manager.send_to_learner(learner_idx=0, step=1, fragment_id=1, data="s_l0_s1_f1")
    manager.send_to_learner(learner_idx=1, step=1, fragment_id=1, data="s_l1_s1_f1")

    self.assertEqual(manager.recv_from_syncer(learner_idx=0, step=1, fragment_id=1), "s_l0_s1_f1")
    self.assertEqual(manager.recv_from_syncer(learner_idx=1, step=1, fragment_id=1), "s_l1_s1_f1")

  def test_transport_manager_out_of_order(self):
    manager = ThreadedTransportManager(num_learners=1)

    # Send out of order
    manager.send_to_syncer(learner_idx=0, step=2, fragment_id=1, data="step2")
    manager.send_to_syncer(learner_idx=0, step=1, fragment_id=1, data="step1")

    # Receive in order
    self.assertEqual(manager.recv_from_learner(learner_idx=0, step=1, fragment_id=1), "step1")
    self.assertEqual(manager.recv_from_learner(learner_idx=0, step=2, fragment_id=1), "step2")

  def test_transport_manager_blocking(self):
    manager = ThreadedTransportManager(num_learners=1)
    results = {}

    def worker():
      results['data'] = manager.recv_from_learner(learner_idx=0, step=1, fragment_id=1)

    t = threading.Thread(target=worker)
    t.start()

    # Sleep to ensure worker is blocked
    time.sleep(0.1)
    self.assertTrue(t.is_alive())
    self.assertNotIn('data', results)

    # Send data
    manager.send_to_syncer(learner_idx=0, step=1, fragment_id=1, data="blocked_data")
    t.join(timeout=1.0)

    self.assertFalse(t.is_alive())
    self.assertEqual(results['data'], "blocked_data")

  def test_syncer_replication_on_multi_device_cpu(self):
    """Replicates syncer values without creating a broadcast executable."""
    script = r"""
import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental.layout import Format, Layout
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from maxtext.trainers.diloco.decomposed_transport import ThreadedTransportManager, LearnerTransport
from maxtext.utils.mesh_utils import replicate_across_submeshes_pytree, stack_across_meshes_pytree

devices = jax.devices("cpu")
assert len(devices) == 4, devices
device_grid = np.asarray(devices).reshape(2, 2)
submeshes = [Mesh(device_grid[i], ("fsdp",)) for i in range(2)]
mesh = Mesh(device_grid, ("diloco", "fsdp"))
source_sharding = NamedSharding(mesh, P(None, "fsdp"))
source_format = Format(Layout((1, 0)), source_sharding)
source = jax.device_put(np.arange(32, dtype=np.float32).reshape(4, 8), source_format)
global_value = {"weight": source}
replicas = replicate_across_submeshes_pytree(global_value, submeshes)
assert len(replicas) == 2
for replica, submesh in zip(replicas, submeshes, strict=True):
  np.testing.assert_array_equal(np.asarray(replica["weight"]), np.arange(32).reshape(4, 8))
  assert replica["weight"].sharding == NamedSharding(submesh, P(None, "fsdp"))
  assert replica["weight"].format.layout.major_to_minor == (1, 0)
np.testing.assert_array_equal(np.asarray(source), np.arange(32).reshape(4, 8))

# Pathways concatenate always donates its inputs. Verify the transport owns a
# distinct buffer even when learner and colocated meshes share CPU devices.
manager = ThreadedTransportManager(2)
transports = [LearnerTransport(manager, i, submeshes[i]) for i in range(2)]
originals = []
for i, transport in enumerate(transports):
  original = jax.device_put(
      jnp.arange(32, dtype=jnp.float32).reshape(4, 8), NamedSharding(submeshes[i], P(None, "fsdp"))
  )
  originals.append(original)
  transport.send_to_syncer(step=1, fragment_id=0, data={"weight": original})
fragments = [manager.recv_from_learner(i, step=1, fragment_id=0) for i in range(2)]
stacked = stack_across_meshes_pytree(fragments, mesh, "diloco")
jax.block_until_ready(stacked)
consume = jax.jit(lambda value: value + 1, donate_argnums=0)
for fragment in fragments:
  consume(fragment["weight"]).block_until_ready()
for original in originals:
  np.testing.assert_array_equal(np.asarray(original), np.arange(32).reshape(4, 8))
for transport in transports:
  transport.close()
"""
    env = os.environ.copy()
    existing_flags = env.get("XLA_FLAGS", "")
    env["XLA_FLAGS"] = f"{existing_flags} --xla_force_host_platform_device_count=4".strip()
    subprocess.run([sys.executable, "-c", script], check=True, env=env, timeout=120)

  def test_layout_canonicalized_jit_accepts_different_committed_layouts(self):
    """Reproduces a committed-layout mismatch and verifies canonicalization."""
    script = r"""
import numpy as np
import jax
import optax
from jax.experimental.layout import Format, Layout
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from maxtext.trainers.diloco.threaded_diloco import _jit_with_layout_canonicalized_inputs, make_step_fns
from maxtext.utils.mesh_utils import _expand_array_dims_with_mesh

devices = np.asarray(jax.devices("cpu"))
assert len(devices) == 4, devices
mesh = Mesh(devices.reshape(2, 2), ("diloco", "fsdp"))
sharding = NamedSharding(mesh, P(None, "fsdp"))
row_major = Format(Layout((0, 1)), sharding)
column_major = Format(Layout((1, 0)), sharding)
first = jax.device_put(np.arange(32, dtype=np.float32).reshape(4, 8), row_major)
second = jax.device_put(np.arange(32, dtype=np.float32).reshape(4, 8), column_major)

# A normal precompiled executable has a strict physical-layout ABI even though
# both arguments have identical shape, dtype, devices, and logical sharding.
raw = jax.jit(lambda x: x + 1, in_shardings=sharding, out_shardings=sharding)
executable = raw.lower(first).compile()
np.testing.assert_array_equal(np.asarray(executable(first)), np.arange(32).reshape(4, 8) + 1)
try:
  executable(second)
except ValueError as error:
  assert "layout" in str(error).lower(), error
else:
  raise AssertionError("Expected a committed physical-layout mismatch")

# The production wrapper reads executable.input_formats and relays every call
# through those exact formats. Calls with either source layout must succeed.
safe = _jit_with_layout_canonicalized_inputs(
    lambda x: x + 1,
    in_shardings=sharding,
    out_shardings=sharding,
)
np.testing.assert_array_equal(np.asarray(safe(first)), np.arange(32).reshape(4, 8) + 1)
np.testing.assert_array_equal(np.asarray(safe(second)), np.arange(32).reshape(4, 8) + 1)

# Reproduce the TPU report more literally with a rank-1 default layout versus
# an explicit T(256)-style physical tile. CPU PJRT supports this layout syntax
# even though it does not implement TPU memory.
vector_sharding = NamedSharding(mesh, P())
default_vector_format = Format(Layout((0,)), vector_sharding)
tiled_vector_format = Format(Layout((0,), tiling=((256,),)), vector_sharding)
default_vector = jax.device_put(np.arange(256, dtype=np.float32), default_vector_format)
tiled_vector = jax.device_put(np.arange(256, dtype=np.float32), tiled_vector_format)
assert default_vector.format.layout.tiling == ()
assert tiled_vector.format.layout.tiling == ((256,),)

tiled_raw = jax.jit(
    lambda x: x + 1,
    in_shardings=tiled_vector_format,
    out_shardings=tiled_vector_format,
)
tiled_executable = tiled_raw.lower(tiled_vector).compile()
try:
  tiled_executable(default_vector)
except ValueError as error:
  assert "layout" in str(error).lower(), error
else:
  raise AssertionError("Expected the default-layout versus T(256) mismatch")

tiled_safe = _jit_with_layout_canonicalized_inputs(
    lambda x: x + 1,
    in_shardings=tiled_vector_format,
    out_shardings=tiled_vector_format,
)
np.testing.assert_array_equal(np.asarray(tiled_safe(tiled_vector)), np.arange(256) + 1)
np.testing.assert_array_equal(np.asarray(tiled_safe(default_vector)), np.arange(256) + 1)

# Fragment stacking expands each learner value before invoking the Pathways
# concatenate primitive. Verify that this earlier compiled boundary also
# accepts alternating committed input layouts.
learner_mesh = Mesh(devices[:2], ("fsdp",))
learner_sharding = NamedSharding(learner_mesh, P(None, "fsdp"))
learner_row = jax.device_put(
    np.arange(32, dtype=np.float32).reshape(4, 8), Format(Layout((0, 1)), learner_sharding)
)
learner_column = jax.device_put(
    np.arange(32, dtype=np.float32).reshape(4, 8), Format(Layout((1, 0)), learner_sharding)
)
for learner_value in (learner_row, learner_column):
  expanded = _expand_array_dims_with_mesh(learner_value, "diloco")
  np.testing.assert_array_equal(np.asarray(expanded), np.arange(32).reshape(1, 4, 8))
  assert expanded.sharding.spec == P("diloco", None, "fsdp")

# Exercise the actual syncer gradient and Nesterov-SGD pytrees, including a
# second call whose committed input layouts differ from the first call.
optimizer = optax.sgd(learning_rate=0.1, momentum=0.9, nesterov=True)
compute_grad, apply_outer_step = make_step_fns(
    mesh, {"weight": sharding}, ["weight"], optimizer
)
stacked_sharding = NamedSharding(mesh, P("diloco", None, "fsdp"))
stacked_row = Format(Layout((0, 1, 2)), stacked_sharding)
stacked_column = Format(Layout((2, 1, 0)), stacked_sharding)

def run_outer_step(param_array, stacked_format):
  params = {"weight": param_array}
  stacked = {
      "weight": jax.device_put(
          np.stack([np.ones((4, 8), np.float32), np.full((4, 8), 2, np.float32)]),
          stacked_format,
      )
  }
  grad = compute_grad(params, stacked)
  np.testing.assert_allclose(np.asarray(grad["weight"]), np.asarray(param_array) - 1.5)
  new_params, new_opt_state = apply_outer_step(grad, optimizer.init(params), params)
  jax.block_until_ready((new_params, new_opt_state))
  assert new_params["weight"].shape == (4, 8)

run_outer_step(first, stacked_row)
run_outer_step(second, stacked_column)
"""
    env = os.environ.copy()
    existing_flags = env.get("XLA_FLAGS", "")
    env["XLA_FLAGS"] = f"{existing_flags} --xla_force_host_platform_device_count=4".strip()
    subprocess.run([sys.executable, "-c", script], check=True, env=env, timeout=120)

if __name__ == "__main__":
  unittest.main()
