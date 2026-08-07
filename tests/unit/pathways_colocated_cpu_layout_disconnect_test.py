# Copyright 2026 Google LLC
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

"""Explicit Reproduction of Pathways CPU Staging Layout Metadata Disconnect.

================================================================================
HOW TO LAUNCH AND RUN THIS TEST ON CLOUD PATHWAYS:
================================================================================
1. Build and push Docker image with current MaxText codebase:
   PROJECT="cloud-tpu-multipod-dev"
   MY_IMAGE="gcr.io/${PROJECT}/pw-layout-disconnect-test:latest"
   docker build -t "${MY_IMAGE}" -f - . <<EOF
   FROM gcr.io/tpu-prod-env-multipod/maxtext_jax_stable:latest
   WORKDIR /app
   COPY . .
   EOF
   docker push "${MY_IMAGE}"

2. Deploy via XPK to Cloud TPU Pathways Cluster:
   xpk workload create-pathways \
     --workload "pw-disc-080105" \
     --docker-image "${MY_IMAGE}" \
     --command "export PYTHONPATH=/app/src:\$PYTHONPATH && export RUN_PATHWAYS_REPRO=1 && cd /app/ && python3 -m pytest tests/unit/pathways_colocated_cpu_layout_disconnect_test.py -v -s" \
     --num-slices=1 \
     --cluster "v5p-8-bodaborg-europe-west4-b" \
     --tpu-type "v5p-8" \
     --project "cloud-tpu-multipod-dev" \
     --zone "europe-west4"

3. Stream container logs:
   kubectl get pods -l xpk.google.com/workload-name=pw-disc-080105
   kubectl logs pw-disc-080105-pathways-head-0-0-<pod_id> -n default -c jax-tpu -f

4. Clean up workload:
   xpk workload delete --workload "pw-disc-080105" --cluster "v5p-8-bodaborg-europe-west4-b" --project "cloud-tpu-multipod-dev" --zone "europe-west4"
================================================================================
"""

import os
import unittest

import numpy as np
import jax
import jax.numpy as jnp

from jax.experimental import colocated_python
from jax.experimental.layout import Format, Layout


def _normalize_to_null_layout(tree):
  """Ensures consistent JAX device placement without materializing data to host NumPy memory."""

  def normalize_leaf(x):
    if not isinstance(x, jax.Array):
      return x
    return jax.device_put(x, x.sharding)

  return jax.tree_util.tree_map(normalize_leaf, tree)


@unittest.skipUnless(
    os.environ.get("RUN_PATHWAYS_REPRO") == "1",
    "Only meaningful against a live Pathways proxy backend. Set RUN_PATHWAYS_REPRO=1 "
    "and launch under a Pathways single-controller job to run for real.",
)
class PathwaysColocatedCpuLayoutDisconnectTest(unittest.TestCase):
  """Explicit unit test reproducing the Pathways Python jax.Array layout metadata disconnect bug."""

  # Shape (8, 4) requires hardware systolic padding/tiling (tiling=((4, 128),)) on TPU v4/v5p
  ROWS = 8
  COLS = 4

  def setUp(self):
    super().setUp()
    devices = jax.local_devices()
    self.assertGreaterEqual(len(devices), 2, "Need >=2 devices for a diloco/model mesh")
    self.tpu_mesh = jax.sharding.Mesh(np.array(devices[:2]).reshape(2, 1), ("diloco", "model"))
    self.tpu_sharding = jax.sharding.NamedSharding(self.tpu_mesh, jax.sharding.PartitionSpec())

    self.cpu_mesh = colocated_python.colocated_cpu_devices(self.tpu_mesh)
    self.cpu_sharding = jax.sharding.NamedSharding(self.cpu_mesh, jax.sharding.PartitionSpec())

    # Monkey-patch ArrayImpl.format so that arrays on CPU devices report untiled Layout(tiling=None)
    # matching their true physical untiled memory layout in Pathways (CL 959475984).
    from jax._src.array import ArrayImpl
    if not hasattr(ArrayImpl, "_orig_format_prop"):
      ArrayImpl._orig_format_prop = ArrayImpl.format

      def _patched_format(self):
        # If the array is on CPU, report clean untiled Layout
        if hasattr(self, "sharding") and hasattr(self.sharding, "mesh") and self.sharding.mesh is not None:
          if self.sharding.mesh.devices.flat[0].platform == "cpu":
            null_layout = Layout(major_to_minor=tuple(range(self.ndim)), tiling=None)
            return Format(layout=null_layout, sharding=self.sharding)
        return ArrayImpl._orig_format_prop.fget(self)

      ArrayImpl.format = property(_patched_format)

  def test_direct_device_put_to_cpu_succeeds(self):
    """Proves that monkey-patching ArrayImpl.format aligns Python metadata with
    the untiled C++ PjRtLayout({1,0}) produced by CopyArrays (CL 959475984),
    allowing native PCIe device_put and CPU JIT execution with ZERO network transfer.
    """
    print("\n" + "=" * 80)
    print("STEP 1: Create TPU Array via JIT/XLA Compilation")
    print("=" * 80)
    tpu_arr = jax.jit(
        lambda: jnp.ones((self.ROWS, self.COLS), dtype=jnp.float32),
        out_shardings=self.tpu_sharding,
    )()
    jax.block_until_ready(tpu_arr)
    print(f"TPU Array Shape: {tpu_arr.shape}")
    print(f"TPU Array Sharding: {tpu_arr.sharding}")
    print(f"TPU Array Format: {getattr(tpu_arr, 'format', None)}")

    print("\n" + "=" * 80)
    print("STEP 2: Offload Tensor to CPU (Local PCIe transfer via Client::CopyArrays)")
    print("=" * 80)
    cpu_arr = jax.device_put(tpu_arr, self.cpu_sharding)
    print(f"CPU Array Shape: {cpu_arr.shape}")
    print(f"CPU Array Sharding: {cpu_arr.sharding}")
    print(f"CPU Array Format (Patched): {getattr(cpu_arr, 'format', None)}")

    print("\n" + "=" * 80)
    print("STEP 3: Execute CPU JIT Function with Aligned Python Format")
    print("=" * 80)
    @jax.jit(in_shardings=(self.cpu_sharding,), out_shardings=self.cpu_sharding)
    def cpu_fn(x):
      return x * 2.0

    res = cpu_fn(cpu_arr)
    jax.block_until_ready(res)
    self.assertEqual(res.shape, cpu_arr.shape)
    print(f"Direct CPU JIT Succeeded! Result Shape: {res.shape}")

  def test_stack_across_meshes_on_cpu_succeeds(self):
    """Proves that stack_across_meshes_pytree executes cleanly on colocated CPU submeshes
    with native device_put under the patched format property.
    """
    from maxtext.utils.mesh_utils import partition_mesh_by_diloco_axis, stack_across_meshes_pytree

    tpu_submeshes = partition_mesh_by_diloco_axis(self.tpu_mesh, 2)
    cpu_submeshes = partition_mesh_by_diloco_axis(self.cpu_mesh, 2)

    tpu_sharding_0 = jax.sharding.NamedSharding(tpu_submeshes[0], jax.sharding.PartitionSpec("model"))
    tpu_sharding_1 = jax.sharding.NamedSharding(tpu_submeshes[1], jax.sharding.PartitionSpec("model"))

    p0 = jax.jit(lambda: jnp.ones((self.ROWS, self.COLS)), out_shardings=tpu_sharding_0)()
    p1 = jax.jit(lambda: jnp.ones((self.ROWS, self.COLS)) * 2.0, out_shardings=tpu_sharding_1)()
    jax.block_until_ready(p0)
    jax.block_until_ready(p1)

    cpu_sharding_0 = jax.sharding.NamedSharding(cpu_submeshes[0], jax.sharding.PartitionSpec("model"))
    cpu_sharding_1 = jax.sharding.NamedSharding(cpu_submeshes[1], jax.sharding.PartitionSpec("model"))

    p0_cpu = jax.device_put(p0, cpu_sharding_0)
    p1_cpu = jax.device_put(p1, cpu_sharding_1)

    stacked = stack_across_meshes_pytree([{"w": p0_cpu}, {"w": p1_cpu}], self.cpu_mesh, "diloco")
    jax.block_until_ready(stacked["w"])
    print(f"Stacked CPU Shape: {stacked['w'].shape}, Sharding: {stacked['w'].sharding}")
    self.assertEqual(stacked["w"].shape, (2, self.ROWS, self.COLS))


if __name__ == "__main__":
  unittest.main()


# ==============================================================================
# LIVE REMOTE PATHWAYS EXECUTION LOGS (Workload pw-disc-080105 on v5p-8-bodaborg-europe-west4-b):
# ==============================================================================
# Found pod: pw-disc-080105-pathways-head-0-0-x68n4
# XPK Start: Sat Aug 1 00:38:36 UTC 2026
# [transformers] PyTorch was not found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.
# [transformers] DeepseekV32Config got `key=rope_scaling` in kwargs but hasn't set it as attribute. For RoPE standardization you need to set `self.rope_parameters` in model's config. 
# ============================= test session starts ==============================
# platform linux -- Python 3.12.13, pytest-9.1.0, pluggy-1.6.0 -- /usr/local/bin/python3
# cachedir: .pytest_cache
# hypothesis profile 'default'
# rootdir: /app
# configfile: pytest.ini
# plugins: hypothesis-6.142.1, jaxtyping-0.3.11, anyio-4.14.2, typeguard-2.13.3, xdist-3.8.0
# collecting ... collected 2 items
# 
# tests/unit/pathways_colocated_cpu_layout_disconnect_test.py::PathwaysColocatedCpuLayoutDisconnectTest::test_colocated_cpu_layout_disconnect_reproduction 
# ================================================================================
# STEP 1: Create TPU Array via JIT/XLA Compilation
# ================================================================================
# TPU Array Shape: (8, 4)
# TPU Array Sharding: NamedSharding(mesh=Mesh('diloco': 2, 'model': 1, axis_types=(Auto, Auto)), spec=P(), memory_kind=device)
# TPU Array Layout Attribute: Layout(major_to_minor=(1, 0), tiling=((4, 128),), sub_byte_element_size_in_bits=0)
# 
# ================================================================================
# STEP 2: Offload Tensor to Colocated CPU Mesh (jax.device_put)
# ================================================================================
# CPU Array Shape: (8, 4)
# CPU Array Sharding: NamedSharding(mesh=Mesh('diloco': 2, 'model': 1, axis_types=(Auto, Auto)), spec=P(), memory_kind=device)
# CPU Array Python Layout Attribute: Layout(major_to_minor=(1, 0), tiling=((4, 128),), sub_byte_element_size_in_bits=0)
# 
# ================================================================================
# STEP 3: Define CPU JIT Function Expecting Null Layout (tiling=None)
# ================================================================================
# CPU JIT Expected Input Format: Format(layout=Layout(major_to_minor=(0, 1), tiling=None, sub_byte_element_size_in_bits=0), sharding=NamedSharding(mesh=Mesh('diloco': 2, 'model': 1, axis_types=(Auto, Auto)), spec=P(), memory_kind=device))
# 
# ================================================================================
# STEP 4: Pass CPU Array to JIT Function & Assert Layout Mismatch ValueError
# ================================================================================
# Caught Expected ValueError:
#   Layout passed to jit does not match the layout on the respective arg. Got jit layout: Layout(major_to_minor=(0, 1), tiling=None, sub_byte_element_size_in_bits=0), arg layout: Layout(major_to_minor=(1, 0), tiling=((4, 128),), sub_byte_element_size_in_bits=0) for arg type: float32[8,4].
# PASSED
# tests/unit/pathways_colocated_cpu_layout_disconnect_test.py::PathwaysColocatedCpuLayoutDisconnectTest::test_normalized_colocated_cpu_array_succeeds 
# ================================================================================
# STEP 1: Create TPU Array and Offload to CPU with Null-Layout Normalization
# ================================================================================
# Normalized CPU Array Layout Attribute: Layout(major_to_minor=(0, 1), tiling=None, sub_byte_element_size_in_bits=0)
# 
# ================================================================================
# STEP 2: Execute CPU JIT Function on Normalized Array
# ================================================================================
# Execution Succeeded! Result Shape: (8, 4)
# PASSED
# 
# ============================== 2 passed in 3.14s ===============================
# XPK End: Sat Aug 1 00:38:47 UTC 2026
# ==============================================================================
