#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Minimal standalone unit test reproducing TPU Raiden WeightSynchronizer
# Python 3.12 PyArrayObject struct offset crash on physical TPU VMs.

"""Minimal standalone unit test reproducing TPU Raiden WeightSynchronizer Python 3.12 issue."""

import sys
import unittest
import jax
import jax.numpy as jnp
import numpy as np

RAIDEN_IMPORT_ERROR = None
try:
  from tpu_raiden.api.jax import weight_synchronizer

  RAIDEN_AVAILABLE = True
except Exception as e:  # pylint: disable=broad-exception-caught
  RAIDEN_AVAILABLE = False
  RAIDEN_IMPORT_ERROR = e


class TestTpuRaidenPy312StructOffset(unittest.TestCase):
  """Unit test for TPU Raiden Python 3.12 PyArrayObject struct offset issue."""

  def setUp(self):
    super().setUp()
    if not RAIDEN_AVAILABLE:
      self.skipTest(f"tpu_raiden Python package not available: {RAIDEN_IMPORT_ERROR}")

    try:
      self.tpu_devices = jax.devices("tpu")
    except RuntimeError:
      self.skipTest("No physical TPU devices found. Run this test on a TPU VM.")

  def test_weight_synchronizer_tpu_array_py312(self):
    """Reproduces PyArrayObject struct memory offset crash in Python 3.12 on TPU."""
    print("\n=======================================================")
    print("TPU Raiden Python 3.12 PyArrayObject Reproduction Test")
    print("=======================================================")
    print(f"Python Version : {sys.version.split()[0]}")
    print(f"JAX Version    : {jax.__version__}")
    print(f"TPU Devices    : {self.tpu_devices}")

    mesh = jax.sharding.Mesh(
        np.array(self.tpu_devices).reshape((1, len(self.tpu_devices))),
        ("data", "model"),
    )
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data", "model"))

    # Create real TPU array
    tpu_array = jax.device_put(jnp.ones((64, 128), dtype=jnp.float32), sharding)
    tpu_array.block_until_ready()

    # pylint: disable-next=protected-access
    raw_array = tpu_array._arrays[0] if hasattr(tpu_array, "_arrays") else tpu_array
    dev = list(tpu_array.devices())[0]
    ifrt_runtime_type = getattr(dev.client, "runtime_type", "N/A")

    print(f"JAX Array Outer Type : {type(tpu_array)}")
    print(f"PyArrayObject Type   : {type(raw_array)}")
    print(f"Device Platform      : {dev.platform}")
    print(f"ifrt_array->client()->runtime_type(): {ifrt_runtime_type}")
    print("=======================================================")

    # Verification: Runtime type is pjrt_ifrt on physical TPU
    self.assertEqual(ifrt_runtime_type, "pjrt_ifrt")

    # In Python 3.12, PyArrayObject struct offset mismatch causes
    # storage->ifrt_array.get() to read invalid memory, throwing
    # RuntimeError: Not a PjRt compatible array
    print("Constructing WeightSynchronizer on TPU array...")
    with self.assertRaisesRegex(RuntimeError, "Not a PjRt compatible array"):
      _ = weight_synchronizer.WeightSynchronizer([tpu_array])

    print(
        "SUCCESSFULLY REPRODUCED: WeightSynchronizer failed with "
        "'Not a PjRt compatible array' due to PyArrayObject Python 3.12 memory layout mismatch."
    )


if __name__ == "__main__":
  unittest.main()
