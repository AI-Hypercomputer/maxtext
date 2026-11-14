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

"""ga_accumulate_fp32 under Zero-1 + gradient accumulation (cast_params_to_compute_dtype) on an 8-device CPU mesh.

What the option does: the micro-batch gradients are summed in float32. What it does not do: a gradient that the SPMD
partitioner reduces at its weight-gradient dot (a parameter replicated over the reduced axes) is still reduced in the
dot's bf16 output dtype, every micro-batch, before the cast. test_dot_gradient_is_still_reduced_in_bf16 pins that
down, so it fails, and the option's description must change, if the reduction ever becomes float32.
"""

# pylint: disable=too-many-positional-arguments

import dataclasses
import glob
import os
import re
import subprocess
import sys
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from maxtext.common.common_types import ShardMode
from maxtext.utils import gradient_accumulation

_REQUIRED_CPU_DEVICES = 8
_DUMP_ENV = "GA_ACCUMULATE_FP32_TEST_DUMP_DIR"
# Large enough for the Zero-1 + GA cast to the compute dtype (cast_params_to_compute_dtype skips tensors < 2**22).
_D = 2048

pytestmark = pytest.mark.cpu_only


@pytest.mark.cpu_only
def test_ga_accumulate_fp32_on_cpu_mesh():
  """Runs the tests in a subprocess with 8 forced CPU devices and the HLO after SPMD partitioning dumped."""
  with tempfile.TemporaryDirectory() as dump_dir:
    env = os.environ.copy()
    env["XLA_FLAGS"] = (
        env.get("XLA_FLAGS", "")
        + f" --xla_force_host_platform_device_count={_REQUIRED_CPU_DEVICES}"
        + f" --xla_dump_to={dump_dir} --xla_dump_hlo_as_text --xla_dump_hlo_pass_re=spmd-partitioning"
    )
    env["JAX_PLATFORMS"] = "cpu"
    env[_DUMP_ENV] = dump_dir
    result = subprocess.run([sys.executable, __file__], env=env, capture_output=True, text=True, check=False)
  assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
  assert "GA_ACCUMULATE_FP32_TESTS_PASSED" in result.stdout


@dataclasses.dataclass
class _Cfg:
  """The config fields gradient_accumulation_loss_and_grad reads."""

  ga_accumulate_fp32: bool
  gradient_accumulation_steps: int = 2
  use_tunix_gradient_accumulation: bool = False
  shard_optimizer_over_data: bool = True
  cast_params_to_compute_dtype: bool = True
  dtype: jnp.dtype = jnp.bfloat16
  shard_mode: int = ShardMode.AUTO
  debug_sharding: bool = False
  ga_params_loop_barrier: bool = False
  global_batch_size_to_train_on: int = 16
  max_target_length: int = 1
  hardware: str = "cpu"


def _aux(xent_sum, data):
  zero = jnp.array(0.0)
  return {
      "xent_sum": xent_sum,
      "total_weights": jnp.array(data["x"].shape[0], jnp.float32),
      "moe_lb_loss": zero,
      "indexer_loss": zero,
      "mtp_loss": zero,
  }


def _scale_loss_fn(model, config, data, dropout_rng, params, is_train=True):
  """d/dw = sum of the micro-batch's c, the same for every element of w."""
  del model, config, dropout_rng, is_train
  xent_sum = jnp.sum(params["w"].astype(jnp.float32)) * jnp.sum(data["c"].astype(jnp.float32))
  return xent_sum, _aux(xent_sum, data)


def _dot_loss_fn(model, config, data, dropout_rng, params, is_train=True):
  """A matmul, so the gradient of w comes from a weight-gradient dot over the data-sharded batch."""
  del model, config, dropout_rng, is_train
  xent_sum = jnp.sum(jnp.sin((data["x"] @ params["w"]).astype(jnp.float32)))
  return xent_sum, _aux(xent_sum, data)


class GaAccumulateFp32Test(unittest.TestCase):
  """Run in the subprocess that test_ga_accumulate_fp32_on_cpu_mesh starts."""

  __test__ = False

  def setUp(self):
    super().setUp()
    if len(jax.devices("cpu")) < _REQUIRED_CPU_DEVICES:
      self.skipTest(f"Needs {_REQUIRED_CPU_DEVICES} CPU devices; run via test_ga_accumulate_fp32_on_cpu_mesh.")
    self.mesh = Mesh(np.array(jax.devices("cpu")[:_REQUIRED_CPU_DEVICES]), ("data",))
    self.w_sharding = {"w": NamedSharding(self.mesh, PartitionSpec("data", None))}  # Zero-1: params sharded over data
    self.w = {"w": jax.device_put(jnp.full((_D, _D), 0.01, jnp.float32), self.w_sharding["w"])}

  def _grads(self, loss_fn, data, accumulate_fp32):
    cfg = _Cfg(ga_accumulate_fp32=accumulate_fp32)
    step = jax.jit(
        lambda params, batch: gradient_accumulation.gradient_accumulation_loss_and_grad(
            loss_fn, cfg, None, params, self.w_sharding, batch, None
        )[2]["w"]
    )
    return step(self.w, data)

  def _batch(self, x):
    return jax.device_put(x, NamedSharding(self.mesh, PartitionSpec("data")))

  def test_micro_batches_are_summed_in_float32(self):
    # Micro-batch 0 is rows 0, 2, ..., 14 and micro-batch 1 rows 1, 3, ..., 15. Their gradients, 256/16 = 16 and
    # 1/16 = 0.0625 (loss scale 1/16), are exact in bf16, but their sum 16.0625 needs 8 mantissa bits: bf16 (7)
    # rounds it to 16.
    c = jnp.tile(jnp.array([32.0, 0.125], jnp.bfloat16), 8)
    data = {"c": self._batch(c), "x": self._batch(jnp.zeros((16, 1), jnp.bfloat16))}
    fp32 = self._grads(_scale_loss_fn, data, accumulate_fp32=True)
    bf16 = self._grads(_scale_loss_fn, data, accumulate_fp32=False)
    self.assertEqual(fp32.dtype, jnp.float32)
    self.assertEqual(bf16.dtype, jnp.bfloat16)
    np.testing.assert_array_equal(np.asarray(fp32), np.float32(257.0 / 16))
    np.testing.assert_array_equal(np.asarray(bf16, np.float32), np.float32(16.0))

  def test_dot_gradient_is_still_reduced_in_bf16(self):
    dump_dir = os.environ[_DUMP_ENV]
    seen = set(glob.glob(os.path.join(dump_dir, "*")))
    x = jax.random.normal(jax.random.key(0), (16, _D), jnp.bfloat16)
    self._grads(_dot_loss_fn, {"x": self._batch(x)}, accumulate_fp32=True)
    dumps = [
        f for f in glob.glob(os.path.join(dump_dir, "*after_spmd-partitioning*")) if f not in seen and "jit__lambda" in f
    ]
    self.assertTrue(dumps, "no HLO dump after SPMD partitioning")
    with open(dumps[0], encoding="utf-8") as f:
      hlo = f.read()
    dots = set(re.findall(r"%([\w.\-]+) = \w+\[\d+,\d+\][^ ]* dot\(", hlo))
    reductions = re.findall(rf"= (\w+)\[{_D},{_D}\][^ ]* all-reduce\(%([\w.\-]+)\)", hlo)
    of_dots = [dtype for dtype, operand in reductions if operand in dots]
    self.assertTrue(of_dots, f"no all-reduce of a weight-gradient dot; all-reduces: {reductions}")
    # bf16 today: if this becomes f32, update ga_accumulate_fp32's description (it would then also cover this case).
    self.assertEqual(set(of_dots), {"bf16"})


if __name__ == "__main__":
  suite = unittest.defaultTestLoader.loadTestsFromTestCase(GaAccumulateFp32Test)
  res = unittest.TextTestRunner(verbosity=2).run(suite)
  if not res.wasSuccessful():
    sys.exit(1)
  print("GA_ACCUMULATE_FP32_TESTS_PASSED")
