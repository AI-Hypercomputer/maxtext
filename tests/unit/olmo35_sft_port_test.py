# Copyright 2023-2026 Google LLC
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

"""CPU tests for the Olmo 3.5 SFT pieces ported onto the mid-training line."""

import os
import sys
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from maxtext.configs import pyconfig
from maxtext.input_pipeline import input_pipeline_utils
from maxtext.layers import moe
from maxtext.models import olmoe3
from maxtext.trainers.pre_train import train
from maxtext.utils.globals import MAXTEXT_PKG_DIR


class KdaPaddingTest(unittest.TestCase):
  """Padding (segment 0) is taken out of the KDA recurrence without changing real tokens."""

  def _inputs(self, seq=128, heads=2, dk=16, dv=16, pad=70):
    rng = np.random.default_rng(0)
    q = jnp.asarray(rng.standard_normal((1, seq, heads, dk)), jnp.float32)
    k = jnp.asarray(rng.standard_normal((1, seq, heads, dk)), jnp.float32)
    q, k = olmoe3._l2_normalize(q), olmoe3._l2_normalize(k)  # pylint: disable=protected-access
    v = jnp.asarray(rng.standard_normal((1, seq, heads, dv)), jnp.float32)
    log_decay = -jnp.asarray(rng.uniform(0.0, 0.1, (1, seq, heads, dk)), jnp.float32)
    # Padding gates as an untrained model may produce them: decays far past the float range of a chunk.
    log_decay = log_decay.at[:, seq - pad :].set(-1e4)
    beta = jnp.asarray(rng.uniform(0.0, 1.0, (1, seq, heads)), jnp.float32)
    seg = jnp.concatenate([jnp.ones((1, 30), jnp.int32), 2 * jnp.ones((1, seq - 30 - pad), jnp.int32)], axis=1)
    seg = jnp.concatenate([seg, jnp.zeros((1, pad), jnp.int32)], axis=1)
    prev = jnp.pad(seg, ((0, 0), (1, 0)), constant_values=-1)[:, :seq]
    return q, k, v, log_decay, beta, seg, seg != prev, seq - pad

  def test_pairwise_padding_is_finite_and_real_tokens_unchanged(self):
    q, k, v, log_decay, beta, seg, resets, n_real = self._inputs()
    ld_m, beta_m = olmoe3._mask_padding_gates(seg, log_decay, beta)  # pylint: disable=protected-access
    masked = olmoe3._delta_rule_chunked(q, k, v, ld_m, beta_m, resets, 64)  # pylint: disable=protected-access
    self.assertTrue(bool(jnp.all(jnp.isfinite(masked))))
    ref = olmoe3._delta_rule_scan(q, k, v, jnp.exp(log_decay), beta, resets)  # pylint: disable=protected-access
    np.testing.assert_allclose(masked[:, :n_real], ref[:, :n_real], atol=2e-4, rtol=2e-4)

  def test_subblock_real_tokens_unchanged(self):
    q, k, v, log_decay, beta, seg, resets, n_real = self._inputs()
    ld_m, beta_m = olmoe3._mask_padding_gates(seg, log_decay, beta)  # pylint: disable=protected-access
    with jax.default_matmul_precision("highest"):
      out = olmoe3._delta_rule_chunked_subblock(q, k, v, ld_m, beta_m, resets, 64)  # pylint: disable=protected-access
      unmasked = olmoe3._delta_rule_chunked_subblock(  # pylint: disable=protected-access
          q, k, v, log_decay, beta, resets, 64
      )
    self.assertTrue(bool(jnp.all(jnp.isfinite(out))))
    np.testing.assert_allclose(out[:, :n_real], unmasked[:, :n_real], atol=1e-5, rtol=1e-5)


class DropProbeTest(unittest.TestCase):
  """REQUIRED_RBF / MOE-DROPPED bookkeeping."""

  def test_combine_required_rbf(self):
    self.assertIsNone(moe._combine_required_rbf([None, None]))  # pylint: disable=protected-access
    self.assertEqual(float(moe._combine_required_rbf([jnp.float32(1.5), jnp.float32(2.0)])), 2.0)  # pylint: disable=protected-access
    combined = moe._combine_required_rbf(  # pylint: disable=protected-access
        [jnp.array([1.5, 3.0, 100.0]), jnp.array([2.5, 1.0, 100.0])]
    )
    np.testing.assert_allclose(np.asarray(combined), [2.5, 4.0, 200.0])

  def test_log_lines(self):
    class Cfg:
      log_moe_dropped_tokens = True

    lines = train.required_rbf_log_lines(Cfg(), 7, np.array([1.2, 0, 50, 2.6, 5, 50], np.float32))
    self.assertEqual(lines[0], "REQUIRED_RBF step=7 max=2.6000 per_layer=[1.2000,2.6000]")
    self.assertTrue(lines[1].startswith("MOE-DROPPED step=7 dropped=5 routed=100 frac=5.000e-02 per_layer=[0,5]"))

    class CfgNoDrops:
      log_moe_dropped_tokens = False

    lines = train.required_rbf_log_lines(CfgNoDrops(), 3, np.array([1.0, 1.25], np.float32))
    self.assertEqual(lines, ["REQUIRED_RBF step=3 max=1.2500 per_layer=[1.0000,1.2500]"])


class LoadedRowsTest(unittest.TestCase):

  def test_expand_to_loaded_rows(self):
    op = input_pipeline_utils.ExpandToLoadedRows([True, False, False, True])
    out = op.map({"inputs": np.array([[1, 2], [3, 4]]), "inputs_segmentation": np.array([[1, 1], [1, 0]])})
    np.testing.assert_array_equal(out["inputs"], [[1, 2], [0, 0], [0, 0], [3, 4]])
    np.testing.assert_array_equal(out["inputs_segmentation"], [[1, 1], [0, 0], [0, 0], [1, 0]])


class Olmo35MeshRuleTest(unittest.TestCase):
  """olmo35-ep-as-cp and the flags the production launcher passes parse on this line."""

  def test_launcher_flags_parse(self):
    config = pyconfig.initialize(
        [sys.argv[0], os.path.join(MAXTEXT_PKG_DIR, "configs", "post_train", "sft.yml")],
        run_name="t",
        base_output_directory="/tmp/olmo35_sft_port_test",
        model_name="olmo35-tiny",
        override_model_config=True,
        moe_aux_loss_reduction="sum",
        warmup_step_offset=1,
        z_loss_multiplier=1e-5,
        emo_enabled=False,
        enable_checkpointing=False,
        checkpoint_period=300,
        max_num_checkpoints_to_keep=10,
        custom_mesh_and_rule="olmo35-ep-as-cp",
        ici_fsdp_parallelism=-1,
        ici_expert_parallelism=1,
        context_parallel_load_balance=False,
        allow_split_physical_axes=True,
        shard_exp_on_fsdp=False,
        ragged_buffer_factor=-1.0,
        log_required_ragged_buffer_factor=True,
        log_moe_dropped_tokens=True,
        sharding_tolerance=0.3,
        olmoe3_per_layer_remat=True,
        sparse_matmul=True,
        megablox=True,
        use_tokamax_kda=False,
        kda_chunked_impl="subblock",
        kda_matmul_precision="highest",
        num_vocab_tiling=16,
        remat_policy="full",
        per_device_batch_size=1,
        max_target_length=128,
        skip_jax_distributed_system=True,
    )
    self.assertEqual(config.kda_matmul_precision, "highest")
    self.assertTrue(config.log_moe_dropped_tokens)
    self.assertEqual(config.max_num_checkpoints_to_keep, 10)
    # KDA runs Ulysses over the expert axis: context_sharding is inferred from activation_length.
    self.assertEqual(config.context_sharding, "expert")


if __name__ == "__main__":
  unittest.main()
