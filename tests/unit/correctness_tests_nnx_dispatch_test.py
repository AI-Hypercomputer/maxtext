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

"""CPU unit coverage for the pure-NNX dispatch in the post-training correctness tests.

The correctness tests themselves are TPU-only and skipped (b/425997645), so this
exercises the changed dispatch code that otherwise has no CPU coverage:
  - `mt.from_config` is exported (the GRPO trainer calls it)
  - the SFT correctness test's `setup_maxtext_model` / `get_maxtext_logits` run on
    both paths (pure_nnx=True -> NNX, pure_nnx=False -> Linen) and stay finite

The GRPO NNX building blocks the other dispatch helpers call
(`compute_log_probs_nnx`, `grpo_loss_fn_nnx`) are already covered by grpo_nnx_test.
"""

import os
import sys
import unittest

import jax
import jax.numpy as jnp

import maxtext as mt
from maxtext.configs import pyconfig
from maxtext.utils.globals import MAXTEXT_PKG_DIR
from tests.post_training.integration import sft_trainer_correctness_test as sft

# Small model so the test stays fast and runs on a single CPU device.
_SMALL = {
    "base_emb_dim": 16,
    "base_num_decoder_layers": 2,
    "base_num_query_heads": 2,
    "base_num_kv_heads": 2,
    "base_mlp_dim": 32,
    "head_dim": 8,
    "vocab_size": 32,
    "max_target_length": 8,
    "per_device_batch_size": 1,
}


def _sft_config(pure_nnx):
  return pyconfig.initialize(
      [sys.argv[0], os.path.join(MAXTEXT_PKG_DIR, "configs/post_train", "sft.yml")],
      run_name=f"unit-sft-{pure_nnx}",
      model_name="default",
      enable_checkpointing=False,
      pure_nnx=pure_nnx,
      enable_nnx=pure_nnx,
      pure_nnx_decoder=pure_nnx,
      **_SMALL,
  )


def _fake_data(config):
  b = jax.device_count() * config.per_device_batch_size
  length = config.max_target_length
  return {
      "inputs": jnp.ones((b, length), jnp.int32),
      "inputs_position": jnp.broadcast_to(jnp.arange(length, dtype=jnp.int32), (b, length)),
      "inputs_segmentation": jnp.ones((b, length), jnp.int32),
  }


class CorrectnessTestNNXDispatchTest(unittest.TestCase):

  def test_from_config_is_exported(self):
    # grpo_trainer calls mt.from_config; the package must export it.
    self.assertTrue(hasattr(mt, "from_config"))

  def test_sft_logits_nnx_path(self):
    config = _sft_config(pure_nnx=True)
    logits = sft.get_maxtext_logits(config, _fake_data(config))
    self.assertTrue(bool(jnp.isfinite(logits).all()))

  def test_sft_logits_linen_path(self):
    config = _sft_config(pure_nnx=False)
    logits = sft.get_maxtext_logits(config, _fake_data(config))
    self.assertTrue(bool(jnp.isfinite(logits).all()))


if __name__ == "__main__":
  unittest.main()
