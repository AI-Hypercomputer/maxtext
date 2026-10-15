# Copyright 2025-2026 Google LLC
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

"""NNX parity test for `fused_lm_head_ce_nnx_loss` vs the full-vocab reference loss.

Runs on CPU (Pallas interpret mode) with 4 fake devices so the shard_map wrapper
(tokens split over fsdp, LM head all-gathered / grad reduce-scattered) is exercised.
"""

import os

os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4"

# pylint: disable=g-import-not-at-top,wrong-import-position
from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils
from maxtext.utils import maxtext_utils_nnx
from maxtext.utils import model_creation_utils
from maxtext.utils.globals import MAXTEXT_PKG_DIR
from maxtext.utils.vocabulary_tiling import fused_lm_head_ce_nnx_loss

SEQ = 64


def _cfg(z_loss, vocab, bwd_fused):
  return pyconfig.initialize(
      [
          "",
          os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml"),
          "run_name=fused_ce_nnx_test",
          "enable_checkpointing=False",
          "enable_dropout=False",
          f"max_target_length={SEQ}",
          "per_device_batch_size=1",
          "base_emb_dim=128",
          "base_mlp_dim=256",
          "base_num_query_heads=2",
          "base_num_kv_heads=2",
          "head_dim=64",
          "base_num_decoder_layers=0",
          f"vocab_size={vocab}",
          "logits_via_embedding=False",
          "dtype=float32",
          "weight_dtype=float32",
          "matmul_precision=highest",
          f"z_loss_multiplier={z_loss}",
          "fused_lm_head_ce=True",
          f"fused_lm_head_ce_bwd_fused={bwd_fused}",
          "fused_lm_head_ce_block_t=64",
          "fused_lm_head_ce_block_v=256",
          "fused_lm_head_ce_block_t_dx=64",
          "fused_lm_head_ce_block_v_dx=128",
          "fused_lm_head_ce_block_t_dw=32",
          "fused_lm_head_ce_block_v_dw=256",
          "fused_lm_head_ce_block_t_bwd=32",
          "fused_lm_head_ce_block_v_bwd=128",
          "skip_jax_distributed_system=True",
      ],
  )


class FusedCENNXTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("no_zloss", 0.0, 1000, True),
      ("zloss", 1e-4, 1536, True),
      ("zloss_split_bwd", 1e-4, 1000, False),
  )
  def test_loss_and_grads_match_full_vocab(self, z_loss, vocab, bwd_fused):
    cfg = _cfg(z_loss, vocab, bwd_fused)
    mesh = maxtext_utils.get_mesh_from_config(cfg)
    rngs = maxtext_utils_nnx.create_nnx_rngs(cfg)
    with nn_partitioning.axis_rules(cfg.logical_axis_rules):
      model = model_creation_utils.from_config(cfg, mesh=mesh, rngs=rngs, model_mode=MODEL_MODE_TRAIN)
    batch = jax.device_count()
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
    hidden = jax.random.normal(k1, (batch, SEQ, cfg.emb_dim), jnp.float32)
    labels = jax.random.randint(k2, (batch, SEQ), 0, cfg.vocab_size)
    seg = (jax.random.uniform(k3, (batch, SEQ)) > 0.2).astype(jnp.int32)
    data = {"targets": labels, "targets_segmentation": seg}
    graphdef, params, rest = nnx.split(model, nnx.Param, ...)

    def ref_fn(p, h):
      m = nnx.merge(graphdef, p, rest, copy=True)
      logits = m.logits_from_hidden_states_for_vocab_tiling(h, True, MODEL_MODE_TRAIN)
      xent, z = max_utils.cross_entropy_with_logits(logits, jax.nn.one_hot(labels, cfg.vocab_size), z_loss=z_loss)
      return jnp.sum(xent * (seg != 0)), jnp.sum(z * (seg != 0))

    def fused_fn(p, h):
      m = nnx.merge(graphdef, p, rest, copy=True)
      return fused_lm_head_ce_nnx_loss(m, h, data, cfg, is_train=True)

    with nn_partitioning.axis_rules(cfg.logical_axis_rules):
      vg = lambda f: jax.jit(jax.value_and_grad(f, argnums=(0, 1), has_aux=True))
      (r_loss, r_z), (r_gp, r_gh) = vg(ref_fn)(params, hidden)
      (f_loss, f_z), (f_gp, f_gh) = vg(fused_fn)(params, hidden)

    np.testing.assert_allclose(f_loss, r_loss, rtol=1e-4)
    np.testing.assert_allclose(f_z, r_z, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(np.asarray(f_gh), np.asarray(r_gh), rtol=1e-3, atol=1e-5)
    r_leaves = jax.tree_util.tree_leaves_with_path(r_gp)
    f_leaves = dict(jax.tree_util.tree_leaves_with_path(f_gp))
    checked = 0
    for path, r in r_leaves:
      f = f_leaves[path]
      np.testing.assert_allclose(np.asarray(f), np.asarray(r), rtol=1e-3, atol=1e-5, err_msg=str(path))
      checked += 1
    self.assertGreater(checked, 0)


if __name__ == "__main__":
  absltest.main()
