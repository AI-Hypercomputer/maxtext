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

"""OLMoE3 Kimi Delta Attention against AI2's PyTorch reference.

Weights are transplanted from the reference module rather than co-initialized,
so this compares the computation and nothing else.

The reference lives in OLMo-core, not in this repo. Point ``OLMO_CORE_STANDALONE``
at the directory holding ``standalone_model.py`` from
``allenai/OLMo-core@akshitab/standalone`` (``src/scripts/standalone/``); the test
skips when it is not available::

    OLMO_CORE_STANDALONE=/path/to/OLMo-core/src/scripts/standalone \\
        python -m pytest tests/unit/olmoe3_vs_reference_test.py
"""

import os
import sys
import unittest

import numpy as np

_STANDALONE = os.environ.get("OLMO_CORE_STANDALONE", "")
if _STANDALONE and _STANDALONE not in sys.path:
  sys.path.insert(0, _STANDALONE)

try:  # pylint: disable=g-import-not-at-top
  import torch
  import standalone_model as reference

  _HAVE_REFERENCE = True
except ImportError:
  _HAVE_REFERENCE = False


@unittest.skipUnless(_HAVE_REFERENCE, "set OLMO_CORE_STANDALONE to the OLMo-core standalone script directory")
class KimiDeltaAttentionReferenceTest(unittest.TestCase):
  """The KDA recurrence, conv, gating and norms must match the reference."""

  @classmethod
  def setUpClass(cls):
    # Imported here, not at module scope: this class is skipped unless the
    # torch reference is available, and these must not be a hard dependency.
    # pylint: disable=g-import-not-at-top,import-outside-toplevel
    import jax.numpy as jnp
    from flax import nnx
    from jax.sharding import Mesh
    from maxtext.configs import pyconfig
    from maxtext.models.olmoe3 import OLMoE3KimiDeltaAttention
    from maxtext.utils import maxtext_utils
    from maxtext.utils.globals import MAXTEXT_PKG_DIR

    cls.jnp, cls.np = jnp, np
    torch.set_grad_enabled(False)
    torch.manual_seed(0)

    cfg = pyconfig.initialize(
        [
            "",
            os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml"),
            "model_name=olmoe3-3p5b",
            "run_name=olmoe3_kda_parity",
            "enable_checkpointing=False",
            "scan_layers=False",
            "per_device_batch_size=1",
            "max_target_length=64",
            "dtype=float32",
            "weight_dtype=float32",
            "megablox=False",
            "sparse_matmul=False",
            "skip_jax_distributed_system=True",
        ],
    )
    cls.cfg = cfg
    cls.mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    cls.reference_layer = reference.KimiDeltaAttention(reference.largest_config).float()

    with cls.mesh:
      cls.layer = OLMoE3KimiDeltaAttention(cfg, cls.mesh, None, rngs=nnx.Rngs(params=0))
    cls._transplant_weights()

  @classmethod
  def _transplant_weights(cls):
    """Copies the torch reference weights into the MaxText layer."""
    jnp = cls.jnp
    ref, mine = cls.reference_layer, cls.layer

    def transposed(t):
      # torch.nn.Linear stores (out, in); DenseGeneral stores (in, out).
      return jnp.asarray(t.detach().numpy().T.copy())

    def raw(t):
      return jnp.asarray(t.detach().numpy().copy())

    for name in ("w_q", "w_k", "w_v", "f_proj_1", "f_proj_2", "w_b", "g_proj_1", "g_proj_2", "w_out"):
      target = getattr(mine, name).kernel
      target[...] = transposed(getattr(ref, name).weight).reshape(target[...].shape)
    mine.g_proj_2.bias[...] = raw(ref.g_proj_2.bias)
    for name in ("q_conv", "k_conv", "v_conv"):
      getattr(mine, name)[...] = raw(getattr(ref, name).weight)[:, 0, :]
    mine.A_log[...] = raw(ref.A_log)
    mine.dt_bias[...] = raw(ref.dt_bias)
    mine.o_norm.scale[...] = raw(ref.o_norm.weight)

  def _compare(self, segment_ids):
    """Runs both layers on identical inputs and returns the relative error."""
    jnp = self.jnp
    batch, seq_len = 2, 64
    inputs = torch.randn(batch, seq_len, self.cfg.emb_dim)
    expected = self.reference_layer(inputs, segment_ids).numpy()
    with self.mesh:
      actual = np.asarray(
          self.layer(
              jnp.asarray(inputs.numpy()),
              None if segment_ids is None else jnp.asarray(segment_ids.numpy()),
          )
      )
    relative = np.max(np.abs(actual - expected)) / np.max(np.abs(expected))
    self.assertLess(relative, 1e-4, f"relative error {relative:.2e}")

  def test_matches_reference_unpacked(self):
    """Checks test matches reference unpacked."""
    self._compare(None)

  def test_matches_reference_packed(self):
    """Packed documents exercise the conv masking and the state reset."""
    batch, seq_len, boundary = 2, 64, 30
    segment_ids = torch.cat(
        [torch.zeros(batch, boundary, dtype=torch.long), torch.ones(batch, seq_len - boundary, dtype=torch.long)],
        dim=1,
    )
    self._compare(segment_ids)


if __name__ == "__main__":
  unittest.main()
