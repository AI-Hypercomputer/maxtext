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

"""OLMo 3.5 partner family: whole-model logits against AI2's PyTorch reference.

Weights are transplanted from the reference model rather than co-initialized, so
this compares the computation and nothing else. Every parameter must map; an
unmapped one fails the test rather than silently leaving MaxText's own init in
place, which would otherwise look like a numerical disagreement.

The reference lives in OLMo-core, not in this repo. Point ``OLMO_CORE_STANDALONE``
at the directory holding ``partner_model.py`` and ``standalone_configs.py`` from
``allenai/OLMo-core@codex/partner-model-family-20260914``
(``src/scripts/standalone/``); the test skips when it is not available::

    OLMO_CORE_STANDALONE=/path/to/OLMo-core/src/scripts/standalone \\
        JAX_PLATFORMS=cpu python -m pytest tests/unit/olmo35_vs_reference_test.py
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
  import partner_model as reference

  _HAVE_REFERENCE = True
except ImportError:
  _HAVE_REFERENCE = False

# Smallest shape satisfying the reference's own validate(): d_model, latent and
# expert hidden all multiples of 256, latent * 2 == d_model, head_dim 128,
# n_heads == 2 * n_kv_heads, n_layers % 8 == 0. With 8 layers the reference puts
# full attention at index 7 and leaves 0-6 on KDA, and layer 0 is the dense
# block, so one forward pass covers every code path in the family.
SHAPE = {
    "vocab_size": 512,
    "d_model": 512,
    "n_layers": 8,
    "n_heads": 2,
    "n_kv_heads": 1,
    "head_dim": 128,
    "expert_hidden_size": 256,
    "num_routed_experts": 8,
    "top_k": 2,
    "latent_dim": 256,
}


def _torch_to_np(tensor):
  return tensor.detach().float().cpu().numpy()


def _path_str(path):
  parts = [str(k.key) if hasattr(k, "key") else str(k) for k in path]
  return "/".join(p for p in parts if p != ".value")


def _reference_weights(ref_model):
  """Map each MaxText parameter path to the reference array, already laid out."""
  out = {}
  params = dict(ref_model.named_parameters())

  def transposed(name):
    return _torch_to_np(params[name]).T

  out["token_embedder/embedding"] = _torch_to_np(params["embedding.weight"])
  out["decoder/embedding_norm/scale"] = _torch_to_np(params["embedding_norm.weight"])
  out["decoder/decoder_norm/scale"] = _torch_to_np(params["lm_norm.weight"])
  out["decoder/logits_dense/kernel"] = transposed("lm_head.weight")

  for i, block in enumerate(ref_model.blocks):
    b, pre = f"blocks.{i}", f"decoder/layers_{i}"
    for norm in ("attn_in_norm", "attn_out_norm", "ffn_in_norm", "ffn_out_norm"):
      out[f"{pre}/{norm}/scale"] = _torch_to_np(params[f"{b}.{norm}.weight"])

    # MaxText computes silu(wi_0) * wi_1 and the reference computes
    # up * silu(gate), so wi_0 is the gate branch and wi_1 the up branch.
    out[f"{pre}/shared_ffn/wi_0/kernel"] = transposed(f"{b}.shared.gate.weight")
    out[f"{pre}/shared_ffn/wi_1/kernel"] = transposed(f"{b}.shared.up.weight")
    out[f"{pre}/shared_ffn/wo/kernel"] = transposed(f"{b}.shared.down.weight")

    if block.router is not None:
      out[f"{pre}/latent_down/kernel"] = transposed(f"{b}.latent_down.weight")
      out[f"{pre}/latent_up/kernel"] = transposed(f"{b}.latent_up.weight")
      out[f"{pre}/moe_block/gate/kernel"] = transposed(f"{b}.router.weight")
      # Reference up/gate are [E, hidden, latent] and down is [E, latent, hidden];
      # MaxText wants [E, latent, hidden] and [E, hidden, latent].
      out[f"{pre}/moe_block/wi_0"] = _torch_to_np(params[f"{b}.routed.gate"]).transpose(0, 2, 1)
      out[f"{pre}/moe_block/wi_1"] = _torch_to_np(params[f"{b}.routed.up"]).transpose(0, 2, 1)
      out[f"{pre}/moe_block/wo"] = _torch_to_np(params[f"{b}.routed.down"]).transpose(0, 2, 1)

    mixer = f"{b}.mixer"
    if isinstance(block.mixer, reference.KimiDeltaAttention):
      for w in ("w_q", "w_k", "w_v", "f_proj_1", "f_proj_2", "w_b", "g_proj_1", "g_proj_2", "w_out"):
        out[f"{pre}/mixer/{w}/kernel"] = transposed(f"{mixer}.{w}.weight")
      out[f"{pre}/mixer/g_proj_2/bias"] = _torch_to_np(params[f"{mixer}.g_proj_2.bias"])
      for conv in ("q_conv", "k_conv", "v_conv"):
        out[f"{pre}/mixer/{conv}"] = _torch_to_np(params[f"{mixer}.{conv}.weight"])[:, 0, :]
      out[f"{pre}/mixer/A_log"] = _torch_to_np(params[f"{mixer}.A_log"])
      out[f"{pre}/mixer/dt_bias"] = _torch_to_np(params[f"{mixer}.dt_bias"])
      out[f"{pre}/mixer/o_norm/scale"] = _torch_to_np(params[f"{mixer}.o_norm.weight"])
    else:
      heads, kv_heads, dim = SHAPE["n_heads"], SHAPE["n_kv_heads"], SHAPE["head_dim"]
      attn = f"{pre}/mixer/attention"
      # MaxText widens the query projection to 2 * head_dim and splits the second
      # half off as the output gate; the reference keeps a separate w_g. Same
      # parameters, different layout.
      w_q = transposed(f"{mixer}.w_q.weight").reshape(-1, heads, dim)
      w_g = transposed(f"{mixer}.w_g.weight").reshape(-1, heads, dim)
      out[f"{attn}/query/kernel"] = np.concatenate([w_q, w_g], axis=-1)
      out[f"{attn}/key/kernel"] = transposed(f"{mixer}.w_k.weight").reshape(-1, kv_heads, dim)
      out[f"{attn}/value/kernel"] = transposed(f"{mixer}.w_v.weight").reshape(-1, kv_heads, dim)
      out[f"{attn}/out/kernel"] = transposed(f"{mixer}.w_out.weight")
      out[f"{attn}/query_norm/scale"] = _torch_to_np(params[f"{mixer}.q_norm.weight"])
      out[f"{attn}/key_norm/scale"] = _torch_to_np(params[f"{mixer}.k_norm.weight"])
      out[f"{attn}/ssmax_scale"] = _torch_to_np(params[f"{mixer}.ssmax_scale"])
  return out


# Embedding-inclusive counts from standalone_configs.FAMILY. These do not need
# the torch reference, so they run everywhere.
FAMILY_TOTAL_PARAMS = {
    "olmo35-tiny": 12_496_341_632,
    "olmo35-small": 72_237_847_936,
    "olmo35-medium": 322_601_566_720,
    "olmo35-large": 1_310_163_554_560,
}


class OLMo35ParameterCountTest(unittest.TestCase):
  """Exact parameter parity with the published family geometries.

  This is the cheap structural check behind the logits test: the counts only
  come out right if every projection, norm gain, bias and conv is present at the
  right size, including the per-head QK gains and the scalable-softmax gain.
  """

  def _assert_total(self, model_name):
    """Build the rung and assert its total parameter count matches the family."""
    # pylint: disable=g-import-not-at-top,import-outside-toplevel
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh
    from maxtext.configs import pyconfig
    from maxtext.layers import quantizations
    from maxtext.models import models
    from maxtext.utils import max_utils, maxtext_utils
    from maxtext.utils.globals import MAXTEXT_PKG_DIR

    cfg = pyconfig.initialize(
        [
            "",
            os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml"),
            f"model_name={model_name}",
            "run_name=olmo35_param_count",
            "enable_checkpointing=False",
            "scan_layers=False",
            "skip_jax_distributed_system=True",
            "per_device_batch_size=1",
            "max_target_length=64",
            "dtype=float32",
            "weight_dtype=float32",
            "megablox=False",
            "sparse_matmul=False",
        ]
    )
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    model = models.transformer_as_linen(cfg, mesh, quant=quantizations.configure_quantization(cfg))
    rng = jax.random.PRNGKey(0)
    ids = jnp.ones((1, 64), dtype=jnp.int32)
    params = jax.eval_shape(lambda: model.init({"params": rng, "dropout": rng}, ids, ids, enable_dropout=False))["params"]
    total = max_utils.calculate_num_params_from_pytree(params)
    expected = FAMILY_TOTAL_PARAMS[model_name]
    self.assertEqual(total, expected, f"{model_name}: {total:,} != reference {expected:,}")

  def test_tiny_total_parameters(self):
    self._assert_total("olmo35-tiny")

  def test_small_total_parameters(self):
    self._assert_total("olmo35-small")

  def test_medium_total_parameters(self):
    self._assert_total("olmo35-medium")

  def test_large_total_parameters(self):
    self._assert_total("olmo35-large")


@unittest.skipUnless(_HAVE_REFERENCE, "set OLMO_CORE_STANDALONE to the OLMo-core standalone script directory")
class OLMo35ReferenceLogitsTest(unittest.TestCase):
  """The whole OLMo 3.5 stack must reproduce the reference's logits."""

  def _compare(self, seq_len, emo_pool, seed=0):
    """Transplant reference weights into MaxText and compare the logits."""
    # pylint: disable=g-import-not-at-top,import-outside-toplevel
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh
    from maxtext.configs import pyconfig
    from maxtext.layers import quantizations
    from maxtext.models import models
    from maxtext.utils import maxtext_utils
    from maxtext.utils.globals import MAXTEXT_PKG_DIR

    torch.set_grad_enabled(False)
    torch.manual_seed(seed)

    ref_config = reference.OLMoE3Config(
        **SHAPE,
        emo_enabled=True,
        emo_min_document_expert_pool=emo_pool,
        emo_max_document_expert_pool=emo_pool,
        emo_eval_document_expert_pool=emo_pool,
    )
    ref_model = reference.OLMoE3(ref_config, dtype=torch.float32)
    ref_model.eval()  # fixes the EMo pool size, so routing is deterministic

    rng = np.random.default_rng(seed)
    tokens = rng.integers(0, SHAPE["vocab_size"], size=(1, seq_len)).astype(np.int32)
    expected = _torch_to_np(ref_model(torch.from_numpy(tokens).long()))

    cfg = pyconfig.initialize(
        [
            "",
            os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml"),
            "model_name=olmo35-tiny",
            "override_model_config=True",  # the test shrinks every dimension
            "run_name=olmo35_reference_parity",
            "enable_checkpointing=False",
            "scan_layers=False",
            "skip_jax_distributed_system=True",
            "per_device_batch_size=1",
            f"max_target_length={seq_len}",
            "dtype=float32",
            "weight_dtype=float32",
            "megablox=False",
            "sparse_matmul=False",
            f"base_emb_dim={SHAPE['d_model']}",
            f"base_num_query_heads={SHAPE['n_heads']}",
            f"base_num_kv_heads={SHAPE['n_kv_heads']}",
            f"head_dim={SHAPE['head_dim']}",
            f"base_num_decoder_layers={SHAPE['n_layers']}",
            f"base_mlp_dim={8 * SHAPE['d_model']}",
            f"base_moe_mlp_dim={SHAPE['expert_hidden_size']}",
            f"moe_expert_input_dim={SHAPE['latent_dim']}",
            f"num_experts={SHAPE['num_routed_experts']}",
            f"num_experts_per_tok={SHAPE['top_k']}",
            f"gdn_num_key_heads={SHAPE['n_heads']}",
            f"gdn_num_value_heads={SHAPE['n_heads']}",
            "gdn_key_head_dim=128",
            "gdn_value_head_dim=256",
            f"vocab_size={SHAPE['vocab_size']}",
            f"emo_min_document_expert_pool={emo_pool}",
            f"emo_max_document_expert_pool={emo_pool}",
            f"emo_eval_document_expert_pool={emo_pool}",
        ]
    )
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    model = models.transformer_as_linen(cfg, mesh, quant=quantizations.configure_quantization(cfg))
    positions = jnp.arange(seq_len, dtype=jnp.int32)[None, :]
    segment_ids = jnp.ones((1, seq_len), dtype=jnp.int32)
    variables = model.init(
        {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1), "aqt": jax.random.PRNGKey(2)},
        jnp.asarray(tokens),
        positions,
        enable_dropout=False,
    )

    lookup = _reference_weights(ref_model)
    unmapped = []

    def transplant(path, leaf):
      key = _path_str(path)
      if key not in lookup:
        unmapped.append(f"missing {key} {tuple(leaf.shape)}")
        return leaf
      value = lookup[key]
      if tuple(value.shape) != tuple(leaf.shape):
        unmapped.append(f"shape {key}: reference {tuple(value.shape)} != maxtext {tuple(leaf.shape)}")
        return leaf
      return jnp.asarray(value, leaf.dtype)

    params = jax.tree_util.tree_map_with_path(transplant, variables["params"])
    self.assertEqual(unmapped, [], "every parameter must come from the reference")

    actual = np.asarray(
        model.apply(
            {"params": params},
            jnp.asarray(tokens),
            positions,
            decoder_segment_ids=segment_ids,
            enable_dropout=False,
            rngs={"dropout": jax.random.PRNGKey(3), "aqt": jax.random.PRNGKey(4)},
        )
    )

    self.assertEqual(actual.shape, expected.shape)
    rel = np.abs(actual - expected).max() / max(float(np.abs(expected).max()), 1e-9)
    # float32 round-off across a different op order measures ~1e-5; anything
    # past 1e-3 means a layer disagrees, not that the accumulation order does.
    self.assertLess(float(rel), 1e-3, f"relative logit error {rel:.2e}")
    np.testing.assert_array_equal(actual.argmax(-1), expected.argmax(-1))

  def test_logits_match_chunked_delta_rule(self):
    """seq 64 is a whole number of KDA chunks, so the chunked rule runs."""
    self._compare(seq_len=64, emo_pool=SHAPE["num_routed_experts"])

  def test_logits_match_multiple_chunks(self):
    """Two KDA chunks, so the inter-chunk state carry is exercised."""
    self._compare(seq_len=128, emo_pool=SHAPE["num_routed_experts"])

  def test_logits_match_unfused_scan(self):
    """seq 16 is shorter than a chunk, which falls back to the token scan."""
    self._compare(seq_len=16, emo_pool=SHAPE["num_routed_experts"])

  def test_logits_match_restricted_emo_pool(self):
    """A pool equal to top_k is the hardest case for the EMo masking."""
    self._compare(seq_len=64, emo_pool=SHAPE["top_k"])


if __name__ == "__main__":
  unittest.main()
