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

"""Unit, golden E2E parity, and TPU v5 tests for WeaverOmniTransformer."""

import dataclasses
import os
import subprocess
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.common.common_types import DecoderBlockType
from maxtext.configs import pyconfig
from maxtext.models import models
from maxtext.models import weaver
from maxtext.utils import globals as maxtext_globals
import numpy as np
import pytest

WeaverConfig = weaver.WeaverConfig
WeaverMoTDecoderLayer = weaver.WeaverMoTDecoderLayer
WeaverOmniTransformer = weaver.WeaverOmniTransformer
WeaverTimeEmbedder = weaver.WeaverTimeEmbedder
build_weaver_3d_position_ids = weaver.build_weaver_3d_position_ids
get_weaver_timestep_embedding = weaver.get_weaver_timestep_embedding
patchify_latents = weaver.patchify_latents
unpatchify_latents = weaver.unpatchify_latents

_TRANSFORMER_GOLDEN_FILENAME = "weaver_transformer_golden_data.npz"
_BASE_CONFIG_PATH = os.path.join(maxtext_globals.MAXTEXT_CONFIGS_DIR, "base.yml")


def _get_golden_data_path(filename: str) -> str | None:
  """Attempts to locate or download golden test asset from GCS bucket."""
  tmp_path = f"/tmp/{filename}"
  if os.path.exists(tmp_path):
    return tmp_path

  gcs_uri = f"gs://maxtext-test-assets/{filename}"
  try:
    ret = subprocess.call(
        ["gcloud", "storage", "cp", gcs_uri, tmp_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=15,
    )
    if ret == 0 and os.path.exists(tmp_path):
      return tmp_path
  except (subprocess.SubprocessError, OSError):
    pass

  return None


def _copy_unscanned_to_scanned_layers(
    unscanned_layers: nnx.List,
    scanned_layers: WeaverMoTDecoderLayer,
) -> None:
  """Stacks per-layer weights from unscanned ``nnx.List`` into ``scanned_layers``."""
  num_layers = len(unscanned_layers)
  per_layer_flat = [
      dict(nnx.to_flat_state(nnx.split(unscanned_layers[i], nnx.Param, ...)[1])) for i in range(num_layers)
  ]

  _, scanned_params, _ = nnx.split(scanned_layers, nnx.Param, ...)
  for path, scanned_leaf in nnx.to_flat_state(scanned_params):
    stacked_val = jnp.stack([per_layer_flat[i][path][...] for i in range(num_layers)], axis=0)
    scanned_leaf[...] = stacked_val


@pytest.mark.cpu_only
class WeaverOmniTransformerUnitTest(parameterized.TestCase):
  """CPU unit tests for WeaverOmniTransformer components and configuration."""

  def test_patchify_and_unpatchify_roundtrip(self):
    """Verifies patchify_latents and unpatchify_latents form an exact roundtrip."""
    bsz, c_lat, t_lat, h_lat, w_lat = 2, 48, 3, 6, 8
    patch_size = 2
    latents = jax.random.normal(jax.random.PRNGKey(0), (bsz, c_lat, t_lat, h_lat, w_lat))

    patches = patchify_latents(latents, patch_size=patch_size)
    expected_s_gen = t_lat * (h_lat // patch_size) * (w_lat // patch_size)
    expected_patch_dim = patch_size * patch_size * c_lat
    self.assertEqual(patches.shape, (bsz, expected_s_gen, expected_patch_dim))

    reconstructed_3d = unpatchify_latents(
        patches,
        t_lat=t_lat,
        h_lat=h_lat,
        w_lat=w_lat,
        patch_size=patch_size,
        latent_channels=c_lat,
    )
    self.assertEqual(reconstructed_3d.shape, (bsz, c_lat, t_lat, h_lat, w_lat))
    np.testing.assert_allclose(np.array(latents), np.array(reconstructed_3d), rtol=1e-6, atol=1e-6)

    # Also verify 2D [B * S_gen, patch_dim] input to unpatchify_latents.
    patches_2d = patches.reshape(bsz * expected_s_gen, expected_patch_dim)
    reconstructed_2d = unpatchify_latents(
        patches_2d,
        t_lat=t_lat,
        h_lat=h_lat,
        w_lat=w_lat,
        patch_size=patch_size,
        latent_channels=c_lat,
    )
    np.testing.assert_allclose(np.array(latents), np.array(reconstructed_2d), rtol=1e-6, atol=1e-6)

  def test_timestep_embedding_and_time_embedder(self):
    """Verifies sinusoidal timestep embedding and WeaverTimeEmbedder shapes and aliases."""
    timesteps = jnp.array([250.0, 750.0], dtype=jnp.float32)
    sin_emb = get_weaver_timestep_embedding(timesteps * 0.001, num_channels=256)
    self.assertEqual(sin_emb.shape, (2, 256))
    self.assertEqual(sin_emb.dtype, jnp.float32)
    # At t=0, cos is 1 and sin is 0 -> first half is 1, second half is 0.
    zero_emb = get_weaver_timestep_embedding(jnp.array([0.0], dtype=jnp.float32), num_channels=256)
    np.testing.assert_allclose(np.array(zero_emb[0, :128]), np.ones(128), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.array(zero_emb[0, 128:]), np.zeros(128), rtol=1e-6, atol=1e-6)

    embedder = WeaverTimeEmbedder(in_channels=256, time_embed_dim=128, rngs=nnx.Rngs(0))
    self.assertIs(embedder.linear_1, embedder.mlp_0)
    self.assertIs(embedder.linear_2, embedder.mlp_2)
    out = embedder(timesteps)
    self.assertEqual(out.shape, (2, 128))
    self.assertTrue(bool(jnp.all(jnp.isfinite(out))))

  def test_build_weaver_3d_position_ids(self):
    """Verifies 3D M-RoPE coordinates for packed [text, vision] samples."""
    pos_3d = build_weaver_3d_position_ids(batch_size=2, s_und=4, t_lat=2, h_patch=2, w_patch=3)
    s_gen = 2 * 2 * 3
    sample_len = 4 + s_gen
    self.assertEqual(pos_3d.shape, (2 * sample_len, 3))

    # Text positions for sample 0 are (i, i, i) for i in 0..3.
    expected_text = np.stack([np.arange(4)] * 3, axis=-1)
    np.testing.assert_array_equal(np.array(pos_3d[:4]), expected_text)

    # First vision token of sample 0 has (t=4, h=0, w=0); last has (t=5, h=1, w=2).
    np.testing.assert_array_equal(np.array(pos_3d[4]), np.array([4, 0, 0]))
    np.testing.assert_array_equal(np.array(pos_3d[sample_len - 1]), np.array([5, 1, 2]))
    # Sample 1 repeats the same per-sample coordinate layout.
    np.testing.assert_array_equal(np.array(pos_3d[:sample_len]), np.array(pos_3d[sample_len:]))

  def test_weaver_nano_diffuser_config_loading(self):
    """Verifies weaver-nano-diffuser.yml loads and populates WeaverConfig."""
    cfg = pyconfig.initialize(
        [
            None,
            _BASE_CONFIG_PATH,
            "model_name=weaver-nano-diffuser",
            "skip_jax_distributed_system=True",
        ]
    )
    self.assertEqual(cfg.decoder_block, DecoderBlockType.WEAVER)
    self.assertEqual(cfg.emb_dim, 4096)
    self.assertEqual(cfg.mlp_dim, 12288)
    self.assertEqual(cfg.num_query_heads, 32)
    self.assertEqual(cfg.num_kv_heads, 8)
    self.assertEqual(cfg.num_decoder_layers, 36)
    self.assertEqual(cfg.head_dim, 128)
    self.assertEqual(cfg.vocab_size, 151936)
    self.assertIs(models.WeaverOmniTransformer, WeaverOmniTransformer)

    weaver_cfg = WeaverConfig.from_maxtext_config(cfg)
    self.assertEqual(weaver_cfg.hidden_size, 4096)
    self.assertEqual(weaver_cfg.intermediate_size, 12288)
    self.assertEqual(weaver_cfg.num_attention_heads, 32)
    self.assertEqual(weaver_cfg.num_key_value_heads, 8)
    self.assertEqual(weaver_cfg.num_hidden_layers, 36)
    self.assertEqual(weaver_cfg.head_dim, 128)
    self.assertEqual(weaver_cfg.vocab_size, 151936)
    self.assertEqual(weaver_cfg.latent_channels, 48)
    self.assertEqual(weaver_cfg.patch_size, 2)
    self.assertEqual(weaver_cfg.mrope_section, (24, 20, 20))
    self.assertEqual(weaver_cfg.rope_theta, 5_000_000.0)

  def test_scan_layers_false_and_true_equivalence(self):
    """Verifies scan_layers=False and scan_layers=True produce identical outputs."""
    base_cfg = WeaverConfig(
        hidden_size=64,
        head_dim=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        num_hidden_layers=3,
        vocab_size=128,
        latent_channels=48,
        patch_size=2,
        mrope_section=(4, 2, 2),
        scan_layers=False,
    )
    model_unscanned = WeaverOmniTransformer(base_cfg, rngs=nnx.Rngs(42))
    model_scanned = WeaverOmniTransformer(
        dataclasses.replace(base_cfg, scan_layers=True),
        rngs=nnx.Rngs(99),
    )

    # Copy top-level weights and per-layer weights from model_unscanned to model_scanned.
    model_scanned.embed_tokens.embedding[...] = model_unscanned.embed_tokens.embedding[...]
    model_scanned.proj_in.kernel[...] = model_unscanned.proj_in.kernel[...]
    model_scanned.proj_in.bias[...] = model_unscanned.proj_in.bias[...]
    model_scanned.time_embedder.mlp_0.kernel[...] = model_unscanned.time_embedder.mlp_0.kernel[...]
    model_scanned.time_embedder.mlp_0.bias[...] = model_unscanned.time_embedder.mlp_0.bias[...]
    model_scanned.time_embedder.mlp_2.kernel[...] = model_unscanned.time_embedder.mlp_2.kernel[...]
    model_scanned.time_embedder.mlp_2.bias[...] = model_unscanned.time_embedder.mlp_2.bias[...]
    model_scanned.norm.scale[...] = model_unscanned.norm.scale[...]
    model_scanned.norm_moe_gen.scale[...] = model_unscanned.norm_moe_gen.scale[...]
    model_scanned.proj_out.kernel[...] = model_unscanned.proj_out.kernel[...]
    model_scanned.proj_out.bias[...] = model_unscanned.proj_out.bias[...]

    assert model_unscanned.layers is not None
    assert model_scanned.scanned_layers is not None
    _copy_unscanned_to_scanned_layers(model_unscanned.layers, model_scanned.scanned_layers)

    self.assertIs(model_unscanned.vae2llm, model_unscanned.proj_in)
    self.assertIs(model_unscanned.llm2vae, model_unscanned.proj_out)

    input_ids = jax.random.randint(jax.random.PRNGKey(1), (2, 5), 0, 128, dtype=jnp.int32)
    latents = jax.random.normal(jax.random.PRNGKey(2), (2, 48, 2, 4, 4))
    timesteps = jnp.array([200.0, 800.0], dtype=jnp.float32)

    out_unscanned, hidden_unscanned = model_unscanned(input_ids, latents, timesteps, return_hidden_states=True)
    out_scanned, hidden_scanned = model_scanned(input_ids, latents, timesteps, return_hidden_states=True)

    self.assertEqual(out_unscanned.shape, (2, 48, 2, 4, 4))
    self.assertEqual(out_scanned.shape, (2, 48, 2, 4, 4))
    np.testing.assert_allclose(np.array(out_unscanned), np.array(out_scanned), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.array(hidden_unscanned), np.array(hidden_scanned), rtol=1e-5, atol=1e-5)


@pytest.mark.cpu_only
@pytest.mark.scheduled_only
class WeaverOmniTransformerGoldenParityTest(parameterized.TestCase):
  """End-to-end numerical parity tests against reference GPU golden outputs."""

  golden_data: Any = None

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    golden_path = _get_golden_data_path(_TRANSFORMER_GOLDEN_FILENAME)
    if golden_path is not None:
      cls.golden_data = np.load(golden_path)

  def setUp(self):
    super().setUp()
    if self.golden_data is None:
      self.skipTest(
          f"Golden test asset {_TRANSFORMER_GOLDEN_FILENAME} not found locally under /tmp "
          "and could not be downloaded from gs://maxtext-test-assets/."
      )

  def _populate_unscanned_weights(
      self,
      model: WeaverOmniTransformer,
      data: dict[str, np.ndarray],
      prefix: str,
      dtype: Any,
  ) -> None:
    """Loads reference weights from the golden archive into an unscanned WeaverOmniTransformer."""

    def arr(key: str) -> jax.Array:
      return jnp.array(data[f"{prefix}_{key}"], dtype=dtype)

    model.embed_tokens.embedding[...] = arr("embed_tokens_weight")
    model.proj_in.kernel[...] = arr("proj_in_weight").T
    model.proj_in.bias[...] = arr("proj_in_bias")
    model.time_embedder.mlp_0.kernel[...] = arr("time_embedder_mlp_0_weight").T
    model.time_embedder.mlp_0.bias[...] = arr("time_embedder_mlp_0_bias")
    model.time_embedder.mlp_2.kernel[...] = arr("time_embedder_mlp_2_weight").T
    model.time_embedder.mlp_2.bias[...] = arr("time_embedder_mlp_2_bias")
    model.norm.scale[...] = arr("norm_weight")
    model.norm_moe_gen.scale[...] = arr("norm_moe_gen_weight")
    model.proj_out.kernel[...] = arr("proj_out_weight").T
    model.proj_out.bias[...] = arr("proj_out_bias")

    assert model.layers is not None
    for l_idx, layer in enumerate(model.layers):
      lp = f"layer_{l_idx}"
      layer.input_layernorm.scale[...] = arr(f"{lp}_input_layernorm_weight")
      layer.input_layernorm_moe_gen.scale[...] = arr(f"{lp}_input_layernorm_moe_gen_weight")
      layer.post_attention_layernorm.scale[...] = arr(f"{lp}_post_attention_layernorm_weight")
      layer.post_attention_layernorm_moe_gen.scale[...] = arr(f"{lp}_post_attention_layernorm_moe_gen_weight")

      layer.self_attn.q_proj.kernel[...] = arr(f"{lp}_self_attn_q_proj_weight").T
      layer.self_attn.k_proj.kernel[...] = arr(f"{lp}_self_attn_k_proj_weight").T
      layer.self_attn.v_proj.kernel[...] = arr(f"{lp}_self_attn_v_proj_weight").T
      layer.self_attn.o_proj.kernel[...] = arr(f"{lp}_self_attn_o_proj_weight").T

      layer.self_attn.q_proj_gen.kernel[...] = arr(f"{lp}_self_attn_q_proj_moe_gen_weight").T
      layer.self_attn.k_proj_gen.kernel[...] = arr(f"{lp}_self_attn_k_proj_moe_gen_weight").T
      layer.self_attn.v_proj_gen.kernel[...] = arr(f"{lp}_self_attn_v_proj_moe_gen_weight").T
      layer.self_attn.o_proj_gen.kernel[...] = arr(f"{lp}_self_attn_o_proj_moe_gen_weight").T

      if layer.self_attn.q_norm is not None and layer.self_attn.k_norm is not None:
        layer.self_attn.q_norm.scale[...] = arr(f"{lp}_self_attn_q_norm_weight")
        layer.self_attn.k_norm.scale[...] = arr(f"{lp}_self_attn_k_norm_weight")
      if layer.self_attn.q_norm_gen is not None and layer.self_attn.k_norm_gen is not None:
        layer.self_attn.q_norm_gen.scale[...] = arr(f"{lp}_self_attn_q_norm_moe_gen_weight")
        layer.self_attn.k_norm_gen.scale[...] = arr(f"{lp}_self_attn_k_norm_moe_gen_weight")
      if layer.self_attn.k_norm_und_for_gen is not None:
        layer.self_attn.k_norm_und_for_gen.scale[...] = arr(f"{lp}_self_attn_k_norm_und_for_gen_weight")

      for mlp_mod, mlp_name in ((layer.mlp, "mlp"), (layer.mlp_moe_gen, "mlp_moe_gen")):
        if mlp_mod.gate_proj is not None:
          mlp_mod.gate_proj.kernel[...] = arr(f"{lp}_{mlp_name}_gate_proj_weight").T
        mlp_mod.up_proj.kernel[...] = arr(f"{lp}_{mlp_name}_up_proj_weight").T
        mlp_mod.down_proj.kernel[...] = arr(f"{lp}_{mlp_name}_down_proj_weight").T

  @parameterized.named_parameters(
      ("weaver_mini_unscanned", "case0", "weaver_mini", False),
      ("weaver_mini_scanned", "case0", "weaver_mini", True),
      ("weaver_max_unscanned", "case1", "weaver_max", False),
      ("weaver_max_scanned", "case1", "weaver_max", True),
  )
  def test_e2e_golden_parity(self, prefix: str, variant: str, scan_layers: bool):
    """Verifies E2E forward pass parity against the reference GPU implementation."""
    assert self.golden_data is not None
    data = dict(self.golden_data)
    dtype = jnp.bfloat16

    is_mini = variant == "weaver_mini"
    cfg = WeaverConfig(
        hidden_size=128,
        head_dim=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=256,
        num_hidden_layers=2,
        vocab_size=256,
        latent_channels=48,
        patch_size=2,
        time_embed_in_channels=256,
        timestep_scale=0.001,
        rms_norm_eps=1e-6,
        hidden_act="silu" if is_mini else "relu2",
        qk_norm_for_text=is_mini,
        qk_norm_for_diffusion=True,
        use_und_k_norm_for_gen=not is_mini,
        mrope_section=(6, 5, 5),
        rope_theta=5_000_000.0,
        scan_layers=False,
        dtype=dtype,
        weight_dtype=dtype,
    )

    ref_unscanned = WeaverOmniTransformer(cfg, rngs=nnx.Rngs(0))
    self._populate_unscanned_weights(ref_unscanned, data, prefix, dtype)

    if scan_layers:
      scanned_cfg = dataclasses.replace(cfg, scan_layers=True)
      model = WeaverOmniTransformer(scanned_cfg, rngs=nnx.Rngs(1))
      model.embed_tokens.embedding[...] = ref_unscanned.embed_tokens.embedding[...]
      model.proj_in.kernel[...] = ref_unscanned.proj_in.kernel[...]
      model.proj_in.bias[...] = ref_unscanned.proj_in.bias[...]
      model.time_embedder.mlp_0.kernel[...] = ref_unscanned.time_embedder.mlp_0.kernel[...]
      model.time_embedder.mlp_0.bias[...] = ref_unscanned.time_embedder.mlp_0.bias[...]
      model.time_embedder.mlp_2.kernel[...] = ref_unscanned.time_embedder.mlp_2.kernel[...]
      model.time_embedder.mlp_2.bias[...] = ref_unscanned.time_embedder.mlp_2.bias[...]
      model.norm.scale[...] = ref_unscanned.norm.scale[...]
      model.norm_moe_gen.scale[...] = ref_unscanned.norm_moe_gen.scale[...]
      model.proj_out.kernel[...] = ref_unscanned.proj_out.kernel[...]
      model.proj_out.bias[...] = ref_unscanned.proj_out.bias[...]
      assert ref_unscanned.layers is not None
      assert model.scanned_layers is not None
      _copy_unscanned_to_scanned_layers(ref_unscanned.layers, model.scanned_layers)
    else:
      model = ref_unscanned

    input_ids = jnp.array(data[f"{prefix}_input_ids"], dtype=jnp.int32)
    latents = jnp.array(data[f"{prefix}_latents"], dtype=dtype)
    timesteps = jnp.array(data[f"{prefix}_timesteps"], dtype=jnp.float32)
    golden_pos_ids = data[f"{prefix}_position_ids"]
    golden_preds_vision = data[f"{prefix}_preds_vision"]
    golden_last_hidden = data[f"{prefix}_last_hidden_state"]

    # Verify automatic 3D M-RoPE position ID builder matches the reference position_ids.
    auto_pos_3d = build_weaver_3d_position_ids(batch_size=2, s_und=6, t_lat=2, h_patch=2, w_patch=2)
    np.testing.assert_array_equal(np.array(auto_pos_3d), golden_pos_ids.T)

    preds_vision, last_hidden = model(
        input_ids,
        latents,
        timesteps,
        return_hidden_states=True,
    )

    self.assertEqual(preds_vision.shape, (2, 48, 2, 4, 4))
    self.assertEqual(last_hidden.shape, (28, 128))

    preds_np = np.array(preds_vision, dtype=np.float32)
    hidden_np = np.array(last_hidden, dtype=np.float32)

    preds_avg_abs = float(np.mean(np.abs(preds_np - golden_preds_vision)))
    hidden_avg_abs = float(np.mean(np.abs(hidden_np - golden_last_hidden)))
    print(
        f"\n[WeaverOmniTransformer E2E Parity: {variant} | scan_layers={scan_layers}]"
        f" preds_max_abs={np.max(np.abs(preds_np - golden_preds_vision)):.4e},"
        f" preds_avg_abs={preds_avg_abs:.4e},"
        f" hidden_max_abs={np.max(np.abs(hidden_np - golden_last_hidden)):.4e},"
        f" hidden_avg_abs={hidden_avg_abs:.4e}"
    )

    self.assertLess(hidden_avg_abs, 1.2e-2)
    self.assertLess(preds_avg_abs, 1.2e-2)
    np.testing.assert_allclose(
        hidden_np,
        golden_last_hidden,
        rtol=3e-2,
        atol=5.5e-2,
        err_msg=f"last_hidden_state mismatch for {variant} (scan_layers={scan_layers})",
    )
    np.testing.assert_allclose(
        preds_np,
        golden_preds_vision,
        rtol=3e-2,
        atol=5.5e-2,
        err_msg=f"preds_vision mismatch for {variant} (scan_layers={scan_layers})",
    )


@pytest.mark.tpu_only
@pytest.mark.scheduled_only
class WeaverOmniTransformerTPUTest(parameterized.TestCase):
  """Scheduled nightly TPU v5 test for full 36-layer WeaverOmniTransformer."""

  @parameterized.named_parameters(
      ("unscanned", False),
      ("scanned", True),
  )
  def test_full_36_layer_backbone_forward_tpu(self, scan_layers: bool):
    """Executes full 36-layer WeaverOmniTransformer on TPU with random weights."""
    cfg = pyconfig.initialize(
        [
            None,
            _BASE_CONFIG_PATH,
            "model_name=weaver-nano-diffuser",
            f"scan_layers={scan_layers}",
            "dtype=bfloat16",
            "weight_dtype=bfloat16",
            "skip_jax_distributed_system=True",
        ]
    )
    model = WeaverOmniTransformer.from_config(cfg, rngs=nnx.Rngs(0))
    self.assertEqual(model.num_hidden_layers, 36)
    self.assertEqual(model.hidden_size, 4096)
    self.assertEqual(model.latent_channels, 48)

    batch_size = 2
    s_und = 8
    t_lat, h_lat, w_lat = 2, 4, 4

    input_ids = jax.random.randint(jax.random.PRNGKey(10), (batch_size, s_und), 0, model.vocab_size, dtype=jnp.int32)
    latents = jax.random.normal(
        jax.random.PRNGKey(11),
        (batch_size, 48, t_lat, h_lat, w_lat),
        dtype=jnp.bfloat16,
    )
    timesteps = jnp.array([250.0, 750.0], dtype=jnp.float32)

    graphdef, state = nnx.split(model)

    @jax.jit
    def forward(model_state, ids, lats, ts):
      merged = nnx.merge(graphdef, model_state)
      return merged(ids, lats, ts)

    output = forward(state, input_ids, latents, timesteps)
    self.assertEqual(output.shape, (batch_size, 48, t_lat, h_lat, w_lat))
    self.assertTrue(bool(jnp.all(jnp.isfinite(output))))


if __name__ == "__main__":
  absltest.main()
