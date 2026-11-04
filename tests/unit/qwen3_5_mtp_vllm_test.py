# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for Qwen3.5 Multi-Token Prediction (MTP) in MaxText and vLLM adapter."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np
import yaml


class _AutoStubModule(types.ModuleType):
  """Lightweight module stub for optional dependencies not installed in minimal CPU test envs."""

  def __init__(self, name: str):
    super().__init__(name)
    self.__path__ = []

  def __getattr__(self, name: str):
    if name.startswith("__"):
      raise AttributeError(name)
    child = _AutoStubModule(f"{self.__name__}.{name}")
    sys.modules[child.__name__] = child
    setattr(self, name, child)
    return child

  def __call__(self, *args, **kwargs):
    return self

  def __or__(self, other):
    return object

  def __ror__(self, other):
    return object


def _ensure_test_stubs():
  """Provides lightweight module stubs when running outside full TPU/MaxText environments."""
  if importlib.util.find_spec("omegaconf") is None:
    oc = types.ModuleType("omegaconf")
    oc.DictConfig = dict
    oc.OmegaConf = types.SimpleNamespace(
        load=lambda p: yaml.safe_load(Path(p).read_text(encoding="utf-8")),
        create=lambda obj=None: obj or {},
        to_container=lambda cfg, resolve=True: cfg,
        from_cli=lambda args: {
            k: yaml.safe_load(v) for a in (args or ()) if "=" in str(a) for k, v in [str(a).split("=", 1)]
        },
        merge=lambda *cfgs: {k: v for c in cfgs if isinstance(c, dict) for k, v in c.items()},
    )
    sys.modules["omegaconf"] = oc

  missing_roots = {r for r in ("aqt", "grain", "drjax", "tokamax") if importlib.util.find_spec(r) is None}
  for pkg in (
      "aqt.jax.v2.aqt_tensor",
      "aqt.jax.v2.config",
      "aqt.jax.v2.flax.aqt_flax",
      "aqt.jax.v2.tiled_dot_general",
      "aqt.jax.v2.calibration",
      "grain.python",
      "drjax",
      "tokamax",
  ):
    if pkg.split(".")[0] in missing_roots:
      parts = pkg.split(".")
      for i in range(1, len(parts) + 1):
        sub = ".".join(parts[:i])
        sys.modules.setdefault(sub, _AutoStubModule(sub))
      if pkg == "aqt.jax.v2.aqt_tensor":
        sys.modules[pkg].QTensor = type("QTensor", (), {})

  try:
    from orbax.checkpoint import v1 as ocp  # pylint: disable=import-outside-toplevel

    if not hasattr(ocp.training, "pathways"):
      ocp.training.pathways = types.SimpleNamespace(Snapshotter=object)
  except ImportError:
    pass

  try:
    import qwix  # pylint: disable=import-outside-toplevel
    import qwix._src.core as qwix_core  # pylint: disable=import-outside-toplevel

    for mod, attr, name in ((qwix_core, "sparsity", "qwix._src.core.sparsity"), (qwix, "contrib", "qwix.contrib")):
      if not hasattr(mod, attr):
        setattr(mod, attr, _AutoStubModule(name))
        sys.modules[name] = getattr(mod, attr)
    sys.modules.setdefault("qwix.contrib.sparsity", _AutoStubModule("qwix.contrib.sparsity"))
  except ImportError:
    pass

  try:
    import jax.experimental.xla_metadata as xla_metadata  # pylint: disable=import-outside-toplevel

    if not hasattr(xla_metadata, "must_fuse_call"):
      xla_metadata.must_fuse_call = lambda f, *a, **k: f
  except ImportError:
    pass

  for name in (
      "tpu_inference",
      "tpu_inference.logger",
      "tpu_inference.layers",
      "tpu_inference.layers.common",
      "tpu_inference.layers.common.attention_metadata",
      "tpu_inference.models",
      "tpu_inference.models.common",
      "tpu_inference.models.common.model_loader",
      "tpu_inference.models.jax",
      "tpu_inference.models.jax.utils",
      "tpu_inference.models.jax.utils.multi_modal_utils",
      "tpu_inference.runner",
      "tpu_inference.runner.kv_cache",
      "tpu_inference.runner.kv_cache_manager",
      "tpu_inference.utils",
      "vllm",
      "vllm.config",
  ):
    sys.modules.setdefault(name, _AutoStubModule(name))

  sys.modules["tpu_inference.logger"].init_logger = lambda *a, **k: types.SimpleNamespace(info=lambda *a, **k: None)
  sys.modules["tpu_inference.models.common.model_loader"].register_model = lambda *a, **k: None
  sys.modules["tpu_inference.models.jax.utils.multi_modal_utils"].merge_multimodal_embeddings = lambda *a, **k: None
  sys.modules["tpu_inference.layers.common.attention_metadata"].AttentionMetadata = types.SimpleNamespace
  sys.modules["vllm.config"].VllmConfig = types.SimpleNamespace


_ensure_test_stubs()

# pylint: disable=wrong-import-position
from maxtext.checkpoint_conversion.utils import hf_shape, param_mapping
from maxtext.common.common_types import AttentionType, DecoderBlockType, MODEL_MODE_AUTOREGRESSIVE
from maxtext.integration.vllm.maxtext_vllm_adapter import adapter
from maxtext.layers import multi_token_prediction
from maxtext.layers.embeddings import Embed
from maxtext.layers.linears import DenseGeneral
from maxtext.models import qwen3, qwen3_5


class _DummyCfg(types.SimpleNamespace):
  def __getattr__(self, name: str):
    if name.startswith("__"):
      raise AttributeError(name)
    return False


def _make_dummy_cfg() -> _DummyCfg:
  """Builds a minimal Qwen3.5 config backed by base.yml defaults for single-device CPU tests."""
  base_yml = Path(__file__).resolve().parents[2] / "src" / "maxtext" / "configs" / "base.yml"
  defaults = yaml.safe_load(base_yml.read_text(encoding="utf-8")) if base_yml.is_file() else {}
  defaults.update(
      decoder_block=DecoderBlockType.QWEN3_5,
      model_name="qwen3.5-35b-a3b",
      dtype=jnp.float32,
      weight_dtype=jnp.float32,
      shard_mode="auto",
      scan_layers=True,
      emb_dim=32,
      head_dim=8,
      base_num_query_heads=4,
      base_num_kv_heads=2,
      num_query_heads=4,
      num_kv_heads=2,
      base_num_decoder_layers=4,
      num_decoder_layers=4,
      inhomogeneous_layer_cycle_interval=4,
      full_attention_interval=4,
      max_position_embeddings=128,
      max_target_length=64,
      max_prefill_predict_length=32,
      micro_batch_size_to_train_on=1,
      partial_rotary_factor=0.25,
      attention="dot_product",
      attention_type=AttentionType.GLOBAL.value,
      use_qk_norm=True,
      num_experts=4,
      num_experts_per_tok=2,
      base_moe_mlp_dim=16,
      moe_mlp_dim=16,
      padded_base_moe_mlp_dim=16,
      shared_experts=1,
      mlp_activations=["silu", "linear"],
      norm_topk_prob=True,
      sparse_matmul=False,
      vocab_size=64,
      logits_via_embedding=False,
      mtp_num_layers=1,
      logical_axis_rules=[],
      lora=types.SimpleNamespace(enable_lora=False, lora_restore_path=None),
  )
  return _DummyCfg(**defaults)


def _make_single_device_mesh(cfg) -> Mesh:
  axis_names = tuple(cfg.mesh_axes)
  return Mesh(np.array(jax.devices()[:1]).reshape((1,) * len(axis_names)), axis_names)


class Qwen3_5MTPTest(unittest.TestCase):
  """Tests for Qwen3.5 Multi-Token Prediction layer, checkpoint mapping, and vLLM adapter."""

  def setUp(self):
    super().setUp()
    self.cfg = _make_dummy_cfg()
    self.mesh = _make_single_device_mesh(self.cfg)
    self.rngs = nnx.Rngs(params=0, dropout=1, gating=2)

  def test_mtp_layer_uses_qwen3_next_rmsnorm_and_single_decoder_layer(self):
    """MultiTokenPredictionLayer for Qwen3.5 uses Qwen3NextRMSNorm (1+scale) and Qwen3_5DecoderLayer."""
    mtp_layer = multi_token_prediction.MultiTokenPredictionLayer(
        config=self.cfg,
        mesh=self.mesh,
        layer_number=1,
        transformer_layer_module=qwen3_5.Qwen3_5ScannableBlock,
        model_mode=MODEL_MODE_AUTOREGRESSIVE,
        quant=None,
        rngs=self.rngs,
    )
    for norm in (mtp_layer.embedding_norm, mtp_layer.hidden_state_norm, mtp_layer.final_norm):
      self.assertIsInstance(norm, qwen3.RMSNorm)
      self.assertEqual(norm.scale_offset, 1.0)
    self.assertIsInstance(mtp_layer.transformer_layer, qwen3_5.Qwen3_5DecoderLayer)
    self.assertTrue(mtp_layer.transformer_layer.is_full_attention_layer)
    self.assertEqual(mtp_layer.transformer_layer.layer_idx, self.cfg.base_num_decoder_layers)

  def test_qwen3_5_mtp_param_mapping_and_shapes(self):
    """QWEN3_5 param_mapping, hooks, and hf_shape include mtp_block.mtp_layer_1 when mtp_num_layers=1."""
    hf_cfg = {
        "text_config": {
            "hidden_size": 32,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "vocab_size": 64,
            "num_experts": 4,
            "moe_intermediate_size": 16,
            "shared_expert_intermediate_size": 16,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_conv_kernel_dim": 4,
            "full_attention_interval": 4,
            "mtp_num_hidden_layers": 1,
        }
    }
    mapping = param_mapping.QWEN3_5_MAXTEXT_TO_HF_PARAM_MAPPING(hf_cfg, self.cfg, scan_layers=False)
    hooks = param_mapping.QWEN3_5_MAXTEXT_TO_HF_PARAM_HOOK_FN(hf_cfg, self.cfg, scan_layers=False)
    shapes = hf_shape.QWEN3_5_HF_WEIGHTS_TO_SHAPE(hf_cfg)

    prefix = "params-mtp_block-mtp_layer_1"
    tl_prefix = f"{prefix}-mtp_1_transformer_layer"
    self.assertEqual(mapping[f"{prefix}-mtp_1_embedding_norm-scale"], "mtp.pre_fc_norm_embedding.weight")
    self.assertEqual(mapping[f"{prefix}-mtp_1_hidden_state_norm-scale"], "mtp.pre_fc_norm_hidden.weight")
    self.assertEqual(mapping[f"{prefix}-mtp_1_projection-kernel"], "mtp.fc.weight")
    self.assertEqual(mapping[f"{prefix}-mtp_1_final_norm-scale"], "mtp.norm.weight")
    self.assertEqual(len(mapping[f"{tl_prefix}-mlp-routed_experts-wi_0"]), 4)
    self.assertIn(f"{prefix}-mtp_1_projection-kernel", hooks)
    self.assertIn(f"{tl_prefix}-mlp-routed_experts-wi_0", hooks)
    self.assertEqual(shapes["mtp.fc.weight"], [32, 64])
    self.assertEqual(shapes["mtp.layers.0.mlp.experts.0.gate_proj.weight"], [16, 32])

  def test_target_model_exposes_post_norm_aux_hidden_states_for_mtp(self):
    """MaxTextForCausalLM.__call__ returns post-decoder_norm aux_hidden_states when method=='mtp'."""
    raw_hidden = jnp.ones((4, 1, self.cfg.emb_dim), dtype=jnp.float32) * 2.0
    normed_hidden = jnp.ones((4, self.cfg.emb_dim), dtype=jnp.float32) * 5.0
    dummy_kv = [jnp.zeros((1, 4, 2, 8))]

    class _DummyMaxTextModel(nnx.Module):

      def __init__(self):
        self.decoder = types.SimpleNamespace(decoder_norm=lambda x: normed_hidden)

      def __call__(self, **kwargs):
        return raw_hidden, dummy_kv, None

    dummy_self = types.SimpleNamespace(
        cfg=self.cfg,
        maxtext_config=self.cfg,
        mesh=self.mesh,
        model_mode=MODEL_MODE_AUTOREGRESSIVE,
        model=_DummyMaxTextModel(),
        vllm_config=types.SimpleNamespace(speculative_config=None),
    )
    attn_meta = types.SimpleNamespace(
        input_positions=jnp.array([0, 1, 2, 3], dtype=jnp.int32),
        seq_lens=jnp.array([4], dtype=jnp.int32),
    )

    _, hidden_out, aux_out, _ = adapter.MaxTextForCausalLM.__call__(
        dummy_self, kv_caches=dummy_kv, input_ids=jnp.zeros((4,), dtype=jnp.int32), attention_metadata=attn_meta
    )
    self.assertEqual(aux_out, [])
    np.testing.assert_allclose(hidden_out, raw_hidden.reshape(-1, self.cfg.emb_dim))

    dummy_self.vllm_config = types.SimpleNamespace(speculative_config=types.SimpleNamespace(method="mtp"))
    _, hidden_out, aux_out, _ = adapter.MaxTextForCausalLM.__call__(
        dummy_self, kv_caches=dummy_kv, input_ids=jnp.zeros((4,), dtype=jnp.int32), attention_metadata=attn_meta
    )
    np.testing.assert_allclose(hidden_out, raw_hidden.reshape(-1, self.cfg.emb_dim))
    self.assertEqual(len(aux_out), 1)
    np.testing.assert_allclose(aux_out[0], normed_hidden.reshape(-1, self.cfg.emb_dim))

  def test_generate_maxtext_config_enables_mtp_num_layers(self):
    """generate_maxtext_config sets mtp_num_layers=1 when speculative_config.method == 'mtp'."""
    vllm_config = types.SimpleNamespace(
        additional_config={"maxtext_config": {"model_name": "qwen3.5-35b-a3b"}},
        load_config=types.SimpleNamespace(load_format="auto"),
        sharding_config=types.SimpleNamespace(tp_size=1, expert_size=1, attn_dp_size=1, attn_dp_expert_size=1),
        parallel_config=types.SimpleNamespace(enable_expert_parallel=False),
        model_config=types.SimpleNamespace(
            hf_config=types.SimpleNamespace(num_key_value_heads=2, moe_intermediate_size=16),
            max_model_len=128,
            get_hf_config_path=lambda: "/tmp/fake_hf_config",
        ),
        scheduler_config=types.SimpleNamespace(max_num_batched_tokens=128),
        cache_config=types.SimpleNamespace(block_size=16),
        speculative_config=types.SimpleNamespace(method="mtp"),
    )
    recorded_kwargs = {}
    with mock.patch.object(
        adapter.pyconfig, "initialize", side_effect=lambda argv, **kw: recorded_kwargs.update(kw) or self.cfg
    ):
      adapter.generate_maxtext_config(vllm_config)
    self.assertEqual(recorded_kwargs.get("mtp_num_layers"), 1)

  def test_maxtext_for_causal_lm_mtp_forward_and_logits(self):
    """MaxTextForCausalLM resolves model.mtp_block.mtp_layer_1 and runs mtp_forward and compute_mtp_logits."""
    dim = self.cfg.emb_dim
    target_lm_head_val = jnp.ones((dim, self.cfg.vocab_size), dtype=jnp.float32) * 0.2

    class _FakeDecoder(nnx.Module):

      def __init__(self, cfg, rngs):
        self.logits_dense = DenseGeneral(
            in_features_shape=dim,
            out_features_shape=cfg.vocab_size,
            dtype=jnp.float32,
            weight_dtype=jnp.float32,
            rngs=rngs,
        )
        self.logits_dense.kernel.value = jnp.copy(target_lm_head_val)

      def apply_output_head(self, shared_embedding, y, deterministic, model_mode, normalize_y=True, reduce_mhc=True):
        del shared_embedding, deterministic, model_mode, reduce_mhc
        return self.logits_dense(y * 10.0 if normalize_y else y)

    class _FakeMTPBlock(nnx.Module):

      def __init__(self, cfg, mesh, rngs):
        self.mtp_layer_1 = multi_token_prediction.MultiTokenPredictionLayer(
            config=cfg,
            mesh=mesh,
            layer_number=1,
            transformer_layer_module=qwen3_5.Qwen3_5ScannableBlock,
            model_mode=MODEL_MODE_AUTOREGRESSIVE,
            quant=None,
            rngs=rngs,
        )

    class _FakeMaxTextModel(nnx.Module):

      def __init__(self, cfg, mesh, rngs):
        self.token_embedder = Embed(
            mesh=mesh,
            num_embeddings=cfg.vocab_size,
            num_features=dim,
            dtype=jnp.float32,
            attend_dtype=jnp.float32,
            embedding_init=jax.nn.initializers.normal(stddev=1.0),
            config=cfg,
            rngs=rngs,
        )
        self.decoder = _FakeDecoder(cfg, rngs)
        self.mtp_block = _FakeMTPBlock(cfg, mesh, rngs)

    vllm_config = types.SimpleNamespace(
        model_config=types.SimpleNamespace(model="Qwen/Qwen3.5-35B-A3B", trust_remote_code=True),
        load_config=types.SimpleNamespace(load_format="auto"),
        cache_config=types.SimpleNamespace(block_size=16, cache_dtype="auto"),
        speculative_config=types.SimpleNamespace(method="mtp"),
        additional_config={},
    )
    fake_trunk = _FakeMaxTextModel(self.cfg, self.mesh, self.rngs)
    with (
        mock.patch.object(adapter, "generate_maxtext_config", return_value=self.cfg),
        mock.patch.object(adapter.model_creation_utils, "from_pretrained", return_value=fake_trunk),
    ):
      target_model = adapter.MaxTextForCausalLM(vllm_config, jax.random.key(0), self.mesh)
      self.assertFalse(target_model.has_mtp())
      target_model.load_weights(jax.random.key(42))

    self.assertTrue(target_model.has_mtp())
    self.assertIs(target_model.mtp_layer, fake_trunk.mtp_block.mtp_layer_1)

    hidden = jnp.ones((4, dim), dtype=jnp.float32)
    logits = target_model.compute_mtp_logits(hidden)
    np.testing.assert_allclose(logits, jnp.dot(hidden, target_lm_head_val), rtol=1e-5, atol=1e-5)

    raw_decoder_out = jnp.ones((4, 1, dim), dtype=jnp.float32) * 2.0
    updated_draft_kv = jnp.ones((1, 4, 2, 8), dtype=jnp.float32)
    with mock.patch.object(
        target_model.mtp_layer, "mtp_1_transformer_layer", return_value=(raw_decoder_out, updated_draft_kv)
    ):
      new_kv_caches, out_hidden, residual_list, aux = target_model.mtp_forward(
          kv_caches=[jnp.zeros((1, 4, 2, 8), dtype=jnp.float32)],
          input_ids=jnp.array([1, 2, 3, 4], dtype=jnp.int32),
          target_hidden_states=jnp.ones((4, dim), dtype=jnp.float32),
          attention_metadata=types.SimpleNamespace(
              input_positions=jnp.array([0, 1, 2, 3], dtype=jnp.int32),
              seq_lens=jnp.array([4], dtype=jnp.int32),
          ),
          _layer_name_to_kvcache_index={"draft_layer.0": 0},
      )
      np.testing.assert_allclose(new_kv_caches[0], updated_draft_kv)
      expected_normed = target_model.mtp_layer.final_norm(raw_decoder_out).reshape(4, dim)
      np.testing.assert_allclose(out_hidden, expected_normed, rtol=1e-5, atol=1e-5)
      np.testing.assert_allclose(residual_list[0], expected_normed, rtol=1e-5, atol=1e-5)
      self.assertIsNone(aux)

  def test_load_mtp_block_from_hf_with_padding_and_moe_fusion(self):
    """load_mtp_block_from_hf loads mtp.* safetensors, repeats KV heads, pads MoE MLP dim, and fuses wi_0/wi_1."""
    import tempfile
    from safetensors.numpy import save_file

    cfg = _make_dummy_cfg()
    cfg.base_num_kv_heads = 4
    cfg.num_kv_heads = 4
    cfg.base_moe_mlp_dim = 12
    cfg.moe_mlp_dim = 12
    cfg.padded_base_moe_mlp_dim = 16
    cfg.prefuse_moe_weights = True

    class _ModelWithMTP(nnx.Module):

      def __init__(self, c, m, r):
        self.mtp_block = multi_token_prediction.MultiTokenPredictionBlock(
            config=c,
            mesh=m,
            transformer_layer_module=qwen3_5.Qwen3_5ScannableBlock,
            decoder=types.SimpleNamespace(model_mode=MODEL_MODE_AUTOREGRESSIVE, quant=None),
            rngs=r,
        )

    with self.mesh:
      model = _ModelWithMTP(cfg, self.mesh, self.rngs)

    emb = cfg.emb_dim  # 32
    n_q = cfg.base_num_query_heads  # 4
    hf_n_kv = 2
    head_dim = cfg.head_dim  # 8
    hf_moe_mlp = 12
    n_exp = cfg.num_experts  # 4

    tensors = {
        "mtp.pre_fc_norm_embedding.weight": np.full((emb,), 1.5, dtype=np.float32),
        "mtp.pre_fc_norm_hidden.weight": np.full((emb,), 2.5, dtype=np.float32),
        "mtp.fc.weight": np.full((emb, 2 * emb), 0.125, dtype=np.float32),
        "mtp.norm.weight": np.full((emb,), 3.5, dtype=np.float32),
        "mtp.layers.0.input_layernorm.weight": np.full((emb,), 0.5, dtype=np.float32),
        "mtp.layers.0.post_attention_layernorm.weight": np.full((emb,), 0.75, dtype=np.float32),
        "mtp.layers.0.self_attn.q_proj.weight": np.ones((n_q * 2 * head_dim, emb), dtype=np.float32),
        "mtp.layers.0.self_attn.k_proj.weight": np.arange(hf_n_kv * head_dim * emb, dtype=np.float32).reshape(
            hf_n_kv * head_dim, emb
        ),
        "mtp.layers.0.self_attn.v_proj.weight": np.ones((hf_n_kv * head_dim, emb), dtype=np.float32) * 2.0,
        "mtp.layers.0.self_attn.o_proj.weight": np.ones((emb, n_q * head_dim), dtype=np.float32) * 0.5,
        "mtp.layers.0.self_attn.q_norm.weight": np.ones((head_dim,), dtype=np.float32),
        "mtp.layers.0.self_attn.k_norm.weight": np.ones((head_dim,), dtype=np.float32),
        "mtp.layers.0.mlp.gate.weight": np.ones((n_exp, emb), dtype=np.float32) * 0.25,
        "mtp.layers.0.mlp.shared_expert.gate_proj.weight": np.ones((hf_moe_mlp, emb), dtype=np.float32),
        "mtp.layers.0.mlp.shared_expert.up_proj.weight": np.ones((hf_moe_mlp, emb), dtype=np.float32),
        "mtp.layers.0.mlp.shared_expert.down_proj.weight": np.ones((emb, hf_moe_mlp), dtype=np.float32),
        "mtp.layers.0.mlp.shared_expert_gate.weight": np.ones((1, emb), dtype=np.float32),
    }
    for e in range(n_exp):
      tensors[f"mtp.layers.0.mlp.experts.{e}.gate_proj.weight"] = np.full(
          (hf_moe_mlp, emb), float(e + 1), dtype=np.float32
      )
      tensors[f"mtp.layers.0.mlp.experts.{e}.up_proj.weight"] = np.full(
          (hf_moe_mlp, emb), float((e + 1) * 10), dtype=np.float32
      )
      tensors[f"mtp.layers.0.mlp.experts.{e}.down_proj.weight"] = np.full(
          (emb, hf_moe_mlp), float(e + 1), dtype=np.float32
      )

    hf_cfg = {
        "text_config": {
            "hidden_size": emb,
            "num_hidden_layers": 4,
            "num_attention_heads": n_q,
            "num_key_value_heads": hf_n_kv,
            "head_dim": head_dim,
            "num_experts": n_exp,
            "moe_intermediate_size": hf_moe_mlp,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
        }
    }

    with tempfile.TemporaryDirectory() as tmpdir:
      save_file(tensors, str(Path(tmpdir) / "model.safetensors"))
      ok = adapter.model_creation_utils.load_mtp_block_from_hf(
          model, cfg, self.mesh, tmpdir, hf_config_dict=hf_cfg
      )
      self.assertTrue(ok)
      self.assertTrue(getattr(model, "_mtp_block_restored", False))

    lyr = model.mtp_block.mtp_layer_1
    np.testing.assert_allclose(lyr.embedding_norm.scale.value, 1.5)
    np.testing.assert_allclose(lyr.final_norm.scale.value, 3.5)
    tl = lyr.transformer_layer
    # KV heads repeated from 2 -> 4: [h0, h0, h1, h1]
    k_val = np.asarray(tl.attention.attention.key.kernel.value)
    self.assertEqual(k_val.shape, (emb, 4, head_dim))
    np.testing.assert_allclose(k_val[:, 0, :], k_val[:, 1, :])
    np.testing.assert_allclose(k_val[:, 2, :], k_val[:, 3, :])
    # MoE wi fused and padded from 12 -> 16 per half (total 32)
    wi_val = np.asarray(tl.mlp.routed_experts.wi.value)
    self.assertEqual(wi_val.shape, (n_exp, emb, 32))
    np.testing.assert_allclose(wi_val[0, :, :12], 1.0)
    np.testing.assert_allclose(wi_val[0, :, 12:16], 0.0)
    np.testing.assert_allclose(wi_val[0, :, 16:28], 10.0)
    np.testing.assert_allclose(wi_val[0, :, 28:32], 0.0)
    # MoE wo zero-padded along axis 1 (mlp_moe) from 12 -> 16
    wo_val = np.asarray(tl.mlp.routed_experts.wo.value)
    self.assertEqual(wo_val.shape, (n_exp, 16, emb))
    np.testing.assert_allclose(wo_val[0, :12, :], 1.0)
    np.testing.assert_allclose(wo_val[0, 12:16, :], 0.0)

  def test_from_pretrained_restores_legacy_cycle_slot_scanned_checkpoint(self):
    """from_pretrained loads legacy layer_0..layer_{k-1} scanned checkpoints into local_layers/global_layer."""
    import tempfile
    from orbax import checkpoint as ocp

    cfg = _make_dummy_cfg()
    cfg.scan_layers = True
    cfg.prefuse_moe_weights = True
    cfg.mtp_num_layers = 0
    cfg.base_num_decoder_layers = 8
    cfg.num_decoder_layers = 8
    cfg.inhomogeneous_layer_cycle_interval = 4
    cfg.param_scan_axis = 1
    cfg.checkpoint_storage_concurrent_gb = 1
    cfg.checkpoint_storage_use_ocdbt = False
    cfg.checkpoint_storage_use_zarr3 = False
    cfg.convert_checkpoint_if_possible = False

    ref_model = adapter.model_creation_utils.from_config(cfg, mesh=self.mesh, rngs=nnx.Rngs(params=42, dropout=1, gating=2))

    def _to_pure(tree):
      if isinstance(tree, nnx.Variable):
        return jnp.asarray(tree.get_value())
      if hasattr(tree, "items"):
        return {k: _to_pure(v) for k, v in tree.items()}
      return tree

    pure = _to_pure(nnx.state(ref_model, nnx.Param))

    def _split_wi(node):
      if isinstance(node, dict):
        if "wi" in node:
          wi = node["wi"]
          half = wi.shape[-1] // 2
          out = {k: _split_wi(v) for k, v in node.items() if k != "wi"}
          out["wi_0"] = wi[..., :half]
          out["wi_1"] = wi[..., half:]
          return out
        return {k: _split_wi(v) for k, v in node.items()}
      return node

    legacy_layers = {}
    for slot in range(3):
      slot_tree = jax.tree.map(lambda x, s=slot: jnp.take(x, s, axis=2), pure["decoder"]["layers"]["local_layers"])
      legacy_layers[f"layer_{slot}"] = _split_wi(slot_tree)
    legacy_layers["layer_3"] = _split_wi(pure["decoder"]["layers"]["global_layer"])
    legacy_ckpt = {
        "params": {
            "params": {
                "token_embedder": pure["token_embedder"],
                "decoder": {
                    "decoder_norm": pure["decoder"]["decoder_norm"],
                    "logits_dense": pure["decoder"]["logits_dense"],
                    "layers": legacy_layers,
                },
            }
        }
    }

    with tempfile.TemporaryDirectory() as tmpdir:
      ckpt_dir = str(Path(tmpdir) / "0" / "items")
      ckptr = ocp.Checkpointer(ocp.PyTreeCheckpointHandler(use_ocdbt=False, use_zarr3=False))
      ckptr.save(ckpt_dir, legacy_ckpt)
      cfg.load_parameters_path = ckpt_dir
      with mock.patch.object(adapter.model_creation_utils, "verify_and_sync_scan_layers", side_effect=lambda c: c):
        loaded_model = adapter.model_creation_utils.from_pretrained(cfg, mesh=self.mesh)

    loaded_pure = _to_pure(nnx.state(loaded_model, nnx.Param))
    ref_leaves = jax.tree_util.tree_leaves_with_path(pure)
    loaded_leaves = jax.tree_util.tree_leaves_with_path(loaded_pure)
    self.assertEqual(len(ref_leaves), len(loaded_leaves))
    for (p1, v1), (p2, v2) in zip(ref_leaves, loaded_leaves):
      self.assertEqual(p1, p2)
      np.testing.assert_allclose(np.asarray(v1), np.asarray(v2), err_msg=str(p1))


if __name__ == "__main__":
  unittest.main()
