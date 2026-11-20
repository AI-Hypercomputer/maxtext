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

"""End-to-end, checkpoint round-trip, and registry tests for the m3 Gemma 3 model."""

from __future__ import annotations

import contextlib
import copy
import json
import os
import pathlib
import types
import unittest

from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.checkpoint_conversion import to_huggingface
from maxtext.checkpoint_conversion.utils import hf_model_configs
from maxtext.checkpoint_conversion.utils import hf_shape
from maxtext.checkpoint_conversion.utils import param_mapping
from maxtext.checkpoint_conversion.utils import tensor_handling
from maxtext.configs import pyconfig
from maxtext.configs import types as config_types
from maxtext.m3 import models as m3_models
from maxtext.m3.core import sharding as m3_sharding
from maxtext.m3.infra import quantization as m3_quant
from maxtext.m3.models.gemma3 import checkpoint_mappings_gemma3
from maxtext.m3.models.gemma3 import modeling_gemma3
from maxtext.utils import globals as maxtext_globals
from maxtext.utils import model_creation_utils
import numpy as np
import optax
import transformers

_TESTDATA = pathlib.Path(__file__).resolve().parent / "testdata" / "gemma3"
_GOLDEN_NPZ = _TESTDATA / "gemma3_hf_golden.npz"
_GOLDEN_MANIFEST = _TESTDATA / "golden_manifest.json"
_BASE_YML = os.path.join(maxtext_globals.MAXTEXT_CONFIGS_DIR, "base.yml")
_RL_YML = os.path.join(maxtext_globals.MAXTEXT_CONFIGS_DIR, "post_train", "rl.yml")
_PREFIXES = ("mini1b", "minigqa")
# (scan_layers, stack_cycles, param_scan_axis, remat_policy)
_LAYOUTS = (
    (True, True, 0, "none"),
    (False, True, 0, "none"),
    (True, True, 1, "full"),
    (False, False, 0, "minimal"),
)


def _case_hf_config(prefix: str) -> dict:
  with open(_GOLDEN_MANIFEST, "r", encoding="utf-8") as f:
    return dict(json.load(f)["cases"][prefix]["config"])


def _hf_weights(data: np.lib.npyio.NpzFile, prefix: str, kind: str) -> dict[str, np.ndarray]:
  tag = f"{prefix}__{kind}__"
  return {k[len(tag) :]: np.asarray(data[k]) for k in data.files if k.startswith(tag)}


def _flat_params(tree: nnx.State) -> dict[str, nnx.Variable]:
  """Returns `{'params-a-b-c': variable}` for an `nnx.State` of parameters or gradients."""
  return {"params-" + "-".join(str(k) for k in path): v for path, v in nnx.to_flat_state(tree)}


def _stack_axis(mt_key: str, cfg: modeling_gemma3.Gemma3Config, stacked_shape: tuple[int, ...]) -> int:
  """Stacking axis used by `to_maxtext` (`tensor_handling.single_stack_axis`)."""
  conv_cfg = types.SimpleNamespace(scan_layers=cfg.stack_cycles, param_scan_axis=cfg.param_scan_axis)
  return tensor_handling.single_stack_axis(mt_key, conv_cfg, stacked_shape)


def _load_hf(model: modeling_gemma3.Gemma3ForCausalLM, hf: dict[str, np.ndarray], hf_cfg: dict) -> None:
  """Loads HF tensors into `model` through the production mapping and hooks, like `to_maxtext`."""
  cfg = model.gemma3_config
  pmap = checkpoint_mappings_gemma3.GEMMA3_M3_MAXTEXT_TO_HF_PARAM_MAPPING(hf_cfg, cfg, cfg.stack_cycles)
  hooks = checkpoint_mappings_gemma3.GEMMA3_M3_MAXTEXT_TO_HF_PARAM_HOOK_FN(hf_cfg, cfg, cfg.stack_cycles, False)
  flat = _flat_params(nnx.state(model, nnx.Param))
  assert set(flat) == set(pmap), sorted(set(flat) ^ set(pmap))
  for mt_key, var in flat.items():
    target = var.get_value()
    hook = hooks[mt_key] if mt_key in hooks else None
    if isinstance(pmap[mt_key], list):
      axis = _stack_axis(mt_key, cfg, target.shape)
      slice_shape = tuple(d for i, d in enumerate(target.shape) if i != axis)
      slices = [tensor_handling.apply_hook_fns(hf[k], slice_shape, hook) for k in pmap[mt_key]]
      value = np.stack(slices, axis=axis)
    else:
      value = tensor_handling.apply_hook_fns(hf[pmap[mt_key]], target.shape, hook)
    assert value.shape == target.shape, (mt_key, value.shape, target.shape)
    var.set_value(jax.device_put(jnp.asarray(value, dtype=target.dtype), target.sharding))


def _export_hf(tree: nnx.State, cfg: modeling_gemma3.Gemma3Config, hf_cfg: dict) -> dict[str, np.ndarray]:
  """Exports a parameter (or gradient) tree to HF tensors through the production mapping and hooks."""
  pmap = checkpoint_mappings_gemma3.GEMMA3_M3_MAXTEXT_TO_HF_PARAM_MAPPING(hf_cfg, cfg, cfg.stack_cycles)
  hooks = checkpoint_mappings_gemma3.GEMMA3_M3_MAXTEXT_TO_HF_PARAM_HOOK_FN(hf_cfg, cfg, cfg.stack_cycles, True)
  shapes = checkpoint_mappings_gemma3.GEMMA3_M3_HF_WEIGHTS_TO_SHAPE(hf_cfg)
  out: dict[str, np.ndarray] = {}
  for mt_key, var in _flat_params(tree).items():
    arr = np.asarray(var.get_value())
    hook = hooks[mt_key] if mt_key in hooks else None
    if isinstance(pmap[mt_key], list):
      axis = _stack_axis(mt_key, cfg, arr.shape)
      for i, hf_key in enumerate(pmap[mt_key]):
        out[hf_key] = np.asarray(tensor_handling.apply_hook_fns(np.take(arr, i, axis=axis), shapes[hf_key], hook))
    else:
      hf_key = pmap[mt_key]
      out[hf_key] = np.asarray(tensor_handling.apply_hook_fns(arr, shapes[hf_key], hook))
  return out


def _ce_loss(logits: jax.Array, targets: jax.Array) -> jax.Array:
  log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
  return -jnp.mean(jnp.take_along_axis(log_probs, targets[..., None], axis=-1))


def _rel_frob(a: np.ndarray, b: np.ndarray) -> float:
  return float(np.linalg.norm(a - b) / max(1.0, float(np.linalg.norm(b))))


class Gemma3ModelTest(unittest.TestCase):
  """Golden HF parity, packing isolation, checkpoint round trip, and registry wiring for m3 Gemma 3."""

  def setUp(self) -> None:
    super().setUp()
    self._stack = contextlib.ExitStack()
    self._mesh = m3_sharding.create_mesh(devices=jax.devices()[:1])
    self._stack.enter_context(jax.set_mesh(self._mesh))
    self._stack.enter_context(nnx.use_eager_sharding(True))
    self._data = self._stack.enter_context(np.load(_GOLDEN_NPZ))

  def tearDown(self) -> None:
    self._stack.close()
    super().tearDown()

  def _build(self, prefix: str, *, scan: bool, stack: bool, axis: int, remat: str) -> modeling_gemma3.Gemma3ForCausalLM:
    """Builds the `prefix` mini-twin in fp32 with the given layout and loads its HF golden weights."""
    cfg = modeling_gemma3.Gemma3Config.from_hf_config_json(
        _case_hf_config(prefix),
        dtype="float32",
        weight_dtype="float32",
        matmul_precision="highest",
        scan_layers=scan,
        stack_cycles=stack,
        param_scan_axis=axis,
        remat_policy=remat,
    )
    model = modeling_gemma3.Gemma3ForCausalLM(cfg, self._mesh, rngs=nnx.Rngs(0))
    _load_hf(model, _hf_weights(self._data, prefix, "param"), _case_hf_config(prefix))
    return model

  def test_e2e_logits_loss_and_all_grads_match_hf_across_layouts(self) -> None:
    """Forward logits, loss, and every parameter gradient match HF goldens in all scan/stack/remat layouts."""
    for prefix in _PREFIXES:
      ids = jnp.asarray(np.asarray(self._data[f"{prefix}__input_ids"]))
      pos = jnp.asarray(np.asarray(self._data[f"{prefix}__position_ids"]))
      targets = jnp.asarray(np.asarray(self._data[f"{prefix}__targets"]))
      ref_logits = np.asarray(self._data[f"{prefix}__logits"])
      ref_loss = float(np.asarray(self._data[f"{prefix}__loss"]))
      ref_grads = _hf_weights(self._data, prefix, "grad")
      for scan, stack, axis, remat in _LAYOUTS:
        with self.subTest(prefix=prefix, scan=scan, stack=stack, axis=axis, remat=remat):
          model = self._build(prefix, scan=scan, stack=stack, axis=axis, remat=remat)

          @nnx.jit
          def _step(m, x, p, t):
            def _loss(mod):
              logits = mod(x, p)
              return _ce_loss(logits, t), logits

            return nnx.value_and_grad(_loss, has_aux=True)(m)

          (loss, logits), grads = _step(model, ids, pos, targets)
          logits_np = np.asarray(logits)
          np.testing.assert_allclose(logits_np, ref_logits, atol=1e-5, rtol=1e-5)
          self.assertLessEqual(_rel_frob(logits_np, ref_logits), 1e-5)
          np.testing.assert_allclose(float(loss), ref_loss, atol=1e-6, rtol=1e-6)
          got = _export_hf(nnx.state(grads, nnx.Param), model.gemma3_config, _case_hf_config(prefix))
          self.assertEqual(set(got), set(ref_grads))
          for hf_key, ref in ref_grads.items():
            np.testing.assert_allclose(got[hf_key], ref, atol=1e-5, rtol=1e-5, err_msg=hf_key)
            self.assertLessEqual(_rel_frob(got[hf_key], ref), 1e-5, hf_key)

  def test_single_weight_perturbation_breaks_logit_parity(self) -> None:
    """Negative control: one perturbed q_norm entry on a global layer moves logits past the parity tolerance."""
    for prefix in _PREFIXES:
      hf = _hf_weights(self._data, prefix, "param")
      hf["model.layers.5.self_attn.q_norm.weight"] = hf["model.layers.5.self_attn.q_norm.weight"].copy()
      hf["model.layers.5.self_attn.q_norm.weight"][0] += 0.25
      model = self._build(prefix, scan=True, stack=True, axis=0, remat="none")
      _load_hf(model, hf, _case_hf_config(prefix))
      ids = jnp.asarray(np.asarray(self._data[f"{prefix}__input_ids"]))
      pos = jnp.asarray(np.asarray(self._data[f"{prefix}__position_ids"]))
      logits = np.asarray(model(ids, pos))
      ref = np.asarray(self._data[f"{prefix}__logits"])
      self.assertFalse(np.allclose(logits, ref, atol=1e-5, rtol=1e-5), prefix)

  def test_right_padding_matches_hf_on_real_tokens(self) -> None:
    """Right-padded batches (`decoder_segment_ids = attention_mask`) match HF logits on non-pad tokens."""
    for prefix in _PREFIXES:
      model = self._build(prefix, scan=True, stack=True, axis=0, remat="none")
      ids = jnp.asarray(np.asarray(self._data[f"{prefix}__pad_input_ids"]))
      mask = np.asarray(self._data[f"{prefix}__pad_attention_mask"])
      pos = jnp.asarray(np.asarray(self._data[f"{prefix}__position_ids"]))
      logits = np.asarray(model(ids, pos, decoder_segment_ids=jnp.asarray(mask)))
      ref = np.asarray(self._data[f"{prefix}__pad_logits"])
      real = mask.astype(bool)
      np.testing.assert_allclose(logits[real], ref[real], atol=1e-5, rtol=1e-5)

  def test_packed_documents_match_standalone_forward_and_isolate_gradients(self) -> None:
    """A packed second document reproduces its standalone logits and gets zero gradient from the first."""
    model = self._build("minigqa", scan=True, stack=True, axis=0, remat="none")
    ids = np.asarray(self._data["minigqa__input_ids"])
    doc_a, doc_b = ids[0, :5], ids[1, :7]
    packed = jnp.asarray(np.concatenate([doc_a, doc_b])[None, :])
    seg = jnp.asarray(np.concatenate([np.full(5, 1), np.full(7, 2)])[None, :].astype(np.int32))
    pos = jnp.asarray(np.concatenate([np.arange(5), np.arange(7)])[None, :].astype(np.int32))
    packed_logits = np.asarray(model(packed, pos, decoder_segment_ids=seg))
    alone = np.asarray(model(jnp.asarray(doc_b[None, :]), jnp.arange(7, dtype=jnp.int32)[None, :]))
    np.testing.assert_allclose(packed_logits[0, 5:], alone[0], atol=1e-5, rtol=1e-5)
    # Negative control: without segment IDs document B attends to document A.
    leaky = np.asarray(model(packed, pos))
    self.assertFalse(np.allclose(leaky[0, 5:], alone[0], atol=1e-5, rtol=1e-5))

    self.assertFalse(hasattr(model.decoder, "layers_remainder"))
    emb = model.token_embedder(packed)

    def _doc_b_loss(e):
      y = model.decoder.layers(e, decoder_positions=pos, decoder_segment_ids=seg)
      for rem_idx in (6, 7):
        y = getattr(model.decoder, f"layers_{rem_idx}")(y, decoder_positions=pos, decoder_segment_ids=seg)
      return jnp.sum(jnp.square(y[:, 5:]))

    g = np.asarray(jax.grad(_doc_b_loss)(emb))
    np.testing.assert_array_equal(g[0, :5], np.zeros_like(g[0, :5]))
    self.assertGreater(float(np.abs(g[0, 5:]).max()), 0.0)

  def test_checkpoint_roundtrip_is_exact_and_multimodal_layout_maps_text_tower(self) -> None:
    """HF -> m3 -> HF is bit-exact in stacked and flat layouts; multimodal keys load, export fails closed."""
    for prefix in _PREFIXES:
      hf = _hf_weights(self._data, prefix, "param")
      hf_cfg = _case_hf_config(prefix)
      expected = {k: v for k, v in hf.items() if not (k == "lm_head.weight" and hf_cfg["tie_word_embeddings"])}
      for scan, stack, axis, _ in _LAYOUTS:
        with self.subTest(prefix=prefix, stack=stack, axis=axis):
          model = self._build(prefix, scan=scan, stack=stack, axis=axis, remat="none")
          out = _export_hf(nnx.state(model, nnx.Param), model.gemma3_config, hf_cfg)
          self.assertEqual(set(out), set(expected))
          for k, v in expected.items():
            np.testing.assert_array_equal(out[k], v, err_msg=k)

      # Multimodal `Gemma3ForConditionalGeneration` layout: text tower lives under `model.language_model`.
      mm_cfg = {"model_type": "gemma3", "text_config": hf_cfg, "vision_config": {"hidden_size": 8}}
      cfg = modeling_gemma3.Gemma3Config.from_hf_config_json(mm_cfg, dtype="float32", weight_dtype="float32")
      pmap = checkpoint_mappings_gemma3.GEMMA3_M3_MAXTEXT_TO_HF_PARAM_MAPPING(mm_cfg, cfg, True)
      text_keys = {k for v in pmap.values() for k in (v if isinstance(v, list) else [v])}
      renamed = {k.replace("model.", "model.language_model.", 1) for k in expected if k.startswith("model.")}
      renamed |= {k for k in expected if k == "lm_head.weight"}
      self.assertEqual(text_keys, renamed)
      with self.assertRaisesRegex(NotImplementedError, "vision tower"):
        checkpoint_mappings_gemma3.GEMMA3_M3_MAXTEXT_TO_HF_PARAM_HOOK_FN(mm_cfg, cfg, True, saving_to_hf=True)

    # Negative control: an identity out-projection hook cannot reproduce the HF layout.
    hf_cfg = _case_hf_config("mini1b")
    hooks = checkpoint_mappings_gemma3.GEMMA3_M3_MAXTEXT_TO_HF_PARAM_HOOK_FN(hf_cfg, None, False, False)
    o_proj = _hf_weights(self._data, "mini1b", "param")["model.layers.0.self_attn.o_proj.weight"]
    good = hooks["params-decoder-layers_0-self_attention-out-kernel"](o_proj, (4, 8, 32))
    self.assertFalse(np.array_equal(good, o_proj.reshape(4, 8, 32)))
    with self.assertRaisesRegex(ValueError, "non-empty HF config"):
      checkpoint_mappings_gemma3.GEMMA3_M3_MAXTEXT_TO_HF_PARAM_MAPPING(None, None, False)

  def test_registries_pyconfig_flops_and_engine_hook(self) -> None:
    """`gemma3-1b` loads through pyconfig and is wired in all three m3 registries and checkpoint tables."""
    for name in ("gemma3-1b", "gemma3-4b", "gemma3-12b", "gemma3-27b"):
      self.assertIn(name, m3_models.MODEL_REGISTRY)
      self.assertIn(name, m3_models.MODEL_FLOPS_REGISTRY)
      self.assertIn(name, m3_models.MODEL_ENGINE_CONFIG_REGISTRY)
    self.assertIn("gemma3-1b", param_mapping.PARAM_MAPPING)
    self.assertIn("gemma3-1b", param_mapping.HOOK_FNS)
    self.assertIn("gemma3-1b", hf_shape.HF_SHAPE)

    mt_cfg = pyconfig.initialize(
        [None, _BASE_YML],
        model_name="gemma3-1b",
        per_device_batch_size=2,
        max_target_length=1024,
        gradient_accumulation_steps=1,
        skip_jax_distributed_system=True,
    )
    self.assertTrue(mt_cfg.use_m3_model)
    g3 = modeling_gemma3.Gemma3Config.from_config(mt_cfg)
    preset = modeling_gemma3.Gemma3Config.preset("gemma3-1b")
    for field in ("vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads"):
      self.assertEqual(getattr(g3, field), getattr(preset, field), field)
    self.assertEqual(g3.num_key_value_heads, 1)
    self.assertEqual(g3.sliding_window, 512)
    self.assertEqual(g3.query_pre_attn_scalar, 256.0)
    self.assertEqual(g3.param_scan_axis, 0)

    # FLOPs: 26 layers = 22 local + 4 global; S=1024 > W=512 so local attention counts S*W - W^2/2 pairs.
    total, weight, attn = m3_models.calculate_tflops_per_device(mt_cfg, log=False)
    b, s, h, i, v, l, hq, hkv, d, w = 2.0, 1024.0, 1152.0, 6912.0, 262144.0, 26, 4.0, 1.0, 256.0, 512.0
    exp_w = 3.0 * 2.0 * b * s * (h * v + l * (h * (2 * hq * d + 2 * hkv * d) + 3 * h * i)) / 1e12
    exp_a = 3.0 * (4 * 2.0 * b * s**2 * hq * d + 22 * 4.0 * b * (s * w - 0.5 * w**2) * hq * d) / 1e12
    self.assertAlmostEqual(weight, exp_w, places=9)
    self.assertAlmostEqual(attn, exp_a, places=9)
    self.assertAlmostEqual(total, weight + attn, places=9)

    # Engine hook registers a loss whose value equals `compute_gemma3_loss`.
    model = self._build("minigqa", scan=True, stack=True, axis=0, remat="none")
    recorded = {}

    class _Engine:
      """Records the hooks `configure_gemma3_engine` installs."""

      def __init__(self):
        self._config = types.SimpleNamespace(model_name="gemma3-1b")

      def with_loss_fn(self, fn, has_aux):
        recorded["fn"], recorded["has_aux"] = fn, has_aux
        return self

      def with_gen_model_input_fn(self, fn):
        recorded["gen_model_input_fn"] = fn
        return self

    m3_models.configure_engine(_Engine())
    self.assertTrue(recorded["has_aux"])
    self.assertIsNone(recorded["gen_model_input_fn"])
    ids = jnp.asarray(np.asarray(self._data["minigqa__input_ids"]))
    tgt = jnp.asarray(np.asarray(self._data["minigqa__targets"]))
    metric, aux = recorded["fn"](model, None, {"inputs": ids, "targets": tgt}, is_train=False)
    ref = float(np.asarray(self._data["minigqa__loss"]))
    np.testing.assert_allclose(float(metric.unreduced_sum / metric.denominator), ref, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(float(aux["total_loss"]), ref, atol=1e-6, rtol=1e-6)

    # Packed + padded `gemma3_engine_loss_fn` binarizes segment IDs (segment 2 gets weight 1.0, 0 gets 0.0)
    # and contributes zero gradient from 0-masked target slots.
    b_sz, s_len = ids.shape
    in_seg = jnp.asarray(np.tile(np.array([1, 1, 1, 0, 2, 2, 2, 2, 2, 0, 0, 0], dtype=np.int32)[:s_len], (b_sz, 1)))
    in_pos = jnp.asarray(np.tile(np.array([0, 1, 2, 0, 0, 1, 2, 3, 4, 0, 0, 0], dtype=np.int32)[:s_len], (b_sz, 1)))
    t_seg = jnp.asarray(np.tile(np.array([1, 1, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0], dtype=np.int32)[:s_len], (b_sz, 1)))
    packed_batch = {
        "inputs": ids,
        "targets": tgt,
        "inputs_segmentation": in_seg,
        "inputs_position": in_pos,
        "targets_segmentation": t_seg,
    }
    packed_metric, packed_aux = recorded["fn"](model, None, packed_batch, is_train=True)
    expected_denom = float(np.count_nonzero(np.asarray(t_seg)))
    self.assertEqual(float(packed_metric.denominator), expected_denom)
    self.assertEqual(float(packed_aux["total_weights"]), expected_denom)
    packed_logits = model(ids, in_pos, decoder_segment_ids=in_seg)
    lp = jax.nn.log_softmax(packed_logits.astype(jnp.float32), axis=-1)
    per_tok = -jnp.take_along_axis(lp, tgt[..., None], axis=-1).squeeze(-1)
    expected_loss = float(jnp.sum(per_tok * (t_seg != 0).astype(jnp.float32)) / expected_denom)
    np.testing.assert_allclose(
        float(packed_metric.unreduced_sum / packed_metric.denominator), expected_loss, atol=1e-6, rtol=1e-6
    )

    def _engine_scalar_loss(m, targets_arr):
      met, _ = recorded["fn"](m, None, {**packed_batch, "targets": targets_arr}, is_train=True)
      return met.unreduced_sum / met.denominator

    loss_orig, grads_orig = nnx.value_and_grad(_engine_scalar_loss)(model, tgt)
    tgt_pad_perturbed = jnp.where(t_seg != 0, tgt, (tgt + 7) % model.gemma3_config.vocab_size)
    loss_pad_pert, grads_pad_pert = nnx.value_and_grad(_engine_scalar_loss)(model, tgt_pad_perturbed)
    np.testing.assert_array_equal(np.asarray(loss_orig), np.asarray(loss_pad_pert))
    for k, g_var in _flat_params(nnx.state(grads_orig, nnx.Param)).items():
      g_pert = _flat_params(nnx.state(grads_pad_pert, nnx.Param))[k]
      np.testing.assert_array_equal(np.asarray(g_var.get_value()), np.asarray(g_pert.get_value()), err_msg=k)
    # Negative control: perturbing a segment-2 target changes loss.
    tgt_seg2_perturbed = tgt.at[0, 5].set((tgt[0, 5] + 7) % model.gemma3_config.vocab_size)
    self.assertNotEqual(float(loss_orig), float(_engine_scalar_loss(model, tgt_seg2_perturbed)))

    # `opt_type="muon"` and `use_multimodal=True` fail closed on Gemma 3 m3.
    with self.assertRaisesRegex(ValueError, "Muon"):
      pyconfig.initialize(
          [None, _BASE_YML],
          model_name="gemma3-1b",
          opt_type="muon",
          skip_jax_distributed_system=True,
      )
    with self.assertRaisesRegex(ValueError, "Multimodal"):
      pyconfig.initialize(
          [None, _BASE_YML],
          model_name="gemma3-4b",
          use_m3_model=True,
          use_multimodal=True,
          skip_jax_distributed_system=True,
      )
    mt_4b_text = pyconfig.initialize(
        [None, _BASE_YML],
        model_name="gemma3-4b",
        use_m3_model=True,
        use_multimodal=False,
        per_device_batch_size=1,
        max_target_length=512,
        gradient_accumulation_steps=1,
        skip_jax_distributed_system=True,
    )
    self.assertGreater(m3_models.calculate_tflops_per_device(mt_4b_text, log=False)[0], 0.0)
    with self.assertRaisesRegex(ValueError, "Multimodal"):
      modeling_gemma3.Gemma3Config.from_config(mt_4b_text.replace(use_multimodal=True))
    with self.assertRaisesRegex(ValueError, "Multimodal"):
      m3_models.calculate_tflops_per_device(mt_4b_text.replace(use_multimodal=True), log=False)
    with self.assertRaisesRegex(ValueError, "use_multimodal"):
      modeling_gemma3.Gemma3Config.from_config(modeling_gemma3.BaseConfig(model_name="gemma3-4b", use_multimodal=True))
    with self.assertRaisesRegex(ValueError, "Multimodal"):
      pyconfig.initialize(
          [None, _RL_YML],
          config_class=config_types.RLConfig,
          model_name="gemma3-4b",
          use_m3_model=True,
          use_multimodal=True,
          remat_policy="full",
          skip_jax_distributed_system=True,
      )
    with self.assertRaisesRegex(ValueError, "Muon"):
      pyconfig.initialize(
          [None, _RL_YML],
          config_class=config_types.RLConfig,
          model_name="gemma3-1b",
          opt_type="muon",
          remat_policy="full",
          skip_jax_distributed_system=True,
      )

    # `m3/infra/quantization.py` covers all 7 projections per layer across scanned, unscanned, and flat layouts.
    for scan_mode, stack_mode in ((True, True), (False, True), (False, False)):
      q_model = self._build("minigqa", scan=scan_mode, stack=stack_mode, axis=0, remat="none")
      q_plan = m3_quant.plan(q_model)
      self.assertEqual(len(q_plan.quantized), q_model.gemma3_config.num_hidden_layers * 7)
      self.assertEqual(q_plan.uncovered, ())

    # Vocab-tiling output projection matches full forward pass, and non-train modes fail closed.
    pos = jnp.asarray(np.asarray(self._data["minigqa__position_ids"]))
    full_logits, hidden = model.decoder(model.token_embedder, ids, decoder_positions=pos)
    tiled_logits = model.logits_from_hidden_states_for_vocab_tiling(hidden)
    np.testing.assert_allclose(np.asarray(tiled_logits), np.asarray(full_logits), atol=1e-6, rtol=1e-6)
    with self.assertRaisesRegex(NotImplementedError, "model_mode"):
      model.logits_from_hidden_states_for_vocab_tiling(hidden, model_mode="prefill")
    with self.assertRaisesRegex(NotImplementedError, "model_mode"):
      model(ids, pos, model_mode="prefill")
    with self.assertRaisesRegex(NotImplementedError, "encoder_images"):
      model(ids, pos, encoder_images=jnp.ones((1, 4, 4, 3), dtype=jnp.float32))
    with self.assertRaisesRegex(NotImplementedError, "images"):
      recorded["fn"](
          model,
          None,
          {"inputs": ids, "targets": tgt, "images": jnp.ones((1, 4, 4, 3), dtype=jnp.float32)},
          is_train=False,
      )
    np.testing.assert_allclose(
        np.asarray(model(ids, pos, encoder_images=None, encoder_image_masks=None)),
        np.asarray(full_logits),
        atol=1e-6,
        rtol=1e-6,
    )

    # HF_SHAPE for 4B/12B/27B handles both legacy multimodal configs and text-only m3 export configs,
    # and `to_huggingface --override_model_architecture` keeps Gemma 3 `layer_types` aligned on shrunk configs.
    mm_shapes = hf_shape.HF_SHAPE["gemma3-4b"](hf_model_configs.HF_MODEL_CONFIGS["gemma3-4b"].to_dict())
    self.assertIn("model.language_model.embed_tokens.weight", mm_shapes)
    text_shapes = hf_shape.HF_SHAPE["gemma3-4b"](_case_hf_config("minigqa"))
    self.assertIn("model.embed_tokens.weight", text_shapes)

    mt_mini_export = pyconfig.initialize(
        [None, _BASE_YML],
        model_name="gemma3-1b",
        override_model_config=True,
        base_num_decoder_layers=8,
        base_emb_dim=64,
        base_mlp_dim=128,
        base_num_query_heads=4,
        base_num_kv_heads=1,
        head_dim=16,
        vocab_size=256,
        sliding_window_size=4,
        skip_jax_distributed_system=True,
    )
    hf_cfg_1b = copy.deepcopy(hf_model_configs.HF_MODEL_CONFIGS["gemma3-1b"])
    to_huggingface._validate_or_update_architecture(hf_cfg_1b, mt_mini_export, override=True)  # pylint: disable=protected-access
    self.assertEqual(hf_cfg_1b.num_hidden_layers, 8)
    self.assertEqual(len(hf_cfg_1b.layer_types), 8)
    reloaded_hf = transformers.Gemma3TextConfig.from_dict(hf_cfg_1b.to_dict())
    self.assertEqual(reloaded_hf.num_hidden_layers, 8)
    mini_mappings = to_huggingface._get_model_mappings(  # pylint: disable=protected-access
        "gemma3-1b", True, hf_cfg_1b.to_dict(), mt_mini_export
    )
    self.assertIn("model.layers.7.mlp.down_proj.weight", mini_mappings["shape_mapping"])
    stale_dict = {**hf_model_configs.HF_MODEL_CONFIGS["gemma3-1b"].to_dict(), "num_hidden_layers": 8}
    self.assertIn("model.layers.7.mlp.down_proj.weight", hf_shape.HF_SHAPE["gemma3-1b"](stale_dict))
    self.assertEqual(hf_model_configs.HF_MODEL_CONFIGS["gemma3-1b"].num_hidden_layers, 26)

  def test_create_model_eval_shape_places_every_param_on_driver_mesh(self) -> None:
    """`m3_models.create_model` on a 4-device mesh shards every parameter and rejects KV-cache decoding."""
    devices = jax.devices()
    if len(devices) < 4:
      self.skipTest(f"Requires >= 4 JAX devices, got {len(devices)}")
    self._stack.close()
    self._stack = contextlib.ExitStack()
    mt_cfg = pyconfig.initialize(
        [None, _BASE_YML],
        model_name="gemma3-1b",
        override_model_config=True,
        base_num_decoder_layers=8,
        base_emb_dim=64,
        base_mlp_dim=128,
        base_num_query_heads=4,
        base_num_kv_heads=1,
        head_dim=16,
        vocab_size=256,
        sliding_window_size=4,
        skip_jax_distributed_system=True,
    )
    mesh4 = m3_sharding.create_mesh(ici={"fsdp": 2, "tensor": 2}, devices=devices[:4])
    abstract = nnx.eval_shape(lambda: m3_models.create_model(mt_cfg, mesh4, rngs=nnx.Rngs(0), quant_mode_str="train"))
    self.assertIs(abstract.mesh, mesh4)
    for key, var in _flat_params(nnx.state(abstract, nnx.Param)).items():
      self.assertIsNotNone(var.get_metadata()["out_sharding"], key)
      self.assertFalse(var.get_metadata()["eager_sharding"], key)

    # Mesh-free (`mesh=None`) and direct `Gemma3ForCausalLM(cfg, mesh4)` construction with no ambient mesh.
    with jax.sharding.use_abstract_mesh(jax.sharding.AbstractMesh((), ())):
      unsharded = m3_models.create_model(mt_cfg, None, rngs=nnx.Rngs(0), quant_mode_str="train")
      self.assertIsNone(unsharded.sharding)
      unsharded_direct = modeling_gemma3.create_gemma3_model(mt_cfg, None, rngs=nnx.Rngs(0))
      self.assertIsNone(unsharded_direct.sharding)
      self.assertEqual(unsharded_direct(jnp.ones((2, 4), jnp.int32)).shape, (2, 4, 256))

      direct_sharded = modeling_gemma3.Gemma3ForCausalLM(mt_cfg, mesh4, rngs=nnx.Rngs(0))
      for key, var in _flat_params(nnx.state(direct_sharded, nnx.Param)).items():
        self.assertIsNotNone(var.get_metadata()["out_sharding"], key)
        self.assertFalse(var.get_metadata()["eager_sharding"], key)
      opt = nnx.Optimizer(direct_sharded, optax.adamw(1e-3), wrt=nnx.Param)
      self.assertIsNotNone(opt)

    mesh1 = m3_sharding.create_mesh(devices=devices[:1])
    with jax.set_mesh(mesh1):
      with self.assertRaisesRegex(ValueError, "different mesh"):
        m3_models.create_model(mt_cfg, mesh4, rngs=nnx.Rngs(0), quant_mode_str="train")
      with self.assertRaisesRegex(ValueError, "different mesh"):
        modeling_gemma3.Gemma3ForCausalLM(mt_cfg, mesh4, rngs=nnx.Rngs(0))

    # `create_nnx_abstract_model` preserves `P(None, None)` on stacked 1D norms across `param_scan_axis` 0 and 1.
    for scan_axis in (0, 1):
      mt_cfg_axis = pyconfig.initialize(
          [None, _BASE_YML],
          model_name="gemma3-1b",
          override_model_config=True,
          base_num_decoder_layers=12,
          base_emb_dim=64,
          base_mlp_dim=128,
          base_num_query_heads=4,
          base_num_kv_heads=1,
          head_dim=16,
          vocab_size=256,
          sliding_window_size=4,
          param_scan_axis=scan_axis,
          skip_jax_distributed_system=True,
      )
      _, abs_utils = model_creation_utils.create_nnx_abstract_model(mt_cfg_axis, mesh4)
      abs_direct = nnx.eval_shape(
          lambda cfg=mt_cfg_axis: m3_models.create_model(cfg, mesh4, rngs=nnx.Rngs(0), quant_mode_str="train")
      )
      utils_flat = _flat_params(nnx.state(abs_utils, nnx.Param))
      direct_flat = _flat_params(nnx.state(abs_direct, nnx.Param))
      self.assertEqual(set(utils_flat), set(direct_flat))
      for key, var in utils_flat.items():
        got_spec = var.get_value().sharding.spec
        exp_spec = direct_flat[key].get_metadata()["out_sharding"]
        self.assertEqual(got_spec, exp_spec, f"scan_axis={scan_axis} {key}")
        if key.startswith("params-decoder-layers-") and key.endswith("_norm-weight"):
          self.assertEqual(got_spec, jax.sharding.PartitionSpec(None, None), key)

    with jax.set_mesh(mesh4):
      model = m3_models.create_model(mt_cfg, mesh4, rngs=nnx.Rngs(0), quant_mode_str="train")
      with self.assertRaisesRegex(NotImplementedError, "KV-cache"):
        model(jnp.ones((2, 4), jnp.int32), cache=object())


if __name__ == "__main__":
  if "XLA_FLAGS" not in os.environ:
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
  unittest.main()
