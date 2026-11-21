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

"""Provenance, SHA-256 integrity, and HF regeneration tests for Gemma 3 m3 goldens."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import importlib.util
import inspect
import json
import pathlib
import re
from typing import Any
import unittest

import numpy as np

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None

_HERE = pathlib.Path(__file__).resolve().parent
_TESTDATA_DIR = _HERE / "testdata" / "gemma3"
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_SUBSTRINGS = ("google" + "3", "blaze ", "/usr/local/", "/home/")

PINNED_REFERENCE_MODULES: dict[str, str] = {
    "transformers.models.gemma3.configuration_gemma3": (
        "0e4da9c802622202a8aaa63fc4909c192a2bdde0078f4250ffd53b4a779219d9"
    ),
    "transformers.models.gemma3.modeling_gemma3": ("4daae3bdf1a6b345df76014a78ce1ca5e66d8e3db00abf565e1bde4198b43855"),
}
PINNED_ORACLE_FILES: dict[str, str] = {}
GOLDEN_MANIFEST = "testdata/gemma3/golden_manifest.json"
GOLDEN_MANIFEST_SHA256 = "e5b0eb77b8f36cf7d2a02a8dfa610710a6133f0e84bb741f7ad0af53d37f4a57"
GOLDEN_NPZ_SHA256 = "25982e5d3416736f6e6ba51c656a6fbb73c4fe35495ee035c40b1a11bd04cda7"


def _sha256_bytes(payload: bytes) -> str:
  return hashlib.sha256(payload).hexdigest()


def verify_gemma3_golden_provenance(
    manifest: Mapping[str, Any],
    npz_bytes: bytes,
) -> None:
  """Verifies manifest schema, clean provenance strings, and golden SHA-256."""
  raw_json = json.dumps(manifest, sort_keys=True)
  for forbidden in _FORBIDDEN_SUBSTRINGS:
    if forbidden in raw_json:
      raise ValueError(f"Manifest contains non-portable substring {forbidden!r}.")

  if "model_family" not in manifest or manifest["model_family"] != "gemma3":
    raise ValueError("Unexpected or missing model_family in golden manifest.")
  if "command" not in manifest or not isinstance(manifest["command"], str) or not manifest["command"].strip():
    raise ValueError("Missing generating command in golden manifest.")

  if "reference" not in manifest or not isinstance(manifest["reference"], Mapping):
    raise ValueError("Missing reference block in golden manifest.")
  ref = manifest["reference"]
  if "pinned_modules" not in ref or ref["pinned_modules"] != PINNED_REFERENCE_MODULES:
    raise ValueError("Manifest reference.pinned_modules does not match PINNED_REFERENCE_MODULES.")
  if "pinned_oracle_files" not in ref or ref["pinned_oracle_files"] != PINNED_ORACLE_FILES:
    raise ValueError("Manifest reference.pinned_oracle_files does not match PINNED_ORACLE_FILES.")
  for sha_key in ("configuration_gemma3_sha256", "modeling_gemma3_sha256"):
    if sha_key not in ref:
      raise ValueError(f"Missing {sha_key} in golden manifest reference.")
    val = ref[sha_key]
    if not isinstance(val, str) or not _HEX64_RE.match(val):
      raise ValueError(f"Invalid {sha_key}: {val!r}.")

  if "golden_archive" not in manifest or not isinstance(manifest["golden_archive"], Mapping):
    raise ValueError("Missing golden_archive block in golden manifest.")
  archive = manifest["golden_archive"]
  if "sha256" not in archive:
    raise ValueError("Missing golden_archive.sha256 in golden manifest.")
  expected_sha = archive["sha256"]
  if not isinstance(expected_sha, str) or not _HEX64_RE.match(expected_sha):
    raise ValueError(f"Invalid golden_archive.sha256: {expected_sha!r}.")

  if "size_bytes" not in archive:
    raise ValueError("Missing golden_archive.size_bytes in golden manifest.")
  expected_size = archive["size_bytes"]
  if len(npz_bytes) != expected_size:
    raise ValueError(f"Golden archive byte size mismatch: got {len(npz_bytes)}, expected {expected_size}.")

  actual_sha = _sha256_bytes(npz_bytes)
  if actual_sha != expected_sha:
    raise ValueError(f"Golden archive SHA-256 mismatch: got {actual_sha}, expected {expected_sha}.")


class Gemma3ProvenanceTest(unittest.TestCase):
  """Torch-free and optional live-HF provenance checks for Gemma 3 m3 goldens."""

  def setUp(self) -> None:
    super().setUp()
    self.manifest_path = _HERE / GOLDEN_MANIFEST
    self.npz_path = _TESTDATA_DIR / "gemma3_hf_golden.npz"
    self.manifest_bytes = self.manifest_path.read_bytes()
    self.manifest = json.loads(self.manifest_bytes.decode("utf-8"))
    self.npz_bytes = self.npz_path.read_bytes()

  def test_manifest_schema_and_sha256_integrity(self) -> None:
    """Verifies pinned manifest/archive SHA-256s and reference module pins."""
    self.assertTrue(PINNED_REFERENCE_MODULES or PINNED_ORACLE_FILES)
    self.assertEqual(_sha256_bytes(self.manifest_bytes), GOLDEN_MANIFEST_SHA256)
    self.assertEqual(_sha256_bytes(self.npz_bytes), GOLDEN_NPZ_SHA256)
    verify_gemma3_golden_provenance(self.manifest, self.npz_bytes)

    goldens = self.manifest["goldens"]
    self.assertGreaterEqual(len(goldens), 1)
    for entry in goldens:
      local_bytes = (_HERE / entry["local_path"]).read_bytes()
      self.assertEqual(_sha256_bytes(local_bytes), entry["sha256"])
      self.assertTrue(entry["command"].strip())

    self.assertEqual(self.manifest["reference"]["deviations"], [])
    self.assertEqual(self.manifest["reference"]["precision"], "float32")
    self.assertIn("mini1b", self.manifest["cases"])
    self.assertIn("minigqa", self.manifest["cases"])

  def test_manifest_integrity_negative_control(self) -> None:
    """Verifies provenance validation fails closed on byte, hash, or pin tampering."""
    verify_gemma3_golden_provenance(self.manifest, self.npz_bytes)

    corrupted = bytearray(self.npz_bytes)
    corrupted[-1] ^= 0x01
    with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
      verify_gemma3_golden_provenance(self.manifest, bytes(corrupted))

    bad_manifest = json.loads(json.dumps(self.manifest))
    bad_manifest["golden_archive"]["size_bytes"] += 1
    with self.assertRaisesRegex(ValueError, "byte size mismatch"):
      verify_gemma3_golden_provenance(bad_manifest, self.npz_bytes)

    tampered_pins = json.loads(json.dumps(self.manifest))
    tampered_pins["reference"]["pinned_modules"]["transformers.models.gemma3.modeling_gemma3"] = "0" * 64
    with self.assertRaisesRegex(ValueError, "PINNED_REFERENCE_MODULES"):
      verify_gemma3_golden_provenance(tampered_pins, self.npz_bytes)

  def test_golden_tensor_inventory_finiteness_and_swa_locality(self) -> None:
    """Verifies tensor inventory, non-zero RMSNorm scales, SWA locality, and right-padding invariance."""
    declared = self.manifest["tensors"]
    with np.load(self.npz_path) as data:
      actual_keys = set(data.files)
      self.assertEqual(actual_keys, set(declared.keys()))
      self.assertEqual(len(actual_keys), self.manifest["golden_archive"]["num_arrays"])

      norm_weight_count = 0
      grad_count = 0
      act_count = 0
      for key, meta in declared.items():
        arr = np.asarray(data[key])
        self.assertEqual(list(arr.shape), meta["shape"], msg=f"Shape mismatch on {key}")
        self.assertEqual(str(arr.dtype), meta["dtype"], msg=f"Dtype mismatch on {key}")
        self.assertTrue(np.all(np.isfinite(arr)), msg=f"Non-finite values in {key}")

        if "__param__" in key and ("layernorm" in key or "_norm" in key or key.endswith("model.norm.weight")):
          norm_weight_count += 1
          self.assertGreater(
              float(np.max(np.abs(arr))),
              0.05,
              msg=f"RMSNorm weight {key} is near-zero and would not test (1 + weight).",
          )
        if "__grad__" in key:
          grad_count += 1
          self.assertGreater(
              float(np.linalg.norm(arr)),
              0.0,
              msg=f"Parameter gradient {key} is identically zero.",
          )
        if "__act__" in key:
          act_count += 1

      self.assertGreaterEqual(norm_weight_count, 22 * 6 + 2)
      self.assertGreater(grad_count, 200)
      self.assertEqual(act_count, 28)

      for prefix in ("mini1b", "minigqa"):
        l0_orig = np.asarray(data[f"{prefix}__act__layer0_out"])
        l0_pert = np.asarray(data[f"{prefix}__pert_tok0_layer0_out"])
        l5_orig = np.asarray(data[f"{prefix}__act__layer5_out"])
        l5_pert = np.asarray(data[f"{prefix}__pert_tok0_layer5_out"])

        # Token 0 sits within sliding_window=4 for positions 0..3 and outside for 4..11.
        self.assertGreater(float(np.max(np.abs(l0_pert[:, :4, :] - l0_orig[:, :4, :]))), 1e-3)
        np.testing.assert_array_equal(l0_pert[:, 4:, :], l0_orig[:, 4:, :])
        # Global full attention at layer 5 propagates token 0 across positions 4..11.
        self.assertGreater(float(np.max(np.abs(l5_pert[:, 4:, :] - l5_orig[:, 4:, :]))), 1e-4)

        # Causal right-padding invariance on valid prefixes (row 0 len 9, row 1 len 7).
        logits = np.asarray(data[f"{prefix}__logits"])
        pad_logits = np.asarray(data[f"{prefix}__pad_logits"])
        np.testing.assert_allclose(pad_logits[0, :9, :], logits[0, :9, :], atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(pad_logits[1, :7, :], logits[1, :7, :], atol=1e-6, rtol=1e-6)
        self.assertGreater(float(np.max(np.abs(pad_logits[0, 9:, :] - logits[0, 9:, :]))), 1e-3)
        self.assertGreater(float(np.max(np.abs(pad_logits[1, 7:, :] - logits[1, 7:, :]))), 1e-3)

  @unittest.skipUnless(
      HAS_TORCH and HAS_TRANSFORMERS,
      "Requires torch and transformers for live HF Gemma 3 regeneration check.",
  )
  def test_live_unmodified_hf_regeneration_matches_goldens(self) -> None:
    """Re-runs unmodified HF Gemma3ForCausalLM and verifies full forward/act/probe/grad parity."""
    import torch  # pylint: disable=import-outside-toplevel
    import torch.nn.functional as F  # pylint: disable=import-outside-toplevel
    import transformers  # pylint: disable=import-outside-toplevel
    from transformers.models.gemma3 import Gemma3ForCausalLM  # pylint: disable=import-outside-toplevel
    from transformers.models.gemma3 import Gemma3TextConfig  # pylint: disable=import-outside-toplevel
    from transformers.models.gemma3 import configuration_gemma3  # pylint: disable=import-outside-toplevel
    from transformers.models.gemma3 import modeling_gemma3  # pylint: disable=import-outside-toplevel

    ref_meta = self.manifest["reference"]
    if transformers.__version__ == ref_meta["transformers_version"]:
      cfg_src = inspect.getsourcefile(configuration_gemma3)
      mod_src = inspect.getsourcefile(modeling_gemma3)
      self.assertIsNotNone(cfg_src)
      self.assertIsNotNone(mod_src)
      assert cfg_src is not None
      assert mod_src is not None
      self.assertEqual(
          _sha256_bytes(pathlib.Path(cfg_src).read_bytes()),
          PINNED_REFERENCE_MODULES["transformers.models.gemma3.configuration_gemma3"],
      )
      self.assertEqual(
          _sha256_bytes(pathlib.Path(mod_src).read_bytes()),
          PINNED_REFERENCE_MODULES["transformers.models.gemma3.modeling_gemma3"],
      )

    with np.load(self.npz_path) as data:
      for prefix, case_info in self.manifest["cases"].items():
        cfg_dict = case_info["config"]
        cfg = Gemma3TextConfig(**cfg_dict, _attn_implementation="eager")
        model = Gemma3ForCausalLM(cfg).to(dtype=torch.float32)
        model.eval()

        state_dict = {
            k.split("__param__", 1)[1]: torch.from_numpy(np.asarray(data[k]))
            for k in data.files
            if k.startswith(f"{prefix}__param__")
        }
        model.load_state_dict(state_dict, strict=True)
        if cfg.tie_word_embeddings:
          model.lm_head.weight = model.model.embed_tokens.weight

        captured: dict[str, np.ndarray] = {}

        def _make_hook(store: dict[str, np.ndarray], tag: str):
          def _fn(_mod, _inp, out):
            tensor = out[0] if isinstance(out, tuple) else out
            store[tag] = tensor.detach().cpu().numpy().astype(np.float32)

          return _fn

        last_idx = int(cfg_dict["num_hidden_layers"]) - 1
        hooks = [
            model.model.embed_tokens.register_forward_hook(_make_hook(captured, "embed_tokens")),
            model.model.layers[0].input_layernorm.register_forward_hook(_make_hook(captured, "layer0_input_layernorm")),
            model.model.layers[0].self_attn.q_norm.register_forward_hook(_make_hook(captured, "layer0_q_norm")),
            model.model.layers[0].self_attn.k_norm.register_forward_hook(_make_hook(captured, "layer0_k_norm")),
            model.model.layers[0].self_attn.register_forward_hook(_make_hook(captured, "layer0_self_attn")),
            model.model.layers[0].post_attention_layernorm.register_forward_hook(
                _make_hook(captured, "layer0_post_attention_layernorm")
            ),
            model.model.layers[0].pre_feedforward_layernorm.register_forward_hook(
                _make_hook(captured, "layer0_pre_feedforward_layernorm")
            ),
            model.model.layers[0].mlp.register_forward_hook(_make_hook(captured, "layer0_mlp")),
            model.model.layers[0].post_feedforward_layernorm.register_forward_hook(
                _make_hook(captured, "layer0_post_feedforward_layernorm")
            ),
            model.model.layers[0].register_forward_hook(_make_hook(captured, "layer0_out")),
            model.model.layers[5].self_attn.register_forward_hook(_make_hook(captured, "layer5_self_attn")),
            model.model.layers[5].register_forward_hook(_make_hook(captured, "layer5_out")),
            model.model.layers[last_idx].register_forward_hook(_make_hook(captured, f"layer{last_idx}_out")),
            model.model.norm.register_forward_hook(_make_hook(captured, "final_norm")),
        ]

        input_ids = torch.from_numpy(np.asarray(data[f"{prefix}__input_ids"])).long()
        targets = torch.from_numpy(np.asarray(data[f"{prefix}__targets"])).long()
        position_ids = torch.from_numpy(np.asarray(data[f"{prefix}__position_ids"])).long()

        model.zero_grad(set_to_none=True)
        out = model(input_ids=input_ids, position_ids=position_ids, use_cache=False)
        log_probs = F.log_softmax(out.logits.float(), dim=-1)
        nll = -torch.gather(log_probs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        loss = nll.mean()
        loss.backward()

        for handle in hooks:
          handle.remove()

        expected_act_tags = {k.split("__act__", 1)[1] for k in data.files if k.startswith(f"{prefix}__act__")}
        self.assertEqual(set(captured.keys()), expected_act_tags)
        for tag, act_val in captured.items():
          np.testing.assert_allclose(
              act_val,
              np.asarray(data[f"{prefix}__act__{tag}"]),
              atol=1e-5,
              rtol=1e-5,
              err_msg=f"Activation mismatch on {prefix}:{tag}",
          )

        np.testing.assert_allclose(
            out.logits.detach().cpu().numpy(),
            np.asarray(data[f"{prefix}__logits"]),
            atol=1e-5,
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            float(loss.detach().cpu().item()),
            float(np.asarray(data[f"{prefix}__loss"])),
            atol=1e-6,
            rtol=1e-6,
        )

        expected_grad_names = {k.split("__grad__", 1)[1] for k in data.files if k.startswith(f"{prefix}__grad__")}
        actual_grad_names: set[str] = set()
        for name, param in model.named_parameters():
          self.assertIsNotNone(param.grad, msg=f"Unexpected None gradient on {prefix}:{name}")
          assert param.grad is not None
          actual_grad_names.add(name)
          np.testing.assert_allclose(
              param.grad.detach().cpu().numpy(),
              np.asarray(data[f"{prefix}__grad__{name}"]),
              atol=1e-5,
              rtol=1e-5,
              err_msg=f"Gradient mismatch on {prefix}:{name}",
          )
        self.assertEqual(actual_grad_names, expected_grad_names)

        # Verify live right-padded and token-0-perturbed probe passes.
        pad_ids = torch.from_numpy(np.asarray(data[f"{prefix}__pad_input_ids"])).long()
        pad_mask = torch.from_numpy(np.asarray(data[f"{prefix}__pad_attention_mask"])).long()
        pert_ids = torch.from_numpy(np.asarray(data[f"{prefix}__pert_tok0_input_ids"])).long()
        with torch.no_grad():
          out_pad = model(
              input_ids=pad_ids,
              attention_mask=pad_mask,
              position_ids=position_ids,
              use_cache=False,
          )
          np.testing.assert_allclose(
              out_pad.logits.cpu().numpy(),
              np.asarray(data[f"{prefix}__pad_logits"]),
              atol=1e-5,
              rtol=1e-5,
          )
          swa_cap: dict[str, np.ndarray] = {}
          h0 = model.model.layers[0].register_forward_hook(_make_hook(swa_cap, "l0"))
          h5 = model.model.layers[5].register_forward_hook(_make_hook(swa_cap, "l5"))
          _ = model(input_ids=pert_ids, position_ids=position_ids, use_cache=False)
          h0.remove()
          h5.remove()
          np.testing.assert_allclose(
              swa_cap["l0"],
              np.asarray(data[f"{prefix}__pert_tok0_layer0_out"]),
              atol=1e-5,
              rtol=1e-5,
          )
          np.testing.assert_allclose(
              swa_cap["l5"],
              np.asarray(data[f"{prefix}__pert_tok0_layer5_out"]),
              atol=1e-5,
              rtol=1e-5,
          )


if __name__ == "__main__":
  unittest.main()
