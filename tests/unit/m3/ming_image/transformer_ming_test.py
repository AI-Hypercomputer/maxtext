# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the Ming-Image diffusion transformer (`transformer_ming.py`).

The parity test runs the Ming-Image-Ant PyTorch reference (`diffusion/transformer.py::DiffusionTransformer`) when its
repo is available (`MING_IMAGE_REPO`, default `~/workspace/diffusion/ming/Ming-Image-Ant`); it is skipped otherwise.
"""

import json
import os
import sys
import unittest

import pytest

pytest.importorskip("diffusers")  # maxtext.m3.models.wan (attention op) needs diffusers.

# pylint: disable=wrong-import-position
from flax import nnx
from flax.traverse_util import flatten_dict
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.m3.models.ming_image.transformer_ming import MingImageTransformer, transformer_param_mapping
from maxtext.m3.models.wan import utils_wan
from maxtext.m3.models.wan.utils_wan import WAN_MESH_AXES, WanSharding

# pylint: enable=wrong-import-position

# The released Ming-Image-0.1-Design transformer/config.json.
_FULL_CONFIG = {
    "all_f_patch_size": [1],
    "all_patch_size": [2],
    "axes_dims": [32, 48, 48],
    "axes_lens": [20480, 512, 512],
    "cap_feat_dim": 2560,
    "dim": 3840,
    "in_channels": 16,
    "n_heads": 30,
    "n_kv_heads": 30,
    "n_layers": 30,
    "n_refiner_layers": 2,
    "norm_eps": 1e-05,
    "qk_norm": True,
    "rope_theta": 256.0,
    "siglip_feat_dim": None,
    "t_scale": 1000.0,
    "alignment_padding_mode": "zero_masked",
    "multi_frame_output": False,
}
# Tiny config: head_dim 32 = sum(axes_dims); FFN hidden int(64 / 3 * 8) = 170.
_TINY_CONFIG = {
    **_FULL_CONFIG,
    "axes_dims": [8, 12, 12],
    "axes_lens": [256, 32, 32],
    "cap_feat_dim": 24,
    "dim": 64,
    "in_channels": 4,
    "n_heads": 2,
    "n_kv_heads": 2,
    "n_layers": 2,
    "n_refiner_layers": 1,
}
_INDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "testdata", "ming_image_transformer.safetensors.index.json")
_REF_REPO = os.path.expanduser(os.environ.get("MING_IMAGE_REPO", "~/workspace/diffusion/ming/Ming-Image-Ant"))


def _model_kwargs(config):
  return {k: v for k, v in config.items() if k != "siglip_feat_dim"}


def _param_keys(model) -> dict:
  params = nnx.state(model, nnx.Param).to_pure_dict()
  return {"-".join(map(str, path)): leaf.shape for path, leaf in flatten_dict(params).items()}


def _load_hf_weights(model, hf_weights: dict, config: dict, scan_layers: bool = False):
  """Loads `{hf_key: np.ndarray}` into `model` through `transformer_param_mapping` (as the converter does)."""
  mapping, hooks = transformer_param_mapping(config, scan_layers=scan_layers)
  state = nnx.state(model, nnx.Param)
  flat = flatten_dict(state.to_pure_dict())
  new_flat = {}
  for path, leaf in flat.items():
    key = "-".join(map(str, path))
    hf_keys = mapping[key]
    hook = hooks.get(key, lambda x, _: x)
    if isinstance(hf_keys, list):
      value = np.stack([hook(hf_weights[k], leaf.shape[1:]) for k in hf_keys], axis=0)
    else:
      value = hook(hf_weights[hf_keys], leaf.shape)
    assert value.shape == leaf.shape, (key, value.shape, leaf.shape)
    new_flat[path] = jnp.asarray(value, leaf.dtype)
  pure = {}
  for path, value in new_flat.items():
    node = pure
    for part in path[:-1]:
      node = node.setdefault(part, {})
    node[path[-1]] = value
  nnx.replace_by_pure_dict(state, pure)
  nnx.update(model, state)


def _random_inputs(config, rng, cap_lens, cap2_lens, *, height=8, width=8):
  """Padded JAX inputs; the padded caption slots hold random values (they must be ignored via the masks)."""
  batch = len(cap_lens)
  c, dim, cap_dim = config["in_channels"], config["dim"], config["cap_feat_dim"]
  lc, lc2 = max(cap_lens), max(cap2_lens) if cap2_lens else 0
  latents = rng.standard_normal((batch, height, width, c)).astype(np.float32)
  timestep = rng.uniform(0.0, 1.0, (batch,)).astype(np.float32)
  cap = rng.standard_normal((batch, lc, cap_dim)).astype(np.float32)
  cap_mask = np.arange(lc)[None, :] < np.asarray(cap_lens)[:, None]
  cap2 = cap2_mask = None
  if cap2_lens:
    cap2 = rng.standard_normal((batch, lc2, dim)).astype(np.float32)
    cap2_mask = np.arange(lc2)[None, :] < np.asarray(cap2_lens)[:, None]
  return latents, timestep, cap, cap_mask, cap2, cap2_mask


class MingImageTransformerTest(unittest.TestCase):
  """Shapes, parity vs. the PyTorch reference and checkpoint-mapping coverage."""

  def test_forward_shape(self):
    model = MingImageTransformer(rngs=nnx.Rngs(0), **_model_kwargs(_TINY_CONFIG))
    latents, t, cap, cap_mask, cap2, cap2_mask = _random_inputs(_TINY_CONFIG, np.random.default_rng(0), [5, 9], [3, 4])
    out = nnx.jit(lambda m, *a: m(*a))(model, latents, t, cap, cap_mask, cap2, cap2_mask)
    self.assertEqual(out.shape, latents.shape)
    self.assertTrue(np.all(np.isfinite(np.asarray(out))))

  def test_scan_matches_unscanned(self):
    """`scan_layers=True` gives the same output for the same weights."""
    rng = np.random.default_rng(1)
    unscanned = MingImageTransformer(rngs=nnx.Rngs(0), **_model_kwargs(_TINY_CONFIG))
    hf_weights = {}
    mapping, _ = transformer_param_mapping(_TINY_CONFIG)
    for key, shape in _param_keys(unscanned).items():
      # Kernels are stored transposed (PyTorch (out, in)) in the HF checkpoint.
      hf_shape = shape[::-1] if key.endswith("kernel") else shape
      hf_weights[mapping[key]] = (rng.standard_normal(hf_shape) * 0.1).astype(np.float32)
    _load_hf_weights(unscanned, hf_weights, _TINY_CONFIG)
    scanned = MingImageTransformer(rngs=nnx.Rngs(0), scan_layers=True, **_model_kwargs(_TINY_CONFIG))
    _load_hf_weights(scanned, hf_weights, _TINY_CONFIG, scan_layers=True)
    inputs = _random_inputs(_TINY_CONFIG, rng, [5, 9], [3, 4])
    np.testing.assert_allclose(np.asarray(scanned(*inputs)), np.asarray(unscanned(*inputs)), atol=1e-5, rtol=1e-5)

  def test_wan_sharding_rules(self):
    """The Wan `logical_axis_rules` table covers every logical axis; sharded == unsharded on a 1-device mesh."""
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape((1,) * len(WAN_MESH_AXES)), WAN_MESH_AXES)
    sharding = WanSharding.from_rules(utils_wan.yml_rules("logical_axis_rules"), WAN_MESH_AXES)
    nnx.eval_shape(lambda: MingImageTransformer(rngs=nnx.Rngs(0), sharding=sharding, **_model_kwargs(_FULL_CONFIG)))
    inputs = _random_inputs(_TINY_CONFIG, np.random.default_rng(2), [5, 9], [3, 4])
    plain = MingImageTransformer(rngs=nnx.Rngs(0), **_model_kwargs(_TINY_CONFIG))
    expected = np.asarray(plain(*inputs))
    with jax.set_mesh(mesh):
      sharded = MingImageTransformer(rngs=nnx.Rngs(0), sharding=sharding, **_model_kwargs(_TINY_CONFIG))
      out = np.asarray(nnx.jit(lambda m, *a: m(*a))(sharded, *inputs))
    np.testing.assert_allclose(out, expected, atol=1e-5, rtol=1e-5)

  def test_full_config_param_mapping_coverage(self):
    """Every param of the real-size model is mapped, and every mapped HF key is in the checkpoint index."""
    with open(_INDEX_PATH, "rt", encoding="utf-8") as f:
      index_keys = set(json.load(f)["weight_map"])
    for scan_layers in (False, True):
      model = nnx.eval_shape(
          lambda s=scan_layers: MingImageTransformer(rngs=nnx.Rngs(0), scan_layers=s, **_model_kwargs(_FULL_CONFIG))
      )
      param_keys = _param_keys(model)
      mapping, hooks = transformer_param_mapping(_FULL_CONFIG, scan_layers=scan_layers)
      self.assertEqual(set(mapping), set(param_keys))
      self.assertLessEqual(set(hooks), set(mapping))
      hf_keys = set()
      for value in mapping.values():
        hf_keys.update(value if isinstance(value, list) else [value])
      self.assertEqual(hf_keys, index_keys)
      self.assertEqual(
          param_keys["layers-0-feed_forward-w1-kernel" if not scan_layers else "layers-feed_forward-w1-kernel"],
          (3840, 10240) if not scan_layers else (30, 3840, 10240),
      )


@unittest.skipUnless(os.path.isdir(os.path.join(_REF_REPO, "diffusion")), "Ming-Image-Ant reference repo not found")
class MingImageTransformerParityTest(unittest.TestCase):
  """Numerical parity with the PyTorch reference on a tiny random config (fp32, CPU)."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    torch = pytest.importorskip("torch")
    if _REF_REPO not in sys.path:
      sys.path.insert(0, _REF_REPO)
    from diffusion.transformer import DiffusionTransformer  # pylint: disable=import-outside-toplevel,import-error

    cls.torch = torch
    torch.manual_seed(0)
    ref = DiffusionTransformer(**_model_kwargs(_TINY_CONFIG)).eval()  # The reference has no siglip_feat_dim.
    with torch.no_grad():
      for name, p in ref.named_parameters():
        if "norm" in name and p.ndim == 1:
          p.copy_(1.0 + 0.2 * torch.randn_like(p))
        elif p.ndim == 2:  # O(1) activations: std 1 / sqrt(fan_in).
          p.copy_(torch.randn_like(p) / p.shape[1] ** 0.5)
        else:
          p.copy_(0.2 * torch.randn_like(p))
    cls.ref = ref
    cls.hf_weights = {k: v.detach().numpy() for k, v in ref.state_dict().items()}

  def _run_parity(self, cap_lens, cap2_lens, scan_layers=False, height=8, width=8):
    """Runs both models on one padded batch (reference: the per-sample unpadded lists) and compares outputs."""
    torch = self.torch
    model = MingImageTransformer(rngs=nnx.Rngs(0), scan_layers=scan_layers, **_model_kwargs(_TINY_CONFIG))
    _load_hf_weights(model, self.hf_weights, _TINY_CONFIG, scan_layers=scan_layers)
    rng = np.random.default_rng(42)
    latents, t, cap, cap_mask, cap2, cap2_mask = _random_inputs(
        _TINY_CONFIG, rng, cap_lens, cap2_lens, height=height, width=width
    )
    out = np.asarray(nnx.jit(lambda m, *a: m(*a))(model, latents, t, cap, cap_mask, cap2, cap2_mask))

    with torch.no_grad():
      x_list = [torch.from_numpy(latents[i]).permute(2, 0, 1)[:, None] for i in range(len(cap_lens))]  # [C, 1, H, W]
      cap_list = [torch.from_numpy(cap[i, : cap_lens[i]]) for i in range(len(cap_lens))]
      cap2_list = None
      if cap2_lens:
        cap2_list = [torch.from_numpy(cap2[i, : cap2_lens[i]]) for i in range(len(cap_lens))]
      ref_out = self.ref(x_list, torch.from_numpy(t), cap_list, patch_size=2, f_patch_size=1, cap_feats_2=cap2_list)
      ref_out = ref_out.sample if hasattr(ref_out, "sample") else ref_out[0]
    ref_np = np.stack([o[:, 0].permute(1, 2, 0).numpy() for o in ref_out], axis=0)  # [B, H, W, C]
    max_diff = float(np.max(np.abs(out - ref_np)))
    print(f"parity cap_lens={cap_lens} cap2_lens={cap2_lens} scan={scan_layers}: max abs diff {max_diff:.3e}")
    np.testing.assert_allclose(out, ref_np, atol=1e-4, rtol=1e-4)
    return max_diff

  def test_parity_with_cap_feats_2(self):
    # Sample 1's caption (20 + 15 = 35) pads to 64, sample 0's (5 + 3) to 32: different image position offsets.
    self._run_parity([5, 20], [3, 15])

  def test_parity_without_cap_feats_2(self):
    self._run_parity([7, 33], [], height=12, width=8)

  def test_parity_scanned(self):
    self._run_parity([5, 20], [3, 15], scan_layers=True)


if __name__ == "__main__":
  unittest.main()
