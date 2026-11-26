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

"""Unit tests for AdamW weight-decay mask and Muon dimension numbers across m3 transforms (`M4E-12`)."""

from types import SimpleNamespace
from typing import Any
from absl.testing import absltest, parameterized
from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.m3.core.sharding import mesh_from_config
from maxtext.m3.infra import offload as m3_offload
from maxtext.m3.infra import remat as m3_remat
from maxtext.m3.infra import scan as m3_scan
from maxtext.optimizers import optimizers
from maxtext.utils import maxtext_muon_utils
from maxtext.utils import model_creation_utils
from maxtext.utils import muon_utils
from tests.utils.test_helpers import get_test_config_path

_DEFAULT_ADAMW_MASK = ["bias", "scale", "embedding", "embed_tokens"]


def _make_qwen3_cfg(output_dir: str, **overrides: Any):
  """Builds a small 4-CPU m3 Qwen3 config for optimizer mask tests."""
  argv = [
      "",
      get_test_config_path("base.yml"),
      f"base_output_directory={output_dir}",
      "run_name=m3_optimizer_mask_test",
      "model_name=qwen3-0.6b",
      "use_m3_model=true",
      "override_model_config=true",
      "base_emb_dim=64",
      "base_mlp_dim=128",
      "base_num_decoder_layers=2",
      "base_num_query_heads=4",
      "base_num_kv_heads=2",
      "head_dim=16",
      "vocab_size=256",
      "ici_fsdp_parallelism=2",
      "ici_tensor_parallelism=2",
      "dtype=float32",
      "weight_dtype=float32",
      f"adamw_mask={_DEFAULT_ADAMW_MASK!r}",
      "enable_checkpointing=false",
      "use_weight_converter=false",
      "skip_jax_distributed_system=true",
  ]
  for key, value in overrides.items():
    argv.append(f"{key}={value}")
  return pyconfig.initialize(argv)


def _normalize_layer_path(path: tuple[Any, ...]) -> tuple[Any, ...]:
  """Collapses unscanned `('decoder', 'layers', 'layers', i, ...)` to `('decoder', 'layers', ...)`."""
  if path[:3] == ("decoder", "layers", "layers") and isinstance(path[3], int):
    return path[:2] + path[4:]
  if path[:2] == ("layers", "layers") and isinstance(path[2], int):
    return ("layers",) + path[3:]
  return path


def _unwrap_var(node: Any) -> Any:
  return node.get_value() if isinstance(node, nnx.Variable) else node


def _extract_optimizer_resolution(
    model: nnx.Module, cfg: Any, mesh: jax.sharding.Mesh
) -> dict[tuple[Any, ...], dict[str, Any]]:
  """Extracts per-parameter AdamW mask, Muon feature-axis sizes, mdn, and physical sharding spec."""
  params = nnx.state(model, nnx.Param)
  mask_fn = optimizers.get_adamw_mask(cfg)
  assert mask_fn is not None
  mask_tree = mask_fn(params)
  mdn_tree = muon_utils.get_muon_weight_dimension_numbers(model, cfg)
  smdn_tree = maxtext_muon_utils.get_maxtext_muon_weight_dimension_numbers(model, cfg, mesh=mesh)

  summary: dict[tuple[Any, ...], dict[str, Any]] = {}
  for (path, var), (_, mask_node), (_, mdn_node), (_, smdn_node) in zip(
      nnx.to_flat_state(params),
      nnx.to_flat_state(mask_tree),
      nnx.to_flat_state(mdn_tree),
      nnx.to_flat_state(smdn_tree),
  ):
    norm_path = _normalize_layer_path(path)
    shape = var.get_raw_value().shape
    mask_val = bool(_unwrap_var(mask_node))
    mdn_val = _unwrap_var(mdn_node)
    smdn_val = _unwrap_var(smdn_node)
    if mdn_val is None:
      red_sizes = None
      out_sizes = None
    else:
      red_sizes = tuple(shape[axis] for axis in mdn_val.reduction_axis)
      out_sizes = tuple(shape[axis] for axis in mdn_val.output_axis)
    entry = {
        "mask": mask_val,
        "red_sizes": red_sizes,
        "out_sizes": out_sizes,
        "mdn": mdn_val,
        "smdn_spec": smdn_val.sharding.spec if smdn_val is not None else None,
        "ndim": len(shape),
    }
    if norm_path in summary:
      # Unscanned layers 0..N-1 must agree identically with layer 0.
      assert summary[norm_path] == entry, f"Layer disagreement at {path}: {summary[norm_path]} vs {entry}"
    summary[norm_path] = entry
  return summary


class _BiasedBlockLayer(nnx.Module):
  """Toy decoder block with 2D kernel, 1D bias, and 1D RMSNorm scale for bias/scale mask testing."""

  def __init__(self, emb_dim: int, mlp_dim: int, mesh: jax.sharding.Mesh, rngs: nnx.Rngs):
    P = jax.sharding.PartitionSpec
    k_val = jax.device_put(
        rngs.params.normal((emb_dim, mlp_dim), dtype=jnp.float32),
        jax.sharding.NamedSharding(mesh, P("fsdp", "tensor")),
    )
    b_val = jax.device_put(
        jnp.zeros((mlp_dim,), dtype=jnp.float32),
        jax.sharding.NamedSharding(mesh, P("tensor")),
    )
    s_val = jax.device_put(
        jnp.ones((emb_dim,), dtype=jnp.float32),
        jax.sharding.NamedSharding(mesh, P(None)),
    )
    self.proj_kernel = nnx.Param(k_val, out_sharding=P("fsdp", "tensor"))
    self.proj_bias = nnx.Param(b_val, out_sharding=P("tensor"))
    self.norm_scale = nnx.Param(s_val, out_sharding=P(None))

  def __call__(self, x: jax.Array) -> jax.Array:
    return (x * self.norm_scale[...]) @ self.proj_kernel[...] + self.proj_bias[...]


class _BiasedToyModel(nnx.Module):
  """Toy m3-style model containing embedding, `nnx.Sequential` layers with bias/scale, and final norm."""

  def __init__(self, num_layers: int, emb_dim: int, mlp_dim: int, mesh: jax.sharding.Mesh):
    P = jax.sharding.PartitionSpec
    rngs = nnx.Rngs(0)
    emb_val = jax.device_put(
        rngs.params.normal((64, emb_dim), dtype=jnp.float32),
        jax.sharding.NamedSharding(mesh, P("tensor", "fsdp")),
    )
    self.token_embedder = nnx.Dict(embedding=nnx.Param(emb_val, out_sharding=P("tensor", "fsdp")))
    self.layers = nnx.Sequential(*[_BiasedBlockLayer(emb_dim, mlp_dim, mesh, rngs) for _ in range(num_layers)])


class OptimizerMaskTest(parameterized.TestCase):
  """Verifies `adamw_mask` and Muon dimension numbers across unscanned, scan, remat, and offload."""

  def test_qwen3_adamw_mask_and_muon_mdn_match_across_unscanned_scan0_remat_and_offload(self):
    out_dir = self.create_tempdir().full_path
    cfg_unscanned = _make_qwen3_cfg(out_dir, scan_layers="false")
    cfg_scan0 = _make_qwen3_cfg(out_dir, scan_layers="true", param_scan_axis="0")
    cfg_scan1 = _make_qwen3_cfg(out_dir, scan_layers="true", param_scan_axis="1")
    cfg_remat = _make_qwen3_cfg(out_dir, scan_layers="true", param_scan_axis="0", remat_policy="full")
    cfg_offload = _make_qwen3_cfg(
        out_dir,
        scan_layers="true",
        param_scan_axis="0",
        remat_policy="full",
        parameter_memory_host_offload="true",
    )
    mesh = mesh_from_config(cfg_unscanned)
    with jax.set_mesh(mesh):
      m_unscanned = model_creation_utils.create_model(cfg_unscanned, mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))
      m_scan0 = model_creation_utils.create_model(cfg_scan0, mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))
      m_scan1 = model_creation_utils.create_model(cfg_scan1, mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))
      m_remat = model_creation_utils.create_model(cfg_remat, mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))
      m_offload = model_creation_utils.create_model(cfg_offload, mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))

    res_unscanned = _extract_optimizer_resolution(m_unscanned, cfg_unscanned, mesh)
    res_scan0 = _extract_optimizer_resolution(m_scan0, cfg_scan0, mesh)
    res_scan1 = _extract_optimizer_resolution(m_scan1, cfg_scan1, mesh)
    res_remat = _extract_optimizer_resolution(m_remat, cfg_remat, mesh)
    res_offload = _extract_optimizer_resolution(m_offload, cfg_offload, mesh)

    self.assertEqual(set(res_unscanned.keys()), set(res_scan0.keys()))
    self.assertEqual(res_scan0, res_remat)
    self.assertEqual(res_scan0, res_offload)

    for path, u_info in res_unscanned.items():
      s0_info = res_scan0[path]
      s1_info = res_scan1[path]
      self.assertEqual(u_info["mask"], s0_info["mask"], f"AdamW mask mismatch at {path}")
      self.assertEqual(u_info["mask"], s1_info["mask"], f"AdamW mask mismatch at {path}")
      self.assertEqual(u_info["red_sizes"], s0_info["red_sizes"], f"Muon reduction sizes mismatch at {path}")
      self.assertEqual(u_info["out_sizes"], s0_info["out_sizes"], f"Muon output sizes mismatch at {path}")
      self.assertEqual(u_info["red_sizes"], s1_info["red_sizes"], f"Muon reduction sizes mismatch at {path}")
      self.assertEqual(u_info["out_sizes"], s1_info["out_sizes"], f"Muon output sizes mismatch at {path}")

      # Norm scales (1D unscanned, 2D stacked under ScannedLayers) and embeddings must be excluded.
      if "scale" in path or "embedding" in path:
        self.assertFalse(s0_info["mask"], f"Expected weight decay excluded for {path}")
        self.assertIsNone(s0_info["mdn"], f"Expected Muon excluded for {path}")
      else:
        self.assertTrue(s0_info["mask"], f"Expected weight decay enabled for {path}")
        self.assertIsNotNone(s0_info["mdn"], f"Expected Muon enabled for {path}")
        # Under param_scan_axis=0, axis 0 is the stacked layer dimension and must NEVER be reduced over.
        self.assertNotIn(0, s0_info["mdn"].reduction_axis, f"Layer axis 0 reduced at {path}")
        self.assertEqual(s0_info["mdn"].reduction_axis[0], 1, f"Expected feature axis 1 at {path}")

  def test_biases_stacked_scales_and_embeddings_excluded_across_transforms(self):
    out_dir = self.create_tempdir().full_path
    cfg = _make_qwen3_cfg(out_dir)
    mesh = mesh_from_config(cfg)
    with jax.set_mesh(mesh):
      unscanned = _BiasedToyModel(num_layers=3, emb_dim=16, mlp_dim=32, mesh=mesh)
      scanned0 = _BiasedToyModel(num_layers=3, emb_dim=16, mlp_dim=32, mesh=mesh)
      m3_scan.apply_scan(scanned0, param_scan_axis=0)
      remat_offload = _BiasedToyModel(num_layers=3, emb_dim=16, mlp_dim=32, mesh=mesh)
      m3_scan.apply_scan(remat_offload, param_scan_axis=0)
      remat_offload = m3_remat.maybe_apply_remat(
          remat_offload, SimpleNamespace(remat_policy="full"), model_mode=MODEL_MODE_TRAIN
      )
      remat_offload = m3_offload.maybe_offload_parameters(
          remat_offload, SimpleNamespace(parameter_memory_host_offload=True)
      )

    res_u = _extract_optimizer_resolution(unscanned, cfg, mesh)
    res_s0 = _extract_optimizer_resolution(scanned0, cfg, mesh)
    res_ro = _extract_optimizer_resolution(remat_offload, cfg, mesh)

    self.assertEqual(res_s0, res_ro)
    for path, u_entry in res_u.items():
      self.assertEqual(u_entry["mask"], res_s0[path]["mask"])
      self.assertEqual(u_entry["red_sizes"], res_s0[path]["red_sizes"])
      self.assertEqual(u_entry["out_sizes"], res_s0[path]["out_sizes"])

    # Verify 2D stacked bias (3, 32), 2D stacked norm scale (3, 16), and 2D embedding (64, 16)
    # are all excluded from both AdamW weight decay and Muon orthogonalization.
    self.assertFalse(res_s0[("layers", "proj_bias")]["mask"])
    self.assertIsNone(res_s0[("layers", "proj_bias")]["mdn"])
    self.assertEqual(res_s0[("layers", "proj_bias")]["ndim"], 2)

    self.assertFalse(res_s0[("layers", "norm_scale")]["mask"])
    self.assertIsNone(res_s0[("layers", "norm_scale")]["mdn"])
    self.assertEqual(res_s0[("layers", "norm_scale")]["ndim"], 2)

    self.assertFalse(res_s0[("token_embedder", "embedding")]["mask"])
    self.assertIsNone(res_s0[("token_embedder", "embedding")]["mdn"])

    # Verify 3D stacked kernel (3, 16, 32) is included and reduces over axis 1 (size 16), not axis 0 (size 3).
    self.assertTrue(res_s0[("layers", "proj_kernel")]["mask"])
    self.assertEqual(res_s0[("layers", "proj_kernel")]["red_sizes"], (16,))
    self.assertEqual(res_s0[("layers", "proj_kernel")]["out_sizes"], (32,))

  def test_negative_control_missing_scan_metadata_or_incomplete_mask_is_caught(self):
    out_dir = self.create_tempdir().full_path
    cfg = _make_qwen3_cfg(out_dir, scan_layers="true", param_scan_axis="0")
    mesh = mesh_from_config(cfg)
    with jax.set_mesh(mesh):
      model = model_creation_utils.create_model(cfg, mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))

    # Negative control 1: stripping `param_scan_axis` metadata from a `param_scan_axis=0` stacked
    # weight causes Muon to reduce over the layer axis (axis 0, size 2) instead of feature axis 1 (size 64).
    wi0_var = model.decoder.layers.mlp.wi_0.kernel
    stripped_var = nnx.Param(wi0_var.get_raw_value(), out_sharding=wi0_var.get_metadata()["out_sharding"])
    self.assertEqual(muon_utils.get_scan_axes(stripped_var), ())
    bad_mdn = muon_utils.to_stacked_layout(
        muon_utils.transform_logic(("decoder", "layers", "mlp", "wi_0", "kernel"), shape=stripped_var.shape),
        stripped_var.ndim,
        muon_utils.get_scan_axes(stripped_var),
    )
    assert bad_mdn is not None
    self.assertEqual(bad_mdn.reduction_axis, (0,))
    self.assertEqual(stripped_var.shape[bad_mdn.reduction_axis[0]], 2)
    self.assertNotEqual(stripped_var.shape[bad_mdn.reduction_axis[0]], 64)

    # Negative control 2: omitting "scale" from `adamw_mask` allows 2D stacked norm scales to decay.
    bad_cfg = _make_qwen3_cfg(
        out_dir,
        scan_layers="true",
        param_scan_axis="0",
        adamw_mask="['bias','embedding','embed_tokens']",
    )
    bad_res = _extract_optimizer_resolution(model, bad_cfg, mesh)
    self.assertTrue(bad_res[("decoder", "layers", "pre_self_attention_layer_norm", "scale")]["mask"])

  def test_m3_moe_cycles_norms_and_vision_muon_dimension_numbers(self):
    """Verifies Muon axes across m3 MoE experts (3D/4D/5D), pre-stacked cycles/layers, 1D norms, and ViT out/pos_embed."""
    mdn = muon_utils.mdn

    # 1. Routed MoE experts: 4D/5D gate_up_proj (..., E, D, 2, M) -> (-3,)->(-1,); down_proj (..., E, M, D) -> (-2,)->(-1,).
    gu_4d = nnx.Param(jnp.zeros((8, 64, 2, 32), dtype=jnp.float32))
    gu_4d_path = ("decoder", "layers", "0", "mlp", "experts", "gate_up_proj")
    self.assertEqual(
        muon_utils.to_stacked_layout(
            muon_utils.transform_logic(gu_4d_path, shape=gu_4d.shape),
            gu_4d.ndim,
            muon_utils.get_scan_axes(gu_4d, gu_4d_path),
        ),
        mdn((-3,), (-1,)),
    )

    gu_5d = nnx.Param(jnp.zeros((3, 8, 64, 2, 32), dtype=jnp.float32))
    gu_5d_path = ("decoder", "cycles", "gdn_0", "mlp", "experts", "gate_up_proj")
    self.assertEqual(muon_utils.get_scan_axes(gu_5d, gu_5d_path), (0,))
    self.assertEqual(
        muon_utils.to_stacked_layout(
            muon_utils.transform_logic(gu_5d_path, shape=gu_5d.shape),
            gu_5d.ndim,
            muon_utils.get_scan_axes(gu_5d, gu_5d_path),
        ),
        mdn((-3,), (-1,)),
    )

    dn_4d = nnx.Param(jnp.zeros((3, 8, 32, 64), dtype=jnp.float32))
    dn_4d_path = ("decoder", "cycles", "gdn_0", "mlp", "experts", "down_proj")
    self.assertEqual(
        muon_utils.to_stacked_layout(
            muon_utils.transform_logic(dn_4d_path, shape=dn_4d.shape),
            dn_4d.ndim,
            muon_utils.get_scan_axes(dn_4d, dn_4d_path),
        ),
        mdn((-2,), (-1,)),
    )

    # 2. Pre-stacked GraniteMoE layers & stacked 1D norms/vectors -> None via get_muon_weight_dimension_numbers.
    granite_q = nnx.Param(jnp.zeros((4, 64, 64), dtype=jnp.float32))
    granite_q_path = ("layers", "self_attn", "q_proj", "kernel")
    self.assertEqual(muon_utils.get_scan_axes(granite_q, granite_q_path), (0,))
    self.assertEqual(
        muon_utils.to_stacked_layout(
            muon_utils.transform_logic(granite_q_path, shape=granite_q.shape),
            granite_q.ndim,
            muon_utils.get_scan_axes(granite_q, granite_q_path),
        ),
        mdn((1,), (-1,)),
    )

    class _StackedVecToy(nnx.Module):
      """Minimal module with stacked 1D norm and custom stacked 1D vector."""

      def __init__(self):
        self.layers = nnx.Dict(input_layernorm=nnx.Dict(weight=nnx.Param(jnp.ones((4, 64), dtype=jnp.float32))))
        self.decoder = nnx.Dict(
            cycles=nnx.Dict(gdn_0=nnx.Dict(alpha_vec=nnx.Param(jnp.ones((4, 64), dtype=jnp.float32))))
        )

    vec_dims = nnx.to_pure_dict(muon_utils.get_muon_weight_dimension_numbers(_StackedVecToy()))
    self.assertIsNone(vec_dims["layers"]["input_layernorm"]["weight"])
    self.assertIsNone(vec_dims["decoder"]["cycles"]["gdn_0"]["alpha_vec"])

    # 3. Multimodal ViT pos_embed / patch_embed / _ape excluded, and 3D ViT attn.out reduces (0, -2) -> (-1,).
    for excluded_path, shp in (
        (("vision", "pos_embed_interpolate", "pos_embed"), (2304, 1024)),
        (("vision", "patch_embed", "proj", "kernel"), (2, 16, 16, 3, 1024)),
        (("super_block", "slot_3", "self_attn", "indexer", "index_kpool_compress_ape"), (2, 4, 128)),
    ):
      self.assertIsNone(muon_utils.transform_logic(excluded_path, shape=shp))

    vit_out = nnx.Param(jnp.zeros((16, 64, 1024), dtype=jnp.float32))
    vit_out_path = ("vision_encoder", "blocks", "0", "attn", "out", "kernel")
    self.assertEqual(
        muon_utils.to_stacked_layout(
            muon_utils.transform_logic(vit_out_path, shape=vit_out.shape),
            vit_out.ndim,
            muon_utils.get_scan_axes(vit_out, vit_out_path),
        ),
        mdn((0, -2), (-1,)),
    )

    # 4. Router weights under `mlp.gate.weight` (glm5.3-flash, qwen3.8-flash-next) honor `include_routers=False`.
    for router_path in (
        ("decoder", "cycles", "moe_0", "mlp", "gate", "weight"),
        ("super_block", "slot_3", "mlp", "gate", "weight"),
    ):
      self.assertIsNone(muon_utils.transform_logic(router_path, shape=(128, 16), include_routers=False))
      self.assertEqual(muon_utils.transform_logic(router_path, shape=(128, 16), include_routers=True), mdn((0,), (-1,)))


if __name__ == "__main__":
  absltest.main()
