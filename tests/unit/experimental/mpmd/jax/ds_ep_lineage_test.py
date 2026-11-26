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

"""ds_ep_lineage: ds_ep's batch-split chunks on Lineage's layers (MaxText src/maxtext/experimental/lineage).

    export XLA_FLAGS=--xla_force_host_platform_device_count=4                    # the CPU parts' 4 CPU devices
    JAX_PLATFORMS=cpu python3 -m pytest tests/unit/experimental/mpmd/jax/ds_ep_lineage_test.py -q    # CPU parts
    MAXTEXT_MPMD_REQUIRE_LINEAGE=1 JAX_PLATFORMS=cpu \\
        python3 -m pytest tests/unit/experimental/mpmd/jax/ds_ep_lineage_test.py -q
    JAX_PLATFORMS=tpu MAXTEXT_MPMD_REQUIRE_TPU=1 MAXTEXT_MPMD_REQUIRE_LINEAGE=1 \\
        python3 -m pytest tests/unit/experimental/mpmd/jax/ds_ep_lineage_test.py -k OnTPU -s

Config / mapping / view checks, the Region's constructor guards and the Region cache are the package's own code and
run everywhere.  The weight conversion, the SparseCore switch and the typed conversion pipeline (on a CPU mesh) need Lineage
importable -- its modules import typeguard and jaxtyping, which MaxText's TPU requirements do not bring -- and skip
without it, giving import_lineage's error.
MAXTEXT_MPMD_REQUIRE_LINEAGE=1 turns a Lineage that does not import into a failure instead -- this module, and
ds_ep_lineage_pipeline_test with it, fail to import, naming why -- for runs meant to have Lineage (the last two commands
above, and ds_ep_lineage_pipeline_test's), where a Lineage dependency left out would otherwise pass as skips.
The TPU command puts jax on the chips with JAX_PLATFORMS=tpu, and MAXTEXT_MPMD_REQUIRE_TPU=1 turns OnTPU's skip off a
TPU7x of 8 devices into a failure, so a run left on another backend cannot pass as skips either (testing.require_tpu).

Lineage's kernels have no CPU path (Pallas TPU, SparseCore), so everything that runs a layer is `OnTPU`: a TPU7x with
8 devices (a tpu7x-8, 2x2x1 chips: ds_ep's stage meshes devices 0-3 and 4-7), skipped elsewhere -- MaxText's TPU CI
(a v6e-4, Lineage's typeguard and jaxtyping not installed) skips it.  OnTPU checks the region
against Lineage's own public entry points (dsv3_sparse_layer, dsv3_sparse_layers with its real hooks), against ds_ep's
own unsplit layer math at top-k routing (what catches a conversion bug), at 1-5 MoE layers and with dense layers, on
the stage without SparseCore offload against the one with it, and ds_ep's F program on the forward-only primal against
the full-bank forward.  On CPU (with Lineage) the routing rule is checked exactly against ds_ep's, and the YaRN tables
and softmax scale Lineage gets against ds_ep's own; the region is traced (not run) for its types, traces, activation
banks and residuals, and called on inputs it refuses.
"""

import collections
import contextlib
import dataclasses
import functools
import gc
import math
import os
import re
import subprocess
import sys
import threading
import types
import unittest
import weakref
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P
import pytest

from maxtext.experimental.mpmd.jax.deepseek.config import DeepSeekConfig
from maxtext.experimental.mpmd.jax import ds_ep
from maxtext.experimental.mpmd.jax import ds_ep_lineage as dl
from maxtext.experimental.mpmd.jax import ds_ep_pipeline as dsp
from maxtext.experimental.mpmd.jax import testing

# Why Lineage does not import here (import_lineage's ImportError, naming the maxtext in use and what is wrong: it has
# no Lineage, a module its Lineage imports is missing, or its Lineage lacks what the adapter calls), else None.
NO_LINEAGE = testing.lineage_missing()
HAVE_LINEAGE = NO_LINEAGE is None
if testing.env_on("MAXTEXT_MPMD_REQUIRE_LINEAGE") and not HAVE_LINEAGE:  # a run meant to have it (module doc)
  raise ImportError(f"MAXTEXT_MPMD_REQUIRE_LINEAGE=1, but Lineage does not import: {NO_LINEAGE}")
needs_lineage = testing.requires_lineage


# Why a check of the stage without SparseCore offload is McJAX's (testing.skip_on_pathways).
SC_STAGE0 = (
    "stage 0 (devices 0-3) holds chip (0,0,0), where a TPU VM's own libtpu offloads no collective to SparseCore; "
    "Pathways' servers offload its collectives too"
)


def _on_tpu7x():
  """A TPU7x of 8 devices (a tpu7x-8, 2x2x1 chips), the TPU the TPU tests are written for -- whose stage meshes they
  hard-code (stage 1, devices 4-7, offloads collectives to SparseCore; stage 0, which holds chip (0,0,0), does not).
  Asked when a TPU class sets up, not at import: it starts jax's backend."""
  devs = testing.tpu_devices()
  return len(devs) == 8 and devs[0].device_kind == "TPU7x"


def _cpu_devices(n=4):
  try:
    devs = jax.devices("cpu")
  except RuntimeError:
    return None
  return devs[:n] if len(devs) >= n else None


def small_cfg(**kw):
  """A CPU-sized DeepSeek layer Lineage accepts (D / 128 even, groups on with k / topk_routing_group == 2)."""
  base = dict(
      vocab_size=256,
      emb_dim=256,
      num_decoder_layers=3,
      first_num_dense_layers=0,
      num_query_heads=2,
      num_kv_heads=2,
      q_lora_rank=128,
      kv_lora_rank=128,
      qk_nope_head_dim=128,
      qk_rope_head_dim=64,
      v_head_dim=128,
      mlp_dim=256,
      moe_mlp_dim=256,
      num_experts=8,
      num_experts_per_tok=4,
      n_routing_groups=4,
      topk_routing_group=2,
      routed_bias=False,
  )
  base.update(kw)
  return DeepSeekConfig(**base)


def stage_cfg(cfg, **kw):
  base = dict(
      model=cfg,
      fsdp=2,
      ep=2,
      shard_mode="explicit",
      random_routing=False,
      batch_split=True,
      fsdp_collectives="xla",
      ep_collectives="xla",
      gmm_impl="ragged_dot",
      use_ragged_sort=False,
  )
  base.update(kw)
  return ds_ep.StageConfig(**base)


def rand_tree(shapes, seed, scale=0.02, scales=None):
  """Host bf16-representable fp32 arrays for a ds_ep shapes tree: N(0, scale^2); norm scales 1 + N(0, 0.1^2), so
  a swapped norm shows; `scales` {leaf name: scale} overrides by the leaf's last path key."""
  rng = np.random.default_rng(seed)

  def one(path, s):
    keys = [getattr(k, "key", None) for k in path]
    name = keys[-2] if keys[-1] in ("kernel", "scale") else keys[-1]
    sc = (scales or {}).get(name, scale)
    a = (1 + 0.1 * rng.standard_normal(s)) if keys[-1] == "scale" else sc * rng.standard_normal(s)
    return np.asarray(jnp.asarray(a, jnp.bfloat16).astype(jnp.float32))

  return jax.tree_util.tree_map_with_path(one, shapes, is_leaf=ds_ep._is_shape)


def chunk_spec(n=0, nd=0, bias=False):
  return ds_ep.ChunkSpec(index=0, layers=tuple(range(n)), first=False, last=False, stage=0, dense=nd, routed_bias=bias)


def stacked_shapes(cfg, n, bias=False, dense=False):
  """ds_ep's shapes of n stacked MoE (or dense) layers (ds_ep.chunk_shapes)."""
  ch = chunk_spec(0, n, bias) if dense else chunk_spec(n, 0, bias)
  return ds_ep.chunk_shapes(stage_cfg(cfg), ch)["dense_layers" if dense else "layers"]


def stacked(cfg, n, bias=False, dense=False, seed=0, scales=None):
  """n stacked MoE (or dense) layers of ds_ep's compute weights: rand_tree's values in ds_ep's compute dtypes
  (ds_ep.leaf_dtype_tree: bf16, but the router kernel and bias in fp32 under float32_gate_logits)."""
  ch = chunk_spec(0, n, bias) if dense else chunk_spec(n, 0, bias)
  dts = ds_ep.leaf_dtype_tree(stage_cfg(cfg), ch, "compute")["dense_layers" if dense else "layers"]
  return jax.tree.map(
      lambda a, dt: jnp.asarray(a, dt), rand_tree(stacked_shapes(cfg, n, bias, dense), seed, scales=scales), dts
  )


def leaves_with_names(tree):
  return [(jax.tree_util.keystr(p), np.asarray(x)) for p, x in jax.tree_util.tree_flatten_with_path(tree)[0]]


# ------------------------------------------------------------- the package's own code: config, mapping, view
class Config(unittest.TestCase):

  def dsv3(self, **kw):
    cfg = DeepSeekConfig(
        emb_dim=7168,
        num_query_heads=128,
        num_kv_heads=128,
        q_lora_rank=1536,
        kv_lora_rank=512,
        moe_mlp_dim=2048,
        num_experts=256,
        num_experts_per_tok=8,
        n_routing_groups=8,
        topk_routing_group=4,
    )
    return cfg.replace(**kw)

  def test_accepts_deepseek_v3(self):
    """DeepSeek-V3, and what only picks among ds_ep's own kernels (IGNORED_MODEL_FIELDS: Lineage's splash is
    base-2 whatever ds_ep's would be) or a k / topk_routing_group other than 2 (`routing`)."""
    cfg = self.dsv3()
    dl.check(cfg, stage_cfg(cfg, fsdp=16, ep=8))
    dl.check(cfg.replace(float32_gate_logits=True, n_routing_groups=-1, topk_routing_group=-1))
    dl.check(cfg.replace(topk_routing_group=2))
    dl.check(cfg.replace(sa_use_base2_exp=False, attention_kernel="splash"))
    self.assertIn("sa_use_base2_exp", dl.IGNORED_MODEL_FIELDS)

  def test_refusals(self):
    cfg = self.dsv3()
    for change, want in (
        (dict(dtype=jnp.float32), "bf16 activations"),
        (dict(routed_score_func="softmax"), "sigmoid"),
        (dict(n_routing_groups=-1), "set both"),
        (dict(n_routing_groups=7), "multiple of n_routing_groups"),
        (dict(n_routing_groups=256), "1 expert(s) a routing group"),
        (dict(topk_routing_group=9), "topk_routing_group 9 > n_routing_groups 8"),
        (dict(n_routing_groups=64, topk_routing_group=1), "num_experts_per_tok 8 > the 1 x 4 experts"),
        (dict(shared_experts=2), "shared_experts 2"),
        (dict(num_kv_heads=64), "num_kv_heads 64"),
        (dict(emb_dim=7168 + 128), "multiple of 256"),
        (dict(moe_mlp_dim=2000), "moe_mlp_dim 2000"),
        (dict(q_lora_rank=0), "query LoRA"),
        (dict(mlp_activations=("gelu", "linear")), "silu x linear"),
        (dict(rope_interleave=False), "interleaved"),
        (dict(rope_truncate=False), "truncates"),
        (dict(rope_attention_scaling=True), "never scales"),
        (dict(rope_factor=40.5), "rope_factor 40.5"),
        (dict(beta_fast=32.5), "beta_fast"),
        (dict(routed_bias_update_rate=1e-3), "expert counts"),
        (dict(load_balance_loss_weight=1e-2), "aux loss"),
        (dict(mtp_num_layers=1), "MTP"),
    ):
      with self.subTest(change=change):
        with self.assertRaisesRegex(ValueError, want.replace("(", r"\(").replace(")", r"\)")):
          dl.check(cfg.replace(**change))
    # The stage's settings, on a stage config that is not batch-split -- one that is refuses them itself
    # (StageConfig._check_lineage) -- as a Region built from any stage config checks them (Region.__init__).
    for change, want in (
        (dict(random_routing=True), "random routing"),
        (dict(ragged_buffer_factor=2.0, use_ragged_sort=True), "dropless"),
        (dict(shard_mode="auto"), "want 'explicit'"),
        (dict(ep=3), "multiple of ep 3"),
    ):
      with self.subTest(change=change):
        with self.assertRaisesRegex(ValueError, "^Lineage's batch-split layers cannot run this: .*" + want):
          dl.check(cfg, stage_cfg(cfg, batch_split=False, **change))
    with self.assertRaisesRegex(ValueError, "cp 2"):
      dl.check(cfg, dataclasses.replace(stage_cfg(cfg, batch_split=False), cp=2))

  def test_reports_every_problem(self):
    with self.assertRaises(ValueError) as e:
      dl.check(self.dsv3(routed_score_func="softmax", shared_experts=2, n_routing_groups=7, topk_routing_group=9))
    for want in (
        "sigmoid",
        "shared_experts 2",
        "not a multiple of n_routing_groups 7",
        "topk_routing_group 9 > n_routing_groups 7",
    ):
      self.assertIn(want, str(e.exception))

  def test_region_guards(self):
    """A Region refuses no MoE layer, a capacity factor <= 0 and a mesh that is not the stage config's before it
    imports Lineage (so here without it); the model and stage checks first (`check`)."""
    devs = _cpu_devices()
    if devs is None:
      self.skipTest("needs 4 CPU devices")
    m2 = Mesh(np.array(devs, dtype=object).reshape(2, 2), ds_ep.FE, axis_types=(AxisType.Explicit,) * 2)
    scfg = stage_cfg(small_cfg())
    with self.assertRaisesRegex(ValueError, r"at least one MoE layer \(n_moe 0\)"):
      dl.Region(scfg, m2, n_moe=0)
    for cf in (0.0, -1.0, float("nan")):
      with self.assertRaisesRegex(ValueError, "capacity_factor must be > 0"):
        dl.Region(scfg, m2, n_moe=1, capacity_factor=cf)
    with self.assertRaisesRegex(ValueError, r"mesh \{'fsdp': 4, 'expert': 1\} is not fsdp 2 x ep 2"):
      dl.Region(scfg, Mesh(m2.devices.reshape(4, 1), ds_ep.FE, axis_types=(AxisType.Explicit,) * 2), n_moe=1)
    with self.assertRaisesRegex(ValueError, "random routing"):
      dl.Region(stage_cfg(small_cfg(), batch_split=False, random_routing=True), m2, n_moe=1)

  def test_capacity_factor_reaches_lineage(self):
    """The capacity factor (ds_ep_pipeline --lineage-capacity-factor) is what Lineage's layer functions get
    (layer_kwargs' capacity_factor, which the router's chunk capacity reads), not its default; here with Lineage's
    modules stubbed (RegionInputs.test_capacity_factor builds a Region with it, RegionTrace.test_capacity_factor_bank
    checks the activation bank is sized with it)."""
    devs = _cpu_devices()
    if devs is None:
      self.skipTest("needs 4 CPU devices")
    m3 = dl.a3_mesh(Mesh(np.array(devs, dtype=object).reshape(2, 2), ds_ep.FE, axis_types=(AxisType.Explicit,) * 2))
    stub = types.SimpleNamespace(
        ops=types.SimpleNamespace(rms_norm=lambda *a, **kw: None),
        dsv3_mla=types.SimpleNamespace(yarn=None),
        expert_mlp=types.SimpleNamespace(make_gmm_fn=lambda **kw: None),
    )
    with mock.patch.object(dl, "import_lineage", lambda: stub):
      for cf in (1.0, 3.0):
        self.assertEqual(dl.layer_kwargs(small_cfg(), m3, cf)["capacity_factor"], cf)
      self.assertEqual(dl.layer_kwargs(small_cfg(), m3)["capacity_factor"], dl.CAPACITY_FACTOR)

  def test_routing(self):
    """Groups on: a group scored by its two best scores whatever k / topk_routing_group (Routing checks the
    selection itself); groups off: one group, plain top-k."""
    self.assertEqual(dl.routing(self.dsv3()), (8, 4, 2))
    self.assertEqual(dl.routing(self.dsv3(topk_routing_group=2)), (8, 2, 2))
    self.assertEqual(dl.routing(self.dsv3(n_routing_groups=-1, topk_routing_group=-1)), (1, 1, 8))

  def test_sa_blocks(self):
    cfg = self.dsv3(sa_block_q=512, sa_block_kv=1024, sa_block_kv_compute=256)
    self.assertEqual(dl.sa_blocks(cfg), (512, 1024, 256, 512, 1024, 256))
    self.assertEqual(
        dl.sa_blocks(cfg.replace(sa_block_q_dkv=2048, sa_block_kv_dkv=128, sa_block_kv_dkv_compute=64)),
        (512, 1024, 256, 2048, 128, 64),
    )

  def test_axis_mapping_rule(self):
    """A3 meets Lineage's typing rule: token axes (fsdp_attention + attention) == expert + fsdp_moe, expert and
    fsdp_moe disjoint; 'attention' a real axis; the only logical name used as a physical one is identity-mapped."""
    am = dl.AXIS_MAPPING
    self.assertEqual(set(am), {"dcn", "attention", "fsdp_attention", "expert", "fsdp_moe"})
    ax = lambda *names: set().union(*(dl._physical((n,)) for n in names))
    self.assertEqual(ax("fsdp_attention", "attention"), ax("expert", "fsdp_moe"))
    self.assertEqual(ax("fsdp_attention", "attention"), {"fsdp", "expert", "attn"})
    self.assertFalse(ax("expert") & ax("fsdp_moe"))
    self.assertEqual(ax("attention"), {"attn"})
    self.assertEqual(ax("dcn"), set())
    for logical in am:
      if logical in {"fsdp", "expert", "attn"}:
        self.assertEqual(am[logical], logical)
    self.assertEqual(dl.ACT2, P(ds_ep.FE, None, None))
    self.assertEqual(dl.ACT3.partitions[0], dl.ACT2.partitions[0])  # the same batch blocks on m3

  def test_a3_mesh(self):
    devs = _cpu_devices()
    if devs is None:
      self.skipTest("needs 4 CPU devices")
    arr = np.array(devs, dtype=object).reshape(2, 2)
    m2 = Mesh(arr, ds_ep.FE, axis_types=(AxisType.Explicit,) * 2)
    m3 = dl.a3_mesh(m2)
    self.assertEqual(m3.axis_names, ("fsdp", "expert", "attn"))
    self.assertEqual(dict(m3.shape), {"fsdp": 2, "expert": 2, "attn": 1})
    self.assertTrue(all(t == AxisType.Explicit for t in m3.axis_types))
    self.assertEqual(m3.devices.reshape(2, 2).tolist(), arr.tolist())  # same devices, same order
    with self.assertRaisesRegex(ValueError, "Explicit"):
      dl.a3_mesh(Mesh(arr, ds_ep.FE))
    with self.assertRaisesRegex(ValueError, "Explicit"):
      dl.a3_mesh(Mesh(arr, ("data", "model"), axis_types=(AxisType.Explicit,) * 2))
    prev = jax.config.jax_remove_size_one_mesh_axis_from_type
    try:
      jax.config.update("jax_remove_size_one_mesh_axis_from_type", True)
      with self.assertRaisesRegex(ValueError, "remove_size_one_mesh_axis_from_type False"):
        dl.a3_mesh(m2)
    finally:
      jax.config.update("jax_remove_size_one_mesh_axis_from_type", prev)
    self.assertIsNone(dl.compiler_options(m2))

  def test_import_error(self):
    """Without Lineage: an ImportError naming it; available() False.  (Loading.test_missing_dependency: a module
    Lineage imports missing.)"""
    saved = dl._LIN
    blocked = {k: sys.modules.get(k) for k in list(sys.modules) if k.startswith("maxtext.experimental.lineage")}
    try:
      dl._LIN = None
      for k in blocked:
        del sys.modules[k]
      sys.modules["maxtext.experimental.lineage"] = None
      with self.assertRaisesRegex(ImportError, r"^ds_ep_lineage could not import MaxText's Lineage stack "):
        dl.import_lineage()
      self.assertFalse(dl.available())
      self.assertIsNone(dl.provenance()["lineage"])
    finally:
      sys.modules.pop("maxtext.experimental.lineage", None)
      sys.modules.update({k: v for k, v in blocked.items() if v is not None})
      dl._LIN = saved

  def test_imports_without_lineage(self):
    """The module imports (and its config checks run) in a process that cannot import Lineage, as on the run
    image."""
    code = (
        "import sys; sys.modules['maxtext.experimental.lineage'] = None\n"
        "from maxtext.experimental.mpmd.jax import ds_ep_lineage as dl\n"
        "assert not dl.available()\n"
        "from maxtext.experimental.mpmd.jax.deepseek.config import DeepSeekConfig\n"
        "dl.check(DeepSeekConfig(emb_dim=256, moe_mlp_dim=256, n_routing_groups=4, topk_routing_group=4))\n"
        "print('ok')\n"
    )
    env = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONPATH=testing.child_pythonpath(__name__))
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300)
    self.assertEqual(out.returncode, 0, out.stderr[-2000:])
    self.assertIn("ok", out.stdout)

  def test_require_lineage(self):
    """MAXTEXT_MPMD_REQUIRE_LINEAGE=1 in a process that cannot import Lineage: this module and
    ds_ep_lineage_pipeline_test fail to import, naming the switch and import_lineage's reason, where without it they
    import and skip what needs Lineage (as this run does where Lineage is missing)."""
    env = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONPATH=testing.child_pythonpath(__name__))
    # The test modules by their full names (tests.unit.experimental.mpmd.jax.* in MaxText: the port renames each).
    for mod in (
        "tests.unit.experimental.mpmd.jax.ds_ep_lineage_test",
        "tests.unit.experimental.mpmd.jax.ds_ep_lineage_pipeline_test",
    ):
      with self.subTest(mod):
        for require in ("1", "0"):
          code = f"import sys; sys.modules['maxtext.experimental.lineage'] = None\n" f"import {mod}\nprint('imported')\n"
          out = subprocess.run(
              [sys.executable, "-c", code],
              env=dict(env, MAXTEXT_MPMD_REQUIRE_LINEAGE=require),
              capture_output=True,
              text=True,
              timeout=300,
          )
          if require == "1":
            self.assertNotEqual(out.returncode, 0, out.stdout)
            self.assertRegex(
                out.stderr,
                r"ImportError: MAXTEXT_MPMD_REQUIRE_LINEAGE=1, but Lineage does not import: "
                r"ds_ep_lineage could not import MaxText's Lineage stack",
            )
          else:
            self.assertEqual(out.returncode, 0, out.stderr[-2000:])
            self.assertIn("imported", out.stdout)

  def test_no_rope_channels(self):
    """qk_rope_head_dim 0 (MLA without rope, which ds_ep's layers and Lineage's run): the rope (de)interleave leaves a
    weight as it is, as Lineage's converter does (interop/maxtext_layout.py) -- w[..., -0:] is all of it, which would
    shuffle every channel into even / odd halves and the inverse fail; with rope channels the two are inverses."""
    w = jnp.arange(2 * 3 * 8, dtype=jnp.float32).reshape(2, 3, 8)
    for f in (dl._deinterleave_rope, dl._interleave_rope):
      np.testing.assert_array_equal(f(w, 0), w, err_msg=f.__name__)
    np.testing.assert_array_equal(dl._deinterleave_rope(w, 4)[0, 0], [0, 1, 2, 3, 4, 6, 5, 7])
    for rope in (0, 2, 4, 8):
      np.testing.assert_array_equal(dl._interleave_rope(dl._deinterleave_rope(w, rope), rope), w, err_msg=rope)


# ------------------------------------------------------------------------- the SparseCore switch (needs Lineage)
@needs_lineage
class SparseCoreSwitch(unittest.TestCase):

  def test_switch(self):
    import jax.experimental.compute_on as jco

    lin = dl.import_lineage()
    f = lambda x: x
    sc = dict(
        compute_type="tpu_sparsecore",
        out_memory_spaces=jax.memory.Space.Device,
        compiler_options={"sparse_core_config": {"core_ids": [0]}},
    )
    dev = dict(compute_type="device", out_memory_spaces=jax.memory.Space.Device)
    owners = (lin.dsv3_router, lin.dsv3_mla, jco)
    for owner in owners:
      self.assertTrue(hasattr(owner.compute_on, "mpmd_sparsecore_switch"), owner)
      self.assertFalse(hasattr(owner.compute_on.mpmd_sparsecore_switch, "mpmd_sparsecore_switch"))  # once
      self.assertIsNot(owner.compute_on(**sc)(f), f)
    seen = {}
    with dl.no_sparsecore():
      for owner in owners:
        self.assertIs(owner.compute_on(**sc)(f), f)
        self.assertIs(owner.compute_on(f, **sc), f)
        self.assertIs(owner.compute_on(f=f, **sc), f)  # jax's signature: f=None, *, compute_type, ...
        self.assertIsNot(owner.compute_on(**dev)(f), f)  # compute_on('device') stays
      with dl.no_sparsecore(False):
        self.assertIsNot(lin.dsv3_router.compute_on(**sc)(f), f)
      self.assertIs(lin.dsv3_router.compute_on(**sc)(f), f)
      t = threading.Thread(target=lambda: seen.update(other=lin.dsv3_router.compute_on(**sc)(f) is f))
      t.start()
      t.join()
    self.assertFalse(seen["other"])  # another thread keeps SparseCore on
    self.assertIsNot(lin.dsv3_router.compute_on(**sc)(f), f)
    self.assertIsNot(jco.compute_on(f=f, **sc), f)

  def test_any_alias(self):
    """Another Lineage module's alias of jax's compute_on, under any name (a later Lineage's: a module-level one
    keeps jax's own, which would leave its SparseCore regions in on a stage that offloads nothing), is switched too
    when the switch is installed; the wrappers already there stay as they are."""
    import jax.experimental.compute_on as jco

    lin = dl.import_lineage()
    name = "maxtext.experimental.lineage._mpmd_alias_probe"
    mod = types.ModuleType(name)
    mod.offload_to = jco.compute_on.mpmd_sparsecore_switch  # jax's own, as an import binds it
    before = {owner: owner.compute_on for owner in (lin.dsv3_router, lin.dsv3_mla, jco)}
    sc = dict(compute_type="tpu_sparsecore", out_memory_spaces=jax.memory.Space.Device)
    f = lambda x: x
    sys.modules[name] = mod
    try:
      dl._install_sparsecore_switch(vars(lin))
      self.assertIs(mod.offload_to.mpmd_sparsecore_flag, dl._SC)
      self.assertEqual({owner: owner.compute_on for owner in before}, before)
      with dl.no_sparsecore():
        self.assertIs(mod.offload_to(**sc)(f), f)
      self.assertIsNot(mod.offload_to(**sc)(f), f)
    finally:
      del sys.modules[name]

  def test_reloaded_module(self):
    """A reloaded copy of the module (a new thread-local flag, the wrappers already installed) adopts the
    wrappers' flag: its switch still switches."""
    lin, saved = dl.import_lineage(), dl._SC
    sc = dict(compute_type="tpu_sparsecore", out_memory_spaces=jax.memory.Space.Device)
    f = lambda x: x
    try:
      dl._SC = threading.local()
      dl._install_sparsecore_switch(vars(lin))
      self.assertIs(dl._SC, saved)
      with dl.no_sparsecore():
        self.assertIs(lin.dsv3_mla.compute_on(**sc)(f), f)
    finally:
      dl._SC = saved


@needs_lineage
class Loading(unittest.TestCase):

  def test_missing_dependency(self):
    """Lineage there but a module it imports not (typeguard here, as on a venv without it): import_lineage names that
    module and that it is missing; available() False."""
    code = (
        "from maxtext.experimental.mpmd.jax import ds_ep_lineage as dl\n"
        "import sys; sys.modules['typeguard'] = None\n"  # (Lineage's modules import it, ops.py:31)
        "try:\n"
        "  dl.import_lineage()\n"
        "except ImportError as e:\n"
        "  print(f'error: {e}')\n"
        "print(f'available: {dl.available()}')\n"
    )
    env = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONPATH=testing.child_pythonpath(__name__))
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300)
    self.assertEqual(out.returncode, 0, out.stderr[-2000:])
    self.assertRegex(
        out.stdout,
        r"error: ds_ep_lineage could not import MaxText's Lineage stack \(maxtext\.experimental\.lineage\): "
        r".*typeguard.* -- a module it imports is missing \(typeguard and jaxtyping",
    )
    self.assertIn("available: False", out.stdout)


class RegionCache(unittest.TestCase):

  def test_cache(self):
    """`region` hands out one Region per (stage config, chunk structure, stage mesh, capacity factor, SparseCore
    answer) -- the same object while cached, whatever else the chunk is (its index, stage, first / last) -- and keeps
    the REGION_CACHE last asked for: asking refreshes one, building one more evicts the least recently used.  An
    evicted Region still alive elsewhere (as a program's trace holds one) is handed out again; one that is not is
    built anew.  Another stage mesh, or another model or stage config, gets a Region of its own, built with them.
    (A stand-in Region: no Lineage needed.)"""
    made = []

    class Stub:

      def __init__(self, *args):
        made.append(args)

    scfg = stage_cfg(small_cfg())
    # fresh empty caches of the module's own kinds
    with (
        mock.patch.object(dl, "Region", Stub),
        mock.patch.object(dl, "REGION_CACHE", 2),
        mock.patch.object(dl, "_REGIONS", type(dl._REGIONS)()),
        mock.patch.object(dl, "_LIVE", type(dl._LIVE)()),
    ):
      get = lambda n, cf=dl.CAPACITY_FACTOR, sc=False, cfg=scfg, mesh="mesh", **kw: dl.region(
          cfg, dataclasses.replace(chunk_spec(n), **kw), mesh, capacity_factor=cf, sparsecore=sc
      )
      a = get(1)
      self.assertIs(get(1, index=3, stage=1, first=True), a)
      b = get(2)
      self.assertIs(get(1), a)  # refreshed: b is the least recently used now
      c = get(3)  # b evicted
      self.assertEqual((len(dl._REGIONS), len(made)), (2, 3))
      self.assertIs(get(2), b)  # still alive (held here): handed out again, not rebuilt; a evicted
      self.assertEqual(len(made), 3)
      gone = weakref.ref(a)
      del a
      gc.collect()
      self.assertIsNone(gone())  # the weak part keeps no Region alive
      self.assertNotIn(get(1), (b, c))
      self.assertEqual(len(made), 4)  # rebuilt
      self.assertIsNot(get(1, cf=1.0), get(1))
      self.assertEqual(made[-1][5:], (1.0, False))  # (Region's capacity_factor, sparsecore)
      self.assertIsNot(get(1, sc=True), get(1))
      self.assertEqual(made[-1][5:], (dl.CAPACITY_FACTOR, True))
      self.assertEqual((len(dl._REGIONS), len(made)), (2, 6))
      # another stage mesh (one stage a slice: every stage offloads, so the SparseCore answers do not tell them apart)
      self.assertIsNot(get(1, mesh="mesh2"), get(1))
      self.assertEqual(made[-1][:2], (scfg, "mesh2"))  # (Region's scfg, mesh)
      other = dataclasses.replace(scfg, model=scfg.model.replace(routed_scaling_factor=1.0))  # layer_kwargs reads it
      self.assertIsNot(get(1, cfg=other), get(1))  # another model or stage config
      self.assertEqual(made[-1][:2], (other, "mesh"))
      self.assertEqual((len(dl._REGIONS), len(made)), (2, 8))


@needs_lineage
class Routing(unittest.TestCase):

  def test_matches_ds_ep_selection(self):
    """Lineage's expert selection (dsv3_router.expert_indices_and_coeffs, its compute_on regions as plain code) with
    the arguments the region gives it (layer_kwargs) picks ds_ep's experts
    (src/maxtext/experimental/mpmd/jax/deepseek/moe.select_experts:
    top-k among each token's best groups, a group scored by its two best post-bias scores) and weighs them as ds_ep
    does, on fp32 scores: DeepSeek-V3's grouping, two where k / topk_routing_group is not 2 (Lineage's own config
    would score a group by its top k / g there), and groups off.  The TPU tests pin the routing, so this is where
    the routing rule is checked exactly."""
    from maxtext.experimental.mpmd.jax.deepseek import moe as ds_moe

    devs = _cpu_devices()
    if devs is None:
      self.skipTest("needs 4 CPU devices")
    m3 = dl.a3_mesh(Mesh(np.array(devs, dtype=object).reshape(2, 2), ds_ep.FE, axis_types=(AxisType.Explicit,) * 2))
    sel = dl.import_lineage().dsv3_router.expert_indices_and_coeffs
    rng = np.random.default_rng(0)
    for E, k, g, tg in ((256, 8, 8, 4), (64, 8, 8, 2), (32, 3, 4, 3), (32, 8, -1, -1)):
      with self.subTest(experts=E, k=k, groups=(g, tg)):
        cfg = small_cfg(
            num_experts=E,
            num_experts_per_tok=k,
            n_routing_groups=g,
            topk_routing_group=tg,
            routed_bias=True,
            float32_gate_logits=True,
        )
        kw = dl.layer_kwargs(cfg, m3)
        pre = jax.nn.sigmoid(jnp.asarray(rng.standard_normal((1024, E)), jnp.float32))
        logits = pre + jnp.asarray(0.05 * rng.standard_normal(E), jnp.float32)
        with dl.no_sparsecore():
          idx, coeffs = sel(
              logits,
              pre,
              **{
                  a: kw[a]
                  for a in (
                      "num_experts_per_tok",
                      "routed_scaling_factor",
                      "n_routing_groups",
                      "topk_routing_group",
                      "topk_in_group",
                  )
              },
          )
        want = ds_moe.select_experts(logits, cfg)
        np.testing.assert_array_equal(np.sort(idx, -1), np.sort(want, -1))
        w = jnp.take_along_axis(pre, want, axis=-1)
        w = cfg.routed_scaling_factor * w / (w.sum(-1, keepdims=True) + 1e-20)
        order, want_order = np.argsort(idx, -1), np.argsort(want, -1)
        np.testing.assert_allclose(
            np.take_along_axis(np.asarray(coeffs), order, -1),
            np.take_along_axis(np.asarray(w), want_order, -1),
            rtol=1e-6,
        )


@needs_lineage
class Yarn(unittest.TestCase):

  def test_matches_ds_ep(self):
    """The YaRN tables and the softmax scale Lineage's MLA gets are ds_ep's own: `kernels`' cos / sin (its
    get_yarn_freqs arguments: rope dim, theta, positions, betas, rope factor) exactly the package's rope.freqs_cis_for
    table in Lineage's split-half layout, in fp32; and the query scale dsv3_mla._softmax_scale computes from
    layer_kwargs' head dims, positions, rope factor and mscale, DeepSeekConfig.softmax_scale -- at DeepSeekConfig's
    YaRN defaults and at other values (mscale != 1 among them), where a swapped or dropped setting shows.  The TPU
    tests against Lineage's entry points build the region and its references from the same kernels / layer_kwargs,
    and those against ds_ep's own math (OnTPU.test_vs_ds_ep_math, test_dense_layers) hold the region to it only
    loosely and at DeepSeekConfig's YaRN defaults (mscale 1): only this checks the settings exactly, on CPU, and at
    other values."""
    from maxtext.experimental.mpmd.jax.deepseek import rope

    devs = _cpu_devices()
    if devs is None:
      self.skipTest("needs 4 CPU devices")
    m3 = dl.a3_mesh(Mesh(np.array(devs, dtype=object).reshape(2, 2), ds_ep.FE, axis_types=(AxisType.Explicit,) * 2))
    scale_of = dl.import_lineage().dsv3_mla._softmax_scale
    seq = 512
    for name, change in (
        ("defaults", {}),
        (
            "other values",
            dict(
                rope_max_timescale=50_000,
                rope_factor=32,
                beta_fast=16,
                original_max_position_embeddings=2048,
                mscale=0.707,
            ),
        ),
    ):
      with self.subTest(name):
        cfg = small_cfg(**change)
        with _as_tpu7x(), jax.set_mesh(m3):
          cos, sin = jax.jit(lambda: dl.kernels(cfg.replace(dtype=jnp.float32), m3, 4, seq)[0])()
        t = np.asarray(rope.freqs_cis_for(cfg, seq))  # [seq, rope / 2] complex64, interleaved pairs' angles
        for got, half in ((cos, t.real), (sin, t.imag)):
          want = np.concatenate([half, half], axis=-1)  # split-half: channel i and i + rope / 2 share one
          np.testing.assert_array_equal(np.asarray(got), np.broadcast_to(want[None, :, None], got.shape), name)
        kw = dl.layer_kwargs(cfg, m3)
        self.assertAlmostEqual(
            scale_of(
                **{
                    k: kw[k]
                    for k in (
                        "qk_head_dim",
                        "rope_head_dim",
                        "max_position_embeddings",
                        "original_max_position_embeddings",
                        "rope_factor",
                        "mscale",
                    )
                }
            ),
            cfg.softmax_scale,
            places=12,
        )


# ------------------------------------------------------------------------- weight conversion (needs Lineage)
@contextlib.contextmanager
def _tensorflow_stub():
  """lineage/interop/maxtext_layout.py imports dsv3_model -> dsv3 -> xla_transforms, which imports tensorflow's
  hlo_pb2 (max_async_overlap_transform.py:66), only to restamp megascale send/recv: stub that package chain while
  importing, and restore sys.modules' tensorflow entries after."""
  names = [
      "tensorflow",
      "tensorflow.compiler",
      "tensorflow.compiler.xla",
      "tensorflow.compiler.xla.service",
      "tensorflow.compiler.xla.service.hlo_pb2",
  ]
  saved = {n: sys.modules[n] for n in names if n in sys.modules}
  for n in names:
    sys.modules.setdefault(n, types.ModuleType(n))
  for parent, child in zip(names[:-1], names[1:]):
    setattr(sys.modules[parent], child.rsplit(".", 1)[1], sys.modules[child])
  try:
    yield
  finally:
    for n in names:
      if n in saved:
        sys.modules[n] = saved[n]
      else:
        sys.modules.pop(n, None)


def _to_port(layer):
  """ds_ep's layer tree in MaxText's NNX names (ds_ep_parity.port_tree's moe_to_port)."""
  moe = layer["moe"]
  return {
      **{k: v for k, v in layer.items() if k != "moe"},
      "DeepSeekMoeBlock_0": {
          "MoeBlock_0": {"gate": moe["gate"], "wi_0": moe["wi_0"], "wi_1": moe["wi_1"], "wo": moe["wo"]},
          "shared_experts": moe["shared_experts"],
      },
  }


@needs_lineage
class Conversion(unittest.TestCase):

  def tree(self, cfg, n=3, bias=False, dense=False, seed=0):
    return stacked(cfg, n, bias, dense, seed)

  def test_round_trip_sparse(self):
    """Weights -> Lineage -> (as gradients) back: bit-exact, every combination of routed bias and
    float32_gate_logits, with rope channels and without (qk_rope_head_dim 0); the bias goes in as given (or zeros of
    router_bias_dtype) and its gradient is zero."""
    for bias, f32, rope in [(b, f, 64) for b in (False, True) for f in (False, True)] + [(True, False, 0)]:
      with self.subTest(bias=bias, f32gl=f32, rope=rope):
        cfg = small_cfg(routed_bias=bias, float32_gate_logits=f32, qk_rope_head_dim=rope)
        w = self.tree(cfg, bias=bias)
        lw = dl.sparse_from_maxtext(w, cfg)
        self.assertEqual(lw.moe.router.kernel.dtype, jnp.bfloat16)
        want_bias = w["moe"]["gate"]["bias"] if bias else np.zeros((3, cfg.num_experts))
        np.testing.assert_array_equal(np.asarray(lw.moe.router.bias, np.float32), want_bias)
        self.assertEqual(lw.moe.router.bias.dtype, jnp.float32 if f32 else jnp.bfloat16)
        back = dl.sparse_to_maxtext(lw, cfg, bias=bias)
        self.assertEqual(jax.tree.structure(back), jax.tree.structure(w))
        for (name, a), (_, b) in zip(leaves_with_names(back), leaves_with_names(w)):
          if name.endswith("['bias']"):
            self.assertFalse(np.any(np.asarray(a)), name)
            self.assertEqual(a.dtype, b.dtype, name)
          else:
            self.assertEqual(a.dtype, b.dtype, name)
            np.testing.assert_array_equal(a, b, err_msg=name)

  def test_round_trip_dense(self):
    cfg = small_cfg()
    w = self.tree(cfg, n=2, dense=True)
    back = dl.dense_to_maxtext(dl.dense_from_maxtext(w, cfg), cfg)
    self.assertEqual(jax.tree.structure(back), jax.tree.structure(w))
    for (name, a), (_, b) in zip(leaves_with_names(back), leaves_with_names(w)):
      np.testing.assert_array_equal(a, b, err_msg=name)

  def test_layout(self):
    """Where each channel lands (arange-filled leaves): split-half rope, k_up / v_up, the fused tiled gate."""
    cfg = small_cfg()
    n, R, N, V = 2, cfg.qk_rope_head_dim, cfg.qk_nope_head_dim, cfg.v_head_dim
    D, E, F = cfg.emb_dim, cfg.num_experts, cfg.moe_mlp_dim
    leaves, tdef = jax.tree.flatten(stacked_shapes(cfg, n), is_leaf=ds_ep._is_shape)
    w = jax.tree.unflatten(
        tdef,
        [jnp.asarray((np.arange(np.prod(s)) + 1000 * i) % 4093, jnp.float32).reshape(s) for i, s in enumerate(leaves)],
    )  # every leaf distinct (wi_0 is not wi_1)
    lw = dl.sparse_from_maxtext(w, cfg)
    att = w["self_attention"]
    perm = np.concatenate([np.arange(N), N + np.arange(0, R, 2), N + np.arange(1, R, 2)])
    np.testing.assert_array_equal(lw.mla.q_up, np.asarray(att["wq_b"]["kernel"])[..., perm])
    kperm = np.concatenate(
        [np.arange(cfg.kv_lora_rank), cfg.kv_lora_rank + np.arange(0, R, 2), cfg.kv_lora_rank + np.arange(1, R, 2)]
    )
    np.testing.assert_array_equal(lw.mla.kv_down, np.asarray(att["wkv_a"]["kernel"])[..., kperm])
    np.testing.assert_array_equal(lw.mla.k_up, np.asarray(att["wkv_b"]["kernel"])[..., :N])
    np.testing.assert_array_equal(lw.mla.v_up, np.asarray(att["wkv_b"]["kernel"])[..., N : N + V])
    self.assertEqual(lw.moe.routed.gate.shape, (n, E, D // 8, 8, 2 * F))
    self.assertEqual(lw.moe.routed.linear.shape, (n, E, F // 8, 8, D))
    gate = np.asarray(lw.moe.routed.gate).reshape(n, E, D, 2 * F)
    np.testing.assert_array_equal(gate[..., :F], w["moe"]["wi_0"])  # SiLU half first
    np.testing.assert_array_equal(gate[..., F:], w["moe"]["wi_1"])
    np.testing.assert_array_equal(np.asarray(lw.moe.routed.linear).reshape(n, E, F, D), w["moe"]["wo"])
    np.testing.assert_array_equal(lw.pre_attn_norm_scale, w["pre_self_attention_layer_norm"]["scale"])
    np.testing.assert_array_equal(lw.post_attn_norm_scale, w["post_self_attention_layer_norm"]["scale"])
    np.testing.assert_array_equal(lw.moe.shared.gate_1, w["moe"]["shared_experts"]["wi_1"]["kernel"])

  def test_matches_maxtext_layout(self):
    """Against Lineage's own MaxText <-> Lineage converter (interop/maxtext_layout's per-layer helpers), with and
    without a routed bias, for the MoE and the dense layer, both directions; and without rope channels
    (qk_rope_head_dim 0), where both leave the q / kv projections as they are."""
    with _tensorflow_stub():
      from maxtext.experimental.lineage.interop import maxtext_layout as ml
    for bias, rope in ((False, 64), (True, 64), (True, 0)):
      with self.subTest(bias=bias, rope=rope):
        cfg = small_cfg(routed_bias=bias, qk_rope_head_dim=rope)
        w = self.tree(cfg, bias=bias)
        port = _to_port(w)
        if not bias:  # the helper wants a bias leaf (KeyError without one); ds_ep's absent bias is zeros
          port["DeepSeekMoeBlock_0"]["MoeBlock_0"]["gate"] = {
              **port["DeepSeekMoeBlock_0"]["MoeBlock_0"]["gate"],
              "bias": jnp.zeros((3, cfg.num_experts), jnp.bfloat16),
          }
        leaf = functools.partial(ml._to_lineage_leaf, param_scan_axis=0, dtype=jnp.bfloat16)
        theirs, their_bias = ml._sparse_layer_from_maxtext(
            port, leaf, jnp.asarray, qk_head_dim=cfg.qk_nope_head_dim, rope_head_dim=cfg.qk_rope_head_dim
        )
        ours = dl.sparse_from_maxtext(w, cfg, tiles=False)
        np.testing.assert_array_equal(np.asarray(ours.moe.router.bias, np.float32), np.asarray(their_bias, np.float32))
        theirs = dataclasses.replace(
            theirs,
            moe=dataclasses.replace(theirs.moe, router=dataclasses.replace(theirs.moe.router, bias=ours.moe.router.bias)),
        )
        self.assertEqual(jax.tree.structure(ours), jax.tree.structure(theirs))
        for (name, a), (_, b) in zip(leaves_with_names(ours), leaves_with_names(theirs)):
          np.testing.assert_array_equal(a, b, err_msg=name)
        # the inverse, but for the bias (theirs passes it through, ours is its gradient: zero)
        back_theirs = ml._sparse_layer_to_maxtext(
            theirs,
            their_bias,
            functools.partial(ml._to_maxtext_leaf, param_scan_axis=0),
            rope_head_dim=cfg.qk_rope_head_dim,
        )
        back_ours = _to_port(dl.sparse_to_maxtext(ours, cfg, bias=True, tiles=False))
        for tree in (back_theirs, back_ours):
          del tree["DeepSeekMoeBlock_0"]["MoeBlock_0"]["gate"]["bias"]
        self.assertEqual(jax.tree.structure(back_ours), jax.tree.structure(back_theirs))
        for (name, a), (_, b) in zip(leaves_with_names(back_ours), leaves_with_names(back_theirs)):
          np.testing.assert_array_equal(a, b, err_msg=name)
    for rope in (64, 0):
      with self.subTest(dense=True, rope=rope):
        cfg = small_cfg(qk_rope_head_dim=rope)
        wd = self.tree(cfg, n=2, dense=True)
        leaf = functools.partial(ml._to_lineage_leaf, param_scan_axis=0, dtype=jnp.bfloat16)
        theirs = ml._dense_layer_from_maxtext(
            wd, leaf, qk_head_dim=cfg.qk_nope_head_dim, rope_head_dim=cfg.qk_rope_head_dim
        )
        ours = dl.dense_from_maxtext(wd, cfg)
        for (name, a), (_, b) in zip(leaves_with_names(ours), leaves_with_names(theirs)):
          np.testing.assert_array_equal(a, b, err_msg=name)
        back_theirs = ml._dense_layer_to_maxtext(
            theirs, functools.partial(ml._to_maxtext_leaf, param_scan_axis=0), rope_head_dim=cfg.qk_rope_head_dim
        )
        for (name, a), (_, b) in zip(leaves_with_names(dl.dense_to_maxtext(ours, cfg)), leaves_with_names(back_theirs)):
          np.testing.assert_array_equal(a, b, err_msg=name)

  def test_a3_types(self):
    """The A3 compute types the region gives Lineage's weights (and their gradients), written out: what traced and
    lowered fwd + bwd at DeepSeek-V3 sizes and ran here on TPU.  Norms, down-projections, router kernel and shared
    experts reduced over every axis; the up / out projections head-sharded over 'attn' and reduced over the token
    axes but 'attn'; q_norm / kv_norm over the token axes only (also over 'attn' fails the backward,
    dsv3_mla.py:623); the routed experts over 'expert', reduced over fsdp_moe; the router bias plain."""
    devs = _cpu_devices()
    if devs is None:
      self.skipTest("needs 4 CPU devices")
    m2 = Mesh(np.array(devs, dtype=object).reshape(2, 2), ds_ep.FE, axis_types=(AxisType.Explicit,) * 2)
    rg = dl.Region(stage_cfg(small_cfg()), m2, n_moe=1, n_dense=1)
    ALL, FE_ = frozenset({"fsdp", "expert", "attn"}), frozenset(ds_ep.FE)
    sp, dn = rg.types[("reduced", "m3", "sparse")], rg.types[("reduced", "m3", "dense")]
    for spec, want in (
        (sp.pre_attn_norm_scale, P(None, reduced=ALL)),
        (sp.mla.q_down, P(None, None, reduced=ALL)),
        (sp.mla.kv_down, P(None, None, reduced=ALL)),
        (sp.mla.q_norm_scale, P(None, reduced=FE_)),
        (sp.mla.kv_norm_scale, P(None, reduced=FE_)),
        (sp.mla.q_up, P(None, "attn", None, reduced=FE_)),
        (sp.mla.k_up, P(None, "attn", None, reduced=FE_)),
        (sp.mla.v_up, P(None, "attn", None, reduced=FE_)),
        (sp.mla.out, P("attn", None, None, reduced=FE_)),
        (sp.moe.router.kernel, P(None, None, reduced=ALL)),
        (sp.moe.router.bias, P(None)),
        (sp.moe.shared.gate_0, P(None, None, reduced=ALL)),
        (sp.moe.routed.gate, P("expert", None, None, None, reduced={"fsdp", "attn"})),
        (sp.moe.routed.linear, P("expert", None, None, None, reduced={"fsdp", "attn"})),
        (dn.mlp.gate_1, P(None, None, reduced=ALL)),
        (dn.mla.out, P("attn", None, None, reduced=FE_)),
        (rg.types[("unreduced", "m2", "sparse")].mla.q_up, P(None, None, None, unreduced=FE_)),
        (rg.types[("unreduced", "m2", "sparse")].moe.routed.gate, P("expert", None, None, None, unreduced={"fsdp"})),
    ):
      self.assertEqual(spec, want)

  def test_typed_pipeline_on_cpu_mesh(self):
    """The region's typed conversions on a CPU stage mesh: ds_ep's compute copies (typed chunk_wc_specs) -> one
    Lineage tree per layer on the A3 view, typed reduced over Lineage's collect axes, values as converted on the
    host; and per-layer Lineage gradients (typed unreduced, each device's partial) -> ds_ep's gradient tree typed
    chunk_gacc_specs(explicit=True), whose sum over the devices is the converted sum of the partials."""
    devs = _cpu_devices()
    if devs is None:
      self.skipTest("needs 4 CPU devices")
    m2 = Mesh(np.array(devs, dtype=object).reshape(2, 2), ds_ep.FE, axis_types=(AxisType.Explicit,) * 2)
    for bias, dense in ((True, False), (False, True)):
      with self.subTest(bias=bias, dense=dense):
        cfg = small_cfg(routed_bias=bias, float32_gate_logits=bias)
        rg = dl.Region(stage_cfg(cfg), m2, n_moe=2, n_dense=2 if dense else 0, routed_bias=bias)
        self.assertFalse(rg.sparsecore)  # CPU: no SparseCore offload, the regions trace as plain code
        key = "dense_layers" if dense else "layers"
        conv = dl.dense_from_maxtext if dense else dl.sparse_from_maxtext
        back = (lambda g: dl.dense_to_maxtext(g, cfg)) if dense else (lambda g: dl.sparse_to_maxtext(g, cfg, bias))
        kd = "dense" if dense else "sparse"
        host = stacked(cfg, 2, bias, dense, seed=1)
        plain = jax.tree.map(lambda s: P(*s.partitions), rg.wc_specs[key], is_leaf=ds_ep._is_spec)
        arrs = jax.tree.map(lambda a, s: jax.device_put(a, NamedSharding(m2, s)), host, plain)
        w = jax.jit(jax.shard_map(lambda t: t, mesh=m2, in_specs=(plain,), out_specs=rg.wc_specs[key], check_vma=False))(
            arrs
        )
        wl = jax.jit(lambda w: rg._to_lineage(w, dense))(w)
        t3 = rg.types[("reduced", "m3", kd)]
        for l in range(2):
          want = conv(jax.tree.map(lambda a: jnp.asarray(a[l]), host), cfg)
          jax.tree_util.tree_map_with_path(
              lambda p, x, s: self.assertEqual(jax.typeof(x).sharding.spec, s, jax.tree_util.keystr(p)), wl[l], t3
          )
          for (name, a), (_, b) in zip(leaves_with_names(wl[l]), leaves_with_names(want)):
            np.testing.assert_array_equal(a, b, err_msg=f"layer {l}{name}")
        # gradients: each device's partial = the layer's converted weights; every leaf is unreduced over 4 devices
        # but the routed experts', over 'fsdp' (x 'attn' of size 1) only
        u3 = rg.types[("unreduced", "m3", kd)]
        if not dense:
          u3 = dl._drop_bias(u3)
        g_in = [dl._drop_bias(x) if not dense else x for x in wl]
        p3 = jax.tree.map(lambda s: P(*s.partitions), u3, is_leaf=ds_ep._is_spec)
        grads = jax.jit(
            lambda gs: [
                jax.shard_map(lambda t: t, mesh=rg.mesh3, in_specs=(p3,), out_specs=u3, check_vma=False)(
                    jax.tree.map(lambda a, s: jax.reshard(a, NamedSharding(rg.mesh3, s)), g, p3)
                )
                for g in gs
            ]
        )(g_in)
        out = jax.jit(lambda gs: rg._from_lineage(gs, dense))(grads)
        jax.tree_util.tree_map_with_path(
            lambda p, x, s: self.assertEqual(jax.typeof(x).sharding.spec, s, jax.tree_util.keystr(p)),
            out,
            rg.gacc_specs[key],
        )
        summed = jax.jit(
            lambda t: jax.tree.map(
                lambda a, s: jax.reshard(a.astype(jnp.float32), NamedSharding(m2, P(*s.partitions))),
                t,
                rg.gacc_specs[key],
            )
        )(out)
        want = back(conv(jax.tree.map(jnp.asarray, host), cfg))
        for (name, a), (_, b) in zip(leaves_with_names(summed), leaves_with_names(want)):
          k = (
              0
              if name.endswith("['bias']")
              else 2
              if name in ("['moe']['wi_0']", "['moe']['wi_1']", "['moe']['wo']")
              else 4
          )
          np.testing.assert_array_equal(a, k * np.asarray(b, np.float32), err_msg=name)


# ------------------------------------------------------------------------- the region traced on CPU
def _eqns(closed, pred):
  """How many equations of a closed jaxpr (sub-jaxprs included) satisfy `pred`."""
  from jax.extend import core as jcore

  n = 0
  todo = [closed.jaxpr]
  while todo:
    jaxpr = todo.pop()
    for e in jaxpr.eqns:
      n += bool(pred(e))
      todo.extend(jcore.jaxprs_in_params(e.params))
  return n


def _sparsecore_eqn(e):
  return e.primitive.name == "compute_on" and e.params.get("compute_type") == "tpu_sparsecore"


def _unreduced_stack(e):
  """A jnp.stack of unreduced values (jax 0.11.1 scales them by the unreduced axes' sizes: the module doc)."""
  return e.primitive.name == "stack" and any(
      getattr(getattr(v.aval, "sharding", None), "spec", None) is not None and v.aval.sharding.spec.unreduced
      for v in e.invars
  )


def _region_jits(closed):
  """[(name, number of outputs)] of the Region's nested jits (`_prim_impl` / `_fwd_impl` / `_bwd_impl`) a closed jaxpr
  calls, sub-jaxprs included."""
  from jax.extend import core as jcore

  out, todo = [], [closed.jaxpr]
  while todo:
    jaxpr = todo.pop()
    for e in jaxpr.eqns:
      if e.primitive.name in ("pjit", "jit") and e.params.get("name") in ("_prim_impl", "_fwd_impl", "_bwd_impl"):
        out.append((e.params["name"], len(e.outvars)))
      todo.extend(jcore.jaxprs_in_params(e.params))
  return sorted(out)


def _device_rows(bank):
  """A device's rows of a half's activation bank (init_activation_bank: [tiles, 8, 2F], over the token axes)."""
  sharding = jax.typeof(bank).sharding
  token = sharding.spec[0]
  shards = math.prod(sharding.mesh.shape[ax] for ax in ((token,) if isinstance(token, str) else token))
  return bank.shape[0] * bank.shape[1] // shards


@contextlib.contextmanager
def _as_tpu7x():
  """jax's TPU-info queries answered for a TPU7x, so Lineage's Pallas kernels trace on CPU (they have no CPU path to
  run)."""
  import jax._src.tpu_info as tpu_info

  with (
      mock.patch.object(tpu_info, "get_device_kind", lambda: "TPU7x"),
      mock.patch.object(tpu_info, "get_num_device_cores", lambda: 1),
  ):
    yield


@needs_lineage
class RegionTrace(unittest.TestCase):
  """The region traced on a CPU stage mesh, as a TPU7x's (`_as_tpu7x`): what a host without a TPU can check of it (no
  CI run has Lineage: module doc).  setUpClass traces one region -- 3 MoE + 1 dense layers, routed bias,
  float32_gate_logits, SparseCore off -- in a program of every kind (`programs`), counting each nested jit's traces
  and recording each one's activation bank."""

  @classmethod
  def setUpClass(cls):
    import jax._src.tpu_info as tpu_info

    devs = _cpu_devices()
    if devs is None or not hasattr(tpu_info, "get_device_kind"):
      raise unittest.SkipTest("needs 4 CPU devices and jax._src.tpu_info")
    cfg = cls.cfg = small_cfg(
        routed_bias=True,
        float32_gate_logits=True,
        sa_block_q=512,
        sa_block_kv=512,
        sa_block_kv_compute=512,
        max_target_length=512,
    )
    m2 = cls.m2 = Mesh(np.array(devs, dtype=object).reshape(2, 2), ds_ep.FE, axis_types=(AxisType.Explicit,) * 2)
    SL = dl.import_lineage().dsv3_sparse_layer
    init_bank, rule = SL.init_activation_bank, []
    cls.traces, cls.banks = collections.Counter(), []

    def counting(name, f):
      @functools.wraps(f)  # the nested jit keeps the method's name (`_region_jits`)
      def g(self, *a):
        cls.traces[name] += 1
        rule.append(name)
        try:
          return f(self, *a)
        finally:
          rule.pop()

      return g

    def bank(mb0, *a, **kw):
      """init_activation_bank, recording (the rule tracing, num_layers, a device's rows of mb0's bank)."""
      (b0, o0), b1 = init_bank(mb0, *a, **kw)
      cls.banks.append((rule[-1], kw["num_layers"], _device_rows(b0)))
      return (b0, o0), b1

    with (
        _as_tpu7x(),
        mock.patch.object(dl.Region, "_prim_impl", counting("prim", dl.Region._prim_impl)),
        mock.patch.object(dl.Region, "_fwd_impl", counting("fwd", dl.Region._fwd_impl)),
        mock.patch.object(dl.Region, "_bwd_impl", counting("bwd", dl.Region._bwd_impl)),
        mock.patch.object(SL, "init_activation_bank", bank),
    ):
      cls.rg = rg = dl.Region(stage_cfg(cfg), m2, n_moe=3, n_dense=1, routed_bias=True)
      cls.w, cls.wd = cls.sds(rg, "layers"), cls.sds(rg, "dense_layers")
      cls.h = jax.ShapeDtypeStruct((8, 512, cfg.emb_dim), jnp.bfloat16, sharding=NamedSharding(m2, dl.ACT2))
      cls.programs = cls.trace_programs(rg)
      _, (_, _, _, (_, cls.res_body, cls.res_epi), _) = jax.eval_shape(rg._fwd, cls.w, cls.wd, cls.h)

  @classmethod
  def tearDownClass(cls):
    # The traced programs pin ~7.4M GC-tracked objects, through these attributes and jax's trace caches both.  Kept
    # for the session, they make CPython put off every automatic full collection, and a later class's dead CPU
    # pipelines (cyclic garbage holding XLA executables and their maps) pile up past vm.max_map_count (65530):
    # "Cannot allocate memory", or an abort, where one process runs this file and ds_ep_test with Lineage importable.
    for k in ("cfg", "m2", "traces", "banks", "rg", "w", "wd", "h", "programs", "res_body", "res_epi"):
      if k in cls.__dict__:
        delattr(cls, k)
    jax.clear_caches()
    gc.collect()
    super().tearDownClass()

  @classmethod
  def sds(cls, region, k):
    """The region's `k` weights ('layers' / 'dense_layers') as ShapeDtypeStructs typed as ds_ep's programs pass
    them."""
    ch = chunk_spec(region.n_moe, region.n_dense, region.routed_bias)
    shapes, dts = ds_ep.chunk_shapes(region.scfg, ch)[k], ds_ep.leaf_dtype_tree(region.scfg, ch, "compute")[k]
    return jax.tree.map(
        lambda s, dt, sp: jax.ShapeDtypeStruct(s, dt, sharding=NamedSharding(region.mesh, sp)),
        shapes,
        dts,
        region.wc_specs[k],
        is_leaf=ds_ep._is_shape,
    )

  @classmethod
  def trace_programs(cls, rg):
    """{kind: traced program} in this order: 'F' (the region alone, as ds_ep's F programs call it: no AD), 'B'
    (jax.vjp), 'B checkpoint' (jax.vjp of jax.checkpoint(region), as the fused last chunk's under
    remat_fused_last), then 'F set_mesh' and 'B set_mesh' (the same under m2's mesh context)."""
    w, wd, h = cls.w, cls.wd, cls.h

    def fprog(w, wd, h):
      return rg(w, wd, h)[0]

    def bprog(region):
      def prog(w, wd, h, d):
        y, vjp, aux = jax.vjp(region, w, wd, h, has_aux=True)
        return (y, aux) + tuple(vjp(d))

      return prog

    out = {
        "F": jax.jit(fprog).trace(w, wd, h),
        "B": jax.jit(bprog(rg)).trace(w, wd, h, h),
        "B checkpoint": jax.jit(bprog(jax.checkpoint(rg))).trace(w, wd, h, h),
    }
    with jax.set_mesh(cls.m2):
      out["F set_mesh"] = jax.jit(fprog).trace(w, wd, h)
      out["B set_mesh"] = jax.jit(bprog(rg)).trace(w, wd, h, h)
    return out

  def test_programs(self):
    """Each nested jit traces once across the five programs (Region._fwd's one context): the primal in F (and in
    the jax.checkpoint'ed B, from the cache), the two rules in B; the gradients come out typed
    chunk_gacc_specs(explicit=True), y and dh P(FE, None, None); no SparseCore region in any program and no stack
    of unreduced values in the B program (3 layers: both step barriers of the chain trace); every scan-body residual
    keeps mb0's MoE input folded (out_proj_out + orig_x, orig_x None) as dsv3_sparse_layers does
    (dsv3_sparse_layers.py:247-259) -- one activation less a layer, which only the residual's structure shows.  And
    a region with SparseCore on has SparseCore regions."""
    rg, cfg, b = self.rg, self.cfg, self.programs["B"]
    self.assertFalse(rg.sparsecore)
    self.assertEqual(dict(self.traces), {"prim": 1, "fwd": 1, "bwd": 1})
    y, aux, dw, dwd, dh = b.out_info
    for name, tree, specs in (("dW", dw, rg.gacc_specs["layers"]), ("dW dense", dwd, rg.gacc_specs["dense_layers"])):
      jax.tree_util.tree_map_with_path(
          lambda p, a, s, name=name: self.assertEqual(a.sharding.spec, s, name + jax.tree_util.keystr(p)), tree, specs
      )
    self.assertEqual((y.sharding.spec, dh.sharding.spec), (dl.ACT2, dl.ACT2))
    self.assertEqual(aux.group_sizes.shape, (3, 4, cfg.num_experts))
    self.assertEqual(aux.selected_experts.shape, (3, 8, 512, cfg.num_experts_per_tok))
    for kind, prog in self.programs.items():
      self.assertEqual(_eqns(prog.jaxpr, _sparsecore_eqn), 0, kind)
    self.assertEqual(_eqns(b.jaxpr, _unreduced_stack), 0)
    self.assertEqual(len(self.res_body), 2)
    for res in self.res_body:
      self.assertIsNone(res.mb0.moe.orig_x)
      self.assertIsNotNone(res.mb1.moe.orig_x)
    self.assertIsNotNone(self.res_epi.mb0.moe.orig_x)  # the epilogue's is not folded (its backward takes both)
    with _as_tpu7x():
      on = dl.Region(stage_cfg(cfg), self.m2, n_moe=1, routed_bias=True, sparsecore=True)
      f = jax.jit(lambda w, h: on(w, None, h)).trace(self.sds(on, "layers"), self.h)
    self.assertGreater(_eqns(f.jaxpr, _sparsecore_eqn), 0)

  def test_forward_only_bank(self):
    """The primal -- what a program without AD runs, ds_ep's F -- is the forward alone: its nested jit returns y and
    the routing (3 values: no residual for the program to discard) and sizes Lineage's activation bank for no layer,
    each half's bank only its scratch chunk (bank_aligned(chunk_capacity) rows a device); the forward rule, which
    jax.vjp runs, sizes it for the 3 MoE layers, BANK_FACTOR x their balanced gating activations more.  The B
    programs call the two rules; the jax.checkpoint'ed one traced the primal (jax.checkpoint stages the region out)
    but runs only the forward rule -- twice, DCE'd to (y, aux) for the forward and whole for the recompute -- so its
    bank is the full one too."""
    lin = dl.import_lineage()
    from maxtext.experimental.lineage import dsv3_experts

    cfg, rg = self.cfg, self.rg
    slots = self.h.shape[0] // self.m2.size // 2 * self.h.shape[1] * cfg.num_experts_per_tok  # balanced, a half
    scratch = dsv3_experts.bank_aligned(lin.dsv3_router.chunk_capacity(slots, rg.scfg.ep, rg.capacity_factor))
    full = dsv3_experts.bank_aligned(int(dl.BANK_FACTOR * rg.n_moe * slots)) + scratch
    self.assertEqual(sorted(self.banks), [("fwd", 3, full), ("prim", 0, scratch)])
    self.assertLess(scratch, full)
    jits = {kind: _region_jits(prog.jaxpr) for kind, prog in self.programs.items()}
    self.assertEqual(jits["F"], [("_prim_impl", 3)])
    self.assertEqual(jits["F set_mesh"], [("_prim_impl", 3)])
    for kind in ("B", "B set_mesh"):
      self.assertEqual([name for name, _ in jits[kind]], ["_bwd_impl", "_fwd_impl"], kind)
    self.assertEqual([name for name, _ in jits["B checkpoint"]], ["_bwd_impl", "_fwd_impl", "_fwd_impl"])
    self.assertIn(("_fwd_impl", 3), jits["B checkpoint"])

  def test_capacity_factor_bank(self):
    """Lineage's activation bank is sized with the Region's capacity factor -- the last hop of ds_ep_pipeline
    --lineage-capacity-factor: `_chain_fwd` hands init_activation_bank the factor the router's chunk capacity reads,
    layer_kwargs' -- here where the factor shows: fsdp 1 x ep 4 at 3.0 (MaxText's DeepSeek-V3 Lineage recipe's), a
    chunk of 3 x a device's balanced load of routed slots (at this class's ep 2 any factor >= 2 clamps to every slot
    routable to a device).  Each half's bank is its checkpoint rows -- none in the primal, BANK_FACTOR x the layer's
    balanced gating activations in the forward rule -- then one such chunk of scratch rows, and the gate GMM finds
    the checkpoint rows ending where that chunk starts (dsv3_experts.bank_chunk_row).  Sized with a smaller factor
    than the router's (the module default here), the bank would end before the chunk does: the gate GMM would write it
    from a negative row, outside the bank, on a TPU."""
    lin = dl.import_lineage()
    from maxtext.experimental.lineage import dsv3_experts

    SL, cfg = lin.dsv3_sparse_layer, self.cfg
    m14 = Mesh(self.m2.devices.reshape(1, 4), ds_ep.FE, axis_types=(AxisType.Explicit,) * 2)
    init_bank, chunk_row, banks, ckpt_rows = SL.init_activation_bank, dsv3_experts.bank_chunk_row, [], []

    def bank(mb0, *a, **kw):
      (b0, o0), b1 = init_bank(mb0, *a, **kw)
      banks.append((kw["num_layers"], _device_rows(b0)))
      return (b0, o0), b1

    def row(b, capacity, *a):  # b: a device's bank, capacity: the chunk's rows
      ckpt_rows.append(b.shape[0] - dsv3_experts.bank_aligned(capacity))
      return chunk_row(b, capacity, *a)

    with (
        _as_tpu7x(),
        mock.patch.object(SL, "init_activation_bank", bank),
        mock.patch.object(dsv3_experts, "bank_chunk_row", row),
    ):
      rg = dl.Region(stage_cfg(cfg, fsdp=1, ep=4), m14, n_moe=1, routed_bias=True, capacity_factor=3.0, sparsecore=False)
      w = self.sds(rg, "layers")
      h = jax.ShapeDtypeStruct(self.h.shape, jnp.bfloat16, sharding=NamedSharding(m14, dl.ACT2))
      jax.jit(lambda w, h: rg(w, None, h)[0]).trace(w, h)  # the primal: an F program
      prim_rows = set(ckpt_rows)
      ckpt_rows.clear()
      jax.eval_shape(rg._fwd, w, None, h)  # the forward rule: a B program's forward
    slots = h.shape[0] // m14.size // 2 * h.shape[1] * cfg.num_experts_per_tok  # balanced, a half
    chunk = dsv3_experts.bank_aligned(lin.dsv3_router.chunk_capacity(slots, 4, rg.capacity_factor))
    ckpt = dsv3_experts.bank_aligned(int(dl.BANK_FACTOR * slots))
    self.assertNotEqual(chunk, dsv3_experts.bank_aligned(lin.dsv3_router.chunk_capacity(slots, 4, dl.CAPACITY_FACTOR)))
    self.assertEqual(banks, [(0, chunk), (1, ckpt + chunk)])
    self.assertEqual((prim_rows, set(ckpt_rows)), ({0}, {ckpt}))


@needs_lineage
class RegionInputs(unittest.TestCase):
  """What a Region is built with and refuses to be called with, on a CPU stage mesh: nothing of Lineage runs or
  traces (`_check_inputs` raises before), so each case takes milliseconds."""

  @classmethod
  def setUpClass(cls):
    devs = _cpu_devices()
    if devs is None:
      raise unittest.SkipTest("needs 4 CPU devices")
    cls.cfg = small_cfg(sa_block_q=512, sa_block_kv=512, sa_block_kv_compute=512, max_target_length=512)
    cls.m2 = Mesh(np.array(devs, dtype=object).reshape(2, 2), ds_ep.FE, axis_types=(AxisType.Explicit,) * 2)

  def test_capacity_factor(self):
    """The capacity factor a Region is built with is its layer functions' (ds_ep_pipeline --lineage-capacity-factor,
    Region.kw), which Lineage's MoE chunk capacity reads (and its activation bank's size: RegionTrace's
    test_capacity_factor_bank)."""
    for cf in (1.0, 3.0):
      self.assertEqual(dl.Region(stage_cfg(self.cfg), self.m2, n_moe=1, capacity_factor=cf).kw["capacity_factor"], cf)

  def test_refused_inputs(self):
    """h not [B, S, D] bf16, an odd number of rows a device, h or a weight not typed as ds_ep's explicit programs
    pass them, dense weights to a region without dense layers or none to one with, a weight tree not ds_ep's:
    each a ValueError naming it, raised before any of Lineage traces -- without the check two of them would trace
    on unnoticed (an untyped h, a stray w_dense) and the rest fail inside Lineage or jax, naming neither."""
    cfg, m2 = self.cfg, self.m2
    rg = dl.Region(stage_cfg(cfg), m2, n_moe=1)
    rgd = dl.Region(stage_cfg(cfg), m2, n_moe=1, n_dense=1)
    w, wd = RegionTrace.sds(rg, "layers"), RegionTrace.sds(rgd, "dense_layers")
    act = NamedSharding(m2, dl.ACT2)
    h = lambda b=8, d=cfg.emb_dim, dt=jnp.bfloat16, sh=act: jax.ShapeDtypeStruct((b, 512, d), dt, sharding=sh)
    plain = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=NamedSharding(m2, P(*x.sharding.spec.partitions))), w
    )
    missing = {k: v for k, v in w.items() if k != "post_self_attention_layer_norm"}
    D = cfg.emb_dim
    for name, region, args, want in (
        ("h fp32", rg, (w, None, h(dt=jnp.float32)), rf"h: want \[B, S, {D}\] bf16"),
        ("h of another width", rg, (w, None, h(d=D // 2)), rf"h: want \[B, S, {D}\] bf16"),
        ("one row a device", rg, (w, None, h(b=4)), "an even number a device"),
        ("h typed replicated", rg, (w, None, h(sh=NamedSharding(m2, P(None, None, None)))), "h: typed"),
        ("w_dense without dense layers", rg, (w, wd, h()), "w_dense given to a region without dense layers"),
        ("no w_dense with dense layers", rgd, (w, None, h()), "w_dense: tree"),
        ("a leaf missing", rg, (missing, None, h()), "w_layers: tree"),
        ("weights not typed reduced", rg, (plain, None, h()), r"w_layers\[.*\]: typed"),
    ):
      with self.subTest(name), _as_tpu7x(), self.assertRaisesRegex(ValueError, want):
        jax.eval_shape(region, *args)


# ------------------------------------------------------------------------- on the TPU (TPU7x, 8 devices)
def tpu_cfg(**kw):
  """The TPU tests' model: Lineage-valid, small enough to compile a 5-layer program in about a minute."""
  base = dict(
      vocab_size=256,
      emb_dim=1024,
      num_decoder_layers=6,
      first_num_dense_layers=0,
      num_query_heads=16,
      num_kv_heads=16,
      q_lora_rank=512,
      kv_lora_rank=512,
      qk_nope_head_dim=128,
      qk_rope_head_dim=64,
      v_head_dim=128,
      mlp_dim=1024,
      moe_mlp_dim=512,
      num_experts=32,
      num_experts_per_tok=8,
      n_routing_groups=8,
      topk_routing_group=4,
      routed_bias=False,
      sa_block_q=2048,
      sa_block_kv=2048,
      sa_block_kv_compute=2048,
      max_target_length=2048,
      attention_kernel="tokamax_splash",
  )
  base.update(kw)
  return DeepSeekConfig(**base)


B, S = 8, 2048  # two rows a device of a 4-device stage: one a half
S1K = dict(max_target_length=1024, sa_block_q=1024, sa_block_kv=1024, sa_block_kv_compute=1024)


def pins(cfg, mode="balanced"):
  """The k experts `OnTPU.setup`'s routed bias pins every token to: 'balanced' half on each of two EP shards (k /
  topk_routing_group experts of the first topk_routing_group / 2 groups of each half of the groups; without
  groups every (E / k)-th expert), 'shard0' the lowest-numbered ones (of the first topk_routing_group groups) -- all
  on EP shard 0 of two."""
  E, k, g, tg = cfg.num_experts, cfg.num_experts_per_tok, cfg.n_routing_groups, cfg.topk_routing_group
  if g <= 0:
    return [i * (E // k) for i in range(k)] if mode == "balanced" else list(range(k))
  groups = [*range(tg // 2), *range(g // 2, g // 2 + tg // 2)] if mode == "balanced" else list(range(tg))
  return sorted(grp * (E // g) + i for grp in groups for i in range(k // tg))


def _rel(a, b):
  a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
  return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def compare(name, got, want, log=print):
  """Per-leaf relative error, cosine of the whole gradient, bitwise-equal leaves -> (worst rel, cos, n bitwise)."""
  ga, wa = leaves_with_names(got), leaves_with_names(want)
  assert [n for n, _ in ga] == [n for n, _ in wa], ([n for n, _ in ga], [n for n, _ in wa])
  rels = sorted(((_rel(a, b), n) for (n, a), (_, b) in zip(ga, wa)), reverse=True)
  flat = lambda ls: np.concatenate([np.asarray(a, np.float64).ravel() for _, a in ls])
  x, y = flat(ga), flat(wa)
  cos = float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))
  nbit = sum(np.array_equal(np.asarray(a), np.asarray(b)) for (_, a), (_, b) in zip(ga, wa))
  log(
      f"  {name}: cos {cos:.7f}, worst leaf rel {rels[0][0]:.3g} {rels[0][1]}, next {rels[1][0]:.3g} {rels[1][1]}, "
      f"bitwise {nbit}/{len(ga)}"
  )
  return rels[0][0], cos, nbit


def _full(tree, specs, mesh):
  """Gradients typed unreduced (or anything) -> fp32, summed / replicated over `mesh`, in the program."""
  return jax.tree.map(lambda g, s: jax.reshard(g.astype(jnp.float32), NamedSharding(mesh, P(*s.partitions))), tree, specs)


def _put_typed(host, mesh, typed):
  """Host arrays -> device arrays typed `typed` (reduced types only come out of a program: an identity shard_map)."""
  plain = jax.tree.map(lambda s: P(*s.partitions), typed, is_leaf=ds_ep._is_spec)
  arrs = jax.tree.map(lambda a, s: jax.device_put(a, NamedSharding(mesh, s)), host, plain)
  return jax.jit(jax.shard_map(lambda t: t, mesh=mesh, in_specs=(plain,), out_specs=typed, check_vma=False))(arrs)


def sparsecore_collectives(hlo):
  """Collectives inside the computations a compiled TPU module runs on SparseCore (`execution_thread="sparsecore"`):
  Lineage's compute_on('tpu_sparsecore') regions.  (`async_execution_thread="sparsecore"` / sparse_core_config alone
  also mark libtpu's own copy and gather offloads, which a program without offloaded collectives has too.)"""
  blocks = re.split(r"\n(?=(?:ENTRY )?%[\w.\-]+ \()", hlo)
  coll = re.compile(r"\b(all-gather|reduce-scatter|all-reduce|all-to-all|collective-permute)(-start)?\(")
  return sum(len(coll.findall(b)) for b in blocks if re.search(r'^}, execution_thread="sparsecore"', b, re.M))


def _storage_types(n):
  """Lineage's storage layout on the A3 view (dsv3_sharding._SPARSE_8X8X8 with FSDP over 'fsdp', heads over 'attn',
  experts over 'expert'), stacked over n layers, routed weights untiled: what dsv3_sparse_layers' hooks gather from
  and reduce-scatter back to."""
  T = dl.import_lineage().dsv3_types
  f, a = "fsdp", dl.ATTN
  return T.DSv3SparseLayerWeightsPytree(
      pre_attn_norm_scale=P(None, None),
      post_attn_norm_scale=P(None, None),
      mla=T.DSv3MLAWeightsPytree(
          q_down=P(None, f, None),
          q_up=P(None, f, a, None),
          q_norm_scale=P(None, None),
          kv_down=P(None, f, None),
          k_up=P(None, None, a, f),
          v_up=P(None, None, a, f),
          kv_norm_scale=P(None, None),
          out=P(None, a, None, f),
      ),
      moe=T.DSv3MoEWeightsPytree(
          router=T.DSv3MoERouterWeightsPytree(kernel=P(None, None, None), bias=P(None, None)),
          routed=T.DSv3MoERoutedExpertWeightsPytree(gate=P(None, "expert", f, None), linear=P(None, "expert", f, None)),
          shared=T.DSv3MoESharedExpertWeightsPytree(
              gate_0=P(None, None, f), gate_1=P(None, None, f), linear=P(None, f, None)
          ),
      ),
  )


@pytest.mark.tpu_only
@pytest.mark.scheduled_only
class OnTPU(unittest.TestCase):
  """The region on a TPU7x of 8 devices (a tpu7x-8).  Stage meshes are ds_ep's own objects
  (ds_ep_pipeline.typed_meshes(stage_meshes(2, 2, 2), 'explicit')): stage 1 = devices 4-7 (libtpu offloads its
  collectives to SparseCore), stage 0 = devices 0-3 (holds chip (0,0,0): no offload under a TPU VM's own client,
  the region traces Lineage's SparseCore regions as plain code; Pathways' servers offload there too, and
  test_sparsecore_off_vs_on skips on Pathways).  Skipped without Lineage, or (asked only then) off a TPU7x of 8
  devices.
  Minutes of Lineage tracing and TPU compiles: scheduled_only.  Numbers go to stdout (-s)."""

  @classmethod
  def setUpClass(cls):
    if not HAVE_LINEAGE:
      raise unittest.SkipTest(f"needs Lineage: {NO_LINEAGE}")
    testing.require_tpu(_on_tpu7x(), "a TPU7x with 8 devices (a tpu7x-8: OnTPU's stage meshes are its halves)")
    cls.meshes = dsp.typed_meshes(dsp.stage_meshes(2, 2, 2), "explicit")
    cls.mesh42 = dsp.typed_meshes(dsp.stage_meshes(1, 4, 2), "explicit")[0]  # all 8 devices: fsdp 4 x ep 2
    cls.mesh41 = Mesh(cls.meshes[1].devices.reshape(4, 1), ds_ep.FE, axis_types=(AxisType.Explicit,) * 2)
    cls.lin = dl.import_lineage()

  def setup(
      self,
      cfg,
      n,
      nd=0,
      bias=False,
      stage=1,
      seed=0,
      router_scale=0.1,
      pinned=False,
      mesh=None,
      rows=2,
      capacity_factor=dl.CAPACITY_FACTOR,
  ):
    """(region, its typed inputs, the host weights, the host activation / cotangent) on stage `stage` (or on
    `mesh`), `rows` rows a device of cfg.max_target_length tokens.

    `pinned` (True: 'balanced', or 'shard0': `pins`, kept in self.pins): a routed bias of 8 on k experts and 0
    elsewhere, so every token picks those whatever its sigmoid scores (in (0, 1)) and their bf16 rounding: two
    programs whose Lineage math rounds differently by a bf16 ulp (the comparisons of this class) then route alike,
    and differ by bf16 reassociation alone.  Unpinned, top-k routing on bf16-rounded scores is chaotic under such
    ulps: tokens near a tie flip experts, more in each later layer (Step 0: 3 / 40 / 73 of 16384 tokens in layers
    0 / 1 / 2), and the router kernel's gradient moves by a few percent with them.  The coefficients, the router
    kernel's gradient and the experts' math are unaffected.  'shard0' puts every token on EP shard 0's experts,
    twice a shard's balanced load: at capacity factor 1 its MoE chunk loop runs twice and the bank overflows."""
    if pinned:
      bias = True
      cfg = cfg.replace(routed_bias=True)
    mesh = self.meshes[stage] if mesh is None else mesh
    scfg = stage_cfg(cfg, fsdp=mesh.shape["fsdp"], ep=mesh.shape["expert"])
    ch = ds_ep.ChunkSpec(
        index=0, layers=tuple(range(n)), first=False, last=False, stage=stage, dense=nd, routed_bias=bias
    )
    rg = dl.region(scfg, ch, mesh, capacity_factor=capacity_factor)
    scales = {"gate": router_scale, "bias": 0.05}
    host = {"layers": stacked(cfg, n, bias, seed=seed, scales=scales)}
    self.pins = pins(cfg, "balanced" if pinned is True else pinned) if pinned else None
    if pinned:
      pin = np.zeros((n, cfg.num_experts), np.float32)
      pin[:, self.pins] = 8.0
      host["layers"]["moe"]["gate"]["bias"] = jnp.asarray(pin, host["layers"]["moe"]["gate"]["bias"].dtype)
    if nd:
      host["dense_layers"] = stacked(cfg, nd, dense=True, seed=seed + 1)
    w = _put_typed(host["layers"], mesh, rg.wc_specs["layers"])
    wd = _put_typed(host["dense_layers"], mesh, rg.wc_specs["dense_layers"]) if nd else None
    rng = np.random.default_rng(seed + 2)
    shape = (rows * mesh.size, cfg.max_target_length, cfg.emb_dim)
    hx = np.asarray(jnp.asarray(rng.standard_normal(shape), jnp.bfloat16))
    dy = np.asarray(jnp.asarray(rng.standard_normal(shape), jnp.bfloat16))
    return rg, w, wd, host, hx, dy

  def assert_pinned(self, sel):
    """Every token of every layer picked exactly the pinned experts (aux.selected_experts)."""
    np.testing.assert_array_equal(np.sort(sel, -1), np.broadcast_to(np.asarray(self.pins), sel.shape))

  def lower_region(self, rg, w, wd, hx, dy):
    """One B program: forward, then the region's VJP -> (lowered program, its inputs); outputs y, group sizes,
    selected experts, dW (summed in fp32), dW_dense, dh."""
    mesh = rg.mesh
    put = lambda a: jax.device_put(a, NamedSharding(mesh, dl.ACT2))

    def prog(w, wd, h, d):
      y, vjp, aux = jax.vjp(rg, w, wd, h, has_aux=True)
      dw, dwd, dh = vjp(d)
      return (
          y,
          aux.group_sizes,
          aux.selected_experts,
          _full(dw, rg.gacc_specs["layers"], mesh),
          _full(dwd, rg.gacc_specs["dense_layers"], mesh) if wd is not None else None,
          dh,
      )

    args = (w, wd, put(hx), put(dy))
    return jax.jit(prog, compiler_options=dl.compiler_options(mesh)).lower(*args), args

  def run_region(self, rg, w, wd, hx, dy):
    """lower_region, compiled and run -> host (y, group sizes, selected experts, dW, dW_dense, dh)."""
    lo, args = self.lower_region(rg, w, wd, hx, dy)
    out = lo.compile()(*args)
    return jax.tree.map(lambda a: np.asarray(a, np.float32) if a.dtype != jnp.int32 else np.asarray(a), out)

  def run_lineage(self, rg, fn, w, hx, dy, convert=None, back=None):
    """Reference program: y, dW, dx of `fn(w3, x, yarn, splash)` on the region's A3 view, under m3's abstract mesh
    (the context enclosing the VJP, as it must).  It mirrors the region's program around `fn`: h and dy enter on
    ds_ep's mesh and are resharded as the region reshards them; `convert` maps the weight inputs to Lineage's
    (default: none, `w` is Lineage's), `back` the weight gradients to the outputs (default: the region's own
    inverse conversion of per-layer MoE gradients, summed in fp32).  Lineage's bf16 numerics follow how XLA lays
    out its operands and results, which the program's other ops steer: a reference that fed Lineage parameters or
    reduced its gradients in Lineage's layout would round the same math differently by a bf16 ulp (measured: dW of
    wq_b / wkv_a, whose rope channels the conversion permutes, 2e-3 apart)."""
    m3, m2, cfg = rg.mesh3, rg.mesh, rg.cfg
    put = lambda a: jax.device_put(a, NamedSharding(m2, dl.ACT2))
    if back is None:
      back = lambda dw: _full(rg._from_lineage(dw, dense=False), rg.gacc_specs["layers"], m2)

    def prog(w, h, d):
      w3 = w if convert is None else convert(w)
      x, d3 = (jax.reshard(a, NamedSharding(m3, dl.ACT3)) for a in (h, d))
      with jax.sharding.use_abstract_mesh(m3.abstract_mesh):
        yarn, splash = dl.kernels(cfg, m3, *h.shape[:2])
        y, vjp = jax.vjp(lambda w, x: fn(w, x, yarn, splash), w3, x)
        dw, dx = vjp(d3)
      return (jax.reshard(y, NamedSharding(m2, dl.ACT2)), back(dw), jax.reshard(dx, NamedSharding(m2, dl.ACT2)))

    with dl.no_sparsecore(not rg.sparsecore):
      out = (
          jax.jit(prog, compiler_options=dl.compiler_options(m2))
          .lower(w, put(hx), put(dy))
          .compile()(w, put(hx), put(dy))
      )
    return jax.tree.map(lambda a: np.asarray(a, np.float32), out)

  def convert(self, rg):
    return lambda w: rg._to_lineage(w, dense=False)

  def to_maxtext(self, rg, g, tiles=True):
    """Lineage-layout fp32 gradients -> ds_ep's layout, fp32 kept (the f32gl branch only skips the bf16 cast)."""
    return jax.tree.map(
        lambda a: np.asarray(a, np.float32),
        dl.sparse_to_maxtext(
            jax.tree.map(jnp.asarray, g), rg.cfg.replace(float32_gate_logits=True), bias=rg.routed_bias, tiles=tiles
        ),
    )

  def chained_single_layers(self, rg):
    """L calls of Lineage's public single-layer entry point, dsv3_sparse_layer (prologue + epilogue of one
    layer, its own custom VJP, collected weights), chained."""
    kw = rg.kw

    def fn(ws, x, yarn, splash):
      for w in ws:
        x, _ = self.lin.dsv3_sparse_layer.dsv3_sparse_layer(x, w, yarn, splash, ragged_buffer_factor=dl.BANK_FACTOR, **kw)
      return x

    return fn

  def stack_layers(self, rg):
    """Lineage's driver dsv3_sparse_layers with its real collect / reduce hooks on storage-layout weights."""
    from maxtext.experimental.lineage import dsv3_sparse_layers

    kw = rg.kw

    def fn(w, x, yarn, splash):
      return dsv3_sparse_layers.dsv3_sparse_layers(x, w, yarn, splash, ragged_buffer_factor=dl.BANK_FACTOR, **kw)[0]

    return fn

  def per_layer(self, tree, l):
    return jax.tree.map(lambda a: a[l], tree)

  # ----------------------------------------------------------------------- (1) vs Lineage's public entry points
  def test_vs_dsv3_sparse_layer_n1(self):
    """n=1 is exactly dsv3_sparse_layer (prologue + epilogue of one layer): same program pieces, same inputs, so the
    output, dx and the summed weight gradients are bitwise Lineage's, whatever the routing (free here but in the
    last case) -- on stage 1 with groups 8/4 and a random routed bias; with groups off and no routed bias (Lineage's
    (1, 1, k) routing and its bias of zeros); over all 8 devices as fsdp 4 x ep 2 with 4 rows a device, groups 8/2
    (k / topk_routing_group 4: `routing`) and the fp32 router; with every token on EP shard 0's experts at
    capacity factor 1 (its MoE chunk loop runs twice, the activation bank overflows: Lineage's scratch rows and
    recomputed gate GMM); and on stage 1's devices as fsdp 4 x ep 1 (Lineage's EP collectives over a size-1 axis,
    SparseCore on)."""
    for seed, (name, cfg, kw) in enumerate(
        (
            ("stage 1, groups 8/4, random routed bias", tpu_cfg(routed_bias=True), dict(bias=True)),
            (
                "groups off, no routed bias",
                tpu_cfg(num_experts=16, num_experts_per_tok=4, n_routing_groups=-1, topk_routing_group=-1),
                {},
            ),
            (
                "fsdp 4 x ep 2, 4 rows a device, groups 8/2, fp32 router",
                tpu_cfg(num_experts=64, topk_routing_group=2, routed_bias=True, float32_gate_logits=True, **S1K),
                dict(bias=True, mesh=self.mesh42, rows=4),
            ),
            ("every token on EP shard 0, capacity factor 1", tpu_cfg(), dict(pinned="shard0", capacity_factor=1.0)),
            ("fsdp 4 x ep 1", tpu_cfg(routed_bias=True), dict(bias=True, mesh=self.mesh41)),
        )
    ):
      with self.subTest(name):
        rg, w, wd, host, hx, dy = self.setup(cfg, 1, seed=seed, **kw)
        (y, gs, sel, dw, _, dh) = self.run_region(rg, w, wd, hx, dy)
        self.assertEqual(gs.sum(), hx.shape[0] * hx.shape[1] * cfg.num_experts_per_tok)
        if self.pins is not None:
          self.assert_pinned(sel)
          self.assertEqual(gs[..., cfg.num_experts // 2 :].sum(), 0)  # EP shard 1's experts: no token
        ry, rdw, rdx = self.run_lineage(rg, self.chained_single_layers(rg), w, hx, dy, convert=self.convert(rg))
        print(
            f"n=1 vs dsv3_sparse_layer ({name}): y bitwise {np.array_equal(y, ry)} rel {_rel(y, ry):.3g}; dh "
            f"bitwise {np.array_equal(dh, rdx)} rel {_rel(dh, rdx):.3g}"
        )
        _, _, nbit = compare("dW", dw, rdw)
        np.testing.assert_array_equal(y, ry)
        np.testing.assert_array_equal(dh, rdx)
        self.assertEqual(nbit, len(jax.tree.leaves(rdw)))

  def test_vs_dsv3_sparse_layers_n3_n4_n5(self):
    """n=3, 4 and 5 against Lineage's own driver, dsv3_sparse_layers, with its real hooks on storage-layout weights
    (SparseCore-1 all-gathers in, reduce-scatter + all-reduce out), routing pinned (`setup`), layer by layer.  (Fed
    gathered weights typed reduced instead, its hooks leave the gradients unreduced and its backward jnp.stacks the
    scanned layers' (ops._interleave_arrays): at n=5 layers 3 and 4 come out 4x -- 2x the routed experts, unreduced
    over one axis fewer -- measured on this TPU.)  Not bitwise: the
    hooks' SparseCore regions hand the layers differently laid-out weights, so the dots round differently by bf16
    ulps (Step 0: with the hooks' regions traced as plain code the forward is bitwise equal), and the hooks reduce
    each layer's gradient in bf16 where this test sums the region's partials in fp32.  Unpinned, the routing flips
    those ulps cause moved the router kernel's gradient by 2.7e-2 (n=3) / 3.8e-2 (n=4)."""
    for n in (3, 4, 5):
      with self.subTest(n=n):
        cfg = tpu_cfg()
        rg, w, wd, host, hx, dy = self.setup(cfg, n, seed=n, pinned=True)
        (y, gs, sel, dw, _, dh) = self.run_region(rg, w, wd, hx, dy)
        self.assertEqual(gs.reshape(n, -1).sum(-1).tolist(), [B * S * cfg.num_experts_per_tok] * n)
        self.assert_pinned(sel)
        st = _storage_types(n)
        lw = dl.sparse_from_maxtext(jax.tree.map(jnp.asarray, host["layers"]), cfg, tiles=False)
        w3 = _put_typed(jax.tree.map(np.asarray, lw), rg.mesh3, st)
        ry, rdw, rdx = self.run_lineage(rg, self.stack_layers(rg), w3, hx, dy, back=lambda dw: _full(dw, st, rg.mesh3))
        print(
            f"n={n} vs dsv3_sparse_layers: y bitwise {np.array_equal(y, ry)} rel {_rel(y, ry):.3g}; dh rel "
            f"{_rel(dh, rdx):.3g}"
        )
        want = self.to_maxtext(rg, rdw, tiles=False)
        worst, cos, _ = compare("dW", dw, want)
        self.assertLess(_rel(y, ry), 1e-2)
        self.assertLess(_rel(dh, rdx), 2e-2)
        for l in range(n):
          worst, cos, _ = compare(f"dW layer {l}", self.per_layer(dw, l), self.per_layer(want, l))
          self.assertGreater(cos, 0.9999, l)
          self.assertLess(worst, 2e-2, l)

  # ----------------------------------------------------------------------- (3) n=2 and n=5, per layer
  def test_vs_chained_single_layers_n2_n5(self):
    """n=2 (no Lineage entry point: its driver asserts n >= 3) and n=5 (where Lineage's own driver would stack
    unreduced gradients with jnp.stack, 4x wrong at this mesh) against n chained dsv3_sparse_layer calls -- the same
    per-layer math without the cross-layer overlap, so not the same rounding (phases group the halves' ops
    differently) -- layer by layer, routing pinned.  Unpinned: n=2 worst leaf 5.5e-3, n=5 3.8e-2 (layer 2's router
    kernel, after routing flips in layers 0-1).  The reference sums Lineage's gradients in Lineage's layout and
    converts them on the host, so the region's inverse conversion is checked here too (a jnp.stack of unreduced
    gradients there, or in the driver, shows as a 4x / 2x layer)."""
    for n in (2, 5):
      with self.subTest(n=n):
        cfg = tpu_cfg()
        rg, w, wd, host, hx, dy = self.setup(cfg, n, seed=10 + n, pinned=True)
        (y, gs, sel, dw, _, dh) = self.run_region(rg, w, wd, hx, dy)
        self.assert_pinned(sel)
        u3 = [rg.types[("unreduced", "m3", "sparse")]] * n
        ry, rdw, rdx = self.run_lineage(
            rg,
            self.chained_single_layers(rg),
            w,
            hx,
            dy,
            convert=self.convert(rg),
            back=lambda dw: _full(dw, u3, rg.mesh3),
        )
        print(
            f"n={n} vs chained dsv3_sparse_layer: y bitwise {np.array_equal(y, ry)} rel {_rel(y, ry):.3g}; dh "
            f"bitwise {np.array_equal(dh, rdx)} rel {_rel(dh, rdx):.3g}"
        )
        want = self.to_maxtext(rg, jax.tree.map(lambda *a: np.stack(a), *rdw))
        compare("dW", dw, want)
        for l in range(n):
          worst, cos, _ = compare(f"dW layer {l}", self.per_layer(dw, l), self.per_layer(want, l))
          self.assertGreater(cos, 0.9999, l)
          self.assertLess(worst, 2e-2, l)
        self.assertLess(_rel(y, ry), 1e-2)
        self.assertLess(_rel(dh, rdx), 2e-2)

  # ----------------------------------------------------------------------- (2) vs ds_ep's own layer math
  def ds_ep_reference(self, rg, host, hx, dy):
    """ds_ep's unsplit layers (dense_layer_fwd, layer_fwd at top-k routing, pure-JAX experts: ragged_dot + the
    argsort dispatch, tokamax splash base-2) in one shard_map over the stage mesh, the weights' partial gradients
    psummed over their partial axes -> host (y, dW tree, dh)."""
    scfg, mesh = rg.scfg, rg.mesh
    plain = {k: jax.tree.map(lambda s: P(*s.partitions), rg.wc_specs[k], is_leaf=ds_ep._is_spec) for k in host}
    w = {k: jax.tree.map(lambda a, s: jax.device_put(a, NamedSharding(mesh, s)), host[k], plain[k]) for k in host}
    put = lambda a: jax.device_put(a, NamedSharding(mesh, dl.ACT2))

    def body(w, x, d):
      def f(w, x):
        for l in range(rg.n_dense):
          x = ds_ep.dense_layer_fwd(ds_ep._take(w["dense_layers"], l), x, scfg)
        for l in range(rg.n_moe):
          x = ds_ep.layer_fwd(ds_ep._take(w["layers"], l), x, None, scfg)
        return x

      y, vjp = jax.vjp(f, w, x)
      dw, dx = vjp(d)
      dw = {k: jax.tree.map(lambda g, s: jax.lax.psum(g, ds_ep._partial_axes(s.partitions)), dw[k], plain[k]) for k in dw}
      return y, dw, dx

    out = jax.jit(
        jax.shard_map(
            body, mesh=mesh, in_specs=(plain, dl.ACT2, dl.ACT2), out_specs=(dl.ACT2, plain, dl.ACT2), check_vma=False
        ),
        compiler_options=dl.compiler_options(mesh),
    )(w, put(hx), put(dy))
    return jax.tree.map(lambda a: np.asarray(a, np.float32), out)

  def test_vs_ds_ep_math(self):
    """Against ds_ep's own layer math (the package's MLA and pure-JAX MoE) at top-k routing, routed bias and
    float32_gate_logits on: loose tolerances -- Lineage's numerics differ by design (bf16 rope cos / sin, the
    coefficient before wo, its own splash and GMM kernels, the router kernel bf16) -- tight enough, with the
    routing pinned (`setup`), that a misplaced channel, a swapped wi_0 / wi_1 or a lost rope de-interleave fails.
    Unpinned, the two implementations' numerics route a few near-tie tokens differently, which moves the router
    kernel's gradient most (measured here: worst leaf 9.4e-2, the router kernel; wi_0 4.6e-2; cosine 0.99954):
    bounded only at the spread such flips make without a bug (Region's docstring: the router kernel's gradient
    0.06-0.14 off, cosine 0.996-0.9995 over the configs measured; bf16 router scores tie more, 2.7% of a layer's
    tokens flipped at top-4 of 16), cosine > 0.995 and a leaf < 0.25, at fp32 router scores -- which libtpu, jax or
    a seed can move with no bug present: the exact checks are the pinned subtest's and Routing's on CPU."""
    for pinned in (True, False):
      with self.subTest(pinned=pinned):
        cfg = tpu_cfg(routed_bias=True, float32_gate_logits=True)
        rg, w, wd, host, hx, dy = self.setup(cfg, 2, bias=True, seed=20, pinned=pinned)
        (y, gs, sel, dw, _, dh) = self.run_region(rg, w, wd, hx, dy)
        if pinned:
          self.assert_pinned(sel)
        ry, rdw, rdh = self.ds_ep_reference(rg, host, hx, dy)
        print(
            f'n=2 vs ds_ep layer math (routing {"pinned" if pinned else "free"}): y rel {_rel(y, ry):.3g}, dh rel '
            f"{_rel(dh, rdh):.3g}"
        )
        del dw["moe"]["gate"]["bias"], rdw["layers"]["moe"]["gate"]["bias"]  # not trained: zero on both sides
        worst, cos, _ = compare("dW", dw, rdw["layers"])
        self.assertLess(_rel(y, ry), 2e-2)
        self.assertLess(_rel(dh, rdh), 5e-2)
        self.assertGreater(cos, 0.999 if pinned else 0.995)
        self.assertLess(worst, 5e-2 if pinned else 0.25)

  # ----------------------------------------------------------------------- (5) dense layers
  def test_dense_layers(self):
    """A chunk with 2 dense layers then 2 MoE layers (routing pinned): against ds_ep's own math (loose, catches the
    dense conversion) and against Lineage's dsv3_dense_layer x 2 then chained dsv3_sparse_layer x 2 under one
    jax.vjp (the region recomputes each dense layer in its backward: the same math).  Not bitwise even in y:
    Lineage's dense layer has no optimization barriers, so XLA rounds it by what else a program computes -- in
    forward-only programs the region's y is bitwise Lineage's dense layers + the region's MoE chain (5e-4 from the
    chained single layers), while each one's backward program moves its forward by ~one bf16 ulp (the region's
    3.2e-3, the composition's 4.0e-3: the 4e-3 printed here)."""
    cfg = tpu_cfg(float32_gate_logits=True)
    rg, w, wd, host, hx, dy = self.setup(cfg, 2, nd=2, seed=30, pinned=True)
    (y, gs, sel, dw, dwd, dh) = self.run_region(rg, w, wd, hx, dy)
    self.assert_pinned(sel)
    ry, rdw, rdh = self.ds_ep_reference(rg, host, hx, dy)
    print(f"dense 2 + moe 2 vs ds_ep layer math: y rel {_rel(y, ry):.3g}, dh rel {_rel(dh, rdh):.3g}")
    worst_d, cos_d, _ = compare("dW dense", dwd, rdw["dense_layers"])
    del dw["moe"]["gate"]["bias"], rdw["layers"]["moe"]["gate"]["bias"]
    worst, cos, _ = compare("dW moe", dw, rdw["layers"])
    self.assertLess(_rel(y, ry), 2e-2)
    self.assertLess(_rel(dh, rdh), 5e-2)
    self.assertGreater(min(cos, cos_d), 0.999)
    self.assertLess(max(worst, worst_d), 5e-2)
    # against Lineage's own layers, composed by jax.vjp
    lin, kw = self.lin, rg.kw
    single = self.chained_single_layers(rg)
    mla_kw = {k: kw[k] for k in dl.MLA_KWARGS}

    def fn(ws, x, yarn, splash):
      for wdl in ws[0]:
        x = lin.dsv3_dense_layer.dsv3_dense_layer(x, wdl, yarn, splash, **mla_kw)
      return single(ws[1], x, yarn, splash)

    both = lambda ws: (rg._to_lineage(ws[0], True), rg._to_lineage(ws[1], False))
    back = lambda dw: (
        _full(rg._from_lineage(dw[0], True), rg.gacc_specs["dense_layers"], rg.mesh),
        _full(rg._from_lineage(dw[1], False), rg.gacc_specs["layers"], rg.mesh),
    )
    ly, (want_d, want), ldx = self.run_lineage(rg, fn, (wd, w), hx, dy, convert=both, back=back)
    print(
        f"dense 2 + moe 2 vs Lineage layers: y bitwise {np.array_equal(y, ly)} rel {_rel(y, ly):.3g}; dh bitwise "
        f"{np.array_equal(dh, ldx)} rel {_rel(dh, ldx):.3g}"
    )
    worst_d, cos_d, _ = compare("dW dense vs Lineage", dwd, want_d)
    del want["moe"]["gate"]["bias"]
    worst, cos, _ = compare("dW moe vs Lineage", dw, want)
    self.assertLess(_rel(y, ly), 1e-2)
    self.assertGreater(min(cos, cos_d), 0.9999)
    self.assertLess(max(worst, worst_d), 2e-2)

  # ----------------------------------------------------------------------- (4) SparseCore off vs on
  def test_sparsecore_off_vs_on(self):
    """The same chunk on stage 0 (devices 0-3: no SparseCore offload, so the region traces Lineage's SparseCore
    regions as plain code -- compiled with them, libtpu 0.0.46 aborts) and on stage 1 (devices 4-7, SparseCore
    on): the SC-on program has Lineage's SparseCore regions (lowered) and collectives on SparseCore (compiled), the
    SC-off one neither; same loss and gradients to bf16 level (routing pinned: the two run their collectives on
    different engines, which round differently).  McJAX only (SC_STAGE0)."""
    testing.skip_on_pathways(SC_STAGE0)
    cfg = tpu_cfg()
    lowered, texts, outs = {}, {}, {}
    for stage in (0, 1):  # lowered text first: compiled with SparseCore regions, stage 0 aborts the process
      rg, w, wd, host, hx, dy = self.setup(cfg, 2, stage=stage, seed=40, pinned=True)
      self.assertEqual(rg.sparsecore, stage == 1)
      lowered[stage] = self.lower_region(rg, w, wd, hx, dy)
      texts[stage] = [lowered[stage][0].as_text().count("sparseoffload")]
    self.assertEqual(texts[0], [0], "SparseCore regions in the stage-0 program")
    self.assertGreater(texts[1][0], 0)
    for stage in (0, 1):
      lo, args = lowered[stage]
      co = lo.compile()
      texts[stage].append(sparsecore_collectives(co.as_text()))
      outs[stage] = jax.tree.map(lambda a: np.asarray(a, np.float32), co(*args))
    print(
        f'SparseCore regions (lowered "sparseoffload", compiled collectives on SparseCore): stage 0 {texts[0]}, '
        f"stage 1 {texts[1]}"
    )
    self.assertEqual(texts[0], [0, 0])
    self.assertGreater(texts[1][1], 0)
    (y0, _, sel0, dw0, _, dh0), (y1, _, sel1, dw1, _, dh1) = outs[0], outs[1]
    self.assert_pinned(sel0)
    self.assert_pinned(sel1)
    loss = lambda y: float(np.sum(y.astype(np.float64) * np.asarray(dy, np.float64)) / y.size)
    print(
        f"stage 0 vs stage 1: loss {loss(y0):.7f} vs {loss(y1):.7f}; y bitwise {np.array_equal(y0, y1)} rel "
        f"{_rel(y0, y1):.3g}; dh rel {_rel(dh0, dh1):.3g}"
    )
    worst, cos, _ = compare("dW", dw0, dw1)
    self.assertLess(abs(loss(y0) - loss(y1)), 1e-3 * abs(loss(y1)) + 1e-6)
    self.assertLess(_rel(y0, y1), 1e-2)
    self.assertGreater(cos, 0.9999)
    self.assertLess(worst, 2e-2)

  # ----------------------------------------------------------------------- (6) the forward-only primal
  def test_forward_only_bank(self):
    """ds_ep's F program of a middle chunk of 2 MoE layers (ds_ep.ChunkPrograms' fwd: the region's primal, the
    activation bank sized for no layer) against the same program on the forward rule's forward (the bank for both
    layers, residuals dropped: the primal before the forward-only bank, `_FullBankPrimal`), on stage 0 (SparseCore
    off) and stage 1 (on): y bitwise equal, and bitwise the B program's (`run_region`: a MoE-only region's forward
    is the same in every program); the F program's temp lower by at least 0.9 x the two halves' checkpoint rows
    (BANK_FACTOR x 2 layers' balanced gating activations, [rows, 2F] bf16 each)."""
    from maxtext.experimental.lineage import dsv3_experts

    cfg = tpu_cfg()
    for stage in (0, 1):
      with self.subTest(stage=stage):
        rg, w, wd, host, hx, dy = self.setup(cfg, 2, stage=stage, seed=50, pinned=True)
        mesh, scfg = rg.mesh, rg.scfg
        ch = ds_ep.ChunkSpec(index=1, layers=(0, 1), first=False, last=False, stage=stage, routed_bias=True)
        args = (
            {"layers": w},
            jax.device_put(hx, NamedSharding(mesh, dl.ACT2)),
            jax.device_put(jnp.int32(0), NamedSharding(mesh, P())),
        )
        full = _FullBankPrimal(scfg, mesh, 2, routed_bias=True, sparsecore=rg.sparsecore)
        out = {}
        for name, region in (("forward-only bank", rg), ("full bank", full)):
          pr = ds_ep.ChunkPrograms(scfg, ch, mesh, hx.shape[0], hx.shape[1], 1.0, lineage_sparsecore=rg.sparsecore)
          self.assertEqual(set(pr.wspec), {"layers"})
          with mock.patch.object(dl, "region", lambda *a, region=region, **kw: region):
            co = pr.fwd.lower(*args).compile()
          y, res = co(*args)
          self.assertEqual(res, ())
          out[name] = (np.asarray(y, np.float32), co.memory_analysis().temp_size_in_bytes)
        (y, temp), (y_full, temp_full) = out["forward-only bank"], out["full bank"]
        y_b = self.run_region(rg, w, wd, hx, dy)[0]
        slots = hx.shape[0] // mesh.size // 2 * hx.shape[1] * cfg.num_experts_per_tok  # balanced, a half
        bank = 2 * dsv3_experts.bank_aligned(int(dl.BANK_FACTOR * 2 * slots)) * 2 * cfg.moe_mlp_dim * 2
        print(
            f"stage {stage} (SparseCore {rg.sparsecore}) F program: temp {temp / 2**30:.3f} GiB, full bank "
            f"{temp_full / 2**30:.3f} ({(temp - temp_full) / 2**30:+.3f}; the halves' checkpoint rows "
            f"{bank / 2**30:.3f}); y bitwise the full bank's {np.array_equal(y, y_full)}, the B program's "
            f"{np.array_equal(y, y_b)}"
        )
        np.testing.assert_array_equal(y, y_full)
        np.testing.assert_array_equal(y, y_b)
        self.assertGreaterEqual(temp_full - temp, 0.9 * bank)


class _FullBankPrimal(dl.Region):
  """A Region whose primal is the forward rule's forward, residuals dropped -- the primal before the forward-only
  bank (test_forward_only_bank's reference)."""

  def _primal(self, w_layers, w_dense, h):
    return self._fwd(w_layers, w_dense, h)[0]


if __name__ == "__main__":
  unittest.main()
