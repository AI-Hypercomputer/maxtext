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

"""Two-sided CPU checks for gdn_gated_norm_layout (models/qwen3.py `_pin_gdn_gated_norm_layout`).

1. Config: the default is 'auto'; a non-'auto' value needs gdn_out_proj_token_minor=True and a Gated DeltaNet decoder; a
   typo'd value or key is refused by config construction.
2. The real Qwen3NextGatedDeltaNet in training, alone and as a remat'd two-layer scan (the production structure: a
   forward body plus a replay in the backward body): the loss and every gradient (parameters and input) are
   bit-identical for 'auto', 'natural' and 'natural_fwd_bwd'. For 'token_minor' the loss, the input gradient and every
   parameter gradient but one are bit-identical; the gated-norm scale gradient, a float32 sum over the tokens whose
   order XLA:CPU takes from the operand layout, matches to a relative 1e-6 (measured: max |d| 9.5e-7, rel 1.3e-7).
   The lowered program carries exactly the expected extra LayoutConstraints on the 4-D
   (B, S, H_v, D_v) tensors: for 'natural' one row-major pin of the norm output per forward evaluation, plus one per
   backward for 'natural_fwd_bwd'; for 'token_minor' one token-minor (0, 2, 3, 1) pin of the GDN output per forward
   evaluation; none in prefill.
3. Negative controls go through the SAME checks and must fail them: the pin dropped, the pin with the permutation
   (0, 1, 3, 2), a value-changing pin (heads reversed), 'natural_fwd_bwd' without its cotangent pin, and for
   'token_minor' the pin dropped and the permutation (0, 2, 1, 3).
"""

import functools
import re
import sys
from unittest import mock

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pydantic

from maxtext.common.common_types import MODEL_MODE_PREFILL, MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.configs import types as config_types
from maxtext.layers import moe
from maxtext.models import qwen3
from tests.utils.test_helpers import get_test_config_path

_PASS_TOKEN = "GDN_GATED_NORM_LAYOUT_CHECKS_PASSED"
_SEQ, _EMB = 64, 64
_NORM_SHAPE = "1x64x2x128xbf16"  # the gated-norm output (B, S, H_v, D_v) of the layer below
_ROW_MAJOR_4D = "3, 2, 1, 0"  # minor_to_major of Layout(major_to_minor=(0, 1, 2, 3))
_TOKEN_MINOR_4D = "1, 3, 2, 0"  # minor_to_major of Layout(major_to_minor=(0, 2, 3, 1))
_TOKEN_MINOR_3D = "1, 2, 0"  # minor_to_major of the out_proj token-minor pin (0, 2, 1)
_LAYOUT_RE = re.compile(
    r"stablehlo\.custom_call @LayoutConstraint\(%[^)]*\)[^\n]*?result_layouts = \[dense<\[([0-9, ]+)\]>[^\n]*?"
    r"->\s*tensor<([0-9a-z]+)>"
)


def _cfg(layout, token_minor=True, decoder_block="qwen3_next"):
  return pyconfig.initialize(
      [
          None,
          get_test_config_path(),
          "run_name=gdn_gated_norm_layout_test",
          "dtype=bfloat16",
          "weight_dtype=float32",
          f"decoder_block={decoder_block}",
          "attention=dot_product",
          f"base_emb_dim={_EMB}",
          "base_num_query_heads=2",
          "base_num_kv_heads=2",
          "head_dim=32",
          "gdn_num_key_heads=1",
          "gdn_num_value_heads=2",
          "gdn_key_head_dim=128",
          "gdn_value_head_dim=128",
          "gdn_conv_kernel_dim=4",
          "gdn_chunk_size=64",
          f"max_target_length={_SEQ}",
          "use_gdn_kernel=False",
          f"gdn_out_proj_token_minor={token_minor}",
          f"gdn_gated_norm_layout={layout}",
      ],
      skip_jax_distributed_system=True,
  )


def _layer(layout, seed=0, model_mode=MODEL_MODE_TRAIN):
  cfg = _cfg(layout)
  mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape([1] * len(cfg.mesh_axes)), cfg.mesh_axes)
  layer = qwen3.Qwen3NextGatedDeltaNet(
      config=cfg, mesh=mesh, model_mode=model_mode, inputs_shape=(1, _SEQ, cfg.emb_dim), rngs=nnx.Rngs(seed)
  )
  return nnx.split(layer, nnx.Param, ...)


def _inputs():
  rng = np.random.default_rng(7)
  hidden = jnp.asarray(rng.standard_normal((1, _SEQ, _EMB)).astype(np.float32)).astype(jnp.bfloat16)
  dy = jnp.asarray(rng.standard_normal((1, _SEQ, _EMB)).astype(np.float32))
  return hidden, dy


def _single_layer_fns(layout, model_mode=MODEL_MODE_TRAIN):
  graphdef, params, rest = _layer(layout, model_mode=model_mode)
  hidden, dy = _inputs()

  def loss(p, h):
    out = nnx.merge(graphdef, p, rest)(h, model_mode=model_mode)[0]
    return jnp.sum(out.astype(jnp.float32) * dy)

  return loss, params, hidden


def _scan_fns(layout):
  """Two layers applied by a remat'd lax.scan: a forward body, and a backward body that replays it."""
  graphdef, p0, rest = _layer(layout, seed=0)
  _, p1, _ = _layer(layout, seed=1)
  stacked = jax.tree.map(lambda a, b: jnp.stack([a, b]), p0, p1)
  hidden, dy = _inputs()

  def loss(p, h):
    @functools.partial(jax.checkpoint, policy=jax.checkpoint_policies.nothing_saveable)
    def step(carry, layer_params):
      out = nnx.merge(graphdef, layer_params, rest)(carry, model_mode=MODEL_MODE_TRAIN)[0]
      return (carry + out).astype(carry.dtype), None

    out, _ = jax.lax.scan(step, h, p)
    return jnp.sum(out.astype(jnp.float32) * dy)

  return loss, stacked, hidden


_SCALE_LEAF = "['norm']['rms_norm']['scale'].value"  # the gated-norm scale gradient (a sum over the tokens)
_SCALE_RTOL = 1e-6


def _run(loss, params, hidden):
  """[(name, array)] of the loss, the input gradient and every parameter gradient."""
  val, (gp, gh) = jax.jit(jax.value_and_grad(loss, argnums=(0, 1)))(params, hidden)
  named = [("loss", val), ("d_input", gh)]
  named += [(jax.tree_util.keystr(k), v) for k, v in jax.tree_util.tree_leaves_with_path(gp)]
  return [(n, np.asarray(jax.device_get(v))) for n, v in named]


def _constraints(text):
  """[(minor_to_major, result tensor type)] of every LayoutConstraint custom call in a StableHLO module."""
  return [(m.group(1), m.group(2)) for m in _LAYOUT_RE.finditer(text)]


def _counts(loss, params, hidden):
  fwd = jax.jit(loss).lower(params, hidden).as_text()
  vg = jax.jit(jax.value_and_grad(loss, argnums=(0, 1))).lower(params, hidden).as_text()
  out = {}
  for name, text in (("fwd", fwd), ("vg", vg)):
    cs = _constraints(text)
    out[name] = {
        "row_major_4d": sum(1 for lay, ty in cs if lay == _ROW_MAJOR_4D and ty == _NORM_SHAPE),
        "token_minor_4d": sum(1 for lay, ty in cs if lay == _TOKEN_MINOR_4D and ty == _NORM_SHAPE),
        "other_4d_norm": sum(1 for lay, ty in cs if lay not in (_ROW_MAJOR_4D, _TOKEN_MINOR_4D) and ty == _NORM_SHAPE),
        "token_minor_3d": sum(1 for lay, _ in cs if lay == _TOKEN_MINOR_3D),
        "total": len(cs),
        "raw_count": text.count("@LayoutConstraint"),
    }
  return out


def _expected(layout, structure):
  """4-D constraints ((row-major fwd, vg), (token-minor fwd, vg)) in the forward and value-and-grad programs."""
  fwd_evals = {"single": 1, "scan": 1}[structure]  # forward program: one forward body
  vg_fwd_evals = {"single": 1, "scan": 2}[structure]  # value-and-grad: forward + (scan) the remat replay
  bwd = 1  # one backward per layer application site
  return {
      "auto": ((0, 0), (0, 0)),
      "natural": ((fwd_evals, vg_fwd_evals), (0, 0)),
      "natural_fwd_bwd": ((fwd_evals, vg_fwd_evals + bwd), (0, 0)),
      "token_minor": ((0, 0), (fwd_evals, vg_fwd_evals)),
  }[layout]


def check_layout(layout, structure, counts):
  """The constraint census of one program pair against `_expected`; returns (ok, detail)."""
  (want_fwd, want_vg), want_t4 = _expected(layout, structure)
  got_fwd, got_vg = counts["fwd"]["row_major_4d"], counts["vg"]["row_major_4d"]
  got_t4 = (counts["fwd"]["token_minor_4d"], counts["vg"]["token_minor_4d"])
  stray = counts["fwd"]["other_4d_norm"] + counts["vg"]["other_4d_norm"]
  # The out_proj token-minor pin is forward-only (identity cotangent): one per forward evaluation, untouched by the key.
  want_tm = {"single": (1, 1), "scan": (1, 2)}[structure]
  got_tm = (counts["fwd"]["token_minor_3d"], counts["vg"]["token_minor_3d"])
  parsed_all = all(counts[k]["total"] == counts[k]["raw_count"] for k in ("fwd", "vg"))
  ok = (got_fwd, got_vg) == (want_fwd, want_vg) and got_t4 == want_t4 and stray == 0 and got_tm == want_tm and parsed_all
  return ok, (
      f"row-major 4-D fwd/vg {got_fwd}/{got_vg} (want {want_fwd}/{want_vg}), token-minor 4-D {got_t4} (want {want_t4}), "
      f"other-layout 4-D {stray} (want 0), "
      f"token-minor 3-D {got_tm} (want {want_tm}), parsed {counts['fwd']['total']}+{counts['vg']['total']} of "
      f"{counts['fwd']['raw_count']}+{counts['vg']['raw_count']}"
  )


def check_bits(ref, got, layout):
  """Bit-identity of every array; for 'token_minor' the gated-norm scale gradient only to a relative _SCALE_RTOL."""
  if [n for n, _ in ref] != [n for n, _ in got]:
    return False, "array names differ"
  bad, tol_used = [], ""
  for (n, a), (_, b) in zip(ref, got):
    if a.tobytes() == b.tobytes():
      continue
    if layout == "token_minor" and n.endswith(_SCALE_LEAF):
      rel = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))) / max(float(np.max(np.abs(a))), 1e-30))
      tol_used = f"; scale grad rel {rel:.2e}"
      if rel <= _SCALE_RTOL:
        continue
    bad.append(n)
  return not bad, f"{len(got)} arrays (loss, d_input, {len(got) - 2} param grads), {len(bad)} differ {bad}{tol_used}"


# Planted variants of `_pin_gdn_gated_norm_layout(x, mode)`: each must fail at least one check.
PLANTED = {
    "N1_pin_dropped": lambda x, mode: x,
    "N2_wrong_perm_0132": lambda x, mode: moe._forward_layout_constraint(x, (0, 1, 3, 2)),  # pylint: disable=protected-access
    "N3_heads_reversed": lambda x, mode: moe.forward_row_major(x[:, :, ::-1, :]),
    "N4_no_cotangent_pin": lambda x, mode: moe.forward_row_major(x),
    "N5_tm_pin_dropped": lambda x, mode: x,
    "N6_tm_wrong_perm_0213": lambda x, mode: moe._forward_layout_constraint(x, (0, 2, 1, 3)),  # pylint: disable=protected-access
    "N7_tm_heads_reversed": lambda x, mode: moe._forward_layout_constraint(  # pylint: disable=protected-access
        x[:, :, ::-1, :], (0, 2, 3, 1)
    ),
}
PLANTED_MODE = {"N1_pin_dropped": "natural", "N2_wrong_perm_0132": "natural", "N3_heads_reversed": "natural",
                "N4_no_cotangent_pin": "natural_fwd_bwd", "N5_tm_pin_dropped": "token_minor",
                "N6_tm_wrong_perm_0213": "token_minor", "N7_tm_heads_reversed": "token_minor"}


def run_all(verbose=True):
  """Every real and planted case; returns the list of (case, kind, ok) and whether the whole check passed."""
  rows, all_good = [], True

  def report(case, kind, ok, detail, expect_ok):
    nonlocal all_good
    good = ok == expect_ok
    all_good &= good
    rows.append((case, kind, ok))
    if verbose:
      tag = "PASS" if ok else "FAIL"
      print(f"{case:28s} {kind:6s} [{tag}] {'(expected)' if good else '(UNEXPECTED)'} {detail}", flush=True)

  for structure, make in (("single", _single_layer_fns), ("scan", _scan_fns)):
    ref = None
    for layout in ("auto", "natural", "natural_fwd_bwd", "token_minor"):
      loss, params, hidden = make(layout)
      bits = _run(loss, params, hidden)
      if ref is None:
        ref = bits
      ok, detail = check_bits(ref, bits, layout)
      report(f"{structure}/{layout}", "bits", ok, detail, True)
      if layout == "token_minor":
        # The relaxed branch is scoped and bounded: the same token_minor result must FAIL the exact check of the other
        # modes, and a scale gradient moved by 1e-5 relative (10x the tolerance) must FAIL the token_minor check.
        ok8, d8 = check_bits(ref, bits, "natural")
        report(f"{structure}/N8_tol_only_for_token_minor", "gate", ok8, d8, False)
        moved = [(n, a * np.float32(1 + 1e-5) if n.endswith(_SCALE_LEAF) else a) for n, a in bits]
        ok9, d9 = check_bits(ref, moved, "token_minor")
        report(f"{structure}/N9_scale_beyond_tol", "gate", ok9, d9, False)
      ok, detail = check_layout(layout, structure, _counts(loss, params, hidden))
      report(f"{structure}/{layout}", "layout", ok, detail, True)
    for name, fn in PLANTED.items():
      layout = PLANTED_MODE[name]
      with mock.patch.object(qwen3, "_pin_gdn_gated_norm_layout", fn):
        loss, params, hidden = make(layout)
        bits = _run(loss, params, hidden)
        ok_bits, d_bits = check_bits(ref, bits, layout)
        ok_lay, d_lay = check_layout(layout, structure, _counts(loss, params, hidden))
      # A planted variant must FAIL the combined gate (bits AND layout).
      report(f"{structure}/{name}", "gate", ok_bits and ok_lay, f"bits: {d_bits}; layout: {d_lay}", False)

  # Training path only: no row-major 4-D pin when the same config runs in prefill. Prefill writes the layer's cache,
  # so the non-parameter state goes in as a jit argument (a closed-over cache cannot be mutated inside the trace).
  for layout in ("natural", "token_minor"):
    graphdef, params, rest = _layer(layout, model_mode=MODEL_MODE_PREFILL)
    hidden, _ = _inputs()
    text = (
        jax.jit(lambda p, h, r, g=graphdef: nnx.merge(g, p, r)(h, model_mode=MODEL_MODE_PREFILL)[0])
        .lower(params, hidden, rest)
        .as_text()
    )
    n = sum(1 for lay, ty in _constraints(text) if ty == _NORM_SHAPE)
    report(f"single/{layout}/prefill", "layout", n == 0, f"4-D (B, S, H_v, D_v) constraints {n} (want 0)", True)
  return rows, all_good


class GdnGatedNormLayoutConfigTest(absltest.TestCase):
  """Config construction: default, accepted and refused settings."""

  def test_default_is_auto(self):
    self.assertEqual(config_types.Qwen3Next.model_fields["gdn_gated_norm_layout"].default, "auto")
    self.assertEqual(_cfg("auto").gdn_gated_norm_layout, "auto")

  def test_accepted_values(self):
    for layout in ("natural", "natural_fwd_bwd", "token_minor"):
      self.assertEqual(_cfg(layout).gdn_gated_norm_layout, layout)

  def test_refused(self):
    with self.assertRaises(pydantic.ValidationError):
      _cfg("naturel")
    with self.assertRaisesRegex(ValueError, "requires gdn_out_proj_token_minor=True"):
      _cfg("natural", token_minor=False)
    with self.assertRaisesRegex(ValueError, "gdn_gated_norm_layout='natural' requires a decoder with Gated DeltaNet"):
      _cfg("natural", token_minor=False, decoder_block="llama2")
    with self.assertRaises(Exception) as ctx:  # extra='forbid': a typo'd key is refused, not ignored
      pyconfig.initialize(
          [None, get_test_config_path(), "run_name=gdn_gated_norm_layout_test", "gdn_gated_norm_layot=natural"],
          skip_jax_distributed_system=True,
      )
    self.assertIn("gdn_gated_norm_layot", str(ctx.exception))


class GdnGatedNormLayoutModuleTest(absltest.TestCase):
  """Bit-identity and the LayoutConstraint census through the real module, with planted negative controls."""

  def test_real_and_planted_cases(self):
    rows, all_good = run_all(verbose=True)
    self.assertTrue(all_good, rows)

  def test_banner_once_per_mode(self):
    qwen3._GDN_GATED_NORM_LAYOUT_LOGGED.clear()  # pylint: disable=protected-access
    with mock.patch.object(qwen3.max_logging, "log") as log:
      for _ in range(2):
        loss, params, hidden = _single_layer_fns("natural_fwd_bwd")
        jax.jit(loss).lower(params, hidden)
    msgs = [c.args[0] for c in log.call_args_list if "[gdn_gated_norm_layout]" in c.args[0]]
    self.assertEqual(msgs, ["[gdn_gated_norm_layout] mode=natural_fwd_bwd: GDN gated-norm output pinned row-major in "
                            "the forward and on its cotangent"])


if __name__ == "__main__":
  if "--harness" in sys.argv:
    _, ok = run_all(verbose=True)
    print(_PASS_TOKEN if ok else "GDN_GATED_NORM_LAYOUT_CHECKS_FAILED", flush=True)
    sys.exit(0 if ok else 1)
  absltest.main()
