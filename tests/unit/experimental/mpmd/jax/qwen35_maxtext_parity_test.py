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

"""The Qwen3.5 port against MaxText's own `qwen3_5` model, on CPU at the tiny shape.

    JAX_PLATFORMS=cpu python3 -m pytest -q tests/unit/experimental/mpmd/jax/qwen35_maxtext_parity_test.py

MaxText builds on CPU, about six minutes in all: the classes are marked scheduled_only (MaxText's scheduled CI
runs them).  Every check passes against the MaxText around it (jax 0.12.0.1), and passed against MaxText
dcafaa2ef (2026-08-29, jax 0.10.2) and b33d0f687 (2026-09-25; jax 0.11.1 with flax 0.12.9, and jax 0.11.2 with
flax 0.12.10 and tokamax 0.0.16).  Those two round their RMSNorms differently --
9a343b3d8 (2026-09-10) moved the zero-centred norms' (1 + s) multiply and the
GDN gated norm to fp32 -- and the port has both
(`Qwen35Config.norm_numerics`): every check reads the rounding off the MaxText
it imports (`_MaxText.norm_numerics`) and sets it.  The port's Gated DeltaNet
is MaxText's as of 4b5b24c92 (2026-11-24; jax 0.12.0.1, flax 0.12.12): padding
zeroed before the conv, the conv window and the recurrent state restarting at
every packed document under its `packing` (on by default), the gates in fp32,
and the WY matrix from k k^T at the key heads, inverted by block doubling;
`qc.from_maxtext` reads MaxText's two options among them, `packing` and
`gdn_chunk_inverse` (`qc.maxtext_gdn_fields`).  These fp32 checks need
XLA:CPU's YNNPACK fusions on, jaxlib's default -- no
`--xla_cpu_experimental_ynn_fusion_type=` in XLA_FLAGS (with them off the ragged-MoE
logits are 1.6e-5 from MaxText's on jax 0.11.1, and 2e-6 to 5e-5 on jax
0.12.0.1 against 4b5b24c92, where with them on they are bit for bit; the
dense MoE's are bit for bit either way).

MaxText initialises every zero-centred norm scale to 0, the GDN gated norm's to
1 and dt_bias to 1: at those values any f(s) with f(0) = 1, and any swap of two
such leaves, gives the same forward.  Every check below first moves them off
(`perturb`), on MaxText's model and the port alike.

fp32 (MaxText `dtype=float32 matmul_precision=highest`):

  * parameter tree: MaxText's NNX state with the block axis moved to 0
    (`convert.from_nnx_state`) is `model.param_avals(from_maxtext(cfg))`;
  * forward logits, max rel-to-max <= 1e-6 (bit-exact in practice): S = 64,
    S = 50 (the GDN padding path), packed + right-padded segment ids, S = 33
    packed documents with per-document positions and padding at the start,
    middle and end of rows (under MaxText's packing the GDN restarts at each
    document), and the dense MoE (`sparse_matmul=false`);
  * gradients of the loss on two of those batches, every leaf within
    max(1e-5, 4 x floor), the floor MEASURED on the same weights and batch as
    the port's own scanned vs unrolled (`scan_layers=False`) gradients.  A_log's
    gradient is a sum with heavy cancellation, so fp32 reassociation between
    two compiled programs of the same model moves it by 1e-5 to 3e-5
    rel-to-max (the custom-VJP WY inverse's own share is ~4e-7); the packed
    GDN of MaxText from fb42453e1 on, which these batches run, moves the
    floor to ~1e-4 (9e-5 / 1.2e-4 on the two batches at 4b5b24c92, its worst
    leaves in_proj_qkvz's and A_log's).

bf16 (`dtype=bfloat16 matmul_precision=default`), the port-correctness
criterion: forward logits BIT-EXACT, in a subprocess started with
`--xla_allow_excess_precision=false` (otherwise fusion decides which bf16
roundings survive).  In fp32 every bf16 rounding the port copies from MaxText
is a no-op cast, so this is the check that holds them: the GDN gates g in fp32,
its WY matrix from a bf16 k k^T times fp32 beta and decay, the packed conv in
fp32 with the fp32 kernel and one rounding, the WY factors U and W rounded to bf16, the
l2 norm in bf16 with its eps inside the sqrt and the GDN q scale applied after
it in bf16, silu in fp32 then rounded (after the conv, and on the gated norm's
gate), beta and the gates' sigmoids in bf16, the RMSNorms' rounding, the
mRoPE's fp32 cos/sin, the router's fp32 softmax then bf16 renormalisation.
Two of those cannot show at
the fp32 checks' shape: Dk^-1/2 is a power of two at Dk = 16 (as at any power
of 4), so every order of the q-scale rounding gives the same bits, and with
top-2 routing the two bf16-rounded softmax weights already sum to 1, so the
renormalisation is an identity.  The bf16 build therefore runs at Dk = 32 and
top-4 (`BF16`), and the subprocess also runs the port with each of those
roundings moved (`ROUNDING_MUTANTS`, and the packed conv's kernel rounded to
bf16), with each of the GDN's November 2026 changes undone (`GDN_MUTANTS`; the
padding zeroed before the conv on a second MaxText build, packing=false, where
the plain conv runs) and with gdn_packing flipped: every mutant must change the
logits.  gdn_chunk_inverse is no mutant:
the two inverses part in fp32's last bits, which no bf16 logit at these
shapes sees; `GdnAgainstMaxText` holds it, with the rest of the GDN's pieces,
bit for bit against MaxText's own functions in fp32 and bf16.
(The attention's q scale, head_dim^-1/2, is a power of two here and in both
real models, 256^-1/2.)  Bit-exact with MaxText's ragged_dot fallback
(`megablox=false`; the port's gmm_fp32_accum=False) and with the dense MoE.
Under MaxText's default megablox `gmm` the routed experts' grouped matmul
accumulates in another order and some rows come out 1 bf16 ulp apart (2e-3 /
4e-3 rel-to-max on the S = 64 / packed logits), so the check pins
megablox=false.  bf16 GRADIENTS of this model are not reproducible across
program structures at a random init (two jitted programs of the same model
disagree at O(1) on some leaves: top-k ties, then chaos), so they are no
parity target.
"""

import contextlib
import inspect
import json
import os
import subprocess
import sys
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maxtext.experimental.mpmd.jax import testing
from maxtext.experimental.mpmd.jax.qwen35 import _ops
from maxtext.experimental.mpmd.jax.qwen35 import config as qc
from maxtext.experimental.mpmd.jax.qwen35 import convert
from maxtext.experimental.mpmd.jax.qwen35 import gdn
from maxtext.experimental.mpmd.jax.qwen35 import layers
from maxtext.experimental.mpmd.jax.qwen35 import model as qm
from maxtext.experimental.mpmd.jax.qwen35 import moe
from maxtext.experimental.mpmd.jax.qwen35 import train as qt

ROOT = testing.import_root()

# MaxText overrides for the tiny shape of qwen3.5-397b-a17b: 3:1 GDN:attention,
# GQA 2:1, 4 value heads per GDN key head, partial RoPE (rotary 16, mrope
# sections summing to 8), top-2 of 8 experts + the gated shared expert.
TINY = dict(
    base_emb_dim=64,
    base_num_decoder_layers=8,
    base_num_query_heads=4,
    base_num_kv_heads=2,
    head_dim=64,
    mrope_section=[3, 3, 2],
    base_mlp_dim=32,
    base_moe_mlp_dim=32,
    num_experts=8,
    num_experts_per_tok=2,
    gdn_num_key_heads=2,
    gdn_num_value_heads=8,
    gdn_key_head_dim=16,
    gdn_value_head_dim=16,
    gdn_chunk_size=16,
    vocab_size=256,
    max_target_length=64,
    max_prefill_predict_length=16,
    per_device_batch_size=2.0,
    dtype="float32",
    matmul_precision="highest",
)
# The bf16 run: MaxText's training numerics, with the ragged_dot grouped matmul
# rather than megablox, at a GDN key head dim whose rsqrt is not a power of two
# and a top-k whose renormalisation is not an identity (see the module docstring).
BF16 = dict(dtype="bfloat16", matmul_precision="default", megablox=False, gdn_key_head_dim=32, num_experts_per_tok=4)
# One-line port edits that each move one bf16 rounding and keep the math: the
# bf16 check must see every one (at Dk = 16, top-2 it saw none of these).
# (module, function, old source, new source)
ROUNDING_MUTANTS = {
    "GDN q scale applied in fp32": (
        gdn,
        "chunk_gated_delta_rule",
        "query = query * scale",
        "query = (query.astype(jnp.float32) * jax.lax.rsqrt(jnp.float32(query.shape[-1]))" ").astype(compute_dtype)",
    ),
    "router renormalisation in fp32": (
        moe,
        "route",
        "weights = jax.nn.softmax(weights.astype(jnp.float32), axis=-1).astype(config.dtype)\n"
        "  if config.norm_topk_prob:\n"
        "    weights = weights / weights.sum(axis=-1, keepdims=True)\n",
        "weights = jax.nn.softmax(weights.astype(jnp.float32), axis=-1)\n"
        "  if config.norm_topk_prob:\n"
        "    weights = weights / weights.sum(axis=-1, keepdims=True)\n"
        "  weights = weights.astype(config.dtype)\n",
    ),
    "no router renormalisation": (moe, "route", "if config.norm_topk_prob:", "if False:"),
}
# The packed conv's rounding (gdn_packing): the fp32 kernel, not rounded to `dtype` as the plain conv's is.
PACKED_CONV_MUTANT = (
    "packed conv with its kernel rounded to dtype",
    (
        gdn,
        "segmented_causal_conv",
        "w = _ops.small_weight(kernel, jnp.float32, config)",
        "w = _ops.small_weight(kernel, config.dtype, config).astype(jnp.float32)",
    ),
)
# The GDN's November 2026 changes, each undone by a port edit (as ROUNDING_MUTANTS are): every edit must change the
# bf16 logits.  The padding zeroed before the conv shows only where the plain conv runs (the packed one zeroes it
# itself), so its edit runs on a second MaxText build, packing=false (`PACKING_OFF`).  gdn_packing, a MaxText option,
# is flipped instead (`PACKING_MUTANT`).  Not gdn_chunk_inverse: the two inverses part in fp32's last bits, which no
# bf16 logit of these shapes sees (chunks of 16, 32 and 64 alike); GdnAgainstMaxText's fp32 checks hold it.
GDN_MUTANTS = {
    "GDN gates in dtype (before fb42453e1)": (
        gdn,
        "gated_delta_net",
        "gate_dtype = jnp.float32",
        "gate_dtype = config.dtype",
    ),
    "WY matrix from (k beta) k^T at the value heads (before e15aaeb00)": (
        gdn,
        "chunk_gated_delta_rule",
        "kk = heads(jnp.matmul(k_c, k_c.swapaxes(-1, -2), precision=_HI).astype(jnp.float32), 2)\n"
        "  g_diff = jnp.where(strict, g_cum[..., :, None] - g_cum[..., None, :], _NEG)\n"
        "  smat = jnp.where(strict, kk * (beta_c.astype(jnp.float32)[..., :, None] * jnp.exp(g_diff)), 0.0)\n",
        "kk = jnp.matmul(heads(k_c, 2) * beta_c[..., None], heads(k_c, 2).swapaxes(-1, -2), precision=_HI)\n"
        "  g_diff = jnp.where(strict, g_cum[..., :, None] - g_cum[..., None, :], _NEG)\n"
        "  smat = jnp.where(strict, kk.astype(jnp.float32) * jnp.exp(g_diff), 0.0)\n",
    ),
}
PACKING_OFF = "ragged, packing=false"
MASK_MUTANT = (
    f"padding not zeroed before the conv (before 79c427d44; {PACKING_OFF})",
    (gdn, "gated_delta_net", "if segment_ids is not None:\n    qkv = jnp.where", "if False:\n    qkv = jnp.where"),
)
PACKING_MUTANT = "gdn_packing flipped"
NO_EXCESS_PRECISION = "--xla_allow_excess_precision=false"
REPORT = "QWEN35_BF16_REPORT "
B, S = 2, 64


def _import_maxtext():
  """MaxText's modules the checks use, imported when a check first needs them (not at collection)."""
  import maxtext  # pylint: disable=g-import-not-at-top
  from maxtext.common import common_types  # pylint: disable=g-import-not-at-top
  from maxtext.configs import pyconfig  # pylint: disable=g-import-not-at-top
  from maxtext.utils import maxtext_utils  # pylint: disable=g-import-not-at-top
  from maxtext.utils import model_creation_utils  # pylint: disable=g-import-not-at-top

  return maxtext, common_types, pyconfig, maxtext_utils, model_creation_utils


# The pyconfig overrides of every MaxText build here: no run directory, checkpoint or jax.distributed, synthetic
# data, the yml's model dims overridable.
BASE_OVERRIDES = dict(
    run_name="parity",
    enable_checkpointing=False,
    dataset_type="synthetic",
    skip_jax_distributed_system=True,
    override_model_config=True,
)


def perturb(params, seed=7):
  """Moves the leaves MaxText initialises to constants off them: zero-centred
  norm scales (0) to N(0, 0.3), the GDN gated norm's (1) to 1 + N(0, 0.3),
  dt_bias (1) to 1 + N(0, 0.5).  `params` is any pytree with MaxText's names."""
  rng = np.random.default_rng(seed)

  def leaf(path, x):
    name = jax.tree_util.keystr(path)
    if "['scale']" in name:
      base, sd = (1.0, 0.3) if "['rms_norm']" in name else (0.0, 0.3)
    elif "['dt_bias']" in name:
      base, sd = 1.0, 0.5
    else:
      return x
    return jnp.asarray(np.asarray(base + sd * rng.standard_normal(x.shape), x.dtype))

  return jax.tree_util.tree_map_with_path(leaf, params)


# MaxText's model the checks build by default: Qwen3.5's yml, the text mRoPE (use_mrope true).  QWEN38 is
# Qwen3.8-2.4T's, which leaves use_mrope at base.yml's false: the partial RoPE (attention.partial_rope).
QWEN35, QWEN38 = "qwen3.5-397b-a17b", "qwen3.8-2.4t-a95b"
# The bf16 check's mutant for that model: the mRoPE where MaxText runs the partial RoPE.
MROPE_ON_QWEN38 = f"{QWEN38} under the text mRoPE (use_mrope=True)"


def maxtext_config(model=QWEN35, **extra):
  """MaxText's pyconfig of `model` at the tiny shape (TINY, `extra` on top), as the checks build it."""
  maxtext, _, pyconfig, _, _ = _import_maxtext()
  base_yml = os.path.join(os.path.dirname(maxtext.__file__), "configs", "base.yml")
  overrides = dict(BASE_OVERRIDES)
  overrides.update(TINY, **extra)
  return pyconfig.initialize([sys.argv[0], base_yml, f"model_name={model}"], **overrides)


class _MaxText:
  """One MaxText build per model and set of overrides (perturbed), cached for the module."""

  _cache = {}

  def __new__(cls, model=QWEN35, **extra):
    key = (model,) + tuple(sorted(extra.items()))
    if key in cls._cache:
      return cls._cache[key]
    _, common_types, _, _, model_creation_utils = _import_maxtext()
    from flax import linen as nn  # pylint: disable=g-import-not-at-top
    from flax import nnx  # pylint: disable=g-import-not-at-top

    self = object.__new__(cls)
    self.nn, self.nnx, self.mode = nn, nnx, common_types.MODEL_MODE_TRAIN
    self.cfg = maxtext_config(model, **extra)
    # One device: even with mesh=None MaxText builds its own mesh over `devices`
    # (fsdp = 8 on 8 fake CPU devices), and the sharded program
    # reassociates fp32 sums (7e-6 on the S = 50 logits).  The port is the same
    # program on any device count.
    with nn.logical_axis_rules(self.cfg.logical_axis_rules):
      self.model = model_creation_utils.get_nnx_create_model_fn(
          self.cfg,
          mesh=None,
          devices=jax.devices()[:1],
          model_mode=self.mode,
          rng_key=jax.random.PRNGKey(self.cfg.init_weights_seed),
      )()
    nnx.update(self.model, perturb(nnx.split(self.model, nnx.Param, ...)[1]))
    self.port_cfg = qc.from_maxtext(self.cfg)
    self.params = convert.from_nnx_state(nnx.split(self.model)[1], self.cfg.param_scan_axis, config=self.port_cfg)
    self.norms = self.norm_numerics()
    if self.norms in qc.NORM_NUMERICS:
      self.port_cfg = self.port_cfg.replace(norm_numerics=self.norms)
    cls._cache[key] = self
    return self

  def logits(self, tokens, positions, segment_ids, model=None):
    with self.nn.logical_axis_rules(self.cfg.logical_axis_rules):
      out = (model or self.model)(
          decoder_input_tokens=tokens,
          decoder_positions=positions,
          decoder_segment_ids=segment_ids,
          enable_dropout=False,
          model_mode=self.mode,
      )
    if model is None:
      # What the call sowed (activation statistics) would otherwise ride along
      # in every later `nnx.split(..., nnx.Param, ...)`'s rest, and flax 0.12.9
      # refuses to update a traced copy of it from inside MaxText's layer scan.
      self.nnx.pop(self.model, self.nnx.Intermediate)
    return out

  def norm_numerics(self):
    """The `Qwen35Config.norm_numerics` this MaxText's RMSNorms round as, read
    off its own modules: the final (zero-centred) norm, and layer 0's GDN gated
    norm with block 0 sliced out of its scanned scale.  Each is compared
    bitwise with the port's norm under either setting (both eagerly, so every
    rounding happens); 'other' when one matches neither, 'mixed' when the two
    want different settings.  In fp32 both settings are one computation, and
    the config's own setting is kept."""
    cfg, nnx = self.port_cfg, self.nnx
    rng = np.random.default_rng(5)
    x = jnp.asarray(rng.standard_normal((2, 7, cfg.emb_dim)), cfg.dtype)
    o, z = (jnp.asarray(rng.standard_normal((2, 7, 3, cfg.gdn_value_head_dim)), cfg.dtype) for _ in range(2))
    axis = self.cfg.param_scan_axis
    blocks = self.model.decoder.layers
    # layer 0 of block 0: the first of MaxText's nested scan's `local_layers`, whose leaves carry the local layers'
    # axis right after the block axis (convert.from_nnx_state)
    graphdef, state = nnx.split(blocks.local_layers.attention.norm)
    first = lambda a: jnp.take(a, 0, axis=axis) if getattr(a, "ndim", 0) > axis else a
    gated = nnx.merge(graphdef, jax.tree.map(lambda a: first(first(a)), state))
    zc_scale = self.params["decoder"]["decoder_norm"]["scale"]
    g_scale = self.params["decoder"]["layers"]["layer_0"]["attention"]["norm"]["rms_norm"]["scale"][0]
    sites = (
        (self.model.decoder.decoder_norm(x), lambda c: layers.rms_norm({"scale": zc_scale}, x, c)),
        (gated(o, z), lambda c: gdn.gated_rms_norm({"rms_norm": {"scale": g_scale}}, o, z, c)),
    )
    match = set(qc.NORM_NUMERICS)
    for got, port in sites:
      same = set()
      for n in qc.NORM_NUMERICS:
        want = port(cfg.replace(norm_numerics=n))
        if got.dtype == want.dtype and bool(jnp.array_equal(got, want)):
          same.add(n)
      if not same:
        return "other"
      match &= same
    if not match:
      return "mixed"
    return cfg.norm_numerics if len(match) > 1 else match.pop()


def batch(seq=S, packed=False):
  rng = np.random.default_rng(1234)
  tokens = jnp.asarray(rng.integers(0, 256, size=(B, seq)), jnp.int32)
  positions = jnp.broadcast_to(jnp.arange(seq, dtype=jnp.int32), (B, seq))
  seg = np.ones((B, seq), np.int32)
  if packed:  # row 0: two documents then 13 padding tokens (segment id 0)
    seg[0, 40:] = 2
    seg[0, seq - 13 :] = 0
  return tokens, positions, jnp.asarray(seg)


def documents(seq=33):
  """S = 33, not a multiple of the GDN chunk (16): packed documents with
  positions restarting at each, and padding (segment 0, position 0) at the end
  of row 0 and at the start and in the middle of row 1."""
  rng = np.random.default_rng(99)
  tokens = jnp.asarray(rng.integers(0, 256, size=(B, seq)), jnp.int32)
  seg, pos = np.zeros((B, seq), np.int32), np.zeros((B, seq), np.int32)
  for row, spans in ((0, ((0, 13), (13, 27))), (1, ((4, 20), (23, seq)))):
    for doc, (a, b) in enumerate(spans):
      seg[row, a:b], pos[row, a:b] = doc + 1, np.arange(b - a)
  return tokens, jnp.asarray(pos), jnp.asarray(seg)


def forward_cases():
  return (
      ("S=64", batch()),
      ("S=50 (padding path)", batch(50)),
      ("packed + padded segments", batch(packed=True)),
      ("S=33 documents", documents()),
  )


def max_rel(a, b):
  a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
  return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-30))


@contextlib.contextmanager
def _edited(module, name, old, new):
  """`module.<name>` redefined from its own source with `old` replaced by `new`,
  for the duration; the module's other functions call the edited one."""
  fn = getattr(module, name)
  src = inspect.getsource(fn)
  if old not in src:
    raise SystemExit(f"{module.__name__}.{name} no longer contains {old!r}: update its mutant")
  scope = dict(vars(module))
  exec(compile(src.replace(old, new, 1), f"<{module.__name__}.{name}, edited>", "exec"), scope)  # pylint: disable=exec-used
  setattr(module, name, scope[name])
  try:
    yield
  finally:
    setattr(module, name, fn)


def expected_mutants():
  """The mutant names `bf16_forward_report` runs."""
  return (
      list(ROUNDING_MUTANTS)
      + list(GDN_MUTANTS)
      + [PACKED_CONV_MUTANT[0], MASK_MUTANT[0], PACKING_MUTANT, MROPE_ON_QWEN38]
  )


def bf16_forward_report():
  """`python3 -m tests.unit.experimental.mpmd.jax.qwen35_maxtext_parity_test --bf16-forward`, with
  XLA_FLAGS containing --xla_allow_excess_precision=false: prints `REPORT
  {"norms": the norm numerics read off MaxText, "cases": {case: [bitwise,
  max_rel]}, "mutants": {mutant: [changed the logits, max_rel]}}`."""
  if NO_EXCESS_PRECISION not in os.environ.get("XLA_FLAGS", ""):
    raise SystemExit(f"XLA_FLAGS must contain {NO_EXCESS_PRECISION} before the backend starts")
  report = {"norms": None, "cases": {}, "mutants": {}}
  sets = [
      ("ragged, megablox=false", {}, forward_cases()),
      ("dense MoE", dict(sparse_matmul=False), forward_cases()[-1:]),
      (PACKING_OFF, dict(packing=False), forward_cases()),
  ]
  sets.append((f"{QWEN38} (partial RoPE)", dict(model=QWEN38), forward_cases()))
  edits = dict(ROUNDING_MUTANTS, **dict([PACKED_CONV_MUTANT]), **GDN_MUTANTS)
  for moe_impl, extra, cases in sets:
    mt = _MaxText(**BF16, **extra)
    cfg = mt.port_cfg
    if (
        jnp.dtype(cfg.dtype) != jnp.bfloat16
        or cfg.gmm_fp32_accum
        or (cfg.gdn_key_head_dim, cfg.num_experts_per_tok) != (32, 4)
    ):
      raise SystemExit(f"not the bf16 configuration: {cfg}")
    if report["norms"] not in (None, mt.norms):
      raise SystemExit(f"two builds of one MaxText read as {report['norms']} and {mt.norms}")
    report["norms"] = mt.norms
    want = {}
    for name, (tok, pos, seg) in cases:
      want[name] = mt.logits(tok, pos, seg)

    def run(c=cfg):
      fwd = jax.jit(lambda p, t, q, s: qm.forward(p, t, q, s, c))
      out = {}
      for name, (tok, pos, seg) in cases:
        got = fwd(mt.params, tok, pos, seg)
        if (got.shape, got.dtype) != (want[name].shape, want[name].dtype):
          raise SystemExit(f"{name}: {got.shape} {got.dtype} vs MaxText {want[name].shape} " f"{want[name].dtype}")
        out[name] = [bool(jnp.array_equal(got, want[name])), max_rel(got, want[name])]
      return out

    for name, res in run().items():
      report["cases"][f"{moe_impl}: {name}"] = res
    if not extra:
      for mutant, edit in edits.items():
        with _edited(*edit):
          res = run().values()
        report["mutants"][mutant] = [not all(b for b, _ in res), max(r for _, r in res)]
      res = run(cfg.replace(gdn_packing=not cfg.gdn_packing)).values()
      report["mutants"][PACKING_MUTANT] = [not all(b for b, _ in res), max(r for _, r in res)]
    if extra.get("packing") is False:
      if cfg.gdn_packing:
        raise SystemExit(f"packing=false read as {cfg}")
      with _edited(*MASK_MUTANT[1]):
        res = run().values()
      report["mutants"][MASK_MUTANT[0]] = [not all(b for b, _ in res), max(r for _, r in res)]
    if extra.get("model") == QWEN38:
      # the port before use_mrope: Qwen3.8 under the Qwen3.5 ymls' text mRoPE
      res = run(cfg.replace(use_mrope=True)).values()
      report["mutants"][MROPE_ON_QWEN38] = [not all(b for b, _ in res), max(r for _, r in res)]
  print(REPORT + json.dumps(report), flush=True)


@pytest.mark.scheduled_only
class AgainstMaxText(unittest.TestCase):

  def test_parameter_tree(self):
    mt = _MaxText()
    want = {
        jax.tree_util.keystr(p): (tuple(a.shape), a.dtype)
        for p, a in jax.tree_util.tree_flatten_with_path(qm.param_avals(mt.port_cfg))[0]
    }
    got = {
        jax.tree_util.keystr(p): (tuple(a.shape), a.dtype) for p, a in jax.tree_util.tree_flatten_with_path(mt.params)[0]
    }
    self.assertEqual(len(got), 70)
    self.assertEqual(got, want)
    # perturbed: off MaxText's constant inits, on the model the checks run
    lay = mt.params["decoder"]["layers"]
    for name, leaf, init in (
        ("decoder_norm", mt.params["decoder"]["decoder_norm"]["scale"], 0.0),
        ("query_norm", lay["layer_3"]["attention"]["attention"]["query_norm"]["scale"], 0.0),
        ("gated norm", lay["layer_0"]["attention"]["norm"]["rms_norm"]["scale"], 1.0),
        ("dt_bias", lay["layer_0"]["attention"]["dt_bias"], 1.0),
    ):
      self.assertGreater(float(np.min(np.abs(np.asarray(leaf) - init))), 0.0, name)

  def test_from_maxtext(self):
    mt = _MaxText()
    c = mt.port_cfg
    self.assertEqual(
        c,
        qc.tiny(
            dtype=jnp.float32,
            matmul_precision="highest",
            sa_block_q_dkv=c.sa_block_q_dkv,
            sa_block_kv_dkv=c.sa_block_kv_dkv,
            sa_block_kv_dkv_compute=c.sa_block_kv_dkv_compute,
            sa_use_base2_exp=bool(mt.cfg.sa_use_base2_exp),
            **qc.maxtext_gdn_fields(mt.cfg),
        ),
    )
    for f in ("gdn_packing", "gdn_chunk_inverse"):  # MaxText's defaults are the port's
      self.assertEqual(getattr(c, f), getattr(qc.Qwen35Config(), f), f)

  def test_from_maxtext_refuses_other_ropes(self):
    """Under use_mrope=false MaxText's gated attention rotates with YaRN or LLaMA-3.1's RoPE when its rope_type (a
    RopeType enum in MaxText's pyconfig) says so, not with the partial RoPE: from_maxtext refuses those; under
    use_mrope=true the mRoPE comes first, whatever rope_type says.  Pyconfigs only, no model."""
    self.assertIs(qc.from_maxtext(maxtext_config(use_mrope=False)).use_mrope, False)
    for rope_type in ("yarn", "llama3.1"):
      with self.subTest(rope_type=rope_type):
        with self.assertRaisesRegex(NotImplementedError, f"use_mrope=false with rope_type='{rope_type}'"):
          qc.from_maxtext(maxtext_config(use_mrope=False, rope_type=rope_type))
        self.assertIs(qc.from_maxtext(maxtext_config(rope_type=rope_type)).use_mrope, True)

  def test_forward_logits(self):
    mt = _MaxText()
    fwd = jax.jit(lambda p, t, q, s: qm.forward(p, t, q, s, mt.port_cfg))
    for name, (tok, pos, seg) in forward_cases():
      with self.subTest(name):
        want = mt.logits(tok, pos, seg)
        got = fwd(mt.params, tok, pos, seg)
        self.assertEqual((got.shape, got.dtype), (want.shape, want.dtype))
        self.assertLessEqual(max_rel(got, want), 1e-6)

  def test_forward_logits_dense_moe(self):
    mt = _MaxText(sparse_matmul=False)
    self.assertEqual(mt.port_cfg.moe_impl, "dense")
    fwd = jax.jit(lambda p, t, q, s: qm.forward(p, t, q, s, mt.port_cfg))
    for name, (tok, pos, seg) in (("packed", batch(packed=True)), ("S=33 documents", documents())):
      with self.subTest(name):
        self.assertLessEqual(max_rel(fwd(mt.params, tok, pos, seg), mt.logits(tok, pos, seg)), 1e-6)

  def test_forward_logits_bf16_bit_exact(self):
    """bf16 logits, bitwise, in a fresh process with --xla_allow_excess_precision=false,
    under the norm numerics read off this MaxText; and each rounding mutant of
    the port changes them."""
    flags = f"{os.environ.get('XLA_FLAGS', '')} {NO_EXCESS_PRECISION}".strip()
    env = dict(os.environ, JAX_PLATFORMS="cpu", XLA_FLAGS=flags, PYTHONPATH=testing.child_pythonpath(__name__))
    r = subprocess.run(
        [sys.executable, "-m", "tests.unit.experimental.mpmd.jax.qwen35_maxtext_parity_test", "--bf16-forward"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=1500,
    )
    self.assertEqual(r.returncode, 0, (r.stdout + r.stderr)[-6000:])
    lines = [l for l in r.stdout.splitlines() if l.startswith(REPORT)]
    self.assertEqual(len(lines), 1, r.stdout[-3000:])
    report = json.loads(lines[0][len(REPORT) :])
    self.assertIn(report["norms"], qc.NORM_NUMERICS, "MaxText's RMSNorms round as neither norm_numerics setting")
    names = [n for n, _ in forward_cases()]
    self.assertEqual(
        sorted(report["cases"]),
        sorted(
            [f"ragged, megablox=false: {n}" for n in names]
            + ["dense MoE: S=33 documents"]
            + [f"{PACKING_OFF}: {n}" for n in names]
            + [f"{QWEN38} (partial RoPE): {n}" for n in names]
        ),
    )
    for name, (bitwise, rel) in sorted(report["cases"].items()):
      with self.subTest(name, norms=report["norms"]):
        self.assertTrue(bitwise, f"max rel-to-max {rel:.3e}")
    self.assertEqual(sorted(report["mutants"]), sorted(expected_mutants()))
    for name, (changed, rel) in sorted(report["mutants"].items()):
      with self.subTest(mutant=name):
        self.assertTrue(changed, "the bf16 check cannot see this rounding at its shape")

  def test_gradients(self):
    mt = _MaxText()
    nnx, cfg = mt.nnx, mt.port_cfg
    nnx.pop(mt.model, nnx.Intermediate)
    graphdef, mt_params, rest = nnx.split(mt.model, nnx.Param, ...)
    port = jax.jit(
        jax.value_and_grad(
            lambda p, c, tok, pos, seg, targets: qt.loss_from_logits(
                qm.forward(p, tok, pos, seg, c), targets, seg, c.vocab_size
            )[0]
        ),
        static_argnums=1,
    )
    for name, (tok, pos, seg) in (("packed + padded", batch(packed=True)), ("S=33 documents", documents())):
      with self.subTest(name):
        targets = jnp.roll(tok, -1, axis=1)

        # `rest` is an argument, not a closure constant: flax 0.12.9 refuses to
        # update the merged module's intermediates from inside the trace otherwise.
        def mt_loss(p, r, tok=tok, pos=pos, seg=seg, targets=targets):
          logits = mt.logits(tok, pos, seg, model=nnx.merge(graphdef, p, r))
          return qt.loss_from_logits(logits, targets, seg, mt.cfg.vocab_size)[0]

        mt_l, mt_g = jax.jit(jax.value_and_grad(mt_loss))(mt_params, rest)
        l, g = port(mt.params, cfg, tok, pos, seg, targets)
        _, g_unrolled = port(mt.params, cfg.replace(scan_layers=False), tok, pos, seg, targets)
        self.assertLess(abs(float(l) - float(mt_l)), 1e-6 * abs(float(mt_l)))
        want = convert.from_nnx_state(mt_g, mt.cfg.param_scan_axis, params_only=False, config=cfg)
        self.assertEqual(jax.tree.structure(g), jax.tree.structure(want))
        floor = max(max_rel(a, b) for a, b in zip(jax.tree.leaves(g_unrolled), jax.tree.leaves(g)))
        tol = max(1e-5, 4 * floor)
        for (path, a), b in zip(jax.tree_util.tree_flatten_with_path(g)[0], jax.tree.leaves(want)):
          self.assertLessEqual(max_rel(a, b), tol, f"{jax.tree_util.keystr(path)} (floor {floor:.2e})")


@pytest.mark.scheduled_only
class GdnAgainstMaxText(unittest.TestCase):
  """The GDN's pieces against MaxText's own functions, bit for bit, in fp32 and bf16: the chunked delta rule
  (`qwen3.jax_chunk_gated_delta_rule`) with and without packed documents, at 2 key heads x 3 value heads each, both
  inverses; and its packed-document helpers (kernels/attention/gdn_packing.py) -- the canonical documents, the
  block-doubling inverse, the segmented conv (both of the port's forms against it: 'shift' to fp32 rounding) and the
  masks and per-document cumulative decay.  Where a model-level check fails, these say which piece moved."""

  @classmethod
  def setUpClass(cls):
    from maxtext.models import qwen3  # pylint: disable=g-import-not-at-top
    from maxtext.kernels.attention import gdn_packing  # pylint: disable=g-import-not-at-top

    cls.qwen3, cls.gdn_packing = qwen3, gdn_packing
    rng = np.random.default_rng(11)
    b, s, hk, h, dk, dv = 2, 37, 2, 6, 16, 8
    cls.args = (
        rng.standard_normal((b, s, hk, dk)),
        rng.standard_normal((b, s, hk, dk)),
        rng.standard_normal((b, s, h, dv)),
        -np.exp(rng.uniform(-3, 2.7, (h,))) * np.log1p(np.exp(rng.standard_normal((b, s, h)) + 1.0)),
        1 / (1 + np.exp(-rng.standard_normal((b, s, h)))),
    )
    seg = np.zeros((b, s), np.int32)
    seg[0] = [1] * 5 + [2] * 12 + [0] * 3 + [2] * 4 + [1] * 13
    seg[1] = [0] * 4 + [3] * 9 + [5] * 10 + [5] * 8 + [0] * 6
    cls.seg = jnp.asarray(seg)

  def test_delta_rule(self):
    """The rule bit for bit, where fp32 tells the two inverses apart (the bf16 logits do not: GDN_MUTANTS)."""
    q, k, v, g, beta = self.args
    outs = {}
    for dt in (jnp.float32, jnp.bfloat16):
      ins = [jnp.asarray(x, dt) for x in (q, k, v)] + [jnp.asarray(g, jnp.float32), jnp.asarray(beta, dt)]
      for seg in (None, self.seg):
        for inverse in qc.GDN_CHUNK_INVERSES:
          want = jax.jit(
              lambda *a, seg=seg, inverse=inverse: self.qwen3.jax_chunk_gated_delta_rule(
                  *a, chunk_size=8, use_qk_norm_in_gdn=True, compute_dtype=dt, segment_ids=seg, chunk_inverse=inverse
              )[0]
          )(*ins)
          got = jax.jit(
              lambda *a, seg=seg, inverse=inverse: gdn.chunk_gated_delta_rule(
                  *a, 8, True, dt, segment_ids=seg, chunk_inverse=inverse
              )
          )(*ins)
          with self.subTest(dtype=jnp.dtype(dt).name, packed=seg is not None, chunk_inverse=inverse):
            self.assertEqual((got.shape, got.dtype), (want.shape, want.dtype))
            self.assertTrue(bool(jnp.array_equal(got, want)), f"max rel-to-max {max_rel(got, want):.3e}")
            outs[jnp.dtype(dt).name, seg is not None, inverse] = got
    for packed in (False, True):  # fp32 sees which inverse ran
      self.assertFalse(bool(jnp.array_equal(outs["float32", packed, "solve"], outs["float32", packed, "block_doubling"])))

  def test_canonical_documents_and_inverse(self):
    np.testing.assert_array_equal(
        np.asarray(gdn.canonical_segment_ids(self.seg)), np.asarray(self.gdn_packing.canonicalize_segment_ids(self.seg))
    )
    rng = np.random.default_rng(12)
    for c in (16, 12):
      sm = jnp.asarray(np.tril(rng.standard_normal((2, 3, 2, c, c)) * 0.3, -1), jnp.float32)
      want = jax.jit(lambda m, c=c: self.gdn_packing.block_doubling_unit_lower_inverse(m, c))(sm)
      for name, fn in (
          ("_block_doubling", gdn._block_doubling),
          ("unit_lower_inv", lambda m: gdn.unit_lower_inv(m, "block_doubling")),
      ):
        with self.subTest(chunk=c, fn=name):
          self.assertTrue(bool(jnp.array_equal(jax.jit(fn)(sm), want)))

  def test_segmented_conv(self):
    rng = np.random.default_rng(13)
    x = rng.standard_normal((2, 37, 24))
    w = jnp.asarray(rng.standard_normal((4, 1, 24)), jnp.float32)
    for dt in (jnp.float32, jnp.bfloat16):
      want = jax.jit(
          lambda x, w: self.gdn_packing.segmented_causal_depthwise_conv1d(
              qkv=x, conv_weight=w, segment_ids=self.seg, kernel_size=4, dtype=dt
          )
      )(jnp.asarray(x, dt), w)
      for impl in qc.GDN_CONV_IMPLS:
        cfg = qc.tiny(dtype=dt, gdn_conv_impl=impl)
        got = jax.jit(lambda x, w, c=cfg: gdn.segmented_causal_conv(x, w, self.seg, c))(jnp.asarray(x, dt), w)
        with self.subTest(dtype=jnp.dtype(dt).name, impl=impl):
          self.assertEqual((got.shape, got.dtype), (want.shape, want.dtype))
          if impl == "conv":
            self.assertTrue(bool(jnp.array_equal(got, want)), f"max rel-to-max {max_rel(got, want):.3e}")
          else:
            self.assertLess(max_rel(got, want), 1e-6 if dt == jnp.float32 else 2**-7)

  def test_segment_masks(self):
    b, s, h, c = 2, 37, 3, 8
    n = -(-s // c)
    rng = np.random.default_rng(14)
    g_c = jnp.asarray(-np.abs(rng.standard_normal((b, n, h, c))), jnp.float32)
    k_c = jnp.asarray(rng.standard_normal((b, n, h, c, 4)), jnp.float32)
    beta_c = jnp.asarray(rng.uniform(size=(b, n, h, c)), jnp.float32)
    out = self.gdn_packing.prepare_jax_gdn_segment_masks(
        self.seg, k_c, k_c, k_c, g_c, beta_c, batch_size=b, seq_len=s, num_chunks=n, chunk_size=c
    )
    g_cum_mt, strict_mt, causal_mt, m_in_mt, m_out_mt, m_keep_mt, valid_mt = out[5:]
    valid, cum, strict, causal, m_in, m_out, m_keep = jax.jit(gdn._segment_masks, static_argnums=(1, 2))(self.seg, n, c)
    for name, got, want in (
        ("valid", valid, valid_mt),
        ("strict", strict, strict_mt),
        ("causal", causal, causal_mt),
        ("m_in", m_in, m_in_mt),
        ("m_out", m_out, m_out_mt),
        ("m_keep", m_keep, m_keep_mt),
        ("cumulative decay", jax.jit(lambda m, g: _ops.segment_cumsum(m, jnp.where(valid, g, 0.0)))(cum, g_c), g_cum_mt),
    ):
      with self.subTest(name):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


@pytest.mark.scheduled_only
class AgainstMaxTextQwen38(unittest.TestCase):
  """The same checks on MaxText's qwen3.8-2.4t-a95b at the tiny shape: its yml leaves use_mrope at base.yml's
  false, so MaxText's gated attention rotates with PartialRotaryEmbedding (sin / cos of position / timescale
  rounded to the activation dtype, the rotation in it), which the port's `partial_rope` follows op for op
  (`Qwen35Config.use_mrope=False`, carried by from_maxtext).  The bf16 bit-exact logits of this model are
  test_forward_logits_bf16_bit_exact's fourth case set."""

  def test_from_maxtext_takes_the_partial_rope(self):
    mt = _MaxText(model=QWEN38)
    self.assertFalse(mt.cfg.use_mrope)
    self.assertIs(mt.port_cfg.use_mrope, False)
    self.assertEqual(mt.port_cfg.replace(use_mrope=True), _MaxText().port_cfg)

  def test_parameter_tree(self):
    mt = _MaxText(model=QWEN38)
    got = {
        jax.tree_util.keystr(p): (tuple(a.shape), a.dtype) for p, a in jax.tree_util.tree_flatten_with_path(mt.params)[0]
    }
    want = {
        jax.tree_util.keystr(p): (tuple(a.shape), a.dtype)
        for p, a in jax.tree_util.tree_flatten_with_path(qm.param_avals(mt.port_cfg))[0]
    }
    self.assertEqual(got, want)

  def test_forward_logits(self):
    """fp32 logits within 1e-6 of MaxText's (the mRoPE port would part from them in the rotation's last bits)."""
    mt = _MaxText(model=QWEN38)
    fwd = jax.jit(lambda p, t, q, s, c: qm.forward(p, t, q, s, c), static_argnums=4)
    for name, (tok, pos, seg) in forward_cases():
      with self.subTest(name):
        want = mt.logits(tok, pos, seg)
        self.assertLessEqual(max_rel(fwd(mt.params, tok, pos, seg, mt.port_cfg), want), 1e-6)

  def test_gradients(self):
    """The loss and every gradient leaf against MaxText's, at test_gradients' floors."""
    mt = _MaxText(model=QWEN38)
    nnx, cfg = mt.nnx, mt.port_cfg
    nnx.pop(mt.model, nnx.Intermediate)
    graphdef, mt_params, rest = nnx.split(mt.model, nnx.Param, ...)
    port = jax.jit(
        jax.value_and_grad(
            lambda p, c, tok, pos, seg, targets: qt.loss_from_logits(
                qm.forward(p, tok, pos, seg, c), targets, seg, c.vocab_size
            )[0]
        ),
        static_argnums=1,
    )
    tok, pos, seg = documents()
    targets = jnp.roll(tok, -1, axis=1)

    def mt_loss(p, r):
      logits = mt.logits(tok, pos, seg, model=nnx.merge(graphdef, p, r))
      return qt.loss_from_logits(logits, targets, seg, mt.cfg.vocab_size)[0]

    mt_l, mt_g = jax.jit(jax.value_and_grad(mt_loss))(mt_params, rest)
    l, g = port(mt.params, cfg, tok, pos, seg, targets)
    _, g_unrolled = port(mt.params, cfg.replace(scan_layers=False), tok, pos, seg, targets)
    self.assertLess(abs(float(l) - float(mt_l)), 1e-6 * abs(float(mt_l)))
    want = convert.from_nnx_state(mt_g, mt.cfg.param_scan_axis, params_only=False, config=cfg)
    floor = max(max_rel(a, b) for a, b in zip(jax.tree.leaves(g_unrolled), jax.tree.leaves(g)))
    tol = max(1e-5, 4 * floor)
    for (path, a), b in zip(jax.tree_util.tree_flatten_with_path(g)[0], jax.tree.leaves(want)):
      self.assertLessEqual(max_rel(a, b), tol, f"{jax.tree_util.keystr(path)} (floor {floor:.2e})")


if __name__ == "__main__":
  if sys.argv[1:] == ["--bf16-forward"]:
    bf16_forward_report()
  else:
    unittest.main()
