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

"""Self-tests for the vendored DeepSeek-V4 reference (tests/utils/deepseek4_reference).

Checks that the vendored model.py differs from upstream only in `# [CHANGE]`
lines, that Block.forward is differentiable (fp64 finite differences and
torch.autograd.gradcheck), and that the pure-torch kernel and Hadamard shims
match their mathematical definitions. CPU only.
"""

import ast
import hashlib
import math
import pathlib

import numpy as np
import pytest

try:
  import torch
  from tests.utils import deepseek4_reference as ref
  from tests.utils.deepseek4_reference import fast_hadamard_transform as fht

  HAS_TORCH = True
except ImportError:
  HAS_TORCH = False

pytestmark = [
    pytest.mark.cpu_only,
    pytest.mark.scheduled_only,
    pytest.mark.skipif(not HAS_TORCH, reason="torch not available"),
]

_MARKER = "# [CHANGE]"
_HEADER_END = "# pylint: skip-file\n# fmt: off\n"
# One marker per edited upstream line (== len(_REVERT)); the exact count also catches duplicated marker lines.
_EXPECTED_MARKERS = 6
# Vendored code (text before the marker) -> upstream line.
_REVERT = {
    "from .kernel import act_quant, fp4_act_quant, fp8_gemm, fp4_gemm, sparse_attn, hc_split_sinkhorn": (
        "from kernel import act_quant, fp4_act_quant, fp8_gemm, fp4_gemm, sparse_attn, hc_split_sinkhorn"
    ),
    "    assert x.dtype == torch.bfloat16 or x.dtype == default_dtype": "    assert x.dtype == torch.bfloat16",
    "    from .fast_hadamard_transform import hadamard_transform": (
        "    from fast_hadamard_transform import hadamard_transform"
    ),
    (
        "            self.compressor.kv_cache = torch.zeros_like(self.kv_cache[:, win:]) if torch.is_grad_enabled()"
        " else self.kv_cache[:, win:]"
    ): "            self.compressor.kv_cache = self.kv_cache[:, win:]",
    "        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)": (
        "        q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)"
    ),
    (
        '        default_dtype = {"fp8": torch.float8_e4m3fn, "fp32": torch.float32, "fp64": torch.float64}'
        ".get(args.dtype, torch.bfloat16)"
    ): '        default_dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16',
}
_ALLOWED_ABSOLUTE_IMPORTS = {"math", "dataclasses", "typing", "functools", "contextlib", "torch"}


@pytest.fixture(name="model_globals")
def _model_globals():
  """Restores model.py globals and kernel mode mutated by a test."""
  names = ("world_size", "rank", "default_dtype", "scale_fmt", "scale_dtype")
  saved = {n: getattr(ref.model, n) for n in names}
  mode = ref.kernel.MODE
  yield
  for n, v in saved.items():
    setattr(ref.model, n, v)
  ref.kernel.set_mode(mode)


# -----------------------------------------------------------------------------
# (a) Diff guard
# -----------------------------------------------------------------------------


def _model_source() -> str:
  return pathlib.Path(ref.model.__file__).read_text(encoding="utf-8")


def _split_header(src: str):
  """Splits the added license/lint header (ending at `# fmt: off`) from the upstream body."""
  head, sep, body = src.partition(_HEADER_END)
  assert sep, "missing header terminator"
  assert head.startswith("# Copyright") and head.count("\n") < 60, "header must be a short comment block"
  assert all(not l or l.startswith("#") for l in head.splitlines()), "header must be comments only"
  for required in ("Apache License", "MIT License", "Copyright (c) 2023 DeepSeek", ref.UPSTREAM_REVISION):
    assert required in head, f"header lacks {required!r}"
  return head + sep, body


def _drift_msg(what: str) -> str:
  return f"vendored {what} drifted from upstream {ref.UPSTREAM_REPO}@{ref.UPSTREAM_REVISION}"


def test_only_marked_lines_differ_from_upstream():
  head, body = _split_header(_model_source())
  assert _MARKER not in head, "marker in header"
  assert _EXPECTED_MARKERS == len(_REVERT)
  assert body.count(_MARKER) == _EXPECTED_MARKERS, f"expected {_EXPECTED_MARKERS} {_MARKER} lines"
  out, used = [], []
  for line in body.splitlines(keepends=True):
    if _MARKER not in line:
      out.append(line)
    else:
      code = line[: line.index("  " + _MARKER)]
      assert code in _REVERT, f"unexpected [CHANGE] line: {code!r}"
      used.append(code)
      out.append(_REVERT[code] + "\n")
  assert sorted(used) == sorted(_REVERT), f"[CHANGE] lines missing or duplicated: {used}"
  digest = hashlib.sha256("".join(out).encode("utf-8")).hexdigest()
  assert digest == ref.UPSTREAM_MODEL_SHA256, _drift_msg("model.py")


def test_encoding_and_testdata_match_upstream():
  ref_dir = pathlib.Path(ref.__file__).parent
  _, body = _split_header((ref_dir / "encoding_dsv4.py").read_text(encoding="utf-8"))
  assert hashlib.sha256(body.encode("utf-8")).hexdigest() == ref.UPSTREAM_ENCODING_SHA256, _drift_msg("encoding_dsv4.py")
  files = sorted(p.name for p in (ref_dir / "testdata").iterdir())
  assert files == sorted(ref.UPSTREAM_TESTDATA_SHA256), f"unexpected testdata files: {files}"
  for name, want in ref.UPSTREAM_TESTDATA_SHA256.items():
    got = hashlib.sha256((ref_dir / "testdata" / name).read_bytes()).hexdigest()
    assert got == want, _drift_msg(f"testdata/{name}")


def test_model_imports_only_package_modules():
  for node in ast.walk(ast.parse(_model_source())):
    if isinstance(node, ast.Import):
      mods = [a.name for a in node.names]
    elif isinstance(node, ast.ImportFrom):
      if node.level:
        assert node.module in ("kernel", "fast_hadamard_transform"), node.module
        continue
      mods = [node.module]
    else:
      continue
    for m in mods:
      assert m.split(".")[0] in _ALLOWED_ABSOLUTE_IMPORTS, m
  assert ref.model.sparse_attn is ref.kernel.sparse_attn
  assert ref.model.hc_split_sinkhorn is ref.kernel.hc_split_sinkhorn


# -----------------------------------------------------------------------------
# (b) Block.forward gradient checks (float64)
# -----------------------------------------------------------------------------


def _build_block(layer_id, args, gen):
  prev = torch.get_default_dtype()
  torch.set_default_dtype(torch.float64)
  try:
    block = ref.model.Block(layer_id, args).double()
  finally:
    torch.set_default_dtype(prev)
  ref.init_params(block, args, gen)
  return block


@pytest.mark.parametrize(
    "layer_id,ratio,is_hash",
    [(0, 0, True), (2, 4, True), (3, 128, False)],
)
def test_block_gradients_fp64(model_globals, layer_id, ratio, is_hash):
  del model_globals
  args = ref.tiny_args()
  assert args.compress_ratios[layer_id] == ratio
  ref.configure(torch.float64, args)
  ref.kernel.set_mode("train")
  seq, eps, n_dirs = ref.TINY_SEQ_LEN, 1e-6, 4
  bsz = args.max_batch_size  # bsz == max_batch_size is needed to hit the upstream model.py:491 view failure.
  gen = torch.Generator().manual_seed(layer_id)
  with ref.fp64_promotion():
    block = _build_block(layer_id, args, gen)
    assert block.ffn.gate.hash == is_hash
    assert (ratio == 4) == (getattr(block.attn, "indexer", None) is not None)
    h = torch.randn(bsz, seq, args.hc_mult, args.dim, generator=gen, dtype=torch.float64, requires_grad=True)
    ids = torch.randint(0, args.vocab_size, (bsz, seq), generator=gen)
    w = torch.randn(bsz, seq, args.hc_mult, args.dim, generator=gen, dtype=torch.float64)

    def f(x):
      return block(x, 0, ids)

    def loss(x):
      return (f(x) * w).sum()

    (g,) = torch.autograd.grad(loss(h), h)
    with torch.no_grad():
      for _ in range(n_dirs):
        u = torch.randn(h.shape, generator=gen, dtype=torch.float64)
        u /= u.norm()
        fd = (loss(h + eps * u) - loss(h - eps * u)).item() / (2 * eps)
        an = (g * u).sum().item()
        rel = abs(fd - an) / max(abs(fd), abs(an), 1e-30)
        assert rel < 1e-5, f"layer {layer_id}: fd={fd:.12e} analytic={an:.12e} rel={rel:.3e}"
    assert torch.autograd.gradcheck(f, (h,), eps=eps, atol=1e-5, rtol=1e-3, fast_mode=True)


@pytest.mark.parametrize("layer_id,yarn", [(0, False), (2, True), (3, True)])
def test_yarn_active_on_compressed_layers(model_globals, layer_id, yarn):
  del model_globals
  args = ref.tiny_args()
  assert args.original_seq_len < args.max_seq_len
  ref.configure(torch.float32, args)
  attn = ref.model.Attention(layer_id, args)
  base = args.compress_rope_theta if yarn else args.rope_theta
  plain = ref.model.precompute_freqs_cis(
      args.rope_head_dim, args.max_seq_len, 0, base, args.rope_factor, args.beta_fast, args.beta_slow
  )
  assert (not torch.equal(attn.freqs_cis, plain)) == yarn


# -----------------------------------------------------------------------------
# (c) Kernel fidelity
# -----------------------------------------------------------------------------


def _dense_sink_attention(q, kv, sink, idx, scale):
  """Dense masked softmax over [selected keys, sink]; the sink carries no value."""
  b, m, _, _ = q.shape
  n = kv.size(1)
  mask = torch.zeros(b, m, n + 1, dtype=torch.bool)
  bi, mi, _ = torch.meshgrid(torch.arange(b), torch.arange(m), torch.arange(idx.size(-1)), indexing="ij")
  sel = idx.long().where(idx >= 0, torch.full_like(idx.long(), n))
  mask[bi, mi, sel] = True
  mask = mask[..., :n]
  s = torch.einsum("bmhd,bnd->bmhn", q, kv) * scale
  s = s.masked_fill(~mask.unsqueeze(2), float("-inf"))
  s = torch.cat([s, sink.view(1, 1, -1, 1).expand(b, m, -1, 1)], dim=-1)
  p = torch.softmax(s, dim=-1)[..., :n]
  return torch.einsum("bmhn,bnd->bmhd", p, kv)


def _random_topk(b, m, n, topk, gen):
  idx = torch.stack([torch.stack([torch.randperm(n, generator=gen)[:topk] for _ in range(m)]) for _ in range(b)])
  drop = torch.rand(idx.shape, generator=gen) < 0.3
  drop[..., 0] = False
  return idx.masked_fill(drop, -1).int()


@pytest.mark.parametrize("dtype,atol", [(torch.float64, 1e-12), (torch.float32, 1e-5)])
def test_sparse_attn_matches_dense_reference(dtype, atol):
  gen = torch.Generator().manual_seed(0)
  b, m, h, d, n, topk = 2, 5, 4, 32, 200, 150  # topk spans 3 kernel blocks of 64.
  q = torch.randn(b, m, h, d, generator=gen, dtype=torch.float64)
  kv = torch.randn(b, n, d, generator=gen, dtype=torch.float64)
  sink = torch.randn(h, generator=gen, dtype=torch.float64)
  idx = _random_topk(b, m, n, topk, gen)
  scale = d**-0.5
  out = ref.kernel.sparse_attn(q.to(dtype), kv.to(dtype), sink.float(), idx, scale)
  assert out.dtype == dtype
  q, kv = q.to(dtype).double(), kv.to(dtype).double()
  expected = _dense_sink_attention(q, kv, sink.float().double(), idx, scale)
  np.testing.assert_allclose(out.double().numpy(), expected.numpy(), atol=atol, rtol=0)


def _sinkhorn_numpy(mixes, scale, base, hc, iters, eps):  # pylint: disable=too-many-positional-arguments
  """Per-row loop transcription of hc_split_sinkhorn_kernel."""
  rows = mixes.reshape(-1, mixes.shape[-1])
  pre = np.empty((rows.shape[0], hc))
  post = np.empty((rows.shape[0], hc))
  comb = np.empty((rows.shape[0], hc, hc))
  for i, r in enumerate(rows):
    pre[i] = 1 / (1 + np.exp(-(r[:hc] * scale[0] + base[:hc]))) + eps
    post[i] = 2 / (1 + np.exp(-(r[hc : 2 * hc] * scale[1] + base[hc : 2 * hc])))
    c = (r[2 * hc :] * scale[2] + base[2 * hc :]).reshape(hc, hc)
    c = np.exp(c - c.max(1, keepdims=True))
    c = c / c.sum(1, keepdims=True) + eps
    c = c / (c.sum(0, keepdims=True) + eps)
    for _ in range(iters - 1):
      c = c / (c.sum(1, keepdims=True) + eps)
      c = c / (c.sum(0, keepdims=True) + eps)
    comb[i] = c
  lead = mixes.shape[:-1]
  return pre.reshape(*lead, hc), post.reshape(*lead, hc), comb.reshape(*lead, hc, hc)


def test_hc_split_sinkhorn_formula_and_doubly_stochastic():
  gen = torch.Generator().manual_seed(0)
  hc, iters, eps = 4, 20, 1e-6
  mixes = torch.randn(2, 3, (2 + hc) * hc, generator=gen, dtype=torch.float64)
  scale = torch.rand(3, generator=gen, dtype=torch.float64) + 0.5
  base = torch.randn((2 + hc) * hc, generator=gen, dtype=torch.float64)
  pre, post, comb = ref.kernel.hc_split_sinkhorn(mixes, scale, base, hc, iters, eps)
  for got, want in zip((pre, post, comb), _sinkhorn_numpy(mixes.numpy(), scale.numpy(), base.numpy(), hc, iters, eps)):
    np.testing.assert_allclose(got.numpy(), want, atol=1e-14, rtol=1e-12)
  np.testing.assert_allclose(comb.sum(-1).numpy(), 1.0, atol=1e-4)
  np.testing.assert_allclose(comb.sum(-2).numpy(), 1.0, atol=1e-5)
  assert bool((pre > eps).all() and (pre < 1 + eps).all() and (post > 0).all() and (post < 2).all())


def _is_power_of_two(s: torch.Tensor) -> bool:
  e = torch.log2(s.double())
  return bool((s > 0).all() and torch.equal(e, torch.round(e)))


def test_qat_act_quant_fp8_e4m3(model_globals):
  del model_globals
  ref.kernel.set_mode("qat")
  gen = torch.Generator().manual_seed(0)
  block = 128
  x = 3 * torch.randn(4, 2 * block, generator=gen)
  y, s = ref.kernel.act_quant(x, block, "ue8m0", torch.float32)
  assert y.dtype == torch.float8_e4m3fn and s.shape == (4, 2)
  assert _is_power_of_two(s)
  ratio = x.unflatten(-1, (2, block)).abs().amax(-1) / s
  assert bool((ratio > 224).all() and (ratio <= 448).all())

  xg = x.clone().requires_grad_(True)
  xq = ref.kernel.act_quant(xg.clone(), block, "ue8m0", torch.float32, inplace=True)
  np.testing.assert_array_equal(xq.detach().numpy(), (y.float().unflatten(-1, (2, block)) * s.unsqueeze(-1)).flatten(-2))
  on_grid = (xq.detach().unflatten(-1, (2, block)) / s.unsqueeze(-1)).flatten(-2)
  np.testing.assert_array_equal(on_grid.to(torch.float8_e4m3fn).float().numpy(), on_grid.numpy())
  xq.sum().backward()
  np.testing.assert_array_equal(xg.grad.numpy(), 1.0)  # Straight-through.

  ref.kernel.set_mode("train")
  assert ref.kernel.act_quant(x, block, "ue8m0", torch.float32, inplace=True) is x


def test_qat_fp4_act_quant_e2m1(model_globals):
  del model_globals
  ref.kernel.set_mode("qat")
  gen = torch.Generator().manual_seed(0)
  block, grid = 32, torch.tensor(ref.kernel._FP4_GRID)  # pylint: disable=protected-access
  x = torch.randn(4, 4 * block, generator=gen)
  q, s = ref.kernel.fp4_act_quant(x, block)
  assert s.dtype == torch.float8_e8m0fnu
  s = s.float()
  assert _is_power_of_two(s)
  assert bool(torch.isin(q.abs(), grid).all())
  ratio = x.unflatten(-1, (4, block)).abs().amax(-1) / s
  assert bool((ratio > 3).all() and (ratio <= 6).all())
  xq = ref.kernel.fp4_act_quant(x.clone(), block, inplace=True)
  np.testing.assert_array_equal(xq.numpy(), (q.unflatten(-1, (4, block)) * s[..., None]).flatten(-2).numpy())

  # Ties round to even.
  vals = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.75, 0.3, 5.9, 6.0])
  want = torch.tensor([0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, -1.0, 0.5, 6.0, 6.0])
  np.testing.assert_array_equal(ref.kernel._cast_fp4_e2m1(vals).numpy(), want.numpy())  # pylint: disable=protected-access


# -----------------------------------------------------------------------------
# (d) Hadamard shim
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("n", [16, 32, 128])
def test_hadamard_orthogonal_and_inner_product_invariant(n):
  hmat = fht.hadamard_matrix(n, torch.float64)
  np.testing.assert_array_equal((hmat @ hmat.T).numpy(), n * np.eye(n))
  i = np.arange(n)
  popcount = np.vectorize(lambda v: bin(v).count("1"))(i[:, None] & i[None, :])
  np.testing.assert_array_equal(hmat.numpy(), (-1.0) ** popcount)  # Natural (Sylvester) order.

  gen = torch.Generator().manual_seed(n)
  q = torch.randn(5, n, generator=gen, dtype=torch.float64)
  k = torch.randn(5, n, generator=gen, dtype=torch.float64)
  hq, hk = fht.hadamard_transform(q, n**-0.5), fht.hadamard_transform(k, n**-0.5)
  np.testing.assert_allclose((hq * hk).sum(-1).numpy(), (q * k).sum(-1).numpy(), atol=1e-12, rtol=0)
  np.testing.assert_allclose(hq.numpy(), (q @ hmat).numpy() / math.sqrt(n), atol=1e-14, rtol=0)


def test_rotate_activation_dtype_guard(model_globals):
  del model_globals
  n = 16
  x = torch.randn(3, n, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
  i = np.arange(n)
  hmat = (-1.0) ** np.vectorize(lambda v: bin(v).count("1"))(i[:, None] & i[None, :])
  ref.configure(torch.float64)
  np.testing.assert_allclose(ref.model.rotate_activation(x).numpy(), x.numpy() @ hmat * n**-0.5, atol=1e-14, rtol=0)
  ref.configure(torch.bfloat16)
  assert ref.model.rotate_activation(x.bfloat16()).dtype == torch.bfloat16
  with pytest.raises(AssertionError):
    ref.model.rotate_activation(x.float())
