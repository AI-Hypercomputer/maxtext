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

"""CPU unit tests for on-policy distillation (OPD) block-diffusion rollout.

Covers the *pure* logic of the OPD student rollout end-to-end on CPU:

  (a) the student block-diffusion generate loop with a STUB forward (canned,
      position-addressed logits): a block fills within its step cap, the
      forced-argmax progress guarantee, the shifted-logit indexing (token i from
      hidden i-1), multi-block advancement, prompt positions held fixed, and the
      step-cap force-fill (no ``mask_id`` leak);
  (b) that the committed-position mask makes the reused forward-KL loss nonzero
      ONLY at committed positions (real ``create_labels`` + ``compute_loss`` with
      crafted student/teacher logits);
  (c) that generation is stop-gradient — no gradient flows to the (differentiable)
      student forward parameters through the rollout (checked via ``jax.grad``),
      plus a source check that ``_train_step`` wires the rollout under
      ``jax.lax.stop_gradient`` and behind the ``opd_on_policy`` gate.

The pure rollout module (``diffusion_generate.py``) is loaded standalone by file
path (it imports only ``jax``), so (a)/(c) run with no MaxText/Tunix deps. The
loss gating test (b) imports the real ``distillation_utils``; when ``tunix`` is
not installed (bare CPU venv) a minimal stub is injected so the real strategy
still loads and runs. The full two-model on-policy ``_train_step`` needs real
model weights and a TPU and is therefore covered by source/AST checks here.
"""

import ast
import importlib.util
import pathlib
import sys
import types
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = [pytest.mark.cpu_only, pytest.mark.post_training]

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_DISTILL_DIR = _REPO_ROOT / "src" / "maxtext" / "trainers" / "post_train" / "distillation"
_DIFFUSION_GENERATE = _DISTILL_DIR / "diffusion_generate.py"
_TRAIN_DISTILL = _DISTILL_DIR / "train_distill.py"


def _load_diffusion_generate_standalone():
  """Load ``diffusion_generate.py`` by file path (it imports only jax)."""
  spec = importlib.util.spec_from_file_location("opd_diffusion_generate_under_test", _DIFFUSION_GENERATE)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


_DG = _load_diffusion_generate_standalone()
diffusion_generate = _DG.diffusion_generate
diffusion_commit = _DG.diffusion_commit


def _import_distillation_utils():
  """Import the real ``distillation_utils``; stub ``tunix`` if it is absent.

  In CI ``tunix`` is installed and imported directly. On a bare ``jax[cpu]`` venv
  we inject the two tiny tunix symbols ``distillation_utils`` needs at import time
  (``peft_trainer.TrainingInput`` / ``PeftTrainer`` and
  ``checkpoint_manager.CheckpointManager``) so the real strategy class still loads.
  """
  try:
    import tunix  # noqa: F401  pylint: disable=unused-import
  except ImportError:
    import flax

    for pkg in ("tunix", "tunix.sft"):
      if pkg not in sys.modules:
        m = types.ModuleType(pkg)
        m.__path__ = []  # mark as package
        sys.modules[pkg] = m

    pt = types.ModuleType("tunix.sft.peft_trainer")

    @flax.struct.dataclass(frozen=True)
    class _TrainingInput:  # minimal base for MaxTextTrainingInput
      input_tokens: Any
      input_mask: Any

    pt.TrainingInput = _TrainingInput
    pt.PeftTrainer = type("PeftTrainer", (), {})
    sys.modules["tunix.sft.peft_trainer"] = pt

    cm = types.ModuleType("tunix.sft.checkpoint_manager")
    cm.CheckpointManager = type("CheckpointManager", (), {})
    sys.modules["tunix.sft.checkpoint_manager"] = cm

  src = str(_REPO_ROOT / "src")
  if src not in sys.path:
    sys.path.insert(0, src)
  from maxtext.trainers.post_train.distillation import distillation_utils  # pylint: disable=import-outside-toplevel

  return distillation_utils


# ---- Test constants ---------------------------------------------------------
VOCAB = 16
MASK_ID = 15  # out of the normal token range so leaks are detectable


def _target_array(n: int) -> np.ndarray:
  """Deterministic absolute-position -> token map in [0, VOCAB-3]."""
  return np.array([p % (VOCAB - 2) for p in range(n)], dtype=np.int32)


def _make_stub_forward(global_target: np.ndarray, logit_scale: float):
  """Pure-JAX stub forward ``(tokens[B,T], positions[B,T]) -> logits[B,T,V]``.

  ``logits[b, j]`` peaks on ``global_target[positions[b, j]]`` with confidence
  set by ``logit_scale`` (large -> commits via threshold in one step; small ->
  below threshold so only the forced-argmax position commits). The stub ignores
  the canvas so per-position argmax is stable and committed values are exact.
  """
  g = jnp.asarray(global_target)

  def forward_fn(tokens, positions):
    del tokens
    target = g[positions]  # [B, T]
    return jax.nn.one_hot(target, VOCAB, dtype=jnp.float32) * logit_scale  # [B, T, V]

  return forward_fn


def _make_batch(batch_size, seq_len, prompt_len):
  """Prompt occupies ``[0, prompt_len)``; completion ``[prompt_len, seq_len)``."""
  g = _target_array(256)
  tokens = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
  # Distinct, non-mask prompt tokens so "prompt untouched" is meaningful.
  prompt = jnp.arange(1, prompt_len + 1, dtype=jnp.int32)
  tokens = tokens.at[:, :prompt_len].set(prompt)
  positions = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32), (batch_size, seq_len))
  gen_mask = jnp.broadcast_to(jnp.arange(seq_len) >= prompt_len, (batch_size, seq_len))
  return g, tokens, positions, gen_mask


class TestDiffusionGenerateStub:
  """(a) Pure student generate loop with a stub forward."""

  def test_block_fills_within_max_steps_high_confidence(self):
    g, tokens, positions, gen_mask = _make_batch(batch_size=2, seq_len=8, prompt_len=4)
    fwd = _make_stub_forward(g, logit_scale=30.0)
    gen, committed = diffusion_generate(
        fwd, tokens, positions, gen_mask, mask_id=MASK_ID, bd_size=4, threshold=0.9, temperature=0.0
    )
    gen = np.asarray(gen)
    committed = np.asarray(committed)
    # High confidence -> the whole completion commits, no mask_id leaks.
    assert not np.any(gen == MASK_ID)
    # Every generated position is reported committed; prompt positions are not.
    assert np.array_equal(committed, np.asarray(gen_mask))

  def test_forced_argmax_progress_low_confidence(self):
    # Below-threshold confidence: only the forced (highest-confidence masked)
    # position commits per step, yet the block still fully fills within bd_size.
    g, tokens, positions, gen_mask = _make_batch(batch_size=1, seq_len=8, prompt_len=4)
    fwd = _make_stub_forward(g, logit_scale=0.05)
    gen, _ = diffusion_generate(
        fwd, tokens, positions, gen_mask, mask_id=MASK_ID, bd_size=4, threshold=0.9, temperature=0.0
    )
    assert not np.any(np.asarray(gen) == MASK_ID)

  def test_shifted_logit_indexing_i_from_i_minus_1(self):
    # Completion starts mid-block to exercise the shift across a block boundary.
    prompt_len = 3
    g, tokens, positions, gen_mask = _make_batch(batch_size=1, seq_len=8, prompt_len=prompt_len)
    fwd = _make_stub_forward(g, logit_scale=30.0)
    gen, _ = diffusion_generate(
        fwd, tokens, positions, gen_mask, mask_id=MASK_ID, bd_size=4, threshold=0.9, temperature=0.0
    )
    gen = np.asarray(gen)[0]
    gnp = np.asarray(g)
    # Prompt untouched.
    assert gen[:prompt_len].tolist() == list(range(1, prompt_len + 1))
    # Shifted convention: committed token at position p == target at (p - 1).
    for p in range(prompt_len, 8):
      assert int(gen[p]) == int(gnp[p - 1]), f"shift mismatch at {p}"

  def test_multi_block_advance_contiguous_across_boundaries(self):
    # prompt block 0 fully prompt; blocks 1 and 2 fully generated.
    prompt_len = 4
    seq_len = 12
    bd = 4
    g, tokens, positions, gen_mask = _make_batch(batch_size=1, seq_len=seq_len, prompt_len=prompt_len)
    fwd = _make_stub_forward(g, logit_scale=30.0)
    gen, committed = diffusion_generate(
        fwd, tokens, positions, gen_mask, mask_id=MASK_ID, bd_size=bd, threshold=0.9, temperature=0.0
    )
    gen = np.asarray(gen)[0]
    gnp = np.asarray(g)
    assert not np.any(gen == MASK_ID)
    # Contiguity across every block boundary proves later blocks were seeded from
    # the previous block's committed tail (shifted convention, full sequence).
    for p in range(prompt_len, seq_len):
      assert int(gen[p]) == int(gnp[p - 1]), f"mismatch at {p}"
    # Explicit boundary check at the start of block 2.
    assert int(gen[2 * bd]) == int(gnp[2 * bd - 1])
    assert np.array_equal(np.asarray(committed)[0], np.asarray(gen_mask)[0])

  def test_prompt_positions_never_masked(self):
    prompt_len = 5
    g, tokens, positions, gen_mask = _make_batch(batch_size=3, seq_len=8, prompt_len=prompt_len)
    fwd = _make_stub_forward(g, logit_scale=30.0)
    gen, committed = diffusion_generate(
        fwd, tokens, positions, gen_mask, mask_id=MASK_ID, bd_size=4, threshold=0.9, temperature=0.0
    )
    gen = np.asarray(gen)
    # Prompt region identical to the input for every row.
    assert np.array_equal(gen[:, :prompt_len], np.asarray(tokens)[:, :prompt_len])
    # Prompt positions are never reported as committed.
    assert not np.any(np.asarray(committed)[:, :prompt_len])

  def test_step_cap_force_fill_no_mask_leak(self):
    # Cap denoise steps far below what full commitment needs; the post-loop
    # force-fill must still leave no mask_id and follow the shifted convention.
    prompt_len = 4
    g, tokens, positions, gen_mask = _make_batch(batch_size=1, seq_len=8, prompt_len=prompt_len)
    fwd = _make_stub_forward(g, logit_scale=0.05)  # below threshold
    gen, _ = diffusion_generate(
        fwd,
        tokens,
        positions,
        gen_mask,
        mask_id=MASK_ID,
        bd_size=4,
        threshold=0.9,
        temperature=0.0,
        max_denoise_steps=1,  # far fewer than the 4 needed per block
    )
    gen = np.asarray(gen)[0]
    gnp = np.asarray(g)
    assert not np.any(gen == MASK_ID)
    for p in range(prompt_len, 8):
      assert int(gen[p]) == int(gnp[p - 1]), f"force-filled mismatch at {p}"


class TestDiffusionCommit:
  """Direct checks on the pure commit primitive."""

  def test_high_confidence_commits_all_masked(self):
    logits = jax.nn.one_hot(jnp.array([1, 2, 3]), VOCAB, dtype=jnp.float32) * 30.0  # [3, V]
    mask = jnp.array([True, True, True])
    tok, new_mask = diffusion_commit(logits, mask, threshold=0.9, temperature=0.0)
    assert not np.any(np.asarray(new_mask))  # all committed
    assert np.asarray(tok).tolist() == [1, 2, 3]

  def test_low_confidence_commits_exactly_one_forced(self):
    # Flat-ish logits below threshold -> only the single highest-confidence
    # masked position is force-committed; the mask shrinks by exactly one.
    logits = jnp.array([[0.10, 0.09, 0.0, 0.0], [0.05, 0.05, 0.05, 0.05]], dtype=jnp.float32)
    mask = jnp.array([True, True])
    _, new_mask = diffusion_commit(logits, mask, threshold=0.9, temperature=0.0)
    assert int(np.asarray(new_mask).sum()) == 1

  def test_no_masked_positions_forces_nothing(self):
    logits = jnp.zeros((2, VOCAB), dtype=jnp.float32)
    mask = jnp.array([False, False])
    _, new_mask = diffusion_commit(logits, mask, threshold=0.9, temperature=0.0)
    assert not np.any(np.asarray(new_mask))  # stays empty; nothing forced


class TestGenerationIsStopGradient:
  """(c) Generation is off the student gradient path."""

  def test_no_grad_through_rollout_to_forward_params(self):
    # forward_fn depends on a differentiable parameter `w`; a scalar built from
    # the generated tokens must have zero gradient w.r.t. `w` (argmax/commit is
    # non-differentiable), i.e. any loss on the rollout gets no grad from it.
    g, tokens, positions, gen_mask = _make_batch(batch_size=1, seq_len=8, prompt_len=4)
    g = jnp.asarray(g)

    def scalar_of_w(w):
      def fwd(toks, pos):
        del toks
        return jax.nn.one_hot(g[pos], VOCAB, dtype=jnp.float32) * 30.0 + w

      gen, _ = diffusion_generate(fwd, tokens, positions, gen_mask, mask_id=MASK_ID, bd_size=4, threshold=0.9)
      return jnp.sum(gen.astype(jnp.float32))

    grad = jax.grad(scalar_of_w)(jnp.float32(0.7))
    assert float(grad) == 0.0

  def test_train_step_wires_rollout_under_stop_gradient(self):
    src = _TRAIN_DISTILL.read_text()
    # Gated behind the (default-False) opd_on_policy flag.
    assert "if self.opd_on_policy:" in src
    assert "self._run_on_policy_rollout(batch, student)" in src
    # Rollout outputs are placed under stop_gradient.
    method = _find_method(ast.parse(src), "MaxTextDistillationTrainer", "_run_on_policy_rollout")
    assert method is not None, "_run_on_policy_rollout not defined"
    body_src = ast.get_source_segment(src, method)
    assert "jax.lax.stop_gradient(gen_tokens)" in body_src
    assert "jax.lax.stop_gradient(committed_mask)" in body_src


class TestCommittedMaskGatesForwardKL:
  """(b) The committed-position mask makes forward-KL nonzero only there."""

  def _strategy(self, du, vocab, alpha=1.0, temperature=1.0):
    return du.CombinedDistillationStrategy(
        student_forward_fn=None,
        teacher_forward_fn=None,
        vocab_size=vocab,
        pad_id=0,
        temperature=temperature,
        alpha=alpha,
    )

  def test_create_labels_mask_is_committed_positions(self):
    du = _import_distillation_utils()
    strat = self._strategy(du, vocab=6)
    targets = jnp.array([[2, 3, 4, 5]])
    committed = jnp.array([[0, 1, 0, 1]])  # committed at positions 1 and 3
    labels = strat.create_labels(targets, targets_segmentation=committed)
    mask = np.asarray(jnp.any(labels != 0, axis=-1))[0]
    assert mask.tolist() == [False, True, False, True]

  def test_forward_kl_counts_only_committed_positions(self):
    du = _import_distillation_utils()
    strat = self._strategy(du, vocab=6, alpha=1.0, temperature=1.0)
    targets = jnp.array([[2, 3, 4, 5]])
    committed = jnp.array([[0, 1, 0, 1]])
    labels = strat.create_labels(targets, targets_segmentation=committed)

    s_logits = jax.random.normal(jax.random.PRNGKey(0), (1, 4, 6))
    t_logits = jax.random.normal(jax.random.PRNGKey(1), (1, 4, 6))
    s_out = du.DistillationForwardOutput(logits=s_logits)
    t_out = du.DistillationForwardOutput(logits=t_logits)

    _, metrics = strat.compute_loss(s_out, t_out, labels, step=None)
    kl_sum = float(metrics[du.METRIC_KL_DIV_AT_T][0])

    def fkl(sl, tl):  # forward KL(teacher || student) at one position
      return float(jnp.sum(jax.nn.softmax(tl) * (jax.nn.log_softmax(tl) - jax.nn.log_softmax(sl))))

    hand = fkl(s_logits[0, 1], t_logits[0, 1]) + fkl(s_logits[0, 3], t_logits[0, 3])
    assert abs(kl_sum - hand) < 1e-4

  def test_loss_invariant_to_non_committed_positions(self):
    du = _import_distillation_utils()
    strat = self._strategy(du, vocab=6, alpha=1.0, temperature=1.0)
    targets = jnp.array([[2, 3, 4, 5]])
    committed = jnp.array([[0, 1, 0, 1]])
    labels = strat.create_labels(targets, targets_segmentation=committed)

    s_logits = jax.random.normal(jax.random.PRNGKey(0), (1, 4, 6))
    t_logits = jax.random.normal(jax.random.PRNGKey(1), (1, 4, 6))
    s_out = du.DistillationForwardOutput(logits=s_logits)
    loss_a, _ = strat.compute_loss(s_out, du.DistillationForwardOutput(logits=t_logits), labels, step=None)
    # Perturb teacher logits ONLY at non-committed positions 0 and 2.
    t2 = t_logits.at[0, 0].add(5.0).at[0, 2].add(-3.0)
    loss_b, _ = strat.compute_loss(s_out, du.DistillationForwardOutput(logits=t2), labels, step=None)
    assert abs(float(loss_a) - float(loss_b)) < 1e-5


class TestSourceStructure:
  """Structural checks that the rollout is on-device and correctly wired."""

  def test_generate_uses_lax_loops(self):
    src = _DIFFUSION_GENERATE.read_text()
    assert "jax.lax.while_loop" in src  # inner per-block denoise
    assert "jax.lax.fori_loop" in src  # outer block advance

  def test_train_distill_imports_diffusion_generate(self):
    src = _TRAIN_DISTILL.read_text()
    assert "import diffusion_generate" in src or "diffusion_generate," in src
    assert "diffusion_generate.diffusion_generate(" in src


def _find_method(module: ast.Module, cls_name: str, method_name: str):
  for node in module.body:
    if isinstance(node, ast.ClassDef) and node.name == cls_name:
      for sub in node.body:
        if isinstance(sub, ast.FunctionDef) and sub.name == method_name:
          return sub
  return None
