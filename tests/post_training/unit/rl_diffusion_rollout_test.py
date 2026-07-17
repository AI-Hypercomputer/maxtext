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

"""CPU unit tests for block-diffusion RL (GRPO/DAPO) rollout + shared logprob.

Covers the pure logic of the block-diffusion RL pieces on CPU:

  (a) the shared block-diffusion per-token logprob fn
      (``diffusion_generate.diffusion_per_token_logps``): correct shape, the
      shifted-logit convention (post-shift logits at ``p`` score token ``p``),
      hand-computed values against a stub forward, temperature scaling, and
      ``completion_mask`` zeroing;
  (b) ``MaxTextDiffusionRollout.generate`` with a stub model produces a valid
      completion (no ``mask_id`` leak, EOS truncation, prompt echoed left-padded);
  (c) THE P0 INVARIANT: the rollout's old-policy logps
      (``MaxTextDiffusionRollout.get_per_token_logps``) and the loss's new-policy
      logps (the ``make_diffusion_per_token_logps_fn`` hook) are the SAME function
      and return identical values on the same (prompt, completion, stub model).

Both modules under test are pure/light: ``diffusion_generate.py`` imports only jax
and is loaded standalone by path; ``maxtext_diffusion_rollout.py`` is loaded
standalone with the real (standalone) ``diffusion_generate`` injected and a minimal
``tunix.rl.rollout.base_rollout`` stub, so the whole file runs on a bare jax[cpu]
venv with no MaxText package import or full tunix install. The tunix loss-side
gating (``grpo_loss_fn`` default = AR) is covered in the tunix repo's test.
"""

import abc
import dataclasses
import importlib.util
import pathlib
import sys
import types
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

pytestmark = [pytest.mark.cpu_only, pytest.mark.post_training]

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_DIFFUSION_GENERATE = (
    _REPO_ROOT / "src" / "maxtext" / "trainers" / "post_train" / "distillation" / "diffusion_generate.py"
)
_DIFFUSION_ROLLOUT = _REPO_ROOT / "src" / "maxtext" / "integration" / "vllm" / "maxtext_diffusion_rollout.py"
_TRAIN_RL = _REPO_ROOT / "src" / "maxtext" / "trainers" / "post_train" / "rl" / "train_rl.py"


def _load_by_path(name, path):
  spec = importlib.util.spec_from_file_location(name, path)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


# ---- Load diffusion_generate standalone (imports only jax) ------------------
_DG = _load_by_path("dgr2_diffusion_generate_under_test", _DIFFUSION_GENERATE)
diffusion_per_token_logps = _DG.diffusion_per_token_logps


def _install_stub_base_rollout():
  """Minimal ``tunix.rl.rollout.base_rollout`` so the rollout module loads bare."""

  @dataclasses.dataclass
  class RolloutConfig:
    max_tokens_to_generate: int = 8
    max_prompt_length: int = 8
    temperature: float = 1.0
    top_p: float | None = 1.0
    top_k: int | None = None
    seed: Any = None

  @dataclasses.dataclass
  class RolloutOutput:
    text: list
    logits: Any
    tokens: list
    left_padded_prompt_tokens: Any
    logprobs: Any

  @dataclasses.dataclass
  class CacheConfig:
    cache_size: int = 0
    num_layers: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0

  class BaseRollout(abc.ABC):
    pass

  base = types.ModuleType("tunix.rl.rollout.base_rollout")
  base.RolloutConfig = RolloutConfig
  base.RolloutOutput = RolloutOutput
  base.CacheConfig = CacheConfig
  base.BaseRollout = BaseRollout
  for pkg in ("tunix", "tunix.rl", "tunix.rl.rollout"):
    if pkg not in sys.modules:
      m = types.ModuleType(pkg)
      m.__path__ = []
      sys.modules[pkg] = m
  sys.modules["tunix.rl.rollout.base_rollout"] = base
  sys.modules["tunix.rl.rollout"].base_rollout = base
  return base


def _install_stub_maxtext_diffusion_generate():
  """Register the standalone ``diffusion_generate`` under its maxtext dotted path."""
  dotted = "maxtext.trainers.post_train.distillation"
  parts = dotted.split(".")
  for i in range(1, len(parts) + 1):
    name = ".".join(parts[:i])
    if name not in sys.modules:
      m = types.ModuleType(name)
      m.__path__ = []
      sys.modules[name] = m
  sys.modules[dotted].diffusion_generate = _DG
  sys.modules[dotted + ".diffusion_generate"] = _DG


_BASE = _install_stub_base_rollout()
_install_stub_maxtext_diffusion_generate()
_DR = _load_by_path("dgr2_maxtext_diffusion_rollout_under_test", _DIFFUSION_ROLLOUT)
MaxTextDiffusionRollout = _DR.MaxTextDiffusionRollout
make_diffusion_per_token_logps_fn = _DR.make_diffusion_per_token_logps_fn


# ---- Test constants and stubs ----------------------------------------------
VOCAB = 16
MASK_ID = 15  # outside the normal token range so leaks are detectable
PAD_ID = 0
EOS_ID = 9


def _target_map(n: int) -> jnp.ndarray:
  """Deterministic absolute-position -> token map in [1, VOCAB-3]."""
  return jnp.asarray([1 + (p % (VOCAB - 3)) for p in range(n)], dtype=jnp.int32)


class _StubModel(nnx.Module):
  """Adapter-shaped stub: ``(tokens, positions, cache, attn_mask) -> (logits, None)``.

  Logits peak on ``target[positions]`` (canvas-independent, so argmax/logps are
  exact and position-addressed), scaled by ``scale``. Carries one real nnx.Param so
  it round-trips through ``nnx.split``/``nnx.merge`` (needed by the loss-side hook).
  """

  def __init__(self, target: jnp.ndarray, scale: float):
    self.target = nnx.Variable(jnp.asarray(target, dtype=jnp.int32))
    self.bias = nnx.Param(jnp.zeros((), dtype=jnp.float32))
    self.scale = float(scale)
    self.vocab = int(VOCAB)

  def __call__(self, input_tokens, positions, cache=None, attention_mask=None, decoder_segment_ids=None):
    del input_tokens, cache, attention_mask, decoder_segment_ids
    tgt = self.target[...][positions]  # [B, T]
    logits = jax.nn.one_hot(tgt, self.vocab, dtype=jnp.float32) * self.scale + self.bias[...]
    return logits, None


class _StubTokenizer:
  """Whitespace tokenizer over integer strings, e.g. "3 4 5" <-> [3, 4, 5]."""

  def encode(self, text: str):
    return [int(t) for t in text.split()] if text.strip() else []

  def decode(self, ids):
    return " ".join(str(int(i)) for i in ids)

  def pad_id(self):
    return PAD_ID

  def eos_id(self):
    return EOS_ID


def _make_config(bd_size=4, mask_id=MASK_ID, threshold=0.9, max_denoise_steps=0):
  return types.SimpleNamespace(
      enable_block_diffusion=True,
      bd_size=bd_size,
      mask_id=mask_id,
      rl=types.SimpleNamespace(
          rl_diffusion_threshold=threshold,
          rl_diffusion_max_denoise_steps=max_denoise_steps,
      ),
  )


def _make_rollout(scale=30.0, bd_size=4, temperature=1.0, target_len=64):
  model = _StubModel(_target_map(target_len), scale=scale)
  rollout_config = _BASE.RolloutConfig(max_tokens_to_generate=4, max_prompt_length=4, temperature=temperature)
  rollout = MaxTextDiffusionRollout(
      rollout_actor=model,
      tokenizer=_StubTokenizer(),
      mesh=None,
      rollout_config=rollout_config,
      maxtext_config=_make_config(bd_size=bd_size),
  )
  return rollout, model, rollout_config


# ============================================================================
# (a) shared block-diffusion per-token logprob fn
# ============================================================================
class TestSharedLogprobFn:

  def _model_apply(self, scale=30.0, target_len=64):
    g = _target_map(target_len)

    def model_apply(tokens, positions):
      del tokens
      return jax.nn.one_hot(g[positions], VOCAB, dtype=jnp.float32) * scale

    return model_apply, g

  def test_shape(self):
    fwd, _ = self._model_apply()
    prompt = jnp.array([[1, 2, 3, 4], [1, 1, 2, 2]], dtype=jnp.int32)
    completion = jnp.array([[3, 4, 5, 6], [2, 3, 4, 5]], dtype=jnp.int32)
    logps = diffusion_per_token_logps(fwd, prompt, completion, None, bd_size=4)
    assert logps.shape == (2, 4)

  def test_shifted_convention_matches_hand_computed(self):
    # positions = 0..7 (all non-pad). Post-shift logits at full position p score
    # token p from logits[p-1], which peaks on g[p-1]. So a completion token equal
    # to g[p-1] gets logp ~ 0; anything else ~ -scale.
    scale = 30.0
    fwd, g = self._model_apply(scale=scale)
    prompt = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)  # full positions 0..3
    aligned = jnp.array([[int(g[3]), int(g[4]), int(g[5]), int(g[6])]], dtype=jnp.int32)
    logps = np.asarray(diffusion_per_token_logps(fwd, prompt, aligned, None, bd_size=4))
    assert np.allclose(logps, 0.0, atol=1e-4), logps

    # Break one token -> that position drops to ~ -scale, others stay ~ 0.
    broken = aligned.at[0, 2].set((int(g[5]) + 1) % (VOCAB - 3) + 1)
    logps_b = np.asarray(diffusion_per_token_logps(fwd, prompt, broken, None, bd_size=4))
    assert np.allclose(logps_b[0, [0, 1, 3]], 0.0, atol=1e-4)
    assert logps_b[0, 2] < -1.0

  def test_matches_manual_log_softmax(self):
    # Small vocab, moderate scale so the log-softmax is non-degenerate; compare to
    # a fully manual computation of the same shifted-logit log-prob.
    scale = 1.3
    fwd, g = self._model_apply(scale=scale)
    prompt = jnp.array([[2, 3, 4, 5]], dtype=jnp.int32)
    completion = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
    logps = np.asarray(diffusion_per_token_logps(fwd, prompt, completion, None, bd_size=4))
    full = np.concatenate([np.asarray(prompt)[0], np.asarray(completion)[0]])  # [8]
    for i in range(4):
      p = 4 + i  # full position of completion token i
      logits_row = np.asarray(jax.nn.one_hot(int(g[p - 1]), VOCAB)) * scale  # logits[p-1]
      manual = logits_row[full[p]] - np.log(np.sum(np.exp(logits_row)))
      assert abs(logps[0, i] - manual) < 1e-4, (i, logps[0, i], manual)

  def test_temperature_scales_logits(self):
    fwd, g = self._model_apply(scale=2.0)
    prompt = jnp.array([[2, 3, 4, 5]], dtype=jnp.int32)
    completion = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
    a = np.asarray(diffusion_per_token_logps(fwd, prompt, completion, None, bd_size=4, temperature=1.0))
    b = np.asarray(diffusion_per_token_logps(fwd, prompt, completion, None, bd_size=4, temperature=2.0))
    assert not np.allclose(a, b)  # temperature changes the distribution

  def test_completion_mask_zeroes_masked_positions(self):
    fwd, _ = self._model_apply()
    prompt = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
    completion = jnp.array([[3, 4, 5, 6]], dtype=jnp.int32)
    mask = jnp.array([[1, 1, 0, 0]], dtype=jnp.int32)
    logps = np.asarray(diffusion_per_token_logps(fwd, prompt, completion, mask, bd_size=4))
    assert np.all(logps[0, 2:] == 0.0)

  def test_requires_divisible_length(self):
    fwd, _ = self._model_apply()
    prompt = jnp.array([[1, 2, 3]], dtype=jnp.int32)  # 3 + 4 = 7, not divisible by 4
    completion = jnp.array([[3, 4, 5, 6]], dtype=jnp.int32)
    with pytest.raises(ValueError):
      diffusion_per_token_logps(fwd, prompt, completion, None, bd_size=4)


# ============================================================================
# (b) MaxTextDiffusionRollout.generate
# ============================================================================
class TestDiffusionRolloutGenerate:

  def test_generate_produces_valid_completion(self):
    rollout, _, cfg = _make_rollout(scale=30.0, bd_size=4)
    out = rollout.generate(["1 2 3 4", "2 3"], cfg)
    assert len(out.text) == 2
    assert len(out.tokens) == 2
    # Prompt echoed, left-padded to max_prompt_length=4.
    assert out.left_padded_prompt_tokens.shape == (2, 4)
    assert out.left_padded_prompt_tokens[0].tolist() == [1, 2, 3, 4]
    assert out.left_padded_prompt_tokens[1].tolist() == [PAD_ID, PAD_ID, 2, 3]
    for toks in out.tokens:
      toks = np.asarray(toks)
      assert toks.size >= 1
      assert not np.any(toks == MASK_ID)  # no mask token leaks into the output
      # EOS-truncated: EOS may appear only as the final token, if at all.
      if np.any(toks == EOS_ID):
        assert int(toks[-1]) == EOS_ID
        assert not np.any(toks[:-1] == EOS_ID)

  def test_requires_block_diffusion_enabled(self):
    model = _StubModel(_target_map(64), scale=30.0)
    cfg = _make_config()
    cfg.enable_block_diffusion = False
    with pytest.raises(ValueError):
      MaxTextDiffusionRollout(
          rollout_actor=model,
          tokenizer=_StubTokenizer(),
          mesh=None,
          rollout_config=_BASE.RolloutConfig(temperature=1.0),
          maxtext_config=cfg,
      )


# ============================================================================
# (c) THE P0 INVARIANT
# ============================================================================
class TestP0Invariant:

  def test_rollout_and_loss_hook_are_same_fn(self):
    # Both sites funnel through the SAME single-source-of-truth object.
    assert _DR.diffusion_generate.diffusion_per_token_logps is diffusion_per_token_logps
    hook = make_diffusion_per_token_logps_fn(bd_size=4, mask_id=MASK_ID)
    assert callable(hook)

  def test_rollout_logps_equals_loss_hook_logps(self):
    """Old-policy (rollout) logps == new-policy (loss hook) logps, identically."""
    rollout, model, _ = _make_rollout(scale=3.0, bd_size=4, temperature=1.0)
    prompt = jnp.array([[1, 2, 3, 4], [1, 1, 2, 2]], dtype=jnp.int32)
    completion = jnp.array([[3, 4, 5, 6], [2, 3, 4, 5]], dtype=jnp.int32)
    completion_mask = jnp.array([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=jnp.int32)

    # Site 1: rollout old-policy logps.
    old_logps = np.asarray(rollout.get_per_token_logps(prompt, completion, completion_mask))

    # Site 2: loss-side new-policy logps via the pluggable hook (drop-in for
    # common.compute_per_token_logps: called with graphdef, state, ...).
    hook = make_diffusion_per_token_logps_fn(bd_size=4, mask_id=MASK_ID)
    graphdef, state = nnx.split(model)
    new_logps = np.asarray(
        hook(
            graphdef,
            state,
            prompt_tokens=prompt,
            completion_tokens=completion,
            pad_id=PAD_ID,
            eos_id=EOS_ID,
            completion_mask=completion_mask,
            stop_gradient=False,
            temperature=1.0,
        )
    )

    assert old_logps.shape == new_logps.shape == (2, 4)
    np.testing.assert_array_equal(old_logps, new_logps)

  def test_consistency_holds_under_temperature(self):
    rollout, model, _ = _make_rollout(scale=2.5, bd_size=4, temperature=0.7)
    prompt = jnp.array([[5, 6, 7, 8]], dtype=jnp.int32)
    completion = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
    old_logps = np.asarray(rollout.get_per_token_logps(prompt, completion))
    hook = make_diffusion_per_token_logps_fn(bd_size=4, mask_id=MASK_ID)
    graphdef, state = nnx.split(model)
    new_logps = np.asarray(
        hook(
            graphdef,
            state,
            prompt_tokens=prompt,
            completion_tokens=completion,
            pad_id=PAD_ID,
            eos_id=EOS_ID,
            temperature=0.7,
        )
    )
    np.testing.assert_array_equal(old_logps, new_logps)


# ============================================================================
# Source-structure checks (train_rl wiring)
# ============================================================================
class TestTrainRlWiring:

  def test_train_rl_gates_diffusion_rollout(self):
    src = _TRAIN_RL.read_text()
    assert "rl_diffusion_rollout" in src
    assert "MaxTextDiffusionRollout" in src
    assert "make_diffusion_per_token_logps_fn" in src
    # Diffusion rollout is gated and requires block diffusion.
    assert "use_diffusion_rollout" in src
    assert "enable_block_diffusion" in src
    # The learner receives the shared logprob hook.
    assert "per_token_logps_fn=diffusion_per_token_logps_fn" in src

  def test_diffusion_rollout_is_not_a_vllm_subclass(self):
    # Fresh JAX rollout (BaseRollout), NOT a VllmRollout subclass.
    src = _DIFFUSION_ROLLOUT.read_text()
    assert "class MaxTextDiffusionRollout(base_rollout.BaseRollout)" in src
    assert "vllm_rollout.VllmRollout" not in src  # does not subclass the vLLM rollout
    assert "diffusion_generate.diffusion_generate(" in src
