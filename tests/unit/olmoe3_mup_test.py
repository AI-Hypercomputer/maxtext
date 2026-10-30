# Copyright 2023-2026 Google LLC
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

"""muP (`mup_base_model_name`) for OLMoE3 against OLMo-core's MuPConfig in base-model mode.

The multipliers are checked against the ones OLMo-core computes for the same base and target models
(tests/assets/olmoe3_mup_multipliers_olmo_core.json); the optimizer against AdamW's arithmetic with
OLMo-core's per-group learning rate `lr * m` and weight decay `wd / m` (scaled) or `wd` (coupled).
"""

# pylint: disable=missing-function-docstring,protected-access

import collections
import json
import os
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import optax

from maxtext.optimizers import mup
from maxtext.optimizers import optimizers
from maxtext.utils.globals import MAXTEXT_REPO_ROOT
import olmoe3_test as t

BASE = "olmo35-tiny-e256k8-sh2x"
TARGET = "olmo35-small-e256k8-sh2x"
# Small enough to init on CPU and keeping every width: 3 layers (dense KDA, MoE attention, MoE KDA), 8 experts.
REDUCED = (
    "override_model_config=True",
    "base_num_decoder_layers=3",
    "inhomogeneous_layer_cycle_interval=2",
    "num_experts=8",
    "num_experts_per_tok=2",
    "vocab_size=1024",
    "emo_enabled=False",
)


def _name(path):
  """The golden file's names: the parameter path without the leading `params/`."""
  return mup._path_str(path).removeprefix("params/")


def _config(model, base=BASE, extra=()):
  return t._config(model, extra=(f"mup_base_model_name={base}", *extra))


def _params(cfg, init=False):
  _, model = t._build(cfg)
  if not init:
    return t._abstract_params(cfg, model, 64)
  rng = jax.random.PRNGKey(0)
  ids = jnp.ones((1, 64), dtype=jnp.int32)
  return model.init({"params": rng, "dropout": rng}, ids, ids, enable_dropout=False)["params"]


class MuPMultiplierTest(unittest.TestCase):
  """The multipliers against OLMo-core's."""

  @classmethod
  def setUpClass(cls):
    with open(os.path.join(MAXTEXT_REPO_ROOT, "tests", "assets", "olmoe3_mup_multipliers_olmo_core.json"), encoding="utf-8") as f:
      cls.golden = json.load(f)

  def _check_against_olmo_core(self, model):
    cfg = _config(model)
    multipliers = mup.lr_multipliers(cfg, mup.load_base_config(cfg), _params(cfg))
    got = {_name(path): m for path, m in jax.tree_util.tree_flatten_with_path(multipliers)[0]}
    expected = self.golden[model]
    self.assertEqual(sorted(got), sorted(expected))
    for name, m in got.items():
      self.assertAlmostEqual(m, expected[name], places=12, msg=name)
    return got

  def test_target_matches_olmo_core(self):
    got = self._check_against_olmo_core(TARGET)
    self.assertEqual(
        collections.Counter(round(m, 6) for m in got.values()), collections.Counter({0.666667: 580, 1.0: 423, 0.5: 110})
    )

  def test_base_model_is_all_ones(self):
    got = self._check_against_olmo_core(BASE)
    self.assertEqual(set(got.values()), {1.0})

  def test_embedding_multiplier(self):
    cfg = _config(TARGET, extra=("mup_embedding_lr_multiplier=3.0",))
    multipliers = mup.lr_multipliers(cfg, mup.load_base_config(cfg), _params(cfg))
    got = {_name(path): m for path, m in jax.tree_util.tree_flatten_with_path(multipliers)[0]}
    self.assertEqual(got["token_embedder/embedding"], 3.0)
    self.assertEqual(got["decoder/logits_dense/kernel"], 1024 / 1536)

  def test_rejects_a_layer_kind_the_base_lacks(self):
    cfg = _config(TARGET)
    # A base whose layers are all KDA: the target's attention layers have no base role.
    base = mup.load_base_config(cfg)
    no_attention = collections.namedtuple("Base", ["inhomogeneous_layer_cycle_interval"])(10_000)
    fields = {k: getattr(base, k) for k in ("emb_dim", "gdn_value_head_dim", "gdn_num_value_heads", "num_query_heads",
                                            "head_dim", "moe_expert_input_dim", "moe_mlp_dim", "mlp_dim",
                                            "shared_expert_mlp_dim", "first_num_dense_layers", "num_decoder_layers")}
    base = collections.namedtuple("Base", [*fields, "inhomogeneous_layer_cycle_interval"])(
        **fields, inhomogeneous_layer_cycle_interval=no_attention.inhomogeneous_layer_cycle_interval
    )
    with self.assertRaisesRegex(ValueError, "moe\\+attention"):
      mup.lr_multipliers(cfg, base, _params(cfg))

  def test_rejects_parameters_without_a_rule(self):
    cfg = _config(TARGET)
    with self.assertRaisesRegex(NotImplementedError, "no parametrization rule"):
      mup.lr_multipliers(cfg, mup.load_base_config(cfg), {"decoder": {"layers_1": {"mixer": {"w_new": {"kernel": jnp.zeros((4, 4))}}}}})

  def test_rejects_a_kernel_whose_shape_disagrees(self):
    cfg = _config(TARGET)
    bad = {"decoder": {"layers_1": {"latent_up": {"kernel": jnp.zeros((cfg.moe_expert_input_dim + 1, cfg.emb_dim))}}}}
    with self.assertRaisesRegex(ValueError, "shape"):
      mup.lr_multipliers(cfg, mup.load_base_config(cfg), bad)

  def test_unsupported_runs_raise(self):
    for extra, match in ((("opt_type=sgd",), "adamw"), (("scan_layers=True",), "scan_layers")):
      with self.assertRaisesRegex(ValueError, match):
        mup.check_supported(_config(TARGET, extra=extra))


class MuPOptimizerTest(unittest.TestCase):
  """mup_adamw's updates: OLMo-core's AdamW groups with lr * m and wd / m (scaled) or wd (coupled)."""

  LR = 3e-3
  WD = 0.1

  @classmethod
  def setUpClass(cls):
    cls.cfg = {}
    for scaled in (True, False):
      extra = (*REDUCED, f"mup_scale_weight_decay={scaled}", "adamw_mask=['token_embedder/embedding']",
               f"adam_weight_decay={cls.WD}", "adam_b1=0.9", "adam_b2=0.95", "adam_eps=1e-8")
      cls.cfg[scaled] = _config(TARGET, extra=extra)
    cls.base = mup.load_base_config(cls.cfg[True])
    cls.params = _params(cls.cfg[True], init=True)
    cls.multipliers = mup.lr_multipliers(cls.cfg[True], cls.base, cls.params)
    cls.mask = optimizers.get_adamw_mask(cls.cfg[True])(cls.params)

  def _step(self, tx, grads, steps=1):
    state = tx.init(self.params)
    params = self.params
    for _ in range(steps):
      updates, state = tx.update(grads, state, params)
      params = optax.apply_updates(params, updates)
    return updates, state

  def _tx(self, scaled, base=None):
    return mup.mup_adamw(self.LR, self.cfg[scaled], base or self.base, mask=optimizers.get_adamw_mask(self.cfg[scaled]))

  def test_multipliers_of_the_reduced_models(self):
    got = collections.Counter(round(m, 6) for m in jax.tree_util.tree_leaves(self.multipliers))
    self.assertEqual(set(got), {0.5, 0.666667, 1.0})

  def test_all_ones_is_optax_adamw_bitwise(self):
    # The target as its own base: every multiplier is 1, so muP must be exactly optax.adamw, state included.
    own_base = mup.load_base_config(_config(BASE, base=BASE, extra=REDUCED))
    cfg = _config(BASE, base=BASE, extra=(*REDUCED, "adamw_mask=['token_embedder/embedding']"))
    params = _params(cfg, init=True)
    grads = jax.tree.map(lambda p: jax.random.normal(jax.random.PRNGKey(7), p.shape, p.dtype), params)
    mask = optimizers.get_adamw_mask(cfg)
    reference = optax.adamw(self.LR, b1=cfg.adam_b1, b2=cfg.adam_b2, eps=cfg.adam_eps, eps_root=cfg.adam_eps_root,
                            weight_decay=cfg.adam_weight_decay, mu_dtype=cfg.mu_dtype, mask=mask)
    for scaled in (True, False):
      cfg_s = _config(BASE, base=BASE, extra=(*REDUCED, "adamw_mask=['token_embedder/embedding']",
                                                  f"mup_scale_weight_decay={scaled}"))
      tx = mup.mup_adamw(self.LR, cfg_s, own_base, mask=mask)
      states = [tx.init(params), reference.init(params)]
      self.assertEqual(jax.tree.structure(states[0]), jax.tree.structure(states[1]))
      p_mup, p_ref = params, params
      for _ in range(3):
        u_mup, states[0] = tx.update(grads, states[0], p_mup)
        u_ref, states[1] = reference.update(grads, states[1], p_ref)
        p_mup, p_ref = optax.apply_updates(p_mup, u_mup), optax.apply_updates(p_ref, u_ref)
      for a, b in zip(jax.tree_util.tree_leaves(p_mup), jax.tree_util.tree_leaves(p_ref)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

  def test_decay_applied_by_a_zero_gradient_step(self):
    # With zero gradients Adam's update is exactly 0, so the step is the decoupled decay alone:
    # lr * wd * p (scaled) or lr * m * wd * p (coupled), and nothing for the embedding (adamw_mask).
    zeros = jax.tree.map(jnp.zeros_like, self.params)
    for scaled in (True, False):
      updates, _ = self._step(self._tx(scaled), zeros)
      for path, u, p, m, decayed in zip(
          [p for p, _ in jax.tree_util.tree_flatten_with_path(self.params)[0]],
          *(jax.tree_util.tree_leaves(x) for x in (updates, self.params, self.multipliers, self.mask)),
      ):
        expected = 0.0 if not decayed else -self.LR * self.WD * (1.0 if scaled else m) * np.asarray(p, np.float64)
        np.testing.assert_allclose(np.asarray(u, np.float64), expected, rtol=1e-5, atol=1e-12, err_msg=_name(path))

  def test_first_adam_step(self):
    # Step 1: Adam's bias-corrected direction is g / (|g| + eps); the update is -lr * m * (direction + wd_p * p).
    grads = jax.tree.map(lambda p: jax.random.normal(jax.random.PRNGKey(3), p.shape, p.dtype), self.params)
    eps = self.cfg[True].adam_eps
    for scaled in (True, False):
      updates, _ = self._step(self._tx(scaled), grads)
      for u, g, p, m, decayed in zip(
          *(jax.tree_util.tree_leaves(x) for x in (updates, grads, self.params, self.multipliers, self.mask))
      ):
        g, p = np.asarray(g, np.float64), np.asarray(p, np.float64)
        wd = (self.WD / m if scaled else self.WD) if decayed else 0.0
        expected = -self.LR * m * (g / (np.abs(g) + eps) + wd * p)
        np.testing.assert_allclose(np.asarray(u, np.float64), expected, rtol=2e-5, atol=1e-9)

  def test_coupled_is_adamw_times_the_multiplier_through_a_schedule(self):
    # Coupled decay is exactly OLMo-core's group lr * m on the whole AdamW update, at every step of a warmup.
    schedule = optax.linear_schedule(0.0, self.LR, transition_steps=5)
    cfg = self.cfg[False]
    mask = optimizers.get_adamw_mask(cfg)
    tx = mup.mup_adamw(schedule, cfg, self.base, mask=mask)
    reference = optax.adamw(schedule, b1=cfg.adam_b1, b2=cfg.adam_b2, eps=cfg.adam_eps, eps_root=cfg.adam_eps_root,
                            weight_decay=cfg.adam_weight_decay, mu_dtype=cfg.mu_dtype, mask=mask)
    grads = jax.tree.map(lambda p: jax.random.normal(jax.random.PRNGKey(5), p.shape, p.dtype), self.params)
    s_mup, s_ref = tx.init(self.params), reference.init(self.params)
    for _ in range(4):
      u_mup, s_mup = tx.update(grads, s_mup, self.params)
      u_ref, s_ref = reference.update(grads, s_ref, self.params)
      for a, b, m in zip(*(jax.tree_util.tree_leaves(x) for x in (u_mup, u_ref, self.multipliers))):
        np.testing.assert_allclose(np.asarray(a), m * np.asarray(b), rtol=1e-6, atol=1e-12)

  def test_get_optimizer_builds_mup(self):
    # Through get_optimizer, with skip-step on as in our runs: init and one jitted update on the real tree.
    cfg = _config(TARGET, extra=(*REDUCED, "skip_step_on_spikes=True"))
    tx = optimizers.get_optimizer(cfg, optax.constant_schedule(self.LR))
    params = _params(cfg, init=True)
    state = tx.init(params)
    grads = jax.tree.map(jnp.ones_like, params)
    updates, _ = jax.jit(tx.update)(grads, state, params)
    self.assertEqual(jax.tree.structure(updates), jax.tree.structure(params))


if __name__ == "__main__":
  unittest.main()
