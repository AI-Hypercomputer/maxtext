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

"""CompletedP (`completedp_base_model_name`) for OLMoE3 against OLMo-core's CompletedPConfig in base-model mode.

Roles, init factors, lr / eps / weight-decay multipliers, betas and residual multipliers are checked against the
values OLMo-core computes for the same base and target models (tests/assets/olmoe3_completedp_olmo_core.json); the
init, forward pass and optimizer against what those values mean.
"""

# pylint: disable=missing-function-docstring,protected-access

import json
import os
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import optax

from maxtext.models import olmoe3_roles as roles
from maxtext.optimizers import completedp
from maxtext.optimizers import optimizers
from maxtext.utils.globals import MAXTEXT_REPO_ROOT
import olmoe3_test as t

BASE = "olmo35-tiny-e256k8-sh2x"
TARGET = "olmo35-small-e256k8-sh2x"
# Reduced (CPU-sized, every width kept): 3 layers -- dense KDA, MoE attention, MoE KDA -- and 8 experts.
REDUCED = (
    "override_model_config=True",
    "base_num_decoder_layers=3",
    "inhomogeneous_layer_cycle_interval=2",
    "num_experts=8",
    "num_experts_per_tok=2",
    "vocab_size=1024",
    "emo_enabled=False",
)
# The reduced model's layers and the full model's layer of the same kind (golden names).
REDUCED_LAYER = {"layers_0/": "layers_0/", "layers_1/": "layers_7/", "layers_2/": "layers_1/"}
TRUNC = 0.986534  # std of a normal truncated at +/-3 std, relative to the untruncated one


def _name(path):
  return roles.path_str(path).removeprefix("params/")


def _config(model, base=BASE, extra=()):
  return t._config(model, extra=(f"completedp_base_model_name={base}", "steps=8", *extra))


def _params(cfg, init=False):
  _, model = t._build(cfg)
  if not init:
    return t._abstract_params(cfg, model, 64)
  rng = jax.random.PRNGKey(0)
  ids = jnp.ones((1, 64), dtype=jnp.int32)
  return model.init({"params": rng, "dropout": rng}, ids, ids, enable_dropout=False)["params"]


def _case_extra(cfg, case):
  """completedp_base_* for a golden optimizer case: m_B = batch_ratio, m_D = duration_ratio."""
  batch, tokens = completedp.run_tokens(cfg)
  return (
      f"completedp_base_batch_tokens={int(batch / case['batch_ratio'])}",
      f"completedp_base_train_tokens={int(tokens / case['duration_ratio'])}",
  )


class CompletedPGoldenTest(unittest.TestCase):
  """Roles, multipliers, betas and residual multipliers against OLMo-core's."""

  @classmethod
  def setUpClass(cls):
    with open(os.path.join(MAXTEXT_REPO_ROOT, "tests", "assets", "olmoe3_completedp_olmo_core.json"), encoding="utf-8") as f:
      cls.golden = json.load(f)

  def _check(self, model, alpha):
    golden = self.golden["models"][f"{model}@{alpha}"]
    for case_name, case in self.golden["optim_cases"].items():
      cfg = _config(model, extra=(f"completedp_alpha={alpha}",))
      cfg = _config(model, extra=(f"completedp_alpha={alpha}", *_case_extra(cfg, case)))
      base = roles.load_base_config(BASE)
      params = _params(cfg)
      got_roles = {_name(p): r.role for p, r in jax.tree_util.tree_flatten_with_path(
          roles.classify_tree(cfg, params), is_leaf=lambda x: isinstance(x, roles.ParamRole))[0]}
      self.assertEqual(sorted(got_roles), sorted(golden["params"]))
      lr, eps, wd = (
          {_name(p): m for p, m in jax.tree_util.tree_flatten_with_path(tree)[0]}
          for tree in completedp.multipliers(cfg, base, params)
      )
      offset = 2 if case_name == "none" else 5
      for name, row in golden["params"].items():
        self.assertEqual(got_roles[name], row[0], name)
        for got, expected, what in zip((lr[name], eps[name], wd[name]), row[offset : offset + 3], ("lr", "eps", "wd")):
          self.assertAlmostEqual(got, expected, places=12, msg=f"{name} {what} ({case_name})")
      self.assertEqual(list(completedp.betas(cfg, base)), case["betas"])
      self.assertAlmostEqual(completedp.residual_multiplier(cfg, base), golden["residual"], places=12)
    return golden

  def test_target_alpha_1(self):
    self._check(TARGET, 1.0)

  def test_target_alpha_half(self):
    self._check(TARGET, 0.5)

  def test_base_model(self):
    golden = self._check(BASE, 1.0)
    self.assertEqual({row[2:5] == [1.0, 1.0, 1.0] for row in golden["params"].values()}, {True})

  def test_init_factors(self):
    # The factor each in-layer matrix's init gets, against OLMo-core's (apply_completedp_init_'s rule).
    golden = self.golden["models"][f"{TARGET}@1.0"]["params"]
    cfg = _config(TARGET)
    base = roles.load_base_config(BASE)
    for name, row in golden.items():
      if row[0] not in roles.MATRIX_ROLES:
        self.assertEqual(row[1], 1.0, name)
        continue
      if name == "decoder/logits_dense/kernel":
        sub, dense = roles.LM_HEAD_KEY, False
      else:
        sub, dense = name.split("/", 2)[2], name.startswith("decoder/layers_0/")
        sub = {"moe_block/wi_1": "moe_block/wi_0"}.get(sub, sub)
      self.assertAlmostEqual(roles.completedp_init_factor(cfg, base, sub, dense), row[1], places=12, msg=name)


class CompletedPModelTest(unittest.TestCase):
  """The init and the residual multipliers in the model."""

  def test_init_stds(self):
    # Every parameter of a reduced small-sh2x: 0.02 x its golden init factor (truncated), vectors as usual.
    with open(os.path.join(MAXTEXT_REPO_ROOT, "tests", "assets", "olmoe3_completedp_olmo_core.json"), encoding="utf-8") as f:
      golden = json.load(f)["models"][f"{TARGET}@1.0"]["params"]
    cfg = _config(TARGET, extra=REDUCED)
    for path, w in jax.tree_util.tree_flatten_with_path(_params(cfg, init=True))[0]:
      name = _name(path)
      for reduced, full in REDUCED_LAYER.items():
        name = name.replace(f"decoder/{reduced}", f"decoder/{full}")
      if name == "token_embedder/embedding":
        name_factor = 1.0
      elif name in golden:
        name_factor = golden[name][1]
      else:
        self.fail(f"{name} not in the golden file")
      w = np.asarray(w, np.float64)
      role = golden.get(name, ["input_embedding"])[0]
      if role in roles.MATRIX_ROLES or role == roles.INPUT_EMBEDDING:
        std = 0.02 * name_factor
        self.assertLess(abs(w.std() / (std * TRUNC) - 1), 0.05, f"{name}: std {w.std():.5f}, expected {std * TRUNC:.5f}")
        self.assertLessEqual(np.abs(w).max(), 3 * std * (1 + 1e-6), name)

  def test_residual_multiplier_in_the_layer(self):
    # Both residual branches of a layer are scaled by the multiplier: out = x + c * attn + c * ffn.
    cfg = _config(TARGET, extra=REDUCED)
    _, model = t._build(cfg)
    del model
    from maxtext.models import olmoe3  # pylint: disable=import-outside-toplevel
    from flax import nnx  # pylint: disable=import-outside-toplevel
    from jax.sharding import Mesh  # pylint: disable=import-outside-toplevel
    from maxtext.utils import maxtext_utils  # pylint: disable=import-outside-toplevel

    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    layer = olmoe3.OLMoE3DecoderLayer(cfg, mesh, "train", layer_idx=2, rngs=nnx.Rngs(0))
    self.assertAlmostEqual(layer.residual_multiplier, (3 / 16) ** -1.0)
    branches = {}
    for norm in ("attn_out_norm", "ffn_out_norm"):
      inner = getattr(layer, norm)

      def record(x, inner=inner, norm=norm):
        branches[norm] = inner(x)
        return branches[norm]

      object.__setattr__(layer, norm, record)
    x = jax.random.normal(jax.random.PRNGKey(1), (1, 64, cfg.emb_dim), jnp.float32)
    seg = jnp.ones((1, 64), jnp.int32)
    out = layer(x, seg, jnp.arange(64)[None], deterministic=True, model_mode="train")
    out = out[0] if isinstance(out, tuple) else out
    c = layer.residual_multiplier
    np.testing.assert_allclose(out, x + c * branches["attn_out_norm"] + c * branches["ffn_out_norm"], rtol=1e-5, atol=1e-5)

  def test_rejects_unsupported_runs(self):
    for extra, match in (
        (("olmo_init_method=nanochat",), "init"),
        (("mup_base_model_name=" + BASE,), "mup"),
        (("scan_layers=True",), "scan_layers"),
        (("completedp_alpha=0.3",), "alpha"),
        (("opt_type=sgd",), "adamw"),
    ):
      with self.assertRaisesRegex(ValueError, match):
        completedp.check_supported(_config(TARGET, extra=extra))


class CompletedPOptimizerTest(unittest.TestCase):
  """completedp_adamw's updates: AdamW with each parameter's lr x m_lr, eps x m_eps, wd x m_wd and scaled betas."""

  LR = 3e-3
  WD = 0.1
  EPS = 1e-8

  @classmethod
  def setUpClass(cls):
    extra = (*REDUCED, "adamw_mask=['token_embedder/embedding']", f"adam_weight_decay={cls.WD}", "adam_b1=0.9",
             "adam_b2=0.95", f"adam_eps={cls.EPS}")
    cfg = _config(TARGET, extra=extra)
    # m_B = 2, m_D = 4: betas (0.95, 0.975), and every multiplier also carries s = sqrt(1/2).
    cls.cfg = _config(TARGET, extra=(*extra, *_case_extra(cfg, {"batch_ratio": 2.0, "duration_ratio": 4.0})))
    cls.base = roles.load_base_config(BASE)
    cls.params = _params(cls.cfg, init=True)
    cls.lr_m, cls.eps_m, cls.wd_m = completedp.multipliers(cls.cfg, cls.base, cls.params)
    cls.mask = optimizers.get_adamw_mask(cls.cfg)(cls.params)

  def _tx(self):
    return completedp.completedp_adamw(self.LR, self.cfg, self.base, mask=optimizers.get_adamw_mask(self.cfg))

  def _leaves(self, *trees):
    return zip(*(jax.tree_util.tree_leaves(x) for x in trees))

  def test_betas(self):
    self.assertEqual(completedp.betas(self.cfg, self.base), (0.95, 0.975))

  def test_all_ones_is_optax_adamw_bitwise(self):
    # The base as its own model (full depth, so m_L = 1) and no batch/duration change: every multiplier is 1, so
    # completedp_adamw must be exactly optax.adamw, state included. Params are small constants (no init needed).
    cfg = _config(BASE, extra=("override_model_config=True", "num_experts=8", "num_experts_per_tok=2", "vocab_size=1024",
                               "emo_enabled=False", "adamw_mask=['token_embedder/embedding']"))
    base = roles.load_base_config(BASE)
    params = jax.tree.map(lambda x: jnp.full(x.shape, 0.01, x.dtype), _params(cfg))
    self.assertEqual({m for tree in completedp.multipliers(cfg, base, params) for m in jax.tree_util.tree_leaves(tree)}, {1.0})
    grads = jax.tree.map(lambda p: jax.random.normal(jax.random.PRNGKey(7), p.shape, p.dtype), params)
    mask = optimizers.get_adamw_mask(cfg)
    reference = optax.adamw(self.LR, b1=cfg.adam_b1, b2=cfg.adam_b2, eps=cfg.adam_eps, eps_root=cfg.adam_eps_root,
                            weight_decay=cfg.adam_weight_decay, mu_dtype=cfg.mu_dtype, mask=mask)
    tx = completedp.completedp_adamw(self.LR, cfg, base, mask=mask)
    states = [tx.init(params), reference.init(params)]
    self.assertEqual(jax.tree.structure(states[0]), jax.tree.structure(states[1]))
    p_cp, p_ref = params, params
    for _ in range(2):
      u_cp, states[0] = tx.update(grads, states[0], p_cp)
      u_ref, states[1] = reference.update(grads, states[1], p_ref)
      p_cp, p_ref = optax.apply_updates(p_cp, u_cp), optax.apply_updates(p_ref, u_ref)
    for a, b in self._leaves(p_cp, p_ref):
      np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

  def test_decay_applied_by_a_zero_gradient_step(self):
    # Zero gradients: Adam's update is exactly 0, so the step is the decay alone, lr m_lr wd m_wd p.
    zeros = jax.tree.map(jnp.zeros_like, self.params)
    updates, _ = self._tx().update(zeros, self._tx().init(self.params), self.params)
    for u, p, m_lr, m_wd, decayed in self._leaves(updates, self.params, self.lr_m, self.wd_m, self.mask):
      expected = -self.LR * m_lr * self.WD * m_wd * np.asarray(p, np.float64) if decayed else 0.0
      np.testing.assert_allclose(np.asarray(u, np.float64), expected, rtol=1e-5, atol=1e-14)

  def test_first_step_uses_each_parameters_eps_and_the_scaled_betas(self):
    # Gradients near eps, so each parameter's epsilon matters: -lr m_lr (g / (|g| + eps m_eps) + wd m_wd p). The
    # moments after one step are (1 - b1') g and (1 - b2') g^2 with the scaled betas.
    grads = jax.tree.map(lambda p: 1e-8 * jax.random.normal(jax.random.PRNGKey(3), p.shape, p.dtype), self.params)
    tx = self._tx()
    updates, state = tx.update(grads, tx.init(self.params), self.params)
    adam = state[0]
    for u, g, p, mu, nu, m_lr, m_eps, m_wd, decayed in self._leaves(
        updates, grads, self.params, adam.mu, adam.nu, self.lr_m, self.eps_m, self.wd_m, self.mask
    ):
      g, p = np.asarray(g, np.float64), np.asarray(p, np.float64)
      np.testing.assert_allclose(np.asarray(mu, np.float64), 0.05 * g, rtol=1e-5, atol=1e-20)
      np.testing.assert_allclose(np.asarray(nu, np.float64), 0.025 * g**2, rtol=1e-4, atol=1e-30)
      wd = self.WD * m_wd if decayed else 0.0
      expected = -self.LR * m_lr * (g / (np.abs(g) + self.EPS * m_eps) + wd * p)
      # The Adam term and the decay can nearly cancel, so the float32 error is relative to the terms (~lr m_lr).
      np.testing.assert_allclose(np.asarray(u, np.float64), expected, rtol=1e-4, atol=1e-6 * self.LR * m_lr)

  def test_get_optimizer_builds_completedp(self):
    cfg = _config(TARGET, extra=(*REDUCED, "skip_step_on_spikes=True"))
    tx = optimizers.get_optimizer(cfg, optax.constant_schedule(self.LR))
    params = _params(cfg, init=True)
    updates, _ = jax.jit(tx.update)(jax.tree.map(jnp.ones_like, params), tx.init(params), params)
    self.assertEqual(jax.tree.structure(updates), jax.tree.structure(params))


if __name__ == "__main__":
  unittest.main()
