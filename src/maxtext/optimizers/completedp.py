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

"""CompletedP (Complete(d)P, arXiv:2512.22382 Table 1) for OLMoE3, as OLMo-core's CompletedPConfig in base-model mode.

Ratios relative to a base model and run: ``m_N = emb_dim / base emb_dim``, ``m_L = layers / base layers``, each
matrix's own ``r = fan_in / base fan_in`` (the same matrix in the base model), and ``s = sqrt(m_B / m_D)`` from the
tokens per step and the training tokens. Per parameter role (`olmoe3_roles`), multipliers on the base AdamW values:

  role                       lr                    eps                  weight decay
  input embedding / vector   s                     1 / (m_N s)          s
  hidden matrix              m_L^(a-1) s / r       m_L^-a / (m_N s)     r s
  hidden readout matrix      m_L^(a-1) s / r       m_L^-a / s           r s
  hidden vector              m_L^(a-1) s           m_L^-a / (m_N s)     s
  hidden head-shared vector  m_L^(a-1) s           m_L^-a / s           s
  output matrix (LM head)    s / r                 1 / s                r s
  output vector              s                     1 / s                s

and both betas become ``1 - (1 - beta) m_B / m_D``. The model side (residual multipliers ``m_L^-a`` and the init
scaling) is in models/olmoe3.py. Reference: OLMo-core ``tanushy/parameterization-ladders`` (``optim/completedp.py``,
``nn/transformer/completedp.py``, docs/COMPLETEDP_HANDOFF.md).
"""

import dataclasses

import jax
import jax.numpy as jnp
import optax
from optax._src import base as optax_base
from optax._src import numerics
from optax._src import wrappers as optax_wrappers

from maxtext.models import olmoe3_roles as roles
from maxtext.utils import max_logging


@dataclasses.dataclass(frozen=True)
class Ratios:
  m_n: float
  m_l: float
  batch: float  # m_B
  duration: float  # m_D
  alpha: float

  @property
  def s(self):
    return (self.batch / self.duration) ** 0.5


def check_supported(config):
  """Raises if `completedp_base_model_name` is set on a run CompletedP can't handle."""
  name = "completedp_base_model_name"
  if config.decoder_block.value != "olmoe3":
    raise ValueError(f"{name} is only implemented for the olmoe3 decoder, not {config.decoder_block}.")
  if config.opt_type != "adamw":
    raise ValueError(f"{name} needs opt_type=adamw, not {config.opt_type}.")
  if config.scan_layers:
    raise ValueError(f"{name} needs scan_layers=False (roles are read from the layers_<i> paths).")
  if config.olmo_init_method != "normal":
    raise ValueError(f"{name} rescales the normal init; olmo_init_method={config.olmo_init_method} isn't supported.")
  if config.mup_base_model_name:
    raise ValueError(f"{name} and mup_base_model_name can't both be set.")
  if config.logits_via_embedding:
    raise ValueError(f"{name} needs an untied LM head (the input and output rules differ).")
  if not 0.5 <= config.completedp_alpha <= 1.0:
    raise ValueError(f"completedp_alpha must be in [0.5, 1], got {config.completedp_alpha}.")


def run_tokens(config):
  """(tokens per optimizer step, training tokens) of this run."""
  batch = config.global_batch_size_to_train_on * config.max_target_length
  return batch, batch * config.steps


def ratios(config, base_config):
  batch, tokens = run_tokens(config)
  base_batch = config.completedp_base_batch_tokens or batch
  base_tokens = config.completedp_base_train_tokens or tokens
  return Ratios(
      m_n=config.emb_dim / base_config.emb_dim,
      m_l=config.num_decoder_layers / base_config.num_decoder_layers,
      batch=batch / base_batch,
      duration=tokens / base_tokens,
      alpha=config.completedp_alpha,
  )


def residual_multiplier(config, base_config):
  """The multiplier on every residual branch, ``m_L^-alpha``."""
  return (config.num_decoder_layers / base_config.num_decoder_layers) ** -config.completedp_alpha


def betas(config, base_config):
  """The scaled AdamW betas, ``1 - (1 - beta) m_B / m_D``; raises if one leaves [0, 1)."""
  rt = ratios(config, base_config)
  ratio = rt.batch / rt.duration
  if ratio == 1.0:
    return config.adam_b1, config.adam_b2
  scaled = (1 - (1 - config.adam_b1) * ratio, 1 - (1 - config.adam_b2) * ratio)
  for b in scaled:
    if not 0.0 <= b < 1.0:
      raise ValueError(f"CompletedP scales a beta to {b} (batch / duration ratio {ratio}); it must stay in [0, 1).")
  return scaled


def role_multipliers(role, r, rt):
  """(lr, eps, weight decay) multipliers of a parameter with `role` and width ratio `r` (1 for non-matrices)."""
  hidden_lr = rt.m_l ** (rt.alpha - 1)
  hidden_eps = rt.m_l**-rt.alpha / rt.m_n
  if role in (roles.INPUT_EMBEDDING, roles.INPUT_VECTOR):
    lr, eps, wd = 1.0, 1 / rt.m_n, 1.0
  elif role == roles.HIDDEN_MATRIX:
    lr, eps, wd = hidden_lr / r, hidden_eps, r
  elif role == roles.HIDDEN_READOUT_MATRIX:
    lr, eps, wd = hidden_lr / r, rt.m_l**-rt.alpha, r
  elif role == roles.HIDDEN_VECTOR:
    lr, eps, wd = hidden_lr, hidden_eps, 1.0
  elif role == roles.HIDDEN_HEAD_SHARED_VECTOR:
    lr, eps, wd = hidden_lr, rt.m_l**-rt.alpha, 1.0
  elif role == roles.OUTPUT_MATRIX:
    lr, eps, wd = 1 / r, 1.0, r
  elif role == roles.OUTPUT_VECTOR:
    lr, eps, wd = 1.0, 1.0, 1.0
  else:
    raise NotImplementedError(role)
  s = rt.s
  return lr * s, eps / s, wd * s


def multipliers(config, base_config, params):
  """Three trees like `params`: the lr, eps and weight-decay multiplier of every parameter (Python floats)."""
  rt = ratios(config, base_config)
  base = roles.base_fan_ins(base_config)

  def of(role):
    r = 1.0 if role.fan_in is None else role.fan_in / roles.base_fan_in(base, role, config.completedp_base_model_name)
    return role_multipliers(role.role, r, rt)

  triples = jax.tree.map(of, roles.classify_tree(config, params), is_leaf=lambda x: isinstance(x, roles.ParamRole))
  is_triple = lambda x: isinstance(x, tuple) and len(x) == 3 and all(isinstance(v, float) for v in x)
  return tuple(jax.tree.map(lambda t, i=i: t[i], triples, is_leaf=is_triple) for i in range(3))


def _log(config, base_config, params, lr, eps, wd):
  rt = ratios(config, base_config)
  max_logging.log(
      f"CompletedP vs {config.completedp_base_model_name}: m_N {rt.m_n:.6g}, m_L {rt.m_l:.6g}, m_B {rt.batch:.6g},"
      f" m_D {rt.duration:.6g}, alpha {rt.alpha}, residual multiplier {residual_multiplier(config, base_config):.6g},"
      f" betas {betas(config, base_config)}"
  )
  groups = {}
  leaves = zip(
      jax.tree_util.tree_flatten_with_path(lr)[0], jax.tree_util.tree_leaves(eps), jax.tree_util.tree_leaves(wd),
      jax.tree_util.tree_leaves(params),
  )
  for (path, m_lr), m_eps, m_wd, leaf in leaves:
    g = groups.setdefault((m_lr, m_eps, m_wd), [0, 0, roles.path_str(path)])
    g[0] += 1
    g[1] += int(getattr(leaf, "size", 0))
  for (m_lr, m_eps, m_wd), (n, size, example) in sorted(groups.items()):
    max_logging.log(
        f"CompletedP: lr x{m_lr:.6g}, eps x{m_eps:.6g}, wd x{m_wd:.6g} on {n} parameters ({size:,} elements),"
        f" e.g. {example}"
    )


def _scale_by_adam_per_param_eps(b1, b2, eps, eps_root, mu_dtype, eps_multipliers_fn):
  """optax.scale_by_adam (same state, same arithmetic) with each parameter's epsilon `eps * m`."""
  inner = optax.scale_by_adam(b1=b1, b2=b2, eps=eps, eps_root=eps_root, mu_dtype=mu_dtype)

  def update_fn(updates, state, params=None):
    mu = optax.tree.update_moment(updates, state.mu, b1, 1)
    nu = optax.tree.update_moment_per_elem_norm(updates, state.nu, b2, 2)
    count_inc = numerics.safe_increment(state.count)
    mu_hat = optax.tree.bias_correction(mu, b1, count_inc)
    nu_hat = optax.tree.bias_correction(nu, b2, count_inc)
    eps_m = eps_multipliers_fn(params)
    updates = jax.tree.map(
        lambda m, v, k: None if m is None else m / (jnp.sqrt(v + eps_root) + (eps if k == 1.0 else eps * k)),
        mu_hat,
        nu_hat,
        eps_m,
        is_leaf=lambda x: x is None,
    )
    mu = optax.tree.cast(mu, mu_dtype)
    nu = optax.tree.cast_like(nu, state.nu)
    return updates, optax.ScaleByAdamState(count=count_inc, mu=mu, nu=nu)

  return optax_base.GradientTransformation(inner.init, update_fn)


def _add_decayed_weights_per_param(weight_decay, wd_multipliers_fn):
  """optax.add_decayed_weights with each parameter's decay `wd * m`."""

  def init_fn(params):
    del params
    return optax_base.EmptyState()

  def update_fn(updates, state, params):
    if params is None:
      raise ValueError(optax_base.NO_PARAMS_MSG)
    updates = jax.tree.map(
        lambda g, p, m: None if g is None else g + (weight_decay if m == 1.0 else weight_decay * m) * p,
        updates,
        params,
        wd_multipliers_fn(params),
        is_leaf=lambda x: x is None,
    )
    return updates, state

  return optax_base.GradientTransformation(init_fn, update_fn)


def _scale_by_learning_rate_and_multiplier(learning_rate, lr_multipliers_fn):
  """optax.scale_by_learning_rate (same state), then each parameter's update times its lr multiplier."""
  inner = optax.scale_by_learning_rate(learning_rate)

  def update_fn(updates, state, params=None):
    updates, state = inner.update(updates, state, params)
    updates = jax.tree.map(lambda u, m: u if m == 1.0 else u * m, updates, lr_multipliers_fn(params))
    return updates, state

  return optax_base.GradientTransformation(inner.init, update_fn)


def completedp_adamw(learning_rate, config, base_config, mask=None):
  """optax.adamw with CompletedP's per-parameter lr, eps and weight decay, and its betas.

  The same three transforms as optax.adamw, so the optimizer state has the same structure: Adam (betas scaled,
  per-parameter eps), the (masked) decoupled decay with each parameter's own decay, and the learning rate with
  each parameter's multiplier. Every parameter's update is ``-lr m_lr (adam_eps' + wd m_wd p)``: PyTorch AdamW
  with OLMo-core's CompletedP group lr, eps and weight decay.
  """
  cache = {}

  def mults(params):
    key = jax.tree.structure(params)
    if key not in cache:
      cache[key] = multipliers(config, base_config, params)
      if len(cache) == 1:
        _log(config, base_config, params, *cache[key])
    return cache[key]

  b1, b2 = betas(config, base_config)
  # The decay sees a masked tree (masked-out leaves removed); its multipliers come from that same tree.
  decay = _add_decayed_weights_per_param(config.adam_weight_decay, lambda p: multipliers(config, base_config, p)[2])
  if mask is not None:
    decay = optax_wrappers.masked(decay, mask)
  return optax.chain(
      _scale_by_adam_per_param_eps(b1, b2, config.adam_eps, config.adam_eps_root, config.mu_dtype, lambda p: mults(p)[1]),
      decay,
      _scale_by_learning_rate_and_multiplier(learning_rate, lambda p: mults(p)[0]),
  )
