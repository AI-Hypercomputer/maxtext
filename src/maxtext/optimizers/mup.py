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

"""muP per-parameter AdamW learning rates for OLMoE3, as OLMo-core's ``MuPConfig`` in base-model mode.

Each weight matrix trains at ``lr * m`` with ``m = base_fan_in(role) / fan_in``, where the role is the same
matrix in a base model (the layer kind -- dense or MoE, KDA or full attention -- plus the path within the
layer), so every matrix of the base model trains at ``lr``. The token embedding gets
``mup_embedding_lr_multiplier`` and every other parameter (norm gains, biases, ``A_log``, ``dt_bias``, the
SSMax gains, the depthwise KDA convs) gets ``lr``. With ``mup_scale_weight_decay`` each parameter's weight
decay is ``wd / m``, so it decays by ``lr * wd`` per step at every width; otherwise by ``lr * m * wd``.

Reference: OLMo-core ``tanushy/parameterization-ladders`` (``optim/mup.py``, ``nn/transformer/mup.py``,
docs/MUP_HANDOFF.md) and scaling-ladders ``tanushy/parameterization-mup-wd`` (MUP_WEIGHT_DECAY.md).
"""

import jax
import optax
from optax._src import base as optax_base
from optax._src import wrappers as optax_wrappers

from maxtext.models import olmoe3_roles as roles
from maxtext.utils import max_logging


def _path_str(path):
  return roles.path_str(path)


def check_supported(config):
  """Raises if `mup_base_model_name` is set on a run muP can't handle."""
  if config.decoder_block.value != "olmoe3":
    raise ValueError(f"mup_base_model_name is only implemented for the olmoe3 decoder, not {config.decoder_block}.")
  if config.opt_type != "adamw":
    raise ValueError(f"mup_base_model_name needs opt_type=adamw, not {config.opt_type}.")
  if config.scan_layers:
    raise ValueError("mup_base_model_name needs scan_layers=False (roles are read from the layers_<i> paths).")


def load_base_config(config):
  """The MaxText config of `mup_base_model_name` (only its widths and layer layout are read)."""
  return roles.load_base_config(config.mup_base_model_name)


def lr_multipliers(config, base_config, params):
  """The muP learning-rate multiplier of every parameter of `params` (same tree, Python floats).

  Matrices (in-layer, readout and the LM head alike): base fan-in / fan-in. The embedding:
  `mup_embedding_lr_multiplier`. Everything else: 1. Raises if a parameter has no rule, if its kernel shape
  disagrees with the fan-in its config gives, or if its layer kind is missing from the base model.
  """
  base = roles.base_fan_ins(base_config)

  def multiplier(role):
    if role.role == roles.INPUT_EMBEDDING:
      return float(config.mup_embedding_lr_multiplier)
    if role.role in roles.MATRIX_ROLES:
      return roles.base_fan_in(base, role, config.mup_base_model_name) / role.fan_in
    return 1.0

  return jax.tree.map(multiplier, roles.classify_tree(config, params), is_leaf=lambda x: isinstance(x, roles.ParamRole))


def _log_multipliers(multipliers, params):
  groups = {}
  for (path, m), leaf in zip(jax.tree_util.tree_flatten_with_path(multipliers)[0], jax.tree_util.tree_leaves(params)):
    count = groups.setdefault(m, [0, 0, _path_str(path)])
    count[0] += 1
    count[1] += int(getattr(leaf, "size", 0))
  for m, (n, size, example) in sorted(groups.items()):
    max_logging.log(f"muP: lr multiplier {m:.6g} on {n} parameters ({size:,} elements), e.g. {example}")


def _add_decayed_weights_per_param(weight_decay, multipliers_fn, scale_weight_decay):
  """optax.add_decayed_weights with each parameter's decay `wd / m` (scaled) or `wd` (coupled)."""

  def init_fn(params):
    del params
    return optax_base.EmptyState()

  def update_fn(updates, state, params):
    if params is None:
      raise ValueError(optax_base.NO_PARAMS_MSG)
    multipliers = multipliers_fn(params)
    updates = jax.tree.map(
        lambda g, p, m: None if g is None else g + (weight_decay / m if scale_weight_decay else weight_decay) * p,
        updates,
        params,
        multipliers,
        is_leaf=lambda x: x is None,
    )
    return updates, state

  return optax_base.GradientTransformation(init_fn, update_fn)


def _scale_by_learning_rate_and_multiplier(learning_rate, multipliers_fn):
  """optax.scale_by_learning_rate (same state), then each parameter's update times its multiplier."""
  inner = optax.scale_by_learning_rate(learning_rate)

  def update_fn(updates, state, params=None):
    updates, state = inner.update(updates, state, params)
    multipliers = multipliers_fn(params)
    updates = jax.tree.map(lambda u, m: u if m == 1.0 else u * m, updates, multipliers)
    return updates, state

  return optax_base.GradientTransformation(inner.init, update_fn)


def mup_adamw(learning_rate, config, base_config, mask=None):
  """optax.adamw with muP per-parameter learning rates and weight decay.

  The same three transforms as optax.adamw, so the optimizer state has the same structure (checkpoints
  load with muP on or off): Adam, the (masked) decoupled decay with each parameter's own decay, and the
  learning rate with each parameter's multiplier folded in. Every parameter's update is
  ``-lr * m * (adam + wd_p * p)`` with ``wd_p = wd / m`` (scaled) or ``wd`` (coupled), which is PyTorch
  AdamW with a group learning rate ``lr * m`` and weight decay ``wd_p``, as OLMo-core's muP groups.
  """
  logged = []

  def multipliers_fn(params):
    multipliers = lr_multipliers(config, base_config, params)
    if not logged:
      logged.append(True)
      _log_multipliers(multipliers, params)
    return multipliers

  decay = _add_decayed_weights_per_param(config.adam_weight_decay, multipliers_fn, config.mup_scale_weight_decay)
  if mask is not None:
    decay = optax_wrappers.masked(decay, mask)
  return optax.chain(
      optax.scale_by_adam(
          b1=config.adam_b1,
          b2=config.adam_b2,
          eps=config.adam_eps,
          eps_root=config.adam_eps_root,
          mu_dtype=config.mu_dtype,
      ),
      decay,
      _scale_by_learning_rate_and_multiplier(learning_rate, multipliers_fn),
  )
