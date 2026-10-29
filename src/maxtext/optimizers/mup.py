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

import os
import re

import jax
import optax
from optax._src import base as optax_base
from optax._src import wrappers as optax_wrappers

from maxtext.utils import max_logging

# (pattern on the path within a layer, fan-in from the config and whether the layer is dense).
_LAYER_MATRICES = (
    (r"mixer/(w_q|w_k|w_v|f_proj_1|g_proj_1|w_b)/kernel", lambda c, dense: c.emb_dim),
    (r"mixer/(f_proj_2|g_proj_2)/kernel", lambda c, dense: c.gdn_value_head_dim),
    (r"mixer/w_out/kernel", lambda c, dense: c.gdn_num_value_heads * c.gdn_value_head_dim),
    (r"mixer/attention/(query|key|value)/kernel", lambda c, dense: c.emb_dim),
    (r"mixer/attention/out/kernel", lambda c, dense: c.num_query_heads * c.head_dim),
    (r"latent_down/kernel", lambda c, dense: c.emb_dim),
    (r"latent_up/kernel", lambda c, dense: c.moe_expert_input_dim),
    (r"moe_block/gate/kernel", lambda c, dense: c.emb_dim),
    (r"moe_block/(wi_0|wi_1)", lambda c, dense: c.moe_expert_input_dim),
    (r"moe_block/wo", lambda c, dense: c.moe_mlp_dim),
    (r"shared_ffn/(wi_0|wi_1)/kernel", lambda c, dense: c.emb_dim),
    (r"shared_ffn/wo/kernel", lambda c, dense: c.mlp_dim if dense else c.shared_expert_mlp_dim),
)
# Parameters inside a layer that are vectors for muP whatever their rank: norm gains (2-D for per-head QK
# norms), biases, KDA's decay parameters and depthwise short convs, and the SSMax gains.
_LAYER_VECTORS = r"(/scale|/bias|ssmax_scale|A_log|dt_bias|_conv)$"


def _path_str(path):
  """The '/'-joined parameter path, as `adamw_mask` matches it, without the boxed-variable `value` key."""
  s = jax.tree_util.keystr(path, simple=True, separator="/")
  return re.sub(r"/value$", "", s)


def _layer_kind(config, layer):
  dense = layer < config.first_num_dense_layers
  attention = (layer + 1) % config.inhomogeneous_layer_cycle_interval == 0
  return ("dense" if dense else "moe") + "+" + ("attention" if attention else "kda")


def _classify(config, path_str):
  """("matrix", role, fan_in) | ("embedding",) | ("vector",) for a parameter path; raises if it has no rule.

  A role is the layer kind plus the matrix's rule in `_LAYER_MATRICES` (or the LM head), so the same matrix
  of a model with another width or depth has the same role.
  """
  if path_str.endswith("token_embedder/embedding"):
    return ("embedding",)
  if path_str.endswith("decoder/logits_dense/kernel"):
    return ("matrix", "logits_dense/kernel", config.emb_dim)
  match = re.search(r"(?:^|/)layers_(\d+)/(.+)$", path_str)
  if match is None:
    if re.search(r"(decoder_norm|embedding_norm)/scale$", path_str):
      return ("vector",)
    raise NotImplementedError(f"muP has no rule for parameter {path_str}")
  layer, sub = int(match.group(1)), match.group(2)
  if re.search(_LAYER_VECTORS, sub):
    return ("vector",)
  kind = _layer_kind(config, layer)
  for pattern, fan_in in _LAYER_MATRICES:
    if re.fullmatch(pattern, sub):
      return ("matrix", f"{kind}/{pattern}", fan_in(config, kind.startswith("dense")))
  raise NotImplementedError(f"muP has no rule for parameter {path_str}")


def _shape_fan_in(config, path_str, shape):
  """The fan-in a kernel's shape implies (dims MaxText contracts over), to check the config-derived one."""
  if re.search(r"moe_block/(wi_0|wi_1|wo)$", path_str):
    # [experts, in, out], or [experts * in, out] under moe_flat_fsdp_weights.
    return shape[1] if len(shape) == 3 else shape[0] // config.num_experts
  return shape[0]


def check_supported(config):
  """Raises if `mup_base_model_name` is set on a run muP can't handle."""
  if config.decoder_block.value != "olmoe3":
    raise ValueError(f"mup_base_model_name is only implemented for the olmoe3 decoder, not {config.decoder_block}.")
  if config.opt_type != "adamw":
    raise ValueError(f"mup_base_model_name needs opt_type=adamw, not {config.opt_type}.")
  if config.scan_layers:
    raise ValueError("mup_base_model_name needs scan_layers=False (roles are read from the layers_<i> paths).")


def load_base_config(config):
  """The MaxText config of `mup_base_model_name`, with base.yml defaults (only its widths and layer layout are read)."""
  from maxtext.configs import pyconfig  # pylint: disable=import-outside-toplevel
  from maxtext.utils.globals import MAXTEXT_PKG_DIR  # pylint: disable=import-outside-toplevel

  return pyconfig.initialize(
      [
          "",
          os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml"),
          f"model_name={config.mup_base_model_name}",
          "run_name=mup_base_model",
          "enable_checkpointing=False",
          "skip_jax_distributed_system=True",
      ]
  )


def base_fan_ins(base_config):
  """{role: fan_in} for every matrix role of the base model, keyed as `_classify` keys them."""
  roles = {"logits_dense/kernel": base_config.emb_dim}
  for layer in range(base_config.num_decoder_layers):
    kind = _layer_kind(base_config, layer)
    for pattern, fan_in in _LAYER_MATRICES:
      roles.setdefault(f"{kind}/{pattern}", fan_in(base_config, kind.startswith("dense")))
  return roles


def lr_multipliers(config, base_config, params):
  """The muP learning-rate multiplier of every parameter of `params` (same tree, Python floats).

  Raises if a parameter has no rule, if its kernel shape disagrees with the fan-in its config gives, or if
  its role (layer kind) is missing from the base model.
  """
  base = base_fan_ins(base_config)

  def multiplier(path, leaf):
    path_str = _path_str(path)
    kind = _classify(config, path_str)
    if kind[0] == "embedding":
      return float(config.mup_embedding_lr_multiplier)
    if kind[0] == "vector":
      return 1.0
    _, role, fan_in = kind
    if hasattr(leaf, "shape") and _shape_fan_in(config, path_str, leaf.shape) != fan_in:
      raise ValueError(f"muP: {path_str} has shape {leaf.shape}, but its config gives fan-in {fan_in}")
    if role not in base:
      raise ValueError(f"muP: the base model {config.mup_base_model_name} has no {role} (no such layer kind)")
    return base[role] / fan_in

  return jax.tree_util.tree_map_with_path(multiplier, params)


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
          b1=config.adam_b1, b2=config.adam_b2, eps=config.adam_eps, eps_root=config.adam_eps_root, mu_dtype=config.mu_dtype
      ),
      decay,
      _scale_by_learning_rate_and_multiplier(learning_rate, multipliers_fn),
  )
