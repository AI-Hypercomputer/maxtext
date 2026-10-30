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

"""The role and fan-in of every OLMoE3 parameter, for width/depth parametrizations (muP, CompletedP).

A parameter's role says which scaling rules apply to it (OLMo-core's ``CompletedPRole``); a matrix also has a
fan-in, read from the config, and a role key -- its layer kind (dense or MoE, KDA or full attention) plus its
rule below -- that names the same matrix in a model of another width or depth, so it can be compared with the
same matrix of a base model (OLMo-core's base-model mode). References: OLMo-core ``tanushy/parameterization-ladders``
(``nn/transformer/mup.py``, ``nn/transformer/completedp.py``).
"""

import dataclasses
import functools
import os
import re

import jax

# In-layer weight matrices: (rule, fan-in from the config and whether the layer is dense, readout). A readout
# matrix's output size doesn't grow with width (MoE router: one output per expert; KDA f_proj_1 / g_proj_1: into
# the fixed head_v_dim-wide gate bottlenecks), so CompletedP gives it the output layer's width rules.
LAYER_MATRICES = (
    (r"mixer/(w_q|w_k|w_v|w_b)/kernel", lambda c, dense: c.emb_dim, False),
    (r"mixer/(f_proj_1|g_proj_1)/kernel", lambda c, dense: c.emb_dim, True),
    (r"mixer/(f_proj_2|g_proj_2)/kernel", lambda c, dense: c.gdn_value_head_dim, False),
    (r"mixer/w_out/kernel", lambda c, dense: c.gdn_num_value_heads * c.gdn_value_head_dim, False),
    (r"mixer/attention/(query|key|value)/kernel", lambda c, dense: c.emb_dim, False),
    (r"mixer/attention/out/kernel", lambda c, dense: c.num_query_heads * c.head_dim, False),
    (r"latent_down/kernel", lambda c, dense: c.emb_dim, False),
    (r"latent_up/kernel", lambda c, dense: c.moe_expert_input_dim, False),
    (r"moe_block/gate/kernel", lambda c, dense: c.emb_dim, True),
    (r"moe_block/(wi_0|wi_1)", lambda c, dense: c.moe_expert_input_dim, False),
    (r"moe_block/wo", lambda c, dense: c.moe_mlp_dim, False),
    (r"shared_ffn/(wi_0|wi_1)/kernel", lambda c, dense: c.emb_dim, False),
    (r"shared_ffn/wo/kernel", lambda c, dense: c.mlp_dim if dense else c.shared_expert_mlp_dim, False),
)


def fan_out(c, sub, dense):
  """The output width of the in-layer matrix at `sub` (e.g. "mixer/w_q/kernel"), from config `c`."""
  kda_heads = c.gdn_num_value_heads  # MaxText's KDA uses gdn_num_value_heads heads for keys and values alike
  widths = (
      (r"mixer/(w_q|w_k|f_proj_2)/kernel", lambda: kda_heads * c.gdn_key_head_dim),
      (r"mixer/(w_v|g_proj_2)/kernel", lambda: kda_heads * c.gdn_value_head_dim),
      (r"mixer/w_b/kernel", lambda: kda_heads),
      (r"mixer/(f_proj_1|g_proj_1)/kernel", lambda: c.gdn_value_head_dim),
      (r"mixer/attention/query/kernel", lambda: c.num_query_heads * c.head_dim),
      (r"mixer/attention/(key|value)/kernel", lambda: c.num_kv_heads * c.head_dim),
      (r"(mixer/w_out|mixer/attention/out|latent_up|shared_ffn/wo)/kernel", lambda: c.emb_dim),
      (r"latent_down/kernel", lambda: c.moe_expert_input_dim),
      (r"moe_block/gate/kernel", lambda: c.num_experts),
      (r"moe_block/(wi_0|wi_1)", lambda: c.moe_mlp_dim),
      (r"moe_block/wo", lambda: c.moe_expert_input_dim),
      (r"shared_ffn/(wi_0|wi_1)/kernel", lambda: c.mlp_dim if dense else c.shared_expert_mlp_dim),
  )
  for pattern, width in widths:
    if re.fullmatch(pattern, sub):
      return width()
  raise NotImplementedError(f"no output width for matrix {sub}")


def is_readout(config, base_config, sub, dense, readout):
  """Whether the in-layer matrix at `sub` is a readout: the fixed `readout` flag (OLMo-core's list), or, under
  `head_dim_transfer` with a base model, whether its output width is the same as in the base model (so it doesn't
  grow with width, e.g. KDA's f_proj_1 / g_proj_1 only while the value head dim stays)."""
  if base_config is None or not getattr(config, "head_dim_transfer", False):
    return readout
  return fan_out(config, sub, dense) == fan_out(base_config, sub, dense)


def head_count(config, sub):
  """The number of heads a head-shared gain at `sub` is shared across."""
  if re.search(_HEAD_SHARED, sub):
    return config.gdn_num_value_heads
  return config.num_query_heads if "query_norm" in sub else config.num_kv_heads


# In-layer element-wise parameters, whatever their rank: norm gains (2-D for per-head QK norms), biases, KDA's
# decay parameters and depthwise short convs, and the SSMax gains.
_LAYER_VECTORS = r"(/scale|/bias|ssmax_scale|A_log|dt_bias|_conv)$"
# Gains shared across heads: their gradient sums over the heads, so CompletedP doesn't width-scale their epsilon.
# KDA's output norm (one head_v_dim gain), and QK norms when not per head (a 1-D gain).
_HEAD_SHARED = r"mixer/o_norm/scale$"
_QK_NORMS = r"mixer/attention/(query|key)_norm/scale$"

INPUT_EMBEDDING = "input_embedding"
INPUT_VECTOR = "input_vector"
HIDDEN_MATRIX = "hidden_matrix"
HIDDEN_READOUT_MATRIX = "hidden_readout_matrix"
HIDDEN_VECTOR = "hidden_vector"
HIDDEN_HEAD_SHARED_VECTOR = "hidden_head_shared_vector"
OUTPUT_MATRIX = "output_matrix"
OUTPUT_VECTOR = "output_vector"
MATRIX_ROLES = (HIDDEN_MATRIX, HIDDEN_READOUT_MATRIX, OUTPUT_MATRIX)
LM_HEAD_KEY = "logits_dense/kernel"


@dataclasses.dataclass(frozen=True)
class ParamRole:
  role: str
  key: None | str = None  # matrices: the role key, to match the same matrix in a base model
  fan_in: None | int = None  # matrices
  sub: None | str = None  # in-layer parameters: the path within the layer, e.g. "mixer/o_norm/scale"


def path_str(path):
  """The '/'-joined parameter path, as `adamw_mask` matches it, without the boxed-variable `value` key."""
  return re.sub(r"/value$", "", jax.tree_util.keystr(path, simple=True, separator="/"))


def layer_kind(config, layer):
  dense = layer < config.first_num_dense_layers
  attention = (layer + 1) % config.inhomogeneous_layer_cycle_interval == 0
  return ("dense" if dense else "moe") + "+" + ("attention" if attention else "kda")


def classify(config, path, shape=None, base_config=None):
  """The `ParamRole` of the parameter at `path` (a '/'-joined path); raises if there's no rule for it. `base_config`
  is only read under `head_dim_transfer` (readouts by output width)."""
  if path.endswith("token_embedder/embedding"):
    return ParamRole(INPUT_EMBEDDING)
  if path.endswith("decoder/embedding_norm/scale"):
    return ParamRole(INPUT_VECTOR)
  if path.endswith("decoder/decoder_norm/scale"):
    return ParamRole(OUTPUT_VECTOR)
  if path.endswith("decoder/" + LM_HEAD_KEY):
    return ParamRole(OUTPUT_MATRIX, LM_HEAD_KEY, config.emb_dim)
  match = re.search(r"(?:^|/)layers_(\d+)/(.+)$", path)
  if match is None:
    raise NotImplementedError(f"no parametrization rule for parameter {path}")
  layer, sub = int(match.group(1)), match.group(2)
  if re.search(_HEAD_SHARED, sub) or (re.search(_QK_NORMS, sub) and shape is not None and len(shape) == 1):
    return ParamRole(HIDDEN_HEAD_SHARED_VECTOR, sub=sub)
  if re.search(_LAYER_VECTORS, sub):
    return ParamRole(HIDDEN_VECTOR)
  kind = layer_kind(config, layer)
  dense = kind.startswith("dense")
  for pattern, fan_in, readout in LAYER_MATRICES:
    if re.fullmatch(pattern, sub):
      role = HIDDEN_READOUT_MATRIX if is_readout(config, base_config, sub, dense, readout) else HIDDEN_MATRIX
      return ParamRole(role, f"{kind}/{pattern}", fan_in(config, dense), sub)
  raise NotImplementedError(f"no parametrization rule for parameter {path}")


def shape_fan_in(config, path, shape):
  """The fan-in a kernel's shape implies (the dims MaxText contracts over), to check the config-derived one."""
  if re.search(r"moe_block/(wi_0|wi_1|wo)$", path):
    # [experts, in, out], or [experts * in, out] under moe_flat_fsdp_weights.
    return shape[1] if len(shape) == 3 else shape[0] // config.num_experts
  return shape[0]


def classify_tree(config, params, check_shapes=True, base_config=None):
  """The `ParamRole` of every leaf of `params`, checking each matrix's fan-in against its kernel shape."""

  def role(path, leaf):
    p = path_str(path)
    shape = getattr(leaf, "shape", None)
    r = classify(config, p, shape, base_config)
    if check_shapes and r.fan_in is not None and shape is not None and shape_fan_in(config, p, shape) != r.fan_in:
      raise ValueError(f"{p} has shape {shape}, but its config gives fan-in {r.fan_in}")
    return r

  return jax.tree_util.tree_map_with_path(role, params)


def base_fan_ins(base_config):
  """{role key: fan_in} for every matrix of the base model, keyed as `classify` keys them."""
  roles = {LM_HEAD_KEY: base_config.emb_dim}
  for layer in range(base_config.num_decoder_layers):
    kind = layer_kind(base_config, layer)
    for pattern, fan_in, _ in LAYER_MATRICES:
      roles.setdefault(f"{kind}/{pattern}", fan_in(base_config, kind.startswith("dense")))
  return roles


def base_fan_in(base, role, base_model_name):
  """The base model's fan-in of a matrix role; raises if the base model has no such layer kind."""
  if role.key not in base:
    raise ValueError(f"the base model {base_model_name} has no {role.key} (no such layer kind)")
  return base[role.key]


@functools.lru_cache(maxsize=None)
def load_base_config(model_name):
  """The MaxText config of a base model, with base.yml defaults (only its widths and layer layout are read)."""
  from maxtext.configs import pyconfig  # pylint: disable=import-outside-toplevel
  from maxtext.utils.globals import MAXTEXT_PKG_DIR  # pylint: disable=import-outside-toplevel

  return pyconfig.initialize(
      [
          "",
          os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml"),
          f"model_name={model_name}",
          "run_name=parametrization_base_model",
          "enable_checkpointing=False",
          "skip_jax_distributed_system=True",
      ]
  )


def completedp_init_factor(config, base_config, sub, dense=False):
  """CompletedP's factor on the normal init of the in-layer matrix at `sub` (a path within a layer, e.g.
  "mixer/w_out/kernel"), or of the LM head for `sub` = LM_HEAD_KEY: ``sqrt(base fan-in / fan-in)`` for a hidden
  matrix, ``base fan-in / fan-in`` for a readout matrix and the LM head."""
  if sub == LM_HEAD_KEY:
    return base_config.emb_dim / config.emb_dim
  for pattern, fan_in, readout in LAYER_MATRICES:
    if re.fullmatch(pattern, sub):
      ratio = fan_in(base_config, dense) / fan_in(config, dense)
      return ratio if is_readout(config, base_config, sub, dense, readout) else ratio ** 0.5
  raise NotImplementedError(f"no parametrization rule for matrix {sub}")
