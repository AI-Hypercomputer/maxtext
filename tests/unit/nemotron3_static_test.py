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

"""Production-plumbing checks for the Nemotron3 modules that CPU fp32 parity cannot see.

1. Every dense contraction in models/nemotron3.py passes matmul_precision (fp32 CPU checks hide a
   missing precision; TPU silently truncates to bf16 passes).
2. Every parameter and cache variable of a Nemotron3 layer carries logical axis names for each dim,
   except the explicitly allowlisted variables of reused upstream modules.
"""

import ast
import os
import re
import unittest

from maxtext.configs import pyconfig
from tests.utils.test_helpers import get_test_config_path

DENSE_CALLS = {
    "DenseGeneral",
    "dot_general",
    "einsum",
    "matmul",
    "Conv",
    "conv_general_dilated",
}
PRECISION_KWARGS = {"matmul_precision", "precision"}
# Variables of reused upstream modules that carry no logical axis names, allowed only in layers of
# the given block type. Everything else, in particular every variable a models/nemotron3.py module
# creates, must be named.
UPSTREAM_UNNAMED_ALLOWLIST = (
    # inference/kvcache.py KVCache._initialize_ar_cache_vars: AR write index, shape (1,), out_sharding=().
    (re.compile(r"^decoder/layers_(\d+)/mixer/KVCache_0/cache_ar_index$"), "*"),
)


def _call_name(node):
  f = node.func
  return f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")


class Nemotron3MatmulPrecisionTest(unittest.TestCase):

  def test_every_dense_op_passes_precision(self):
    test_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(test_dir, os.pardir, os.pardir))
    nemotron3_py = os.path.join(repo_root, "src", "maxtext", "models", "nemotron3.py")
    with open(nemotron3_py, encoding="utf-8") as f:
      tree = ast.parse(f.read())
    missing, matmul_ops = [], []
    for node in ast.walk(tree):
      if isinstance(node, ast.Call) and _call_name(node) in DENSE_CALLS:
        if not PRECISION_KWARGS & {k.arg for k in node.keywords}:
          missing.append((node.lineno, _call_name(node)))
      if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
        matmul_ops.append(node.lineno)
    self.assertEqual(missing, [], "dense op without matmul_precision/precision")
    self.assertEqual(matmul_ops, [], "use einsum/dot_general with precision instead of @")


def _allowlisted(key, pattern):
  for regex, block in UPSTREAM_UNNAMED_ALLOWLIST:
    match = regex.match(key)
    if match and pattern[int(match.group(1))] == block:
      return True
  return False


class Nemotron3LogicalAxesTest(unittest.TestCase):

  def test_every_new_variable_has_logical_axes(self):
    # pylint: disable=import-outside-toplevel
    from flax import nnx
    from maxtext.utils import model_creation_utils

    bad, allowed, n = [], [], 0
    config_args = [None, get_test_config_path("base.yml"), "run_name=test", "skip_jax_distributed_system=True"]
    for mode in ("train", "autoregressive"):
      cfg = pyconfig.initialize(
          config_args,
          decoder_block="nemotron3",
          base_emb_dim=2688,
          base_num_decoder_layers=3,
          hybrid_override_pattern="ME*",
          vocab_size=131072,
          base_num_query_heads=32,
          base_num_kv_heads=2,
          head_dim=128,
          mamba_num_heads=64,
          mamba_head_dim=64,
          mamba_ssm_state_size=128,
          mamba_n_groups=8,
          mamba_conv_kernel=4,
          mamba_conv_bias=True,
          mamba_chunk_size=128,
          num_experts=128,
          num_experts_per_tok=6,
          base_moe_mlp_dim=1856,
          base_mlp_dim=1856,
          shared_experts=2,
          mlp_activations=["relu2"],
          scan_layers=False,
          enable_checkpointing=False,
      )
      _, abstract = model_creation_utils.create_nnx_abstract_model(cfg, model_mode=mode)
      for path, var in nnx.to_flat_state(nnx.state(abstract)):
        if isinstance(var, nnx.RngState):
          continue
        key = "/".join(str(p) for p in path)
        if "layers_" not in key:
          continue
        n += 1
        names = getattr(var, "out_sharding", None) or getattr(var, "sharding_names", None)
        if names is None or len(tuple(names)) != len(var.get_value().shape):
          if _allowlisted(key, cfg.hybrid_override_pattern):
            allowed.append((mode, key))
          else:
            bad.append((mode, key, names, var.get_value().shape))
    self.assertGreater(n, 0)
    self.assertEqual(bad, [])


if __name__ == "__main__":
  unittest.main()
