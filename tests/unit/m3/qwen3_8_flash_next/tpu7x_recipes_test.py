# Copyright 2026 Google LLC
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

"""Unit tests validating Qwen3.8-Flash-Next 64-chip TPU7x training recipes."""

import os

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax

from benchmarks.qwen3_8_flash_next_tpu7x import config as recipe_cfg
from benchmarks.qwen3_8_flash_next_tpu7x import xla_flags
from maxtext.configs import pyconfig
from maxtext.m3.core import moe as m3_moe
from maxtext.m3.core import sharding as m3_sharding
from maxtext.m3.models.qwen3_8_flash_next import modeling_qwen3_8_flash_next
from maxtext.utils.globals import MAXTEXT_PKG_DIR

_BASE_YML = os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")


def _format_cli_arg(key: str, val: object) -> str:
  if isinstance(val, (list, tuple)):
    inner = ",".join(f"'{v}'" if isinstance(v, str) else str(v) for v in val)
    return f"{key}=[{inner}]"
  if isinstance(val, bool):
    return f"{key}={str(val).lower()}"
  return f"{key}={val}"


class Qwen38Tpu7xRecipesTest(parameterized.TestCase):
  """Validates `RECIPES` (`8k`, `8k_2m`, `64k`) against `pyconfig`, `validate_moe_config`, and 4-CPU `eval_shape`."""

  def test_recipe_keys_and_xla_flag_bundles(self):
    self.assertEqual(set(recipe_cfg.RECIPES.keys()), {"8k", "8k_2m", "64k"})
    self.assertIn("--xla_tpu_enable_concurrent_sparse_core_offloading=true", xla_flags.BASE_FLAGS)
    self.assertIn("--xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=6000000", xla_flags.HIGH_TOKEN_FLAGS)

  @parameterized.parameters("8k", "8k_2m", "64k")
  def test_recipe_pyconfig_moe_validation_and_mini_model_eval_shape(self, recipe_name: str):
    recipe = recipe_cfg.RECIPES[recipe_name]
    self.assertEqual(recipe.topology, "4x4x4")

    overrides = dict(recipe.maxtext_flags)
    overrides.update(
        {
            "run_name": f"test_{recipe_name}",
            "dataset_type": "synthetic",
            "enable_checkpointing": False,
            "skip_jax_distributed_system": True,
            "hardware": "cpu",
            "ici_fsdp_parallelism": 2,
            "ici_expert_parallelism": 2,
        }
    )
    argv = ["", _BASE_YML, *(_format_cli_arg(k, v) for k, v in overrides.items())]
    cfg = pyconfig.initialize(argv)

    self.assertTrue(cfg.compound_fsdp_ep)
    self.assertTrue(cfg.moe_fused_bwd_quant)
    self.assertTrue(cfg.use_hc_pallas)
    self.assertTrue(cfg.use_gdn_conv_pallas)
    # Note: `pyconfig` disables `megablox` on `hardware='cpu'`, while `use_tokamax_gmm=True` stays active.
    m3_moe.validate_moe_config(cfg, require_moe=True)

    shd = m3_sharding.sharding_from_config(cfg)
    self.assertTrue(shd.compound_fsdp_ep)
    if recipe_name in ("8k", "64k"):
      self.assertIsInstance(shd, m3_sharding.EPasCP)
    else:
      self.assertIs(type(shd), m3_sharding.DefaultSharding)

    if jax.device_count() < 4:
      return

    mini_overrides = dict(overrides)
    mini_overrides.update(
        {
            "base_emb_dim": 128,
            "base_mlp_dim": 64,
            "base_moe_mlp_dim": 64,
            "base_num_query_heads": 4,
            "base_num_kv_heads": 2,
            "head_dim": 64,
            "mrope_section": [3, 3, 2],
            "base_num_decoder_layers": 4,
            "inhomogeneous_layer_cycle_interval": 4,
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "vocab_size": 256,
            "ple_embed_dim": 128,
            "ngram_vocab_size_base": 64,
            "max_target_length": 64,
        }
    )
    mini_cfg = pyconfig.initialize(["", _BASE_YML, *(_format_cli_arg(k, v) for k, v in mini_overrides.items())])
    mesh = m3_sharding.create_mesh(ici={"fsdp": 2, "expert": 2}, devices=jax.devices()[:4])
    with jax.set_mesh(mesh):
      abs_model = nnx.eval_shape(
          lambda: modeling_qwen3_8_flash_next.create_qwen3_8_flash_next_model(
              mini_cfg, mesh=mesh, rngs=nnx.Rngs(params=0)
          )
      )
    gu_param = abs_model.decoder.first_cycle.gdn_0.mlp.experts.gate_up_proj
    gu_spec = gu_param.get_metadata()["out_sharding"]
    self.assertEqual(gu_spec[0], ("fsdp", "fsdp_transpose", "context", "expert"))
    self.assertIsNone(gu_spec[1])


if __name__ == "__main__":
  os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
  absltest.main()
