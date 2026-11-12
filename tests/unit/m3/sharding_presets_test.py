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

"""Unit and virtual-topology tests for compound MoE sharding and scale-tuned `Sharding` presets."""

from types import SimpleNamespace

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from maxtext.m3.core import moe as moe_lib
from maxtext.m3.core import sharding as sharding_lib
from maxtext.m3.core.sharding import CPasEP
from maxtext.m3.core.sharding import DefaultSharding
from maxtext.m3.core.sharding import EPasCP
from maxtext.m3.core.sharding import EPasDP
from maxtext.m3.core.sharding import FSDPasDPForAttn
from maxtext.m3.core.sharding import FSDPxEP
from maxtext.m3.core.sharding import FSDPxEP_2D
from maxtext.m3.core.sharding import MESH_AXIS_NAMES
from maxtext.m3.core.sharding import SHARDING_PRESETS
from maxtext.m3.core.sharding import TPasEP
from maxtext.m3.core.sharding import TPxEP
from maxtext.m3.core.sharding import TensorType
from maxtext.m3.core.sharding import compound_moe_param_metadata
from maxtext.m3.core.sharding import compound_moe_spec
from maxtext.m3.core.sharding import create_mesh
from maxtext.m3.core.sharding import fsdp_x_ep_axes
from maxtext.m3.core.sharding import param_metadata
from maxtext.m3.core.sharding import project_spec_to_mesh
from maxtext.m3.core.sharding import sharding_from_config
from maxtext.m3.core.sharding import sharding_preset_from_name
from maxtext.m3.core.sharding import tp_x_ep_axes


class CompoundMoEHelpersTest(parameterized.TestCase):
  """Tests `FSDPxEP`, `TPxEP`, `compound_moe_spec`, and `compound_moe_param_metadata`."""

  def test_axis_constants_and_helpers(self):
    self.assertEqual(FSDPxEP, ("fsdp", "expert"))
    self.assertEqual(FSDPxEP_2D, ("fsdp", "fsdp_transpose", "expert"))
    self.assertEqual(TPxEP, ("expert", "tensor"))
    self.assertEqual(tp_x_ep_axes(), ("expert", "tensor"))
    self.assertEqual(fsdp_x_ep_axes(two_d_fsdp=False), ("fsdp", "expert"))
    self.assertEqual(fsdp_x_ep_axes(two_d_fsdp=True), ("fsdp", "fsdp_transpose", "expert"))

    with jax.sharding.set_mesh(None):
      self.assertEqual(fsdp_x_ep_axes(), ("fsdp", "expert"))

    abs_1d = jax.sharding.AbstractMesh((1, 1, 16, 1, 1, 1, 1, 8), MESH_AXIS_NAMES)
    with jax.sharding.use_abstract_mesh(abs_1d):
      self.assertEqual(fsdp_x_ep_axes(), ("fsdp", "expert"))

    abs_2d = jax.sharding.AbstractMesh((1, 1, 8, 2, 1, 1, 1, 8), MESH_AXIS_NAMES)
    with jax.sharding.use_abstract_mesh(abs_2d):
      self.assertEqual(fsdp_x_ep_axes(), ("fsdp", "fsdp_transpose", "expert"))

  def test_compound_moe_spec_fsdp_x_ep_and_tp_x_ep(self):
    with jax.sharding.set_mesh(None):
      self.assertEqual(
          compound_moe_spec(None, "wi_0", ("experts", "expert_embed", "expert_mlp"), mode="fsdp_x_ep"),
          P(("fsdp", "expert"), None, ("tensor", "tensor_sequence")),
      )
      self.assertEqual(
          compound_moe_spec(None, "wi_0", ("experts", "expert_embed", "expert_mlp"), mode="fsdp_x_ep", two_d_fsdp=True),
          P(("fsdp", "fsdp_transpose", "expert"), None, ("tensor", "tensor_sequence")),
      )
      self.assertEqual(
          compound_moe_spec(
              None, "act", ("experts", "expert_embed", "expert_mlp"), mode="fsdp_x_ep", tensor_type=TensorType.ACTIVATION
          ),
          P("expert", None, ("tensor", "tensor_sequence")),
      )
      self.assertEqual(
          compound_moe_spec(None, "wi_0", ("experts", "expert_embed", "expert_mlp"), mode="tp_x_ep"),
          P(("expert", "tensor"), None, "tensor_sequence"),
      )
      self.assertEqual(
          compound_moe_spec(None, "wi_0", ("experts", "expert_embed", "expert_mlp"), mode="tp_x_ep", muon_friendly=False),
          P(("expert", "tensor"), ("fsdp", "fsdp_transpose", "context"), "tensor_sequence"),
      )
      self.assertEqual(
          compound_moe_spec(
              None, "wi_0", ("experts", "expert_embed", "expert_mlp"), mode="fsdp_x_ep", muon_friendly=False
          ),
          P(("fsdp", "expert"), ("fsdp_transpose", "context"), ("tensor", "tensor_sequence")),
      )
      self.assertEqual(
          compound_moe_spec(
              None,
              "wi_0",
              ("experts", "expert_embed", "expert_mlp"),
              mode="fsdp_x_ep",
              two_d_fsdp=True,
              muon_friendly=False,
          ),
          P(("fsdp", "fsdp_transpose", "expert"), "context", ("tensor", "tensor_sequence")),
      )
      self.assertEqual(
          compound_moe_spec(
              CPasEP(), "wi_0", ("experts", "expert_embed", "expert_mlp"), mode="fsdp_x_ep", muon_friendly=False
          ),
          P(("fsdp", "context", "expert"), "fsdp_transpose", ("tensor", "tensor_sequence")),
      )
      self.assertEqual(
          compound_moe_spec(
              TPasEP(), "wi_0", ("experts", "expert_embed", "expert_mlp"), mode="fsdp_x_ep", muon_friendly=False
          ),
          P(("fsdp", "expert", "tensor"), ("fsdp_transpose", "context"), "tensor_sequence"),
      )
      self.assertEqual(
          compound_moe_spec(CPasEP(), "wi_0", ("experts", "expert_embed", "expert_mlp"), mode="fsdp_x_ep"),
          P(("fsdp", "context", "expert"), None, ("tensor", "tensor_sequence")),
      )
      self.assertEqual(
          compound_moe_spec(TPasEP(two_d_fsdp=True), "wi_0", ("experts", "expert_embed", "expert_mlp"), mode="fsdp_x_ep"),
          P(("fsdp", "fsdp_transpose", "expert", "tensor"), None, "tensor_sequence"),
      )
      with self.assertRaisesRegex(ValueError, "Unknown compound MoE mode"):
        compound_moe_spec(None, "wi_0", ("experts", "expert_embed", "expert_mlp"), mode="bad_mode")

    # Verify compound_moe_param_metadata matches param_metadata eager_sharding contract.
    with jax.sharding.set_mesh(None):
      meta_strict = compound_moe_param_metadata(None, "wi_0", ("experts", "expert_embed", "expert_mlp"))
      self.assertTrue(meta_strict["eager_sharding"])
      self.assertEqual(
          meta_strict["eager_sharding"],
          param_metadata(DefaultSharding(), "wi_0", ("experts", "expert_embed", "expert_mlp"))["eager_sharding"],
      )
      with sharding_lib.allow_unsharded_construction():
        meta_relaxed = compound_moe_param_metadata(None, "wi_0", ("experts", "expert_embed", "expert_mlp"))
        self.assertFalse(meta_relaxed["eager_sharding"])


class ShardingStrategyPresetsTest(parameterized.TestCase):
  """Tests `EPasDP`, `EPasCP`, `CPasEP`, `TPasEP`, `FSDPasDPForAttn`, and `sharding_preset_from_name`."""

  def test_ep_as_dp_rules(self):
    std = EPasDP()
    self.assertEqual(
        std("wi_0", ("experts", "expert_embed", "expert_mlp")),
        P("expert", ("fsdp", "fsdp_transpose", "context"), ("tensor", "tensor_sequence")),
    )
    self.assertEqual(std("q_lora", ("q_lora", "q_lora_up_proj")), P(("fsdp", "fsdp_transpose", "expert"), None))
    self.assertEqual(std("router", ("embed_router", None)), P(("fsdp", "fsdp_transpose"), None))
    self.assertEqual(
        std("act", ("batch", "length", "embed"), TensorType.ACTIVATION),
        P(("data", "fsdp", "fsdp_transpose", "expert"), "context", None),
    )

    muon = EPasDP(compound_fsdp_ep=True)
    self.assertEqual(
        muon("wi_0", ("experts", "expert_embed", "expert_mlp")),
        P(("fsdp", "expert"), None, ("tensor", "tensor_sequence")),
    )
    self.assertEqual(
        muon("wo", ("experts", "expert_mlp", "expert_embed")), P(("fsdp", "expert"), ("tensor", "tensor_sequence"), None)
    )
    self.assertEqual(
        muon("act", ("experts", "expert_embed", "expert_mlp"), TensorType.ACTIVATION),
        P("expert", None, ("tensor", "tensor_sequence")),
    )

    muon_2d = EPasDP(compound_fsdp_ep=True, two_d_fsdp=True)
    self.assertEqual(
        muon_2d("wi_0", ("experts", "expert_embed", "expert_mlp")),
        P(("fsdp", "fsdp_transpose", "expert"), None, ("tensor", "tensor_sequence")),
    )

  def test_ep_as_cp_rules(self):
    s = EPasCP()
    self.assertEqual(
        s("attn_act", ("batch", "length", "heads", "head_dim"), TensorType.ACTIVATION),
        P(("data", "fsdp", "fsdp_transpose"), ("context", "expert"), ("tensor", "tensor_sequence"), None),
    )
    self.assertEqual(
        s("norm_act", ("batch", "norm_length", "embed"), TensorType.ACTIVATION),
        P(("data", "fsdp", "fsdp_transpose"), ("context", "expert", "tensor_sequence"), None),
    )
    self.assertEqual(
        s("wi_0", ("experts", "expert_embed", "expert_mlp")),
        P("expert", ("fsdp", "fsdp_transpose", "context"), ("tensor", "tensor_sequence")),
    )
    self.assertEqual(
        EPasCP(compound_fsdp_ep=True)("wi_0", ("experts", "expert_embed", "expert_mlp")),
        P(("fsdp", "expert"), None, ("tensor", "tensor_sequence")),
    )

  def test_cp_as_ep_rules(self):
    s = CPasEP()
    self.assertEqual(
        s("wi_0", ("experts", "expert_embed", "expert_mlp")),
        P(("context", "expert"), ("fsdp", "fsdp_transpose"), ("tensor", "tensor_sequence")),
    )
    self.assertEqual(
        s("gate", ("embed", "mlp")),
        P(("fsdp", "fsdp_transpose", "context", "expert"), ("tensor", "tensor_sequence")),
    )
    self.assertEqual(
        s("q_lora", ("q_lora", "q_lora_up_proj")),
        P(("fsdp", "fsdp_transpose", "context", "expert"), None),
    )
    self.assertEqual(
        s("attn_act", ("batch", "length", "heads", "head_dim"), TensorType.ACTIVATION),
        P(("data", "fsdp", "fsdp_transpose", "expert"), "context", ("tensor", "tensor_sequence"), None),
    )
    self.assertEqual(
        s("act", ("experts", "expert_embed", "expert_mlp"), TensorType.ACTIVATION),
        P(("context", "expert"), None, ("tensor", "tensor_sequence")),
    )
    s_comp = CPasEP(compound_fsdp_ep=True)
    self.assertEqual(
        s_comp("wi_0", ("experts", "expert_embed", "expert_mlp")),
        P(("fsdp", "context", "expert"), None, ("tensor", "tensor_sequence")),
    )

    class _BrokenCPasEP(DefaultSharding):
      """Negative control: claiming `context` on `experts` without excluding it from `expert_embed`."""

      def map_axis(self, axis: str, tensor_name: str, tensor_type: TensorType):
        if axis == "experts":
          return ("context", "expert")
        return super().map_axis(axis, tensor_name, tensor_type)

    with self.assertRaisesRegex(ValueError, "maps both 'experts' and 'expert_embed' onto mesh axis 'context'"):
      _BrokenCPasEP()("wi_0", ("experts", "expert_embed", "expert_mlp"))

  def test_tp_as_ep_rules(self):
    s = TPasEP()
    self.assertEqual(
        s("wi_0", ("experts", "expert_embed", "expert_mlp")),
        P(("expert", "tensor"), ("fsdp", "fsdp_transpose", "context"), "tensor_sequence"),
    )
    self.assertEqual(
        s("q_proj", ("embed", "heads", "head_dim")),
        P(("fsdp", "fsdp_transpose", "context", "expert"), ("tensor", "tensor_sequence"), None),
    )
    s_muon = TPasEP(muon_friendly=True)
    self.assertEqual(
        s_muon("wi_0", ("experts", "expert_embed", "expert_mlp")),
        P(("expert", "tensor"), None, "tensor_sequence"),
    )
    s_comp = TPasEP(compound_fsdp_ep=True)
    self.assertEqual(
        s_comp("wi_0", ("experts", "expert_embed", "expert_mlp")),
        P(("fsdp", "expert", "tensor"), None, "tensor_sequence"),
    )

  def test_fsdp_as_dp_for_attn_rules(self):
    s = FSDPasDPForAttn()
    # MLA compressed projections shard over "expert" and replicate over FSDP.
    self.assertEqual(s("q_lora", ("q_lora", "q_lora_up_proj")), P("expert", None))
    self.assertEqual(s("kv_lora", ("kv_lora", "kv_lora_up_proj")), P("expert", None))
    self.assertEqual(s("q_proj", ("embed_attn", "heads", "head_dim")), P("expert", ("tensor", "tensor_sequence"), None))
    self.assertEqual(s("wq_a", ("embed", None)), P("expert", None))
    self.assertEqual(s("wkv_a", ("embed", None)), P("expert", None))
    self.assertEqual(s("wk", ("embed", None)), P("expert", None))
    self.assertEqual(s("kv_a_proj_with_mqa", ("embed", None)), P("expert", None))
    # Dense MLP and token embeddings still shard over FSDP x CP x EP.
    self.assertEqual(
        s("mlp_gate", ("embed", "mlp")),
        P(("fsdp", "fsdp_transpose", "context", "expert"), ("tensor", "tensor_sequence")),
    )
    # Attention activations still shard batch across data x fsdp x fsdp_transpose x expert.
    self.assertEqual(
        s("mla_q_act", ("batch", "length", "q_lora"), TensorType.ACTIVATION),
        P(("data", "fsdp", "fsdp_transpose", "expert"), "context", None),
    )
    # Routed experts stay EP + FSDP x CP sharded.
    self.assertEqual(
        s("wi_0", ("experts", "expert_embed", "expert_mlp")),
        P("expert", ("fsdp", "fsdp_transpose", "context"), ("tensor", "tensor_sequence")),
    )
    # Compound FSDPxEP mode shards experts over ("fsdp", "expert") with expert_embed=None.
    s_comp = FSDPasDPForAttn(compound_fsdp_ep=True)
    self.assertEqual(
        s_comp("wi_0", ("experts", "expert_embed", "expert_mlp")),
        P(("fsdp", "expert"), None, ("tensor", "tensor_sequence")),
    )
    # Router weight shards on FSDP by default (-rf) and replicates when shard_router_on_fsdp=False.
    self.assertEqual(s("router", ("embed_router", None)), P(("fsdp", "fsdp_transpose"), None))
    self.assertEqual(s("router", ("embed", None)), P(("fsdp", "fsdp_transpose"), None))
    s_no_rf = FSDPasDPForAttn(shard_router_on_fsdp=False)
    self.assertEqual(s("router_weight", ("embed", None)), P(("fsdp", "fsdp_transpose"), None))
    self.assertEqual(s_no_rf("router", ("embed_router", None)), P(None, None))
    self.assertEqual(s_no_rf("router", ("embed", None)), P(None, None))
    self.assertEqual(s_no_rf("router_weight", ("embed", None)), P(None, None))

  def test_preset_registry_and_sharding_from_config(self):
    self.assertEqual(sharding_preset_from_name(None), DefaultSharding())
    self.assertEqual(sharding_preset_from_name("default"), DefaultSharding())
    self.assertEqual(sharding_preset_from_name("ep-as-dp"), EPasDP())
    self.assertEqual(sharding_preset_from_name("ep_as_cp"), EPasCP())
    self.assertEqual(sharding_preset_from_name("cp-as-ep"), CPasEP())
    self.assertEqual(sharding_preset_from_name("tp-as-ep"), TPasEP())
    self.assertEqual(sharding_preset_from_name("tp_x_ep"), TPasEP())
    self.assertEqual(sharding_preset_from_name("fsdp-as-dp-for-attn"), FSDPasDPForAttn(shard_router_on_fsdp=True))
    self.assertEqual(sharding_preset_from_name("fsdp-as-dp-for-attn-rf"), FSDPasDPForAttn(shard_router_on_fsdp=True))
    self.assertEqual(sharding_preset_from_name("fsdp-as-dp-for-attn-no-rf"), FSDPasDPForAttn(shard_router_on_fsdp=False))
    self.assertEqual(sharding_preset_from_name("fsdp-x-ep"), EPasDP(compound_fsdp_ep=True))
    self.assertEqual(sharding_preset_from_name("expert-on-fsdp-ep"), EPasDP(compound_fsdp_ep=True))
    self.assertEqual(sharding_preset_from_name("shard-exp-on-fsdp"), EPasDP(compound_fsdp_ep=True))

    preset_instance = FSDPasDPForAttn(compound_fsdp_ep=True)
    self.assertIs(sharding_preset_from_name(preset_instance), preset_instance)
    self.assertEqual(sharding_preset_from_name(EPasDP(), compound_fsdp_ep=True), EPasDP(compound_fsdp_ep=True))
    self.assertEqual(
        sharding_preset_from_name(FSDPasDPForAttn(), compound_fsdp_ep=True, two_d_fsdp=True),
        FSDPasDPForAttn(compound_fsdp_ep=True, two_d_fsdp=True),
    )

    for key in SHARDING_PRESETS:
      self.assertIsInstance(sharding_preset_from_name(key), DefaultSharding)

    with self.assertRaisesRegex(ValueError, "Unknown m3 sharding preset"):
      sharding_preset_from_name("nonexistent_preset")
    with self.assertRaisesRegex(TypeError, "Expected str or None"):
      sharding_preset_from_name(123)  # pytype: disable=wrong-arg-types

    cfg_default = SimpleNamespace()
    self.assertEqual(sharding_from_config(cfg_default), DefaultSharding())

    cfg_mla = SimpleNamespace(
        sharding_preset="fsdp-as-dp-for-attn",
        compound_fsdp_ep=True,
    )
    self.assertEqual(sharding_from_config(cfg_mla), FSDPasDPForAttn(compound_fsdp_ep=True))

    cfg_exp_fsdp = SimpleNamespace(compound_fsdp_ep=True)
    self.assertEqual(sharding_from_config(cfg_exp_fsdp), EPasDP(compound_fsdp_ep=True))

    # Legacy `sharding_strategy="experimental"` and `shard_exp_on_fsdp=True` are NOT read by `sharding_from_config`.
    cfg_legacy = SimpleNamespace(sharding_strategy="experimental", sharding_preset="", shard_exp_on_fsdp=True)
    self.assertEqual(sharding_from_config(cfg_legacy), DefaultSharding())

    # Declared fields validate cleanly through both `DeviceConfig` and `pyconfig.initialize` with EP > 1.
    from maxtext.configs import pyconfig  # pylint: disable=import-outside-toplevel
    from maxtext.m3.configs.config import DeviceConfig  # pylint: disable=import-outside-toplevel
    import os  # pylint: disable=import-outside-toplevel

    dev_cfg = DeviceConfig(sharding_preset="fsdp_as_dp_for_attn", compound_fsdp_ep=True)
    self.assertEqual(sharding_from_config(dev_cfg), FSDPasDPForAttn(compound_fsdp_ep=True))

    from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR  # pylint: disable=import-outside-toplevel

    base_yml = os.path.join(MAXTEXT_CONFIGS_DIR, "base.yml")
    py_cfg = pyconfig.initialize(
        ["train.py", base_yml],
        enable_single_controller=True,
        use_m3_model=True,
        model_name="qwen3.8-flash-next-mini",
        override_model_config=True,
        sharding_strategy="experimental",
        sharding_preset="fsdp_as_dp_for_attn",
        compound_fsdp_ep=True,
        ici_fsdp_parallelism=2,
        ici_expert_parallelism=2,
        enable_checkpointing=False,
        base_output_directory="/tmp/m3_preset_test",
    )
    self.assertEqual(sharding_from_config(py_cfg), FSDPasDPForAttn(compound_fsdp_ep=True))

    py_cfg_exp_only = pyconfig.initialize(
        ["train.py", base_yml],
        enable_single_controller=True,
        sharding_strategy="experimental",
        enable_checkpointing=False,
    )
    self.assertEqual(sharding_from_config(py_cfg_exp_only), DefaultSharding())

  def test_sharding_from_config_2d_fsdp_outside_mesh_and_default_validation(self):
    with jax.sharding.set_mesh(None):
      # (a) Outside `with jax.set_mesh(mesh):`, `sharding_from_config` extracts 2D FSDP from ICI/DCN config flags.
      s_no_preset = sharding_from_config(SimpleNamespace(ici_fsdp_transpose_parallelism=2, compound_fsdp_ep=True))
      self.assertEqual(s_no_preset, EPasDP(compound_fsdp_ep=True, two_d_fsdp=True))
      self.assertEqual(
          s_no_preset("wi_0", ("experts", "expert_embed", "expert_mlp")),
          P(("fsdp", "fsdp_transpose", "expert"), None, ("tensor", "tensor_sequence")),
      )

      s_dcn = sharding_from_config(SimpleNamespace(dcn_fsdp_transpose_parallelism=2, compound_fsdp_ep=True))
      self.assertEqual(s_dcn, EPasDP(compound_fsdp_ep=True, two_d_fsdp=True))

      for preset_name, expected_experts_axis, expected_mlp_axis in (
          ("ep_as_dp", ("fsdp", "fsdp_transpose", "expert"), ("tensor", "tensor_sequence")),
          ("fsdp_as_dp_for_attn", ("fsdp", "fsdp_transpose", "expert"), ("tensor", "tensor_sequence")),
          ("ep_as_cp", ("fsdp", "fsdp_transpose", "expert"), ("tensor", "tensor_sequence")),
          ("cp_as_ep", ("fsdp", "fsdp_transpose", "context", "expert"), ("tensor", "tensor_sequence")),
          ("tp_as_ep", ("fsdp", "fsdp_transpose", "expert", "tensor"), "tensor_sequence"),
      ):
        s_preset = sharding_from_config(
            SimpleNamespace(
                sharding_preset=preset_name,
                compound_fsdp_ep=True,
                ici_fsdp_transpose_parallelism=2,
            )
        )
        self.assertEqual(
            s_preset("wi_0", ("experts", "expert_embed", "expert_mlp")),
            P(expected_experts_axis, None, expected_mlp_axis),
        )

      s_alias = sharding_from_config(SimpleNamespace(sharding_preset="fsdp-x-ep", ici_fsdp_transpose_parallelism=2))
      self.assertEqual(
          s_alias("wi_0", ("experts", "expert_embed", "expert_mlp")),
          P(("fsdp", "fsdp_transpose", "expert"), None, ("tensor", "tensor_sequence")),
      )

      # (b) `sharding_preset="default"` (or `None` / `DefaultSharding()`) + `ici_fsdp_transpose_parallelism=2`
      # (`compound_fsdp_ep=False`) succeeds cleanly and shards over `fsdp_transpose` natively.
      for default_preset in (None, "", "default", "DEFAULT", "default-sharding", DefaultSharding()):
        s_def = sharding_from_config(SimpleNamespace(sharding_preset=default_preset, ici_fsdp_transpose_parallelism=2))
        self.assertEqual(s_def, DefaultSharding())
        self.assertEqual(
            s_def("wi_0", ("experts", "expert_embed", "expert_mlp")),
            P("expert", ("fsdp", "fsdp_transpose", "context"), ("tensor", "tensor_sequence")),
        )

    # (d) Passing incompatible options to `"default"` or `DefaultSharding()` fails closed with `ValueError`.
    with self.assertRaisesRegex(ValueError, "does not accept options"):
      sharding_preset_from_name("default", compound_fsdp_ep=True)
    with self.assertRaisesRegex(ValueError, "does not accept options"):
      sharding_preset_from_name("", compound_fsdp_ep=True)
    with self.assertRaisesRegex(ValueError, "does not accept options"):
      sharding_preset_from_name(DefaultSharding(), compound_fsdp_ep=True)
    with self.assertRaisesRegex(ValueError, "does not accept options"):
      sharding_preset_from_name(EPasDP(), unknown_flag=True)
    with self.assertRaisesRegex(ValueError, "does not accept options"):
      sharding_from_config(SimpleNamespace(sharding_preset="default", compound_fsdp_ep=True))

  def test_moe_from_config_integrates_with_presets(self):
    cfg = SimpleNamespace(
        num_experts=8,
        sparse_matmul=True,
        use_tokamax_gmm=True,
        use_gmm_v2=True,
        capacity_factor=-1.0,
        ragged_buffer_factor=-1.0,
        use_ragged_sort=False,
        use_ring_of_experts=False,
    )
    exec_mla = moe_lib.moe_from_config(cfg, FSDPasDPForAttn(compound_fsdp_ep=True))
    self.assertEqual(exec_mla.batch, ("data", "fsdp", "fsdp_transpose", "expert"))
    self.assertEqual(exec_mla.experts, ("fsdp", "expert"))
    self.assertEqual(exec_mla.mlp, ("tensor", "tensor_sequence"))

    exec_ep_cp = moe_lib.moe_from_config(cfg, EPasCP())
    self.assertEqual(exec_ep_cp.batch, ("data", "fsdp", "fsdp_transpose"))
    self.assertEqual(exec_ep_cp.experts, "expert")
    self.assertEqual(exec_ep_cp.mlp, ("tensor", "tensor_sequence"))

    exec_tp_ep = moe_lib.moe_from_config(cfg, TPasEP())
    self.assertEqual(exec_tp_ep.experts, ("expert", "tensor"))
    self.assertEqual(exec_tp_ep.mlp, "tensor_sequence")


class ShardingPresetsMultiDeviceAndNegativeControlTest(parameterized.TestCase):
  """4-CPU and virtual-topology (`tpu7x-128`) verification with negative controls."""

  def setUp(self):
    super().setUp()
    if jax.device_count() < 4 or jax.device_count() % 2 != 0:
      self.skipTest(f"Requires >= 4 even devices; found {jax.device_count()}.")
    self.devices = jax.devices()[:4]

  def test_eager_sharding_survives_nnx_wrappers_import_with_negative_control(self):
    """Proves `eager_sharding=True` shards weights after importing `nnx_wrappers`, while omitting it replicates."""
    # Importing legacy `nnx_wrappers` sets `flax_always_shard_variable=False` globally.
    import maxtext.layers.nnx_wrappers as _nnx_wrappers  # pylint: disable=import-outside-toplevel,unused-import

    self.enter_context(nnx.use_eager_sharding(False))
    self.assertFalse(nnx.using_eager_sharding())

    mesh = create_mesh(ici={"fsdp": 2, "expert": 2}, devices=self.devices)
    preset = FSDPasDPForAttn(compound_fsdp_ep=True)
    gu_shape = (8, 16, 32)
    q_lora_shape = (16, 8)

    with jax.sharding.set_mesh(mesh):
      # Positive path: `param_metadata` and `compound_moe_param_metadata` inject `eager_sharding=True`.
      gu_meta = compound_moe_param_metadata(preset, "gate_up_proj", ("experts", "expert_embed", "expert_mlp"))
      q_meta = param_metadata(preset, "q_lora", ("q_lora", "q_lora_up_proj"))
      self.assertTrue(gu_meta["eager_sharding"])
      self.assertTrue(q_meta["eager_sharding"])

      gu_param = nnx.Param(jnp.zeros(gu_shape, dtype=jnp.float32), **gu_meta)
      q_param = nnx.Param(jnp.zeros(q_lora_shape, dtype=jnp.float32), **q_meta)

      # Negative control: omitting `eager_sharding=True` leaves parameters 100% replicated when
      # `nnx_wrappers` turns off global `flax_always_shard_variable`.
      gu_broken = nnx.Param(jnp.zeros(gu_shape, dtype=jnp.float32), out_sharding=gu_meta["out_sharding"])
      q_broken = nnx.Param(jnp.zeros(q_lora_shape, dtype=jnp.float32), out_sharding=q_meta["out_sharding"])

    # Positive assertions:
    # 8 experts sharded across fsdp=2 x expert=2 -> 2 complete (16, 32) expert matrices per device (Muon-friendly).
    self.assertEqual(gu_param[...].sharding.shard_shape(gu_shape), (2, 16, 32))
    # q_lora (16, 8) sharded across expert=2 (replicated over fsdp=2) -> (8, 8) per device.
    self.assertEqual(q_param[...].sharding.shard_shape(q_lora_shape), (8, 8))

    # Negative-control assertions: without `eager_sharding=True`, both parameters silently replicate.
    self.assertEqual(gu_broken[...].sharding.shard_shape(gu_shape), gu_shape)
    self.assertEqual(q_broken[...].sharding.shard_shape(q_lora_shape), q_lora_shape)
    self.assertNotEqual(gu_param[...].sharding.shard_shape(gu_shape), gu_broken[...].sharding.shard_shape(gu_shape))

  def test_fsdp_as_dp_for_attn_activation_and_hlo_negative_controls(self):
    """Proves `FSDPasDPForAttn` shards MLA activations over FSDP x EP and avoids expert all-gather in HLO."""
    mesh = create_mesh(ici={"fsdp": 2, "expert": 2}, devices=self.devices)
    preset = FSDPasDPForAttn(compound_fsdp_ep=True)

    batch_size, seq_len, emb_dim, q_lora_dim, mlp_dim, num_experts = 4, 8, 16, 8, 32, 8
    x = jnp.ones((batch_size, seq_len, emb_dim), dtype=jnp.float32)
    wq_a = jnp.ones((emb_dim, q_lora_dim), dtype=jnp.float32)
    wi_0 = jnp.ones((num_experts, emb_dim, mlp_dim), dtype=jnp.float32)

    with jax.sharding.set_mesh(mesh):
      x_shd = NamedSharding(mesh, preset("x", ("batch", "length", "embed"), TensorType.ACTIVATION))
      wq_shd = NamedSharding(mesh, preset("q_lora", ("q_lora", "q_lora_up_proj")))
      wi_shd = NamedSharding(mesh, preset("wi_0", ("experts", "expert_embed", "expert_mlp")))

      x_dev = jax.device_put(x, x_shd)
      wq_dev = jax.device_put(wq_a, wq_shd)
      wi_dev = jax.device_put(wi_0, wi_shd)

      @jax.jit
      def mla_and_expert_step(x_in, wq_in, wi_in):
        q_latent = jnp.einsum("bsd,dq->bsq", x_in, wq_in)
        q_latent = preset.constrain(q_latent, "q_latent", ("batch", "length", "q_lora"))
        # Each of the 4 (fsdp=2, expert=2) shards owns 1 batch row and 2 local experts:
        # local token-to-local-expert contraction requires zero all-gather on `wi_in`.
        return q_latent, wi_in

      q_out, wi_out = mla_and_expert_step(x_dev, wq_dev, wi_dev)

      # Negative control 1: a broken rule that shards attention activations only on "expert"
      # replicates MLA activations across `fsdp=2`, doubling per-device activation memory.
      @jax.jit
      def broken_mla_activation_step(x_in, wq_in):
        q_latent = jnp.einsum("bsd,dq->bsq", x_in, wq_in)
        return jax.lax.with_sharding_constraint(q_latent, P("expert", None, None))

      q_broken = broken_mla_activation_step(x_dev, wq_dev)

      # Negative control 2: a broken rule that leaves `wi_in` replicated in output forces an all-gather in HLO.
      hlo_good = (
          jax.jit(lambda w: w * 2.0, in_shardings=(wi_shd,), out_shardings=wi_shd).lower(wi_dev).compile().as_text()
      )
      replicated_shd = NamedSharding(mesh, P(None, None, None))
      hlo_bad = (
          jax.jit(lambda w: w * 2.0, in_shardings=(wi_shd,), out_shardings=replicated_shd)
          .lower(wi_dev)
          .compile()
          .as_text()
      )

    # Positive assertions:
    # 1. MLA activation `q_out` is sharded across fsdp=2 x expert=2 -> 1 batch row per device `(1, 8, 8)`.
    self.assertEqual(q_out.sharding.shard_shape(q_out.shape), (1, seq_len, q_lora_dim))
    # 2. Expert weights `wi_out` stay sharded across ("fsdp", "expert") -> `(2, 16, 32)` per device, zero all-gather.
    self.assertEqual(wi_out.sharding.shard_shape(wi_out.shape), (2, emb_dim, mlp_dim))
    self.assertNotIn("all-gather", hlo_good)

    # Negative-control assertions:
    # 1. Broken MLA activation rule replicates across `fsdp=2` -> `(2, 8, 8)` per device (2x activation memory).
    self.assertEqual(q_broken.sharding.shard_shape(q_broken.shape), (2, seq_len, q_lora_dim))
    # 2. Broken expert output rule emits an `all-gather` in compiled HLO.
    self.assertIn("all-gather", hlo_bad)

  def test_tpu7x_128_virtual_topology_abstract_sharding_and_hlo(self):
    """Verifies abstract shard shapes and lowered HLO on virtual 128-chip `tpu7x-128` topologies."""
    # `tpu7x-128` (128 chips): fsdp=16, expert=8 (16 * 8 = 128).
    tpu7x_128_ep_mesh = jax.sharding.AbstractMesh(
        (1, 1, 16, 1, 1, 1, 1, 8),
        MESH_AXIS_NAMES,
        axis_types=(jax.sharding.AxisType.Explicit,) * len(MESH_AXIS_NAMES),
    )
    mla_comp = FSDPasDPForAttn(compound_fsdp_ep=True)
    with jax.sharding.use_abstract_mesh(tpu7x_128_ep_mesh):
      q_lora_spec = mla_comp("q_lora", ("q_lora", "q_lora_up_proj"))
      mla_act_spec = mla_comp("q_latent", ("batch", "length", "heads", "head_dim"), TensorType.ACTIVATION)
      wi_comp_spec = mla_comp("wi_0", ("experts", "expert_embed", "expert_mlp"))
      router_spec = mla_comp("router", ("embed_router", None))

    self.assertEqual(q_lora_spec, P("expert", None))
    self.assertEqual(mla_act_spec, P(("fsdp", "expert"), None, None, None))
    self.assertEqual(wi_comp_spec, P(("fsdp", "expert"), None, ("tensor", "tensor_sequence")))
    self.assertEqual(router_spec, P(("fsdp", "fsdp_transpose"), None))

    # Verify per-chip shard shapes at DeepSeek/GLM scale on `tpu7x-128` (256 experts, 7168 emb, 1536 q_lora, 2048 mlp):
    self.assertEqual(NamedSharding(tpu7x_128_ep_mesh, q_lora_spec).shard_shape((7168, 1536)), (896, 1536))
    self.assertEqual(NamedSharding(tpu7x_128_ep_mesh, mla_act_spec).shard_shape((128, 4096, 16, 128)), (1, 4096, 16, 128))
    self.assertEqual(NamedSharding(tpu7x_128_ep_mesh, wi_comp_spec).shard_shape((256, 7168, 2048)), (2, 7168, 2048))
    self.assertEqual(NamedSharding(tpu7x_128_ep_mesh, router_spec).shard_shape((7168, 256)), (448, 256))

    # Also check `EPasCP`, `CPasEP`, and `TPasEP` on a multi-axis 128-chip virtual mesh
    # (`fsdp=8, fsdp_transpose=2, context=2, tensor=2, expert=2` -> 8*2*2*2*2 = 128 chips).
    tpu7x_128_hybrid_mesh = jax.sharding.AbstractMesh(
        (1, 1, 8, 2, 2, 2, 1, 2),
        MESH_AXIS_NAMES,
    )
    ep_cp = EPasCP()
    cp_ep = CPasEP()
    tp_ep = TPasEP()

    self.assertEqual(
        NamedSharding(
            tpu7x_128_hybrid_mesh,
            project_spec_to_mesh(
                ep_cp("norm_in", ("batch", "norm_length", "embed"), TensorType.ACTIVATION), tpu7x_128_hybrid_mesh
            ),
        ).shard_shape((128, 4096, 7168)),
        (8, 1024, 7168),
    )
    self.assertEqual(
        NamedSharding(
            tpu7x_128_hybrid_mesh,
            project_spec_to_mesh(cp_ep("wi_0", ("experts", "expert_embed", "expert_mlp")), tpu7x_128_hybrid_mesh),
        ).shard_shape((256, 7168, 2048)),
        (64, 448, 1024),
    )
    self.assertEqual(
        NamedSharding(
            tpu7x_128_hybrid_mesh,
            project_spec_to_mesh(tp_ep("wi_0", ("experts", "expert_embed", "expert_mlp")), tpu7x_128_hybrid_mesh),
        ).shard_shape((256, 7168, 2048)),
        (64, 224, 2048),
    )
    self.assertEqual(
        NamedSharding(
            tpu7x_128_hybrid_mesh,
            project_spec_to_mesh(
                TPasEP(muon_friendly=True)("wi_0", ("experts", "expert_embed", "expert_mlp")), tpu7x_128_hybrid_mesh
            ),
        ).shard_shape((256, 7168, 2048)),
        (64, 7168, 2048),
    )

    # Verify `sharding_from_config` resolved outside `with mesh:` does not replicate across `fsdp_transpose=2`
    # when placed on a 4-CPU 2D FSDP x EP mesh (`fsdp=1, fsdp_transpose=2, expert=2`).
    mesh_2d = create_mesh(ici={"fsdp": 1, "fsdp_transpose": 2, "expert": 2}, devices=self.devices)
    with jax.sharding.set_mesh(None):
      s_outside = sharding_from_config(
          SimpleNamespace(
              sharding_preset="fsdp_as_dp_for_attn",
              compound_fsdp_ep=True,
              ici_fsdp_transpose_parallelism=2,
          )
      )
      spec_outside = s_outside("wi_0", ("experts", "expert_embed", "expert_mlp"))
    arr_good = jax.device_put(jnp.zeros((8, 16, 32), dtype=jnp.float32), NamedSharding(mesh_2d, spec_outside))
    arr_bad_1d = jax.device_put(
        jnp.zeros((8, 16, 32), dtype=jnp.float32),
        NamedSharding(mesh_2d, P(("fsdp", "expert"), None, ("tensor", "tensor_sequence"))),
    )
    self.assertEqual(arr_good.sharding.shard_shape((8, 16, 32)), (2, 16, 32))
    self.assertEqual(arr_bad_1d.sharding.shard_shape((8, 16, 32)), (4, 16, 32))

    # Confirm sharding module remains free of numpy and flax.linen imports.
    self.assertNotIn("np", vars(sharding_lib))
    self.assertNotIn("numpy", vars(sharding_lib))
    self.assertFalse(any(k == "linen" for k in vars(sharding_lib)))

  def test_moe_from_config_compound_presets_execute_forward_and_backward_parity(self):
    """Executes 4-CPU `shard_map` forward and backward passes for all 5 compound MoE presets."""
    cfg_moe = SimpleNamespace(
        num_experts=8,
        sparse_matmul=True,
        use_tokamax_gmm=True,
        use_gmm_v2=True,
        capacity_factor=-1.0,
        ragged_buffer_factor=-1.0,
        use_ragged_sort=False,
        use_ring_of_experts=False,
    )
    batch_size, seq_len, emb_dim, mlp_dim, num_experts, top_k = 4, 4, 16, 12, 8, 2
    x = jax.random.normal(jax.random.key(90), (batch_size, seq_len, emb_dim), dtype=jnp.float32)
    cot = jax.random.normal(jax.random.key(91), x.shape, dtype=jnp.float32)
    logits = jax.random.normal(jax.random.key(92), (batch_size, seq_len, num_experts), dtype=jnp.float32)
    top_vals, indices = jax.lax.top_k(jax.nn.softmax(logits, axis=-1), top_k)
    weights = top_vals / jnp.sum(top_vals, axis=-1, keepdims=True)
    wi_0 = jax.random.normal(jax.random.key(93), (num_experts, emb_dim, mlp_dim), dtype=jnp.float32) * 0.1
    wi_1 = jax.random.normal(jax.random.key(94), (num_experts, emb_dim, mlp_dim), dtype=jnp.float32) * 0.1
    wo = jax.random.normal(jax.random.key(95), (num_experts, mlp_dim, emb_dim), dtype=jnp.float32) * 0.1
    w_gu = jnp.stack([wi_0, wi_1], axis=2)

    def _ref_loss(x_i, w0, w1, w_o):
      y = moe_lib.DENSE(x_i, weights, indices, w0, w1, w_o)
      return jnp.sum(y * cot), y

    (_, ref_out), ref_grads = jax.jit(jax.value_and_grad(_ref_loss, argnums=(0, 1, 2, 3), has_aux=True))(
        x, wi_0, wi_1, wo
    )

    cases = (
        ("fsdp_as_dp_for_attn_comp", FSDPasDPForAttn(compound_fsdp_ep=True), {"fsdp": 2, "expert": 2}),
        ("ep_as_dp_comp", EPasDP(compound_fsdp_ep=True), {"fsdp": 2, "expert": 2}),
        ("ep_as_cp_comp", EPasCP(compound_fsdp_ep=True), {"fsdp": 2, "expert": 2}),
        ("cp_as_ep", CPasEP(), {"context": 2, "expert": 2}),
        ("tp_as_ep", TPasEP(), {"expert": 2, "tensor": 2}),
        ("tp_as_ep_seq_tp", TPasEP(), {"expert": 2, "tensor_sequence": 2}),
    )
    for case_name, preset, ici_axes in cases:
      with self.subTest(case=case_name):
        mesh = create_mesh(ici=ici_axes, devices=self.devices)
        exec_preset = moe_lib.moe_from_config(cfg_moe, preset)
        with jax.set_mesh(mesh):
          (_, got_out), got_grads = jax.jit(
              jax.value_and_grad(
                  lambda x_i, w0, w1, w_o, ex=exec_preset: (
                      jnp.sum(ex(x_i, weights, indices, w0, w1, w_o) * cot),
                      ex(x_i, weights, indices, w0, w1, w_o),
                  ),
                  argnums=(0, 1, 2, 3),
                  has_aux=True,
              )
          )(x, wi_0, wi_1, wo)
          got_fused = jax.jit(lambda x_i, gu, dn, ex=exec_preset: ex.fused(x_i, weights, indices, gu, dn))(x, w_gu, wo)

        self.assertLess(float(jnp.max(jnp.abs(got_out - ref_out))), 1e-4)
        self.assertLess(float(jnp.max(jnp.abs(got_fused - ref_out))), 1e-4)
        for g_got, g_ref in zip(got_grads, ref_grads):
          self.assertLess(float(jnp.max(jnp.abs(g_got - g_ref))), 1e-4)

    # Verify compound ('fsdp', 'expert') tuple axis_index works under capacity_factor > 0 and ragged_buffer_factor > 0.
    mesh_comp = create_mesh(ici={"fsdp": 2, "expert": 2}, devices=self.devices)
    for cap_kw in ({"capacity_factor": 0.75}, {"ragged_buffer_factor": 0.75}):
      cfg_cap = SimpleNamespace(num_experts=8, sparse_matmul=True, use_tokamax_gmm=True, use_gmm_v2=True, **cap_kw)
      exec_cap = moe_lib.moe_from_config(cfg_cap, EPasDP(compound_fsdp_ep=True))
      with jax.set_mesh(mesh_comp):
        out_cap = jax.jit(lambda x_i, ex=exec_cap: ex(x_i, weights, indices, wi_0, wi_1, wo))(x)
      self.assertEqual(out_cap.shape, x.shape)
      self.assertTrue(bool(jnp.all(jnp.isfinite(out_cap))))

    # Also verify ShardMode.EXPLICIT on TPasEP (where extra_ep=('tensor',) reshards inside _sharded_grouped
    # and restores the caller's P('expert', None, None) output sharding for downstream residual adds).
    mesh_exp = create_mesh(ici={"expert": 2, "tensor": 2}, devices=self.devices, explicit=True)
    tp_ep_preset = TPasEP(muon_friendly=True)
    exec_exp = moe_lib.moe_from_config(cfg_moe, tp_ep_preset)
    tok_shd = NamedSharding(mesh_exp, P("expert", None, None))
    w_shd = NamedSharding(mesh_exp, P(("expert", "tensor"), None, None))
    x_exp = jax.device_put(x, tok_shd)
    weights_exp = jax.device_put(weights, tok_shd)
    indices_exp = jax.device_put(indices, tok_shd)
    wi0_exp = jax.device_put(wi_0, w_shd)
    wi1_exp = jax.device_put(wi_1, w_shd)
    wo_exp = jax.device_put(wo, w_shd)
    with jax.set_mesh(mesh_exp):
      res_out = jax.jit(lambda x_i, w_i, idx_i, w0, w1, w_o: x_i + exec_exp(x_i, w_i, idx_i, w0, w1, w_o))(
          x_exp, weights_exp, indices_exp, wi0_exp, wi1_exp, wo_exp
      )
    self.assertEqual(res_out.sharding.spec, P("expert", None, None))
    self.assertLess(float(jnp.max(jnp.abs(res_out - (x + ref_out)))), 1e-4)

  def test_end_to_end_qwen3_8_preset_wiring_config_validation_and_tp_as_ep_context(self):
    """Verifies live Qwen3.8 + loop.setup_model preset wiring, config validation, and TPasEP+context sharding."""
    import os  # pylint: disable=import-outside-toplevel
    from maxtext.configs import pyconfig  # pylint: disable=import-outside-toplevel
    from maxtext.m3 import models as m3_models  # pylint: disable=import-outside-toplevel
    from maxtext.m3.configs.config import DeviceConfig  # pylint: disable=import-outside-toplevel
    from maxtext.m3.train import loop as m3_loop  # pylint: disable=import-outside-toplevel
    from maxtext.utils import model_creation_utils  # pylint: disable=import-outside-toplevel
    from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR  # pylint: disable=import-outside-toplevel

    base_yml = os.path.join(MAXTEXT_CONFIGS_DIR, "base.yml")

    # 1. Invalid preset names and use_m3_model=False fail closed at config initialization.
    with self.assertRaisesRegex(ValueError, "Unknown m3 sharding preset"):
      DeviceConfig(sharding_preset="not_a_real_preset")
    with self.assertRaisesRegex(ValueError, "cannot be combined with sharding_preset='default'"):
      DeviceConfig(sharding_preset="default", compound_fsdp_ep=True)
    with self.assertRaisesRegex(ValueError, "Unknown m3 sharding preset"):
      pyconfig.initialize(
          ["train.py", base_yml],
          enable_single_controller=True,
          use_m3_model=True,
          model_name="qwen3.8-flash-next-mini",
          override_model_config=True,
          sharding_preset="not_a_real_preset",
          enable_checkpointing=False,
          base_output_directory="/tmp/m3_preset_bad",
      )
    with self.assertRaisesRegex(ValueError, "require use_m3_model=True"):
      pyconfig.initialize(
          ["train.py", base_yml],
          enable_single_controller=True,
          use_m3_model=False,
          sharding_preset="ep_as_dp",
          enable_checkpointing=False,
          base_output_directory="/tmp/m3_preset_non_m3",
      )

    # 2. Non-DefaultSharding models fail closed when preset/compound_fsdp_ep is active.
    mesh_fsdp_ep = create_mesh(ici={"fsdp": 2, "expert": 2}, devices=self.devices)
    glm_cfg = SimpleNamespace(
        model_name="glm5.3-flash",
        sharding_preset="ep_as_dp",
        compound_fsdp_ep=True,
        shard_mode="auto",
    )
    with self.assertRaisesRegex(ValueError, "custom non-DefaultSharding"):
      m3_models.create_model(glm_cfg, mesh_fsdp_ep, rngs=nnx.Rngs(0))

    # 3. End-to-end pyconfig + model_creation_utils.create_model + loop.setup_model + fsdp_as_dp_for_attn_no_rf.
    q38_cfg = pyconfig.initialize(
        ["train.py", base_yml],
        model_name="qwen3.8-flash-next-mini",
        use_m3_model=True,
        override_model_config=True,
        scan_layers=False,
        sharding_preset="fsdp_as_dp_for_attn_no_rf",
        compound_fsdp_ep=True,
        ici_fsdp_parallelism=2,
        ici_expert_parallelism=2,
        per_device_batch_size=1.0,
        max_target_length=32,
        quantization="",
        enable_checkpointing=False,
        base_output_directory="/tmp/m3_preset_q38",
    )
    expected_gu_spec = P(("fsdp", "expert"), None, None, ("tensor", "tensor_sequence"))
    model_mcu = model_creation_utils.create_model(q38_cfg, mesh_fsdp_ep)
    gu_shd = model_mcu.decoder.layers_0.mlp.experts.gate_up_proj[...].sharding.spec
    self.assertEqual(gu_shd, expected_gu_spec)
    router_shd = model_mcu.decoder.layers_0.mlp.gate.linear.kernel[...].sharding.spec
    self.assertEqual(router_shd, P(None, None))

    # Verify abstract sharding fallback spares replicated router under fsdp_as_dp_for_attn_no_rf.
    _, abs_shd = model_creation_utils.create_nnx_abstract_model(q38_cfg, mesh=mesh_fsdp_ep)
    self.assertEqual(abs_shd.decoder.layers_0.mlp.gate.linear.kernel.get_value().sharding.spec, P(None, None))
    self.assertEqual(abs_shd.decoder.layers_0.mlp.experts.gate_up_proj.get_value().sharding.spec, expected_gu_spec)

    with jax.set_mesh(mesh_fsdp_ep):
      model_loop, _ = m3_loop.setup_model(q38_cfg, mesh_fsdp_ep, sharding=sharding_from_config(q38_cfg))
    self.assertEqual(model_loop.decoder.layers_0.mlp.experts.gate_up_proj[...].sharding.spec, expected_gu_spec)

    # 4. TPasEP with active context=2, tensor=2 preserves 'context' on sequence dim 1 without sequence all-gather.
    mesh_cp_tp = create_mesh(ici={"context": 2, "tensor": 2}, devices=self.devices)
    tp_ep = TPasEP(muon_friendly=True)
    ex_tp_ep = moe_lib.moe_from_config(
        SimpleNamespace(num_experts=4, sparse_matmul=True, use_tokamax_gmm=True, use_gmm_v2=True), tp_ep
    )
    # S=6 divides by context=2 (6 % 2 == 0) and tensor=2 (6 % 2 == 0), NOT context*tensor=4 (6 % 4 != 0);
    # B=2 divides by tensor=2, so tokens_spec becomes P('tensor', 'context', None) with 0 sequence all-gathers.
    x_cp = jax.device_put(
        jax.random.normal(jax.random.key(91), (2, 6, 8), dtype=jnp.float32),
        NamedSharding(mesh_cp_tp, P(None, "context", None)),
    )
    w_cp = jax.device_put(
        jnp.full((2, 6, 2), 0.5, dtype=jnp.float32), NamedSharding(mesh_cp_tp, P(None, "context", None))
    )
    i_cp = jax.device_put(jnp.zeros((2, 6, 2), dtype=jnp.int32), NamedSharding(mesh_cp_tp, P(None, "context", None)))
    gu_cp = jax.device_put(
        jax.random.normal(jax.random.key(92), (4, 8, 2, 6), dtype=jnp.float32),
        NamedSharding(mesh_cp_tp, P("tensor", None, None, None)),
    )
    dn_cp = jax.device_put(
        jax.random.normal(jax.random.key(93), (4, 6, 8), dtype=jnp.float32),
        NamedSharding(mesh_cp_tp, P("tensor", None, None)),
    )
    with jax.set_mesh(mesh_cp_tp):
      hlo_cp_tp = jax.jit(ex_tp_ep.fused).lower(x_cp, w_cp, i_cp, gu_cp, dn_cp).compile().as_text()
      out_cp_tp = jax.jit(ex_tp_ep.fused)(x_cp, w_cp, i_cp, gu_cp, dn_cp)
    seq_ag = [
        ln for ln in hlo_cp_tp.splitlines() if "=" in ln and "all-gather(" in ln.split("=", 1)[1] and "f32[2,6,8]" in ln
    ]
    self.assertEmpty(seq_ag)
    self.assertEqual(tuple(out_cp_tp.sharding.spec)[:2], (None, "context"))
    self.assertNotIn("tensor", tuple(out_cp_tp.sharding.spec))

  def test_ep_as_cp_and_cp_as_ep_preserve_sequence_sharding_without_sequence_all_gather(self):
    # 1. EPasCP on mesh(fsdp=2, expert=2): x sharded P('fsdp', 'expert', None) -> out P('fsdp', 'expert', None), 0 seq AG.
    mesh_ep_cp = create_mesh(ici={"fsdp": 2, "expert": 2}, devices=self.devices)
    ex_ep_cp = moe_lib.moe_from_config(
        SimpleNamespace(num_experts=4, sparse_matmul=True, use_tokamax_gmm=True, use_gmm_v2=True), EPasCP()
    )
    x1 = jax.device_put(
        jax.random.normal(jax.random.key(101), (4, 8, 8), dtype=jnp.float32),
        NamedSharding(mesh_ep_cp, P("fsdp", "expert", None)),
    )
    w1 = jax.device_put(jnp.full((4, 8, 2), 0.5, dtype=jnp.float32), NamedSharding(mesh_ep_cp, P("fsdp", "expert", None)))
    i1 = jax.device_put(jnp.zeros((4, 8, 2), dtype=jnp.int32), NamedSharding(mesh_ep_cp, P("fsdp", "expert", None)))
    gu1 = jax.device_put(
        jax.random.normal(jax.random.key(102), (4, 8, 2, 6), dtype=jnp.float32),
        NamedSharding(mesh_ep_cp, P("expert", None, None, None)),
    )
    dn1 = jax.device_put(
        jax.random.normal(jax.random.key(103), (4, 6, 8), dtype=jnp.float32),
        NamedSharding(mesh_ep_cp, P("expert", None, None)),
    )
    with jax.set_mesh(mesh_ep_cp):
      hlo_ep_cp = jax.jit(ex_ep_cp.fused).lower(x1, w1, i1, gu1, dn1).compile().as_text()
      out_ep_cp = jax.jit(ex_ep_cp.fused)(x1, w1, i1, gu1, dn1)
    seq_ag_ep_cp = [
        ln
        for ln in hlo_ep_cp.splitlines()
        if "=" in ln and "all-gather(" in ln.split("=", 1)[1] and "dimensions={1}" in ln
    ]
    self.assertEmpty(seq_ag_ep_cp, f"Expected 0 sequence all-gathers under EPasCP, got: {seq_ag_ep_cp}")
    self.assertEqual(tuple(out_ep_cp.sharding.spec)[:2], ("fsdp", "expert"))

    # 2. CPasEP on mesh(fsdp=1, context=2, expert=2): x sharded P('expert', 'context', None) -> 0 seq AG.
    mesh_cp_ep = create_mesh(ici={"fsdp": 1, "context": 2, "expert": 2}, devices=self.devices)
    ex_cp_ep = moe_lib.moe_from_config(
        SimpleNamespace(num_experts=4, sparse_matmul=True, use_tokamax_gmm=True, use_gmm_v2=True), CPasEP()
    )
    x2 = jax.device_put(
        jax.random.normal(jax.random.key(111), (4, 8, 8), dtype=jnp.float32),
        NamedSharding(mesh_cp_ep, P("expert", "context", None)),
    )
    w2 = jax.device_put(
        jnp.full((4, 8, 2), 0.5, dtype=jnp.float32), NamedSharding(mesh_cp_ep, P("expert", "context", None))
    )
    i2 = jax.device_put(jnp.zeros((4, 8, 2), dtype=jnp.int32), NamedSharding(mesh_cp_ep, P("expert", "context", None)))
    gu2 = jax.device_put(
        jax.random.normal(jax.random.key(112), (4, 8, 2, 6), dtype=jnp.float32),
        NamedSharding(mesh_cp_ep, P(("context", "expert"), None, None, None)),
    )
    dn2 = jax.device_put(
        jax.random.normal(jax.random.key(113), (4, 6, 8), dtype=jnp.float32),
        NamedSharding(mesh_cp_ep, P(("context", "expert"), None, None)),
    )
    with jax.set_mesh(mesh_cp_ep):
      hlo_cp_ep = jax.jit(ex_cp_ep.fused).lower(x2, w2, i2, gu2, dn2).compile().as_text()
      out_cp_ep = jax.jit(ex_cp_ep.fused)(x2, w2, i2, gu2, dn2)
    seq_ag_cp_ep = [
        ln
        for ln in hlo_cp_ep.splitlines()
        if "=" in ln and "all-gather(" in ln.split("=", 1)[1] and "dimensions={1}" in ln
    ]
    self.assertEmpty(seq_ag_cp_ep, f"Expected 0 sequence all-gathers under CPasEP, got: {seq_ag_cp_ep}")
    self.assertEqual(tuple(out_cp_ep.sharding.spec)[:2], ("expert", "context"))

  def test_embed_router_preserve_unsharded_and_qwen3_8_ep_as_cp_batch_axes(self):
    from maxtext.m3.models.qwen3_8_flash_next import modeling_qwen3_8_flash_next as q38  # pylint: disable=import-outside-toplevel
    from maxtext.utils import sharding as max_sharding  # pylint: disable=import-outside-toplevel

    # 1. param_metadata sets preserve_unsharded=True for "embed"-axis router names
    # under FSDPasDPForAttn(shard_router_on_fsdp=False).
    s_no_rf = FSDPasDPForAttn(shard_router_on_fsdp=False)
    for r_name in ("router", "router_weight", "gate_router", "moe_router"):
      meta = param_metadata(s_no_rf, r_name, ("embed", None))
      self.assertEqual(meta["out_sharding"], P(None, None))
      self.assertTrue(meta.get("preserve_unsharded"))

    mesh_fsdp4 = create_mesh(ici={"fsdp": 4, "expert": 1}, devices=self.devices)

    class _EmbedRouterMod(nnx.Module):

      def __init__(self, shd, rngs):
        self.router = nnx.Param(
            jax.random.normal(rngs.params(), (8, 4), dtype=jnp.float32),
            **param_metadata(shd, "router", ("embed", None)),
        )
        self.non_router = nnx.Param(
            jax.random.normal(rngs.params(), (8, 4), dtype=jnp.float32),
            **param_metadata(DefaultSharding(), "other_weight", (None, None)),
        )

    with jax.set_mesh(mesh_fsdp4):
      abs_mod = nnx.eval_shape(lambda: _EmbedRouterMod(s_no_rf, nnx.Rngs(0)))
    _, abs_var_state = nnx.split(abs_mod)
    named_state = max_sharding.nnx_construct_named_sharding(abs_var_state, mesh_fsdp4)
    sharded_state = max_sharding.apply_m3_default_named_sharding(abs_var_state, named_state, mesh_fsdp4)
    r_shd = sharded_state["router"].get_value()
    nr_shd = sharded_state["non_router"].get_value()
    self.assertEqual(r_shd.spec, P(None, None))
    self.assertEqual(nr_shd.spec, P(None, "fsdp"))

    # 2. Qwen3_8 get_batch_mesh_axes / shard_batch / Qwen3_8FlashNextModel exclude "expert" from batch under EPasCP.
    mesh_ep_cp = create_mesh(ici={"fsdp": 2, "expert": 2}, devices=self.devices)
    with jax.set_mesh(mesh_ep_cp):
      self.assertEqual(q38.get_batch_mesh_axes(EPasCP()), ("fsdp",))
      self.assertEqual(q38.get_batch_mesh_axes(DefaultSharding()), ("fsdp", "expert"))
      x_tok = jax.device_put(
          jnp.ones((2, 8, 16), dtype=jnp.float32),
          NamedSharding(mesh_ep_cp, P("fsdp", "expert", None)),
      )
      y_ep_cp = jax.jit(lambda t: q38.shard_batch(t, sharding=EPasCP()))(x_tok)
      self.assertEqual(tuple(y_ep_cp.sharding.spec)[:2], ("fsdp", "expert"))

      mini_cfg = SimpleNamespace(
          emb_dim=64,
          num_decoder_layers=4,
          vocab_size=128,
          eos_token_id=127,
          ple_eos_token_id=127,
          hc_count=4,
          hc_lowrank=16,
          scan_layers=False,
          dtype=jnp.float32,
          weight_dtype=jnp.float32,
          normalization_layer_epsilon=1e-6,
          inhomogeneous_layer_cycle_interval=4,
          gdn_num_key_heads=2,
          gdn_num_value_heads=4,
          gdn_key_head_dim=16,
          gdn_value_head_dim=16,
          gdn_conv_kernel_dim=4,
          gdn_chunk_size=16,
          num_query_heads=4,
          num_kv_heads=1,
          head_dim=16,
          partial_rotary_factor=0.25,
          rope_max_timescale=1e4,
          mrope_section=(1, 1, 0),
          index_n_heads=2,
          index_kv_heads=1,
          index_head_dim=16,
          index_select_budget=8,
          index_select_block_size=4,
          moe_mlp_dim=32,
          mlp_dim=32,
          num_experts=4,
          num_experts_per_tok=2,
          norm_topk_prob=True,
          routed_scaling_factor=1.0,
          use_ple_layer=False,
          heads_per_ngram=2,
          ple_seed=0,
          ple_kernel_size=3,
          split_ngram_parts=2,
          ple_hash_seed=0,
          ngram_vocab_size_base=500,
          mtp_num_layers=0,
          mtp_num_streams=4,
          mtp_loss_scaling_factor=0.1,
          load_balance_loss_weight=0.001,
          gdn_output_gate_activation="silu",
          gdn_chunk_inverse="block_doubling",
          qsa_indexer_mask="gather",
          qsa_attention_kernel="dense",
          ple_layer_ids=[],
          sharding_preset="ep_as_cp",
          sparse_matmul=True,
          use_tokamax_gmm=True,
          use_gmm_v2=True,
      )
      model_ep_cp = q38.Qwen3_8FlashNextModel(mini_cfg, mesh=mesh_ep_cp, sharding=EPasCP(), rngs=nnx.Rngs(params=0))
      tok_ep_cp = jax.device_put(
          jnp.ones((2, 16), dtype=jnp.int32),
          NamedSharding(mesh_ep_cp, P("fsdp", "expert")),
      )
      logits_ep_cp = jax.jit(lambda m, t: m(t))(model_ep_cp, tok_ep_cp)
      self.assertEqual(logits_ep_cp.shape, (2, 16, 128))
      self.assertEqual(tuple(logits_ep_cp.sharding.spec)[:2], ("fsdp", "expert"))


if __name__ == "__main__":
  absltest.main()
