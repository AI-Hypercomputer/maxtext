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

"""Comprehensive unit and behavioral verification tests for `maxtext.configs.model_reducer`."""

import copy
import tempfile
import os
import unittest
import jax
import jax.numpy as jnp
import omegaconf
import pydantic

from maxtext.configs import model_reducer
from maxtext.configs import pyconfig

_BASE_CONFIG_PATH = "src/maxtext/configs/base.yml"


class ModelReducerTest(unittest.TestCase):
  """Verifies all 10 behavioral, mathematical, and architectural contracts of `model_reducer`."""

  def test_1_gemma3_27b_partial_tail_preserves_true_6_layer_cycle(self):
    """Gemma3-27B (62 layers = 10x6 + 2 tail) must preserve true 6-layer cycle, not gcd(62, 6)=2."""
    cfg = pyconfig.initialize_pydantic(
        ["", _BASE_CONFIG_PATH, "model_name=gemma3-27b", "override_model_config=True", "skip_jax_distributed_system=True"]
    )
    spec = model_reducer.get_architecture_spec(cfg)
    self.assertEqual(spec.arch_cycle_length, 6)
    self.assertEqual(spec.full_cycles, 10)
    self.assertEqual(spec.tail_layers, 2)

    # Minimum structural slice must retain 1 full 6-layer cycle (5 local + 1 global), never a 2-layer local-only prefix
    min_struct = model_reducer.compute_minimal_representative_model(cfg, "structural")
    self.assertEqual(min_struct["min_layers"], 6)

    # Direct cycle selection `target_num_cycles=2` gives 12 layers (2 full 6-layer cycles)
    cfg_2cycles = copy.deepcopy(cfg)
    cfg_2cycles.target_num_cycles = 2
    cfg_2cycles.model_reduce_strategy = "structural"
    plan = model_reducer.plan_model_reduction(cfg_2cycles)
    self.assertIsNotNone(plan)
    self.assertEqual(plan.planned_fields["base_num_decoder_layers"], 12)
    self.assertIn("Omitted partial tail layers", " ".join(plan.relaxed_invariants))

  def test_2_gemma4_small_parameter_counting_includes_full_ple_and_double_wide_mlp(self):
    """Gemma4-E2B PLE tensor must count `vocab_ple * num_layers * hidden_ple` (~2.35B), not `vocab_ple * hidden_ple`."""
    cfg = pyconfig.initialize_pydantic(
        [
            "",
            _BASE_CONFIG_PATH,
            "model_name=gemma4-e2b",
            "override_model_config=True",
            "skip_jax_distributed_system=True",
            "scan_layers=False",
        ]
    )
    stats = model_reducer.estimate_model_parameters(cfg)
    expected_token_emb = 262144 * 1536
    expected_ple_emb = (262144 + 3 * 1536) * 35 * 256
    self.assertEqual(stats["embedding_params"], expected_token_emb + expected_ple_emb)
    self.assertGreater(stats["embedding_params"], 2_700_000_000)
    # Also verify against jax.eval_shape abstract parameter count on a scaled Gemma4-Small config
    small_gemma4 = pyconfig.initialize([
        "", _BASE_CONFIG_PATH, "model_name=gemma4-e2b", "override_model_config=True",
        "skip_jax_distributed_system=True", "hardware=cpu", "scan_layers=False",
        "base_emb_dim=256", "base_mlp_dim=512", "base_num_query_heads=4", "base_num_kv_heads=1",
        "base_num_decoder_layers=10", "num_kv_shared_layers=5", "vocab_size=1024",
        "vocab_size_per_layer_input=1024", "hidden_size_per_layer_input=64",
    ])
    est_p = model_reducer.estimate_model_parameters(small_gemma4)["total_params"]
    exact_p = model_reducer.count_abstract_model_parameters(small_gemma4)
    self.assertAlmostEqual(est_p / exact_p, 1.0, delta=0.01)

  def test_3_metadata_synchronization_idempotence(self):
    """Applying reduction or re-deriving dimensions multiple times must not repeatedly rescale `num_kv_shared_layers`."""
    orig_cfg = pyconfig.initialize_pydantic(
        [
            "",
            _BASE_CONFIG_PATH,
            "model_name=gemma4-e2b",
            "override_model_config=True",
            "skip_jax_distributed_system=True",
            "scan_layers=False",
        ]
    )
    self.assertEqual(orig_cfg.num_kv_shared_layers, 20)

    # Reduce 35 layers -> 20 layers (4 cycles of 5)
    req_cfg = copy.deepcopy(orig_cfg)
    req_cfg.target_num_cycles = 4
    req_cfg.model_reduce_strategy = "structural"
    plan = model_reducer.plan_model_reduction(req_cfg)
    self.assertEqual(plan.planned_fields["base_num_decoder_layers"], 20)
    self.assertEqual(plan.remapped_metadata["num_kv_shared_layers"], 10)

    # Apply plan once, then call derive_dimensions and apply_reduction_plan repeatedly
    model_reducer.apply_reduction_plan(req_cfg, plan)
    self.assertEqual(req_cfg.num_kv_shared_layers, 10)
    for _ in range(4):
      model_reducer.derive_dimensions(req_cfg)
      model_reducer.apply_reduction_plan(req_cfg, plan)
      self.assertEqual(req_cfg.num_kv_shared_layers, 10)

  def test_4_qwen3_vl_decoder_reduction_preserves_vision_extraction_indices(self):
    """`deepstack_visual_indexes_for_vit` indexes into `num_hidden_layers_for_vit` (24) and must remain [5, 11, 17]."""
    qwen_vl_cfg = pyconfig.initialize(
        [
            "",
            _BASE_CONFIG_PATH,
            "model_name=qwen3-vl-2b",
            "override_model_config=True",
            "hardware=cpu",
            "scan_layers=False",
            "model_reduce_factor=2.0",
        ]
    )
    self.assertLess(qwen_vl_cfg.num_decoder_layers, 28)
    self.assertEqual(qwen_vl_cfg.num_hidden_layers_for_vit, 24)
    self.assertEqual(list(qwen_vl_cfg.deepstack_visual_indexes_for_vit), [5, 11, 17])

  def test_5_export_reload_round_trip_contract(self):
    """`architecture(reload(export(C))) == architecture(C)` including variant identity and overrides."""
    reduced_cfg = pyconfig.initialize_pydantic(
        [
            "",
            _BASE_CONFIG_PATH,
            "model_name=gemma4-e2b",
            "override_model_config=True",
            "skip_jax_distributed_system=True",
            "scan_layers=False",
            "target_num_cycles=4",
            "model_reduce_strategy=structural",
        ]
    )
    with tempfile.TemporaryDirectory() as tmpdir:
      out_yml = os.path.join(tmpdir, "gemma4-e2b-sliced.yml")
      model_reducer.save_reduced_model_yaml("gemma4-e2b", reduced_cfg, out_yml)
      reloaded_cfg = pyconfig.initialize_pydantic(["", out_yml, "skip_jax_distributed_system=True"])

      self.assertEqual(reloaded_cfg.model_name, "gemma4-e2b")
      self.assertEqual(reloaded_cfg.num_decoder_layers, reduced_cfg.num_decoder_layers)
      self.assertEqual(reloaded_cfg.num_kv_shared_layers, reduced_cfg.num_kv_shared_layers)
      self.assertEqual(reloaded_cfg.scan_layers, False)
      self.assertEqual(
          model_reducer.estimate_model_parameters(reloaded_cfg),
          model_reducer.estimate_model_parameters(reduced_cfg),
      )

  def test_6_strategy_contract_and_experts_agreement(self):
    """Each strategy modifies only its permitted fields, and `experts` never touches MLP dimensions."""
    orig_cfg = pyconfig.initialize_pydantic(
        ["", _BASE_CONFIG_PATH, "model_name=kimi-k2-1t", "override_model_config=True", "skip_jax_distributed_system=True"]
    )
    for strat in ("structural", "depth", "experts", "width", "compact", "budget"):
      with self.subTest(strategy=strat):
        allowed = model_reducer.STRATEGY_MUTABLE_FIELDS[strat]
        min_info = model_reducer.compute_minimal_representative_model(orig_cfg, strat)
        min_cfg = min_info["min_config"]

        # Check minimum config obeys allowed fields contract
        for candidate_attr in (
            "base_num_decoder_layers",
            "num_experts",
            "num_experts_per_tok",
            "base_moe_mlp_dim",
            "base_mlp_dim",
            "base_emb_dim",
        ):
          if candidate_attr not in allowed:
            self.assertEqual(
                getattr(min_cfg, candidate_attr),
                getattr(orig_cfg, candidate_attr),
                msg=f"Strategy '{strat}' minimum modified forbidden field '{candidate_attr}'",
            )

        # Check planner obeys allowed fields contract
        probe = copy.deepcopy(orig_cfg)
        probe.model_reduce_factor = 2.0
        probe.model_reduce_strategy = strat
        plan = model_reducer.plan_model_reduction(probe)
        self.assertTrue(set(plan.planned_fields.keys()).issubset(allowed))
        if strat == "experts":
          self.assertEqual(probe.base_moe_mlp_dim, orig_cfg.base_moe_mlp_dim)
          self.assertNotIn("base_moe_mlp_dim", plan.planned_fields)

  def test_7_aligned_interval_rounding_and_global_parameter_scale(self):
    """`_round_to_multiple` strictly stays within aligned multiples and `global_parameter_scale` is reflected."""
    # Reproduction from review: _round_to_multiple(21, 8, min_val=1, max_val=22) must return 16, not 22
    val = model_reducer._round_to_multiple(21, 8, min_val=1, max_val=22)
    self.assertEqual(val, 16)
    self.assertEqual(val % 8, 0)

    # Empty aligned interval [17, 22] for multiple=8 must raise ValueError
    with self.assertRaises(ValueError):
      model_reducer._round_to_multiple(20, 8, min_val=17, max_val=22)

    # global_parameter_scale != 1 must scale estimate_model_parameters
    cfg_scale1 = pyconfig.initialize_pydantic(
        ["", _BASE_CONFIG_PATH, "global_parameter_scale=1", "skip_jax_distributed_system=True"]
    )
    cfg_scale2 = pyconfig.initialize_pydantic(
        ["", _BASE_CONFIG_PATH, "global_parameter_scale=2", "skip_jax_distributed_system=True"]
    )
    self.assertEqual(
        model_reducer.estimate_model_parameters(cfg_scale2)["dense_ffn_params"],
        model_reducer.estimate_model_parameters(cfg_scale1)["dense_ffn_params"] * 2,
    )

  def test_8_moe_behavioral_coverage_never_collapses_to_top1(self):
    """Multi-expert MoE (`num_experts_per_tok=8`) never collapses to top-1 in any strategy."""
    orig_cfg = pyconfig.initialize_pydantic(
        ["", _BASE_CONFIG_PATH, "model_name=kimi-k2-1t", "override_model_config=True", "skip_jax_distributed_system=True"]
    )
    for strat in ("structural", "compact", "experts", "width"):
      min_info = model_reducer.compute_minimal_representative_model(orig_cfg, strat)
      self.assertEqual(min_info["min_config"].num_experts_per_tok, 8)

    budget_min = model_reducer.compute_minimal_representative_model(orig_cfg, "budget")
    self.assertGreaterEqual(budget_min["min_config"].num_experts_per_tok, 2)

  def test_9_construction_and_execution_forward_backward(self):
    """Constructs a reduced model with `jax.eval_shape` and runs an actual forward + backward step."""
    cfg = pyconfig.initialize(
        [
            "",
            _BASE_CONFIG_PATH,
            "base_emb_dim=256",
            "base_mlp_dim=512",
            "base_num_query_heads=4",
            "base_num_kv_heads=4",
            "head_dim=64",
            "base_num_decoder_layers=4",
            "vocab_size=1024",
            "per_device_batch_size=1",
            "max_target_length=16",
            "model_reduce_factor=1.5",
            "model_reduce_strategy=structural",
            "hardware=cpu",
            "skip_jax_distributed_system=True",
            "attention=dot_product",
        ]
    )
    self.assertEqual(cfg.num_decoder_layers, 2)
    abstract_count = model_reducer.count_abstract_model_parameters(cfg)
    est_count = model_reducer.estimate_model_parameters(cfg)["total_params"]
    # Abstract parameter tree count and analytical estimate agree within <2% (LayerNorm vectors account for <0.5%)
    self.assertAlmostEqual(abstract_count / est_count, 1.0, delta=0.02)

    from maxtext.common.common_types import MODEL_MODE_TRAIN
    from maxtext.layers import quantizations
    from maxtext.models import models
    from maxtext.utils import maxtext_utils

    devices_array = maxtext_utils.create_device_mesh(cfg)
    mesh = jax.sharding.Mesh(devices_array, cfg.mesh_axes)
    model = models.transformer_as_linen(cfg, mesh, quant=quantizations.configure_quantization(cfg), model_mode=MODEL_MODE_TRAIN)

    rng = jax.random.PRNGKey(0)
    tokens = jnp.ones((1, 16), dtype=jnp.int32)
    positions = jnp.broadcast_to(jnp.arange(16, dtype=jnp.int32), (1, 16))
    segment_ids = jnp.ones((1, 16), dtype=jnp.int32)

    variables = model.init({"params": rng, "dropout": rng, "aqt": rng}, tokens, positions, decoder_segment_ids=segment_ids, enable_dropout=False)

    def loss_fn(params):
      logits = model.apply({"params": params}, tokens, positions, decoder_segment_ids=segment_ids, enable_dropout=False)
      return jnp.mean(logits**2)

    loss, grads = jax.value_and_grad(loss_fn)(variables["params"])
    self.assertTrue(jnp.isfinite(loss))
    grad_norms = jax.tree_util.tree_leaves(jax.tree_util.tree_map(lambda g: jnp.linalg.norm(g), grads))
    self.assertTrue(all(bool(jnp.isfinite(gn)) for gn in grad_norms))

  def test_10_failure_atomicity_and_hard_budget_sweep(self):
    """Failed planning leaves source config untouched; `hard_budget=True` never exceeds target across factor sweep."""
    orig_cfg = pyconfig.initialize_pydantic(
        ["", _BASE_CONFIG_PATH, "model_name=kimi-k2-1t", "override_model_config=True", "skip_jax_distributed_system=True"]
    )
    snapshot_before = copy.deepcopy(orig_cfg.__dict__)

    # Trigger an infeasible reduction (100x under structural strategy)
    bad_req = copy.deepcopy(orig_cfg)
    bad_req.model_reduce_factor = 100.0
    bad_req.model_reduce_strategy = "structural"
    snapshot_bad_before = copy.deepcopy(bad_req.__dict__)
    with self.assertRaisesRegex(ValueError, "minimum supported model size"):
      model_reducer.apply_model_reduce_factor(bad_req)
    # Source config must remain completely unmutated on failure
    self.assertEqual(bad_req.base_num_decoder_layers, snapshot_bad_before["base_num_decoder_layers"])
    self.assertEqual(bad_req.num_experts, snapshot_bad_before["num_experts"])
    self.assertEqual(orig_cfg.__dict__, snapshot_before)

    # Sweep non-2.0 reduction factors with hard_budget=True
    orig_total = model_reducer.estimate_model_parameters(orig_cfg)["total_params"]
    for factor in (1.35, 1.75, 2.0, 2.6, 3.3):
      with self.subTest(factor=factor):
        probe = copy.deepcopy(orig_cfg)
        probe.model_reduce_factor = factor
        probe.hard_budget = True
        probe.model_reduce_strategy = "compact"
        plan = model_reducer.plan_model_reduction(probe)
        self.assertLessEqual(plan.final_stats["total_params"], int(orig_total / factor))


  # ----------------------------------------------------------------------------------------------
  # Regression tests for the second review: request-combination handling, metadata-aware scoring,
  # grouped-routing floors, scale coordinate system, and export/inheritance correctness.
  # ----------------------------------------------------------------------------------------------

  def _tiny_dense_config(self, **overrides):
    """Builds the small dense fixture used by several request-combination tests (4 layers, cycle 1)."""
    args = [
        "",
        _BASE_CONFIG_PATH,
        "base_emb_dim=256",
        "base_mlp_dim=512",
        "base_num_query_heads=4",
        "base_num_kv_heads=4",
        "head_dim=64",
        "base_num_decoder_layers=4",
        "vocab_size=1024",
        "skip_jax_distributed_system=True",
    ]
    cfg = pyconfig.initialize_pydantic(args)
    for key, val in overrides.items():
      setattr(cfg, key, val)
    return cfg

  def test_11_width_strategy_rejects_target_num_cycles(self):
    """`width` cannot change depth, so combining it with `target_num_cycles` must be rejected up front.

    Previously the cycle branch set `base_num_decoder_layers` on the candidate unconditionally while
    `width` excluded that field from `planned_fields`, producing a plan that reported 2 layers but
    applied 4.
    """
    cfg = self._tiny_dense_config(target_num_cycles=2, model_reduce_strategy="width")
    with self.assertRaisesRegex(ValueError, "forbidden from changing depth"):
      model_reducer.plan_model_reduction(cfg)

    # The rejected request must leave the source configuration untouched.
    self.assertEqual(cfg.base_num_decoder_layers, 4)
    self.assertEqual(cfg.num_decoder_layers, 4)

    # A depth-capable strategy accepts the same request and stays self-consistent.
    ok_cfg = self._tiny_dense_config(target_num_cycles=2, model_reduce_strategy="structural")
    plan = model_reducer.plan_model_reduction(ok_cfg)
    self.assertEqual(plan.planned_fields["base_num_decoder_layers"], 2)
    self.assertEqual(plan.retained_resolved_layers, 2)

  def test_12_cycle_selection_preserves_explicit_hard_budget(self):
    """A budget supplied alongside `target_num_cycles` must be preserved and enforced, never overwritten."""
    spec = model_reducer.get_architecture_spec(self._tiny_dense_config())
    two_cycle_cfg = model_reducer.materialize_candidate(self._tiny_dense_config(), spec=spec, resolved_layers=2)
    two_cycle_params = model_reducer.estimate_model_parameters(two_cycle_cfg)["total_params"]

    # An unsatisfiable budget must FAIL rather than be silently redefined as the slice's own size.
    infeasible = self._tiny_dense_config(
        target_num_cycles=2,
        target_model_size="1M",
        hard_budget=True,
        model_reduce_strategy="structural",
    )
    self.assertLess(1_000_000, two_cycle_params)  # the fixture really does overshoot a 1M budget
    with self.assertRaises(ValueError) as caught:
      model_reducer.plan_model_reduction(infeasible)
    # The decisive check: the rejection must be measured against the SUPPLIED 1M budget. Previously
    # the cycle branch overwrote `target_params` with the slice's own size, so this request
    # succeeded and silently reported a budget of ~1.84M.
    message = str(caught.exception)
    self.assertIn("1.00M", message)
    self.assertNotIn(model_reducer._format_params(two_cycle_params), message.split("has")[0])

    # A satisfiable budget is retained verbatim on the plan.
    feasible_budget = int(two_cycle_params * 1.5)
    feasible = self._tiny_dense_config(
        target_num_cycles=2,
        target_model_size=str(feasible_budget),
        hard_budget=True,
        model_reduce_strategy="structural",
    )
    plan = model_reducer.plan_model_reduction(feasible)
    self.assertEqual(plan.target_params, feasible_budget)
    self.assertEqual(plan.budget_source, "target_model_size")
    self.assertTrue(plan.explicit_budget)
    self.assertLessEqual(plan.final_stats["total_params"], feasible_budget)

    # A cycles-only request is labelled as such, and derives its target from the slice.
    cycles_only = self._tiny_dense_config(target_num_cycles=2, model_reduce_strategy="structural")
    cycles_plan = model_reducer.plan_model_reduction(cycles_only)
    self.assertEqual(cycles_plan.budget_source, "target_num_cycles")
    self.assertFalse(cycles_plan.explicit_budget)

    # `hard_budget` without any budget is a contradiction.
    with self.assertRaisesRegex(ValueError, "no budget was supplied"):
      model_reducer.plan_model_reduction(
          self._tiny_dense_config(target_num_cycles=2, hard_budget=True, model_reduce_strategy="structural")
      )

  def test_13_applied_plan_statistics_equal_plan_statistics(self):
    """Applying a plan must reproduce exactly the architecture and statistics the plan reports."""
    scenarios = (
        ("kimi-k2-1t", {"model_reduce_factor": 2.0, "model_reduce_strategy": "compact"}),
        ("kimi-k2-1t", {"model_reduce_factor": 2.0, "model_reduce_strategy": "structural"}),
        ("kimi-k2-1t", {"model_reduce_factor": 3.0, "model_reduce_strategy": "budget", "hard_budget": True}),
        ("gemma3-27b", {"target_num_cycles": 4, "model_reduce_strategy": "structural"}),
        ("gemma3-27b", {"model_reduce_factor": 2.0, "model_reduce_strategy": "width"}),
    )
    for model_name, request in scenarios:
      with self.subTest(model=model_name, **request):
        base_cfg = pyconfig.initialize_pydantic(
            ["", _BASE_CONFIG_PATH, f"model_name={model_name}", "override_model_config=True",
             "skip_jax_distributed_system=True"]
        )
        req_cfg = copy.deepcopy(base_cfg)
        for key, val in request.items():
          setattr(req_cfg, key, val)

        plan = model_reducer.plan_model_reduction(req_cfg)
        self.assertIsNotNone(plan)

        applied = copy.deepcopy(base_cfg)
        model_reducer.apply_reduction_plan(applied, plan)

        self.assertEqual(
            model_reducer.estimate_model_parameters(applied)["total_params"],
            plan.final_stats["total_params"],
        )
        self.assertEqual(int(applied.num_decoder_layers), plan.retained_resolved_layers)
        # The manifest's own cycle arithmetic must reconstruct the applied depth.
        self.assertEqual(
            plan.arch_spec.prefix_layers
            + plan.retained_cycles * plan.arch_spec.arch_cycle_length
            + plan.retained_tail_layers,
            int(applied.num_decoder_layers),
        )
        # Reported sizes are analytical until explicitly verified.
        self.assertEqual(plan.verification_mode, "estimated")
        self.assertIn("ESTIMATED ONLY", plan.format_fidelity_report())

  def test_14_gemma4_structural_search_finds_feasible_metadata_aware_candidate(self):
    """Structural search must score probes AFTER metadata remapping, or it rejects feasible slices.

    Gemma4-E2B's `num_kv_shared_layers` changes both attention and double-wide MLP parameter counts.
    Scoring a reduced-depth probe with the 35-layer model's metadata over-counts it, which
    previously excluded the exactly-fitting 20-layer slice and then rejected the smaller fallback
    for being too far below target.
    """
    cfg = pyconfig.initialize_pydantic(
        ["", _BASE_CONFIG_PATH, "model_name=gemma4-e2b", "override_model_config=True",
         "skip_jax_distributed_system=True", "scan_layers=False"]
    )
    spec = model_reducer.get_architecture_spec(cfg)
    self.assertEqual(spec.arch_cycle_length, 5)

    # The correctly-scored 20-layer (4-cycle) slice.
    correct = model_reducer.materialize_candidate(cfg, spec=spec, resolved_layers=20)
    self.assertEqual(correct.num_kv_shared_layers, 10)
    correct_params = model_reducer.estimate_model_parameters(correct)["total_params"]

    # The same depth scored with the SOURCE model's stale metadata (the old behavior).
    stale = copy.deepcopy(cfg)
    stale.base_num_decoder_layers = 20
    model_reducer.derive_dimensions(stale)
    self.assertEqual(stale.num_kv_shared_layers, 20)  # stale: unchanged from the 35-layer model
    stale_params = model_reducer.estimate_model_parameters(stale)["total_params"]
    self.assertGreater(stale_params, correct_params)

    # With a hard budget set to exactly the correct 20-layer size, the search must find 20 layers.
    req = copy.deepcopy(cfg)
    req.target_model_size = str(correct_params)
    req.hard_budget = True
    req.model_reduce_strategy = "structural"
    plan = model_reducer.plan_model_reduction(req)
    self.assertEqual(plan.retained_resolved_layers, 20)
    self.assertEqual(plan.retained_cycles, 4)
    self.assertEqual(plan.final_stats["total_params"], correct_params)
    self.assertEqual(plan.remapped_metadata["num_kv_shared_layers"], 10)

  def test_15_deepseek_grouped_routing_minimum_respects_group_capacity(self):
    """Grouped routers need `E % G == 0`, `E/G >= 2`, and `K <= G_selected * (E/G)`, not just `E >= K`."""
    cfg = pyconfig.initialize_pydantic(
        ["", _BASE_CONFIG_PATH, "model_name=deepseek3-671b", "override_model_config=True",
         "skip_jax_distributed_system=True"]
    )
    cfg.num_experts = 256
    cfg.num_experts_per_tok = 8
    cfg.n_routing_groups = 8
    cfg.topk_routing_group = 4

    floors = model_reducer._get_width_and_expert_floors(cfg, strategy="experts")
    min_experts = floors["min_experts"]
    # 8 experts would leave 1 expert per group, which a top-2-per-group router cannot score.
    self.assertGreaterEqual(min_experts, 16)
    self.assertEqual(min_experts % 8, 0)
    self.assertGreaterEqual(min_experts // 8, 2)
    self.assertLessEqual(floors["min_experts_per_tok"], 4 * (min_experts // 8))

    # The minimum model actually produced must satisfy the same constraints...
    min_info = model_reducer.compute_minimal_representative_model(cfg, "experts")
    min_cfg = min_info["min_config"]
    self.assertEqual(min_cfg.num_experts % min_cfg.n_routing_groups, 0)
    self.assertGreaterEqual(min_cfg.num_experts // min_cfg.n_routing_groups, 2)
    self.assertLessEqual(
        min_cfg.num_experts_per_tok,
        min_cfg.topk_routing_group * (min_cfg.num_experts // min_cfg.n_routing_groups),
    )

    # ...and an unroutable grouped configuration must be rejected outright, not silently produced.
    spec = model_reducer.get_architecture_spec(cfg)
    bad = copy.deepcopy(cfg)
    bad.num_experts = 8  # 1 expert per group
    with self.assertRaisesRegex(ValueError, "expert\\(s\\) per group"):
      model_reducer._validate_candidate_constraints(bad, spec)

    unreachable = copy.deepcopy(cfg)
    unreachable.num_experts = 16
    unreachable.topk_routing_group = 1  # only 2 experts reachable, but top-8 requested
    with self.assertRaisesRegex(ValueError, "reachable experts"):
      model_reducer._validate_candidate_constraints(unreachable, spec)

  def test_16_global_parameter_scale_depth_agrees_with_manifest(self):
    """At `global_parameter_scale > 1`, depth choices must be representable in base units.

    `get_individual_scales` multiplies `base_num_decoder_layers` by `2**layer_scale`. Writing a
    resolved layout number straight into `base_num_decoder_layers` made the manifest report a depth
    the configuration never resolved to (scale 8, 1 cycle -> manifest 1 layer, actual 2).
    """
    for scale, multiplier in ((8, 2), (64, 4)):
      with self.subTest(scale=scale):
        cfg = pyconfig.initialize_pydantic(
            ["", _BASE_CONFIG_PATH, f"global_parameter_scale={scale}", "skip_jax_distributed_system=True"]
        )
        spec = model_reducer.get_architecture_spec(cfg)
        self.assertEqual(spec.layer_scale_multiplier, multiplier)

        # Every legal depth must be exactly representable in base units.
        for cycles, tail in model_reducer._legal_cycle_counts(spec):
          resolved = spec.prefix_layers + cycles * spec.arch_cycle_length + tail
          self.assertEqual(resolved % multiplier, 0, msg=f"depth {resolved} not representable")

        # A non-representable request is rejected rather than silently doubled.
        bad = copy.deepcopy(cfg)
        bad.target_num_cycles = 1
        bad.model_reduce_strategy = "structural"
        with self.assertRaisesRegex(ValueError, "not a legal slice"):
          model_reducer.plan_model_reduction(bad)

        # A representable request produces a manifest that matches the resolved depth.
        good = copy.deepcopy(cfg)
        good.target_num_cycles = multiplier
        good.model_reduce_strategy = "structural"
        plan = model_reducer.plan_model_reduction(good)
        applied = copy.deepcopy(cfg)
        model_reducer.apply_reduction_plan(applied, plan)
        self.assertEqual(int(applied.num_decoder_layers), multiplier)
        self.assertEqual(plan.retained_resolved_layers, multiplier)
        self.assertEqual(int(applied.base_num_decoder_layers), 1)

  def test_17_export_preserves_resolved_overrides_on_reload(self):
    """Export must re-write every key the original model YAML declares with the RESOLVED value.

    Seeding the export from the original model YAML and updating only a hand-maintained whitelist
    let untouched keys (e.g. `rope_max_timescale`, `use_multimodal`) revert to the original model's
    value and override the user's resolved value on reload.
    """
    reduced_cfg = pyconfig.initialize_pydantic(
        ["", _BASE_CONFIG_PATH, "model_name=qwen3-vl-2b", "override_model_config=True",
         "skip_jax_distributed_system=True", "scan_layers=False",
         "rope_max_timescale=1000000", "model_reduce_factor=2.0",
         "model_reduce_strategy=structural"]
    )
    self.assertEqual(reduced_cfg.rope_max_timescale, 1000000)

    model_yml = os.path.join(pyconfig.MAXTEXT_CONFIGS_DIR, "models", "qwen3-vl-2b.yml")
    declared_keys = set(omegaconf.OmegaConf.load(model_yml).keys()) - {"base_config"}

    with tempfile.TemporaryDirectory() as tmpdir:
      out_yml = os.path.join(tmpdir, "qwen3-vl-2b-sliced.yml")
      model_reducer.save_reduced_model_yaml("qwen3-vl-2b", reduced_cfg, out_yml)
      reloaded = pyconfig.initialize_pydantic(["", out_yml, "skip_jax_distributed_system=True"])

      self.assertEqual(reloaded.rope_max_timescale, 1000000)
      self.assertEqual(reloaded.num_decoder_layers, reduced_cfg.num_decoder_layers)
      # Reloading must NOT re-apply the reduction that produced this slice.
      self.assertEqual(reloaded.model_reduce_factor, 1.0)
      self.assertEqual(reloaded.target_num_cycles, -1)

      # Every key the model YAML declares must round-trip at its resolved value.
      for key in sorted(declared_keys):
        if key in ("model_reduce_factor", "target_model_size", "target_num_cycles",
                   "hard_budget", "model_reduce_strategy"):
          continue
        if not hasattr(reduced_cfg, key) or not hasattr(reloaded, key):
          continue
        with self.subTest(key=key):
          self.assertEqual(getattr(reloaded, key), getattr(reduced_cfg, key))

  def test_18_exported_slice_used_as_base_config_preserves_reduced_dimensions(self):
    """A job YAML that inherits from an exported slice must keep the slice's architecture.

    Protecting only keys physically present in the TOP-LEVEL YAML lost the slice as soon as the
    exported file was used via `base_config`, because the model YAML then re-applied the original
    architecture.
    """
    reduced_cfg = pyconfig.initialize_pydantic(
        ["", _BASE_CONFIG_PATH, "model_name=gemma4-e2b", "override_model_config=True",
         "skip_jax_distributed_system=True", "scan_layers=False",
         "target_num_cycles=4", "model_reduce_strategy=structural"]
    )
    self.assertEqual(reduced_cfg.num_decoder_layers, 20)

    with tempfile.TemporaryDirectory() as tmpdir:
      sliced_yml = os.path.join(tmpdir, "sliced.yml")
      model_reducer.save_reduced_model_yaml("gemma4-e2b", reduced_cfg, sliced_yml)

      # Direct reload (already covered by test 5, re-asserted here as the control).
      direct = pyconfig.initialize_pydantic(["", sliced_yml, "skip_jax_distributed_system=True"])
      self.assertEqual(direct.num_decoder_layers, 20)

      # Reload through a child job YAML that inherits from the slice.
      job_yml = os.path.join(tmpdir, "job.yml")
      with open(job_yml, "w", encoding="utf-8") as handle:
        handle.write(f"base_config: {sliced_yml}\nrun_name: test-job\n")

      child = pyconfig.initialize_pydantic(["", job_yml, "skip_jax_distributed_system=True"])
      self.assertEqual(child.run_name, "test-job")
      self.assertEqual(child.num_decoder_layers, 20)
      self.assertEqual(child.num_kv_shared_layers, reduced_cfg.num_kv_shared_layers)
      self.assertEqual(
          model_reducer.estimate_model_parameters(child)["total_params"],
          model_reducer.estimate_model_parameters(reduced_cfg)["total_params"],
      )

  def test_19_all_depth_decisions_stay_within_the_legal_depth_set(self):
    """Direct selection and the hard-budget step-down loop must both respect pipeline placement."""
    cfg = pyconfig.initialize_pydantic(
        ["", _BASE_CONFIG_PATH, "base_emb_dim=256", "base_mlp_dim=512", "base_num_query_heads=4",
         "base_num_kv_heads=4", "head_dim=64", "base_num_decoder_layers=8", "vocab_size=1024",
         "skip_jax_distributed_system=True"]
    )
    cfg.ici_pipeline_parallelism = 4
    cfg.num_layers_per_pipeline_stage = 1

    spec = model_reducer.get_architecture_spec(cfg)
    self.assertEqual(spec.pp_divisor, 4)
    legal_cycles = {cycles for cycles, _ in model_reducer._legal_cycle_counts(spec)}
    self.assertEqual(legal_cycles, {4, 8})

    # Direct selection of an illegal depth must be rejected, not silently accepted.
    illegal = copy.deepcopy(cfg)
    illegal.target_num_cycles = 1
    illegal.model_reduce_strategy = "structural"
    with self.assertRaisesRegex(ValueError, "not a legal slice"):
      model_reducer.plan_model_reduction(illegal)

    # The hard-budget step-down loop may only land on legal depths.
    tight = copy.deepcopy(cfg)
    tight.model_reduce_factor = 1.9
    tight.hard_budget = True
    tight.model_reduce_strategy = "compact"
    plan = model_reducer.plan_model_reduction(tight)
    self.assertIn(plan.retained_cycles, legal_cycles)
    self.assertEqual(plan.retained_resolved_layers % 4, 0)

    applied = copy.deepcopy(cfg)
    model_reducer.apply_reduction_plan(applied, plan)
    self.assertEqual(int(applied.num_decoder_layers) % 4, 0)

  def test_20_inferred_parallelism_degrees_are_treated_as_unset(self):
    """Raw parallelism fields may be `-1` ("infer"); they must never be multiplied as-is."""
    cfg = self._tiny_dense_config()
    cfg.ici_pipeline_parallelism = -1
    cfg.dcn_pipeline_parallelism = -1
    cfg.ici_tensor_parallelism = -1
    cfg.dcn_tensor_parallelism = -1
    cfg.ici_expert_parallelism = -1

    spec = model_reducer.get_architecture_spec(cfg)
    # Two `-1` axes must not multiply to a spurious positive degree.
    self.assertEqual(spec.pp_divisor, 1)

    floors = model_reducer._get_width_and_expert_floors(cfg, strategy="compact")
    self.assertGreaterEqual(floors["min_emb_dim"], 1)
    self.assertGreaterEqual(floors["min_kv_heads"], 1)
    self.assertLessEqual(floors["min_kv_heads"], cfg.base_num_kv_heads)
    self.assertLessEqual(floors["min_emb_dim"], cfg.base_emb_dim)

    # Planning must still succeed and produce a self-consistent plan.
    req = copy.deepcopy(cfg)
    req.target_num_cycles = 2
    req.model_reduce_strategy = "structural"
    plan = model_reducer.plan_model_reduction(req)
    self.assertEqual(plan.retained_resolved_layers, 2)

  def test_21_export_preserves_overrides_absent_from_the_original_model_yaml(self):
    """Export must carry architectural overrides the original model YAML never declares.

    The previous exporter wrote `keys(original model YAML) | hand_maintained_whitelist`. An override
    in NEITHER set vanished. `deepseek3-671b.yml` is the concrete case: it recommends
    `n_routing_groups` / `topk_routing_group` in COMMENTS only, so a grouped-routing slice exported
    without them silently reloaded with the defaults - a different router.

    Unlike test 17, this test deliberately overrides keys that are NOT declared by the model YAML,
    which is structurally the only way to catch this class of failure.
    """
    model_yml = os.path.join(pyconfig.MAXTEXT_CONFIGS_DIR, "models", "deepseek3-671b.yml")
    declared_keys = set(omegaconf.OmegaConf.load(model_yml).keys())

    # Overrides chosen to span routing, sharding and attention, and asserted to be undeclared.
    undeclared_overrides = {
        "n_routing_groups": 8,
        "topk_routing_group": 4,
        "ici_expert_parallelism": 4,
        "attention": "dot_product",
    }
    for key in undeclared_overrides:
      self.assertNotIn(
          key, declared_keys, f"Test premise broken: '{key}' is now declared by deepseek3-671b.yml."
      )

    args = ["", _BASE_CONFIG_PATH, "model_name=deepseek3-671b", "override_model_config=True",
            "skip_jax_distributed_system=True", "model_reduce_factor=2.0",
            "model_reduce_strategy=structural"]
    args += [f"{k}={v}" for k, v in undeclared_overrides.items()]
    reduced_cfg = pyconfig.initialize_pydantic(args)

    with tempfile.TemporaryDirectory() as tmpdir:
      out_yml = os.path.join(tmpdir, "deepseek3-671b-sliced.yml")
      model_reducer.save_reduced_model_yaml("deepseek3-671b", reduced_cfg, out_yml)

      exported = omegaconf.OmegaConf.load(out_yml)
      for key, value in undeclared_overrides.items():
        with self.subTest(key=key, stage="serialized"):
          self.assertIn(key, exported, f"'{key}' was dropped from the exported model spec.")
          self.assertEqual(exported[key], value)

      reloaded = pyconfig.initialize_pydantic(["", out_yml, "skip_jax_distributed_system=True"])
      for key, value in undeclared_overrides.items():
        with self.subTest(key=key, stage="reloaded"):
          self.assertEqual(getattr(reloaded, key), value)

      # The grouped router must survive as a whole, not just key-by-key.
      self.assertEqual(reloaded.num_experts, reduced_cfg.num_experts)
      self.assertEqual(reloaded.num_experts_per_tok, reduced_cfg.num_experts_per_tok)
      self.assertEqual(reloaded.num_decoder_layers, reduced_cfg.num_decoder_layers)
      self.assertEqual(
          model_reducer.estimate_model_parameters(reloaded)["total_params"],
          model_reducer.estimate_model_parameters(reduced_cfg)["total_params"],
      )

      # The export is a model SPECIFICATION, not a job dump: run/job state must not leak into it.
      for job_key in ("run_name", "base_output_directory", "hf_access_token", "dataset_path",
                      "load_parameters_path", "checkpoint_dir", "tokenizer_path"):
        with self.subTest(job_key=job_key):
          self.assertNotIn(job_key, exported, f"Job state '{job_key}' leaked into the model spec.")

  def test_22_exported_slice_is_location_independent(self):
    """The same exported file must describe the same model wherever it is stored.

    `pyconfig._is_packaged_default_config` classifies EVERY file directly under `configs/models/`
    as a MaxText-shipped default, so an export written there was skipped when collecting user
    intent and the ORIGINAL architecture was silently restored (35 layers instead of 20). The
    snapshot marker makes recognition independent of directory.
    """
    reduced_cfg = pyconfig.initialize_pydantic(
        ["", _BASE_CONFIG_PATH, "model_name=gemma4-e2b", "override_model_config=True",
         "skip_jax_distributed_system=True", "scan_layers=False",
         "target_num_cycles=4", "model_reduce_strategy=structural"]
    )
    expected_layers = reduced_cfg.num_decoder_layers
    self.assertEqual(expected_layers, 20)

    packaged_models_dir = os.path.join(pyconfig.MAXTEXT_CONFIGS_DIR, "models")
    inside_yml = os.path.join(packaged_models_dir, "gemma4-e2b-sliced-pytest.yml")

    with tempfile.TemporaryDirectory() as tmpdir:
      outside_yml = os.path.join(tmpdir, "gemma4-e2b-sliced.yml")
      try:
        model_reducer.save_reduced_model_yaml("gemma4-e2b", reduced_cfg, outside_yml)
        model_reducer.save_reduced_model_yaml("gemma4-e2b", reduced_cfg, inside_yml)

        # The marker must actually be written; it is what makes the two locations agree.
        self.assertTrue(omegaconf.OmegaConf.load(inside_yml).get("is_resolved_architecture_snapshot"))

        for label, slice_path in (("outside", outside_yml), ("inside", inside_yml)):
          with self.subTest(location=label, load="direct"):
            direct = pyconfig.initialize_pydantic(["", slice_path, "skip_jax_distributed_system=True"])
            self.assertEqual(direct.num_decoder_layers, expected_layers)
            self.assertEqual(direct.num_kv_shared_layers, reduced_cfg.num_kv_shared_layers)

          # A child job YAML inheriting the slice must also keep the sliced architecture.
          job_yml = os.path.join(tmpdir, f"job_{label}.yml")
          with open(job_yml, "w", encoding="utf-8") as handle:
            handle.write(f"base_config: {slice_path}\nrun_name: test-job-{label}\n")
          with self.subTest(location=label, load="inherited"):
            child = pyconfig.initialize_pydantic(["", job_yml, "skip_jax_distributed_system=True"])
            self.assertEqual(child.run_name, f"test-job-{label}")
            self.assertEqual(child.num_decoder_layers, expected_layers)
      finally:
        # Never leave an artifact inside the packaged config directory.
        if os.path.exists(inside_yml):
          os.remove(inside_yml)

  def test_23_qwen3_vl_minimum_slice_reports_deepstack_consumer_coverage(self):
    """DeepStack feature `i` is consumed at decoder layer `i`, so consumers are bounded by depth.

    `deepstack_visual_indexes_for_vit: [5, 11, 17]` yields three features. A one-layer slice
    consumes only the first; the other two are silently dropped. The requirement is a CONSUMER
    COUNT (3 decoder layers), not the largest extraction index (18).
    """
    cfg = pyconfig.initialize_pydantic(
        ["", _BASE_CONFIG_PATH, "model_name=qwen3-vl-2b", "override_model_config=True",
         "skip_jax_distributed_system=True", "scan_layers=False"]
    )
    num_features = len(cfg.deepstack_visual_indexes_for_vit)
    self.assertEqual(num_features, 3)

    spec = model_reducer.get_architecture_spec(cfg)
    self.assertEqual(spec.min_layers_for_modality_coverage, num_features)

    # The floor is the consumer count, and it must not be inflated to the extraction index.
    min_info = model_reducer.compute_minimal_representative_model(cfg, "structural")
    self.assertGreaterEqual(min_info["min_layers"], num_features)
    self.assertLess(min_info["min_layers"], max(cfg.deepstack_visual_indexes_for_vit))

    # Every feature retains a consumer at the minimum, and the report says so.
    req = copy.deepcopy(cfg)
    req.model_reduce_factor = min_info["max_reduce_factor"]
    req.model_reduce_strategy = "structural"
    plan = model_reducer.plan_model_reduction(req)
    self.assertGreaterEqual(plan.retained_resolved_layers, num_features)
    self.assertTrue(
        any("DeepStack" in item for item in plan.preserved_invariants),
        f"Coverage not reported as preserved: {plan.preserved_invariants}",
    )
    self.assertFalse(any("DeepStack" in item for item in plan.relaxed_invariants))

    # If a shallower slice is ever produced, the loss must be reported, never silent.
    shallow = copy.deepcopy(plan)
    shallow.retained_resolved_layers = 1
    shallow.relaxed_invariants = [
        f"DeepStack visual injection coverage: only 1 of {num_features} visual features are consumed"
    ]
    self.assertIn("DeepStack", shallow.format_fidelity_report())

  def test_24_head_floors_use_resolved_not_base_dimensions(self):
    """Divisibility must be checked against RESOLVED head counts, not raw `base_*` values.

    With `global_parameter_scale=2` the head multiplier is 2, so `base_num_kv_heads=4` resolves to
    8 heads and satisfies tensor parallelism of 8. Applying the resolved requirement directly to
    the base value demanded 4 -> 8 in the base domain and raised
    `Empty feasible aligned interval for multiple=8 within [min_val=4, max_val=4]`.
    """
    cfg = pyconfig.initialize_pydantic(
        ["", _BASE_CONFIG_PATH, "global_parameter_scale=2", "base_emb_dim=1024",
         "base_mlp_dim=2048", "base_num_query_heads=8", "base_num_kv_heads=4",
         "head_dim=128", "base_num_decoder_layers=8", "vocab_size=1024",
         "skip_jax_distributed_system=True"]
    )
    cfg.ici_tensor_parallelism = 8
    self.assertEqual(cfg.num_kv_heads % 8, 0, "Test premise: resolved KV heads already divide TP=8.")

    # A depth-only strategy must not fail because of a width floor it can never use.
    structural_floors = model_reducer._get_width_and_expert_floors(cfg, strategy="structural")
    self.assertEqual(structural_floors["min_kv_heads"], cfg.base_num_kv_heads)
    self.assertEqual(structural_floors["min_emb_dim"], cfg.base_emb_dim)

    # Width strategies may compute floors, but only ones the base lattice can actually represent.
    width_floors = model_reducer._get_width_and_expert_floors(cfg, strategy="compact")
    self.assertLessEqual(width_floors["min_kv_heads"], cfg.base_num_kv_heads)
    self.assertGreaterEqual(width_floors["min_kv_heads"], 1)

    # Planning must succeed rather than raise on the empty aligned interval.
    for strategy in ("structural", "compact"):
      with self.subTest(strategy=strategy):
        req = copy.deepcopy(cfg)
        req.model_reduce_factor = 2.0
        req.model_reduce_strategy = strategy
        plan = model_reducer.plan_model_reduction(req)
        applied = copy.deepcopy(cfg)
        model_reducer.apply_reduction_plan(applied, plan, log=False)
        # The APPLIED resolved head counts are what the mesh actually shards.
        self.assertEqual(int(applied.num_kv_heads) % 8, 0)
        self.assertEqual(int(applied.num_query_heads) % 8, 0)

  def test_25_manifest_distinguishes_deferred_from_verified_mesh_validation(self):
    """`-1` parallelism is a planning assumption, not proof of placement validity."""
    deferred_cfg = self._tiny_dense_config()
    deferred_cfg.ici_tensor_parallelism = -1
    deferred_cfg.target_num_cycles = 2
    deferred_cfg.model_reduce_strategy = "structural"

    deferred_plan = model_reducer.plan_model_reduction(deferred_cfg)
    self.assertEqual(deferred_plan.mesh_validation, "deferred")
    self.assertIn("ici_tensor_parallelism", deferred_plan.unresolved_parallelism_axes)
    report = deferred_plan.format_fidelity_report()
    self.assertIn("architecture_validation: passed", report)
    self.assertIn("mesh_validation: deferred", report)

    resolved_cfg = self._tiny_dense_config()
    for axis in model_reducer._SHARDING_RELEVANT_AXES:
      if hasattr(resolved_cfg, axis):
        setattr(resolved_cfg, axis, 1)
    resolved_cfg.target_num_cycles = 2
    resolved_cfg.model_reduce_strategy = "structural"

    resolved_plan = model_reducer.plan_model_reduction(resolved_cfg)
    self.assertEqual(resolved_plan.mesh_validation, "passed")
    self.assertEqual(resolved_plan.unresolved_parallelism_axes, [])
    self.assertIn("mesh_validation: passed", resolved_plan.format_fidelity_report())

  def test_26_applied_plan_is_bound_to_its_own_config(self):
    """A plan must be retrievable by config identity, never as a process-global 'last result'."""
    first = self._tiny_dense_config()
    first.target_num_cycles = 2
    first.model_reduce_strategy = "structural"
    model_reducer.apply_model_reduce_factor(first)
    first_plan = model_reducer.get_last_applied_plan(first)
    self.assertIsNotNone(first_plan)

    # A config that was never reduced must report no plan, even though one exists in this process.
    untouched = self._tiny_dense_config()
    self.assertIsNone(model_reducer.get_last_applied_plan(untouched))

    # A second reduction must not rewrite the first config's manifest.
    second = self._tiny_dense_config()
    second.target_num_cycles = 3
    second.model_reduce_strategy = "structural"
    model_reducer.apply_model_reduce_factor(second)
    second_plan = model_reducer.get_last_applied_plan(second)
    self.assertIsNotNone(second_plan)
    self.assertIsNot(second_plan, first_plan)
    self.assertIs(model_reducer.get_last_applied_plan(first), first_plan)
    self.assertNotEqual(second_plan.retained_resolved_layers, first_plan.retained_resolved_layers)

    # A request that produces no plan must CLEAR the association rather than leave a stale one.
    no_op = self._tiny_dense_config()
    model_reducer.apply_model_reduce_factor(no_op)
    self.assertIsNone(model_reducer.get_last_applied_plan(no_op))

  def test_27_width_reduction_never_violates_grouped_query_attention(self):
    """`num_query_heads % num_kv_heads == 0` is a hard invariant of every width choice.

    Regression: deriving the query lattice from `lcm(hardware_quantum, new_kv_heads)` and then
    falling back to `gcd(base_num_query_heads, lattice)` when the original was not a multiple of it
    produced pairs such as `num_query_heads=26 / num_kv_heads=6` (llama3-8b) and `53 / 7`
    (llama3-70b). The candidate validator rejected them, so 98 model x strategy combinations across
    llama2, llama3, gemma3 and cosmos3 turned into spurious "infeasible" results.

    The floors and the applicator are both checked, because either one can break the pair.
    """
    models = ["llama3-8b", "llama3-70b", "llama2-70b", "gemma3-4b", "gemma3-27b"]
    strategies = ["compact", "balanced", "width", "budget", "auto"]

    for model_name in models:
      cfg = pyconfig.initialize_pydantic(
          ["", _BASE_CONFIG_PATH, f"model_name={model_name}", "override_model_config=True",
           "skip_jax_distributed_system=True", "attention=dot_product"]
      )
      self.assertEqual(cfg.num_query_heads % cfg.num_kv_heads, 0, "Test premise: source model is GQA-valid.")

      for strategy in strategies:
        floors = model_reducer._get_width_and_expert_floors(cfg, strategy=strategy)
        with self.subTest(model=model_name, strategy=strategy, stage="floors"):
          self.assertEqual(
              floors["min_q_heads"] % floors["min_kv_heads"], 0,
              f"Floor pair violates GQA: q={floors['min_q_heads']} kv={floors['min_kv_heads']}",
          )

        req = copy.deepcopy(cfg)
        req.model_reduce_factor = 2.0
        req.model_reduce_strategy = strategy
        with self.subTest(model=model_name, strategy=strategy, stage="applied"):
          plan = model_reducer.plan_model_reduction(req)
          applied = copy.deepcopy(cfg)
          model_reducer.apply_reduction_plan(applied, plan, log=False)
          self.assertEqual(
              int(applied.num_query_heads) % int(applied.num_kv_heads), 0,
              f"Applied pair violates GQA: q={applied.num_query_heads} kv={applied.num_kv_heads}",
          )
          self.assertLessEqual(int(applied.num_kv_heads), int(cfg.num_kv_heads))
          self.assertLessEqual(int(applied.num_query_heads), int(cfg.num_query_heads))

  def test_28_head_pair_chooser_degrades_instead_of_emitting_invalid_pairs(self):
    """When no smaller GQA-valid pair exists, the heads must be left alone, not made invalid."""
    # A prime query-head count with a hardware lattice that cannot divide it: the only GQA-valid
    # option at or below the original is the original itself.
    floors = {"kv_quantum": 1, "min_kv_heads": 1, "q_quantum": 13}
    q, kv = model_reducer._choose_head_pair(13, 13, 2.0, floors)
    self.assertEqual(q % kv, 0)
    self.assertLessEqual(q, 13)
    self.assertLessEqual(kv, 13)

    # The regression case: 32 query / 8 kv must never yield a non-divisible pair at any scale.
    for scale in (1.1, 1.25, 1.41, 1.7, 2.0, 2.8, 4.0):
      with self.subTest(dim_scale=scale):
        q, kv = model_reducer._choose_head_pair(32, 8, scale, {"kv_quantum": 1, "min_kv_heads": 1, "q_quantum": 1})
        self.assertGreater(kv, 0)
        self.assertEqual(q % kv, 0, f"scale={scale} produced q={q} kv={kv}")
        self.assertLessEqual(q, 32)
        self.assertLessEqual(kv, 8)


if __name__ == "__main__":
  unittest.main()
