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

"""Unit tests for `maxtext.checkpoint_conversion.slicing`.

- reducer: pure config tests of the depth / expert / width rules.
- slice_model: greedy search and CLI/report flow with compile + checkpoint mocked.
- slice_utils: peak-HBM extraction, OOM classification, and a real (tiny, CPU)
  random-checkpoint write -> MaxText load round trip.
"""

import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file as save_safetensors_file
import torch

from maxtext.checkpoint_conversion.slicing import reducer
from maxtext.checkpoint_conversion.slicing import slice_hf_checkpoint
from maxtext.checkpoint_conversion.slicing import slice_model
from maxtext.checkpoint_conversion.slicing import slice_utils
from maxtext.checkpoint_conversion.slicing.reducer import DepthPlan
from maxtext.checkpoint_conversion.slicing.reducer import UnsupportedSliceError
from maxtext.common import checkpointing
from maxtext.utils import maxtext_utils
from maxtext.utils import model_creation_utils
from tests.utils.test_helpers import get_test_config_path

GiB = 1024**3
NO_MESH = {"tensor": 1, "expert": 1}


def make_config(**fields):
  """A MaxText-like config namespace with base.yml defaults for the fields the slicer reads."""
  defaults = {
      "decoder_block": "llama2",
      "global_parameter_scale": 1,
      "num_decoder_layers": 32,
      "first_num_dense_layers": 0,
      "first_num_hash_layers": 0,
      "inhomogeneous_layer_cycle_interval": 1,
      "interleave_moe_layer_step": 1,
      "nope_layer_interval": -1,
      "num_kv_shared_layers": 0,
      "engram_layers": [],
      "compress_ratios": [],
      "ici_pipeline_parallelism": 1,
      "dcn_pipeline_parallelism": 1,
      "use_multimodal": False,
      "deepstack_visual_indexes_for_vit": [],
      "num_experts": 1,
      "num_experts_per_tok": 1,
      "n_routing_groups": -1,
      "topk_routing_group": -1,
      "base_emb_dim": 4096,
      "base_mlp_dim": 14336,
      "base_moe_mlp_dim": -1,
      "base_num_query_heads": 32,
      "base_num_kv_heads": 8,
  }
  defaults.update(fields)
  return types.SimpleNamespace(**defaults)


def layers_of(overrides_list):
  return [o["base_num_decoder_layers"] for o in overrides_list]


# ----------------------------------------------------------------------------- reducer


class DepthPlanTest(unittest.TestCase):

  def test_depth_schedule_roughly_halves(self):
    self.assertEqual(reducer.depth_schedule(10, 1), [5, 3, 2, 1])
    self.assertEqual(reducer.depth_schedule(58, 1), [29, 15, 8, 4, 2, 1])
    self.assertEqual(reducer.depth_schedule(10, 3), [5, 3])
    self.assertEqual(reducer.depth_schedule(1, 1), [])

  def test_homogeneous_cycle_is_one(self):
    plan = reducer.get_depth_plan(make_config(num_decoder_layers=80))
    self.assertEqual(plan, DepthPlan(prefix=0, cycle=1, tail=0, num_cycles=80, min_cycles=1))
    self.assertEqual(layers_of(reducer.reduce_depth(make_config(num_decoder_layers=80), plan))[-1], 1)

  def test_qwen35_hybrid_cycle(self):
    config = make_config(decoder_block="qwen3_5", num_decoder_layers=40, inhomogeneous_layer_cycle_interval=4)
    plan = reducer.get_depth_plan(config)
    self.assertEqual(plan, DepthPlan(prefix=0, cycle=4, tail=0, num_cycles=10, min_cycles=1))
    self.assertEqual(layers_of(reducer.reduce_depth(config, plan)), [20, 12, 8, 4])

  def test_deepseek_dense_prefix(self):
    config = make_config(decoder_block="deepseek", num_decoder_layers=61, first_num_dense_layers=3, num_experts=256)
    plan = reducer.get_depth_plan(config)
    self.assertEqual((plan.prefix, plan.cycle, plan.tail, plan.num_cycles), (3, 1, 0, 58))
    self.assertEqual(layers_of(reducer.reduce_depth(config, plan)), [32, 18, 11, 7, 5, 4])

  def test_gemma3_pattern_keeps_tail(self):
    plan = reducer.get_depth_plan(make_config(decoder_block="gemma3", num_decoder_layers=34))
    self.assertEqual((plan.prefix, plan.cycle, plan.tail, plan.num_cycles), (0, 6, 4, 5))
    self.assertEqual(plan.layers(plan.min_cycles), 10)

  def test_block_patterns_and_intervals(self):
    llama4 = make_config(
        decoder_block="llama4",
        num_decoder_layers=48,
        nope_layer_interval=4,
        interleave_moe_layer_step=2,
        inhomogeneous_layer_cycle_interval=4,
    )
    self.assertEqual(reducer.get_depth_plan(llama4).cycle, 4)
    # olmo3 / gpt_oss patterns apply even without inhomogeneous_layer_cycle_interval.
    self.assertEqual(reducer.get_depth_plan(make_config(decoder_block="olmo3")).cycle, 4)
    self.assertEqual(reducer.get_depth_plan(make_config(decoder_block="gpt_oss", num_decoder_layers=24)).cycle, 2)
    # Gemma2 merges [local, global] into one MaxText layer.
    self.assertEqual(reducer.get_depth_plan(make_config(decoder_block="gemma2", num_decoder_layers=21)).cycle, 1)

  def test_enum_decoder_block(self):
    import enum  # pylint: disable=import-outside-toplevel

    class Block(enum.Enum):
      QWEN3_5 = "qwen3_5"

    config = make_config(decoder_block=Block.QWEN3_5, num_decoder_layers=8, inhomogeneous_layer_cycle_interval=4)
    self.assertEqual(reducer.get_depth_plan(config).cycle, 4)

  def test_deepseek4_hash_prefix_and_compress_ratios(self):
    ratios = [0, 0, 4] + [128, 4] * 20
    config = make_config(
        decoder_block="deepseek4", num_decoder_layers=43, first_num_hash_layers=3, compress_ratios=ratios
    )
    plan = reducer.get_depth_plan(config)
    self.assertEqual((plan.prefix, plan.cycle, plan.tail, plan.num_cycles), (3, 2, 0, 20))
    smallest = reducer.reduce_depth(config, plan)[-1]
    self.assertEqual(smallest, {"base_num_decoder_layers": 5, "compress_ratios": [0, 0, 4, 128, 4]})

  def test_deepstack_consumers_set_min_depth(self):
    config = make_config(
        decoder_block="qwen3", num_decoder_layers=28, use_multimodal=True, deepstack_visual_indexes_for_vit=[5, 11, 17]
    )
    plan = reducer.get_depth_plan(config)
    self.assertEqual(plan.min_cycles, 18)
    self.assertEqual(layers_of(reducer.reduce_depth(config, plan))[-1], 18)

  def test_unsupported_models_fail_closed(self):
    cases = {
        "unknown block": make_config(decoder_block="brand_new"),
        "kv sharing": make_config(decoder_block="gemma4", num_kv_shared_layers=20),
        "engram": make_config(decoder_block="deepseek", engram_layers=[1, 4]),
        "pipeline": make_config(ici_pipeline_parallelism=2),
        "global scale": make_config(global_parameter_scale=2),
        "no full cycle": make_config(decoder_block="gemma3", num_decoder_layers=4),
        "compress_ratios mismatch": make_config(decoder_block="deepseek4", num_decoder_layers=7, compress_ratios=[0, 4]),
    }
    for name, config in cases.items():
      with self.subTest(name), self.assertRaises(UnsupportedSliceError):
        reducer.get_depth_plan(config)


class ExpertReductionTest(unittest.TestCase):

  def test_halving_respects_top_k_and_expert_parallelism(self):
    config = make_config(decoder_block="qwen3_moe", num_experts=256, num_experts_per_tok=8)
    steps = reducer.reduce_experts(config, {"expert": 4})
    self.assertEqual([s["num_experts"] for s in steps], [128, 64, 32, 16, 8])

  def test_min_experts(self):
    config = make_config(decoder_block="qwen3_moe", num_experts=256, num_experts_per_tok=8)
    self.assertEqual([s["num_experts"] for s in reducer.reduce_experts(config, NO_MESH, 32)], [128, 64, 32])

  def test_expert_parallel_floor(self):
    config = make_config(decoder_block="mixtral", num_experts=8, num_experts_per_tok=2)
    self.assertEqual([s["num_experts"] for s in reducer.reduce_experts(config, {"expert": 8})], [])
    self.assertEqual([s["num_experts"] for s in reducer.reduce_experts(config, NO_MESH)], [4, 2])

  def test_top1_routing_keeps_at_least_two_experts(self):
    config = make_config(decoder_block="llama4", num_experts=16, num_experts_per_tok=1)
    self.assertEqual([s["num_experts"] for s in reducer.reduce_experts(config, NO_MESH)], [8, 4, 2])

  def test_grouped_routing_capacity(self):
    config = make_config(
        decoder_block="deepseek", num_experts=256, num_experts_per_tok=8, n_routing_groups=8, topk_routing_group=4
    )
    counts = [s["num_experts"] for s in reducer.reduce_experts(config, NO_MESH)]
    self.assertEqual(counts, [128, 64, 32, 16])
    for n in counts:
      self.assertEqual(n % 8, 0)
      self.assertGreaterEqual((n // 8) * 4, 8)

  def test_unsupported(self):
    with self.assertRaises(UnsupportedSliceError):
      reducer.reduce_experts(make_config(), NO_MESH)
    hashed = make_config(decoder_block="deepseek4", num_experts=256, num_experts_per_tok=6, first_num_hash_layers=3)
    with self.assertRaises(UnsupportedSliceError):
      reducer.reduce_experts(hashed, NO_MESH)


class WidthReductionTest(unittest.TestCase):

  def test_llama_gqa_with_tensor_parallelism(self):
    steps = reducer.reduce_width(make_config(), {"tensor": 4})
    self.assertEqual(
        steps,
        [
            {"base_emb_dim": 3072, "base_mlp_dim": 10752, "base_num_query_heads": 24, "base_num_kv_heads": 6},
            {"base_emb_dim": 2048, "base_mlp_dim": 7168, "base_num_query_heads": 16, "base_num_kv_heads": 4},
            {"base_emb_dim": 1024, "base_mlp_dim": 3584, "base_num_query_heads": 8, "base_num_kv_heads": 2},
        ],
    )
    for step in steps:
      self.assertEqual(step["base_num_query_heads"] % 4, 0)
      self.assertEqual(step["base_num_query_heads"] // step["base_num_kv_heads"], 4)

  def test_min_width_ratio(self):
    self.assertEqual(len(reducer.reduce_width(make_config(), NO_MESH, min_width_ratio=0.5)), 2)
    self.assertEqual(reducer.reduce_width(make_config(), NO_MESH, min_width_ratio=1.0), [])

  def test_moe_expert_width(self):
    config = make_config(decoder_block="qwen3_moe", num_experts=128, base_mlp_dim=768, base_moe_mlp_dim=768)
    step = reducer.reduce_width(config, NO_MESH)[1]
    self.assertEqual((step["base_mlp_dim"], step["base_moe_mlp_dim"]), (384, 384))

  def test_unsupported(self):
    for config in (
        make_config(decoder_block="qwen3_5"),
        make_config(decoder_block="deepseek"),
        make_config(decoder_block="qwen3", use_multimodal=True),
        make_config(base_num_query_heads=12, base_num_kv_heads=8),
        make_config(global_parameter_scale=2),
    ):
      with self.subTest(config.decoder_block), self.assertRaises(UnsupportedSliceError):
        reducer.reduce_width(config, NO_MESH)


# ----------------------------------------------------------------------------- slice_model


def make_qwen35_measure(bytes_for_shape):
  """A Qwen3.5-like config and a measure(overrides) backed by `bytes_for_shape(layers, experts, width_ratio)`."""
  base = make_config(
      decoder_block="qwen3_5",
      num_decoder_layers=40,
      inhomogeneous_layer_cycle_interval=4,
      num_experts=256,
      num_experts_per_tok=8,
  )

  def measure(overrides):
    shape = reducer.describe(base, overrides)
    return bytes_for_shape(shape["layers"], shape["experts"], shape["width_ratio"])

  return base, measure


def qwen35_steps(config):
  """steps_for_axis for the Qwen3.5-like config (width is unsupported for qwen3_5)."""
  plan = reducer.get_depth_plan(config)

  def steps(axis):
    if axis == "depth":
      return reducer.reduce_depth(config, plan)
    if axis == "expert":
      return reducer.reduce_experts(config, NO_MESH)
    return reducer.reduce_width(config, NO_MESH)

  return steps


class GreedySearchTest(unittest.TestCase):

  def run_search(self, bytes_for_shape, strategy=reducer.AXES):
    config, measure = make_qwen35_measure(bytes_for_shape)
    log = slice_model.SearchLog()
    selected = slice_model.search(config, strategy, measure, 24 * GiB, qwen35_steps(config), log=log)
    return selected, log

  def test_depth_first_returns_first_fit(self):
    peaks = {40: None, 20: 38 * GiB, 12: 27 * GiB, 8: 19 * GiB, 4: 10 * GiB}
    selected, log = self.run_search(lambda layers, experts, width: peaks[layers])
    self.assertEqual(selected, {"base_num_decoder_layers": 8})
    self.assertEqual([a.result for a in log.attempts], ["oom", "too_large", "too_large", "fits"])
    self.assertEqual({a.axis for a in log.attempts}, {"full", "depth"})

  def test_expert_runs_only_after_depth_fails(self):
    selected, log = self.run_search(lambda layers, experts, width: (experts / 256) * 40 * GiB)
    self.assertEqual(selected, {"base_num_decoder_layers": 4, "num_experts": 128})
    self.assertEqual([a.axis for a in log.attempts], ["full"] + ["depth"] * 4 + ["expert"])

  def test_width_runs_only_after_earlier_axes_fail(self):
    config = make_config(decoder_block="qwen3_moe", num_decoder_layers=4, num_experts=8, num_experts_per_tok=2)
    plan = reducer.get_depth_plan(config)

    def steps(axis):
      if axis == "depth":
        return reducer.reduce_depth(config, plan)
      if axis == "expert":
        return reducer.reduce_experts(config, NO_MESH)
      return reducer.reduce_width(config, NO_MESH)

    def measure(overrides):
      return 1 * GiB if "base_emb_dim" in overrides else 100 * GiB

    log = slice_model.SearchLog()
    selected = slice_model.search(config, reducer.AXES, measure, 24 * GiB, steps, log=log)
    self.assertEqual([a.axis for a in log.attempts], ["full", "depth", "depth", "expert", "expert", "width"])
    self.assertEqual(selected["base_num_decoder_layers"], 1)
    self.assertEqual(selected["num_experts"], 2)
    self.assertEqual(selected["base_emb_dim"], 3072)

  def test_full_model_fits(self):
    selected, log = self.run_search(lambda *_: GiB)
    self.assertEqual(selected, {})
    self.assertEqual(len(log.attempts), 1)

  def test_no_fit_records_skipped_axes(self):
    selected, log = self.run_search(lambda *_: 100 * GiB)
    self.assertIsNone(selected)
    self.assertEqual(len(log.attempts), 1 + 4 + 5)  # full + depth steps + expert steps
    self.assertEqual(len(log.notes), 1)
    self.assertIn("width: skipped", log.notes[0])

  def test_custom_strategy(self):
    selected, log = self.run_search(lambda layers, experts, width: (experts / 256) * 40 * GiB, ("expert",))
    self.assertEqual(selected, {"num_experts": 128})
    self.assertEqual([a.axis for a in log.attempts], ["full", "expert"])

  def test_measure_error_is_recorded_and_stops(self):
    config, _ = make_qwen35_measure(lambda *_: 0)

    def measure(overrides):
      if overrides:
        raise slice_utils.CompileFailedError("boom")
      return 100 * GiB

    log = slice_model.SearchLog()
    with self.assertRaises(slice_utils.CompileFailedError):
      slice_model.search(config, reducer.AXES, measure, GiB, qwen35_steps(config), log=log)
    self.assertEqual([a.result for a in log.attempts], ["too_large", "error"])


class CliTest(unittest.TestCase):

  def test_parse_size(self):
    self.assertEqual(slice_model.parse_size("24GiB"), 24 * GiB)
    self.assertEqual(slice_model.parse_size("24G"), 24 * GiB)
    self.assertEqual(slice_model.parse_size("24GB"), 24 * 10**9)
    self.assertEqual(slice_model.parse_size("1.5GiB"), int(1.5 * GiB))
    self.assertEqual(slice_model.parse_size("1024"), 1024)
    with self.assertRaises(Exception):
      slice_model.parse_size("24 parsecs")

  def test_parse_flags_keeps_maxtext_argv(self):
    flags, maxtext_argv = slice_model.parse_flags(
        [
            "prog",
            "my/base.yml",
            "model_name=qwen3.5-35b-a3b",
            "--hbm-budget-per-device=24GiB",
            "compile_topology=v6e-8",
            "--output-dir=/tmp/x",
            "--strategy=depth,width",
        ]
    )
    self.assertEqual(maxtext_argv, ["prog", "my/base.yml", "model_name=qwen3.5-35b-a3b", "compile_topology=v6e-8"])
    self.assertEqual(flags.strategy, ("depth", "width"))
    self.assertEqual(flags.hbm_budget_per_device, 24 * GiB)
    head, kv = slice_model.split_maxtext_argv(maxtext_argv)
    self.assertEqual(
        (head, kv), (["prog", "my/base.yml"], {"model_name": "qwen3.5-35b-a3b", "compile_topology": "v6e-8"})
    )
    self.assertEqual(slice_model.split_maxtext_argv(["prog", "a=1"]), (["prog"], {"a": "1"}))

  def test_parse_flags_rejects_bad_input(self):
    base = ["prog", "--hbm-budget-per-device=24GiB", "--output-dir=/tmp/x"]
    for extra in (["--strategy=depth,depth"], ["--strategy=layers"], ["not_a_kv"], ["--min-width-ratio=0"]):
      with self.subTest(extra), mock.patch("sys.stderr"), self.assertRaises(SystemExit):
        slice_model.parse_flags(base + extra)

  def test_check_onboarded(self):
    slice_model.check_onboarded("qwen3.5-35b-a3b")
    slice_model.check_onboarded("llama3.1-8b-Instruct")
    for name in (None, "default", "kimi-v3"):
      with self.subTest(name), self.assertRaises(slice_model.SliceError) as ctx:
        slice_model.check_onboarded(name)
      self.assertEqual(ctx.exception.status, "unsupported_model")
      self.assertIn("Please onboard the model into MaxText first", str(ctx.exception))

  def test_load_config_arguments(self):
    with mock.patch.object(slice_model, "pyconfig") as fake_pyconfig:
      argv = ["prog", "base.yml", "model_name=m", "compile_topology=v6e-8", "ici_tensor_parallelism=4"]
      argv.append("base_num_decoder_layers=9")
      slice_model.load_config(argv, {"base_num_decoder_layers": 8, "compress_ratios": [0, 4]})
      (cli,), kwargs = fake_pyconfig.initialize.call_args
      self.assertEqual(cli[:2], ["prog", "base.yml"])
      self.assertIn("override_model_config=true", cli)
      self.assertIn("compile_topology_num_slices=1", cli)
      self.assertNotIn("base_num_decoder_layers=9", cli)
      self.assertEqual(kwargs, {"base_num_decoder_layers": 8, "compress_ratios": [0, 4]})

      slice_model.load_config(argv, {}, drop_topology=True)
      (cli,), _ = fake_pyconfig.initialize.call_args
      self.assertFalse([a for a in cli if a.startswith(("compile_topology", "ici_"))])


class MainTest(unittest.TestCase):
  """End-to-end CLI flow with pyconfig, compilation, and checkpoint writing faked."""

  def run_main(self, extra_args=(), peak_for_layers=None, model_name="qwen3.5-35b-a3b", ckpt_error=None):
    """Run `slice_model.main` on a Qwen3.5-like model; returns (exit code, report, out dir, ckpt mock, loads)."""
    base_fields = {"decoder_block": "qwen3_5", "num_decoder_layers": 40, "inhomogeneous_layer_cycle_interval": 4}
    base_fields.update(num_experts=256, num_experts_per_tok=8, base_moe_mlp_dim=512)
    peaks = peak_for_layers or {40: None, 20: 38 * GiB, 12: 27 * GiB, 8: 19 * GiB, 4: 10 * GiB}
    loaded = []

    def fake_load_config(maxtext_argv, overrides=None, drop_topology=False):
      del maxtext_argv
      loaded.append((dict(overrides or {}), drop_topology))
      fields = dict(base_fields)
      if overrides and "base_num_decoder_layers" in overrides:
        fields["num_decoder_layers"] = overrides["base_num_decoder_layers"]
      return make_config(**fields)

    def fake_peak_hbm(config):
      peak = peaks[config.num_decoder_layers]
      if isinstance(peak, Exception):
        raise peak
      return peak

    out_dir = tempfile.mkdtemp()
    argv = ["prog", f"model_name={model_name}", "compile_topology=v6e-8", "per_device_batch_size=1"]
    argv += ["--hbm-budget-per-device=24GiB", f"--output-dir={out_dir}", *extra_args]
    flags, maxtext_argv = slice_model.parse_flags(argv)
    ckpt = mock.Mock(side_effect=ckpt_error, return_value=f"{out_dir}/checkpoint/0/items")
    with (
        mock.patch.object(slice_model, "load_config", side_effect=fake_load_config),
        mock.patch.object(slice_utils, "target_mesh_shape", return_value=NO_MESH),
        mock.patch.object(slice_utils, "measure_peak_hbm", side_effect=fake_peak_hbm),
        mock.patch.object(slice_utils, "write_random_checkpoint", ckpt),
    ):
      code = slice_model.main(maxtext_argv, flags)
    report = json.loads((Path(out_dir) / "slice_report.json").read_text(encoding="utf-8"))
    return code, report, Path(out_dir), ckpt, loaded

  def test_selects_depth_slice_and_writes_outputs(self):
    code, report, out_dir, ckpt, loaded = self.run_main()
    self.assertEqual(code, 0)
    self.assertEqual(
        json.loads((out_dir / "model_overrides.json").read_text(encoding="utf-8")), {"base_num_decoder_layers": 8}
    )
    self.assertEqual(report["status"], "selected")
    self.assertEqual(report["depth_plan"], {"prefix": 0, "cycle": 4, "tail": 0, "num_cycles": 10, "min_cycles": 1})
    self.assertEqual([a["layers"] for a in report["attempts"]], [40, 20, 12, 8])
    self.assertEqual([a["result"] for a in report["attempts"]], ["oom", "too_large", "too_large", "fits"])
    self.assertEqual(report["selected"]["layers"], 8)
    self.assertEqual(report["selected"]["experts"], 256)
    self.assertEqual(report["selected"]["peak_hbm_gib"], 19.0)
    self.assertEqual(report["checkpoint"]["type"], "random_initialized")
    self.assertIn("base_num_decoder_layers=8", report["launch_args"])
    # The checkpoint config is built on local devices with the selected overrides.
    self.assertEqual(loaded[-1], ({"base_num_decoder_layers": 8, "dataset_type": "synthetic"}, True))
    ckpt.assert_called_once()

  def test_skip_checkpoint(self):
    code, report, _, ckpt, _ = self.run_main(["--skip-checkpoint"])
    self.assertEqual((code, report["checkpoint"]), (0, {"type": "skipped"}))
    ckpt.assert_not_called()

  def test_no_model_fits(self):
    code, report, out_dir, _, _ = self.run_main(["--strategy=depth"], {n: 100 * GiB for n in (40, 20, 12, 8, 4)})
    self.assertEqual((code, report["status"]), (1, "no_model_fits"))
    self.assertEqual(len(report["attempts"]), 5)
    self.assertFalse((out_dir / "model_overrides.json").exists())

  def test_compile_failure_stops(self):
    peaks = {40: None, 20: slice_utils.CompileFailedError("sharding error")}
    code, report, _, _, _ = self.run_main(peak_for_layers=peaks)
    self.assertEqual((code, report["status"]), (1, "compile_failed"))
    self.assertEqual([a["result"] for a in report["attempts"]], ["oom", "error"])

  def test_unsupported_model(self):
    code, report, _, _, loaded = self.run_main(model_name="kimi-v3")
    self.assertEqual((code, report["status"]), (1, "unsupported_model"))
    self.assertEqual(loaded, [])

  def test_checkpoint_failure(self):
    code, report, _, _, _ = self.run_main(ckpt_error=RuntimeError("disk full"))
    self.assertEqual((code, report["status"]), (1, "checkpoint_failed"))


# ----------------------------------------------------------------------------- slice_utils


class SliceUtilsTest(unittest.TestCase):

  def test_peak_bytes(self):
    stats = types.SimpleNamespace(
        output_size_in_bytes=10, temp_size_in_bytes=100, argument_size_in_bytes=50, alias_size_in_bytes=5
    )
    self.assertEqual(slice_utils.peak_bytes(stats), 155)
    stats.peak_memory_in_bytes = 400
    self.assertEqual(slice_utils.peak_bytes(stats), 400)

  def test_oom_classification(self):
    self.assertTrue(slice_utils.is_oom_error(RuntimeError("RESOURCE_EXHAUSTED: Ran out of memory in memory space hbm")))
    self.assertFalse(slice_utils.is_oom_error(ValueError("shape mismatch")))

  def test_measure_peak_hbm(self):
    with mock.patch.object(slice_utils, "compile_train_step", side_effect=RuntimeError("RESOURCE_EXHAUSTED")):
      self.assertIsNone(slice_utils.measure_peak_hbm(object()))
    with mock.patch.object(slice_utils, "compile_train_step", side_effect=ValueError("bad sharding")):
      with self.assertRaises(slice_utils.CompileFailedError):
        slice_utils.measure_peak_hbm(object())
    compiled = mock.Mock()
    compiled.memory_analysis.return_value = None
    with mock.patch.object(slice_utils, "compile_train_step", return_value=compiled):
      with self.assertRaises(slice_utils.CompileFailedError):
        slice_utils.measure_peak_hbm(object())

  def test_math_helpers(self):
    self.assertEqual(slice_utils.lcm(4, 2, -1, 1, 6), 12)
    self.assertEqual(slice_utils.lcm(), 1)
    self.assertEqual((slice_utils.align_down(300, 128), slice_utils.align_up(300, 128)), (256, 384))


class RandomCheckpointRoundTripTest(unittest.TestCase):
  """Write a random checkpoint for a tiny model and load it back through MaxText's loader."""

  TINY = {
      "base_num_decoder_layers": 2,
      "base_emb_dim": 64,
      "base_mlp_dim": 128,
      "base_num_query_heads": 2,
      "base_num_kv_heads": 2,
      "head_dim": 32,
      "vocab_size": 256,
      "max_target_length": 16,
      "per_device_batch_size": 1,
      "attention": "dot_product",
  }

  def _init_params(self, config, seed):
    mesh = maxtext_utils.get_mesh_from_config(config)
    create_model_fn = model_creation_utils.get_nnx_create_model_fn(config, mesh, rng_key=jax.random.PRNGKey(seed))
    abstract, _, shardings = maxtext_utils.get_abstract_state(config, mesh, create_model_fn, is_training=False)
    with jax.set_mesh(mesh), nn_partitioning.axis_rules(config.logical_axis_rules):
      state = jax.jit(lambda: nnx.state(create_model_fn()), out_shardings=shardings)()
    return nnx.split_state(abstract, nnx.Param, ...)[0], nnx.split_state(state, nnx.Param, ...)[0]

  def test_round_trip(self):
    out = tempfile.mkdtemp()
    argv = [sys.argv[0], get_test_config_path(), "run_name=slice_ckpt_test", f"base_output_directory={out}"]
    config = slice_model.load_config(argv, {**self.TINY, "dataset_type": "synthetic"})
    path = slice_utils.write_random_checkpoint(config, os.path.join(out, "checkpoint"), seed=3)
    self.assertEqual(path, os.path.join(out, "checkpoint", "0", "items"))

    abstract_params, expected = self._init_params(config, seed=3)
    with nn_partitioning.axis_rules(config.logical_axis_rules):
      loaded = checkpointing.load_params_from_path(
          path,
          abstract_params,
          config.checkpoint_storage_concurrent_gb,
          config.checkpoint_storage_use_ocdbt,
          config.checkpoint_storage_use_zarr3,
      )

    def flat(tree):
      pure = tree.to_pure_dict() if isinstance(tree, nnx.State) else tree
      return {k: np.asarray(v) for k, v in nnx.traversals.flatten_mapping(pure).items()}

    loaded_flat, expected_flat = flat(loaded), flat(expected)
    self.assertEqual(sorted(loaded_flat), sorted(expected_flat))
    for key, value in expected_flat.items():
      np.testing.assert_array_equal(loaded_flat[key], value, err_msg=str(key))


# ----------------------------------------------------------------------------- slice_hf_checkpoint


def _qwen35_like_config(layers=8, experts=8, hidden=64, moe_dim=32):
  return {
      "architectures": ["Qwen3_5MoeForConditionalGeneration"],
      "model_type": "qwen3_5_moe",
      "text_config": {
          "hidden_size": hidden,
          "num_hidden_layers": layers,
          "num_attention_heads": 4,
          "num_key_value_heads": 2,
          "head_dim": 16,
          "num_experts": experts,
          "moe_intermediate_size": moe_dim,
          "shared_expert_intermediate_size": moe_dim,
          "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * (layers // 4),
          "mlp_only_layers": [],
          "mtp_num_hidden_layers": 1,
          "vocab_size": 128,
      },
      "vision_config": {"depth": 1, "hidden_size": 16, "out_hidden_size": hidden},
  }


def _qwen35_like_tensors(cfg):
  """Tensors with the Qwen3.5 HF key layout (fused experts, router, GDN, MTP, vision)."""
  t = cfg["text_config"]
  h, e, m, n = t["hidden_size"], t["num_experts"], t["moe_intermediate_size"], t["num_hidden_layers"]
  gen = torch.Generator().manual_seed(0)

  def rnd(*shape):
    return torch.randn(*shape, generator=gen).to(torch.bfloat16)

  tensors = {
      "model.language_model.embed_tokens.weight": rnd(t["vocab_size"], h),
      "model.language_model.norm.weight": rnd(h),
      "lm_head.weight": rnd(t["vocab_size"], h),
      "model.visual.blocks.0.norm1.weight": rnd(16),
      "mtp.fc.weight": rnd(h, 2 * h),
      f"mtp.layers.0.mlp.experts.{e - 1}.down_proj.weight": rnd(h, m),
  }
  for i in range(n):
    p = f"model.language_model.layers.{i}"
    tensors[f"{p}.input_layernorm.weight"] = rnd(h)
    tensors[f"{p}.linear_attn.in_proj_qkv.weight"] = rnd(96, h)
    tensors[f"{p}.linear_attn.A_log"] = rnd(4)
    tensors[f"{p}.mlp.gate.weight"] = rnd(e, h)
    tensors[f"{p}.mlp.experts.gate_up_proj"] = rnd(e, 2 * m, h)
    tensors[f"{p}.mlp.experts.down_proj"] = rnd(e, h, m)
    tensors[f"{p}.mlp.shared_expert.gate_proj.weight"] = rnd(m, h)
    tensors[f"{p}.mlp.shared_expert_gate.weight"] = rnd(1, h)
  return tensors


def _write_sharded(directory, config, tensors, shard_of):
  """Write `tensors` into multiple shards (`shard_of(key) -> shard index`) plus index.json and config.json."""
  os.makedirs(directory, exist_ok=True)
  shards = {}
  for key, value in tensors.items():
    shards.setdefault(shard_of(key), {})[key] = value.contiguous()
  weight_map = {}
  for idx, part in sorted(shards.items()):
    name = f"model-{idx:05d}-of-{len(shards):05d}.safetensors"
    save_safetensors_file(part, os.path.join(directory, name))
    weight_map.update({k: name for k in part})
  Path(directory, "model.safetensors.index.json").write_text(
      json.dumps({"metadata": {}, "weight_map": weight_map}), encoding="utf-8"
  )
  Path(directory, "config.json").write_text(json.dumps(config), encoding="utf-8")
  Path(directory, "tokenizer.json").write_text("{}", encoding="utf-8")


def _read_all(directory):
  index = json.loads(Path(directory, "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
  out = {}
  for key, shard in index.items():
    with safe_open(os.path.join(directory, shard), framework="pt") as f:
      out[key] = f.get_tensor(key)
  return out


class SliceHFConfigTest(unittest.TestCase):

  def test_depth_and_expert_config(self):
    cfg = _qwen35_like_config(layers=8, experts=8)
    spec = slice_hf_checkpoint.build_hf_slice_spec(cfg, {"base_num_decoder_layers": 4, "num_experts": 4})
    self.assertEqual((spec.new_layers, spec.new_experts), (4, 4))
    self.assertFalse(spec.width_sliced)
    out = slice_hf_checkpoint.slice_hf_config(cfg, spec)
    t = out["text_config"]
    self.assertEqual((t["num_hidden_layers"], t["num_experts"], t["mtp_num_hidden_layers"]), (4, 4, 0))
    self.assertEqual(t["layer_types"], cfg["text_config"]["layer_types"][:4])
    self.assertEqual(cfg["text_config"]["num_hidden_layers"], 8)  # input is not mutated

  def test_gemma2_bundles_two_hf_layers(self):
    cfg = {"model_type": "gemma2", "num_hidden_layers": 26, "hidden_size": 64, "num_attention_heads": 4}
    spec = slice_hf_checkpoint.build_hf_slice_spec(cfg, {"base_num_decoder_layers": 3}, model_name="gemma2-2b")
    self.assertEqual(spec.new_layers, 6)

  def test_rejects_unknown_override(self):
    with self.assertRaises(ValueError):
      slice_hf_checkpoint.build_hf_slice_spec(_qwen35_like_config(), {"head_dim": 8})

  def test_load_overrides_from_report_and_inline(self):
    report = Path(tempfile.mkdtemp(), "slice_report.json")
    report.write_text(
        json.dumps({"model_name": "qwen3.5-35b-a3b", "selected": {"overrides": {"base_num_decoder_layers": 4}}}),
        encoding="utf-8",
    )
    overrides, name = slice_hf_checkpoint.load_overrides(str(report), None, ["num_experts=16"])
    self.assertEqual(name, "qwen3.5-35b-a3b")
    self.assertEqual(overrides, {"base_num_decoder_layers": 4, "num_experts": 16})


class SliceHFCheckpointTest(unittest.TestCase):

  def test_depth_expert_slice_qwen35_layout(self):
    src, dst = tempfile.mkdtemp(), tempfile.mkdtemp()
    cfg = _qwen35_like_config(layers=8, experts=8)
    tensors = _qwen35_like_tensors(cfg)

    def shard_of(key):  # layers 4..7 and MTP live only in shard 2, which must never be opened.
      match = slice_hf_checkpoint._DECODER_LAYER_RE.match(key)  # pylint: disable=protected-access
      return 2 if key.startswith("mtp.") or (match and int(match.group(1)) >= 4) else 1

    _write_sharded(src, cfg, tensors, shard_of)
    report = slice_hf_checkpoint.slice_hf_checkpoint(src, dst, {"base_num_decoder_layers": 4, "num_experts": 4})

    self.assertEqual((report["stats"]["source_shards_read"], report["stats"]["source_shards_skipped"]), (1, 1))
    out = _read_all(dst)
    expected_keys = {k for k in tensors if not k.startswith("mtp.") and shard_of(k) == 1}
    self.assertEqual(set(out), expected_keys)
    for key in expected_keys:
      if key.endswith(("mlp.gate.weight", "experts.gate_up_proj", "experts.down_proj")):
        torch.testing.assert_close(out[key], tensors[key][:4], rtol=0, atol=0, msg=key)
      else:
        torch.testing.assert_close(out[key], tensors[key], rtol=0, atol=0, msg=key)
    self.assertEqual(json.loads(Path(dst, "config.json").read_text(encoding="utf-8"))["text_config"]["num_experts"], 4)
    self.assertTrue(Path(dst, "tokenizer.json").exists())
    self.assertTrue(Path(dst, "hf_slice_report.json").exists())

  def test_width_slice_gqa_mlp_and_fp8_scales(self):
    src, dst = tempfile.mkdtemp(), tempfile.mkdtemp()
    h, q_heads, kv_heads, hd, m = 256, 4, 2, 64, 512
    cfg = {
        "model_type": "llama", "num_hidden_layers": 2, "hidden_size": h, "intermediate_size": m,
        "num_attention_heads": q_heads, "num_key_value_heads": kv_heads, "head_dim": hd, "vocab_size": 32,
    }  # fmt: skip
    gen = torch.Generator().manual_seed(1)
    tensors = {"model.embed_tokens.weight": torch.randn(32, h, generator=gen), "model.norm.weight": torch.randn(h)}
    for i in range(2):
      p = f"model.layers.{i}"
      tensors.update(
          {
              f"{p}.input_layernorm.weight": torch.randn(h, generator=gen),
              f"{p}.self_attn.q_proj.weight": torch.randn(q_heads * hd, h, generator=gen),
              f"{p}.self_attn.k_proj.weight": torch.randn(kv_heads * hd, h, generator=gen),
              f"{p}.self_attn.v_proj.weight": torch.randn(kv_heads * hd, h, generator=gen),
              f"{p}.self_attn.o_proj.weight": torch.randn(h, q_heads * hd, generator=gen),
              f"{p}.mlp.gate_proj.weight": torch.randn(m, h, generator=gen),
              f"{p}.mlp.gate_proj.weight_scale_inv": torch.randn(m // 128, h // 128, generator=gen),
              f"{p}.mlp.down_proj.weight": torch.randn(h, m, generator=gen),
          }
      )
    _write_sharded(src, cfg, tensors, lambda k: 1)
    overrides = {"base_emb_dim": 128, "base_mlp_dim": 256, "base_num_query_heads": 2, "base_num_kv_heads": 1}
    slice_hf_checkpoint.slice_hf_checkpoint(src, dst, overrides)

    out = _read_all(dst)
    p = "model.layers.1"
    # First KV group and its G=2 query heads are kept together.
    torch.testing.assert_close(out[f"{p}.self_attn.q_proj.weight"], tensors[f"{p}.self_attn.q_proj.weight"][:128, :128])
    torch.testing.assert_close(out[f"{p}.self_attn.k_proj.weight"], tensors[f"{p}.self_attn.k_proj.weight"][:64, :128])
    torch.testing.assert_close(out[f"{p}.self_attn.o_proj.weight"], tensors[f"{p}.self_attn.o_proj.weight"][:128, :128])
    torch.testing.assert_close(out[f"{p}.mlp.down_proj.weight"], tensors[f"{p}.mlp.down_proj.weight"][:128, :256])
    self.assertEqual(tuple(out[f"{p}.mlp.gate_proj.weight_scale_inv"].shape), (2, 1))
    self.assertEqual(tuple(out["model.embed_tokens.weight"].shape), (32, 128))
    new_cfg = json.loads(Path(dst, "config.json").read_text(encoding="utf-8"))
    self.assertEqual(
        (new_cfg["hidden_size"], new_cfg["intermediate_size"], new_cfg["num_attention_heads"]), (128, 256, 2)
    )

  def test_width_slice_fails_closed_on_unknown_tensor(self):
    spec = slice_hf_checkpoint.build_hf_slice_spec(
        {"num_hidden_layers": 2, "hidden_size": 256, "num_attention_heads": 4, "head_dim": 64},
        {"base_emb_dim": 128},
    )
    with self.assertRaises(ValueError):
      slice_hf_checkpoint.plan_tensor_slice("model.layers.0.mystery.weight", (256, 256), spec)

  def test_stream_reader_matches_safe_open(self):
    path = os.path.join(tempfile.mkdtemp(), "t.safetensors")
    value = torch.randn(16, 8, 4).to(torch.bfloat16)
    save_safetensors_file({"x": value, "y": torch.ones(3)}, path)
    slices = (slice(0, 5), slice(0, 8), slice(0, 2))
    with open(path, "rb") as stream:
      base, header = slice_hf_checkpoint._read_safetensors_header_from_stream(stream)  # pylint: disable=protected-access
      got = slice_hf_checkpoint._load_sliced_tensor_from_stream(stream, base, header["x"], slices)  # pylint: disable=protected-access
    torch.testing.assert_close(got, value[slices], rtol=0, atol=0)


if __name__ == "__main__":
  unittest.main()
