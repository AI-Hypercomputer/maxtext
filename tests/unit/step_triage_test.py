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

"""Unit tests for `maxtext.diagnostics.step_triage`."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from maxtext.diagnostics.step_triage import analysis
from maxtext.diagnostics.step_triage import cli
from maxtext.diagnostics.step_triage import events
from maxtext.diagnostics.step_triage import report
from maxtext.diagnostics.step_triage import trigger


FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "step_triage")


def _generate_3d_slow_core_log(
    bounds: tuple[int, int, int] = (4, 4, 4),
    cores_per_chip: int = 2,
    pipeline_depth: int = 13,
    slow_cores_p: dict[str, int] | None = None,
    waves_per_core: int = 2,
    core0_run_offset: int = 0,
) -> list[str]:
  """Generates synthetic 3D torus ICI-SDC logs for slow-core testing."""
  if slow_cores_p is None:
    slow_cores_p = {"1x1x1_TC0": 131}
  w, h, z = bounds
  step = pipeline_depth + 1
  lines: list[str] = [
      "I1021 12:00:00.000000 100 llo_sdc_checker_emitters.cc:423] "
      "IciSdcTestEmitter :: Emitting code for buffer_size_chunks:1024 "
      "packet_size_chunks:32 iterations:1000 max_neighbors:6 "
      f"max_distance:1 pipeline_depth:{pipeline_depth} "
      "pipeline_factor:4 delay_mask:16777215"
  ]

  all_chips = [(gx, gy, gz) for gz in range(z) for gy in range(h) for gx in range(w)]
  chip_to_cores: dict[tuple[int, int, int], list[str]] = {}
  core_to_chip: dict[str, tuple[int, int, int]] = {}
  core_to_g: dict[str, int] = {}

  for gx, gy, gz in all_chips:
    chip = (gx, gy, gz)
    chip_to_cores[chip] = []
    for tc in range(cores_per_chip):
      cname = f"{gx}x{gy}x{gz}_TC{tc}"
      g_idx = (gx + w * (gy + h * gz)) * cores_per_chip + tc
      chip_to_cores[chip].append(cname)
      core_to_chip[cname] = chip
      core_to_g[cname] = g_idx
      r_id = 1 + (core0_run_offset if cname == "0x0x0_TC0" else 0)
      p_val = slow_cores_p.get(cname, 0)
      bw = 42.5 if p_val > 0 else 162.0
      lines.append(
          f"I1021 12:00:01.000000 100 sdc.cc:100] [job.{cname}@{r_id}]: "
          f"IciSdcTest :: ICI BW: {bw} (GiB/s), clock: 1250 (MHz), "
          f"data: 32 (MiB), time: 0.2 (ms), delay iterations: {p_val}"
      )

  wrap_axes = (w >= 3, h >= 3, z >= 3)
  for slow_c, p_val in slow_cores_p.items():
    slow_chip = core_to_chip[slow_c]
    for k in range(1, waves_per_core + 1):
      s0 = k * p_val - 1
      for chip in all_chips:
        dist = analysis.chip_manhattan_distance(slow_chip, chip, bounds, wrap_axes)
        for cname in chip_to_cores[chip]:
          if cname == slow_c:
            continue
          r_id = 1 + (core0_run_offset if cname == "0x0x0_TC0" else 0)
          g_idx = core_to_g[cname]
          if dist == 1:
            it = s0
          elif dist == 0:
            # Sibling TensorCore on the same chip stalls at S1 = S0 + D + 1.
            it = s0 + step
          elif dist <= 2:
            it = s0 + (dist - 1) * step
          else:
            continue
          lines.append(
              f"I1021 12:00:02.000000 100 sdc.cc:200] [job.{cname}@{r_id}]: "
              f"ICI transfer wait on RECV flag timed out on {cname} "
              f"at iteration {it} on core with global core index {g_idx}"
          )
  return lines


class StepTriageTest(unittest.TestCase):
  """Tests event parsing, wave/sflag attribution, reporting, and CLI."""

  def test_fixture_exp1_r1_waves(self):
    path = os.path.join(FIXTURES_DIR, "exp1_r1_waves.log")
    parsed = events.parse_paths([path])
    res = analysis.analyze_parsed_logs(parsed)
    rep = report.build_straggler_report(res, job_name="exp1_r1")

    self.assertEqual(rep["verdict"]["primary_cause"], "TPU_SLOW_CORE")
    stage_c_a1 = rep["probes"]["stage_c_approach1_ici_sdc"]
    self.assertEqual(stage_c_a1["total_waves"], 2)
    culprit_map = {c["core"]: c for c in stage_c_a1["culprit_cores"]}
    self.assertIn("1x3x3_TC0", culprit_map)
    self.assertIn("0x3x1_TC0", culprit_map)
    self.assertEqual(culprit_map["1x3x3_TC0"]["logged_delay_iterations_P"], 139)
    self.assertTrue(culprit_map["1x3x3_TC0"]["p_divides_s0_plus_1"])
    self.assertEqual(culprit_map["0x3x1_TC0"]["logged_delay_iterations_P"], 151)
    self.assertTrue(culprit_map["0x3x1_TC0"]["p_divides_s0_plus_1"])

  def test_fixture_exp2_b_clean(self):
    path = os.path.join(FIXTURES_DIR, "exp2_b_clean.log")
    parsed = events.parse_paths([path])
    res = analysis.analyze_parsed_logs(parsed)
    rep = report.build_straggler_report(res, job_name="exp2_b_clean")

    self.assertEqual(rep["verdict"]["primary_cause"], "CLEAN")
    self.assertEqual(rep["verdict"]["recommended_action"], "NO_ACTION")
    self.assertEqual(rep["verdict"]["suspect_cores"], [])
    self.assertEqual(rep["probes"]["stage_c_approach1_ici_sdc"]["total_waves"], 0)

  def test_fixture_xm_approach2_sc_and_tc_and_healthy(self):
    # 1. 2-host SparseCore delay on 1x0x0_TC0
    sc_path = os.path.join(FIXTURES_DIR, "xm_2h_c_delay_sc.log")
    res_sc = analysis.analyze_parsed_logs(events.parse_paths([sc_path]))
    rep_sc = report.build_straggler_report(res_sc, job_name="2h_c_delay_sc")
    self.assertEqual(rep_sc["verdict"]["primary_cause"], "TPU_SLOW_CORE")
    self.assertEqual(rep_sc["verdict"]["suspect_cores"], ["1x0x0_TC0"])
    a2_sc = rep_sc["probes"]["stage_c_approach2_sflag"]["culprit_cores"]
    self.assertEqual(len(a2_sc), 1)
    self.assertEqual(a2_sc[0]["core"], "1x0x0_TC0")
    self.assertEqual(a2_sc[0]["peer_Wait_sflag_timeouts"], 15)
    self.assertEqual(a2_sc[0]["total_cores"], 16)
    self.assertGreaterEqual(a2_sc[0]["consistent_steps"], 3)

    # 2. 1-host TensorCore delay on 1x0x0_TC0 (includes self-satisfied timeouts)
    tc_path = os.path.join(FIXTURES_DIR, "xm_1h_c_delay_tc.log")
    res_tc = analysis.analyze_parsed_logs(events.parse_paths([tc_path]))
    rep_tc = report.build_straggler_report(res_tc, job_name="1h_c_delay_tc")
    self.assertEqual(rep_tc["verdict"]["primary_cause"], "TPU_SLOW_CORE")
    self.assertEqual(rep_tc["verdict"]["suspect_cores"], ["1x0x0_TC0"])
    a2_tc = rep_tc["probes"]["stage_c_approach2_sflag"]["culprit_cores"]
    self.assertEqual(len(a2_tc), 1)
    self.assertEqual(a2_tc[0]["core"], "1x0x0_TC0")
    self.assertEqual(a2_tc[0]["peer_Wait_sflag_timeouts"], 7)
    self.assertEqual(a2_tc[0]["total_cores"], 8)
    self.assertGreaterEqual(a2_tc[0]["self_satisfied_timeouts"], 3)

    # 3. 2-host healthy (only launch-barrier and startup noise)
    h_path = os.path.join(FIXTURES_DIR, "xm_2h_b_healthy.log")
    res_h = analysis.analyze_parsed_logs(events.parse_paths([h_path]))
    rep_h = report.build_straggler_report(res_h, job_name="2h_b_healthy")
    self.assertEqual(rep_h["verdict"]["primary_cause"], "LAUNCH_BARRIER_NOISE_ONLY")
    self.assertEqual(rep_h["verdict"]["suspect_cores"], [])

  def test_fixture_cloud_p1_compiler_options(self):
    path = os.path.join(FIXTURES_DIR, "cloud_p1_compiler_options.log")
    parsed = events.parse_paths([path])
    res = analysis.analyze_parsed_logs(parsed)
    rep = report.build_straggler_report(res, job_name="p1_compiler_options")

    self.assertEqual(rep["verdict"]["primary_cause"], "TPU_SLOW_CORE")
    self.assertEqual(rep["verdict"]["suspect_cores"], ["1x0x0_TC0"])
    a2 = rep["probes"]["stage_c_approach2_sflag"]["culprit_cores"]
    self.assertEqual(len(a2), 1)
    self.assertEqual(a2[0]["core"], "1x0x0_TC0")
    self.assertEqual(a2[0]["peer_Wait_sflag_timeouts"], 127)
    self.assertEqual(a2[0]["total_cores"], 128)

  def test_fixture_exp3_d_compact_code(self):
    path = os.path.join(FIXTURES_DIR, "exp3_d_compact_code.log")
    parsed = events.parse_paths([path])
    res = analysis.analyze_parsed_logs(parsed)
    rep = report.build_straggler_report(res, job_name="exp3_d_compact")

    self.assertEqual(rep["verdict"]["primary_cause"], "TPU_SLOW_CORE")
    stage_c_a1 = rep["probes"]["stage_c_approach1_ici_sdc"]
    self.assertEqual(stage_c_a1["total_waves"], 3)
    culprit_cores = [c["core"] for c in stage_c_a1["culprit_cores"]]
    self.assertIn("0x0x0_TC1", culprit_cores)
    self.assertIn("0x0x1_TC1", culprit_cores)

  def test_fixture_xm_sdk_callback_events(self):
    path = os.path.join(FIXTURES_DIR, "xm_sdk_callback_events.log")
    parsed = events.parse_paths([path])
    res = analysis.analyze_parsed_logs(parsed)
    rep = report.build_straggler_report(res, job_name="xm_sdk_callbacks")

    self.assertEqual(rep["verdict"]["primary_cause"], "TPU_SLOW_CORE")
    self.assertIn("1x0x0_TC0", rep["verdict"]["suspect_cores"])
    self.assertEqual(
        [c["core"] for c in rep["probes"]["stage_c_approach2_sflag"]["culprit_cores"]],
        ["1x0x0_TC0"],
    )
    self.assertGreaterEqual(rep["probes"]["stage_c_approach1_ici_sdc"]["total_waves"], 1)

  def test_synth_bad_link_unidirectional_and_bidirectional(self):
    # Unidirectional SEND timeout at S0=41 from 1x1x0_TC0 -> 1x2x0_TC0,
    # followed by S1=55 RECV propagation on neighbors of 1x1x0_TC0.
    lines_uni = [
        "I1021 12:00:00.000000 1 sdc.cc:1] IciSdcTestEmitter: max_distance=1, "
        "pipeline_depth=13, send_timeout_cycles=6250000, recv_timeout_cycles=6250000",
        "I1021 12:00:01.000000 1 sdc.cc:2] [job.1x1x0_TC0@1]: ICI SDC Test "
        "transfer SEND timeout : 1x1x0_TC0 -> 1x2x0_TC0 - 4194304 bytes - iteration 41",
        "I1021 12:00:01.000100 1 sdc.cc:2] [job.1x2x0_TC0@1]: ICI SDC Test "
        "transfer RECV timeout : 1x1x0_TC0 -> 1x2x0_TC0 - 4194304 bytes - iteration 41",
        "I1021 12:00:01.000200 1 sdc.cc:2] [job.0x1x0_TC0@1]: ICI SDC Test "
        "transfer RECV timeout : 1x1x0_TC0 -> 0x1x0_TC0 - 4194304 bytes - iteration 55",
        "I1021 12:00:01.000300 1 sdc.cc:2] [job.3x3x0_TC1@1]: IciSdcTest :: "
        "ICI BW: 160.0 GiB/s, time: 1.0 ms (global core index: 31) (delay iterations: 0)",
    ]
    res_uni = analysis.analyze_parsed_logs(events.parse_lines(lines_uni))
    rep_uni = report.build_straggler_report(res_uni)
    self.assertEqual(rep_uni["verdict"]["primary_cause"], "ICI_LINK_DEGRADATION")
    self.assertEqual(rep_uni["verdict"]["suspect_links"], ["1x1x0_TC0 -> 1x2x0_TC0"])
    self.assertEqual(res_uni["waves"][0]["hop_histogram"], {0: 2, 1: 1})

    # Bidirectional SEND timeouts at S0=99 between 1x1x0_TC0 and 1x2x0_TC0.
    lines_bi = [
        "I1021 12:00:01.000000 1 sdc.cc:2] [job.1x1x0_TC0@1]: ICI SDC Test "
        "transfer SEND timeout : 1x1x0_TC0 -> 1x2x0_TC0 - 4194304 bytes - iteration 99",
        "I1021 12:00:01.000100 1 sdc.cc:2] [job.1x2x0_TC0@1]: ICI SDC Test "
        "transfer SEND timeout : 1x2x0_TC0 -> 1x1x0_TC0 - 4194304 bytes - iteration 99",
        "I1021 12:00:01.000300 1 sdc.cc:2] [job.3x3x0_TC1@1]: IciSdcTest :: "
        "ICI BW: 160.0 GiB/s, time: 1.0 ms (global core index: 31) (delay iterations: 0)",
    ]
    res_bi = analysis.analyze_parsed_logs(events.parse_lines(lines_bi))
    self.assertEqual(res_bi["overall_verdict"], "SUSPECTED_BAD_LINK")
    self.assertIn("1x1x0_TC0 <-> 1x2x0_TC0", res_bi["culprit_links"])

  def test_synth_slow_core_single_and_interleaved_with_run_offset(self):
    # Single slow core 1x1x1_TC0 (P=131), testing both P-checked and blind attribution.
    lines_single = _generate_3d_slow_core_log(
        bounds=(4, 4, 4),
        cores_per_chip=2,
        slow_cores_p={"1x1x1_TC0": 131},
        waves_per_core=2,
    )
    res_s = analysis.analyze_parsed_logs(events.parse_lines(lines_single))
    self.assertEqual(res_s["overall_verdict"], "SUSPECTED_SLOW_CORE")
    self.assertEqual(res_s["culprit_slow_cores"], ["1x1x1_TC0"])
    for w in res_s["waves"]:
      self.assertEqual(w["blind_top_core"], "1x1x1_TC0")
      self.assertEqual(w["p_checked_top_core"], "1x1x1_TC0")

    # Interleaved two slow cores (1x1x1_TC0 P=101, 2x2x2_TC1 P=139) with core0 @run_id offset +1.
    lines_inter = _generate_3d_slow_core_log(
        bounds=(4, 4, 4),
        cores_per_chip=2,
        slow_cores_p={"1x1x1_TC0": 101, "2x2x2_TC1": 139},
        waves_per_core=2,
        core0_run_offset=1,
    )
    parsed_i = events.parse_lines(lines_inter)
    self.assertEqual(parsed_i.run_id_offsets.get("0x0x0_TC0"), 1)
    res_i = analysis.analyze_parsed_logs(parsed_i)
    self.assertEqual(set(res_i["culprit_slow_cores"]), {"1x1x1_TC0", "2x2x2_TC1"})
    self.assertEqual(len(res_i["waves"]), 4)

  def test_per_step_run_id_drift_with_launch_id_and_multirun_p(self):
    # Simulates a 2x2x2 topology where 0x0x0_TC0 runs single-device programs
    # every step (drifting from @13 on run 1 to @36 on run 2) with shared hex
    # launch_id tags, and 0x0x1_TC0 is slow with re-randomized P across runs
    # (P=101 on run 1 -> S0=100; P=131 on run 2 -> S0=130).
    lines: list[str] = [
        "I1021 12:00:00.000000 100 llo_sdc_checker_emitters.cc:423] "
        "IciSdcTestEmitter :: Emitting code for buffer_size_chunks:32 "
        "packet_size_chunks:32 iterations:200 max_neighbors:6 "
        "max_distance:1 pipeline_depth:13 pipeline_factor:14 delay_mask:134217727"
    ]
    all_chips = [(gx, gy, gz) for gz in range(2) for gy in range(2) for gx in range(2)]
    runs = [
        (1, 13, "5fcde628", 101),
        (2, 36, "7a1b2c3d", 131),
    ]
    slow_c = "0x0x1_TC0"
    slow_chip = (0, 0, 1)
    for peer_run, c0_run, launch_hex, p_val in runs:
      s0 = p_val - 1
      for gx, gy, gz in all_chips:
        chip = (gx, gy, gz)
        dist = analysis.chip_manhattan_distance(slow_chip, chip, (2, 2, 2), (False, False, False))
        for tc in range(2):
          cname = f"{gx}x{gy}x{gz}_TC{tc}"
          g_idx = (gx + 2 * (gy + 2 * gz)) * 2 + tc
          r_id = c0_run if cname == "0x0x0_TC0" else peer_run
          cur_p = p_val if cname == slow_c else 0
          lines.append(
              f"I1021 12:00:01.000000 100 llo_log_report_handler.cc:100] [{cname}@{r_id}:{launch_hex}:0] "
              f"IciSdcTest :: ICI BW: 150.0 (GiB/s), clock: 1250 (MHz), "
              f"data: 32 (MiB), time: 0.2 (ms), delay iterations: {cur_p}"
          )
          if cname == slow_c:
            continue
          it = s0 if dist == 1 else s0 + 14
          lines.append(
              f"I1021 12:00:02.000000 100 llo_log_report_handler.cc:200] [{cname}@{r_id}:{launch_hex}:0] "
              f"ICI transfer wait on RECV flag timed out on {cname} "
              f"at iteration {it} on core with global core index {g_idx}"
          )

    parsed = events.parse_lines(lines)
    res = analysis.analyze_parsed_logs(parsed)
    rep = report.build_straggler_report(res, job_name="drift_multirun_p")

    self.assertEqual(res["total_waves_clustered"], 2)
    self.assertEqual(res["noise_wave_count"], 0)
    self.assertEqual(res["culprit_slow_cores"], [slow_c])
    stage_c_a1 = rep["probes"]["stage_c_approach1_ici_sdc"]
    self.assertEqual(len(stage_c_a1["culprit_cores"]), 1)
    self.assertEqual(stage_c_a1["culprit_cores"][0]["core"], slow_c)
    self.assertEqual(stage_c_a1["culprit_cores"][0]["s0_iterations"], [100, 130])
    self.assertTrue(stage_c_a1["culprit_cores"][0]["p_divides_s0_plus_1"])

  def test_synth_noise_only_and_sdk_dict_callback(self):
    noise_lines = [
        "I1021 12:00:01.000000 1 sdc.cc:2] [job.0x0x0_TC0@1]: "
        "ICI transfer wait on RECV flag timed out on 0x0x0_TC0 at iteration 17 "
        "on core with global core index 0",
        "I1021 12:00:01.000100 1 sdc.cc:2] [job.3x3x3_TC1@1]: "
        "IciSdcTest :: ICI BW: 162.0 GiB/s, time: 0.2 ms (global core index: 127) (delay iterations: 0)",
    ]
    res_n = analysis.analyze_parsed_logs(events.parse_lines(noise_lines))
    rep_n = report.build_straggler_report(res_n)
    self.assertEqual(rep_n["verdict"]["primary_cause"], "NOISE_ONLY")
    self.assertEqual(rep_n["verdict"]["suspect_cores"], [])

    # Direct dict callback parsing
    cb_evs = events.parse_sdk_callback_event(
        {
            "chip": (1, 0, 0),
            "core": 1,
            "message": ("ICI SDC Test transfer recv timeout (dynamic) : 2 -> 3 - " "4194304 bytes - iteration 27"),
        }
    )
    self.assertEqual(len(cb_evs), 1)
    self.assertEqual(cb_evs[0].coord, "1x0x0_TC1")
    self.assertEqual(cb_evs[0].iteration, 27)

  def test_cause_shares_decision_tree_and_cli(self):
    shares = report.compute_cause_shares(
        baseline_step_ms=1000.0,
        anomalous_step_ms=1500.0,
        stage_a_step_ms=1100.0,
        stage_b_step_ms=1050.0,
        has_slow_core=True,
    )
    self.assertAlmostEqual(shares["HOST_INPUT_PIPELINE"], 0.80, places=4)
    self.assertAlmostEqual(shares["HOST_CALLBACK_BARRIER"], 0.10, places=4)
    self.assertAlmostEqual(shares["TPU_SLOW_CORE"], 0.10, places=4)

    self.assertEqual(
        report.next_probe_decision(baseline_step_ms=1000.0, anomalous_step_ms=1050.0),
        "HEALTHY_NO_PROBE",
    )
    self.assertEqual(
        report.next_probe_decision(baseline_step_ms=1000.0, anomalous_step_ms=1500.0),
        "STAGE_A_HOST_TIMING",
    )
    self.assertEqual(
        report.next_probe_decision(
            baseline_step_ms=1000.0,
            anomalous_step_ms=1500.0,
            stage_a_ran=True,
            stage_a_step_ms=1490.0,
        ),
        "STAGE_B_METRICS",
    )
    self.assertEqual(
        report.next_probe_decision(
            baseline_step_ms=1000.0,
            anomalous_step_ms=1500.0,
            stage_a_ran=True,
            stage_a_step_ms=1490.0,
            stage_b_ran=True,
            stage_b_step_ms=1485.0,
        ),
        "STAGE_C_ICI_PROBE",
    )
    self.assertEqual(
        report.next_probe_decision(
            baseline_step_ms=1000.0,
            anomalous_step_ms=1500.0,
            stage_a_ran=True,
            stage_a_step_ms=1490.0,
            stage_b_ran=True,
            stage_b_step_ms=1485.0,
            stage_c_sflag_ran=True,
            stage_c_ici_ran=True,
        ),
        "STAGE_D_DIAG_STEPS",
    )
    self.assertEqual(
        report.next_probe_decision(
            baseline_step_ms=1000.0,
            anomalous_step_ms=1500.0,
            stage_a_ran=True,
            stage_a_step_ms=1490.0,
            stage_b_ran=True,
            stage_b_step_ms=1485.0,
            stage_c_sflag_ran=True,
            stage_c_sflag_culprits=["1x0x0_TC0"],
        ),
        "EMIT_REPORT",
    )

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
      json_path = tf.name
    try:
      rc = cli.main(
          [
              "--logs",
              os.path.join(FIXTURES_DIR, "cloud_p1_compiler_options.log"),
              "--json",
              json_path,
          ]
      )
      self.assertEqual(rc, 0)
      with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
      self.assertEqual(data["schema_version"], "1.0")
      self.assertEqual(data["verdict"]["primary_cause"], "TPU_SLOW_CORE")
      self.assertEqual(data["verdict"]["suspect_cores"], ["1x0x0_TC0"])
    finally:
      if os.path.exists(json_path):
        os.unlink(json_path)

  def test_trigger_config_defaults_and_from_dict(self):
    cfg = trigger.TriggerConfig()
    self.assertEqual(cfg.mode, "target")
    self.assertIsNone(cfg.target_step_ms)
    self.assertAlmostEqual(cfg.tolerance, 0.05)
    self.assertEqual(cfg.consecutive_steps, 5)
    self.assertEqual(cfg.warmup_steps, 10)
    self.assertEqual(cfg.window, 20)
    self.assertEqual(
        cfg.exclude,
        ("profiler", "checkpoint", "eval", "data_wait", "dropless_replay", "recompile"),
    )
    self.assertEqual(cfg.exclude_after_checkpoint_steps, 2)
    self.assertEqual(cfg.cooldown_steps, 50)

    mapped = trigger.TriggerConfig.from_dict(
        {
            "straggler_probe_trigger_mode": "both",
            "straggler_probe_target_step_ms": 200.0,
            "straggler_probe_tolerance": 0.08,
            "straggler_probe_slowdown_ratio_tolerance": 0.12,
            "straggler_probe_consecutive_steps": 4,
            "straggler_probe_warmup_steps": 6,
            "straggler_probe_window": 15,
            "straggler_probe_exclude": "profiler,checkpoint,eval",
            "straggler_probe_exclude_after_checkpoint_steps": 3,
            "straggler_probe_cooldown_steps": 25,
        }
    )
    self.assertEqual(mapped.mode, "both")
    self.assertEqual(mapped.target_step_ms, 200.0)
    self.assertAlmostEqual(mapped.tolerance, 0.08)
    self.assertAlmostEqual(mapped.effective_slowdown_tolerance, 0.12)
    self.assertEqual(mapped.consecutive_steps, 4)
    self.assertEqual(mapped.warmup_steps, 6)
    self.assertEqual(mapped.window, 15)
    self.assertEqual(mapped.exclude, ("profiler", "checkpoint", "eval"))
    self.assertEqual(mapped.exclude_after_checkpoint_steps, 3)
    self.assertEqual(mapped.cooldown_steps, 25)

  def test_trigger_target_mode_fires_after_n_slow_steps(self):
    trig = trigger.StepTimeTrigger(
        mode="target",
        target_step_ms=100.0,
        tolerance=0.05,
        consecutive_steps=5,
        warmup_steps=0,
    )
    # Healthy steps within 100 * 1.05 = 105.0 ms
    for s in range(3):
      d = trig.observe(s, 103.0)
      self.assertFalse(d.fired)
      self.assertEqual(d.cause, "none")
      self.assertEqual(d.streak_steps, [])

    # 4 slow steps followed by 1 healthy step resets the streak
    for s in range(3, 7):
      d = trig.observe(s, 120.0)
      self.assertFalse(d.fired)
    d_reset = trig.observe(7, 100.0)
    self.assertFalse(d_reset.fired)
    self.assertEqual(d_reset.streak_steps, [])

    # 5 consecutive clean slow steps fire on the 5th step; regression is device_step_ms - target
    decisions = [trig.observe(s, 130.0, data_wait_ms=5.0) for s in range(8, 13)]
    for d in decisions[:-1]:
      self.assertFalse(d.fired)
    final = decisions[-1]
    self.assertTrue(final.fired)
    self.assertEqual(final.mode, "target")
    self.assertEqual(final.cause, "DEVICE_SLOW")
    self.assertEqual(final.streak_steps, [8, 9, 10, 11, 12])
    self.assertEqual(final.reference_ms, 100.0)
    self.assertEqual(final.observed_ms, 125.0)
    self.assertEqual(final.regression_ms, 25.0)
    self.assertEqual(final.recommended_stage, "STAGE_A_HOST_TIMING")
    self.assertIn("STAGE_C_ICI_PROBE", final.recommended_stages)

  def test_trigger_exclusions_checkpoint_eval_profiler_and_streak_preservation(self):
    # Default exclude_after_checkpoint_steps=2 -> checkpoint at step k excludes k, k+1, k+2.
    trig = trigger.StepTimeTrigger(
        mode="target",
        target_step_ms=100.0,
        tolerance=0.05,
        consecutive_steps=4,
        warmup_steps=2,
    )
    # Steps 0 and 1 are warmup-excluded even though they are 500 ms.
    self.assertFalse(trig.observe(0, 500.0).fired)
    self.assertFalse(trig.observe(1, 500.0).fired)
    self.assertEqual(trig.excluded_steps[0], "warmup")
    self.assertEqual(trig.excluded_steps[1], "warmup")

    # Start a slow streak at steps 2 and 3 (2 of 4 needed).
    trig.observe(2, 140.0)
    d3 = trig.observe(3, 140.0)
    self.assertEqual(d3.streak_steps, [2, 3])

    # Step 4 saves a checkpoint (marked after observe, matching train.py L981 -> L986).
    trig.observe(4, 95.0)
    trig.mark(4, "checkpoint")
    # Steps 5 and 6 are excluded by the checkpoint at step 4 (even if fast or slow, streak stays [2, 3]).
    d5 = trig.observe(5, 90.0)
    d6 = trig.observe(6, 300.0)
    self.assertEqual(trig.excluded_steps[4], "checkpoint")
    self.assertEqual(trig.excluded_steps[5], "checkpoint")
    self.assertEqual(trig.excluded_steps[6], "checkpoint")
    self.assertFalse(d5.fired)
    self.assertFalse(d6.fired)
    self.assertEqual(d6.streak_steps, [2, 3])

    # Step 7 is a clean slow step -> streak becomes [2, 3, 7] (3 of 4 needed).
    d7 = trig.observe(7, 140.0)
    self.assertFalse(d7.fired)
    self.assertEqual(d7.streak_steps, [2, 3, 7])

    # Step 8 runs eval (marked after observe, matching train.py L981 -> L1001); excludes 8 and 9.
    trig.observe(8, 90.0)
    trig.mark(8, "eval")
    d9 = trig.observe(9, 400.0)
    self.assertEqual(trig.excluded_steps[8], "eval")
    self.assertEqual(trig.excluded_steps[9], "eval")
    self.assertFalse(d9.fired)
    self.assertEqual(d9.streak_steps, [2, 3, 7])

    # Step 10 hits profiler deactivation boundary; excludes 10 and 11.
    trig.observe(10, 95.0, tags=("profiler",))
    d11 = trig.observe(11, 250.0)
    self.assertEqual(trig.excluded_steps[10], "profiler")
    self.assertEqual(trig.excluded_steps[11], "profiler")
    self.assertFalse(d11.fired)
    self.assertEqual(d11.streak_steps, [2, 3, 7])

    # Step 12 has dropless_replay, Step 13 has recompile, Step 14 has custom tag.
    trig.observe(12, 85.0, tags=("dropless_replay",))
    trig.observe(13, 500.0, tags=("recompile",))
    d14 = trig.observe(14, 90.0, tags=("custom_gc_pause",))
    self.assertEqual(d14.streak_steps, [2, 3, 7])

    # Step 15 is the 4th clean slow step -> fires with streak [2, 3, 7, 15]!
    d15 = trig.observe(15, 140.0)
    self.assertTrue(d15.fired)
    self.assertEqual(d15.streak_steps, [2, 3, 7, 15])

    # Also verify configurable exclude_after_checkpoint_steps=1 only excludes k and k+1.
    trig_ckpt1 = trigger.StepTimeTrigger(
        mode="target",
        target_step_ms=100.0,
        consecutive_steps=2,
        warmup_steps=0,
        exclude_after_checkpoint_steps=1,
    )
    trig_ckpt1.observe(0, 100.0, tags=("checkpoint",))
    self.assertIn(0, trig_ckpt1.excluded_steps)
    self.assertIn(1, trig_ckpt1.excluded_steps)
    self.assertNotIn(2, trig_ckpt1.excluded_steps)

  def test_trigger_data_wait_subtraction_input_bound_stage_a_only(self):
    trig = trigger.StepTimeTrigger(
        mode="target",
        target_step_ms=100.0,
        tolerance=0.05,
        consecutive_steps=3,
        warmup_steps=0,
    )
    # step_ms=140.0 > 105.0, but data_wait_ms=45.0 so device_step_ms=95.0 <= 105.0.
    d0 = trig.observe(0, 140.0, data_wait_ms=45.0)
    d1 = trig.observe(1, 135.0, data_wait_ms=35.0)
    self.assertFalse(d0.fired)
    self.assertFalse(d1.fired)
    d2 = trig.observe(2, 150.0, data_wait_ms=52.0)
    self.assertTrue(d2.fired)
    self.assertEqual(d2.cause, "INPUT_BOUND")
    self.assertEqual(d2.streak_steps, [0, 1, 2])
    self.assertEqual(d2.recommended_stage, "STAGE_A_HOST_TIMING")
    self.assertEqual(d2.recommended_stages, ("STAGE_A_HOST_TIMING",))
    self.assertNotIn("STAGE_C_ICI_PROBE", d2.recommended_stages)
    self.assertNotIn("STAGE_D_DIAG_STEPS", d2.recommended_stages)

    # report.next_probe_decision also respects INPUT_BOUND and does not escalate to Stages C/D.
    self.assertEqual(
        report.next_probe_decision(
            baseline_step_ms=100.0,
            anomalous_step_ms=150.0,
            trigger_decision=d2,
        ),
        "STAGE_A_HOST_TIMING",
    )
    self.assertEqual(
        report.next_probe_decision(
            baseline_step_ms=100.0,
            anomalous_step_ms=150.0,
            stage_a_ran=True,
            stage_a_step_ms=145.0,
            trigger_decision=d2,
        ),
        "EMIT_REPORT",
    )

  def test_trigger_baseline_mode_with_warmup_and_rolling_median(self):
    trig = trigger.StepTimeTrigger(
        mode="baseline",
        slowdown_ratio_tolerance=0.10,
        consecutive_steps=3,
        warmup_steps=4,
        window=5,
    )
    # Warmup steps 0..3 (slow compilation/startup times are ignored).
    for s in range(4):
      d = trig.observe(s, 800.0)
      self.assertFalse(d.fired)
    self.assertIsNone(trig.baseline_median_ms)

    # 5 clean steady-state steps around 200 ms -> median is 200.0 ms, threshold is 220.0 ms.
    for s, ms in enumerate([198.0, 200.0, 202.0, 199.0, 201.0], start=4):
      d = trig.observe(s, ms)
      self.assertFalse(d.fired)
    self.assertAlmostEqual(trig.baseline_median_ms, 200.0, places=3)

    # 3 consecutive clean steps > 220.0 ms fire; baseline median remains uncontaminated at 200.0 ms.
    self.assertFalse(trig.observe(9, 250.0).fired)
    self.assertFalse(trig.observe(10, 255.0).fired)
    d11 = trig.observe(11, 260.0)
    self.assertTrue(d11.fired)
    self.assertEqual(d11.mode, "baseline")
    self.assertEqual(d11.cause, "DEVICE_SLOW")
    self.assertEqual(d11.reference_ms, 200.0)
    self.assertEqual(d11.observed_ms, 260.0)
    self.assertEqual(d11.regression_ms, 60.0)
    self.assertEqual(d11.streak_steps, [9, 10, 11])

  def test_trigger_cooldown_suppresses_and_expires(self):
    trig = trigger.StepTimeTrigger(
        mode="target",
        target_step_ms=100.0,
        tolerance=0.05,
        consecutive_steps=3,
        warmup_steps=0,
        cooldown_steps=4,
    )
    # Fire at step 2 (streak [0, 1, 2]).
    self.assertFalse(trig.observe(0, 130.0).fired)
    self.assertFalse(trig.observe(1, 130.0).fired)
    d2 = trig.observe(2, 130.0)
    self.assertTrue(d2.fired)

    # Cooldown covers steps 3, 4, 5, 6 (4 steps after step 2).
    for s in range(3, 7):
      d_cool = trig.observe(s, 150.0)
      self.assertFalse(d_cool.fired)
      self.assertTrue(d_cool.in_cooldown)

    # After cooldown expires at step 7, 3 consecutive slow steps (7, 8, 9) fire again at step 9.
    d7 = trig.observe(7, 130.0)
    self.assertFalse(d7.in_cooldown)
    self.assertFalse(d7.fired)
    self.assertFalse(trig.observe(8, 130.0).fired)
    d9 = trig.observe(9, 130.0)
    self.assertTrue(d9.fired)
    self.assertEqual(d9.streak_steps, [7, 8, 9])

  def test_trigger_both_mode_and_report_integration(self):
    # Case 1: baseline median (100 ms) is tighter than target (150 ms); fires via baseline.
    trig_base_fires = trigger.StepTimeTrigger(
        mode="both",
        target_step_ms=150.0,
        tolerance=0.05,
        slowdown_ratio_tolerance=0.05,
        consecutive_steps=3,
        warmup_steps=0,
        window=5,
    )
    for s in range(4):
      trig_base_fires.observe(s, 100.0)
    # 120 ms exceeds baseline (100 * 1.05 = 105) even though < target (150 * 1.05 = 157.5).
    self.assertFalse(trig_base_fires.observe(4, 120.0).fired)
    self.assertFalse(trig_base_fires.observe(5, 120.0).fired)
    d_both_base = trig_base_fires.observe(6, 120.0)
    self.assertTrue(d_both_base.fired)
    self.assertEqual(d_both_base.reference_ms, 100.0)

    # Case 2: target (100 ms) is tighter than baseline median (130 ms); fires via target.
    trig_target_fires = trigger.StepTimeTrigger(
        mode="both",
        target_step_ms=100.0,
        tolerance=0.05,
        slowdown_ratio_tolerance=0.50,
        consecutive_steps=3,
        warmup_steps=1,
        window=5,
    )
    trig_target_fires.observe(0, 300.0)  # warmup
    trig_target_fires.observe(1, 100.0)
    trig_target_fires.observe(2, 115.0)
    trig_target_fires.observe(3, 115.0)
    trig_target_fires.mark(4, "eval")
    trig_target_fires.observe(4, 400.0)
    trig_target_fires.observe(5, 250.0)  # excluded by eval at step 4
    d_both_target = trig_target_fires.observe(6, 118.0)
    self.assertTrue(d_both_target.fired)
    self.assertEqual(d_both_target.reference_ms, 100.0)
    self.assertEqual(d_both_target.observed_ms, 118.0)
    self.assertEqual(d_both_target.regression_ms, 18.0)

    # Verify report.build_straggler_report includes the trigger summary under "trigger".
    parsed = events.parse_paths([os.path.join(FIXTURES_DIR, "exp2_b_clean.log")])
    res = analysis.analyze_parsed_logs(parsed)
    rep = report.build_straggler_report(
        res,
        job_name="trigger_report_test",
        trigger_reason="step_time_trigger",
        trigger_summary=trig_target_fires.summary(),
    )
    trig_sec = rep["trigger"]
    self.assertEqual(trig_sec["mode"], "both")
    self.assertTrue(trig_sec["fired"])
    self.assertEqual(trig_sec["cause"], "DEVICE_SLOW")
    self.assertEqual(trig_sec["reference_ms"], 100.0)
    self.assertEqual(trig_sec["target_step_ms"], 100.0)
    self.assertEqual(trig_sec["observed_ms"], 118.0)
    self.assertEqual(trig_sec["regression_ms"], 18.0)
    self.assertEqual(trig_sec["streak_steps"], [2, 3, 6])
    self.assertEqual(trig_sec["excluded_steps"], {0: "warmup", 4: "eval", 5: "eval"})
    self.assertEqual(trig_sec["baseline_step_ms"], 100.0)
    self.assertEqual(trig_sec["anomalous_step_ms"], 118.0)
    self.assertEqual(trig_sec["delta_step_ms"], 18.0)


if __name__ == "__main__":
  unittest.main()
