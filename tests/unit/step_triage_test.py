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
    p2_a1 = rep["probes"]["p2_approach1_ici_sdc"]
    self.assertEqual(p2_a1["total_waves"], 2)
    culprit_map = {c["core"]: c for c in p2_a1["culprit_cores"]}
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
    self.assertEqual(rep["probes"]["p2_approach1_ici_sdc"]["total_waves"], 0)

  def test_fixture_xm_approach2_sc_and_tc_and_healthy(self):
    # 1. 2-host SparseCore delay on 1x0x0_TC0
    sc_path = os.path.join(FIXTURES_DIR, "xm_2h_c_delay_sc.log")
    res_sc = analysis.analyze_parsed_logs(events.parse_paths([sc_path]))
    rep_sc = report.build_straggler_report(res_sc, job_name="2h_c_delay_sc")
    self.assertEqual(rep_sc["verdict"]["primary_cause"], "TPU_SLOW_CORE")
    self.assertEqual(rep_sc["verdict"]["suspect_cores"], ["1x0x0_TC0"])
    a2_sc = rep_sc["probes"]["p2_approach2_sflag"]["culprit_cores"]
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
    a2_tc = rep_tc["probes"]["p2_approach2_sflag"]["culprit_cores"]
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
    a2 = rep["probes"]["p2_approach2_sflag"]["culprit_cores"]
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
    p2_a1 = rep["probes"]["p2_approach1_ici_sdc"]
    self.assertEqual(p2_a1["total_waves"], 3)
    culprit_cores = [c["core"] for c in p2_a1["culprit_cores"]]
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
        [c["core"] for c in rep["probes"]["p2_approach2_sflag"]["culprit_cores"]],
        ["1x0x0_TC0"],
    )
    self.assertGreaterEqual(rep["probes"]["p2_approach1_ici_sdc"]["total_waves"], 1)

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
        p0_step_ms=1100.0,
        p1_step_ms=1050.0,
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
        "RUN_P0_SYNTHETIC_DATA",
    )
    self.assertEqual(
        report.next_probe_decision(
            baseline_step_ms=1000.0,
            anomalous_step_ms=1500.0,
            p0_ran=True,
            p0_step_ms=1490.0,
            p1_ran=True,
            p1_step_ms=1485.0,
            p2_sflag_ran=True,
            p2_sflag_culprits=["1x0x0_TC0"],
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


if __name__ == "__main__":
  unittest.main()
