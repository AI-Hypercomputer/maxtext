# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Canonical Benchmark v2 schemas, Sheet v2 column specs, validation rules, and models matrix loader."""

from collections.abc import Iterator
from dataclasses import dataclass
import json
import math
import os
from typing import Any

SCHEMA_VERSION = 2

RUN_STATUS_CODES = ("OK", "FAILED", "NOT RUN", "SKIPPED", "BLOCKED", "PENDING", "N/A")
RESULT_JSON_STATUS_CODES = ("OK", "FAILED", "NOT_RUN", "SKIPPED")
FAILURE_CATEGORY_CODES = ("OOM-HBM", "OOM-HOST", "TIMEOUT", "ERROR", "NOT-TPU", "PREEMPTED")
STEP1_VALUE_SOURCES = ("cold_download", "warm_cache", "copied_from_other_run", "local_synthetic_files")
DEEPSEEK_BF16_SOURCES = ("converted_by_this_benchmark", "copied_from_maxtext_team_bucket")
CACHE_STATE_CODES = ("empty", "warm")
PIPELINES = ("to_maxtext", "safetensors_dynamic")
SCAN_MODES = ("scanned", "unscanned")
PLATFORMS = ("v7x", "v6e", "v5e")

NON_NUMERIC_CODES = frozenset(
    {
        "",
        "N/A",
        "LOCAL",
        "NOT RUN",
        "BLOCKED",
        "SKIPPED",
        "NOT-TPU",
        "OOM-HBM",
        "OOM-HOST",
        "TIMEOUT",
        "ERROR",
        "PREEMPTED",
        "PENDING",
        "—",
    }
)

NUMERIC_FORMATS = frozenset({"s", "gb", "gbps", "x", "pct", "n", "ratio"})

MODELS_V2_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "models_v2.json")


@dataclass(frozen=True)
class ColumnSpec:
  """One column definition shared across v2 JSON records and Sheet v2 tabs."""

  header: str
  key: str
  fmt: str
  group: str
  definition: str
  source: str


def _c(  # pylint: disable=too-many-positional-arguments
    header: str, key: str, fmt: str, group: str, definition: str, source: str
) -> ColumnSpec:
  return ColumnSpec(header=header, key=key, fmt=fmt, group=group, definition=definition, source=source)


SUMMARY_COLUMNS: tuple[ColumnSpec, ...] = (
    _c("Model", "model_name", "t", "identity", "MaxText model name.", "Run config"),
    _c("Model tier", "model_test_tier", "c", "identity", "Model tier: S, A, B or C.", "Models tab"),
    _c("Pipeline (to_maxtext or safetensors_dynamic)", "pipeline_name", "t", "identity", "Pipeline.", "Run config"),
    _c("Scan layers (scanned or unscanned)", "scan_layers_mode", "c", "identity", "Scan mode.", "Run config"),
    _c("TPU platform", "tpu_platform", "c", "identity", "TPU generation: v5e, v6e or v7x.", "Platform profile"),
    _c("Number of slices", "number_of_slices", "n", "identity", "TPU slices used.", "Run config"),
    _c("Slice topology", "slice_topology", "c", "identity", "Topology per slice.", "Run config"),
    _c("Hosts per slice", "hosts_per_slice", "n", "identity", "Hosts in each slice.", "Platform profile"),
    _c("TPU devices per slice", "tpu_devices_per_slice", "n", "identity", "Devices per slice.", "Platform profile"),
    _c("Total TPU devices", "total_tpu_devices", "n", "identity", "Slices x devices per slice.", "Platform profile"),
    _c("Run status", "run_status", "c", "status", "OK or Legend status/failure code.", "Runs tab"),
    _c("HF checkpoint dtype (as downloaded)", "hf_checkpoint_dtype", "c", "checkpoint", "HF file dtype.", "Headers"),
    _c(
        "Loaded file dtype (after conversion, if any)",
        "loaded_checkpoint_dtype",
        "c",
        "checkpoint",
        "Loaded dtype.",
        "Headers",
    ),
    _c(
        "MaxText weight dtype (weight_dtype)",
        "maxtext_load_weight_dtype",
        "c",
        "checkpoint",
        "MaxText dtype.",
        "Config",
    ),
    _c("Weights (real or synthetic)", "checkpoint_weights_kind", "c", "checkpoint", "real in v2.", "Run config"),
    _c("Loaded checkpoint size (GB)", "checkpoint_size_gb", "gb", "checkpoint", "Loaded size in GB.", "Headers"),
    _c(
        "Load ran on (JAX backend and TPU device kind)",
        "load_backend_and_tpu_device_kind",
        "t",
        "checkpoint",
        "Backend and TPU device kind from pre-load TPU check.",
        "Runs tab",
    ),
    _c("Step 1: download HF → GCS (seconds)", "step1_total_seconds", "s", "step1", "Cold Step 1 total.", "Step 1"),
    _c(
        "DeepSeek only, Step 1b: convert FP8/FP4 → BF16 (seconds)",
        "step1b_deepseek_total_seconds",
        "s",
        "step1",
        "DeepSeek Step 1b total.",
        "Step 1",
    ),
    _c(
        "to_maxtext only, Step 1: copy GCS → local disk (seconds)",
        "to_maxtext_step1_total_seconds",
        "s",
        "pipeline",
        "to_maxtext Step 1 GCS-to-local copy total.",
        "Stages: to_maxtext",
    ),
    _c(
        "to_maxtext only, Step 2: convert to Orbax on CPU (seconds)",
        "to_maxtext_step2_total_seconds",
        "s",
        "pipeline",
        "to_maxtext Step 2 CPU conversion total.",
        "Stages: to_maxtext",
    ),
    _c(
        "Load into TPU HBM: to_maxtext Step 3 or safetensors_dynamic Step 2 (seconds)",
        "load_into_tpu_hbm_total_seconds",
        "s",
        "pipeline",
        "HBM load step total.",
        "Stages tabs",
    ),
    _c("Setup before load (seconds)", "setup_total_before_load_seconds", "s", "setup", "Setup total.", "Stages tabs"),
    _c(
        "Total without the HF download (seconds)",
        "total_without_hf_download_seconds",
        "s",
        "totals",
        "Total excluding cold HF download.",
        "Derived",
    ),
    _c("Total end-to-end (seconds)", "total_end_to_end_seconds", "s", "totals", "End-to-end total.", "Derived"),
    _c(
        "Speedup end-to-end (to_maxtext ÷ safetensors_dynamic)",
        "speedup_end_to_end_to_maxtext_vs_safetensors_dynamic",
        "x",
        "totals",
        "End-to-end speedup ratio.",
        "Derived",
    ),
    _c(
        "Speedup without the HF download (to_maxtext ÷ safetensors_dynamic)",
        "speedup_without_hf_download_to_maxtext_vs_safetensors_dynamic",
        "x",
        "totals",
        "No-HF speedup ratio.",
        "Derived",
    ),
    _c(
        "Speedup load only (to_maxtext Step 3 ÷ safetensors_dynamic Step 2)",
        "speedup_load_only_to_maxtext_step3_vs_safetensors_dynamic_step2",
        "x",
        "totals",
        "Load-only speedup ratio.",
        "Derived",
    ),
    _c(
        "Peak host RAM, max over hosts (GB)",
        "memory_peak_host_ram_container_gb",
        "gb",
        "memory",
        "Peak container host RAM over hosts.",
        "Stages tabs",
    ),
    _c(
        "Peak TPU HBM per device (GB)",
        "memory_tpu_hbm_peak_per_device_gb",
        "gb",
        "memory",
        "Peak TPU HBM per device.",
        "Stages tabs",
    ),
    _c("Repetitions (count)", "repetitions_count", "n", "repetitions", "Run repetition count.", "Runs tab"),
    _c(
        "Load time range, min–max (seconds)",
        "load_time_range_min_max_seconds",
        "c",
        "repetitions",
        "Min-max load range text.",
        "Runs tab",
    ),
    _c("Log link", "log_url", "url", "links", "Deciding step log link.", "Runs tab"),
    _c("Note", "note", "note", "links", "Exception/status notes.", "Sync script"),
    _c("Row ID", "row_id", "t", "links", "Shared row identifier.", "Sync script"),
)

STEP1_COLUMNS: tuple[ColumnSpec, ...] = (
    _c("TPU platform", "tpu_platform", "c", "identity", "Platform served.", "Platform profile"),
    _c("GCS bucket region", "step1_gcs_bucket_region", "c", "identity", "Target regional bucket region.", "Step 1"),
    _c("Model", "model_name", "t", "identity", "MaxText model name.", "Step 1"),
    _c("Step 1 · HF repo", "step1_hf_repo_id", "t", "step1", "HF repo downloaded.", "Step 1"),
    _c(
        "Step 1 · List HF repo files (seconds)",
        "step1_list_hf_repo_files_seconds",
        "s",
        "step1",
        "Repo list.",
        "Step 1",
    ),
    _c(
        "Step 1 · Files already in GCS before the download (count)",
        "step1_files_already_in_gcs_count",
        "n",
        "step1",
        "Pre-existing GCS file count (must be 0 for cold download).",
        "Step 1",
    ),
    _c(
        "Step 1 · Download HF → GCS (seconds)",
        "step1_download_hf_to_gcs_seconds",
        "s",
        "step1",
        "Download s.",
        "Step 1",
    ),
    _c("Step 1 · Download HF → GCS (GB)", "step1_download_hf_to_gcs_gb", "gb", "step1", "Downloaded GB.", "Step 1"),
    _c(
        "Step 1 · Download HF → GCS (GB/s)",
        "step1_download_hf_to_gcs_gb_per_second",
        "gbps",
        "step1",
        "Download GB/s.",
        "Step 1",
    ),
    _c("Step 1 · Number of files (count)", "step1_number_of_files", "n", "step1", "Files count.", "Step 1"),
    _c(
        "Step 1 · Slowest file download (seconds)",
        "step1_slowest_file_download_seconds",
        "s",
        "step1",
        "Slowest shard download.",
        "Step 1",
    ),
    _c("Step 1 · Download workers (count)", "step1_download_workers_count", "n", "step1", "Workers.", "Step 1"),
    _c("Step 1 · Host machine type", "step1_host_machine_type", "t", "step1", "CPU machine type.", "Step 1"),
    _c(
        "Step 1 · Total: list + download (seconds)", "step1_total_seconds", "s", "step1", "List + download s.", "Step 1"
    ),
    _c("Step 1 · Value source", "step1_value_source", "t", "step1", "cold_download or legacy source.", "Step 1"),
    _c("Step 1 · Measured by run ID", "step1_measured_by_run_id", "t", "step1", "Download job run ID.", "Step 1"),
    _c(
        "DeepSeek Step 1b · Convert FP8/FP4 → BF16 (seconds)",
        "step1b_deepseek_convert_fp8_fp4_to_bf16_seconds",
        "s",
        "step1",
        "CPU dequantization time.",
        "Step 1b",
    ),
    _c(
        "DeepSeek Step 1b · Convert input, FP8/FP4 (GB)",
        "step1b_deepseek_convert_input_gb",
        "gb",
        "step1",
        "Input GB.",
        "Step 1b",
    ),
    _c(
        "DeepSeek Step 1b · Convert output, BF16 (GB)",
        "step1b_deepseek_convert_output_gb",
        "gb",
        "step1",
        "Output GB.",
        "Step 1b",
    ),
    _c(
        "DeepSeek Step 1b · Upload BF16 to GCS (seconds)",
        "step1b_deepseek_upload_bf16_to_gcs_seconds",
        "s",
        "step1",
        "BF16 upload time.",
        "Step 1b",
    ),
    _c(
        "DeepSeek Step 1b · Copy BF16 to this region (seconds)",
        "step1b_deepseek_copy_bf16_to_region_seconds",
        "s",
        "step1",
        "Cross-region BF16 copy time.",
        "Step 1b",
    ),
    _c(
        "DeepSeek Step 1b · Host machine type",
        "step1b_deepseek_host_machine_type",
        "t",
        "step1",
        "Step 1b VM.",
        "Step 1b",
    ),
    _c("DeepSeek Step 1b · BF16 source", "step1b_deepseek_bf16_source", "t", "step1", "BF16 provenance.", "Step 1b"),
    _c(
        "DeepSeek Step 1b · Total (seconds)", "step1b_deepseek_total_seconds", "s", "step1", "Step 1b total.", "Step 1b"
    ),
)


def _stage_identity_columns() -> tuple[ColumnSpec, ...]:
  return (
      _c("Row ID", "row_id", "t", "identity", "Summary Row ID.", "Sync script"),
      _c("TPU platform", "tpu_platform", "c", "identity", "v5e, v6e or v7x.", "Platform profile"),
      _c("Model", "model_name", "t", "identity", "MaxText model name.", "Run config"),
      _c("Scan layers", "scan_layers_mode", "c", "identity", "scanned or unscanned.", "Run config"),
      _c("Number of slices", "number_of_slices", "n", "identity", "TPU slices.", "Run config"),
      _c("Slice topology", "slice_topology", "c", "identity", "Slice topology.", "Run config"),
  )


def _setup_columns(pipeline: str) -> tuple[ColumnSpec, ...]:
  """Builds common Setup/Teardown stage column specs for the requested pipeline."""
  cols = [
      ("Setup · Process start to harness code (seconds)", "setup_process_start_to_harness_code_seconds", "Startup."),
      ("Setup · Python imports (seconds)", "setup_python_imports_seconds", "Imports."),
      ("Setup · jax.distributed initialize (seconds)", "setup_jax_distributed_initialize_seconds", "Rendezvous."),
      ("Setup · TPU backend initialize (seconds)", "setup_tpu_backend_initialize_seconds", "TPU init."),
      ("Setup · TPU check (seconds)", "setup_tpu_check_seconds", "TPU check."),
      ("Setup · MaxText config build (seconds)", "setup_maxtext_config_build_seconds", "pyconfig."),
      ("Setup · Device mesh build (seconds)", "setup_device_mesh_build_seconds", "Mesh."),
      ("Setup · Abstract model build (seconds)", "setup_abstract_model_build_seconds", "Abstract model."),
      ("Setup · Abstract train state build (seconds)", "setup_abstract_train_state_build_seconds", "Abstract state."),
  ]
  if pipeline == "safetensors_dynamic":
    cols.append(("Setup · Delete old GCS cache (seconds)", "setup_delete_old_gcs_cache_seconds", "Cache cleanup."))
  cols += [
      (
          "Setup · Wait for all hosts before load (seconds)",
          "setup_wait_for_all_hosts_before_load_seconds",
          "Pre-load sync.",
      ),
      ("Setup · Total before load (seconds)", "setup_total_before_load_seconds", "Total setup."),
      ("Teardown · Write results and exit (seconds)", "teardown_write_results_and_exit_seconds", "Teardown."),
      ("Phase · Wall clock of the whole phase process (seconds)", "phase_wall_clock_seconds", "Phase wall time."),
  ]
  return tuple(_c(h, k, "s", "setup", d, "Stage recorder") for h, k, d in cols)


_SD_PIPELINE_SPECS = (
    ("Step 1 · Verify GCS cache (seconds)", "step1_verify_gcs_cache_seconds", "s"),
    ("Step 1 · Other hosts wait (seconds)", "step1_other_hosts_wait_seconds", "s"),
    ("Step 1 · Total (seconds)", "step1_total_seconds", "s"),
    ("Step 2 · Build HF → MaxText mappings (seconds)", "step2_build_hf_to_maxtext_mappings_seconds", "s"),
    ("Step 2 · Read file headers and plan (seconds)", "step2_read_file_headers_and_plan_seconds", "s"),
    ("Step 2 · Read from GCS to host RAM (seconds)", "step2_read_from_gcs_to_host_ram_seconds", "s"),
    ("Step 2 · Read from GCS to host RAM (GB)", "step2_read_from_gcs_to_host_ram_gb", "gb"),
    ("Step 2 · Read from GCS to host RAM (GB/s)", "step2_read_from_gcs_to_host_ram_gb_per_second", "gbps"),
    ("Step 2 · Bytes needed by this host (GB)", "step2_bytes_needed_by_this_host_gb", "gb"),
    ("Step 2 · Bytes read ÷ bytes needed (ratio)", "step2_bytes_read_divided_by_bytes_needed_ratio", "ratio"),
    ("Step 2 · Copy host RAM → TPU HBM (seconds)", "step2_copy_host_ram_to_tpu_hbm_seconds", "s"),
    ("Step 2 · Wait for other hosts during load (seconds)", "step2_wait_for_other_hosts_during_load_seconds", "s"),
    ("Step 2 · Load total (seconds)", "step2_load_total_seconds", "s"),
    ("Step 2 · Transform hooks (seconds)", "step2_transform_hooks_seconds", "s"),
    ("Step 2 · Reshard to target layout (seconds)", "step2_reshard_to_target_layout_seconds", "s"),
    ("Step 2 · Stack layers and experts (seconds)", "step2_stack_layers_and_experts_seconds", "s"),
    ("Step 2 · Cast to load dtype (seconds)", "step2_cast_to_load_dtype_seconds", "s"),
    ("Step 2 · Gather-to-host fallback (seconds)", "step2_gather_to_host_fallback_seconds", "s"),
    ("Step 2 · Gather-to-host fallback (count)", "step2_gather_to_host_fallback_count", "n"),
    ("Step 2 · XLA compile (seconds)", "step2_xla_compile_seconds", "s"),
    ("Step 2 · Slowest parameter (seconds)", "step2_slowest_parameter_seconds", "s"),
    ("Step 2 · Slowest parameter (name)", "step2_slowest_parameter_name", "t"),
    ("Step 2 · Transform total (seconds)", "step2_transform_total_seconds", "s"),
    ("Step 2 · Harness memory cleanup (seconds)", "step2_harness_memory_cleanup_seconds", "s"),
    ("Step 2 · Final weight check and sync (seconds)", "step2_final_weight_check_and_sync_seconds", "s"),
    ("Step 2 · Total (seconds)", "step2_total_seconds", "s"),
)

_TM_PIPELINE_SPECS = (
    ("Step 1 · Verify GCS cache (seconds)", "step1_verify_gcs_cache_seconds", "s"),
    ("Step 1 · Copy GCS → local ramdisk (seconds)", "step1_copy_gcs_to_local_ramdisk_seconds", "s"),
    ("Step 1 · Copy GCS → local ramdisk (GB)", "step1_copy_gcs_to_local_ramdisk_gb", "gb"),
    ("Step 1 · Copy GCS → local ramdisk (GB/s)", "step1_copy_gcs_to_local_ramdisk_gb_per_second", "gbps"),
    ("Step 1 · Local copy mode", "step1_local_copy_mode", "t"),
    ("Step 1 · Cleanup (seconds)", "step1_cleanup_seconds", "s"),
    ("Step 1 · Total (seconds)", "step1_total_seconds", "s"),
    ("Step 2 on CPU · Startup and config (seconds)", "step2_startup_and_config_seconds", "s"),
    ("Step 2 on CPU · Build abstract model (seconds)", "step2_build_abstract_model_seconds", "s"),
    ("Step 2 on CPU · Read HF tensors to host RAM (seconds)", "step2_read_hf_tensors_to_host_ram_seconds", "s"),
    ("Step 2 on CPU · Read HF tensors to host RAM (GB)", "step2_read_hf_tensors_to_host_ram_gb", "gb"),
    ("Step 2 on CPU · Stream files from GCS (seconds)", "step2_stream_files_from_gcs_seconds", "s"),
    ("Step 2 on CPU · Stream files from GCS (GB)", "step2_stream_files_from_gcs_gb", "gb"),
    ("Step 2 on CPU · Transform hooks (seconds)", "step2_transform_hooks_seconds", "s"),
    ("Step 2 on CPU · Stack layers and experts (seconds)", "step2_stack_layers_and_experts_seconds", "s"),
    ("Step 2 on CPU · Place on CPU devices (seconds)", "step2_place_on_cpu_devices_seconds", "s"),
    ("Step 2 on CPU · Garbage collection (seconds)", "step2_garbage_collection_seconds", "s"),
    ("Step 2 on CPU · Transform total (seconds)", "step2_transform_total_seconds", "s"),
    ("Step 2 on CPU · Orbax save: copy device → host (seconds)", "step2_orbax_save_copy_device_to_host_seconds", "s"),
    ("Step 2 on CPU · Orbax save: write to GCS (seconds)", "step2_orbax_save_write_to_gcs_seconds", "s"),
    ("Step 2 on CPU · Orbax save: merge OCDBT (seconds)", "step2_orbax_save_merge_ocdbt_seconds", "s"),
    ("Step 2 on CPU · Orbax save: commit (seconds)", "step2_orbax_save_commit_seconds", "s"),
    ("Step 2 on CPU · Orbax save: written (GB)", "step2_orbax_save_written_gb", "gb"),
    ("Step 2 on CPU · Orbax save (GB/s)", "step2_orbax_save_gb_per_second", "gbps"),
    ("Step 2 on CPU · Orbax save total (seconds)", "step2_orbax_save_total_seconds", "s"),
    ("Step 2 on CPU · Re-copy local files (seconds)", "step2_recopy_local_files_seconds", "s"),
    ("Step 2 on CPU · Host machine type", "step2_host_machine_type", "t"),
    ("Step 2 on CPU · Total (seconds)", "step2_total_seconds", "s"),
    ("Step 3 · Read Orbax metadata (seconds)", "step3_read_orbax_metadata_seconds", "s"),
    ("Step 3 · Plan read (seconds)", "step3_plan_read_seconds", "s"),
    ("Step 3 · Read from GCS to TPU HBM (seconds)", "step3_read_from_gcs_to_tpu_hbm_seconds", "s"),
    ("Step 3 · Read from GCS to TPU HBM (GB)", "step3_read_from_gcs_to_tpu_hbm_gb", "gb"),
    ("Step 3 · Read from GCS to TPU HBM (GB/s)", "step3_read_from_gcs_to_tpu_hbm_gb_per_second", "gbps"),
    ("Step 3 · Wait for other hosts inside Orbax (seconds)", "step3_wait_for_other_hosts_inside_orbax_seconds", "s"),
    ("Step 3 · Post-load checks (seconds)", "step3_post_load_checks_seconds", "s"),
    ("Step 3 · Final check and sync (seconds)", "step3_final_check_and_sync_seconds", "s"),
    ("Step 3 · Total (seconds)", "step3_total_seconds", "s"),
)


def _pipeline_columns(pipeline: str) -> tuple[ColumnSpec, ...]:
  specs = _SD_PIPELINE_SPECS if pipeline == "safetensors_dynamic" else _TM_PIPELINE_SPECS
  return tuple(_c(f"{pipeline} · {h}", f"{pipeline}_{k}", fmt, "pipeline", h, "Stage recorder") for h, k, fmt in specs)


def _memory_columns(pipeline: str) -> tuple[ColumnSpec, ...]:
  """Builds Memory stage column specs (adding CPU Step 2 RAM columns for `to_maxtext`)."""
  cols: list[ColumnSpec] = []
  if pipeline == "to_maxtext":
    cols.extend(
        [
            _c(
                "Memory · Step 2 on CPU · Peak host RAM, container (GB)",
                "memory_to_maxtext_step2_peak_host_ram_container_gb",
                "gb",
                "memory",
                "Peak container RAM during Step 2.",
                "RAM sampler",
            ),
            _c(
                "Memory · Step 2 on CPU · Peak process RSS (GB)",
                "memory_to_maxtext_step2_peak_process_rss_gb",
                "gb",
                "memory",
                "Peak RSS during Step 2.",
                "RAM sampler",
            ),
            _c(
                "Memory · Step 2 on CPU · Host RAM size (GB)",
                "memory_to_maxtext_step2_host_ram_size_gb",
                "gb",
                "memory",
                "Step 2 host RAM capacity.",
                "RAM sampler",
            ),
        ]
    )
  cols.extend(
      [
          _c(
              "Memory · Load · Peak host RAM, container, max over hosts (GB)",
              "memory_peak_host_ram_container_gb",
              "gb",
              "memory",
              "Peak container RAM during load.",
              "RAM sampler",
          ),
          _c(
              "Memory · Load · Host with the peak host RAM (name)",
              "memory_peak_host_ram_host_name",
              "t",
              "memory",
              "Host hitting peak RAM.",
              "RAM sampler",
          ),
          _c(
              "Memory · Load · Peak process RSS (GB)",
              "memory_peak_process_rss_gb",
              "gb",
              "memory",
              "Peak process RSS during load.",
              "RAM sampler",
          ),
          _c(
              "Memory · Load · Host RAM size (GB)",
              "memory_host_ram_size_gb",
              "gb",
              "memory",
              "Host RAM size.",
              "RAM sampler",
          ),
          _c(
              "Memory · Load · TPU HBM usable per device (GB)",
              "memory_tpu_hbm_usable_per_device_gb",
              "gb",
              "memory",
              "Usable HBM per device.",
              "memory_stats()",
          ),
          _c(
              "Memory · Load · TPU HBM used after read, per device (GB)",
              "memory_tpu_hbm_used_after_read_per_device_gb",
              "gb",
              "memory",
              "HBM used after read.",
              "memory_stats()",
          ),
          _c(
              "Memory · Load · TPU HBM peak per device (GB)",
              "memory_tpu_hbm_peak_per_device_gb",
              "gb",
              "memory",
              "Peak HBM per device.",
              "memory_stats()",
          ),
          _c(
              "Memory · Load · TPU HBM used after load, per device (GB)",
              "memory_tpu_hbm_used_after_load_per_device_gb",
              "gb",
              "memory",
              "HBM used after load.",
              "memory_stats()",
          ),
          _c(
              "Memory · Load · TPU HBM peak, % of usable (percent)",
              "memory_tpu_hbm_peak_percent_of_usable",
              "pct",
              "memory",
              "Peak HBM % of usable.",
              "memory_stats()",
          ),
      ]
  )
  return tuple(cols)


def _totals_columns(pipeline: str) -> tuple[ColumnSpec, ...]:
  if pipeline != "to_maxtext":
    return ()
  return (
      _c(
          "to_maxtext · Total end-to-end, old formula without the local copy (seconds)",
          "to_maxtext_total_end_to_end_old_formula_seconds",
          "s",
          "totals",
          "Legacy total without local copy.",
          "Derived",
      ),
      _c(
          "to_maxtext · Load only = Step 3 (seconds)",
          "to_maxtext_load_only_seconds",
          "s",
          "totals",
          "Step 3 load total.",
          "Derived",
      ),
  )


STAGES_SD_COLUMNS: tuple[ColumnSpec, ...] = (
    _stage_identity_columns()
    + _setup_columns("safetensors_dynamic")
    + _pipeline_columns("safetensors_dynamic")
    + _memory_columns("safetensors_dynamic")
)
STAGES_TM_COLUMNS: tuple[ColumnSpec, ...] = (
    _stage_identity_columns()
    + _setup_columns("to_maxtext")
    + _pipeline_columns("to_maxtext")
    + _memory_columns("to_maxtext")
    + _totals_columns("to_maxtext")
)

RUNS_COLUMNS: tuple[ColumnSpec, ...] = (
    _c("Row ID", "row_id", "t", "identity", "Summary Row ID.", "Sync script"),
    _c("Phase", "phase_name", "t", "identity", "Phase name.", "Result JSON"),
    _c("Run ID", "run_id", "t", "identity", "Run ID.", "Result JSON"),
    _c("Experiment name", "experiment_name", "t", "identity", "Experiment name.", "Result JSON"),
    _c("Repetition index", "repetition_index", "n", "repetitions", "Repetition 1..N.", "Result JSON"),
    _c("Queue attempt number", "queue_attempt_number", "n", "identity", "Queue attempt.", "Result JSON"),
    _c("JobSet restart attempt", "jobset_restart_attempt", "n", "identity", "JobSet restart count.", "Downward API"),
    _c("Phase started (UTC)", "phase_started_at_utc", "ts", "identity", "Start timestamp UTC.", "Result JSON"),
    _c("Phase finished (UTC)", "phase_finished_at_utc", "ts", "identity", "Finish timestamp UTC.", "Result JSON"),
    _c("TPU platform", "tpu_platform", "c", "identity", "v5e, v6e or v7x.", "Env"),
    _c("GKE cluster", "gke_cluster_name", "t", "identity", "GKE cluster.", "Env"),
    _c("GCP project", "gcp_project_id", "t", "identity", "GCP project.", "Env"),
    _c("GCP region", "gcp_region", "t", "identity", "GCP region.", "Env"),
    _c("GCP zone", "gcp_zone", "t", "identity", "GCP zone.", "Env"),
    _c("GKE version", "gke_version", "t", "identity", "GKE version.", "Env"),
    _c("GKE node pool", "gke_node_pool_name", "t", "identity", "Node pool.", "Env"),
    _c("Capacity type (spot, on_demand or reserved)", "capacity_type", "c", "identity", "Capacity type.", "Env"),
    _c("Kueue LocalQueue", "kueue_local_queue_name", "t", "identity", "LocalQueue.", "Env"),
    _c("Host machine type", "host_machine_type", "t", "identity", "Host machine type.", "Env"),
    _c("TPU accelerator type", "tpu_accelerator_type", "t", "identity", "TPU accelerator label.", "Env"),
    _c("Number of slices", "number_of_slices", "n", "identity", "Slices.", "Profile"),
    _c("Slice topology", "slice_topology", "c", "identity", "Topology.", "Profile"),
    _c("Hosts per slice", "hosts_per_slice", "n", "identity", "Hosts per slice.", "Profile"),
    _c("TPU chips per host", "tpu_chips_per_host", "n", "identity", "Chips per host.", "Profile"),
    _c(
        "TPU devices per chip (TensorCores)",
        "tpu_devices_per_chip",
        "n",
        "identity",
        "TensorCores per chip.",
        "Profile",
    ),
    _c("TPU devices per slice", "tpu_devices_per_slice", "n", "identity", "Devices per slice.", "Profile"),
    _c("Total hosts", "total_hosts", "n", "identity", "Total hosts.", "Profile"),
    _c("Total TPU devices", "total_tpu_devices", "n", "identity", "Total TPU devices.", "Profile"),
    _c("Interconnect (ICI, or ICI + DCN)", "interconnect_type", "c", "identity", "Interconnect.", "Profile"),
    _c("Total TPU HBM (GB)", "total_tpu_hbm_gb", "gb", "identity", "Total TPU HBM GB.", "Profile"),
    _c("JAX backend platform (measured)", "jax_backend_platform", "c", "identity", "Measured backend.", "TPU check"),
    _c("TPU device kind (as JAX reports it)", "tpu_device_kind", "c", "identity", "Measured device kind.", "TPU check"),
    _c("Total TPU devices (measured)", "total_tpu_devices_measured", "n", "identity", "Measured devices.", "TPU check"),
    _c(
        "TPU devices per host (measured)",
        "tpu_devices_per_host_measured",
        "n",
        "identity",
        "Devices/host.",
        "TPU check",
    ),
    _c(
        "TPU HBM per device, measured (GB)",
        "tpu_hbm_per_device_gb_measured",
        "gb",
        "identity",
        "Usable HBM.",
        "TPU check",
    ),
    _c("Host RAM, measured (GB)", "host_ram_gb_measured", "gb", "identity", "Measured host RAM.", "Result JSON"),
    _c("Host vCPUs (count)", "host_vcpu_count", "n", "identity", "Host vCPUs.", "Result JSON"),
    _c("Container image", "container_image", "t", "setup", "Container image.", "Env"),
    _c("Python version", "python_version", "c", "setup", "Python version.", "Metadata"),
    _c("jax version", "jax_version", "c", "setup", "jax version.", "Metadata"),
    _c("jaxlib version", "jaxlib_version", "c", "setup", "jaxlib version.", "Metadata"),
    _c("libtpu version", "libtpu_version", "c", "setup", "libtpu version.", "Metadata"),
    _c("flax version", "flax_version", "c", "setup", "flax version.", "Metadata"),
    _c("orbax-checkpoint version", "orbax_checkpoint_version", "c", "setup", "orbax-checkpoint version.", "Metadata"),
    _c("transformers version", "transformers_version", "c", "setup", "transformers version.", "Metadata"),
    _c("safetensors version", "safetensors_version", "c", "setup", "safetensors version.", "Metadata"),
    _c("numpy version", "numpy_version", "c", "setup", "numpy version.", "Metadata"),
    _c("Git branch", "git_branch_name", "t", "setup", "Git branch.", "CODE_VERSION.json"),
    _c("Git commit", "git_commit", "t", "setup", "Git commit.", "CODE_VERSION.json"),
    _c(
        "Git working tree clean (TRUE or FALSE)",
        "git_working_tree_clean",
        "bool",
        "setup",
        "Clean flag.",
        "CODE_VERSION.json",
    ),
    _c("Code tarball SHA-256", "code_tarball_sha256", "t", "setup", "Tarball SHA-256.", "CODE_VERSION.json"),
    _c("Active timing wrappers", "active_timing_wrappers", "t", "setup", "Active wrappers.", "Stage recorder"),
    _c("Model", "model_name", "t", "identity", "Model name.", "Run config"),
    _c("HF repo", "hf_repo_id", "t", "identity", "HF repo.", "Run config"),
    _c(
        "Parameters (billions)", "model_parameter_count_billions", "gb", "identity", "Billions of params.", "Models tab"
    ),
    _c("Architecture (dense or moe)", "model_architecture", "c", "identity", "dense or moe.", "Models tab"),
    _c("Model tier", "model_test_tier", "c", "identity", "S, A, B or C.", "Models tab"),
    _c(
        "Scan layers (scanned or unscanned)", "scan_layers_mode", "c", "identity", "scanned or unscanned.", "Run config"
    ),
    _c("Parameter leaves (count)", "number_of_parameter_leaves", "n", "identity", "Leaf count.", "Result JSON"),
    _c("HF checkpoint dtype (as downloaded)", "hf_checkpoint_dtype", "c", "checkpoint", "HF dtype.", "Headers"),
    _c(
        "Loaded file dtype (after conversion, if any)",
        "loaded_checkpoint_dtype",
        "c",
        "checkpoint",
        "Loaded dtype.",
        "Headers",
    ),
    _c(
        "MaxText weight dtype (weight_dtype)",
        "maxtext_load_weight_dtype",
        "c",
        "checkpoint",
        "MaxText dtype.",
        "Config",
    ),
    _c("to_maxtext Orbax save dtype", "to_maxtext_orbax_save_dtype", "c", "checkpoint", "Orbax save dtype.", "Step 2"),
    _c("Weights (real or synthetic)", "checkpoint_weights_kind", "c", "checkpoint", "real in v2.", "Run config"),
    _c("Checkpoint size (GB)", "checkpoint_size_gb", "gb", "checkpoint", "Checkpoint GB.", "Headers"),
    _c("Checkpoint files (count)", "checkpoint_number_of_files", "n", "checkpoint", "File count.", "Headers"),
    _c("Checkpoint tensors (count)", "checkpoint_number_of_tensors", "n", "checkpoint", "Tensor count.", "Headers"),
    _c("Pipeline (to_maxtext or safetensors_dynamic)", "pipeline_name", "t", "pipeline", "Pipeline.", "Run config"),
    _c("Device mesh shape", "device_mesh_shape", "t", "pipeline", "Mesh shape.", "Result JSON"),
    _c("ICI FSDP parallelism", "ici_fsdp_parallelism", "n", "pipeline", "ici_fsdp_parallelism.", "Result JSON"),
    _c("DCN parallelism axis", "dcn_parallelism_axis", "c", "pipeline", "DCN axis.", "Result JSON"),
    _c("DCN parallelism size", "dcn_parallelism_size", "n", "pipeline", "DCN size.", "Result JSON"),
    _c(
        "Orbax checkpoint storage concurrency (GB)",
        "checkpoint_storage_concurrent_gb",
        "gb",
        "pipeline",
        "Orbax cap.",
        "Result JSON",
    ),
    _c(
        "safetensors read concurrency (bytes)",
        "safetensors_read_concurrent_bytes",
        "n",
        "pipeline",
        "ST read cap.",
        "Result JSON",
    ),
    _c(
        "JAX compilation cache (empty or warm)",
        "jax_compilation_cache_state",
        "c",
        "pipeline",
        "Cache state.",
        "Result JSON",
    ),
    _c("Harness environment flags", "harness_environment_flags", "t", "pipeline", "Env flags.", "Result JSON"),
    _c("Run status (OK, FAILED, NOT_RUN or SKIPPED)", "run_status", "c", "status", "Run status.", "Result JSON"),
    _c("Failure category (Legend code)", "failure_category", "c", "status", "Failure category.", "Result JSON"),
    _c("Failed stage", "failed_stage_name", "t", "status", "Failed stage.", "Result JSON"),
    _c("Process exit code", "process_exit_code", "n", "status", "Exit code.", "Result JSON"),
    _c("Process signal", "process_signal", "c", "status", "Signal.", "Result JSON"),
    _c(
        "Error summary (exception class and first line)",
        "error_summary",
        "t",
        "status",
        "Error summary.",
        "Result JSON",
    ),
    _c("Preempted (TRUE or FALSE)", "was_preempted", "bool", "status", "Preemption flag.", "Result JSON"),
    _c("Step 1 download log", "step1_log_url", "url", "links", "Step 1 log.", "Result JSON"),
    _c("DeepSeek Step 1b log", "step1b_log_url", "url", "links", "Step 1b log.", "Result JSON"),
    _c("Pipeline Step 1 log", "pipeline_step1_log_url", "url", "links", "Pipeline Step 1 log.", "Result JSON"),
    _c("Step 2 log", "step2_log_url", "url", "links", "Step 2 log.", "Result JSON"),
    _c(
        "Load log (to_maxtext Step 3 or safetensors_dynamic Step 2)",
        "step3_or_load_log_url",
        "url",
        "links",
        "Load log.",
        "Result JSON",
    ),
    _c("All hosts' logs (folder)", "all_hosts_log_folder_url", "url", "links", "Logs folder.", "Result JSON"),
    _c("Result JSON", "result_json_url", "url", "links", "Result JSON URL.", "Result JSON"),
    _c("Pod names", "pod_names", "t", "links", "Pod names.", "Downward API"),
    _c("Node names", "node_names", "t", "links", "Node names.", "Downward API"),
)

DATA_TABS: tuple[tuple[str, tuple[ColumnSpec, ...]], ...] = (
    ("Summary", SUMMARY_COLUMNS),
    ("Step 1 downloads", STEP1_COLUMNS),
    ("Stages: safetensors_dynamic", STAGES_SD_COLUMNS),
    ("Stages: to_maxtext", STAGES_TM_COLUMNS),
    ("Runs", RUNS_COLUMNS),
)


def make_row_id(  # pylint: disable=too-many-positional-arguments
    source: str,
    platform: str,
    model_name: str,
    scan_mode: str,
    pipeline: str,
    num_slices: int,
    topology: str,
) -> str:
  """Builds the canonical cross-tab Row ID key."""
  return f"{source}|{platform}|{model_name}|{scan_mode}|{pipeline}|{num_slices}x{topology}"


def normalize_sheet_status(run_status: str, failure_category: str | None = None) -> str:
  """Maps a result JSON `(run_status, failure_category)` to a Legend Summary `Run status` code."""
  if run_status == "NOT_RUN":
    return "NOT RUN"
  if run_status == "FAILED":
    if failure_category in FAILURE_CATEGORY_CODES:
      return str(failure_category)
    return "ERROR"
  if run_status in RUN_STATUS_CODES:
    return run_status
  return "ERROR"


def is_valid_numeric_cell(value: Any) -> bool:
  """Returns True if `value` is a real number (not bool/NaN/Inf) or a valid Legend non-numeric code."""
  if isinstance(value, bool):
    return False
  if isinstance(value, (int, float)):
    return math.isfinite(float(value))
  if isinstance(value, str):
    return value in NON_NUMERIC_CODES
  return False


def validate_row_dict(row: dict[str, Any], columns: tuple[ColumnSpec, ...]) -> list[str]:
  """Checks that numeric columns hold only finite numbers or exact Legend codes."""
  errors: list[str] = []
  for col in columns:
    if col.key not in row or row[col.key] is None:
      continue
    val = row[col.key]
    if col.fmt in NUMERIC_FORMATS and not is_valid_numeric_cell(val):
      errors.append(f"{col.key}: expected finite number or Legend code, got {val!r}")
    elif col.fmt == "bool" and not isinstance(val, bool) and val not in ("", "TRUE", "FALSE"):
      errors.append(f"{col.key}: expected bool or TRUE/FALSE, got {val!r}")
  return errors


def validate_step1_record(record: dict[str, Any]) -> list[str]:
  """Validates a v2 Step 1 / Step 1b download JSON record."""
  errors = validate_row_dict(record, STEP1_COLUMNS)
  for req in ("tpu_platform", "step1_gcs_bucket_region", "model_name", "step1_hf_repo_id", "step1_value_source"):
    if not record.get(req):
      errors.append(f"missing required field {req!r}")
  val_src = record.get("step1_value_source")
  if val_src and val_src not in STEP1_VALUE_SOURCES:
    errors.append(f"invalid step1_value_source {val_src!r}")
  if val_src == "cold_download":
    if record.get("step1_files_already_in_gcs_count") != 0:
      errors.append(
          f"cold_download requires step1_files_already_in_gcs_count == 0, "
          f"got {record.get('step1_files_already_in_gcs_count')!r}"
      )
    if not record.get("step1_measured_by_run_id"):
      errors.append("cold_download requires step1_measured_by_run_id")
  bf16_src = record.get("step1b_deepseek_bf16_source")
  if bf16_src not in (None, "", "N/A") and bf16_src not in DEEPSEEK_BF16_SOURCES:
    errors.append(f"invalid step1b_deepseek_bf16_source {bf16_src!r}")
  return errors


def validate_v2_phase_record(record: dict[str, Any]) -> list[str]:
  """Validates a v2 per-phase result JSON record."""
  errors: list[str] = []
  if record.get("schema_version") != SCHEMA_VERSION:
    errors.append(f"schema_version must be {SCHEMA_VERSION}, got {record.get('schema_version')!r}")

  for req in ("phase_name", "model_name", "scan_layers_mode", "tpu_platform", "run_status"):
    if not record.get(req):
      errors.append(f"missing required field {req!r}")

  status = record.get("run_status")
  if status not in RESULT_JSON_STATUS_CODES:
    errors.append(f"invalid run_status {status!r}")

  if status == "FAILED":
    fcat = record.get("failure_category")
    if fcat not in FAILURE_CATEGORY_CODES:
      errors.append(f"FAILED record must set valid failure_category, got {fcat!r}")

  phase = record.get("phase_name", "")
  if status == "OK" and phase in ("to_maxtext_step3", "safetensors_dynamic"):
    if record.get("jax_backend_platform") != "tpu":
      errors.append(
          f"TPU load OK record must have jax_backend_platform == 'tpu', got {record.get('jax_backend_platform')!r}"
      )
    expected_dev = record.get("total_tpu_devices")
    measured_dev = record.get("total_tpu_devices_measured")
    if not isinstance(measured_dev, int) or measured_dev <= 0:
      errors.append(f"invalid total_tpu_devices_measured {measured_dev!r}")
    elif isinstance(expected_dev, int) and measured_dev != expected_dev:
      errors.append(f"total_tpu_devices_measured ({measured_dev}) != total_tpu_devices ({expected_dev})")

  errors.extend(validate_row_dict(record, RUNS_COLUMNS))
  if record.get("pipeline_name") == "safetensors_dynamic":
    errors.extend(validate_row_dict(record, STAGES_SD_COLUMNS))
  elif record.get("pipeline_name") == "to_maxtext":
    errors.extend(validate_row_dict(record, STAGES_TM_COLUMNS))
  return errors


def load_models_v2(path: str = MODELS_V2_PATH) -> dict[str, Any]:
  """Loads the canonical `configs/models_v2.json` config."""
  with open(path, "r", encoding="utf-8") as f:
    return json.load(f)


def iter_planned_matrix(models_cfg: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
  """Yields every planned `(platform, model, slices, topology, scan_mode, pipeline)` entry."""
  cfg = models_cfg if models_cfg is not None else load_models_v2()
  scan_modes = tuple(cfg["scan_modes"])
  pipelines = tuple(cfg["pipelines"])
  for model in cfg["models"]:
    for platform, placements in model["topologies_by_platform"].items():
      for placement in placements:
        slices = int(placement["slices"])
        topology = str(placement["topology"])
        for scan_mode in scan_modes:
          for pipeline in pipelines:
            yield {
                "platform": platform,
                "model_name": model["model_name"],
                "tier": model["tier"],
                "slices": slices,
                "topology": topology,
                "scan_mode": scan_mode,
                "pipeline": pipeline,
                "row_id": make_row_id("v2", platform, model["model_name"], scan_mode, pipeline, slices, topology),
                "model": model,
            }
