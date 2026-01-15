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

"""Centralized decoupling helpers.

Set DECOUPLE_GCLOUD=TRUE in the environment to disable optional Google Cloud / JetStream / GCS / diagnostics
integrations while still allowing local unit tests to import modules. This module provides:

- is_decoupled(): returns True if decoupled flag set.
- cloud_diagnostics(): tuple(diagnostic, debug_configuration, diagnostic_configuration, stack_trace_configuration)
  providing either real objects or lightweight stubs.
- jetstream(): returns a namespace-like object exposing Engine, Devices, ResultTokens etc. or stubs.
- gcs_storage(): returns google.cloud.storage module or stub namespace with Client/Blob/Bucket.
- goodput_modules(): returns (goodput, monitoring, is_stub) for ml_goodput_measurement integration or stubs.
- monitoring_modules(): returns (monitoring_v3, metric_pb2, monitored_resource_pb2, GoogleAPIError, is_stub)
    for Google Cloud Monitoring integration or stubs.

All stubs raise RuntimeError only when actually invoked, not at import time, so test collection proceeds.
"""
from __future__ import annotations

from types import SimpleNamespace
import importlib.util
import os


def is_decoupled() -> bool:  # dynamic check so setting env after initial import still works
  """Return True when DECOUPLE_GCLOUD environment variable is set to TRUE."""
  return os.environ.get("DECOUPLE_GCLOUD", "").upper() == "TRUE"


# ---------------- Cloud Diagnostics -----------------


def _cloud_diag_stubs():
  """Return lightweight stubs for cloud TPU diagnostics."""
  import contextlib  # pylint: disable=import-outside-toplevel

  class _StubDiag:
    """Stub diagnostic object returning skip metadata."""

    def run(self, *_a, **_k):
      return {"status": "skipped"}

    def diagnose(self, *_a, **_k):
      """Return a context manager that swallows diagnostic errors in stub mode."""

      @contextlib.contextmanager
      def _graceful_diagnose():
        try:
          yield
        except Exception as exc:  # pylint: disable=broad-exception-caught
          print("Warning: using stubs for cloud_diagnostics diagnose() - " f"caught: {exc}")

      return _graceful_diagnose()

  class _StubDebugConfig:
    """Stub debug configuration."""

    def __init__(self, *a, **k):  # pylint: disable=unused-argument
      pass

  class _StubStackTraceConfig:
    """Stub stack trace configuration."""

    def __init__(self, *a, **k):  # pylint: disable=unused-argument
      pass

  class _StubDiagnosticConfig:
    """Stub diagnostic configuration wrapper."""

    def __init__(self, *a, debug_config=None, **k):  # pylint: disable=unused-argument
      del a, k
      self.debug_config = debug_config

  return (
      _StubDiag(),
      SimpleNamespace(DebugConfig=_StubDebugConfig, StackTraceConfig=_StubStackTraceConfig),
      SimpleNamespace(DiagnosticConfig=_StubDiagnosticConfig),
      SimpleNamespace(StackTraceConfig=_StubStackTraceConfig),
  )


def cloud_diagnostics():
  """Return real cloud diagnostics modules or stubs.

  If a dependency is missing and we are decoupled, return stubs. Otherwise
  re-raise the import error so callers fail fast.
  """
  try:
    from cloud_tpu_diagnostics import diagnostic  # type: ignore  # pylint: disable=import-outside-toplevel
    from cloud_tpu_diagnostics.configuration import (  # type: ignore  # pylint: disable=import-outside-toplevel
        debug_configuration,
        diagnostic_configuration,
        stack_trace_configuration,
    )

    return diagnostic, debug_configuration, diagnostic_configuration, stack_trace_configuration
  except ModuleNotFoundError:
    if is_decoupled():
      print("[DECOUPLED NO-OP] cloud_diagnostics: dependency missing; using stubs.")
      return _cloud_diag_stubs()
    raise


# ---------------- JetStream -----------------


def _jetstream_stubs():
  """Return lightweight stubs for JetStream modules."""

  class Engine:  # minimal base class stub
    """Stub Engine accepting any initialization signature."""

    def __init__(self, *a, **k):  # pylint: disable=unused-argument
      pass

  class ResultTokens:
    """Container for result token arrays used by JetStream."""

    def __init__(
        self,
        *args,
        data=None,
        tokens_idx=None,
        valid_idx=None,
        length_idx=None,
        log_prob=None,
        samples_per_slot: int | None = None,
        **kwargs,
    ):
      del args, kwargs  # unused
      self.data = data
      self.tokens_idx = tokens_idx
      self.valid_idx = valid_idx
      self.length_idx = length_idx
      self.log_prob = log_prob
      self.samples_per_slot = samples_per_slot

  # Tokenizer placeholders (unused in decoupled tests due to runtime guard).
  class TokenizerParameters:  # pragma: no cover - placeholder
    """Stub tokenizer parameters object."""

    def __init__(self, *a, **k):  # pylint: disable=unused-argument
      pass

  class TokenizerType:  # emulate enum descriptor access pattern
    """Stub tokenizer type descriptor container."""

    DESCRIPTOR = SimpleNamespace(values_by_name={})

  config_lib = SimpleNamespace()  # not used directly in decoupled tests
  engine_api = SimpleNamespace(Engine=Engine, ResultTokens=ResultTokens)
  token_utils = SimpleNamespace()  # build_tokenizer guarded in MaxEngine when decoupled
  tokenizer_api = SimpleNamespace()  # placeholder
  token_params_ns = SimpleNamespace(TokenizerParameters=TokenizerParameters, TokenizerType=TokenizerType)
  return config_lib, engine_api, token_utils, tokenizer_api, token_params_ns


def jetstream():
  """Return JetStream modules or stubs based on availability and decoupling."""
  needed = [
      "jetstream.core.config_lib",
      "jetstream.engine.engine_api",
      "jetstream.engine.token_utils",
      "jetstream.engine.tokenizer_api",
      "jetstream.engine.tokenizer_pb2",
  ]
  try:
    for mod in needed:
      if importlib.util.find_spec(mod) is None:
        if is_decoupled():
          print("[DECOUPLED NO-OP] jetstream: dependency missing; using stubs.")
          return _jetstream_stubs()
        raise ModuleNotFoundError(mod)

    from jetstream.core import config_lib  # type: ignore  # pylint: disable=import-outside-toplevel
    from jetstream.engine import engine_api, token_utils, tokenizer_api  # type: ignore  # pylint: disable=import-outside-toplevel
    from jetstream.engine.tokenizer_pb2 import TokenizerParameters, TokenizerType  # type: ignore  # pylint: disable=import-outside-toplevel

    return (
        config_lib,
        engine_api,
        token_utils,
        tokenizer_api,
        SimpleNamespace(TokenizerParameters=TokenizerParameters, TokenizerType=TokenizerType),
    )
  except ModuleNotFoundError:
    if is_decoupled():
      print("[DECOUPLED NO-OP] jetstream: dependency missing; using stubs.")
      return _jetstream_stubs()
    raise


# ---------------- GCS -----------------


def _gcs_stubs():  # pragma: no cover - simple no-op placeholders
  """Return stub implementations of the google.cloud.storage API."""

  class _StubBlob:
    """Stub GCS blob with no-op operations."""

    def __init__(self, *a, **k):  # pylint: disable=unused-argument
      del a, k

    def upload_from_filename(self, *a, **k):  # pylint: disable=unused-argument
      return False

    def upload_from_string(self, *a, **k):  # pylint: disable=unused-argument
      return False

    def exists(self):
      return False

    def download_as_string(self):
      return b"{}"

  class _StubListPages:
    """Stub for iterable pages returned by list_blobs."""

    def __init__(self):
      self.pages = [SimpleNamespace(prefixes=[])]

  class _StubBucket:
    """Stub GCS bucket returning stub blobs and pages."""

    def __init__(self, *a, **k):  # pylint: disable=unused-argument
      del a, k

    def blob(self, *a, **k):  # pylint: disable=unused-argument
      return _StubBlob()

    def list_blobs(self, *a, **k):  # pylint: disable=unused-argument
      return _StubListPages()

  class _StubClient:
    """Stub GCS client exposing bucket helpers."""

    def __init__(self, *a, **k):  # pylint: disable=unused-argument
      del a, k

    def get_bucket(self, *a, **k):  # pylint: disable=unused-argument
      return _StubBucket()

    def bucket(self, *a, **k):  # pylint: disable=unused-argument
      return _StubBucket()

  return SimpleNamespace(Client=_StubClient, _IS_STUB=True)


def gcs_storage():
  """Return google.cloud.storage module or stub when decoupled or missing."""
  # In decoupled mode always prefer the stub, even if the library is installed,
  # to avoid accidental GCS calls in tests or local runs.
  if is_decoupled():  # fast path
    print("[DECOUPLED NO-OP] gcs_storage: dependency missing; using stubs.")
    return _gcs_stubs()

  try:  # pragma: no cover - attempt real import when not decoupled
    from google.cloud import storage  # type: ignore  # pylint: disable=import-outside-toplevel

    setattr(storage, "_IS_STUB", False)
    return storage
  except Exception:  # ModuleNotFoundError / ImportError for partial installs  # pylint: disable=broad-exception-caught
    print("[DECOUPLED NO-OP] gcs_storage: dependency missing; using stubs.")
    return _gcs_stubs()


# ---------------- Goodput (ml_goodput_measurement) -----------------


def _goodput_stubs():
  """Return stubs for ml_goodput_measurement integration."""

  class _StubGoodputRecorder:
    """Recorder stub exposing no-op methods and disabled flag."""

    def __init__(self, *a, **k):  # pylint: disable=unused-argument
      self.enabled = False

    def __getattr__(self, name):
      def _noop(*_a, **_k):
        pass

      return _noop

  class _StubMonitoringOptions:
    """Stub monitoring options container."""

    def __init__(self, *a, **k):  # pylint: disable=unused-argument
      pass

  class _StubGoodputMonitor:
    """Stub goodput monitor with no-op uploader methods."""

    def __init__(self, *a, **_k):  # pylint: disable=unused-argument
      pass

    def start_goodput_uploader(self):
      print("[DECOUPLED NO-OP] goodput uploader skipped.")

    def start_step_deviation_uploader(self):
      print("[DECOUPLED NO-OP] goodput step deviation uploader skipped.")

  monitoring_ns = SimpleNamespace(GCPOptions=_StubMonitoringOptions, GoodputMonitor=_StubGoodputMonitor)
  goodput_ns = SimpleNamespace(GoodputRecorder=_StubGoodputRecorder)
  return goodput_ns, monitoring_ns, True


def goodput_modules():
  """Return real goodput modules or stubs when missing and decoupled."""
  try:
    from ml_goodput_measurement import goodput, monitoring  # type: ignore  # pylint: disable=import-outside-toplevel

    return goodput, monitoring, False
  except ModuleNotFoundError:
    if is_decoupled():
      print("[DECOUPLED NO-OP] ml_goodput_measurement: dependency missing; using stubs.")
      return _goodput_stubs()
    raise


__all__ = ["is_decoupled", "cloud_diagnostics", "jetstream", "gcs_storage", "goodput_modules"]

# ---------------- Cloud Monitoring (monitoring_v3 / metric_pb2) -----------------


def _monitoring_stubs():  # pragma: no cover - simple placeholders
  """Return stub implementations for Cloud Monitoring APIs."""

  class GoogleAPIError(Exception):
    """Stub GoogleAPIError mirroring the real exception name."""

  class _DummyMonitoringV3:
    """Dummy monitoring module providing minimal types."""

    class TimeSeries:

      def __init__(self, *a, **k):  # pylint: disable=unused-argument
        del a, k

    class Point:

      def __init__(self, *a, **k):  # pylint: disable=unused-argument
        del a, k

    class TimeInterval:

      def __init__(self, *a, **k):  # pylint: disable=unused-argument
        del a, k

    class TypedValue:

      def __init__(self, *a, **k):  # pylint: disable=unused-argument
        del a, k

    class MetricServiceClient:

      def __init__(self, *a, **k):  # pylint: disable=unused-argument
        del a, k

      def create_time_series(self, *a, **k):  # pylint: disable=unused-argument
        return False

  class _DummyMetricPB2:
    """Dummy metric_pb2 module namespace."""

    class Metric:

      def __init__(self, *a, **k):  # pylint: disable=unused-argument
        del a, k

  class _DummyMonitoredResourcePB2:
    """Dummy monitored_resource_pb2 module namespace."""

    class MonitoredResource:

      def __init__(self, *a, **k):  # pylint: disable=unused-argument
        del a, k

  return _DummyMonitoringV3(), _DummyMetricPB2(), _DummyMonitoredResourcePB2(), GoogleAPIError, True


def monitoring_modules():
  """Return monitoring modules or stubs.

  Stubs only if decoupled AND dependency missing; if not decoupled and missing ->
  re-raise.
  """
  try:  # Attempt real imports first
    from google.cloud import monitoring_v3  # type: ignore  # pylint: disable=import-outside-toplevel
    from google.api import metric_pb2, monitored_resource_pb2  # type: ignore  # pylint: disable=import-outside-toplevel
    from google.api_core.exceptions import GoogleAPIError  # type: ignore  # pylint: disable=import-outside-toplevel

    return monitoring_v3, metric_pb2, monitored_resource_pb2, GoogleAPIError, False
  except (ModuleNotFoundError, ImportError):  # broaden to handle partial google installs
    if is_decoupled():
      print("[DECOUPLED NO-OP] monitoring: dependency missing; using stubs.")
      return _monitoring_stubs()
    raise


__all__.append("monitoring_modules")

# ---------------- Workload Monitor (GCPWorkloadMonitor) -----------------


def _workload_monitor_stub():  # pragma: no cover - simple placeholder
  """Return stub GCPWorkloadMonitor implementation and stub flag."""

  class GCPWorkloadMonitor:
    """Stub of GCPWorkloadMonitor exposing no-op methods."""

    def __init__(self, *a, **k):  # pylint: disable=unused-argument
      pass

    def start_heartbeat_reporting_thread(self, *a, **k):  # pylint: disable=unused-argument
      pass

    def start_performance_reporting_thread(self, *a, **k):  # pylint: disable=unused-argument
      pass

  return GCPWorkloadMonitor, True


def workload_monitor():
  """Return (GCPWorkloadMonitor, is_stub) centralizing stub logic.

  If decoupled OR import fails, returns stub class; otherwise real class.
  """
  if is_decoupled():  # fast path: never attempt heavy import
    print("[DECOUPLED NO-OP] workload_monitor: using stub.")
    return _workload_monitor_stub()

  try:
    from MaxText.gcp_workload_monitor import GCPWorkloadMonitor  # type: ignore  # pylint: disable=import-outside-toplevel

    return GCPWorkloadMonitor, False
  except Exception:  # ModuleNotFoundError / ImportError  # pylint: disable=broad-exception-caught
    print("[NO-OP] workload_monitor dependency missing; using stub.")
    return _workload_monitor_stub()


__all__.append("workload_monitor")

# ---------------- Vertex Tensorboard -----------------


def _vertex_tb_stub():  # pragma: no cover - simple placeholder
  """Return stub VertexTensorboardManager implementation and stub flag."""

  class VertexTensorboardManager:
    """Stub VertexTensorboardManager with no-op configure method."""

    def __init__(self, *a, **k):  # pylint: disable=unused-argument
      pass

    def configure_vertex_tensorboard(self, *a, **k):  # pylint: disable=unused-argument
      # NO-OP in decoupled / missing dependency mode
      pass

  return VertexTensorboardManager, True


def vertex_tensorboard_components():
  """Return (VertexTensorboardManager, is_stub).

  Decoupled or missing dependency -> stub class with no-op configure method.
  """
  if is_decoupled():
    print("[DECOUPLED NO-OP] vertex_tensorboard: using stub.")
    return _vertex_tb_stub()

  try:
    from MaxText.vertex_tensorboard import VertexTensorboardManager  # type: ignore  # pylint: disable=import-outside-toplevel

    return VertexTensorboardManager, False
  except Exception:  # pylint: disable=broad-exception-caught
    print("[NO-OP] vertex_tensorboard dependency missing; using stub.")
    return _vertex_tb_stub()


__all__.append("vertex_tensorboard_components")

# ---------------- TensorBoardX (moved stub) -----------------

try:
  if not is_decoupled():  # Only attempt real import when not decoupled
    from tensorboardX import writer  # type: ignore  # pylint: disable=import-outside-toplevel,unused-import

    _TENSORBOARDX_AVAILABLE = True
  else:
    raise ModuleNotFoundError("Decoupled mode skips tensorboardX import")
except Exception:  # pragma: no cover - provide stub fallback  # pylint: disable=broad-exception-caught
  _TENSORBOARDX_AVAILABLE = False

  class _DummySummaryWriter:
    """Stubbed TensorBoardX SummaryWriter replacement."""

    def __init__(self, *args, **kwargs):  # pylint: disable=unused-argument
      del args, kwargs

    def add_text(self, *args, **kwargs):
      pass

    def add_scalar(self, *args, **kwargs):
      pass

    def add_histogram(self, *args, **kwargs):
      pass

    def flush(self):
      pass

    def close(self):
      pass

  class writer:  # pylint: disable=too-few-public-methods
    SummaryWriter = _DummySummaryWriter


__all__.append("writer")
__all__.append("_TENSORBOARDX_AVAILABLE")
