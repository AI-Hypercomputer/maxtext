"""Centralized decoupling helpers for JetStream / Tunix / cloud diagnostics / GCS.

Set DECOUPLE_GCLOUD=TRUE in the environment to disable optional Google Cloud / JetStream / Tunix
integrations while still allowing local unit tests to import modules. This module provides:

- is_decoupled(): returns True if decoupled flag set.
- cloud_diagnostics(): tuple(diagnostic, debug_configuration, diagnostic_configuration, stack_trace_configuration)
  providing either real objects or lightweight stubs.
- jetstream(): returns a namespace-like object exposing Engine, Devices, ResultTokens etc. or stubs.
- tunix(): returns peft_trainer, DataHooks, TrainingHooks stubs or real imports if available and not decoupled.
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
    return os.environ.get("DECOUPLE_GCLOUD", "").upper() == "TRUE"

# ---------------- Cloud Diagnostics -----------------

def _cloud_diag_stubs():
    class _StubDiag:
        def run(self, *_a, **_k):
            return {"status": "skipped"}
        def diagnose(self, *_a, **_k):
            # Return a context manager that gracefully handles any errors
            import contextlib
            @contextlib.contextmanager
            def _graceful_diagnose():
                try:
                    yield
                except Exception as e:
                    # Log error but don't crash
                    print(f"Warning: Using stubs in decoupling mode for cloud_diagnostics replacement. This stub is for diagnose function: {e}")
            return _graceful_diagnose()
    class _StubDebugConfig:
        def __init__(self, *a, **k):
            pass
    class _StubStackTraceConfig:
        def __init__(self, *a, **k):
            pass
    class _StubDiagnosticConfig:
        def __init__(self, debug_config=None, *a, **k):
            self.debug_config = debug_config
    return (
        _StubDiag(),
        SimpleNamespace(DebugConfig=_StubDebugConfig, StackTraceConfig=_StubStackTraceConfig),
        SimpleNamespace(DiagnosticConfig=_StubDiagnosticConfig),
        SimpleNamespace(StackTraceConfig=_StubStackTraceConfig),
    )

def cloud_diagnostics():
    """Return diagnostics libs or stubs.

        Stubs only if decoupled AND dependency missing.
        If not decoupled and missing -> re-raise.
    """
    try:
        from cloud_tpu_diagnostics import diagnostic  # type: ignore
        from cloud_tpu_diagnostics.configuration import (  # type: ignore
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
    # Provide class-based stubs to allow subclassing and instantiation without import-time errors.
    class Engine:  # minimal base class stub
        def __init__(self, *a, **k):  # accept any signature
            pass

    class ResultTokens:
        def __init__(
            self,
            *,
            data=None,
            tokens_idx=None,
            valid_idx=None,
            length_idx=None,
            log_prob=None,
            samples_per_slot: int | None = None,
        ):
            self.data = data
            self.tokens_idx = tokens_idx
            self.valid_idx = valid_idx
            self.length_idx = length_idx
            self.log_prob = log_prob
            self.samples_per_slot = samples_per_slot

    # Tokenizer placeholders (unused in decoupled tests due to runtime guard).
    class TokenizerParameters:  # pragma: no cover - placeholder
        def __init__(self, *a, **k):
            pass

    class _FakeDescriptorValues:
        def __init__(self):
            self.values_by_name = {}

    class TokenizerType:  # emulate enum descriptor access pattern
        DESCRIPTOR = SimpleNamespace(values_by_name={})

    config_lib = SimpleNamespace()  # not used directly in decoupled tests
    engine_api = SimpleNamespace(Engine=Engine, ResultTokens=ResultTokens)
    token_utils = SimpleNamespace()  # build_tokenizer guarded in MaxEngine when decoupled
    tokenizer_api = SimpleNamespace()  # placeholder
    token_params_ns = SimpleNamespace(TokenizerParameters=TokenizerParameters, TokenizerType=TokenizerType)
    return config_lib, engine_api, token_utils, tokenizer_api, token_params_ns

def jetstream():
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
        from jetstream.core import config_lib  # type: ignore
        from jetstream.engine import engine_api, token_utils, tokenizer_api  # type: ignore
        from jetstream.engine.tokenizer_pb2 import TokenizerParameters, TokenizerType  # type: ignore
        return config_lib, engine_api, token_utils, tokenizer_api, SimpleNamespace(TokenizerParameters=TokenizerParameters, TokenizerType=TokenizerType)
    except ModuleNotFoundError:
        if is_decoupled():
            print("[DECOUPLED NO-OP] jetstream: dependency missing; using stubs.")
            return _jetstream_stubs()
        raise

# ---------------- Tunix -----------------

def _tunix_stubs():
    class DataHooks:  # simple base type
        def __init__(self, *a, **k):
            pass
    class TrainingHooks:  # simple base type
        def __init__(self, *a, **k):
            pass
    class _StubPeftTrainer:
        def __init__(self, *a, **k):
            pass
    peft_trainer = SimpleNamespace(PeftTrainer=_StubPeftTrainer)
    hooks = SimpleNamespace(DataHooks=DataHooks, TrainingHooks=TrainingHooks)
    return peft_trainer, hooks

def tunix():
    try:
        if importlib.util.find_spec("tunix") is None:
            if is_decoupled():
                print("[DECOUPLED NO-OP] tunix: dependency missing; using stubs.")
                return _tunix_stubs()
            raise ModuleNotFoundError("tunix")
        from tunix.sft import peft_trainer  # type: ignore
        from tunix.sft import hooks as tunix_hooks  # type: ignore
        return peft_trainer, tunix_hooks
    except ModuleNotFoundError:
        if is_decoupled():
            print("[DECOUPLED NO-OP] tunix: dependency missing; using stubs.")
            return _tunix_stubs()
        raise

# ---------------- GCS -----------------

def _gcs_stubs():  # pragma: no cover - simple no-op placeholders
    class _StubBlob:
        def __init__(self, *a, **k):
            pass
        def upload_from_filename(self, *a, **k):
            return False
        def upload_from_string(self, *a, **k):
            return False
        def exists(self):
            return False
        def download_as_string(self):
            return b"{}"
    class _StubListPages:
        def __init__(self):
            self.pages = [SimpleNamespace(prefixes=[])]
    class _StubBucket:
        def __init__(self, *a, **k):
            pass
        def blob(self, *a, **k):
            return _StubBlob()
        def list_blobs(self, *a, **k):
            return _StubListPages()
    class _StubClient:
        def __init__(self, *a, **k):
            pass
        def get_bucket(self, *a, **k):
            return _StubBucket()
        def bucket(self, *a, **k):
            return _StubBucket()
    return SimpleNamespace(Client=_StubClient, _IS_STUB=True)

def gcs_storage():
    # In decoupled mode always prefer the stub, even if the library is installed,
    # to avoid accidental GCS calls in tests or local runs.
    if is_decoupled():  # fast path
        print("[DECOUPLED NO-OP] gcs_storage: dependency missing; using stubs.")
        return _gcs_stubs()
    try:  # pragma: no cover - attempt real import when not decoupled
        from google.cloud import storage  # type: ignore
        setattr(storage, "_IS_STUB", False)
        return storage
    except Exception:  # ModuleNotFoundError / ImportError for partial installs
        print("[DECOUPLED NO-OP] gcs_storage: dependency missing; using stubs.")
        return _gcs_stubs()

# ---------------- Goodput (ml_goodput_measurement) -----------------

def _goodput_stubs():
    class _StubGoodputRecorder:
        def __init__(self, *a, **k):
            self.enabled = False
        def __getattr__(self, name):
            def _noop(*_a, **_k):
                pass
            return _noop
    class _StubMonitoringOptions:
        def __init__(self, *a, **k):
            pass
    class _StubGoodputMonitor:
        def __init__(self, *a, **_k):
            pass
        def start_goodput_uploader(self):
            print("[DECOUPLED NO-OP] goodput uploader skipped.")
        def start_step_deviation_uploader(self):
            print("[DECOUPLED NO-OP] goodput step deviation uploader skipped.")
    monitoring_ns = SimpleNamespace(GCPOptions=_StubMonitoringOptions, GoodputMonitor=_StubGoodputMonitor)
    goodput_ns = SimpleNamespace(GoodputRecorder=_StubGoodputRecorder)
    return goodput_ns, monitoring_ns, True

def goodput_modules():
    try:
        from ml_goodput_measurement import goodput, monitoring  # type: ignore
        return goodput, monitoring, False
    except ModuleNotFoundError:
        if is_decoupled():
            print("[DECOUPLED NO-OP] ml_goodput_measurement: dependency missing; using stubs.")
            return _goodput_stubs()
        raise

__all__ = ["is_decoupled", "cloud_diagnostics", "jetstream", "tunix", "gcs_storage", "goodput_modules"]

# ---------------- Cloud Monitoring (monitoring_v3 / metric_pb2) -----------------

def _monitoring_stubs():  # pragma: no cover - simple placeholders
    class GoogleAPIError(Exception):  # mirror real exception name
        pass
    class _DummyMonitoringV3:
        class TimeSeries:
            def __init__(self, *a, **k):
                pass
        class Point:
            def __init__(self, *a, **k):
                pass
        class TimeInterval:
            def __init__(self, *a, **k):
                pass
        class TypedValue:
            def __init__(self, *a, **k):
                pass
        class MetricServiceClient:
            def __init__(self, *a, **k):
                pass
            def create_time_series(self, *a, **k):
                return False
    class _DummyMetricPB2:
        class Metric:
            def __init__(self, *a, **k):
                pass
    class _DummyMonitoredResourcePB2:
        class MonitoredResource:
            def __init__(self, *a, **k):
                pass
    return _DummyMonitoringV3(), _DummyMetricPB2(), _DummyMonitoredResourcePB2(), GoogleAPIError, True

def monitoring_modules():
    """Return monitoring modules or stubs.

    Stubs only if decoupled AND dependency missing; if not decoupled and missing -> re-raise.
    """
    try:  # Attempt real imports first
        from google.cloud import monitoring_v3  # type: ignore
        from google.api import metric_pb2, monitored_resource_pb2  # type: ignore
        from google.api_core.exceptions import GoogleAPIError  # type: ignore
        return monitoring_v3, metric_pb2, monitored_resource_pb2, GoogleAPIError, False
    except (ModuleNotFoundError, ImportError):  # broaden to handle partial google installs without monitoring_v3
        if is_decoupled():
            print("[DECOUPLED NO-OP] monitoring: dependency missing; using stubs.")
            return _monitoring_stubs()
        raise

__all__.append("monitoring_modules")

# ---------------- Workload Monitor (GCPWorkloadMonitor) -----------------

def _workload_monitor_stub():  # pragma: no cover - simple placeholder
    class GCPWorkloadMonitor:
        def __init__(self, *a, **k):
            pass
        def start_heartbeat_reporting_thread(self, *a, **k):
            pass
        def start_performance_reporting_thread(self, *a, **k):
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
        from MaxText.gcp_workload_monitor import GCPWorkloadMonitor  # type: ignore
        return GCPWorkloadMonitor, False
    except Exception:  # ModuleNotFoundError / ImportError
        print("[NO-OP] workload_monitor dependency missing; using stub.")
        return _workload_monitor_stub()

__all__.append("workload_monitor")

# ---------------- Vertex Tensorboard -----------------

def _vertex_tb_stub():  # pragma: no cover - simple placeholder
    class VertexTensorboardManager:
        def __init__(self, *a, **k):
            pass
        def configure_vertex_tensorboard(self, *a, **k):
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
        from MaxText.vertex_tensorboard import VertexTensorboardManager  # type: ignore
        return VertexTensorboardManager, False
    except Exception:
        print("[NO-OP] vertex_tensorboard dependency missing; using stub.")
        return _vertex_tb_stub()

__all__.append("vertex_tensorboard_components")

# ---------------- TensorBoardX (moved stub) -----------------

try:
    if not is_decoupled():  # Only attempt real import when not decoupled
        from tensorboardX import writer  # type: ignore
        _TENSORBOARDX_AVAILABLE = True
    else:
        raise ModuleNotFoundError("Decoupled mode skips tensorboardX import")
except Exception:  # pragma: no cover - provide stub fallback
    _TENSORBOARDX_AVAILABLE = False

    class _DummySummaryWriter:
        def __init__(self, *args, **kwargs):
            pass
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


