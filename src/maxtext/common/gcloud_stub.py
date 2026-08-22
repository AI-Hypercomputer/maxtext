# Copyright 2025 Google LLC
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
- jetstream(): returns a namespace-like object exposing Engine, Devices, ResultTokens etc. or stubs.
- gcs_storage(): returns google.cloud.storage module or stub namespace with Client/Blob/Bucket.
- goodput_modules(): returns (goodput, monitoring, is_stub) for ml_goodput_measurement integration or stubs.
- monitoring_modules(): returns (monitoring_v3, metric_pb2, monitored_resource_pb2, GoogleAPIError, is_stub)
    for Google Cloud Monitoring integration or stubs.
- vertex_tensorboard_modules(): returns (VertexTensorboardManager, is_stub) for Vertex Tensorboard integration.

All stubs raise RuntimeError only when actually invoked, not at import time, so test collection proceeds.
"""
from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
import importlib.util
import os
from typing import TypeVar


def is_decoupled() -> bool:  # dynamic check so setting env after initial import still works
  """Return True when DECOUPLE_GCLOUD environment variable is set to TRUE."""
  return os.environ.get("DECOUPLE_GCLOUD", "").upper() == "TRUE"


T = TypeVar("T")


def _import_or_stub(
    import_fn: Callable[[], T],
    stub_fn: Callable[[], T],
    *,
    label: str,
    stub_if_decoupled: bool = False,
    stub_on_error_when_not_decoupled: bool = False,
) -> T:
  """Import and return real deps or return stubs based on decoupled/error policy.

  This centralizes the common try-import / fallback-to-stub logic used throughout
  this file, so each public helper remains short and consistent.
  """
  if stub_if_decoupled and is_decoupled():
    print(f"[DECOUPLED NO-OP] {label}: using stub.")
    return stub_fn()

  try:
    return import_fn()
  except Exception as exc:  # pylint: disable=broad-exception-caught
    if is_decoupled() or stub_on_error_when_not_decoupled:
      prefix = "[DECOUPLED NO-OP]" if is_decoupled() else "[NO-OP]"
      print(f"{prefix} {label}: dependency missing; using stub. ({type(exc).__name__})")
      return stub_fn()
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

    def get_result_at_slot(self, slot: int):
      from types import SimpleNamespace
      if self.data is not None and self.tokens_idx is not None:
        if isinstance(self.tokens_idx, tuple) and len(self.tokens_idx) == 2:
          tokens = self.data[slot, self.tokens_idx[0]:self.tokens_idx[1]]
        else:
          tokens = self.data[slot, self.tokens_idx]
      else:
        tokens = self.data

      if self.data is not None and self.valid_idx is not None:
        if isinstance(self.valid_idx, tuple) and len(self.valid_idx) == 2:
          valid = self.data[slot, self.valid_idx[0]:self.valid_idx[1]]
        else:
          valid = self.data[slot, self.valid_idx]
      else:
        valid = None

      if self.data is not None and self.length_idx is not None:
        if isinstance(self.length_idx, tuple) and len(self.length_idx) == 2:
          length = self.data[slot, self.length_idx[0]:self.length_idx[1]]
        else:
          length = self.data[slot, self.length_idx]
      else:
        length = None

      log_prob = self.log_prob[slot] if self.log_prob is not None else None
      return SimpleNamespace(
          tokens=tokens,
          valid=valid,
          length=length,
          log_prob=log_prob,
      )

    def _tree_flatten(self):
      children = (self.data, self.tokens_idx, self.valid_idx, self.length_idx, self.log_prob)
      aux_data = (self.samples_per_slot,)
      return children, aux_data

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
      return cls(
          data=children[0],
          tokens_idx=children[1],
          valid_idx=children[2],
          length_idx=children[3],
          log_prob=children[4],
          samples_per_slot=aux_data[0],
      )

  try:
    import jax
    jax.tree_util.register_pytree_node(
        ResultTokens,
        ResultTokens._tree_flatten,
        ResultTokens._tree_unflatten,
    )
  except Exception:
    pass

  class TokenizerParameters:
    def __init__(self, path=None, tokenizer_type=None, access_token=None, use_chat_template=False, extra_ids=0, **kwargs):
      self.path = path
      self.tokenizer_type = tokenizer_type
      self.access_token = access_token
      self.use_chat_template = use_chat_template
      self.extra_ids = extra_ids

  class TokenizerType:
    tiktoken = 1
    sentencepiece = 2
    huggingface = 3
    DESCRIPTOR = SimpleNamespace(
        values_by_name={
            "tiktoken": SimpleNamespace(number=1),
            "sentencepiece": SimpleNamespace(number=2),
            "huggingface": SimpleNamespace(number=3),
        }
    )

  class HuggingFaceTokenizer:
    def __init__(self, metadata):
      import os
      import transformers
      try:
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            metadata.path,
            token=metadata.access_token or None,
            trust_remote_code=True,
        )
      except Exception:
        try:
          import huggingface_hub
          from tokenizers import Tokenizer
          tok_file = metadata.path
          if not os.path.exists(tok_file):
            tok_file = huggingface_hub.hf_hub_download(
                repo_id=metadata.path,
                filename="tokenizer.json",
                token=metadata.access_token or None,
            )
          self.tokenizer = Tokenizer.from_file(tok_file)
        except Exception:
          from tokenizers import Tokenizer
          self.tokenizer = Tokenizer.from_pretrained(metadata.path)

      self.pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
      self.eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
      self.bos_token_id = getattr(self.tokenizer, "bos_token_id", None)
      if self.pad_token_id is None and hasattr(self.tokenizer, "token_to_id"):
        self.eos_token_id = self.tokenizer.token_to_id("<|endoftext|>")
        self.pad_token_id = self.eos_token_id
      try:
        self.tokenizer.pad_token_id = self.pad_token_id
        self.tokenizer.eos_token_id = self.eos_token_id
      except Exception:
        pass

    def encode(self, text, is_bos=True, prefill_lengths=None):
      import numpy as np
      if hasattr(self.tokenizer, "encode"):
        res = self.tokenizer.encode(text)
        token_ids = res.ids if hasattr(res, "ids") else res
      else:
        token_ids = []
      bos_id = getattr(self.tokenizer, "bos_token_id", None)
      if is_bos and bos_id is not None:
        token_ids = [bos_id] + list(token_ids)
      true_length = len(token_ids)
      target_len = prefill_lengths[0] if prefill_lengths else true_length
      pad_id = getattr(self.tokenizer, "pad_token_id", 0) or 0
      padded = list(token_ids) + [pad_id] * max(0, target_len - true_length)
      return np.array(padded[:target_len], dtype=np.int32), true_length

    def decode(self, token_ids):
      if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
      if hasattr(self.tokenizer, "decode"):
        try:
          return self.tokenizer.decode(token_ids, skip_special_tokens=True)
        except TypeError:
          return self.tokenizer.decode(token_ids)
      return ""

  config_lib = SimpleNamespace()  # not used directly in decoupled tests
  engine_api = SimpleNamespace(Engine=Engine, ResultTokens=ResultTokens)
  token_utils = SimpleNamespace(HuggingFaceTokenizer=HuggingFaceTokenizer)
  tokenizer_api = SimpleNamespace()  # placeholder
  token_params_ns = SimpleNamespace(TokenizerParameters=TokenizerParameters, TokenizerType=TokenizerType)

  # Mark these stub namespaces so callers can detect stubbed jetstream components.
  setattr(config_lib, "_IS_STUB", True)
  setattr(engine_api, "_IS_STUB", True)
  setattr(token_utils, "_IS_STUB", True)
  setattr(tokenizer_api, "_IS_STUB", True)
  setattr(token_params_ns, "_IS_STUB", True)
  return config_lib, engine_api, token_utils, tokenizer_api, token_params_ns


def jetstream():
  """Return JetStream modules or stub implementations.

  When running in decoupled mode or when JetStream dependencies are not
  available, this function returns lightweight stub namespaces that mimic the
  real APIs closely enough for tests and non-serving code paths.
  """
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
    # Mark real modules as not stubs so consumers can detect the difference.
    try:
      setattr(config_lib, "_IS_STUB", False)
    except Exception:  # pylint: disable=broad-exception-caught
      pass
    try:
      setattr(engine_api, "_IS_STUB", False)
    except Exception:  # pylint: disable=broad-exception-caught
      pass
    try:
      setattr(token_utils, "_IS_STUB", False)
    except Exception:  # pylint: disable=broad-exception-caught
      pass
    try:
      setattr(tokenizer_api, "_IS_STUB", False)
    except Exception:  # pylint: disable=broad-exception-caught
      pass
    token_params_ns = SimpleNamespace(TokenizerParameters=TokenizerParameters, TokenizerType=TokenizerType)
    setattr(token_params_ns, "_IS_STUB", False)
    return config_lib, engine_api, token_utils, tokenizer_api, token_params_ns
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

    def list_blobs(self, *a, **k):  # pylint: disable=unused-argument
      return iter([])

  def _stub_upload_many_from_filenames(*_a, **_k):
    """No-op stub for transfer_manager.upload_many_from_filenames."""
    return []

  transfer_manager_stub = SimpleNamespace(
      upload_many_from_filenames=_stub_upload_many_from_filenames,
      _IS_STUB=True,
  )

  return SimpleNamespace(Client=_StubClient, transfer_manager=transfer_manager_stub, _IS_STUB=True)


def gcs_storage():
  """Return google.cloud.storage module (with transfer_manager attached) or stub.

  The returned object always exposes both ``.Client`` and ``.transfer_manager``
  so callers can use ``storage.transfer_manager.upload_many_from_filenames(...)``
  without an extra import. ``transfer_manager`` is a submodule of
  ``google.cloud.storage`` and is not auto-imported by ``from google.cloud
  import storage``; we explicitly import and attach it here.
  """
  # In decoupled mode always prefer the stub, even if the library is installed,
  # to avoid accidental GCS calls in tests or local runs.
  if is_decoupled():  # fast path
    print("[DECOUPLED NO-OP] gcs_storage: using stubs.")
    return _gcs_stubs()

  try:  # pragma: no cover - attempt real import when not decoupled
    from google.cloud import storage  # type: ignore  # pylint: disable=import-outside-toplevel
    from google.cloud.storage import transfer_manager  # type: ignore  # pylint: disable=import-outside-toplevel

    setattr(storage, "transfer_manager", transfer_manager)
    setattr(storage, "_IS_STUB", False)
    return storage
  except Exception:  # ModuleNotFoundError / ImportError for partial installs  # pylint: disable=broad-exception-caught
    print("[NO-OP] gcs_storage dependency missing; using stubs.")
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

  def _import():
    from ml_goodput_measurement import goodput, monitoring  # type: ignore  # pylint: disable=import-outside-toplevel

    return goodput, monitoring, False

  return _import_or_stub(
      _import,
      _goodput_stubs,
      label="ml_goodput_measurement",
      stub_if_decoupled=False,
  )


__all__ = ["is_decoupled", "jetstream", "gcs_storage", "goodput_modules"]

# ---------------- Cloud Monitoring (monitoring_v3 / metric_pb2) -----------------


def _monitoring_stubs():  # pragma: no cover - simple placeholders
  """Return stub implementations for Cloud Monitoring APIs."""

  class GoogleAPIError(Exception):
    """Stub GoogleAPIError mirroring the real exception name."""

  class _StubMonitoringV3:
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

  class _StubMetricPB2:
    """Dummy metric_pb2 module namespace."""

    class Metric:

      def __init__(self, *a, **k):  # pylint: disable=unused-argument
        del a, k

  class _StubMonitoredResourcePB2:
    """Dummy monitored_resource_pb2 module namespace."""

    class MonitoredResource:

      def __init__(self, *a, **k):  # pylint: disable=unused-argument
        del a, k

  return _StubMonitoringV3(), _StubMetricPB2(), _StubMonitoredResourcePB2(), GoogleAPIError, True


def monitoring_modules():
  """Return monitoring modules or stubs.

  Stubs only if decoupled AND dependency missing; if not decoupled and missing ->
  re-raise.
  """

  def _import():  # Attempt real imports first
    from google.cloud import monitoring_v3  # type: ignore  # pylint: disable=import-outside-toplevel
    from google.api import metric_pb2, monitored_resource_pb2  # type: ignore  # pylint: disable=import-outside-toplevel
    from google.api_core.exceptions import GoogleAPIError  # type: ignore  # pylint: disable=import-outside-toplevel

    return monitoring_v3, metric_pb2, monitored_resource_pb2, GoogleAPIError, False

  return _import_or_stub(_import, _monitoring_stubs, label="monitoring", stub_if_decoupled=False)


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

  def _import():
    from maxtext.common.gcp_workload_monitor import GCPWorkloadMonitor  # type: ignore  # pylint: disable=import-outside-toplevel

    return GCPWorkloadMonitor, False

  return _import_or_stub(
      _import,
      _workload_monitor_stub,
      label="workload_monitor",
      stub_if_decoupled=True,
      stub_on_error_when_not_decoupled=True,
  )


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
      print("[DECOUPLED NO-OP] skipping Vertex Tensorboard configuration.")

  return VertexTensorboardManager, True


def vertex_tensorboard_modules():
  """Return (VertexTensorboardManager, is_stub).

  Decoupled or missing dependency -> stub class with no-op configure method.
  """

  def _import():
    from maxtext.common.vertex_tensorboard import VertexTensorboardManager  # type: ignore  # pylint: disable=import-outside-toplevel

    return VertexTensorboardManager, False

  return _import_or_stub(
      _import,
      _vertex_tb_stub,
      label="vertex_tensorboard",
      stub_if_decoupled=True,
      stub_on_error_when_not_decoupled=True,
  )


vertex_tensorboard_components = vertex_tensorboard_modules  # backward-compatible alias

__all__.append("vertex_tensorboard_modules")
__all__.append("vertex_tensorboard_components")

# ---------------- ML Diagnostics (google_cloud_mldiagnostics) -----------------


def _mldiagnostics_stub():  # pragma: no cover - simple placeholder
  """Return stub for google_cloud_mldiagnostics."""

  class _StubXprof:
    """Stub of mldiag.xprof context manager."""

    def __init__(self, *a, **k):  # pylint: disable=unused-argument
      pass

    def __enter__(self):
      return self

    def __exit__(self, *a, **k):  # pylint: disable=unused-argument
      pass

  class _StubMldiag:
    """Stub of mldiag module."""

    def xprof(self, *a, **k):  # pylint: disable=unused-argument
      """Return a stub context manager."""
      return _StubXprof()

  return _StubMldiag(), True


def mldiagnostics_modules():
  """Return (mldiag, is_stub) centralizing stub logic.

  If decoupled OR import fails, returns stub object; otherwise real module.
  """

  def _import():
    import google_cloud_mldiagnostics as mldiag  # type: ignore  # pylint: disable=import-outside-toplevel

    return mldiag, False

  return _import_or_stub(
      _import,
      _mldiagnostics_stub,
      label="mldiagnostics",
      stub_if_decoupled=True,
      stub_on_error_when_not_decoupled=True,
  )


__all__.append("mldiagnostics_modules")

# ------------------------- TensorBoardX --------------------------


class StubSummaryWriter:
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


try:
  if not is_decoupled():  # Only attempt real import when not decoupled
    from tensorboardX import writer  # type: ignore  # pylint: disable=import-outside-toplevel,unused-import

    _TENSORBOARDX_AVAILABLE = True
  else:
    raise ModuleNotFoundError("Decoupled mode skips tensorboardX import")
except Exception:  # pragma: no cover - provide stub fallback  # pylint: disable=broad-exception-caught
  _TENSORBOARDX_AVAILABLE = False

  class writer:  # pylint: disable=too-few-public-methods
    SummaryWriter = StubSummaryWriter


__all__.append("writer")
__all__.append("_TENSORBOARDX_AVAILABLE")
__all__.append("StubSummaryWriter")
