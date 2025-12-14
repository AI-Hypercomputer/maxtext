# Copyright 2024 Google LLC
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

"""Dynamic tracing to capture executed files during training.

Usage:
    # Enable via environment variable:
    MAXTEXT_TRACE=1 python -m MaxText.train ...

    # Or import and use directly:
    from MaxText.tracer import start_tracing, stop_tracing, get_traced_files
    start_tracing()
    # ... run training ...
    stop_tracing()
    files = get_traced_files()
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Set


# Global state
_traced_files: Set[str] = set()
_lock = threading.Lock()
_tracing_active = False

# Directories to exclude from tracing
EXCLUDE_PREFIXES = (
    # Standard library
    "/usr/lib/python",
    "/usr/local/lib/python",
    # Site packages (but not maxtext/flax)
    "site-packages/jax/",
    "site-packages/numpy/",
    "site-packages/tensorflow/",
    "site-packages/orbax/",
    "site-packages/absl/",
    "site-packages/google/",
    # Built-in modules
    "<frozen",
    "<string>",
)

# Directories to always include
INCLUDE_PATTERNS = (
    "maxtext",
    "MaxText",
    "flax",
)


def _should_trace(filename: str) -> bool:
    """Determine if a file should be traced."""
    if not filename:
        return False

    # Always include files matching our patterns
    for pattern in INCLUDE_PATTERNS:
        if pattern in filename:
            return True

    # Exclude standard library and common packages
    for prefix in EXCLUDE_PREFIXES:
        if prefix in filename:
            return False

    # Include everything else (user code)
    return True


def _trace_calls(frame, event: str, arg):
    """Profile function called for each Python event."""
    global _traced_files

    if event != "call":
        return

    filename = frame.f_code.co_filename

    if not _should_trace(filename):
        return

    with _lock:
        _traced_files.add(filename)


def start_tracing() -> None:
    """Start tracing executed files."""
    global _tracing_active, _traced_files

    if _tracing_active:
        return

    _traced_files = set()

    sys.setprofile(_trace_calls)
    threading.setprofile(_trace_calls)
    _tracing_active = True
    print("[tracer] Started tracing. Output: ~/maxtext/unique_files.txt")


def stop_tracing() -> None:
    """Stop tracing and write unique_files.txt."""
    global _tracing_active

    if not _tracing_active:
        return

    sys.setprofile(None)
    threading.setprofile(None)
    _tracing_active = False

    # Write file list to ~/maxtext/unique_files.txt
    output_path = Path.home() / "maxtext" / "unique_files.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for file in sorted(_traced_files):
            f.write(file + "\n")

    print(f"[tracer] Wrote {len(_traced_files)} files to {output_path}")


def get_traced_files() -> Set[str]:
    """Return set of all traced file paths."""
    return _traced_files.copy()


def maybe_start_tracing() -> bool:
    """Start tracing if MAXTEXT_TRACE env var is set.

    Returns:
        True if tracing was started, False otherwise.
    """
    trace_env = os.environ.get("MAXTEXT_TRACE", "").lower()
    if trace_env in ("1", "true", "yes"):
        start_tracing()
        return True
    return False


def maybe_stop_tracing() -> None:
    """Stop tracing and write results if tracing was active."""
    if _tracing_active:
        stop_tracing()
