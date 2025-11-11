"""Top-level shim for importing test_utils

This shim lets test modules import `maxtext.tests`.

"""

from importlib import import_module as _imp

try:
    test_utils = _imp("maxtext.tests.test_utils")  # noqa: F401
except Exception:  # pragma: no cover - fail silently if tests not present
    pass

__all__ = ["test_utils"]
