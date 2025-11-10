"""Pytest-based environment smoke test for MaxText (used esp for decoupling testing).

Checks:
  - Core imports (jax, flax, numpy)
  - Optional imports
  - JAX device enumeration
  - Format/Layout alias compatibility (MaxText.layout_compat)

Fails only on missing core imports or device query failure; alias test asserts mapping rules.
"""
from __future__ import annotations
import os, time, importlib
from MaxText.gcloud_stub import is_decoupled

CORE_IMPORTS = ["jax", "jax.numpy", "flax", "numpy"]
OPTIONAL_IMPORTS = ["transformers", "MaxText", "MaxText.pyconfig", "MaxText.maxengine"]

_defects: list[str] = []


def _import(name: str):
    t0 = time.time()
    try:
        mod = importlib.import_module(name)
        return name, mod, time.time() - t0, None
    except Exception as e:  # pragma: no cover
        return name, None, time.time() - t0, e


def test_environment_core_imports():
    results = [_import(n) for n in CORE_IMPORTS]
    missing = [n for (n, m, _, err) in results if m is None]
    if missing:
        raise AssertionError(f"Missing core imports: {missing}")


def test_environment_optional_imports():
    results = [_import(n) for n in OPTIONAL_IMPORTS]
    for (n, m, dt, err) in results:
        if err is not None:
            _defects.append(f"{n} FAIL: {err}")
        else:
            if dt > 8.0:
                _defects.append(f"{n} SLOW_IMPORT ({dt:.1f}s)")


def test_format_layout_alias():
    # Verify Format/Layout behavior based on JAX version threshold logic (>=0.7.0 native Format else alias)
    try:
        import jax  # type: ignore
        from MaxText.layout_compat import Format, Layout, DLL  # type: ignore
    except Exception as e:  # pragma: no cover
        raise AssertionError(f"Failed to import layout compatibility modules: {e}")

    ver = getattr(jax, "__version_info__", None)
    assert ver is not None, "jax.__version_info__ missing"

    sample = Format(DLL.AUTO)  # should construct regardless of version due to alias logic
    # For versions < 0.7.0 Format should be Layout (same type); for >=0.7.0 it may differ.
    if ver < (0, 7, 0):
        assert isinstance(sample, Layout), "Format should alias Layout for jax < 0.7.0"
    else:
        # We can't assert specific class name without importing jax.experimental.layout internals; just ensure not Layout instance.
        assert not isinstance(sample, Layout), "Format should NOT alias Layout for jax >= 0.7.0"

    # AUTO sentinel presence path differs by version; ensure attribute access does not raise.
    dll_attr = "layout" if ver >= (0, 7, 0) else "device_local_layout"
    _ = getattr(sample, dll_attr)


def test_jax_devices():
    try:
        import jax  # type: ignore
    except Exception as e:  # pragma: no cover
        raise AssertionError(f"jax not importable for device test: {e}")
    try:
        devices = jax.devices()
    except Exception as e:  # pragma: no cover
        raise AssertionError(f"jax.devices() failed: {e}")
    assert len(devices) >= 1, "No JAX devices found"


def test_decoupled_flag_consistency():
    decoupled = is_decoupled()
    # Soft check only; logic exercised in other tests.
    if decoupled:
        pass
    else:
        pass


def test_report_defects():
    if _defects:
        print("Environment optional issues:\n" + "\n".join(_defects))
