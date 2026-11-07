# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for `save_weights_to_checkpoint` in checkpoint conversion.

Regression coverage for https://github.com/AI-Hypercomputer/maxtext/issues/5106:
with `--lazy_load_tensors` and a single simulated device, `LazyTensor` leaves were
never materialized and the Orbax v1 save failed with `NoEntryError`.
"""

import numpy as np
import orbax.checkpoint as ocp
import pytest

from maxtext.checkpoint_conversion.to_maxtext import LazyTensor
from maxtext.checkpoint_conversion.utils.utils import materialize_lazy_weights, save_weights_to_checkpoint

pytestmark = [pytest.mark.decoupled_target]


def _lazy(arr: np.ndarray, calls: list, name: str) -> LazyTensor:
  """Wraps `arr` in a LazyTensor whose loader records each invocation in `calls`."""

  def load():
    calls.append(name)
    return arr.copy()

  return LazyTensor(load, shape=arr.shape, dtype=arr.dtype, name=name)


class TestMaterializeLazyWeights:
  """Tests for `materialize_lazy_weights`."""

  def test_materializes_lazy_leaves_and_keeps_arrays(self):
    expected = np.arange(6, dtype=np.float32).reshape(2, 3)
    calls = []
    plain = np.ones((2,), dtype=np.int32)
    tree = {"a": {"lazy": _lazy(expected, calls, "a.lazy"), "plain": plain}, "scalar": np.float32(2.0)}

    out = materialize_lazy_weights(tree)

    assert calls == ["a.lazy"]
    assert isinstance(out["a"]["lazy"], np.ndarray)
    np.testing.assert_array_equal(out["a"]["lazy"], expected)
    assert out["a"]["plain"] is plain
    assert isinstance(out["scalar"], np.ndarray)
    assert out["scalar"].shape == ()
    assert out["scalar"] == np.float32(2.0)

  def test_leaves_non_array_leaves_untouched(self):
    tree = {"step": 3, "name": "x"}
    assert materialize_lazy_weights(tree) == tree


class TestSaveWeightsToCheckpointSingleDevice:
  """Regression tests for the `device_count == 1` path of `save_weights_to_checkpoint`."""

  @pytest.mark.parametrize("use_ocdbt,use_zarr3", [(True, True), (False, False)])
  def test_lazy_tensors_are_saved_on_single_device(self, tmp_path, use_ocdbt, use_zarr3):
    w = np.arange(12, dtype=np.float32).reshape(3, 4)
    b = np.array([0.5, -0.5], dtype=np.float32)
    calls = []
    jax_weights = {
        "decoder": {
            "w": _lazy(w, calls, "decoder.w"),
            "b": _lazy(b, calls, "decoder.b"),
        },
        "eager": np.full((2, 2), 7.0, dtype=np.float32),
    }

    save_weights_to_checkpoint(
        str(tmp_path),
        jax_weights,
        device_count=1,
        use_ocdbt=use_ocdbt,
        use_zarr3=use_zarr3,
    )

    # Lazy loaders ran exactly once each.
    assert sorted(calls) == ["decoder.b", "decoder.w"]

    # A checkpoint was written and round-trips the materialized values.
    step_dir = tmp_path / "0"
    assert step_dir.exists(), f"No checkpoint step directory under {tmp_path}"
    restored = ocp.PyTreeCheckpointer().restore(step_dir / "items")
    # TrainState.params wraps the {"params": jax_weights} dict built by save_weights_to_checkpoint.
    params = restored["params"]["params"]
    np.testing.assert_array_equal(params["decoder"]["w"], w)
    np.testing.assert_array_equal(params["decoder"]["b"], b)
    np.testing.assert_array_equal(params["eager"], jax_weights["eager"])
    assert int(restored["step"]) == 0
