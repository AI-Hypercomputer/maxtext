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

"""
Tests for verifying losses and gradients match using/without using tiling methods:
- Gradient accumulation (GA)
- Vocabulary tiling (VT)
"""

import unittest
import pytest

from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp

from maxtext.configs import pyconfig
from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils
from maxtext.utils import maxtext_utils_nnx
from maxtext.utils import model_creation_utils
from maxtext.utils.vocabulary_tiling import vocab_tiling_nnx_loss

from tests.utils.test_helpers import get_test_config_path


class LossAndGradientCorrectnessTest(unittest.TestCase):
  """
  Unit tests for verifying loss and gradient correctness of:
  - Gradient accumulation (GA)
  - Vocabulary tiling (VT)
  """

  def setUp(self):
    """
    Set up common configurations and dummy data for the tests.
    """
    self.base_config = [
        None,
        get_test_config_path(),
        "base_emb_dim=32",
        "vocab_size=128",
    ]
    self.rng = jax.random.PRNGKey(1234)
    self.batch_size = 1
    self.seq_len = 64
    self.rtol = 1e-2
    self.atol = 1e-2

  @pytest.mark.tpu_only
  def test_vocab_tiling_nnx_loss(self):
    """
    Tests loss correctness of vocab_tiling_nnx_loss on the NNX path: the tiled loss
    should match the non-tiled cross-entropy computed from the same hidden states.
    """
    cfg = pyconfig.initialize(
        self.base_config,
        run_name="nnx_vocab_tiling_loss",
        enable_checkpointing=False,
        enable_dropout=False,
        max_target_length=self.seq_len,
        per_device_batch_size=self.batch_size,
        logits_via_embedding=False,
        base_num_decoder_layers=0,
        dtype="float32",
        matmul_precision="high",
        num_vocab_tiling=4,
        z_loss_multiplier=1e-4,
    )
    rng_model, rng_hidden, rng_targets = jax.random.split(self.rng, 3)
    rngs = maxtext_utils_nnx.create_nnx_rngs(cfg, rng_key=rng_model)
    mesh = maxtext_utils.get_mesh_from_config(cfg)
    model = model_creation_utils.from_config(cfg, mesh=mesh, rngs=rngs)

    hidden_states = jax.random.normal(rng_hidden, (self.batch_size, self.seq_len, cfg.emb_dim), dtype=jnp.float32)
    data = {
        "targets": jax.random.randint(rng_targets, (self.batch_size, self.seq_len), 0, cfg.vocab_size),
        "targets_segmentation": jnp.ones((self.batch_size, self.seq_len)),
    }

    xent_sum_tiled, _ = vocab_tiling_nnx_loss(model, hidden_states, data, cfg, is_train=True)

    # Reference: full logits with no tiling, same masking as the tiled path.
    logits = model.logits_from_hidden_states_for_vocab_tiling(hidden_states, True, MODEL_MODE_TRAIN)
    one_hot_targets = jax.nn.one_hot(data["targets"], cfg.vocab_size)
    xent_ref, _ = max_utils.cross_entropy_with_logits(logits, one_hot_targets, z_loss=cfg.z_loss_multiplier)
    xent_sum_ref = jnp.sum(xent_ref * (data["targets_segmentation"] != 0))

    assert jnp.allclose(
        xent_sum_tiled, xent_sum_ref, rtol=self.rtol, atol=self.atol
    ), f"NNX vocab tiling loss {xent_sum_tiled} does not match non-tiled reference {xent_sum_ref}."


class VocabTilingNNXTest(unittest.TestCase):
  """Loss + gradient parity for the NNX vocab-tiling `custom_vjp` path.

  Compares two computations against the same NNX model:
    - reference: full-vocab `model.logits_from_hidden_states_for_vocab_tiling(...)` then xent over the whole vocab.
    - tiled: `vocab_tiling_nnx_loss(...)` which scans over `num_vocab_tiling` chunks
      and uses a `custom_vjp` for the backward.

  Both paths share the same params; the test checks that loss values and parameter
  gradients match within tolerance, exercising both forward and backward.
  """

  def setUp(self):
    self.base_config = [None, get_test_config_path()]
    self.rng = jax.random.PRNGKey(1234)
    # Global batch must divide fsdp axis (= jax.device_count() by default), so the
    # batch sharding constraints inside vocab_tiling_nnx_loss are satisfied.
    self.batch_size = jax.device_count()
    self.seq_len = 64
    self.rtol = 1e-2
    self.atol = 1e-2

  def _build_cfg_and_model(
      self,
      *,
      num_vocab_tiling=4,
      logits_via_embedding=False,
      z_loss_multiplier=1e-4,
  ):
    """Build a pyconfig + matching NNX `Transformer` for the test."""
    cfg = pyconfig.initialize(
        self.base_config,
        run_name=f"vt_nnx_n{num_vocab_tiling}_emb{logits_via_embedding}_z{z_loss_multiplier}",
        enable_checkpointing=False,
        enable_dropout=False,
        max_target_length=self.seq_len,
        per_device_batch_size=1,
        logits_via_embedding=logits_via_embedding,
        base_num_decoder_layers=0,
        dtype="float32",
        matmul_precision="high",
        num_vocab_tiling=num_vocab_tiling,
        z_loss_multiplier=z_loss_multiplier,
    )
    mesh = maxtext_utils.get_mesh_from_config(cfg)
    rngs = maxtext_utils_nnx.create_nnx_rngs(cfg)
    with nn_partitioning.axis_rules(cfg.logical_axis_rules):
      model = model_creation_utils.from_config(cfg, mesh=mesh, rngs=rngs, model_mode=MODEL_MODE_TRAIN)
    return cfg, model

  def _make_inputs(self, cfg, *, dtype=jnp.float32, pad_half=False):
    """Synthetic hidden_states/labels/segmentation; `pad_half=True` zeros the back half of seg."""
    rng_hidden, rng_targets = jax.random.split(self.rng)
    hidden_states = jax.random.normal(rng_hidden, (self.batch_size, self.seq_len, cfg.emb_dim), dtype=dtype)
    labels = jax.random.randint(rng_targets, (self.batch_size, self.seq_len), 0, cfg.vocab_size)
    if pad_half:
      half = self.seq_len // 2
      segmentation = jnp.concatenate(
          [
              jnp.ones((self.batch_size, half), dtype=jnp.int32),
              jnp.zeros((self.batch_size, self.seq_len - half), dtype=jnp.int32),
          ],
          axis=1,
      )
    else:
      segmentation = jnp.ones((self.batch_size, self.seq_len), dtype=jnp.int32)
    return hidden_states, labels, segmentation

  def _reference_loss_fn(self, cfg, graphdef, rest, hidden_states, labels, segmentation):
    """Full-vocab xent loss closure (params, hidden_states) -> scalar loss."""

    def loss_fn(p, h):
      local_model = nnx.merge(graphdef, p, rest, copy=True)
      logits = local_model.logits_from_hidden_states_for_vocab_tiling(h, True, "train")
      one_hot = jax.nn.one_hot(labels, cfg.vocab_size)
      xent, _ = max_utils.cross_entropy_with_logits(logits, one_hot, z_loss=cfg.z_loss_multiplier)
      return jnp.sum(xent * (segmentation != 0))

    return loss_fn

  def _tiled_loss_fn(self, cfg, graphdef, rest, hidden_states, labels, segmentation):
    """vocab_tiling_nnx_loss closure (params, hidden_states) -> scalar loss."""
    # hidden_states unused at the closure boundary (it comes via h), but kept in the
    # signature so the two closures are callable interchangeably.
    del hidden_states
    data = {"targets": labels, "targets_segmentation": segmentation}

    def loss_fn(p, h):
      local_model = nnx.merge(graphdef, p, rest, copy=True)
      total_loss, _ = vocab_tiling_nnx_loss(local_model, h, data, cfg, is_train=True)
      return total_loss

    return loss_fn

  def _split_and_axes(self, cfg, model):
    """Common boilerplate: split the model and bind the logical axis rules."""
    graphdef, params, rest = nnx.split(model, nnx.Param, ...)
    return graphdef, params, rest

  def _assert_pytrees_close(self, ref, tiled, msg, *, rtol=None, atol=None):
    rtol = self.rtol if rtol is None else rtol
    atol = self.atol if atol is None else atol
    leaves_close = jax.tree_util.tree_map(lambda x, y: jnp.allclose(x, y, rtol=rtol, atol=atol), ref, tiled)
    if not all(jax.tree_util.tree_leaves(leaves_close)):
      raise AssertionError(msg)

  @staticmethod
  def _vg(fn, *, argnums=0):
    """value_and_grad wrapped in jit. Eager value_and_grad trips an IndivisibleError
    on the fsdp reshape inside vocab_tiling_nnx_loss; jit lets XLA reshard cleanly,
    which is also how train.py runs it."""
    return jax.jit(jax.value_and_grad(fn, argnums=argnums))

  @staticmethod
  def _g(fn, *, argnums=0):
    """grad wrapped in jit — see `_vg`."""
    return jax.jit(jax.grad(fn, argnums=argnums))

  def _run_parity(self, *, logits_via_embedding):
    """Compare full-vocab xent loss/grads against the tiled custom_vjp path."""
    cfg, model = self._build_cfg_and_model(num_vocab_tiling=4, logits_via_embedding=logits_via_embedding)
    hidden_states, labels, segmentation = self._make_inputs(cfg)
    graphdef, params, rest = self._split_and_axes(cfg, model)

    ref_loss_fn = self._reference_loss_fn(cfg, graphdef, rest, hidden_states, labels, segmentation)
    tile_loss_fn = self._tiled_loss_fn(cfg, graphdef, rest, hidden_states, labels, segmentation)

    with nn_partitioning.axis_rules(cfg.logical_axis_rules):
      ref_loss, ref_grads = self._vg(ref_loss_fn)(params, hidden_states)
      tile_loss, tile_grads = self._vg(tile_loss_fn)(params, hidden_states)

    assert jnp.allclose(
        ref_loss, tile_loss, rtol=self.rtol, atol=self.atol
    ), f"Losses differ: ref={ref_loss} tiled={tile_loss}"
    self._assert_pytrees_close(ref_grads, tile_grads, "Param gradients differ between full-vocab and tiled paths.")

  # ---------- Original parity tests (params gradient under both embedding modes) ----------

  @pytest.mark.tpu_only
  def test_nnx_vocab_tiling_non_tied_embedding(self):
    """custom_vjp parity for non-tied embedding (separate logits_dense)."""
    self._run_parity(logits_via_embedding=False)

  @pytest.mark.tpu_only
  def test_nnx_vocab_tiling_tied_embedding(self):
    """custom_vjp parity when logits share the input embedding table."""
    self._run_parity(logits_via_embedding=True)

  # ---------- Coverage expansion ----------

  @pytest.mark.tpu_only
  def test_nnx_vocab_tiling_total_z_loss_value_parity(self):
    """The second tuple element (total_z_loss) must match the full-vocab reference."""
    cfg, model = self._build_cfg_and_model(num_vocab_tiling=4)
    hidden_states, labels, segmentation = self._make_inputs(cfg)
    graphdef, params, rest = self._split_and_axes(cfg, model)
    data = {"targets": labels, "targets_segmentation": segmentation}

    def _ref(p, h):
      local_model = nnx.merge(graphdef, p, rest, copy=True)
      logits = local_model.logits_from_hidden_states_for_vocab_tiling(h, True, "train")
      one_hot = jax.nn.one_hot(labels, cfg.vocab_size)
      xent_ref, z_ref = max_utils.cross_entropy_with_logits(logits, one_hot, z_loss=cfg.z_loss_multiplier)
      return jnp.sum(xent_ref * (segmentation != 0)), jnp.sum(z_ref * (segmentation != 0))

    def _tile(p, h):
      local_model = nnx.merge(graphdef, p, rest, copy=True)
      return vocab_tiling_nnx_loss(local_model, h, data, cfg, is_train=True)

    with nn_partitioning.axis_rules(cfg.logical_axis_rules):
      ref_total_loss, ref_total_z_loss = jax.jit(_ref)(params, hidden_states)
      tile_total_loss, tile_total_z_loss = jax.jit(_tile)(params, hidden_states)

    assert jnp.allclose(ref_total_loss, tile_total_loss, rtol=self.rtol, atol=self.atol)
    assert jnp.allclose(
        ref_total_z_loss, tile_total_z_loss, rtol=self.rtol, atol=self.atol
    ), f"total_z_loss differs: ref={ref_total_z_loss} tiled={tile_total_z_loss}"

  @pytest.mark.tpu_only
  def test_nnx_vocab_tiling_padded_segmentation(self):
    """Half-padded segmentation: mask actually changes the loss, and parity holds."""
    cfg, model = self._build_cfg_and_model(num_vocab_tiling=4)

    # Compare unpadded vs padded loss to confirm the mask is wired through.
    hs, labels, full_seg = self._make_inputs(cfg, pad_half=False)
    _, _, pad_seg = self._make_inputs(cfg, pad_half=True)
    graphdef, params, rest = self._split_and_axes(cfg, model)

    def _tile_loss_only(p, h, seg):
      local_model = nnx.merge(graphdef, p, rest, copy=True)
      total, _ = vocab_tiling_nnx_loss(
          local_model, h, {"targets": labels, "targets_segmentation": seg}, cfg, is_train=True
      )
      return total

    with nn_partitioning.axis_rules(cfg.logical_axis_rules):
      full_loss = jax.jit(_tile_loss_only)(params, hs, full_seg)
      pad_loss = jax.jit(_tile_loss_only)(params, hs, pad_seg)
    assert float(pad_loss) < float(
        full_loss
    ), f"Padded loss should be strictly smaller (fewer tokens contribute). full={full_loss} pad={pad_loss}"

    # Now check parity against the full-vocab reference using the padded mask.
    ref_loss_fn = self._reference_loss_fn(cfg, graphdef, rest, hs, labels, pad_seg)
    tile_loss_fn = self._tiled_loss_fn(cfg, graphdef, rest, hs, labels, pad_seg)
    with nn_partitioning.axis_rules(cfg.logical_axis_rules):
      ref_loss, ref_grads = self._vg(ref_loss_fn)(params, hs)
      tile_loss, tile_grads = self._vg(tile_loss_fn)(params, hs)
    assert jnp.allclose(ref_loss, tile_loss, rtol=self.rtol, atol=self.atol)
    self._assert_pytrees_close(ref_grads, tile_grads, "Padded-segmentation gradients differ.")

  @pytest.mark.tpu_only
  def test_nnx_vocab_tiling_grad_over_hidden_states(self):
    """Gradient w.r.t. hidden_states (argnums=1) matches the reference: exercises the
    custom_vjp's hidden_states cotangent, which the params-only tests don't reach."""
    cfg, model = self._build_cfg_and_model(num_vocab_tiling=4)
    hidden_states, labels, segmentation = self._make_inputs(cfg)
    graphdef, params, rest = self._split_and_axes(cfg, model)

    ref_loss_fn = self._reference_loss_fn(cfg, graphdef, rest, hidden_states, labels, segmentation)
    tile_loss_fn = self._tiled_loss_fn(cfg, graphdef, rest, hidden_states, labels, segmentation)
    with nn_partitioning.axis_rules(cfg.logical_axis_rules):
      ref_grad_h = self._g(ref_loss_fn, argnums=1)(params, hidden_states)
      tile_grad_h = self._g(tile_loss_fn, argnums=1)(params, hidden_states)

    assert ref_grad_h.shape == hidden_states.shape
    assert tile_grad_h.shape == hidden_states.shape
    assert ref_grad_h.dtype == hidden_states.dtype
    assert tile_grad_h.dtype == hidden_states.dtype
    assert jnp.allclose(ref_grad_h, tile_grad_h, rtol=self.rtol, atol=self.atol), "grad_hidden_states diverged"

  @pytest.mark.tpu_only
  def test_nnx_vocab_tiling_bf16_hidden_states(self):
    """bf16 hidden_states: loss/grad parity holds and the grad keeps the bf16 dtype."""
    cfg, model = self._build_cfg_and_model(num_vocab_tiling=4)
    hidden_states, labels, segmentation = self._make_inputs(cfg, dtype=jnp.bfloat16)
    graphdef, params, rest = self._split_and_axes(cfg, model)

    ref_loss_fn = self._reference_loss_fn(cfg, graphdef, rest, hidden_states, labels, segmentation)
    tile_loss_fn = self._tiled_loss_fn(cfg, graphdef, rest, hidden_states, labels, segmentation)
    with nn_partitioning.axis_rules(cfg.logical_axis_rules):
      ref_loss, ref_grad_h = self._vg(ref_loss_fn, argnums=1)(params, hidden_states)
      tile_loss, tile_grad_h = self._vg(tile_loss_fn, argnums=1)(params, hidden_states)

    # bf16 has ~3 decimal digits — loosen tolerance.
    assert jnp.allclose(ref_loss, tile_loss, rtol=5e-2, atol=5e-2)
    assert tile_grad_h.dtype == jnp.bfloat16, f"grad cast to primal dtype expected bf16, got {tile_grad_h.dtype}"
    assert jnp.allclose(ref_grad_h, tile_grad_h, rtol=5e-2, atol=5e-2)

  @pytest.mark.tpu_only
  def test_nnx_vocab_tiling_z_loss_zero(self):
    """z_loss=0: total_z_loss is exactly zero; loss/grad parity still holds."""
    cfg, model = self._build_cfg_and_model(num_vocab_tiling=4, z_loss_multiplier=0.0)
    hidden_states, labels, segmentation = self._make_inputs(cfg)
    graphdef, params, rest = self._split_and_axes(cfg, model)
    data = {"targets": labels, "targets_segmentation": segmentation}

    def _tile_fn(p, h):
      local_model = nnx.merge(graphdef, p, rest, copy=True)
      return vocab_tiling_nnx_loss(local_model, h, data, cfg, is_train=True)

    with nn_partitioning.axis_rules(cfg.logical_axis_rules):
      total_loss, total_z_loss = jax.jit(_tile_fn)(params, hidden_states)
    assert float(total_z_loss) == 0.0, f"z_loss=0 but tile path returned {total_z_loss}"
    assert float(total_loss) > 0.0  # cross-entropy on random logits should be positive

    ref_loss_fn = self._reference_loss_fn(cfg, graphdef, rest, hidden_states, labels, segmentation)
    tile_loss_fn = self._tiled_loss_fn(cfg, graphdef, rest, hidden_states, labels, segmentation)
    with nn_partitioning.axis_rules(cfg.logical_axis_rules):
      ref_loss, ref_grads = self._vg(ref_loss_fn)(params, hidden_states)
      tile_loss, tile_grads = self._vg(tile_loss_fn)(params, hidden_states)
    assert jnp.allclose(ref_loss, tile_loss, rtol=self.rtol, atol=self.atol)
    self._assert_pytrees_close(ref_grads, tile_grads, "z_loss=0 gradients differ.")

  @pytest.mark.tpu_only
  def test_nnx_vocab_tiling_other_params_get_zero_grad(self):
    """Carve-out invariant: non-head params get zero grad, head params don't.

    logits_from_hidden_states_for_vocab_tiling only uses the output-head params, so
    the loss gradient for every other param must be exactly zero. The "at least one
    head grad is non-zero" check guards against a bug that just zeros everything.
    """
    cfg, model = self._build_cfg_and_model(num_vocab_tiling=4)
    hidden_states, labels, segmentation = self._make_inputs(cfg)
    graphdef, params, rest = self._split_and_axes(cfg, model)

    tile_loss_fn = self._tiled_loss_fn(cfg, graphdef, rest, hidden_states, labels, segmentation)
    with nn_partitioning.axis_rules(cfg.logical_axis_rules):
      _, tile_grads = self._vg(tile_loss_fn)(params, hidden_states)

    head_keywords = ("token_embedder", "shared_embedding", "decoder_norm", "logits_dense")
    head_nonzero_seen = False
    for path, leaf in jax.tree_util.tree_leaves_with_path(tile_grads):
      path_str = jax.tree_util.keystr(path)
      is_head = any(kw in path_str for kw in head_keywords)
      if is_head:
        if jnp.any(leaf != 0):
          head_nonzero_seen = True
      else:
        assert jnp.all(leaf == 0), f"non-head leaf {path_str} has non-zero grad — carve-out is wrong"
    assert head_nonzero_seen, "expected at least one head leaf with non-zero grad; got all zeros"

  @pytest.mark.tpu_only
  def test_nnx_vocab_tiling_num_vocab_tiling_variants(self):
    """Different num_vocab_tiling values (2, 4, 8) all produce identical loss + grads."""
    losses = []
    grads_list = []
    for n in (2, 4, 8):
      cfg, model = self._build_cfg_and_model(num_vocab_tiling=n)
      hidden_states, labels, segmentation = self._make_inputs(cfg)
      graphdef, params, rest = self._split_and_axes(cfg, model)
      tile_loss_fn = self._tiled_loss_fn(cfg, graphdef, rest, hidden_states, labels, segmentation)
      with nn_partitioning.axis_rules(cfg.logical_axis_rules):
        loss, grads = self._vg(tile_loss_fn)(params, hidden_states)
      losses.append(loss)
      grads_list.append(grads)

    base_loss = losses[0]
    base_grads = grads_list[0]
    for n, loss, grads in zip((2, 4, 8), losses, grads_list):
      assert jnp.allclose(
          loss, base_loss, rtol=self.rtol, atol=self.atol
      ), f"num_vocab_tiling={n}: loss diverges from n=2 baseline ({loss} vs {base_loss})"
      self._assert_pytrees_close(base_grads, grads, f"num_vocab_tiling={n}: grads diverge from n=2 baseline.")
