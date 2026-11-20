# Copyright 2026 Google LLC
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

"""End-to-end 1-step RL smoke test: m3 model -> M3TunixAdapter -> RLCluster -> rollout weight sync."""

from __future__ import annotations

import contextlib
import dataclasses
import importlib.util
import tempfile
from typing import Any
import unittest

from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec
import numpy as np
import optax
import pytest

from maxtext.m3.core import checkpoint as m3_checkpoint
from maxtext.m3.infra import scan as m3_scan
from maxtext.m3.models.granitemoe import modeling_granitemoe
from maxtext.m3.models.qwen3 import modeling_qwen3
from maxtext.m3.train.tunix_adapter import M3TunixAdapter


def _has_real_tunix() -> bool:
  """Returns True when the full `google-tunix` RL package is importable."""
  try:
    return importlib.util.find_spec("tunix.rl.rl_cluster") is not None
  except (ImportError, ValueError):
    return False


HAS_TUNIX = _has_real_tunix()

_PAD_ID = 0
_BOS_ID = 1
_EOS_ID = 2


class _DeterministicByteTokenizer:
  """Offline deterministic tokenizer satisfying `tunix.generate.tokenizer_adapter.TokenizerAdapter`."""

  def __init__(self, vocab_size: int = 64):
    self._vocab_size = vocab_size

  def encode(self, text: str, **unused_kwargs: Any) -> list[int]:
    return [_BOS_ID] + [3 + (ord(ch) % (self._vocab_size - 4)) for ch in text[:5]]

  def decode(self, ids: list[int], **unused_kwargs: Any) -> str:
    chars = [chr(97 + ((int(tok) - 3) % 26)) for tok in ids if int(tok) >= 3]
    return "".join(chars)

  def bos_id(self) -> int:
    return _BOS_ID

  def eos_id(self) -> int:
    return _EOS_ID

  def pad_id(self) -> int:
    return _PAD_ID


def _split_actor_and_rollout_meshes() -> tuple[Mesh, Mesh]:
  """Splits host CPU devices into actor trainer and rollout sampler meshes."""
  devs = jax.devices()
  if len(devs) >= 4:
    actor_mesh = Mesh(np.asarray(devs[:2]).reshape(1, 2, 1, 1), ("data", "fsdp", "tensor", "expert"))
    rollout_mesh = Mesh(np.asarray(devs[2:4]).reshape(1, 1, 2, 1), ("data", "fsdp", "tensor", "expert"))
    return actor_mesh, rollout_mesh
  single = np.asarray(devs[:1]).reshape(1, 1, 1, 1)
  mesh = Mesh(single, ("data", "fsdp", "tensor", "expert"))
  return mesh, mesh


@dataclasses.dataclass(frozen=True, kw_only=True)
class _Qwen3MiniConfig:
  """Minimal typed Qwen3 config matching `modeling_qwen3.py` and `m3/infra/scan.py`."""

  model_name: str = "qwen3-0.6b"
  vocab_size: int = 64
  emb_dim: int = 32
  mlp_dim: int = 64
  num_decoder_layers: int = 2
  num_query_heads: int = 2
  num_kv_heads: int = 2
  head_dim: int = 16
  normalization_layer_epsilon: float = 1e-6
  rope_max_timescale: float = 10000.0
  use_qk_norm: bool = True
  float32_logits: bool = True
  logits_dot_in_fp32: bool = True
  normalize_embedding_logits: bool = False
  final_logits_soft_cap: float | None = None
  cast_logits_to_fp32: bool = True
  dtype: Any = jnp.float32
  weight_dtype: Any = jnp.float32
  scan_layers: bool = False
  param_scan_axis: int = 1


def _build_qwen3(*, scan_layers: bool, mesh: Mesh, seed: int = 0) -> modeling_qwen3.Qwen3Model:
  """Builds a mini fp32 Qwen3Model placed on `mesh`."""
  cfg = _Qwen3MiniConfig(scan_layers=scan_layers, param_scan_axis=1)
  with jax.set_mesh(mesh):
    model = modeling_qwen3.create_qwen3_model(cfg, mesh=mesh, rngs=nnx.Rngs(params=seed))
    return m3_scan.maybe_apply_scan(model, cfg)


def _build_granitemoe(*, scan_layers: bool, mesh: Mesh, seed: int = 7) -> modeling_granitemoe.GraniteMoeModel:
  """Builds a mini fp32 MoE GraniteMoeModel placed on `mesh`."""
  cfg = modeling_granitemoe.GraniteMoeConfig(
      vocab_size=64,
      hidden_size=32,
      intermediate_size=48,
      num_hidden_layers=2,
      num_attention_heads=4,
      num_key_value_heads=2,
      num_local_experts=4,
      num_experts_per_tok=2,
      rms_norm_eps=1e-5,
      rope_theta=10000.0,
      embedding_multiplier=1.0,
      logits_scaling=1.5,
      residual_multiplier=0.5,
      attention_multiplier=0.125,
      tie_word_embeddings=True,
      scan_layers=scan_layers,
      dtype="float32",
      weight_dtype="float32",
      matmul_precision="highest",
  )
  with jax.set_mesh(mesh):
    return modeling_granitemoe.GraniteMoeModel(cfg, mesh=mesh, rngs=nnx.Rngs(params=seed))


def _completion_per_token_logps(
    adapter: M3TunixAdapter,
    prompt_ids: jax.Array,
    completion_ids: jax.Array,
    *,
    pad_id: int = _PAD_ID,
) -> jax.Array:
  """Computes per-token completion log-probabilities through `M3TunixAdapter`."""
  full_tokens = jnp.concatenate([prompt_ids, completion_ids], axis=1)
  valid = full_tokens != pad_id
  cumsum = jnp.cumsum(valid.astype(jnp.int32), axis=-1)
  positions = jnp.maximum(cumsum - 1, 0)
  seq_len = full_tokens.shape[1]
  attn_mask = valid[:, None, :] & jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))[None, :, :]
  logits, _ = adapter(full_tokens, positions=positions, cache=None, attention_mask=attn_mask)
  comp_len = completion_ids.shape[1]
  comp_logits = logits[:, -comp_len - 1 : -1, :].astype(jnp.float32)
  log_probs = jax.nn.log_softmax(comp_logits, axis=-1)
  return jnp.take_along_axis(log_probs, completion_ids[..., None], axis=-1).squeeze(-1)


def _sync_adapter_to_unscanned_rollout(
    actor_adapter: M3TunixAdapter,
    rollout_adapter: M3TunixAdapter,
    rollout_mesh: Mesh,
    *,
    scan_axis: int = 1,
) -> dict[str, Any]:
  """Syncs scanned or unscanned actor weights into an unscanned rollout model on `rollout_mesh`."""
  exported = actor_adapter.to_rollout_pure_dict(legacy_layer_names=True)["base"]
  dst_pure = rollout_adapter.to_rollout_pure_dict(legacy_layer_names=True)["base"]
  dst_specs = nnx.to_pure_dict(nnx.get_partition_spec(nnx.state(rollout_adapter.base, nnx.Param)))
  dst_specs_legacy = m3_checkpoint.to_legacy_layer_names(dst_specs)

  def _reshard_leaf(arr: jax.Array, spec: PartitionSpec | None) -> jax.Array:
    pspec = spec if isinstance(spec, PartitionSpec) else PartitionSpec()
    return jax.device_put(arr, NamedSharding(rollout_mesh, pspec))

  synced_legacy: dict[str, Any] = {}
  for top_key, dst_sub in dst_pure.items():
    if top_key == "decoder" and isinstance(dst_sub, dict) and "layers" in exported.get("decoder", {}):
      # Unstack scanned actor `decoder.layers` along `scan_axis` into unscanned `layers_<i>`.
      src_dec = exported["decoder"]
      dec_out: dict[str, Any] = {}
      for d_key in dst_sub:
        if d_key.startswith("layers_"):
          layer_idx = int(d_key.split("_", 1)[1])
          sliced = jax.tree_util.tree_map(
              lambda x, idx=layer_idx: jnp.take(x, idx, axis=scan_axis),
              src_dec["layers"],
          )
          dec_out[d_key] = jax.tree_util.tree_map(_reshard_leaf, sliced, dst_specs_legacy["decoder"][d_key])
        else:
          dec_out[d_key] = jax.tree_util.tree_map(_reshard_leaf, src_dec[d_key], dst_specs_legacy["decoder"][d_key])
      synced_legacy[top_key] = dec_out
    else:
      synced_legacy[top_key] = jax.tree_util.tree_map(_reshard_leaf, exported[top_key], dst_specs_legacy[top_key])

  restored_m3 = m3_checkpoint.from_legacy_layer_names(
      synced_legacy, nnx.to_pure_dict(nnx.state(rollout_adapter.base, nnx.Param))
  )
  nnx.update(rollout_adapter.base, restored_m3)
  return {"model": synced_legacy}


@pytest.mark.integration_test
class M3RlSmokeTest(parameterized.TestCase):
  """Verifies 1-step RL training, cross-mesh rollout weight sync, and policy logprob shift."""

  def setUp(self):
    super().setUp()
    self.enterContext(nnx.use_eager_sharding(True))

  @parameterized.named_parameters(
      ("qwen3_unscanned", "qwen3", False),
      ("qwen3_scanned", "qwen3", True),
      ("granitemoe_scanned", "granitemoe", True),
  )
  def test_minimal_wheel_one_step_rl_sync_shifts_rollout_logprobs(self, arch: str, scan_layers: bool):
    """Runs a 1-step GRPO update + cross-mesh rollout weight sync without requiring `tunix`."""
    actor_mesh, rollout_mesh = _split_actor_and_rollout_meshes()
    if arch == "qwen3":
      actor_base = _build_qwen3(scan_layers=scan_layers, mesh=actor_mesh, seed=0)
      ref_base = _build_qwen3(scan_layers=scan_layers, mesh=actor_mesh, seed=0)
      rollout_base = _build_qwen3(scan_layers=False, mesh=rollout_mesh, seed=99)
    else:
      actor_base = _build_granitemoe(scan_layers=scan_layers, mesh=actor_mesh, seed=7)
      ref_base = _build_granitemoe(scan_layers=scan_layers, mesh=actor_mesh, seed=7)
      rollout_base = _build_granitemoe(scan_layers=scan_layers, mesh=rollout_mesh, seed=99)

    actor = M3TunixAdapter(actor_base, pad_id=_PAD_ID, use_no_op_mappings=True)
    reference = M3TunixAdapter(ref_base, pad_id=_PAD_ID, use_no_op_mappings=True)
    rollout = M3TunixAdapter(rollout_base, pad_id=_PAD_ID, use_no_op_mappings=True)
    self.assertIsNone(actor.config)
    self.assertIsNone(reference.config)

    # Initial sync aligns disjointly initialized rollout model with step-0 actor.
    if arch == "qwen3":
      _sync_adapter_to_unscanned_rollout(actor, rollout, rollout_mesh)
    else:
      src_pure = actor.to_rollout_pure_dict(legacy_layer_names=False)["base"]
      dst_specs = nnx.to_pure_dict(nnx.get_partition_spec(nnx.state(rollout.base, nnx.Param)))
      synced = jax.tree_util.tree_map(
          lambda x, sp: jax.device_put(
              x, NamedSharding(rollout_mesh, sp if isinstance(sp, PartitionSpec) else PartitionSpec())
          ),
          src_pure,
          dst_specs,
      )
      nnx.update(rollout.base, synced)

    prompt_ids = jnp.array([[_PAD_ID, _BOS_ID, 4, 5], [_BOS_ID, 6, 7, 8]], dtype=jnp.int32)
    completion_ids = jnp.array([[9, 10, 11, _PAD_ID], [12, 13, _EOS_ID, _PAD_ID]], dtype=jnp.int32)
    comp_mask = (completion_ids != _PAD_ID).astype(jnp.float32)
    advantages = jnp.array([1.25, -0.85], dtype=jnp.float32)[:, None]

    with jax.set_mesh(rollout_mesh):
      old_rollout_logps = _completion_per_token_logps(rollout, prompt_ids, completion_ids)
    with jax.set_mesh(actor_mesh):
      ref_logps = jax.lax.stop_gradient(_completion_per_token_logps(reference, prompt_ids, completion_ids))
      np.testing.assert_allclose(old_rollout_logps, ref_logps, rtol=1e-5, atol=1e-5)

      ref_before = [jnp.copy(x) for x in jax.tree_util.tree_leaves(nnx.state(reference.base, nnx.Param))]
      actor_before = [jnp.copy(x) for x in jax.tree_util.tree_leaves(nnx.state(actor.base, nnx.Param))]
      optimizer = nnx.Optimizer(actor, optax.adamw(learning_rate=1e-2, eps=1e-4), wrt=nnx.Param)

      @nnx.jit
      def _grpo_step(model: M3TunixAdapter, opt: nnx.Optimizer):
        def _loss_fn(m: M3TunixAdapter):
          cur_logps = _completion_per_token_logps(m, prompt_ids, completion_ids)
          ratio = jnp.exp(cur_logps - jax.lax.stop_gradient(old_rollout_logps))
          clipped = jnp.clip(ratio, 0.8, 1.2)
          pg_loss = -jnp.minimum(ratio * advantages, clipped * advantages)
          kl_diff = jnp.clip(ref_logps - cur_logps, -20.0, 20.0)
          kl = jnp.exp(kl_diff) - kl_diff - 1.0
          token_loss = (pg_loss + 0.04 * kl) * comp_mask
          return jnp.sum(token_loss) / jnp.sum(comp_mask)

        loss, grads = nnx.value_and_grad(_loss_fn)(model)
        opt.update(model, grads)
        return loss

      step_loss = float(_grpo_step(actor, optimizer))
      self.assertTrue(np.isfinite(step_loss))

    actor_after = jax.tree_util.tree_leaves(nnx.state(actor.base, nnx.Param))
    ref_after = jax.tree_util.tree_leaves(nnx.state(reference.base, nnx.Param))
    for r0, r1 in zip(ref_before, ref_after):
      self.assertTrue(bool(jnp.array_equal(r0, r1)))
    max_param_delta = max(float(jnp.max(jnp.abs(a1 - a0))) for a0, a1 in zip(actor_before, actor_after))
    self.assertGreater(max_param_delta, 1e-5)

    # Sync updated actor weights to rollout mesh and verify rollout logprobs match updated policy.
    if arch == "qwen3":
      _sync_adapter_to_unscanned_rollout(actor, rollout, rollout_mesh)
    else:
      src_pure = actor.to_rollout_pure_dict(legacy_layer_names=False)["base"]
      dst_specs = nnx.to_pure_dict(nnx.get_partition_spec(nnx.state(rollout.base, nnx.Param)))
      synced = jax.tree_util.tree_map(
          lambda x, sp: jax.device_put(
              x, NamedSharding(rollout_mesh, sp if isinstance(sp, PartitionSpec) else PartitionSpec())
          ),
          src_pure,
          dst_specs,
      )
      nnx.update(rollout.base, synced)

    with jax.set_mesh(actor_mesh):
      new_actor_logps = _completion_per_token_logps(actor, prompt_ids, completion_ids)
    with jax.set_mesh(rollout_mesh):
      new_rollout_logps = _completion_per_token_logps(rollout, prompt_ids, completion_ids)

    np.testing.assert_allclose(new_rollout_logps, new_actor_logps, rtol=1e-5, atol=1e-5)
    logprob_shift = float(jnp.max(jnp.abs((new_rollout_logps - old_rollout_logps) * comp_mask)))
    self.assertGreater(logprob_shift, 1e-4)

  @unittest.skipUnless(HAS_TUNIX, "Requires google-tunix package for RLCluster and GrpoLearner")
  @parameterized.named_parameters(
      ("qwen3_scanned_chunked", "qwen3", True, 2, True),
      ("qwen3_unscanned_legacy_vllm", "qwen3", False, 0, True),
      ("qwen3_unscanned_m3_direct", "qwen3", False, 0, False),
      ("granitemoe_scanned", "granitemoe", True, 0, False),
  )
  def test_tunix_rl_cluster_to_vllm_direct_sync_one_step_e2e(
      self,
      arch: str,
      scan_layers: bool,
      chunk_size: int,
      legacy_vllm_dst: bool,
  ):
    """Executes 1 full `RLCluster` + `GrpoLearner` step matching `VllmSampler.update_params` weight sync."""
    # pylint: disable=import-outside-toplevel
    from tunix.generate import mappings as tunix_mappings
    from tunix.generate import utils as tunix_gen_utils
    from tunix.rl import common as tunix_common
    from tunix.rl import reshard as tunix_reshard
    from tunix.rl import rl_cluster as rl_cluster_lib
    from tunix.rl.grpo import grpo_learner
    from tunix.rl.rollout import base_rollout

    actor_mesh, rollout_mesh = _split_actor_and_rollout_meshes()
    if arch == "qwen3":
      actor_base = _build_qwen3(scan_layers=scan_layers, mesh=actor_mesh, seed=3)
      ref_base = _build_qwen3(scan_layers=scan_layers, mesh=actor_mesh, seed=3)
      rollout_target_base = _build_qwen3(scan_layers=False, mesh=rollout_mesh, seed=101)
    else:
      actor_base = _build_granitemoe(scan_layers=scan_layers, mesh=actor_mesh, seed=11)
      ref_base = _build_granitemoe(scan_layers=scan_layers, mesh=actor_mesh, seed=11)
      rollout_target_base = _build_granitemoe(scan_layers=scan_layers, mesh=rollout_mesh, seed=101)

    sync_target = "legacy_maxtext" if legacy_vllm_dst else "m3"
    actor_adapter = M3TunixAdapter(
        actor_base,
        pad_id=_PAD_ID,
        use_no_op_mappings=True,
        unscanned_direct_sync_target=sync_target,
    )
    ref_adapter = M3TunixAdapter(
        ref_base,
        pad_id=_PAD_ID,
        use_no_op_mappings=True,
        unscanned_direct_sync_target=sync_target,
    )
    rollout_target_adapter = M3TunixAdapter(rollout_target_base, pad_id=_PAD_ID, use_no_op_mappings=True)
    expected_dst_leaves = len(jax.tree_util.tree_leaves(nnx.state(rollout_target_adapter.base, nnx.Param)))

    sync_events: list[int] = []

    class _CpuVllmDirectSyncRollout(base_rollout.BaseRollout):
      """CPU rollout worker executing the exact `VllmSampler.update_params` weight-sync branch."""

      def __init__(
          self,
          *,
          rollout_actor: M3TunixAdapter,
          tokenizer: Any,
          mesh: Mesh,
          rollout_config: base_rollout.RolloutConfig,
      ):
        self._tokenizer = tokenizer
        self.mesh = mesh
        self.config = rollout_config
        self.rollout_model = rollout_target_adapter
        self._mapping_cfg = tunix_mappings.MappingConfig.build(
            mapping_obj=rollout_config.rollout_mapping_config,
            model=rollout_actor,
            backend="vllm_jax",
        )
        if not scan_layers and legacy_vllm_dst:
          assert self._mapping_cfg.to_hf_mappings, "Expected structural layer mappings for unscanned -> legacy sync."
        else:
          assert not self._mapping_cfg.to_hf_mappings, "Expected empty mappings for direct structural sync."
        self.update_params(nnx.state(rollout_actor, nnx.Param))

      def _build_vllm_dst_state(self) -> nnx.State:
        """Builds an `nnx.State({'model': ...})` destination matching vLLM `MaxTextForCausalLM` or `m3`."""
        raw_state = nnx.state(self.rollout_model.base, nnx.Param)
        if legacy_vllm_dst:
          return nnx.State({"model": m3_checkpoint.to_legacy_layer_names(raw_state)})
        return nnx.State({"model": raw_state})

      def update_params(
          self,
          params: Any,
          filter_types: tuple[Any, ...] | None = None,
      ) -> None:
        """Transfers raw `params` (`nnx.state(actor, nnx.Param)`) using `VllmSampler.update_params` dispatch."""
        del filter_types
        dst_state = self._build_vllm_dst_state()
        captured: dict[str, Any] = {}

        def _capture_reshard(source: Any, target: Any) -> Any:
          # Cross-mesh jax.device_put on CPU stages arrays via host memory inside sync_weights' disallow guard.
          guard_ctx = jax.transfer_guard("allow") if jax.default_backend() == "cpu" else contextlib.nullcontext()
          with guard_ctx:
            resharded = tunix_reshard.reshard_pytree(source, target)
          captured["synced"] = resharded
          return resharded

        if self._mapping_cfg.to_hf_mappings:
          src_weights = params
          if self._mapping_cfg.preprocess_src_state is not None:
            src_weights = self._mapping_cfg.preprocess_src_state(src_weights)
          tunix_gen_utils.transfer_state_with_mappings(
              src_state=src_weights,
              dst_state=dst_state,
              key_mappings=self._mapping_cfg.to_hf_mappings,
              key_mapping_hook_fns=self._mapping_cfg.to_hf_hook_fns,
              transpose_keys=self._mapping_cfg.to_hf_transpose_keys,
              reshard_fn=_capture_reshard,
              delete_dst_buffers=False,
          )
        else:
          tunix_gen_utils.transfer_state_directly(
              src_state=params,
              dst_state=dst_state,
              reshard_fn=_capture_reshard,
              scan_axis=1,
              delete_dst_buffers=False,
          )

        assert "synced" in captured and captured["synced"], "Weight transfer produced 0 leaves."
        synced_raw = captured["synced"]
        synced_count = len(jax.tree_util.tree_leaves(synced_raw))
        assert (
            synced_count == expected_dst_leaves
        ), f"Expected all {expected_dst_leaves} rollout leaves to sync, got {synced_count}."

        synced_tree = nnx.to_pure_dict(dst_state)["model"]
        if legacy_vllm_dst:
          synced_tree = m3_checkpoint.from_legacy_layer_names(
              synced_tree, nnx.to_pure_dict(nnx.state(self.rollout_model.base, nnx.Param))
          )
        nnx.update(self.rollout_model.base, synced_tree)
        sync_events.append(synced_count)

      def generate(
          self,
          prompts: list[str],
          rollout_config: base_rollout.RolloutConfig,
          **unused_kwargs: Any,
      ) -> base_rollout.RolloutOutput:
        """Generates deterministic group completions and per-token log-probabilities on `self.mesh`."""
        max_prompt = rollout_config.max_prompt_length
        gen_steps = rollout_config.max_tokens_to_generate
        encoded = [self._tokenizer.encode(p)[-max_prompt:] for p in prompts]
        prompt_lens = np.asarray([len(row) for row in encoded], dtype=np.int32)
        left_padded = np.full((len(prompts), max_prompt), _PAD_ID, dtype=np.int32)
        for idx, row in enumerate(encoded):
          left_padded[idx, max_prompt - len(row) :] = np.asarray(row, dtype=np.int32)

        cur_tokens = jnp.asarray(left_padded, dtype=np.int32)
        step_logprobs: list[np.ndarray] = []
        with jax.set_mesh(self.mesh):
          for step_idx in range(gen_steps):
            valid = cur_tokens != _PAD_ID
            pos = jnp.maximum(jnp.cumsum(valid.astype(jnp.int32), axis=-1) - 1, 0)
            slen = cur_tokens.shape[1]
            mask = valid[:, None, :] & jnp.tril(jnp.ones((slen, slen), dtype=jnp.bool_))[None, :, :]
            logits, _ = self.rollout_model(cur_tokens, positions=pos, cache=None, attention_mask=mask)
            next_logits = logits[:, -1, :].astype(jnp.float32)
            # Mask special pad/bos/eos ids during smoke rollout so generated tokens stay valid.
            masked_logits = next_logits.at[:, :3].set(-1e9)
            logp_all = jax.nn.log_softmax(next_logits, axis=-1)
            # Diversify generation choices deterministically across group candidates.
            row_offsets = jnp.arange(cur_tokens.shape[0], dtype=jnp.int32)
            top2 = jax.lax.top_k(masked_logits, 2)[1]
            chosen = top2[jnp.arange(cur_tokens.shape[0]), (row_offsets + step_idx) % 2]
            chosen_lp = jnp.take_along_axis(logp_all, chosen[:, None], axis=-1).squeeze(-1)
            step_logprobs.append(np.asarray(chosen_lp, dtype=np.float32))
            cur_tokens = jnp.concatenate([cur_tokens, chosen[:, None]], axis=1)

        gen_np = np.asarray(cur_tokens[:, max_prompt:], dtype=np.int32)
        stacked_lp = np.stack(step_logprobs, axis=1)
        texts = [self._tokenizer.decode(row.tolist()) for row in gen_np]
        return base_rollout.RolloutOutput(
            text=texts,
            logits=None,
            tokens=[gen_np[i] for i in range(gen_np.shape[0])],
            left_padded_prompt_tokens=left_padded,
            logprobs=[stacked_lp[i] for i in range(stacked_lp.shape[0])],
            prompt_lengths=prompt_lens,
        )

      def get_per_token_logps(
          self,
          prompt_tokens: jax.Array,
          completion_tokens: jax.Array,
      ) -> jax.Array:
        """Evaluates completion per-token log-probabilities via Tunix `compute_per_token_logps`."""
        graphdef, state = nnx.split(self.rollout_model)
        with jax.set_mesh(self.mesh):
          return tunix_common.compute_per_token_logps(
              graphdef,
              state,
              prompt_tokens=prompt_tokens,
              completion_tokens=completion_tokens,
              pad_id=_PAD_ID,
              eos_id=_EOS_ID,
              stop_gradient=True,
              chunk_size=chunk_size,
          )

      def pad_id(self) -> int:
        """Returns rollout padding token ID."""
        return _PAD_ID

      def eos_id(self) -> int:
        """Returns rollout end-of-sequence token ID."""
        return _EOS_ID

      def model(self) -> nnx.Module | None:
        """Matches `VllmRollout.model()` returning `None` when actor and rollout trees differ."""
        return None

    with tempfile.TemporaryDirectory() as ckpt_dir:
      role_to_mesh = {
          rl_cluster_lib.Role.ACTOR: actor_mesh,
          rl_cluster_lib.Role.REFERENCE: actor_mesh,
          rl_cluster_lib.Role.ROLLOUT: rollout_mesh,
      }
      training_cfg = rl_cluster_lib.RLTrainingConfig(
          actor_optimizer=optax.adamw(learning_rate=1e-2, eps=1e-4),
          eval_every_n_steps=100,
          max_steps=1,
          mini_batch_size=2,
          train_micro_batch_size=2,
          rollout_micro_batch_size=2,
          compute_logps_micro_batch_size=2,
          compute_logps_chunk_size=chunk_size,
          checkpoint_root_directory=ckpt_dir,
      )
      rollout_cfg = base_rollout.RolloutConfig(
          max_tokens_to_generate=4,
          max_prompt_length=6,
          temperature=1.0,
          return_logprobs=True,
      )
      cluster_cfg = rl_cluster_lib.ClusterConfig(
          role_to_mesh=role_to_mesh,
          rollout_engine=_CpuVllmDirectSyncRollout,
          offload_to_cpu=False,
          training_config=training_cfg,
          rollout_config=rollout_cfg,
      )

      tokenizer = _DeterministicByteTokenizer(vocab_size=64)
      rl_cluster = rl_cluster_lib.RLCluster(
          actor=actor_adapter,
          reference=ref_adapter,
          tokenizer=tokenizer,
          cluster_config=cluster_cfg,
      )

      # Capture pre-step parameters and per-token logprobs on fixed probe sequences.
      probe_prompts = jnp.array([[_PAD_ID, _BOS_ID, 5, 6, 7, 8], [_BOS_ID, 9, 10, 11, 12, 13]], dtype=jnp.int32)
      probe_completions = jnp.array([[14, 15, 16, 17], [18, 19, 20, 21]], dtype=jnp.int32)
      pre_rollout_logps = rl_cluster.rollout.get_per_token_logps(probe_prompts, probe_completions)
      ref_model = rl_cluster.inference_worker.get_model("reference")
      ref_params_before = [jnp.copy(x) for x in jax.tree_util.tree_leaves(nnx.state(ref_model.base, nnx.Param))]
      actor_params_before = [
          jnp.copy(x) for x in jax.tree_util.tree_leaves(nnx.state(rl_cluster.actor_trainer.model.base, nnx.Param))
      ]

      def _distinct_group_reward(prompts: list[str], completions: list[str], **unused_kw: Any) -> list[float]:
        del prompts
        # Assign distinct rewards per completion within each group so GRPO advantages are non-zero.
        return [float((idx % 2) * 2.0 - 0.5 + 0.01 * sum(ord(c) for c in comp)) for idx, comp in enumerate(completions)]

      algo_cfg = grpo_learner.GRPOConfig(
          num_generations=2,
          num_iterations=1,
          beta=0.04,
      )
      learner = grpo_learner.GrpoLearner(
          rl_cluster,
          algo_config=algo_cfg,
          reward_fns=[_distinct_group_reward],
      )

      learner.train([{"prompts": np.array(["alpha", "bravo"])}])
      # Explicitly invoke sync_weights as well so post-final-step rollout weights reflect step 1.
      rl_cluster.sync_weights()

      self.assertGreaterEqual(rl_cluster.global_steps, 1)
      self.assertGreaterEqual(len(sync_events), 2)
      for count in sync_events:
        self.assertEqual(count, expected_dst_leaves)

      actor_params_after = jax.tree_util.tree_leaves(nnx.state(rl_cluster.actor_trainer.model.base, nnx.Param))
      ref_params_after = jax.tree_util.tree_leaves(nnx.state(ref_model.base, nnx.Param))
      for r0, r1 in zip(ref_params_before, ref_params_after):
        self.assertTrue(bool(jnp.array_equal(r0, r1)), "Reference model weights mutated during RL step.")
      max_actor_delta = max(float(jnp.max(jnp.abs(a1 - a0))) for a0, a1 in zip(actor_params_before, actor_params_after))
      self.assertGreater(max_actor_delta, 1e-5, "Actor model weights did not update during GRPO step.")

      # Verify every parameter leaf (including all decoder layers) synced identically to rollout model.
      actor_legacy = rl_cluster.actor_trainer.model.to_rollout_pure_dict(legacy_layer_names=True)["base"]
      rollout_legacy = rl_cluster.rollout.rollout_model.to_rollout_pure_dict(legacy_layer_names=True)["base"]
      if arch == "qwen3" and scan_layers:
        for layer_idx in range(2):
          actor_layer = jax.tree_util.tree_map(
              lambda x, i=layer_idx: jnp.take(x, i, axis=1), actor_legacy["decoder"]["layers"]
          )
          rollout_layer = rollout_legacy["decoder"][f"layers_{layer_idx}"]
          for a_leaf, r_leaf in zip(jax.tree_util.tree_leaves(actor_layer), jax.tree_util.tree_leaves(rollout_layer)):
            np.testing.assert_allclose(np.asarray(r_leaf), np.asarray(a_leaf), rtol=1e-6, atol=1e-6)
      else:
        for a_leaf, r_leaf in zip(jax.tree_util.tree_leaves(actor_legacy), jax.tree_util.tree_leaves(rollout_legacy)):
          np.testing.assert_allclose(np.asarray(r_leaf), np.asarray(a_leaf), rtol=1e-6, atol=1e-6)

      # Verify synced rollout logprobs shifted and match the updated actor policy (including chunked logps).
      post_rollout_logps = rl_cluster.rollout.get_per_token_logps(probe_prompts, probe_completions)
      actor_graphdef, actor_state = nnx.split(rl_cluster.actor_trainer.model)
      with jax.set_mesh(actor_mesh):
        post_actor_logps = tunix_common.compute_per_token_logps(
            actor_graphdef,
            actor_state,
            prompt_tokens=probe_prompts,
            completion_tokens=probe_completions,
            pad_id=_PAD_ID,
            eos_id=_EOS_ID,
            stop_gradient=True,
            chunk_size=chunk_size,
        )
      np.testing.assert_allclose(post_rollout_logps, post_actor_logps, rtol=1e-5, atol=1e-5)
      logprob_shift = float(jnp.max(jnp.abs(post_rollout_logps - pre_rollout_logps)))
      self.assertGreater(logprob_shift, 1e-4)


if __name__ == "__main__":
  unittest.main()
