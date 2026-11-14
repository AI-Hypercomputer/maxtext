# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""query_proj_kernel_gather_once: one backward all-gather of the attention query kernel per layer.

The attention query (+ Qwen3.5 output gate) kernel is sharded on its embed dimension, so the partitioner all-gathers
it for every dot that reads it. Under remat the backward of a scanned layer reads it twice, in the replay of the
projection and in its input gradient, and the compiler may gather it twice. With the option on, the projection is a
custom VJP (layers/linears.py _dot_general_reusing_gathered_kernel) whose residual is the explicitly gathered kernel,
so the backward gathers it once.

  - Programs and values, in a child process with 8 CPU devices (see main): a stack of Qwen3NextFullAttention layers
    scanned under jax.checkpoint with the decoder's own remat policy, as nnx_decoders scans them, each followed by
    an RMS norm of the residual stream as in the decoder layer, on the 397B trainer's cp-as-ep rules (embed on fsdp x
    context x expert) and on the default rules. With the option: every dot that reads the query kernel (forward,
    replay, input gradient) reads the explicitly gathered kernel; the loss and every gradient are bit-identical to the
    option off; the backward loop body has exactly one all-gather of the query kernel; the output kernel's gathers are
    untouched; and no other collective changes (so the kernel gradient is still reduce-scattered, not all-reduced).
    The same program checks run for a TPU target (tpu7x-8, compiled on this host) when libtpu is installed; there the
    backward gathers the query kernel twice without the option, as in the 397B trainer step, and once with it.
  - In this process: the custom VJP gives the same bits as lax.dot_general; the attention layer passes the gathered
    axes to the query projection, for training only; DenseGeneral rejects what it cannot serve; the config validator
    rejects the option wherever it would not remove an all-gather, in MaxTextConfig and RLConfig.
"""

import collections
import functools
import importlib.util
import inspect
import os
import re
import subprocess
import sys
import types

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
from jax.ad_checkpoint import checkpoint_name
from jax.experimental import topologies
from jax.extend import core as jax_core
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
import pytest

from maxtext.common.common_types import MODEL_MODE_PREFILL, MODEL_MODE_TRAIN, ShardMode
from maxtext.configs import pyconfig
from maxtext.configs import types as config_types
from maxtext.layers import linears
from maxtext.layers import moe
from maxtext.layers import nnx_decoders
from maxtext.models import qwen3
from maxtext.utils import accelerator_to_spec_map
from maxtext.utils import maxtext_utils
from maxtext.utils import sharding
from tests.utils.test_helpers import get_test_config_path

_PASS_TOKEN = "QUERY_PROJ_KERNEL_GATHER_ONCE_CHECKS_PASSED"
_QUERY_KEY = "query_proj_kernel_gather_once"
_NUM_DEVICES = 8
_LAYERS = 3
# Distinct sizes, so each kernel's gathered shape is recognisable among the program's collectives: query kernel
# [256, 4, 2 * 32] (query + gate), output kernel [4 * 32, 256], activations [4, 64, 256] (batch 8 x 0.5 per device).
_EMB, _HEADS, _KV_HEADS, _HEAD_DIM, _SEQ = 256, 4, 2, 32, 64


def _bits(x):
  """The array's bytes, so equality means bit-identical (NaN-safe, -0.0 != 0.0)."""
  return np.asarray(jax.device_get(x)).tobytes()


def _cfg(**overrides):
  kwargs = {
      "run_name": "query_proj_kernel_gather_once",
      "enable_checkpointing": False,
      "log_config": False,
      "skip_jax_distributed_system": True,
  }
  kwargs.update(overrides)
  return pyconfig.initialize([sys.argv[0], get_test_config_path()], **kwargs)


# A small Qwen3.5 attention layer in the trainer's setting: bf16 compute on fp32 weights, scanned layers,
# remat_policy=custom with the projections replayed and the layer input saved.
_SMALL_QWEN35 = {
    "decoder_block": "qwen3_5",
    "base_emb_dim": _EMB,
    "base_num_query_heads": _HEADS,
    "base_num_kv_heads": _KV_HEADS,
    "head_dim": _HEAD_DIM,
    "base_mlp_dim": 256,
    "base_moe_mlp_dim": 256,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "base_num_decoder_layers": 4,
    "max_target_length": _SEQ,
    "per_device_batch_size": 0.5,
    "attention": "dot_product",
    "dtype": "bfloat16",
    "weight_dtype": "float32",
    "scan_layers": True,
    "remat_policy": "custom",
    "query_proj": "remat",
    "out_proj": "remat",
    "decoder_layer_input": "device",
    "context_parallel_load_balance": False,
}

# The 397B trainer's rules (embed_attn -> fsdp x context x expert), and the default rules (embed_attn -> fsdp x
# fsdp_transpose x context x ..., here fsdp 4 x context 2).
_MESHES = {
    "cp_as_ep": {
        "custom_mesh_and_rule": "cp-as-ep",
        "ici_fsdp_parallelism": 2,
        "ici_context_parallelism": 2,
        "ici_expert_parallelism": 2,
    },
    "default": {"ici_fsdp_parallelism": 4, "ici_context_parallelism": 2},
}


class CustomVjpTest(absltest.TestCase):
  """_dot_general_reusing_gathered_kernel is lax.dot_general, value and VJP, bit for bit."""

  def test_matches_dot_general(self):
    rng = np.random.default_rng(0)
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("fsdp",))
    gathered = NamedSharding(mesh, P())
    precision = jax.lax.Precision.DEFAULT
    cases = (
        (
            (2, 8, 16),
            (16, 4, 6),
            (((2,), (0,)), ((), ())),
        ),  # query: [b, s, e] x [e, h, d]
        (
            (2, 8, 4, 6),
            (4, 6, 16),
            (((2, 3), (0, 1)), ((), ())),
        ),  # output: [b, s, h, d] x [h, d, e]
    )
    for x_shape, w_shape, dims in cases:
      x = jnp.asarray(rng.standard_normal(x_shape), jnp.bfloat16)
      w = jnp.asarray(rng.standard_normal(w_shape), jnp.bfloat16)

      def ref(x, w, dims=dims):
        return jax.lax.dot_general(x, w, dims, precision=precision)

      def new(x, w, dims=dims):
        return linears._dot_general_reusing_gathered_kernel(x, w, dims, precision, gathered)  # pylint: disable=protected-access

      ref_out, ref_vjp = jax.vjp(ref, x, w)
      new_out, new_vjp = jax.vjp(new, x, w)
      g = jnp.asarray(rng.standard_normal(ref_out.shape), jnp.bfloat16)
      self.assertEqual(_bits(ref_out), _bits(new_out))
      for a, b in zip(ref_vjp(g), new_vjp(g)):
        self.assertEqual(a.shape, b.shape)
        self.assertEqual(_bits(a), _bits(b))
      self.assertEqual(_bits(jax.jit(ref)(x, w)), _bits(jax.jit(new)(x, w)))


class WiringTest(parameterized.TestCase):
  """The attention layer hands the query DenseGeneral the gathered axes exactly when the option is on, in training."""

  def _projections(self, model_mode, **overrides):
    """Builds a Qwen3NextFullAttention layer and returns its query and output projections plus mesh."""
    cfg = _cfg(
        **{
            **_SMALL_QWEN35,
            "per_device_batch_size": 1.0,
            "max_prefill_predict_length": _SEQ // 2,
            **overrides,
        }
    )
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    with nn_partitioning.axis_rules(cfg.logical_axis_rules):
      layer = qwen3.Qwen3NextFullAttention(cfg, mesh, model_mode, layer_idx=3, rngs=nnx.Rngs(0))
    return {"query": layer.attention.query, "out": layer.attention.out}, mesh

  def test_train_on(self):
    projections, mesh = self._projections(MODEL_MODE_TRAIN, **{_QUERY_KEY: True})
    # Exactly the embed dimension is gathered: the query kernel's axes are (embed_attn, q_heads, kv).
    self.assertEqual(projections["query"].gathered_kernel_axes, (None, "q_heads", "kv"))
    self.assertIs(projections["query"].mesh, mesh)
    self.assertIsNone(projections["out"].gathered_kernel_axes)
    self.assertIsNone(projections["out"].mesh)

  @parameterized.named_parameters(("train_off", MODEL_MODE_TRAIN, False), ("prefill_on", MODEL_MODE_PREFILL, True))
  def test_off(self, model_mode, on):
    projections, _ = self._projections(model_mode, **{_QUERY_KEY: on})
    for projection in projections.values():
      self.assertIsNone(projection.gathered_kernel_axes)
      self.assertIsNone(projection.mesh)

  def test_call_time_prefill_skips_gathered_kernel_custom_vjp(self):
    """A train-constructed Attention module skips _dot_general_reusing_gathered_kernel on non-train calls."""
    cfg = _cfg(
        **{
            **_SMALL_QWEN35,
            "ici_fsdp_parallelism": len(jax.devices()),
            "per_device_batch_size": 1.0,
            "max_prefill_predict_length": _SEQ // 2,
            _QUERY_KEY: True,
        }
    )
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    with nn_partitioning.axis_rules(cfg.logical_axis_rules):
      layer = qwen3.Qwen3NextFullAttention(cfg, mesh, MODEL_MODE_TRAIN, layer_idx=3, rngs=nnx.Rngs(0))
    self.assertEqual(layer.attention.query.gathered_kernel_axes, (None, "q_heads", "kv"))
    x = jnp.ones((len(jax.devices()), 8, _EMB), dtype=jnp.bfloat16)
    graphdef, state = nnx.split(layer.attention)
    kernel_shape = (_EMB, _HEADS, 2 * _HEAD_DIM)
    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
      train_jaxpr = jax.make_jaxpr(
          lambda s, inp: nnx.merge(graphdef, s).query_projection(inp, model_mode=MODEL_MODE_TRAIN)
      )(state, x).jaxpr
      prefill_jaxpr = jax.make_jaxpr(
          lambda s, inp: nnx.merge(graphdef, s).query_projection(inp, model_mode=MODEL_MODE_PREFILL)
      )(state, x).jaxpr
    train_reads = _kernel_reads(train_jaxpr, kernel_shape)
    prefill_reads = _kernel_reads(prefill_jaxpr, kernel_shape)
    self.assertTrue(all(isinstance(r[1], tuple) and r[1][0] == "gathered" for r in train_reads), msg=train_reads)
    self.assertFalse(any(isinstance(r[1], tuple) for r in prefill_reads), msg=prefill_reads)

  def test_dense_general_guard(self):
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("fsdp",))
    x = jnp.ones((2, 16), jnp.float32)
    common = {
        "kernel_axes": ("embed", "mlp"),
        "gathered_kernel_axes": (None, "mlp"),
        "rngs": nnx.Rngs(0),
    }
    for kwargs in (
        {"mesh": None},
        {"mesh": mesh, "shard_mode": ShardMode.EXPLICIT},
    ):
      dense = linears.DenseGeneral(16, 8, **common, **kwargs)
      with self.assertRaisesRegex(ValueError, "gathered_kernel_axes supports only"):
        dense(x)
    dense = linears.DenseGeneral(16, 8, mesh=mesh, **common)
    with self.assertRaisesRegex(ValueError, "gathered_kernel_axes supports only"):
      dense(x, slice_bounds=(0, 4))
    self.assertEqual(dense(x).shape, (2, 8))
    # Calling with use_gathered_kernel=False bypasses the training-only custom VJP path cleanly.
    dense_no_mesh = linears.DenseGeneral(16, 8, mesh=None, **common)
    self.assertEqual(dense_no_mesh(x, use_gathered_kernel=False).shape, (2, 8))

  def test_mesh_none_and_multimodal_encoders_skip_gathered_kernel_axes(self):
    """Mesh-less Attention and vision/audio encoder Attention layers skip gathered_kernel_axes."""
    # pylint: disable=import-outside-toplevel
    from maxtext.layers import attentions as attn_mod

    cfg = _cfg(
        **{
            **_SMALL_QWEN35,
            "model_name": "qwen3.5-397b-a17b",
            "override_model_config": True,
            "ici_fsdp_parallelism": max(len(jax.devices()), 1) if len(jax.devices()) > 1 else -1,
            "per_device_batch_size": 1.0,
            "hidden_size_for_vit": 64,
            "num_attention_heads_for_vit": 4,
            "intermediate_size_for_vit": 128,
            "num_position_embeddings_for_vit": 16,
            "spatial_merge_size_for_vit": 2,
            "attention_for_vit": "dot_product",
            _QUERY_KEY: True,
        }
    )
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    with nn_partitioning.axis_rules(cfg.logical_axis_rules):
      common_attn = {
          "config": cfg,
          "num_query_heads": 4,
          "num_kv_heads": 4,
          "head_dim": 16,
          "max_target_length": 16,
          "attention_kernel": "dot_product",
          "inputs_q_shape": (1, 4, 64),
          "inputs_kv_shape": (1, 4, 64),
          "dtype": jnp.bfloat16,
          "weight_dtype": jnp.float32,
          "model_mode": MODEL_MODE_TRAIN,
          "rngs": nnx.Rngs(0),
      }
      # 1. Mesh-less Attention (e.g. share_kv_layer=True with mesh=None) skips gathered_kernel_axes cleanly.
      no_mesh_attn = attn_mod.Attention(mesh=None, share_kv_layer=True, **common_attn)
      self.assertIsNone(no_mesh_attn.query.gathered_kernel_axes)
      q_out = no_mesh_attn.query_projection(jnp.ones((1, 4, 64), dtype=jnp.bfloat16), model_mode=MODEL_MODE_TRAIN)
      self.assertEqual(q_out.shape, (1, 4, 4, 32))

      # 2. Vision, audio, and ViT-full encoder Attention layers keep gathered_kernel_axes=None even with a train mesh.
      vis_with_mesh = qwen3.Qwen3OmniMoeVisionAttention(config=cfg, mesh=mesh, rngs=nnx.Rngs(0))
      self.assertIsNone(vis_with_mesh.attn.query.gathered_kernel_axes)
      batch = len(jax.devices())
      with jax.set_mesh(mesh):
        out = vis_with_mesh(jnp.ones((batch, 4, 64), dtype=jnp.bfloat16), num_frames=1, height=2, width=2)
      self.assertEqual(out.shape, (batch, 4, 64))
      audio_attn = attn_mod.Attention(mesh=mesh, name="self_attention_audio", attention_type="full", **common_attn)
      vit_full_attn = attn_mod.Attention(mesh=mesh, attention_type="full", **common_attn)
      self.assertIsNone(audio_attn.query.gathered_kernel_axes)
      self.assertIsNone(vit_full_attn.query.gathered_kernel_axes)


def _validator_cfg(**overrides):
  """The validator's view of the 397B trainer cell (attribute names as in MaxTextConfig and RLConfig)."""
  rules = config_types.MaxTextConfig._load_mesh_config_from_yaml("cp-as-ep")["logical_axis_rules"]  # pylint: disable=protected-access
  fields = {
      _QUERY_KEY: False,
      "gather_scanned_kernels_once": False,
      "scan_layers": True,
      "prefetch_fsdp_weights": False,
      "parameter_memory_host_offload": False,
      "lora": config_types.LoRA(),
      "attention_type": "global",
      "fused_qkv": False,
      "use_fused_qkv_kernel": False,
      "quantization": config_types.QuantizationType.NONE,
      "shard_mode": ShardMode.AUTO,
      "ici_context_autoregressive_parallelism": 1,
      "logical_axis_rules": rules,
      "ici_fsdp_parallelism": 32,
      "ici_context_parallelism": 4,
      "ici_expert_parallelism": 2,
      "dcn_fsdp_parallelism": 1,
      "dcn_context_parallelism": 1,
      "dcn_expert_parallelism": 1,
      "remat_policy": "custom",
      "query_proj": "remat",
      "out_proj": "remat",
  }
  fields.update(overrides)
  return types.SimpleNamespace(**fields)


_UNSHARDED = {
    "ici_fsdp_parallelism": 1,
    "ici_context_parallelism": 1,
    "ici_expert_parallelism": 1,
}


def _tensor_parallel(*logical):
  """The trainer cell with `logical` axes mapped to 'tensor' and tensor parallelism 2."""
  rules = [r for r in _validator_cfg().logical_axis_rules if r[0] not in logical] + [[n, ["tensor"]] for n in logical]
  return {"logical_axis_rules": rules, "ici_tensor_parallelism": 2}


class ValidatorTest(parameterized.TestCase):
  """The validator rejects the option wherever it would not remove an all-gather from the backward."""

  def _validate(self, **overrides):
    config_types.MaxTextConfig.validate_query_proj_kernel_gather_once(_validator_cfg(**overrides))

  @parameterized.named_parameters(
      ("trainer_cell", {}),
      ("full", {"remat_policy": "full"}),
      ("save_out_proj", {"remat_policy": "save_out_proj"}),
      ("out_saved", {"out_proj": "device"}),
      ("fsdp_auto", {**_UNSHARDED, "ici_fsdp_parallelism": -1}),
      ("dcn_fsdp", {**_UNSHARDED, "dcn_fsdp_parallelism": 2}),
      ("tensor_on_mlp_only", _tensor_parallel("mlp")),
  )
  def test_accepted(self, overrides):
    self._validate(**{_QUERY_KEY: True, **overrides})

  @parameterized.named_parameters(
      ("mla", {"attention_type": "mla"}, "builds its own query projections"),
      (
          "compressed",
          {"attention_type": "compressed"},
          "builds its own query projections",
      ),
      ("fused_qkv", {"fused_qkv": True}, "requires fused_qkv=False"),
      ("use_fused_qkv_kernel", {"use_fused_qkv_kernel": True}, "requires use_fused_qkv_kernel=False"),
      (
          "int8",
          {"quantization": config_types.QuantizationType.INT8},
          "requires quantization=''",
      ),
      (
          "fp8",
          {"quantization": config_types.QuantizationType.FP8},
          "requires quantization=''",
      ),
      (
          "lora",
          {"lora": config_types.LoRA(enable_lora=True, lora_rank=8)},
          "is not supported with LoRA",
      ),
      ("explicit", {"shard_mode": ShardMode.EXPLICIT}, "requires shard_mode='auto'"),
      (
          "autoregressive",
          {"ici_context_autoregressive_parallelism": 2},
          "the query kernel is replicated",
      ),
      ("unsharded", _UNSHARDED, "none of which has a parallelism other than 1"),
      (
          "tensor_on_q_heads",
          _tensor_parallel("q_heads"),
          "requires the kernel's other dimensions to be replicated",
      ),
      (
          "tensor_on_kv",
          _tensor_parallel("kv"),
          "requires the kernel's other dimensions to be replicated",
      ),
      (
          "minimal",
          {"remat_policy": "minimal"},
          "requires a remat policy that replays the query projection",
      ),
      (
          "save_qkv_proj",
          {"remat_policy": "save_qkv_proj"},
          "requires a remat policy that replays",
      ),
      (
          "qkv_proj_offloaded",
          {"remat_policy": "qkv_proj_offloaded"},
          "requires a remat policy that replays",
      ),
      (
          "save_dot_except_mlp",
          {"remat_policy": "save_dot_except_mlp"},
          "requires a remat policy that replays",
      ),
      ("none", {"remat_policy": "none"}, "requires a remat policy that replays"),
      (
          "custom_saved",
          {"query_proj": "device"},
          "requires a remat policy that replays",
      ),
      (
          "custom_offloaded",
          {"query_proj": "offload"},
          "requires a remat policy that replays",
      ),
  )
  def test_rejected(self, overrides, message):
    with self.assertRaisesRegex(ValueError, f"{_QUERY_KEY}=True .*{message}"):
      self._validate(**{_QUERY_KEY: True, **overrides})
    # The same config with the option off passes.
    self._validate(**overrides)

  def test_pyconfig(self):
    self.assertFalse(_cfg().query_proj_kernel_gather_once)
    self.assertFalse(config_types.Attention().query_proj_kernel_gather_once)
    small = {**_SMALL_QWEN35, **_MESHES["cp_as_ep"]}
    self.assertTrue(_cfg(**small, **{_QUERY_KEY: True}).query_proj_kernel_gather_once)
    with self.assertRaisesRegex(ValueError, "requires a remat policy that replays the query projection"):
      _cfg(**{**small, _QUERY_KEY: True, "query_proj": "device"})
    with self.assertRaisesRegex(ValueError, "requires fused_qkv=False"):
      _cfg(**{**small, _QUERY_KEY: True, "fused_qkv": True})
    with self.assertRaisesRegex(ValueError, "is not supported with LoRA"):
      _cfg(**{**small, _QUERY_KEY: True, "lora": {"enable_lora": True, "lora_rank": 8}})
    tensor_parallel = {
        **_SMALL_QWEN35,
        "ici_fsdp_parallelism": 2,
        "ici_context_parallelism": 2,
        "ici_tensor_parallelism": 2,
    }
    with self.assertRaisesRegex(ValueError, "requires the kernel's other dimensions to be replicated"):
      _cfg(**{**tensor_parallel, _QUERY_KEY: True})

  def test_rl_config_runs_the_validator(self):
    # RLConfig does not inherit MaxTextConfig's validators; its own validator must call the shared one.
    source = inspect.getsource(config_types.RLConfig.set_derived_values_and_validate)
    self.assertIn("MaxTextConfig.validate_query_proj_kernel_gather_once(self)", source)
    self.assertIn("MaxTextConfig.validate_gather_scanned_kernels_once(self)", source)
    source = inspect.getsource(config_types.MaxTextConfig.set_derived_and_validate_values)
    self.assertIn("self.validate_query_proj_kernel_gather_once(self)", source)
    self.assertIn("self.validate_gather_scanned_kernels_once(self)", source)

    # Also instantiate RLConfig directly to verify both validators fire without widening RLConfig's schema with LoRA.
    self.assertNotIn("lora", config_types.RLConfig.model_fields)
    rl_base = {
        "model_name": "qwen3-8b",
        "tokenizer_path": "src/maxtext/assets/tokenizers/tokenizer.llama2",
        "ici_fsdp_parallelism": 4,
        "remat_policy": "full",
    }
    self.assertTrue(config_types.RLConfig(**rl_base, **{_QUERY_KEY: True}).query_proj_kernel_gather_once)
    self.assertTrue(config_types.RLConfig(**rl_base, gather_scanned_kernels_once=True).gather_scanned_kernels_once)
    with self.assertRaisesRegex(ValueError, f"{_QUERY_KEY}=True requires fused_qkv=False"):
      config_types.RLConfig(**rl_base, **{_QUERY_KEY: True}, fused_qkv=True)
    with self.assertRaisesRegex(ValueError, "gather_scanned_kernels_once=True requires scan_layers=True"):
      config_types.RLConfig(**rl_base, gather_scanned_kernels_once=True, scan_layers=False)
    with self.assertRaisesRegex(ValueError, "incompatible with prefetch_fsdp_weights=True"):
      config_types.RLConfig(**rl_base, gather_scanned_kernels_once=True, prefetch_fsdp_weights=True)
    with self.assertRaisesRegex(ValueError, "incompatible with parameter_memory_host_offload=True"):
      config_types.RLConfig(**rl_base, gather_scanned_kernels_once=True, parameter_memory_host_offload=True)

  @parameterized.named_parameters(
      ("explicit", {"shard_mode": "explicit"}, "requires shard_mode='auto'"),
      ("unscanned", {"scan_layers": False}, "requires scan_layers=True"),
      ("prefetch", {"prefetch_fsdp_weights": True, "remat_policy": "full"}, "incompatible with prefetch_fsdp_weights"),
      (
          "host_offload",
          {"parameter_memory_host_offload": True},
          "incompatible with parameter_memory_host_offload",
      ),
      ("lora", {"lora": {"enable_lora": True, "lora_rank": 8}}, "is not supported with LoRA"),
  )
  def test_gather_scanned_kernels_once_validator(self, overrides, message):
    self.assertTrue(_cfg(gather_scanned_kernels_once=True).gather_scanned_kernels_once)
    with self.assertRaisesRegex(ValueError, f"gather_scanned_kernels_once=True .*{message}"):
      _cfg(gather_scanned_kernels_once=True, **overrides)
    self.assertFalse(_cfg(gather_scanned_kernels_once=False, **overrides).gather_scanned_kernels_once)

  def test_qwix_lora_raises_under_custom_vjp_gather_once(self):
    """Confirms Qwix LoRA fails under _dot_general_reusing_gathered_kernel's custom VJP (closed-over LoRAParam)."""
    if importlib.util.find_spec("qwix") is None:
      self.skipTest("requires qwix")
    import qwix  # pylint: disable=import-outside-toplevel

    mesh = Mesh(np.asarray(jax.devices()[:1]), ("fsdp",))
    x = jnp.ones((2, 16), dtype=jnp.float32)
    provider = qwix.LoraProvider(module_path=".*", rank=4, alpha=8.0)

    class _Box(nnx.Module):
      """Minimal module wrapping DenseGeneral for Qwix LoRA testing."""

      def __init__(self, gather_once):
        self.proj = linears.DenseGeneral(
            16,
            8,
            kernel_axes=("embed_attn", "mlp"),
            gathered_kernel_axes=(None, "mlp") if gather_once else None,
            mesh=mesh if gather_once else None,
            rngs=nnx.Rngs(0),
        )

      def __call__(self, inp):
        return self.proj(inp)

    with jax.set_mesh(mesh):
      ok_model = qwix.apply_lora_to_model(_Box(False), provider, x, rngs=nnx.Rngs(1))
      loss, _ = nnx.value_and_grad(lambda m, inp: jnp.sum(jnp.square(m(inp))), argnums=nnx.DiffState(0, nnx.LoRAParam))(
          ok_model, x
      )
      self.assertEqual(loss.shape, ())

      bad_model = qwix.apply_lora_to_model(_Box(True), provider, x, rngs=nnx.Rngs(1))
      with self.assertRaises(jax.errors.UnexpectedTracerError):
        nnx.value_and_grad(lambda m, inp: jnp.sum(jnp.square(m(inp))), argnums=nnx.DiffState(0, nnx.LoRAParam))(
            bad_model, x
        )


class GatherScannedKernelsOnceTest(absltest.TestCase):
  """Tests for moe.gather_scanned_kernels_once and moe.gathered_kernel_grad_shardings."""

  def test_nested_local_layers_metadata_preserved(self):
    # pylint: disable=protected-access
    class _NestedKernelModule(nnx.Module):

      def __init__(self, metadata_axis_name):
        self.kernel_axes = ("gdn_head", "embed_attn")
        self.kernel = nnx.Param(
            jnp.ones((2, 3, 2, 16), dtype=jnp.float32),
            param_scan_axis=1,
            out_sharding=("gdn_head", metadata_axis_name, "local_layers", "embed_attn"),
        )

    class _Block(nnx.Module):

      def __init__(self, metadata_axis_name):
        self.proj = _NestedKernelModule(metadata_axis_name)

    mesh = jax.sharding.AbstractMesh((4, 2, 2), ("fsdp", "context", "tensor"))
    rules = (("embed_attn", ("fsdp", "context")), ("gdn_head", "tensor"), ("local_layers", None))
    for axis_name in ("layers", "scanned_blocks"):
      block = _Block(axis_name)
      shardings = moe.gathered_kernel_grad_shardings(block, mesh, rules, metadata_axis_name=axis_name)
      self.assertEqual(shardings[("proj", "kernel")].spec, P("tensor", None, ("fsdp", "context")))
      both = moe._gathered_kernel_shardings(block, mesh, rules, metadata_axis_name=axis_name)
      self.assertEqual(both[("proj", "kernel")][0].spec, P(None, "tensor", None, None))
      self.assertEqual(both[("proj", "kernel")][1].spec, P("tensor", None, ("fsdp", "context")))

  def test_partition_name_metadata_overrides_default_axis_name(self):
    """Scanned stacks recording nnx.PARTITION_NAME != 'layers' (e.g. layers_outside_pipeline, moe_layers) resolve cleanly."""
    # pylint: disable=protected-access
    mesh = jax.sharding.AbstractMesh((4, 2), ("fsdp", "tensor"))
    rules = (("embed_gdn", "fsdp"), ("gdn_head", "tensor"), ("local_layers", None))

    class _PipelineNestedProj(nnx.Module):

      def __init__(self):
        self.kernel_axes = ("gdn_head", "embed_gdn")
        self.kernel = nnx.Param(
            jnp.ones((2, 3, 3, 16), dtype=jnp.float32),
            param_scan_axis=1,
            out_sharding=("gdn_head", "layers_outside_pipeline", "local_layers", "embed_gdn"),
            **{nnx.PARTITION_NAME: "layers_outside_pipeline"},
        )

    class _MoeLayersProj(nnx.Module):

      def __init__(self):
        self.kernel_axes = ("embed_gdn", "gdn_head")
        self.kernel = nnx.Param(
            jnp.ones((16, 4, 2), dtype=jnp.float32),
            param_scan_axis=1,
            out_sharding=("embed_gdn", "moe_layers", "gdn_head"),
            **{nnx.PARTITION_NAME: "moe_layers"},
        )

    class _Holder(nnx.Module):

      def __init__(self):
        self.nested = _PipelineNestedProj()
        self.moe_proj = _MoeLayersProj()

    holder = _Holder()
    # Called with default metadata_axis_name="layers" (matching NNXDecoder call sites that omit metadata_axis_name).
    both = moe._gathered_kernel_shardings(holder, mesh, rules)
    self.assertEqual(both[("nested", "kernel")][0].spec, P(None, "tensor", None, None))
    self.assertEqual(both[("nested", "kernel")][1].spec, P("tensor", None, "fsdp"))
    self.assertEqual(both[("moe_proj", "kernel")][0].spec, P(None, None, "tensor"))
    self.assertEqual(both[("moe_proj", "kernel")][1].spec, P("fsdp", "tensor"))

    # Verify _is_small_scanned_kernel divides by 3 * 3 = 9 layers (not just 3 local_layers) under layers_outside_pipeline:
    # shape (2, 3, 3, 4096 * 1024) float32 = 288 MiB stack, 32 MiB if divided by 9 (> 16 MiB -> False),
    # whereas shape (2, 3, 3, 1024 * 1024) float32 = 72 MiB stack, 8 MiB/layer if divided by 9 (<= 16 MiB -> True;
    # if it missed layers_outside_pipeline and divided only by 3, it would be 24 MiB > 16 MiB -> False!).
    class _NineLayer8MiB(nnx.Module):

      def __init__(self):
        self.kernel_axes = ("gdn_head", "embed_gdn")
        self.kernel = nnx.Param(
            jax.ShapeDtypeStruct((2, 3, 3, 1024 * 1024), jnp.float32),
            param_scan_axis=1,
            out_sharding=("gdn_head", "layers_outside_pipeline", "local_layers", "embed_gdn"),
            **{nnx.PARTITION_NAME: "layers_outside_pipeline"},
        )

    self.assertTrue(moe._is_small_scanned_kernel(_NineLayer8MiB()))

  def test_size_caps_apply_uniformly_to_query_and_gate_logit(self):
    """Oversized Attention.query and GateLogit kernels exceed _GATHER_ONCE_MAX_LAYER_BYTES and are not hoisted."""
    # pylint: disable=import-outside-toplevel,protected-access
    from maxtext.utils import model_creation_utils

    cfg = _cfg(
        model_name="qwen3.5-397b-a17b",
        override_model_config=True,
        ici_fsdp_parallelism=len(jax.devices()),
        per_device_batch_size=1.0,
        max_target_length=64,
    )
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    with nn_partitioning.axis_rules(cfg.logical_axis_rules):
      _, abstract_model = model_creation_utils.create_nnx_abstract_model(
          cfg, mesh, model_mode=MODEL_MODE_TRAIN, rng_key=jax.random.PRNGKey(0)
      )
    gathered = moe._gathered_kernels(abstract_model.decoder.layers)
    # Qwen3.5-397B Attention.query is 256 MiB/layer (> 16 MiB cap) and must NOT be hoisted.
    self.assertNotIn(("global_layer", "attention", "attention", "query", "kernel"), gathered)
    # Local-layer shared_expert projections are 720 MiB per stack (> 512 MiB stack cap) and must NOT be hoisted.
    self.assertNotIn(("local_layers", "mlp", "shared_expert", "wi_0", "kernel"), gathered)
    # Both global and local MoE router GateLogit kernels (8 MiB/layer) and GQA K/V kernels (8 MiB/layer) ARE hoisted.
    self.assertIn(("global_layer", "mlp", "routed_experts", "gate", "kernel"), gathered)
    self.assertIn(("local_layers", "mlp", "routed_experts", "gate", "kernel"), gathered)
    self.assertIn(("global_layer", "attention", "attention", "key", "kernel"), gathered)
    self.assertIn(("global_layer", "attention", "attention", "value", "kernel"), gathered)

    # An oversized GateLogit (> 16 MiB per layer, e.g. [4096, 2, 2048] f32 = 32 MiB/layer) is also excluded.
    class _HugeGateBlock(nnx.Module):
      """Synthetic block with a 32 MiB/layer GateLogit kernel."""

      def __init__(self):
        self.gate = moe.GateLogit(16, 4, "qwen3_moe", mesh, nnx.Rngs(0), kernel_axes=("embed", None))
        self.gate.kernel = nnx.Param(
            jax.ShapeDtypeStruct((4096, 2, 2048), jnp.float32),
            param_scan_axis=1,
            out_sharding=("embed", "layers", None),
        )

    self.assertEqual(moe._gathered_kernels(_HugeGateBlock()), {})

  def test_scanned_decoder_hoists_all_gathers_and_preserves_bit_identical_sharded_grads(self):
    """End-to-end NNXDecoder scan on FSDP CPU devices hoists FSDP gathers, emits 0 bwd stack gathers, and matches bits."""
    # pylint: disable=import-outside-toplevel,protected-access
    from unittest import mock
    from maxtext.utils import maxtext_utils_nnx, model_creation_utils

    devices = jax.devices("cpu")[:4]
    if len(devices) < 2:
      self.skipTest("requires at least 2 CPU devices")
    fsdp = len(devices)
    # Distinct shapes with all scanned kernels well within _GATHER_ONCE_MAX_LAYER_BYTES (16 MiB):
    # - moe_block.gate.kernel:       [64, 4] f32     = 1 KiB  (stacked param [64, 2, 4])
    # - self_attention.key/value:    [64, 2, 12] f32 = 6 KiB  (stacked param [64, 2, 2, 12])
    # - self_attention.query.kernel: [64, 4, 12] f32 = 12 KiB (stacked param [64, 2, 4, 12])
    # - self_attention.out.kernel:   [4, 12, 64] f32 = 12 KiB (stacked param [4, 2, 12, 64])
    num_layers, emb_dim, q_heads, kv_heads, head_dim, mlp_dim, num_experts, seq_len = 2, 64, 4, 2, 12, 64, 4, 16

    def build_and_run(gather_once, init_param_dict=None, byte_cap=None, query_gather_once=False):
      cfg = _cfg(
          decoder_block="qwen3_moe",
          base_emb_dim=emb_dim,
          base_num_query_heads=q_heads,
          base_num_kv_heads=kv_heads,
          head_dim=head_dim,
          base_mlp_dim=mlp_dim,
          base_moe_mlp_dim=mlp_dim,
          num_experts=num_experts,
          num_experts_per_tok=2,
          base_num_decoder_layers=num_layers,
          vocab_size=128,
          max_target_length=seq_len,
          per_device_batch_size=1.0,
          ici_fsdp_parallelism=fsdp,
          attention="dot_product",
          dtype="float32",
          weight_dtype="float32",
          scan_layers=True,
          remat_policy="full",
          gather_scanned_kernels_once=gather_once,
          query_proj_kernel_gather_once=query_gather_once,
      )
      mesh = Mesh(maxtext_utils.create_device_mesh(cfg, devices), cfg.mesh_axes)
      cap = moe._GATHER_ONCE_MAX_LAYER_BYTES if byte_cap is None else byte_cap
      with mock.patch.object(moe, "_GATHER_ONCE_MAX_LAYER_BYTES", cap):
        with nn_partitioning.axis_rules(cfg.logical_axis_rules):
          create_model, abstract_model = model_creation_utils.create_nnx_abstract_model(
              cfg, mesh, model_mode=MODEL_MODE_TRAIN, rng_key=jax.random.PRNGKey(0)
          )
          model = maxtext_utils_nnx.create_nnx_sharded_model(abstract_model, create_model, mesh=mesh)
        decoder = model.decoder
        gathered_paths = set(moe._gathered_kernels(decoder.layers))
        graphdef, params, rest = nnx.split(decoder.layers, nnx.Param, ...)
        param_shardings = jax.tree.map(lambda x: x.sharding, params)
        if init_param_dict is not None:
          placed = jax.tree_util.tree_map_with_path(
              lambda p, leaf: jax.device_put(init_param_dict[jax.tree_util.keystr(p)], leaf.sharding),
              nnx.to_pure_dict(params),
          )
          nnx.replace_by_pure_dict(params, placed)

        batch = int(cfg.global_batch_size_to_train_on)
        x_sharding = sharding.create_sharding(
            mesh, ("activation_batch", "activation_length", "activation_embed"), cfg.logical_axis_rules
        )
        x = jax.device_put(
            jnp.asarray(np.random.default_rng(3).standard_normal((batch, seq_len, emb_dim), dtype=np.float32)),
            x_sharding,
        )
        positions = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32), (batch, seq_len))
        segment_ids = jnp.ones((batch, seq_len), dtype=jnp.int32)

        def loss_fn(layer_params, hidden):
          scanned_layers = nnx.merge(graphdef, layer_params, rest)
          y = decoder._apply_layers_sequentially(
              scanned_layers,
              hidden,
              segment_ids,
              positions,
              True,
              MODEL_MODE_TRAIN,
              length=num_layers,
          )[0]
          return jnp.mean(jnp.square(y))

        step = jax.jit(
            jax.value_and_grad(loss_fn, argnums=(0, 1)),
            in_shardings=(param_shardings, x_sharding),
            out_shardings=(NamedSharding(mesh, P()), (param_shardings, x_sharding)),
        )
        with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
          compiled = step.lower(params, x).compile()
          loss, (grads, x_grad) = compiled(params, x)
      raw_params = {
          jax.tree_util.keystr(p): np.asarray(jax.device_get(v))
          for p, v in jax.tree_util.tree_leaves_with_path(nnx.to_pure_dict(params))
      }
      return compiled.as_text(), loss, grads, x_grad, param_shardings, gathered_paths, raw_params

    def split_collectives(hlo):
      comps = _computations(hlo)
      bodies = _while_bodies(comps)
      body_set = {c for reach in bodies.values() for c in reach}
      body_lines = [line for c in body_set for line in comps[c]]
      outer_lines = [line for c, lines in comps.items() if c not in body_set for line in lines]
      return _collectives(body_lines), _collectives(outer_lines), outer_lines

    def count_gathers(counter, shape_str):
      return sum(n for (op, dims, _), n in counter.items() if _is_gather_of(op, dims, shape_str))

    # A. Unmocked default _GATHER_ONCE_MAX_LAYER_BYTES (16 MiB): all 5 small scanned kernels are hoisted.
    hlo_off, loss_off, grads_off, xg_off, shardings_off, gathered_paths, init_params = build_and_run(False)
    hlo_on, loss_on, grads_on, xg_on, shardings_on, _, _ = build_and_run(True, init_params)
    _, loss_both, grads_both, xg_both, _, _, _ = build_and_run(True, init_params, query_gather_once=True)

    expected_all = {
        ("moe_block", "gate", "kernel"),
        ("self_attention", "query", "kernel"),
        ("self_attention", "key", "kernel"),
        ("self_attention", "value", "kernel"),
        ("self_attention", "out", "kernel"),
    }
    self.assertEqual(gathered_paths, expected_all)

    # Bit-identical loss, input gradient, and FSDP-sharded parameter gradients (assert_array_equal + _bits).
    self.assertEqual(shardings_on, shardings_off)
    for path, grad_leaf in jax.tree_util.tree_leaves_with_path(grads_on):
      key_str = jax.tree_util.keystr(path)
      if any(f"['{k}']['kernel']" in key_str for k in ("query", "key", "value", "out", "gate")):
        self.assertIn("fsdp", str(grad_leaf.sharding.spec), msg=f"unsharded grad at {key_str}")
    for label, l_cand, xg_cand, g_cand in (
        ("gather_once", loss_on, xg_on, grads_on),
        ("both_flags", loss_both, xg_both, grads_both),
    ):
      np.testing.assert_array_equal(np.asarray(jax.device_get(l_cand)), np.asarray(jax.device_get(loss_off)), label)
      np.testing.assert_array_equal(np.asarray(jax.device_get(xg_cand)), np.asarray(jax.device_get(xg_off)), label)
      self.assertEqual(_bits(l_cand), _bits(loss_off), label)
      self.assertEqual(_bits(xg_cand), _bits(xg_off), label)
      for (p, g_off_leaf), (_, g_on_leaf) in zip(
          jax.tree_util.tree_leaves_with_path(grads_off), jax.tree_util.tree_leaves_with_path(g_cand)
      ):
        err_label = f"{label}:{jax.tree_util.keystr(p)}"
        np.testing.assert_array_equal(
            np.asarray(jax.device_get(g_on_leaf)), np.asarray(jax.device_get(g_off_leaf)), err_msg=err_label
        )
        self.assertEqual(_bits(g_on_leaf), _bits(g_off_leaf), msg=err_label)

    loop_off, outer_off, _ = split_collectives(hlo_off)
    loop_on, outer_on, outer_lines_on = split_collectives(hlo_on)

    # Per-layer gathers disappear from the scan while-loop bodies.
    for layer_shape in ("64,4,12", "64,2,12", "4,12,64", "64,4"):
      self.assertGreater(count_gathers(loop_off, layer_shape), 0, msg=layer_shape)
      self.assertEqual(count_gathers(loop_on, layer_shape), 0, msg=layer_shape)

    # Forward hoists appear outside the loop with their actual param_scan_axis=1 shapes, and ZERO post-backward
    # full-size stacked gradient all-gathers ([2, 64, ...] or extra outer all-gathers) appear anywhere in HLO.
    self.assertEqual(sum(n for (op, _, _), n in outer_off.items() if op == "all-gather"), 0)
    self.assertEqual(count_gathers(outer_on, "64,2,4,12"), 1)
    self.assertEqual(count_gathers(outer_on, "64,2,2,12"), 2)
    self.assertEqual(count_gathers(outer_on, "4,2,12,64"), 1)
    self.assertEqual(count_gathers(outer_on, "64,2,4"), 1)
    for bwd_stack_shape in ("2,64,4,12", "2,64,2,12", "2,4,12,64", "2,64,4"):
      self.assertEqual(count_gathers(outer_on, bwd_stack_shape), 0, msg=bwd_stack_shape)
      self.assertEqual(count_gathers(loop_on, bwd_stack_shape), 0, msg=bwd_stack_shape)
    total_outer_ags = sum(n for (op, _, _), n in outer_on.items() if op == "all-gather")
    self.assertEqual(total_outer_ags, len(expected_all))
    outer_ag_lines = [l for l in outer_lines_on if _COLLECTIVE_RE.search(l) and "all-gather" in l]
    self.assertTrue(all("layer_params[" in l for l in outer_ag_lines), msg=outer_ag_lines)

    # B. Tighter 8 KiB cap: query (12 KiB) and out (12 KiB) exceed the cap and stay inside the scan loop.
    hlo_cap, loss_cap, grads_cap, _, _, capped_paths, _ = build_and_run(True, init_params, byte_cap=8 * 1024)
    self.assertEqual(
        capped_paths,
        {
            ("moe_block", "gate", "kernel"),
            ("self_attention", "key", "kernel"),
            ("self_attention", "value", "kernel"),
        },
    )
    self.assertEqual(_bits(loss_cap), _bits(loss_off))
    for (_, g_off_leaf), (_, g_cap_leaf) in zip(
        jax.tree.leaves_with_path(grads_off), jax.tree.leaves_with_path(grads_cap)
    ):
      self.assertEqual(_bits(g_cap_leaf), _bits(g_off_leaf))
    loop_cap, outer_cap, _ = split_collectives(hlo_cap)
    self.assertEqual(count_gathers(loop_cap, "64,2,12"), 0)
    self.assertEqual(count_gathers(loop_cap, "64,4"), 0)
    self.assertGreater(count_gathers(loop_cap, "64,4,12"), 0)
    self.assertGreater(count_gathers(loop_cap, "4,12,64"), 0)
    self.assertEqual(sum(n for (op, _, _), n in outer_cap.items() if op == "all-gather"), 3)

  def test_tensor_parallel_axes_stay_sharded_on_hoisted_stacks(self):
    """On (fsdp=2, tensor=2), hoisted stacks gather ONLY fsdp (group_size=2) and keep tensor sharded."""
    # pylint: disable=import-outside-toplevel,protected-access
    from maxtext.utils import maxtext_utils_nnx, model_creation_utils

    devices = jax.devices("cpu")[:4]
    if len(devices) < 4:
      self.skipTest("requires 4 CPU devices for (fsdp=2, tensor=2)")
    num_layers, emb_dim, q_heads, kv_heads, head_dim, mlp_dim, num_experts, seq_len = 2, 64, 4, 2, 12, 64, 4, 16
    cfg = _cfg(
        decoder_block="qwen3_moe",
        base_emb_dim=emb_dim,
        base_num_query_heads=q_heads,
        base_num_kv_heads=kv_heads,
        head_dim=head_dim,
        base_mlp_dim=mlp_dim,
        base_moe_mlp_dim=mlp_dim,
        num_experts=num_experts,
        num_experts_per_tok=2,
        base_num_decoder_layers=num_layers,
        vocab_size=128,
        max_target_length=seq_len,
        per_device_batch_size=1.0,
        ici_fsdp_parallelism=2,
        ici_tensor_parallelism=2,
        attention="dot_product",
        dtype="float32",
        weight_dtype="float32",
        scan_layers=True,
        remat_policy="full",
        gather_scanned_kernels_once=True,
    )
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg, devices), cfg.mesh_axes)
    with nn_partitioning.axis_rules(cfg.logical_axis_rules):
      create_model, abstract_model = model_creation_utils.create_nnx_abstract_model(
          cfg, mesh, model_mode=MODEL_MODE_TRAIN, rng_key=jax.random.PRNGKey(0)
      )
      model = maxtext_utils_nnx.create_nnx_sharded_model(abstract_model, create_model, mesh=mesh)
    decoder = model.decoder
    both = moe._gathered_kernel_shardings(decoder.layers, mesh, cfg.logical_axis_rules)
    # Query/key/value keep 'tensor' on heads (dim 2 of stacked [layers, emb, heads, head_dim]); out keeps 'tensor' on dim 1.
    self.assertEqual(both[("self_attention", "query", "kernel")][0].spec, P(None, None, "tensor", None))
    self.assertEqual(both[("self_attention", "key", "kernel")][0].spec, P(None, None, "tensor", None))
    self.assertEqual(both[("self_attention", "value", "kernel")][0].spec, P(None, None, "tensor", None))
    self.assertEqual(both[("self_attention", "out", "kernel")][0].spec, P(None, "tensor", None, None))
    self.assertEqual(both[("moe_block", "gate", "kernel")][0].spec, P(None, None, None))

    graphdef, params, rest = nnx.split(decoder.layers, nnx.Param, ...)
    param_shardings = jax.tree.map(lambda x: x.sharding, params)
    batch = int(cfg.global_batch_size_to_train_on)
    x_sharding = sharding.create_sharding(
        mesh, ("activation_batch", "activation_length", "activation_embed"), cfg.logical_axis_rules
    )
    x = jax.device_put(jnp.ones((batch, seq_len, emb_dim), dtype=jnp.float32), x_sharding)
    positions = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32), (batch, seq_len))
    segment_ids = jnp.ones((batch, seq_len), dtype=jnp.int32)

    def loss_fn(layer_params, hidden):
      scanned_layers = nnx.merge(graphdef, layer_params, rest)
      y = decoder._apply_layers_sequentially(
          scanned_layers, hidden, segment_ids, positions, True, MODEL_MODE_TRAIN, length=num_layers
      )[0]
      return jnp.mean(jnp.square(y))

    step = jax.jit(
        jax.value_and_grad(loss_fn, argnums=(0, 1)),
        in_shardings=(param_shardings, x_sharding),
        out_shardings=(NamedSharding(mesh, P()), (param_shardings, x_sharding)),
    )
    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
      hlo = step.lower(params, x).compile().as_text()

    comps = _computations(hlo)
    bodies = _while_bodies(comps)
    body_set = {c for reach in bodies.values() for c in reach}
    outer = _collectives([line for c, lines in comps.items() if c not in body_set for line in lines])
    outer_ags = {(dims, group): n for (op, dims, group), n in outer.items() if op == "all-gather"}
    # Every hoisted all-gather has group_size == 2 (fsdp=2, NOT fsdp*tensor=4) and TP-sharded head dimensions:
    # query: [64, 2, 4//2=2, 12], key/value: [64, 2, 2//2=1, 12], out: [4//2=2, 2, 12, 64], gate: [64, 2, 4].
    self.assertEqual(
        outer_ags,
        {
            (("64,2,2,12",), 2): 1,
            (("64,2,1,12",), 2): 2,
            (("2,2,12,64",), 2): 1,
            (("64,2,4",), 2): 1,
        },
    )

  def test_call_time_prefill_mode_skips_hoisting(self):
    """Forward-only prefill/non-train calls do not materialize hoisted stacks even when gather_scanned_kernels_once=True."""
    # pylint: disable=import-outside-toplevel,protected-access
    from maxtext.utils import maxtext_utils_nnx, model_creation_utils

    devices = jax.devices("cpu")[:4]
    if len(devices) < 2:
      self.skipTest("requires at least 2 CPU devices")
    cfg = _cfg(
        decoder_block="qwen3_moe",
        base_emb_dim=64,
        base_num_query_heads=4,
        base_num_kv_heads=2,
        head_dim=12,
        base_mlp_dim=64,
        base_moe_mlp_dim=64,
        num_experts=4,
        num_experts_per_tok=2,
        base_num_decoder_layers=2,
        vocab_size=128,
        max_target_length=16,
        max_prefill_predict_length=8,
        per_device_batch_size=1.0,
        ici_fsdp_parallelism=len(devices),
        attention="dot_product",
        dtype="float32",
        weight_dtype="float32",
        scan_layers=True,
        remat_policy="full",
        gather_scanned_kernels_once=True,
    )
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg, devices), cfg.mesh_axes)

    def outer_ag_count(init_mode, call_mode):
      with nn_partitioning.axis_rules(cfg.logical_axis_rules):
        create_model, abstract_model = model_creation_utils.create_nnx_abstract_model(
            cfg, mesh, model_mode=init_mode, rng_key=jax.random.PRNGKey(0)
        )
        model = maxtext_utils_nnx.create_nnx_sharded_model(abstract_model, create_model, mesh=mesh)
      decoder = model.decoder
      decoder.model_mode = MODEL_MODE_TRAIN
      graphdef, params, rest = nnx.split(decoder.layers, nnx.Param, ...)
      param_shardings = jax.tree.map(lambda x: x.sharding, params)
      rest_shardings = jax.tree.map(lambda x: getattr(x, "sharding", None), rest)
      batch = int(cfg.global_batch_size_to_train_on)
      x_sharding = sharding.create_sharding(
          mesh, ("activation_batch", "activation_length", "activation_embed"), cfg.logical_axis_rules
      )
      x = jax.device_put(jnp.ones((batch, 8, 64), dtype=jnp.float32), x_sharding)
      pos = jnp.broadcast_to(jnp.arange(8, dtype=jnp.int32), (batch, 8))
      seg = jnp.ones((batch, 8), dtype=jnp.int32)

      @functools.partial(jax.jit, in_shardings=(param_shardings, rest_shardings, x_sharding))
      def fwd(layer_params, layer_rest, hidden):
        scanned = nnx.merge(graphdef, layer_params, layer_rest)
        return decoder._apply_layers_sequentially(scanned, hidden, seg, pos, True, call_mode, length=2)[0]

      with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
        hlo = fwd.lower(params, rest, x).compile().as_text()
      comps = _computations(hlo)
      bodies = _while_bodies(comps)
      body_set = {c for reach in bodies.values() for c in reach}
      outer = _collectives([line for c, lines in comps.items() if c not in body_set for line in lines])
      return sum(n for (op, _, _), n in outer.items() if op == "all-gather")

    # Prefill without external stacked KV caches (internal KVCache_0, kv_caches_stacked=None) emits 0 outer hoists.
    self.assertEqual(outer_ag_count(MODEL_MODE_PREFILL, MODEL_MODE_PREFILL), 0)
    # Train-constructed decoder emits 5 outer hoists in MODEL_MODE_TRAIN.
    self.assertEqual(outer_ag_count(MODEL_MODE_TRAIN, MODEL_MODE_TRAIN), 5)


def _child_env(targets):
  """The environment of a `main` child process compiling for `targets` (comma-separated: cpu, tpu7x-8)."""
  env = os.environ.copy()
  env["XLA_FLAGS"] = f"{env.get('XLA_FLAGS', '')} --xla_force_host_platform_device_count={_NUM_DEVICES}".strip()
  # A TPU target needs the TPU plugin registered (and skipped as a backend, as there is no TPU here), possibly while
  # this process also has libtpu loaded.
  env["JAX_PLATFORMS"] = "cpu" if targets == "cpu" else ""
  if targets != "cpu":
    env["ALLOW_MULTIPLE_LIBTPU_LOAD"] = "1"
  env["QPGO_TARGETS"] = targets
  repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
  paths = [os.path.join(repo_root, "src"), repo_root] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
  env["PYTHONPATH"] = os.pathsep.join(paths)
  return env


def _run_child(targets):
  """Executes this test module as a child process compiling for `targets` and asserts it passes."""
  result = subprocess.run(
      [sys.executable, __file__],
      env=_child_env(targets),
      capture_output=True,
      text=True,
      check=False,
  )
  print(result.stdout)
  assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
  assert _PASS_TOKEN in result.stdout, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


@pytest.mark.cpu_only
def test_programs_in_a_child_process():
  """Runs `main` below in a fresh process with 8 CPU devices: values and the CPU compiler's programs."""
  _run_child("cpu")


@pytest.mark.cpu_only
@pytest.mark.skipif(
    importlib.util.find_spec("libtpu") is None,
    reason="needs libtpu to compile for a TPU target",
)
def test_tpu_programs_in_a_child_process():
  """The same program checks, compiled for a tpu7x-8 target on this host (no TPU used)."""
  _run_child("tpu7x-8")


# ----------------------------------------------------------------------------------------------------------------------
# Program checks (main).

_COMPUTATION_RE = re.compile(r"^(?:ENTRY )?%?([^\s(]+) .*\{\s*$")
_CALLEE_RE = re.compile(
    r"(?:body|condition|to_apply|calls|called_computations|branch_computations)=\{?([^,}\s]+(?:, *%?[^,}\s]+)*)\}?"
)
_COLLECTIVE_RE = re.compile(
    r"=\s*(\([^=]*?\)|\S+)\s+(all-gather|all-gather-start|reduce-scatter|all-reduce|all-reduce-start|all-to-all|"
    r"collective-permute|collective-permute-start)\("
)


def _group_size(line):
  """Devices per replica group, from any of the HLO printer's three notations (iota, mesh, explicit)."""
  iota = re.search(r"replica_groups=\[(\d+),(\d+)\]<=", line)
  if iota:
    return int(iota.group(2))
  mesh = re.search(r"replica_groups=mesh\[([^\]]*)\]\s*\{([^}]*)\}", line)
  if mesh:
    sizes = dict(re.findall(r"'([^']+)'=(\d+)", mesh.group(1)))
    return int(np.prod([int(sizes[a]) for a in re.findall(r"'([^']+)'", mesh.group(2))]))
  explicit = re.search(r"replica_groups=\{\{([^}]*)\}", line)
  return len(explicit.group(1).split(",")) if explicit else 0


def _computations(hlo):
  """{computation name: [instruction lines]} of an HLO module's text."""
  comps, name = {}, None
  for line in hlo.splitlines():
    if not line.startswith(" "):
      m = _COMPUTATION_RE.match(line)
      name = m.group(1).lstrip("%") if m else None
      if name:
        comps[name] = []
    elif name:
      comps[name].append(line)
  return comps


def _while_bodies(comps):
  """{while body name: set of computations it reaches (itself included)}."""
  callees = {
      name: {c.strip().lstrip("%") for line in lines for group in _CALLEE_RE.findall(line) for c in group.split(",")}
      for name, lines in comps.items()
  }
  bodies = {
      m.group(1).lstrip("%")
      for lines in comps.values()
      for line in lines
      if " while(" in line
      for m in [re.search(r"body=(%?[^,\s]+)", line)]
      if m
  }

  def reach(name, seen):
    if name in seen or name not in comps:
      return seen
    seen.add(name)
    for c in callees[name]:
      reach(c, seen)
    return seen

  return {b: reach(b, set()) for b in bodies}


def _collectives(lines):
  """Counter of (opcode, result dimensions, devices per group) of the collectives in `lines`.

  Element types and layouts are left out: the CPU compiler moves bfloat16 collectives to float32, and the same
  replica groups print in different notations depending on how the sharding was introduced.
  """
  out = collections.Counter()
  for line in lines:
    m = _COLLECTIVE_RE.search(line)
    if not m:
      continue
    dims = tuple(re.findall(r"[a-z0-9]+\[([0-9,]*)\]", m.group(1)))
    out[(m.group(2).replace("-start", ""), dims, _group_size(line))] += 1
  return out


@functools.cache
def _target_devices(target):
  """The devices to compile for: this process's CPU devices, or a TPU topology (compiled for, not run, on this host).

  A TPU topology must be built before anything initializes the local backends (main builds them first).
  """
  if target == "cpu":
    return jax.devices("cpu")
  spec = accelerator_to_spec_map.get_system_characteristics(target)
  return topologies.get_topology_desc(
      platform=spec.platform,
      topology_name=spec.topology_name,
      chip_config_name=spec.chip_config_name,
      chips_per_host_bounds=spec.chips_per_host_bounds,
      num_slices=1,
      wrap=spec.wrap,
  ).devices


def _run(mesh_name, options, target):
  """A scanned, rematerialized stack of Qwen3.5 attention layers: (compiled HLO text, values or None, cfg, mesh, jaxpr).

  The layers are scanned as nnx_decoders scans them (jax.checkpoint with the decoder's remat policy around each
  layer, the layer input saved under its checkpoint name). On CPU the step also runs: values = (loss, parameter
  gradients, input gradient). For a TPU target it is only compiled.
  """
  cfg = _cfg(**{**_SMALL_QWEN35, **_MESHES[mesh_name], **options})
  mesh = Mesh(maxtext_utils.create_device_mesh(cfg, _target_devices(target)), cfg.mesh_axes)
  rules = cfg.logical_axis_rules
  with nn_partitioning.axis_rules(rules):
    layers = [
        qwen3.Qwen3NextFullAttention(
            cfg,
            mesh,
            MODEL_MODE_TRAIN,
            layer_idx=3,
            rngs=nnx.Rngs(params=i, dropout=i),
        )
        for i in range(_LAYERS)
    ]
  graphdef, template, rest = nnx.split(layers[0], nnx.Param, ...)
  stacked = jax.tree.map(
      lambda *xs: jnp.stack(xs),
      *[nnx.to_pure_dict(nnx.split(l, nnx.Param, ...)[1]) for l in layers],
  )
  logical = nnx.to_pure_dict(nnx.get_partition_spec(template))
  param_shardings = jax.tree.map(
      lambda spec: NamedSharding(mesh, P(None, *sharding.logical_to_mesh_axes(spec, mesh, rules))),
      logical,
      is_leaf=lambda s: isinstance(s, P),
  )
  batch, seq = int(cfg.global_batch_size_to_train_on), cfg.max_target_length
  x_sharding = sharding.create_sharding(mesh, ("activation_batch", "activation_length", "activation_embed"), rules)
  x = jnp.asarray(np.random.default_rng(7).standard_normal((batch, seq, _EMB)), jnp.bfloat16)
  segment_ids = jnp.ones((batch, seq), jnp.int32)
  positions = jnp.broadcast_to(jnp.arange(seq, dtype=jnp.int32), (batch, seq))
  policy = nnx_decoders.NNXDecoder.get_remat_policy(
      types.SimpleNamespace(config=cfg, _effective_remat_policy=lambda: cfg.remat_policy)
  )

  def loss_fn(params, x):
    def layer_fn(carry, layer_params):
      carry = checkpoint_name(carry, "decoder_layer_input")
      state = jax.tree.map(lambda v: v, template)
      nnx.replace_by_pure_dict(state, layer_params)
      out, _ = nnx.merge(graphdef, state, rest)(carry, segment_ids, positions, True, MODEL_MODE_TRAIN)
      # A parameter-free RMS norm of the residual stream, as the decoder layer's post-attention norm: its backward needs
      # the attention output, so the backward replays the output projection too.
      y = (carry + out).astype(jnp.float32)
      y = y * jax.lax.rsqrt(jnp.mean(jnp.square(y), axis=-1, keepdims=True) + 1e-6)
      return y.astype(carry.dtype), None

    y, _ = jax.lax.scan(jax.checkpoint(layer_fn, policy=policy, prevent_cse=False), x, params)
    return jnp.mean(jnp.square(y.astype(jnp.float32)))

  step = jax.jit(
      jax.value_and_grad(loss_fn, argnums=(0, 1)),
      in_shardings=(param_shardings, x_sharding),
      out_shardings=(NamedSharding(mesh, P()), (param_shardings, x_sharding)),
  )
  abstract = jax.tree.map(
      lambda a, s: jax.ShapeDtypeStruct(a.shape, a.dtype, sharding=s),
      (stacked, x),
      (param_shardings, x_sharding),
  )
  with jax.set_mesh(mesh), nn_partitioning.axis_rules(rules):
    jaxpr = jax.make_jaxpr(jax.value_and_grad(loss_fn, argnums=(0, 1)))(*abstract).jaxpr
    compiled = step.lower(*abstract).compile()
  values = None
  if target == "cpu":
    loss, (grads, x_grad) = compiled(jax.device_put(stacked, param_shardings), jax.device_put(x, x_sharding))
    values = (loss, grads, x_grad)
  return compiled.as_text(), values, cfg, mesh, jaxpr


def _sub_jaxprs(eqn):
  for value in eqn.params.values():
    for item in value if isinstance(value, (tuple, list)) else (value,):
      if isinstance(item, jax_core.ClosedJaxpr):
        yield item.jaxpr
      elif isinstance(item, jax_core.Jaxpr):
        yield item


def _kernel_reads(jaxpr, kernel_shape):
  """[(dot output shape, source)] for every dot_general anywhere in the program that reads a `kernel_shape` operand.

  source is ("gathered", spec) when the operand is the output of a sharding constraint, else the name of whatever
  produced it ("input" for a program input). Values are followed through scan, remat and call boundaries: into a
  sub-jaxpr whose inputs are the eqn's operands one to one, and out of one whose results are the eqn's results.
  """
  producers = {}

  def producer(j, var):
    if id(j) not in producers:
      producers[id(j)] = {v: (eqn, i) for eqn in j.eqns for i, v in enumerate(eqn.outvars)}
    return producers[id(j)].get(var)

  def source(j, var, env):  # env: {input of j: (outer jaxpr, outer var, outer env)}, resolved lazily.
    if not isinstance(var, jax_core.Var):
      return "literal"
    found = producer(j, var)
    if found is None:
      return source(*env[var]) if var in env else "input"
    eqn, index = found
    if eqn.primitive.name == "sharding_constraint":
      return ("gathered", tuple(eqn.params["sharding"].spec))
    subs = [k for k in _sub_jaxprs(eqn) if len(k.invars) == len(eqn.invars) and len(k.outvars) == len(eqn.outvars)]
    if len(subs) == 1:
      return source(
          subs[0],
          subs[0].outvars[index],
          {iv: (j, ov, env) for iv, ov in zip(subs[0].invars, eqn.invars)},
      )
    return eqn.primitive.name

  reads = []

  def walk(j, env):
    for eqn in j.eqns:
      if eqn.primitive.name == "dot_general":
        for operand in eqn.invars:
          if tuple(getattr(operand.aval, "shape", ())) == kernel_shape:
            reads.append((tuple(eqn.outvars[0].aval.shape), source(j, operand, env)))
      for sub in _sub_jaxprs(eqn):
        walk(
            sub,
            {iv: (j, ov, env) for iv, ov in zip(sub.invars, eqn.invars)} if len(sub.invars) == len(eqn.invars) else {},
        )

  walk(jaxpr, {})
  return reads


def _strip_spec(spec):
  spec = list(spec)
  while spec and spec[-1] is None:
    spec.pop()
  return tuple(spec)


def _check(name, good, detail=""):
  print(
      f"{'PASS' if good else 'FAIL'} {name}{': ' + detail if detail else ''}",
      flush=True,
  )
  return good


def _gathered_dims(cfg, mesh):
  """{kernel: its gathered per-device dimensions}: the embed dimension whole, the rest sharded as the rules say."""

  def shards(logical):
    axes = sharding.logical_to_mesh_axes((logical,), mesh, cfg.logical_axis_rules)[0]
    return 1 if axes is None else int(np.prod([mesh.shape[a] for a in np.atleast_1d(axes)]))

  return {
      "query": f"{_EMB},{_HEADS // shards('q_heads')},{2 * _HEAD_DIM}",
      "out": f"{_HEADS * _HEAD_DIM // shards('mlp')},{_EMB}",
  }


def _is_gather_of(op, dims, gathered):
  """Whether a collective produces a gathered kernel (the TPU compiler may keep the scan's leading 1)."""
  return op == "all-gather" and any(d in (gathered, "1," + gathered) for d in dims)


def _program(hlo, gathered):
  """({kernel: sorted per-loop-body gather counts}, Counter of every collective in the module)."""
  comps = _computations(hlo)
  bodies = _while_bodies(comps)
  per_body = [_collectives([l for c in reach for l in comps[c]]) for reach in bodies.values()]
  counts = {
      kernel: sorted(sum(n for (op, d, _), n in body.items() if _is_gather_of(op, d, dims)) for body in per_body)
      for kernel, dims in gathered.items()
  }
  return counts, _collectives([l for lines in comps.values() for l in lines])


# The kernels' global shapes in the jaxpr, and the logical axes the attention layer gathers them to.
_KERNEL_SHAPES = {
    "query": (_EMB, _HEADS, 2 * _HEAD_DIM),
    "out": (_HEADS * _HEAD_DIM, _EMB),
}
_GATHERED_AXES = {"query": (None, "q_heads", "kv"), "out": ("mlp", None)}
# The projections' input shapes, the result shape of their input-gradient dots (batch 8 devices x 0.5 = 4).
_INPUT_SHAPES = {"query": (4, _SEQ, _EMB), "out": (4, _SEQ, _HEADS * _HEAD_DIM)}

# Per (target, mesh): the query kernel's all-gathers in the [forward, backward] loop bodies (sorted), option off and on.
# Whether the compiler already shares one gather between the two backward dots depends on the compiler and the program;
# with the option the backward has exactly one gather of the kernel and no other collective changes.
_EXPECTED = {
    ("cpu", "cp_as_ep"): ([1, 1], [1, 1]),
    ("cpu", "default"): ([1, 1], [1, 1]),
    ("tpu7x-8", "cp_as_ep"): ([1, 2], [1, 1]),
    ("tpu7x-8", "default"): ([1, 2], [1, 1]),
}


def _check_mesh(mesh_name, target):
  """Compiles (and on CPU runs) the stack with the option off and on, and checks values and programs."""
  ok = True
  runs = {
      "off": _run(mesh_name, {}, target),
      "on": _run(mesh_name, {_QUERY_KEY: True}, target),
  }
  _, off_values, cfg, mesh, _ = runs["off"]
  gathered = _gathered_dims(cfg, mesh)
  programs = {arm: _program(run[0], gathered) for arm, run in runs.items()}
  if os.environ.get("QPGO_DUMP_DIR"):
    for arm, run in runs.items():
      path = os.path.join(os.environ["QPGO_DUMP_DIR"], f"{target}.{mesh_name}.{arm}.hlo.txt")
      with open(path, "w", encoding="utf-8") as f:
        f.write(run[0])
  name = f"{target} {mesh_name}"
  on_values = runs["on"][1]
  if off_values is not None:
    # Values: the loss and every gradient, bit for bit.
    (off_loss, off_grads, off_xg), (on_loss, on_grads, on_xg) = (
        off_values,
        on_values,
    )
    flat_off = jax.tree_util.tree_leaves_with_path(off_grads)
    flat_on = jax.tree_util.tree_leaves_with_path(on_grads)
    differing = [jax.tree_util.keystr(p) for (p, a), (_, b) in zip(flat_off, flat_on) if _bits(a) != _bits(b)]
    same = _bits(off_loss) == _bits(on_loss) and _bits(off_xg) == _bits(on_xg) and len(flat_off) == len(flat_on)
    ok &= _check(
        f"{name}: loss, input gradient and {len(flat_off)} parameter gradients bit-identical",
        same and not differing,
        f"loss {float(off_loss)!r} vs {float(on_loss)!r}, differing {differing}",
    )
  # The program handed to the compiler: with the option every dot that reads the query kernel (forward, replay and
  # input gradient) reads the explicitly gathered kernel; without it none does. This holds whichever compiler runs it.
  shape = _KERNEL_SHAPES["query"]
  want_spec = _strip_spec(sharding.create_sharding(mesh, _GATHERED_AXES["query"], cfg.logical_axis_rules).spec)
  on_reads, off_reads = _kernel_reads(runs["on"][4], shape), _kernel_reads(runs["off"][4], shape)
  good_on = [r for r in on_reads if isinstance(r[1], tuple) and _strip_spec(r[1][1]) == want_spec]
  gathered_off = [r for r in off_reads if isinstance(r[1], tuple)]
  ok &= _check(
      f"{name}: all {len(on_reads)} dots reading the query kernel {list(shape)} read it gathered to {want_spec};"
      f" off: {len(gathered_off)} of {len(off_reads)}",
      len(on_reads) == len(off_reads) >= 3
      and len(good_on) == len(on_reads)
      and _INPUT_SHAPES["query"] in {r[0] for r in good_on}
      and not gathered_off,
      f"on {on_reads}, off {off_reads}",
  )
  (off_counts, off_every), (on_counts, on_every) = programs["off"], programs["on"]
  want_off, want_on = _EXPECTED[(target, mesh_name)]
  ok &= _check(
      f"{name}: query kernel [{gathered['query']}] all-gathers per loop body, off {off_counts['query']} -> on"
      f" {on_counts['query']}",
      off_counts["query"] == want_off and on_counts["query"] == want_on,
      f"want {want_off} -> {want_on}",
  )
  ok &= _check(
      f"{name}: output kernel [{gathered['out']}] all-gathers unchanged",
      on_counts["out"] == off_counts["out"],
      f"{off_counts['out']} -> {on_counts['out']}",
  )
  # No other collective changes (the kernel gradient is still reduce-scattered, not all-reduced).
  removed, added = off_every - on_every, on_every - off_every
  want_removed = sum(want_off) - sum(want_on)
  ok &= _check(
      f"{name}: removes {want_removed} query kernel all-gather(s) and changes no other collective",
      sum(removed.values()) == want_removed
      and all(_is_gather_of(op, d, gathered["query"]) for op, d, _ in removed)
      and not added,
      f"removed {dict(removed)}, added {dict(added)}",
  )
  return ok


def _check_gather_scanned_kernels_once(target):
  """Verifies gather_scanned_kernels_once HLO on 8-device `target` across (fsdp=8) and (fsdp=4, tensor=2)."""
  # pylint: disable=import-outside-toplevel,protected-access
  from maxtext.utils import model_creation_utils

  ok = True
  for label, fsdp, tp in (("fsdp8", 8, 1), ("fsdp4_tp2", 4, 2)):
    cfg = _cfg(
        decoder_block="qwen3_moe",
        base_emb_dim=256,
        base_num_query_heads=8,
        base_num_kv_heads=4,
        head_dim=32,
        base_mlp_dim=256,
        base_moe_mlp_dim=256,
        num_experts=8,
        num_experts_per_tok=2,
        base_num_decoder_layers=2,
        vocab_size=256,
        max_target_length=64,
        per_device_batch_size=1.0,
        ici_fsdp_parallelism=fsdp,
        ici_tensor_parallelism=tp,
        attention="dot_product",
        dtype="bfloat16",
        weight_dtype="float32",
        scan_layers=True,
        remat_policy="full",
        gather_scanned_kernels_once=True,
    )
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg, _target_devices(target)), cfg.mesh_axes)
    rules = cfg.logical_axis_rules
    with nn_partitioning.axis_rules(rules):
      _, abstract_model = model_creation_utils.create_nnx_abstract_model(
          cfg, mesh, model_mode=MODEL_MODE_TRAIN, rng_key=jax.random.PRNGKey(0)
      )
      decoder = abstract_model.decoder
      param_shardings = sharding.nnx_construct_named_sharding(nnx.state(decoder.layers, nnx.Param), mesh)
    graphdef, abs_params, abs_rest = nnx.split(decoder.layers, nnx.Param, ...)
    abs_params_sharded = jax.tree.map(
        lambda v, s: v.replace(jax.ShapeDtypeStruct(v.shape, v.dtype, sharding=s.get_value())),
        abs_params,
        param_shardings,
        is_leaf=lambda v: isinstance(v, nnx.Variable),
    )
    rest_concrete = jax.tree.map(lambda v: jnp.zeros(v.shape, v.dtype) if hasattr(v, "shape") else v, abs_rest)
    batch = _NUM_DEVICES
    x_sharding = sharding.create_sharding(mesh, ("activation_batch", "activation_length", "activation_embed"), rules)
    abs_x = jax.ShapeDtypeStruct((batch, 64, 256), jnp.bfloat16, sharding=x_sharding)

    def loss_fn(layer_params, hidden, decoder=decoder, graphdef=graphdef, rest_concrete=rest_concrete, batch=batch):
      pos = jnp.broadcast_to(jnp.arange(64, dtype=jnp.int32), (batch, 64))
      seg = jnp.ones((batch, 64), dtype=jnp.int32)
      scanned = nnx.merge(graphdef, layer_params, rest_concrete)
      y = decoder._apply_layers_sequentially(scanned, hidden, seg, pos, True, MODEL_MODE_TRAIN, length=2)[0]
      return jnp.mean(jnp.square(y.astype(jnp.float32)))

    step = jax.jit(
        jax.value_and_grad(loss_fn, argnums=(0, 1)),
        in_shardings=(param_shardings, x_sharding),
        out_shardings=(NamedSharding(mesh, P()), (param_shardings, x_sharding)),
    )
    with jax.set_mesh(mesh), nn_partitioning.axis_rules(rules):
      hlo = step.lower(abs_params_sharded, abs_x).compile().as_text()

    comps = _computations(hlo)
    bodies = _while_bodies(comps)
    body_set = {c for reach in bodies.values() for c in reach}
    outer_lines = [line for c, lines in comps.items() if c not in body_set for line in lines]
    body_lines = [line for c in body_set for line in comps[c]]
    outer_ags = {k: n for k, n in _collectives(outer_lines).items() if k[0] == "all-gather"}
    body_ags = {k: n for k, n in _collectives(body_lines).items() if k[0] == "all-gather"}
    # Hoisted kernels: CPU gathers before moveaxis ([d0, 2, ...]); TPU gathers after moveaxis ([2, d0, ...]).
    # In both cases there are EXACTLY 5 group=fsdp hoists (1 per kernel in forward, 0 post-backward gradient gathers)
    # and 0 group=8 (full-mesh) gathers when tensor parallelism is active.
    q_h, kv_h = 8 // tp, 4 // tp
    if target == "cpu":
      want_fsdp_hoists = {
          ("all-gather", (f"256,2,{q_h},32",), fsdp): 1,
          ("all-gather", (f"256,2,{kv_h},32",), fsdp): 2,
          ("all-gather", (f"{q_h},2,32,256",), fsdp): 1,
          ("all-gather", ("256,2,8",), fsdp): 1,
      }
    else:
      want_fsdp_hoists = {
          ("all-gather", (f"2,256,{q_h},32",), fsdp): 1,
          ("all-gather", (f"2,256,{kv_h},32",), fsdp): 2,
          ("all-gather", (f"2,{q_h},32,256",), fsdp): 1,
          ("all-gather", ("2,256,8",), fsdp): 1,
      }
    got_fsdp_hoists = {k: n for k, n in outer_ags.items() if k[2] == fsdp}
    got_full_mesh_ags = {k: n for k, n in outer_ags.items() if tp > 1 and k[2] == fsdp * tp}
    ok &= _check(
        f"{target} {label} gather_scanned_kernels_once: exact 5 FSDP-only outer hoists (group={fsdp}) and 0 bwd stack"
        " all-gathers",
        got_fsdp_hoists == want_fsdp_hoists and not got_full_mesh_ags,
        f"got_fsdp {got_fsdp_hoists}, full_mesh {got_full_mesh_ags}, want {want_fsdp_hoists}",
    )
    in_loop_hoisted = sum(
        n
        for (op, dims, _), n in body_ags.items()
        if any(_is_gather_of(op, dims, s) for s in (f"256,{q_h},32", f"256,{kv_h},32", f"{q_h},32,256", "256,8"))
    )
    ok &= _check(
        f"{target} {label} gather_scanned_kernels_once: 0 remaining in-loop gathers of hoisted kernels",
        in_loop_hoisted == 0,
        f"body_ags {body_ags}",
    )
  return ok


def main():
  targets = os.environ.get("QPGO_TARGETS", "cpu").split(",")
  # TPU topologies first: building one after the local backends are initialized fails.
  for target in sorted(targets, key=lambda t: t == "cpu"):
    assert len(_target_devices(target)) == _NUM_DEVICES, (
        target,
        _target_devices(target),
    )
  ok = True
  for target in targets:
    for mesh_name in _MESHES:
      ok &= _check_mesh(mesh_name, target)
    ok &= _check_gather_scanned_kernels_once(target)
  if not ok:
    sys.exit(1)
  print(_PASS_TOKEN, flush=True)


if __name__ == "__main__":
  main()
