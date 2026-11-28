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

"""Tests for the MPMD pipeline's model (models.py) on CPU devices.

These run the MPMD-only code paths (pre-gather, deferred expert-gradient reduce-scatter) without ``mpmd.jit``, so they
need neither Shardy MPMD nor Pathways.
"""

import dataclasses
import gc
import re
import unittest
from unittest import mock

from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
import numpy as np

from maxtext.common.common_types import DecoderBlockType
from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.utils import model_creation_utils

from maxtext.experimental.mpmd.partir import config as config_lib
from maxtext.experimental.mpmd.partir import hparams as hp
from maxtext.experimental.mpmd.partir import mesh_utils
from maxtext.experimental.mpmd.partir import models
from maxtext.experimental.mpmd.partir import train_step as ts


def IDENTITY(name):  # pylint: disable=invalid-name
  """A stage wrapper that leaves the stage as it is: the MPMD-only code paths, without mpmd.jit."""
  del name
  return lambda fn: fn


def cpu_devices(count):
  """The first `count` CPU devices, or a skip when there are fewer."""
  devices = jax.devices("cpu")
  if len(devices) < count:
    raise unittest.SkipTest(f"needs {count} CPU devices (XLA_FLAGS=--xla_force_host_platform_device_count={count})")
  return devices[:count]


def rel_l2(mine, theirs):
  mine, theirs = np.asarray(mine, np.float64), np.asarray(theirs, np.float64)
  return np.linalg.norm(mine - theirs) / max(np.linalg.norm(theirs), 1e-30)


def batch_of(rows, seq_len, vocab_size, seed=0):
  """A synthetic batch: random tokens, targets shifted by one, no padding."""
  tokens = np.random.default_rng(seed).integers(0, vocab_size, size=(rows, seq_len + 1), dtype=np.int32)
  positions = np.tile(np.arange(seq_len + 1, dtype=np.int32), (rows, 1))
  segmentation = np.ones((rows, seq_len), dtype=np.int32)
  return {
      "inputs": tokens[:, :-1],
      "inputs_position": positions[:, :-1],
      "inputs_segmentation": segmentation,
      "targets": tokens[:, 1:],
      "targets_position": positions[:, 1:],
      "targets_segmentation": segmentation,
  }


def build(hparams, devices, unit_shard_fn=None):
  """The MaxText config, stage topology, graphdef and parameters `runner.build` would make, without Shardy."""
  cfg = config_lib.build_maxtext_config(hparams)
  topology = mesh_utils.create_stage_meshes(devices, hparams.num_stages, hparams.fsdp, hparams.expert)
  graphdef, params = models.PipelinedTransformer.init_split(
      config=cfg,
      mesh=topology.meshes["m0"],
      rngs=nnx.Rngs(hparams.seed),
      unit_shard_fn=unit_shard_fn,
      remat_policy=hparams.mpmd_remat_policy,
      stage_layer_counts=hparams.stage_layer_counts,
  )
  return cfg, topology, graphdef, params


def deepseek3_tiny(**kw):
  return hp.Hparams(
      model_name="deepseek3-tiny",
      max_target_length=16,
      emb_dim=128,
      mlp_dim=256,
      vocab_size=512,
      first_num_dense_layers=1,
      maxtext_overrides=("dtype=float32",),
      **kw,
  )


# A model of each supported decoder block (hparams.SUPPORTED_DECODER_BLOCKS).
MODELS_BY_BLOCK = {
    "simple_mlp": "simple",
    "deepseek": "deepseek3-tiny",
    "llama2": "llama3.1-8b",
    "mistral": "mistral-7b",
    "mixtral": "mixtral-8x7b",
    "qwen3": "qwen3-8b",
    "qwen3_moe": "qwen3-30b-a3b",
}


class SupportedBlocksTest(unittest.TestCase):
  """Every supported decoder block builds through MaxText's private NNXDecoder helpers and runs forward."""

  def test_every_supported_block_builds_and_runs(self):
    devices = cpu_devices(1)
    small = {"num_layers": 2, "num_stages": 1, "max_target_length": 8, "emb_dim": 64, "mlp_dim": 128, "vocab_size": 128}
    self.assertEqual(set(MODELS_BY_BLOCK), set(hp.SUPPORTED_DECODER_BLOCKS))
    for block, model_name in MODELS_BY_BLOCK.items():
      with self.subTest(block=block):
        h = hp.Hparams(model_name=model_name, **small)
        cfg, _, graphdef, params = build(h, devices)
        self.assertEqual(cfg.decoder_block.value, block)
        self.assertEqual(graphdef.num_layers, 2)
        batch = batch_of(2, 8, 128)
        logits = jax.jit(lambda p, b, g=graphdef: models.forward(g, p, b, wrap=models.UNWRAPPED))(params, batch)
        self.assertEqual(logits.shape, (2, 8, 128))
        self.assertTrue(bool(jnp.isfinite(logits).all()))

  def test_refuses_an_unsupported_block_at_build_time(self):
    cfg = config_lib.build_maxtext_config(hp.Hparams(model_name="simple", num_layers=2, num_stages=1))
    with self.assertRaisesRegex(ValueError, "Unsupported decoder_block"):
      models.PipelinedTransformer(
          cfg.replace(decoder_block=DecoderBlockType.GPT3),
          Mesh(np.asarray(cpu_devices(1)), ("fsdp",)),
          rngs=nnx.Rngs(0),
          remat_policy="none",
          stage_layer_counts=(),
      )


class MaxTextTransformerTest(unittest.TestCase):
  """The model computes MaxText's: from the parameters of MaxText's own (unscanned) NNX Transformer, the same logits.

  Parity compares MPMD against this same model unpipelined, so it cannot see the model drift from MaxText's. The
  output head once had no final norm (llama3.1-8b's first loss 24.8, MaxText's 12.3). simple_mlp is left out: MaxText's
  own unscanned NNX decoder returns logits of shape [vocab] for it.
  """

  SMALL = {"num_layers": 2, "num_stages": 1, "max_target_length": 8, "emb_dim": 64, "mlp_dim": 128, "vocab_size": 128}

  def assert_logits_match(self, hparams):
    """Builds both models, copies MaxText's parameters into the per-unit dict, and compares float32 logits."""
    devices = cpu_devices(1)
    cfg, _, graphdef, params = build(hparams, devices)
    with nn_partitioning.axis_rules(cfg.logical_axis_rules):
      reference = model_creation_utils.from_config(cfg, devices=devices, rngs=nnx.Rngs(1))
    reference_def, reference_state = nnx.split(reference)
    leaves = {
        jax.tree_util.keystr(path): leaf
        for path, leaf in jax.tree_util.tree_flatten_with_path(nnx.state(reference, nnx.Param))[0]
    }
    decoder = nnx.state(reference, nnx.Param)["decoder"]

    def index(prefix):
      return [key for _, key in sorted((int(k[len(prefix) :]), k) for k in decoder if re.fullmatch(prefix + r"\d+", k))]

    # MaxText's decoder layers from input to output (DeepSeek: its dense layers, then its MoE layers).
    layer_keys = index("dense_layers_") + index("moe_layers_") + index("layers_")
    self.assertEqual(len(layer_keys), graphdef.num_layers)

    def maxtext_path(path):
      unit, rest = re.fullmatch(r"\['(\w+)'\](.*)", path).groups()
      if unit == mesh_utils.EMBED_NAME:
        return "['token_embedder']" + rest
      if unit == mesh_utils.UNEMBED_NAME:  # decoder_norm and logits_dense
        return "['decoder']" + rest
      return f"['decoder']['{layer_keys[int(unit[len('layer'):])]}']" + rest

    flat, treedef = jax.tree_util.tree_flatten_with_path(params)
    paths = [maxtext_path(jax.tree_util.keystr(path)) for path, _ in flat]
    self.assertEqual(sorted(paths), sorted(leaves))  # every parameter of MaxText's model, each once
    for (_, leaf), path in zip(flat, paths):
      self.assertEqual(leaf.shape, leaves[path].shape, path)
    params = jax.tree_util.tree_unflatten(treedef, [leaves[path] for path in paths])

    batch = batch_of(2, hparams.max_target_length, hparams.vocab_size)
    mine = jax.jit(lambda p, b: models.forward(graphdef, p, b, wrap=models.UNWRAPPED))(params, batch)

    def run_reference(state, b):
      logits = nnx.merge(reference_def, state)(
          b["inputs"],
          b["inputs_position"],
          decoder_segment_ids=b["inputs_segmentation"],
          enable_dropout=False,
          model_mode=MODEL_MODE_TRAIN,
      )
      return logits[0] if isinstance(logits, tuple) else logits

    with nn_partitioning.axis_rules(cfg.logical_axis_rules):
      theirs = jax.jit(run_reference)(reference_state, batch)
    self.assertEqual(mine.shape, theirs.shape)
    self.assertLess(rel_l2(mine, theirs), 1e-5)

  def test_same_logits_as_maxtexts_transformer(self):
    float32 = ("dtype=float32", "weight_dtype=float32", "matmul_precision=highest")
    for block, model_name in MODELS_BY_BLOCK.items():
      if block == "simple_mlp":
        continue
      with self.subTest(block=block):
        dense = {"first_num_dense_layers": 1} if block == "deepseek" else {}
        self.assert_logits_match(hp.Hparams(model_name=model_name, maxtext_overrides=float32, **dense, **self.SMALL))
    with self.subTest(block="deepseek with hyper-connections"):
      self.assert_logits_match(
          hp.Hparams(
              model_name="deepseek3-tiny",
              first_num_dense_layers=1,
              maxtext_overrides=(*float32, "mhc_expansion_rate=4"),
              **self.SMALL,
          )
      )


class ModelSettingsTest(unittest.TestCase):
  """The MPMD-only settings travel with the model, not as extra keys of MaxText's config."""

  def test_graphdef_carries_the_remat_policy_and_stage_layer_counts(self):
    h = hp.Hparams(
        model_name="simple",
        num_layers=4,
        num_stages=2,
        emb_dim=64,
        mlp_dim=128,
        vocab_size=128,
        max_target_length=8,
        remat=True,
        remat_policy="minimal",
        stage_layer_counts=(1, 3),
    )
    cfg, _, graphdef, params = build(h, cpu_devices(2))
    self.assertEqual((graphdef.remat_policy, graphdef.stage_layer_counts), ("minimal", (1, 3)))
    self.assertEqual(
        models._unit_stages(graphdef, params),  # pylint: disable=protected-access
        [("embed", 0), ("layer0", 0), ("layer1", 1), ("layer2", 1), ("layer3", 1), ("unembed", 1)],
    )
    # Required: a caller that forgot them would silently lose remat or uneven stages.
    with self.assertRaisesRegex(TypeError, "missing 2 required keyword-only arguments: 'remat_policy' and"):
      models.PipelinedTransformer.init_split(cfg, graphdef.mesh, rngs=nnx.Rngs(0))  # pylint: disable=missing-kwoa
    fields = {f.name: getattr(graphdef, f.name) for f in dataclasses.fields(graphdef) if f.name != "remat_policy"}
    with self.assertRaisesRegex(TypeError, "missing 1 required positional argument: 'remat_policy'"):
      models.PipelineGraphDef(**fields)
    # A copy of MaxText's config (replace() rebuilds it from its typed fields) still describes the same model.
    self.assertEqual(cfg.replace(learning_rate=0.5).remat_policy, cfg.remat_policy)


def checkpoint_policies(jaxpr):
  """The policy of every jax.checkpoint in `jaxpr` and the jaxprs it calls, in order (None: save nothing)."""
  found = []
  for eqn in jaxpr.eqns:
    if eqn.primitive.name in ("checkpoint", "remat2"):
      found.append(eqn.params["policy"])
    for value in eqn.params.values():
      inner = getattr(value, "jaxpr", value)
      if hasattr(inner, "eqns"):
        found += checkpoint_policies(inner)
  return found


class RematPolicyTest(unittest.TestCase):
  """Each layer is checkpointed with the policy MaxText's own resolver (NNXDecoder.get_remat_policy) gives the name."""

  def test_maxtexts_resolver_takes_every_policy(self):
    """hparams.REMAT_POLICIES is a literal: each must be a name MaxText's resolver (and so its trainer) accepts."""
    cfg = config_lib.build_maxtext_config(hp.Hparams(model_name="simple", num_layers=2, num_stages=1))
    for policy in hp.REMAT_POLICIES:
      enabled, jax_policy = models._resolve_remat_policy(cfg, policy)  # pylint: disable=protected-access
      self.assertEqual((enabled, jax_policy is None), (True, policy == "full"), policy)
    self.assertEqual(models._resolve_remat_policy(cfg, "none"), (False, None))  # pylint: disable=protected-access

  def test_every_layer_is_checkpointed_with_it(self):
    devices = cpu_devices(1)
    small = {"num_layers": 3, "num_stages": 1, "max_target_length": 8, "emb_dim": 64, "mlp_dim": 128, "vocab_size": 128}
    for remat, policy in ((False, "full"), (True, "full"), (True, "save_dot_except_mlp")):
      _, _, graphdef, params = build(hp.Hparams(model_name="simple", remat=remat, remat_policy=policy, **small), devices)
      jaxpr = jax.make_jaxpr(lambda p, b, g=graphdef: models.forward(g, p, b, wrap=models.UNWRAPPED))(
          params, batch_of(2, 8, 128)
      )
      found = checkpoint_policies(jaxpr.jaxpr)
      if not remat:
        self.assertEqual(found, [])
        continue
      # The embedding, then each of the 3 layers with the policy.
      self.assertEqual([p is None for p in found], [True] + [policy == "full"] * 3, policy)


class HyperConnectionsTest(unittest.TestCase):
  """DeepSeek's mHC streams, expanded after the embedding and summed before the output head as MaxText's decoder does."""

  SMALL = {"num_layers": 2, "num_stages": 1, "max_target_length": 8, "emb_dim": 64, "mlp_dim": 128, "vocab_size": 128}

  def test_deepseek_builds_and_runs_with_hyper_connections(self):
    devices = cpu_devices(1)
    plain = hp.Hparams(model_name="deepseek3-tiny", **self.SMALL)
    counts = {}
    for name, hparams in (
        ("plain", plain),
        ("mhc", dataclasses.replace(plain, maxtext_overrides="mhc_expansion_rate=4")),
    ):
      cfg, _, graphdef, params = build(hparams, devices)
      counts[name] = sum(leaf.size for leaf in jax.tree.leaves(params))
      logits = jax.jit(lambda p, b, g=graphdef: models.forward(g, p, b, wrap=models.UNWRAPPED))(
          params, batch_of(2, 8, 128)
      )
      self.assertEqual(logits.shape, (2, 8, 128))
      self.assertTrue(bool(jnp.isfinite(logits).all()), name)
    self.assertEqual(cfg.mhc_expansion_rate, 4)
    self.assertGreater(counts["mhc"], counts["plain"])  # each layer's attention and MLP hyper-connections

  def test_refused_for_layers_without_streams(self):
    hparams = hp.Hparams(model_name="simple", maxtext_overrides="mhc_expansion_rate=4", **self.SMALL)
    with self.assertRaisesRegex(ValueError, "does not build mhc_expansion_rate=4"):
      config_lib.build_maxtext_config(hparams)


class SuppressFsdpPsumTest(unittest.TestCase):
  """The private JAX function the deferred expert-gradient reduce-scatter patches."""

  def test_jax_still_has_it(self):
    # A rename in jax would turn the deferral off (correct, but one reduce-scatter per microbatch).
    self.assertTrue(models.can_defer_moe_reduce_scatter())

  def test_3d_inputs_keep_their_partial_gradients(self):
    """Without the patch the transpose of a check_vma=False shard_map psums a 3D input's gradient over fsdp."""
    mesh = Mesh(np.asarray(cpu_devices(2)), ("fsdp",))
    weight = jnp.ones((2, 3, 4), jnp.float32)
    x = jax.device_put(jnp.arange(2 * 3 * 4, dtype=jnp.float32).reshape(2, 3, 4) + 1, NamedSharding(mesh, P("fsdp")))
    body = jax.shard_map(
        lambda w, v: jnp.sum(w * v, keepdims=True),
        mesh=mesh,
        in_specs=(P(None, None, None), P("fsdp")),
        out_specs=P("fsdp"),
        check_vma=False,
    )
    grad_fn = jax.jit(jax.grad(lambda w, v: jnp.sum(body(w, v)), argnums=0))
    halves = np.asarray(x).reshape(2, 1, 3, 4)
    summed = grad_fn(weight, x)
    for shard in summed.addressable_shards:
      np.testing.assert_allclose(np.asarray(shard.data), np.broadcast_to(halves.sum(axis=0), (2, 3, 4)))
    with models.suppress_fsdp_psum_of_3d_inputs():
      jax.clear_caches()
      partial = jax.jit(jax.grad(lambda w, v: jnp.sum(body(w, v)), argnums=0))(weight, x)
    jax.clear_caches()
    shards = sorted(partial.addressable_shards, key=lambda shard: mesh.devices.tolist().index(shard.device))
    for index, shard in enumerate(shards):
      np.testing.assert_allclose(np.asarray(shard.data), np.broadcast_to(halves[index], (2, 3, 4)))


class DeferredExpertGradTest(unittest.TestCase):
  """MoE expert gradients under FSDP must match the unwrapped step whatever pre-gathering covers."""

  def check_expert_grads(self, h, budgets):
    """Every MoE unit's expert gradients against the unwrapped step's, at each of budgets(unit bytes)."""
    _, _, graphdef, params = build(h, cpu_devices(h.num_stages * h.fsdp))
    # Every stage's units on the one mesh a plain jit runs on.
    params = models.shard_tree_on_fsdp(graphdef.mesh, params, graphdef.fsdp_size, device_put=True)
    rows = int(h.per_device_batch_size * h.fsdp)
    batch = jax.device_put(batch_of(rows, h.max_target_length, 512), NamedSharding(graphdef.mesh, P("fsdp")))

    def step(params, batch):
      preg = models.pregather_params(graphdef, params, microbatch_rows=batch["inputs"].shape[0], wrap=IDENTITY)
      with models.defer_fsdp_shard_map_psum(graphdef, preg, params):
        carry = models.build_grad_carry(graphdef, params, preg)
        grads = jax.grad(
            lambda p: ts.compute_loss(graphdef, p, batch, vocab_size=512, wrap=IDENTITY, pregathered_params=preg)
        )(carry)
      return models.reduce_scatter_deferred_grads(graphdef, params, preg, grads, num_microbatches=1, wrap=IDENTITY)

    reference = jax.jit(jax.grad(lambda p, b: ts.compute_loss(graphdef, p, b, vocab_size=512, wrap=models.UNWRAPPED)))(
        params, batch
    )
    moe_units = [name for name in params if "DeepSeekMoeBlock_0" in params[name]]
    self.assertTrue(moe_units)  # a renamed MoE block must not pass vacuously
    unit_bytes = {
        name: models._unit_pregather_bytes(  # pylint: disable=protected-access
            graphdef.mesh, params[name], graphdef.fsdp_size, graphdef.config
        )
        for name in params
    }
    saved = models._MAX_UNIT_PREGATHER_BYTES  # pylint: disable=protected-access
    self.addCleanup(setattr, models, "_MAX_UNIT_PREGATHER_BYTES", saved)
    for budget in budgets(unit_bytes):
      models._MAX_UNIT_PREGATHER_BYTES = budget  # pylint: disable=protected-access
      # The budget is read while tracing, and jit(step) would replay the first budget's trace.
      grads = jax.jit(lambda p, b: step(p, b))(params, batch)  # pylint: disable=unnecessary-lambda
      for unit in moe_units:
        mine, theirs = (tree[unit]["DeepSeekMoeBlock_0"]["MoeBlock_0"] for tree in (grads, reference))
        for name in ("wi_0", "wi_1", "wo"):
          self.assertLess(rel_l2(mine[name].get_value(), theirs[name].get_value()), 1e-4, (budget, unit, name))

  def test_expert_grads_whatever_is_pregathered(self):
    # Every unit; only embed and unembed (the MoE layer over budget: the case that dropped the FSDP
    # reduction of the expert gradients); nothing.
    self.check_expert_grads(
        deepseek3_tiny(num_layers=2, num_stages=1, fsdp=4, num_microbatches=1, per_device_batch_size=1.0),
        lambda b: (1 << 30, b["embed"] + b["unembed"], 1),
    )

  def test_one_moe_unit_pregathered_and_another_not(self):
    """Two stages: every unit, then a budget only stage 0 (embed, layer0, MoE layer1) fits, so that stage 1 (MoE
    layer2 and layer3, unembed) is not gathered. Gathering layer1's experts deferred the psum of every expert weight,
    so layer2's and layer3's were never reduced."""
    self.check_expert_grads(
        deepseek3_tiny(
            num_layers=4, num_stages=2, fsdp=2, num_microbatches=1, per_device_batch_size=1.0, stage_layer_counts=(2, 2)
        ),
        lambda b: (1 << 30, b["embed"] + b["layer0"] + b["layer1"]),
    )

  def test_expert_grads_without_the_private_jax_function(self):
    """If jax renames the function the deferral patches, every expert gradient is reduced in its microbatch."""
    with mock.patch.object(models, "can_defer_moe_reduce_scatter", lambda: False):
      self.check_expert_grads(
          deepseek3_tiny(num_layers=2, num_stages=1, fsdp=4, num_microbatches=1, per_device_batch_size=1.0),
          lambda b: (1 << 30,),
      )

  def test_microbatch_the_stage_cannot_split(self):
    """2-row microbatches on 4 devices: MaxText's MoE runs all of a microbatch on every device, and the deferred
    reduce-scatter added up 4 copies of the expert gradients. make_train_step's MPMD branch against its SPMD branch,
    on the step-1 gradients in AdamW's first moment."""
    h = deepseek3_tiny(num_layers=2, num_stages=1, fsdp=4, num_microbatches=4, per_device_batch_size=2.0)
    _, _, graphdef, params = build(h, cpu_devices(4))
    params = models.shard_tree_on_fsdp(graphdef.mesh, params, graphdef.fsdp_size, device_put=True)
    state = ts.create_train_state(params, h.learning_rate)
    batch = jax.device_put(batch_of(8, h.max_target_length, 512), NamedSharding(graphdef.mesh, P("fsdp")))
    with (
        mock.patch.object(models, "NAMED_COMPUTATION", IDENTITY),
        mock.patch.object(ts, "microbatch_call", lambda fn, call_counter: fn),
    ):
      mpmd, spmd = (
          jax.jit(ts.make_train_step(graphdef, vocab_size=512, num_microbatches=4, mode=mode))(state, batch)[0]
          .opt_state[0]
          .mu
          for mode in (ts.ExecutionMode.MPMD, ts.ExecutionMode.SPMD)
      )
    theirs = dict(jax.tree_util.tree_flatten_with_path(spmd)[0])
    for path, mine in jax.tree_util.tree_flatten_with_path(mpmd)[0]:
      self.assertLess(rel_l2(mine, theirs[path]), 1e-4, jax.tree_util.keystr(path))

    # Still deferred for microbatches of 4 rows.
    def defers(rows):
      pregathered = jax.eval_shape(
          lambda p: models.pregather_params(graphdef, p, microbatch_rows=rows, wrap=IDENTITY), params
      )
      return models._defers_moe_reduce_scatter(graphdef, params, pregathered)  # pylint: disable=protected-access

    self.assertEqual((defers(2), defers(4)), (False, True))


class InitSplitTest(unittest.TestCase):

  def test_no_unsharded_weights_left_on_device_0(self):
    """MaxText's Attention outlives the split (its query_init closure is in the GraphDef): each layer used to leave
    its unsharded float32 q/k/v/o kernels on device 0, 5 GiB for llama3.1-8b."""
    h = hp.Hparams(
        model_name="llama3.1-8b",
        num_layers=2,
        num_stages=2,
        fsdp=2,
        num_microbatches=2,
        per_device_batch_size=1.0,
        max_target_length=16,
        emb_dim=128,
        mlp_dim=256,
        vocab_size=512,
    )
    devices = cpu_devices(4)
    topology = mesh_utils.create_stage_meshes(devices, h.num_stages, h.fsdp)
    name_to_mesh = mesh_utils.create_name_to_mesh_assignment(num_layers=h.num_layers, num_stages=h.num_stages)

    def shard_unit(name, unit_params):  # as runner.build places each unit on its stage's mesh
      return models.shard_tree_on_fsdp(topology.meshes[name_to_mesh[name]], unit_params, h.fsdp, device_put=True)

    before = {id(a) for a in jax.live_arrays()}
    _, _, _, params = build(h, devices, unit_shard_fn=shard_unit)
    gc.collect()
    held = {id(a) for a in jax.tree_util.tree_leaves(params)}
    left = [
        (a.shape, str(a.dtype))
        for a in jax.live_arrays()
        if id(a) not in before | held and devices[0] in a.devices() and a.nbytes >= 1024
    ]
    self.assertEqual(left, [])  # RNG keys and scalars stay; no weights


if __name__ == "__main__":
  unittest.main()
