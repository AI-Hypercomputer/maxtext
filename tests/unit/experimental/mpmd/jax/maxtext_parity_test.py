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

"""maxtext_parity's weight mapping, chunk files and comparison; its `init` and synthetic batch against MaxText
itself (CPU, tiny DeepSeek and Qwen3.5-family dims).

    python3 -m pytest -q tests/unit/experimental/mpmd/jax/maxtext_parity_test.py
"""

import functools
import json
import os
import subprocess
import sys
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maxtext.experimental.mpmd.jax import ds_ep_pipeline as dsp
from maxtext.experimental.mpmd.jax import maxtext_parity as mp
from maxtext.experimental.mpmd.jax import testing
from maxtext.experimental.mpmd.jax.qwen35 import config as qc
from maxtext.experimental.mpmd.jax.qwen35 import model as qm

ROOT = testing.import_root()
# A ds_ep pipeline of 2 stages x 2 chunks of one layer, and the same model in MaxText's terms.
TINY = (
    "--emb 256 --heads 2 --q-lora 128 --kv-lora 128 --moe-mlp 256 --experts 8 --topk 2 --vocab 512 "
    "--seq 256 --sa-block 128 --sa-block-kv-compute 128 --gmm-impl ragged_dot --ragged-sort off "
    "--attn reference --routing topk --stages 2 --vv 2 --fsdp 1 --ep 1 --gbs 4 --mb 2 --layers 4"
).split()
MAXTEXT_TINY = (
    "base_emb_dim=256 base_num_query_heads=2 base_num_kv_heads=2 q_lora_rank=128 kv_lora_rank=128 "
    "base_moe_mlp_dim=256 base_mlp_dim=256 num_experts=8 num_experts_per_tok=2 vocab_size=512 "
    "base_num_decoder_layers=4 pipeline_parallel_layers=4 num_layers_per_pipeline_stage=1 "
    "num_pipeline_repeats=2 ici_pipeline_parallelism=2 max_target_length=256 attention=dot_product "
    "megablox=false"
).split()
# DeepSeek-V3's features on the tiny model: 2 dense layers of 5 (a wider MLP), routed biases, 4 routing groups
# of 2 experts with 2 kept, the fp32 router, the sequence-wise aux loss, one MTP layer; chunks of 1, 2, 1, 1
# layers (chunk 1 mixed); and the same model as a scanned MaxText (`init --exact` adds the routed bias and MTP).
EXACT_TINY = (
    TINY
    + (
        "--layers 5 --dense-layers 2 --mlp 384 --routed-bias on --bias-update-rate 0.01 "
        "--routing-groups 4 --topk-groups 2 --float32-gate-logits on --lb-loss-weight 0.01 --mtp 1 "
        "--layer-split 1,2,1,1"
    ).split()
)
MAXTEXT_EXACT_TINY = (
    "base_emb_dim=256 base_num_query_heads=2 base_num_kv_heads=2 q_lora_rank=128 kv_lora_rank=128 "
    "base_moe_mlp_dim=256 base_mlp_dim=384 num_experts=8 num_experts_per_tok=2 vocab_size=512 "
    "base_num_decoder_layers=5 first_num_dense_layers=2 n_routing_groups=4 topk_routing_group=2 "
    "routed_bias_update_rate=0.01 float32_gate_logits=true load_balance_loss_weight=0.01 "
    "moe_use_megatron_seq_aux_loss=true max_target_length=256 attention=dot_product "
    "megablox=false use_tokamax_gmm=false"
).split()
# The same pipeline with DeepSeek-V3's structure: 2 dense layers (MLP 384) ahead of the 4 MoE layers, in chunk 0,
# and MaxText's model of it without PP (scanned layers): the layout of a DP-over-DCN run.
TINY_DENSE = TINY[: TINY.index("--layers")] + ["--layers", "6", "--dense-layers", "2", "--mlp", "384"]
MAXTEXT_TINY_DP = (
    "base_emb_dim=256 base_num_query_heads=2 base_num_kv_heads=2 q_lora_rank=128 kv_lora_rank=128 "
    "base_moe_mlp_dim=256 base_mlp_dim=384 num_experts=8 num_experts_per_tok=2 vocab_size=512 "
    "base_num_decoder_layers=6 first_num_dense_layers=2 ici_pipeline_parallelism=1 "
    "max_target_length=256 attention=dot_product megablox=false"
).split()
# Qwen3.8-2.4T's layers at tiny widths (ds_ep_qwen_test.Q_TINY's), two blocks of four: a ds_ep pipeline of 2 stages
# x 2 chunks cut inside the blocks, and the same model as MaxText's qwen3.8-2.4t-a95b yml cut down, scanned (no PP).
QWEN_TINY = (
    "--model qwen3.8-2.4t-a95b --layers 8 --emb 256 --heads 4 --kv-heads 2 --head-dim 64 --gdn-key-heads 2 "
    "--gdn-value-heads 16 --gdn-key-head-dim 32 --gdn-value-head-dim 32 --moe-mlp 256 --experts 8 --topk 2 "
    "--vocab 512 --seq 256 --sa-block 128 --sa-block-kv-compute 128 --gmm-impl ragged_dot --ragged-sort off "
    "--attn dot_product --routing topk --stages 2 --vv 2 --fsdp 1 --ep 1 --gbs 4 --mb 2 "
    "--layer-split 1,3,3,1"
).split()
MAXTEXT_QWEN_TINY = (
    "base_emb_dim=256 base_num_decoder_layers=8 base_num_query_heads=4 base_num_kv_heads=2 head_dim=64 "
    "base_mlp_dim=256 base_moe_mlp_dim=256 num_experts=8 num_experts_per_tok=2 gdn_num_key_heads=2 "
    "gdn_num_value_heads=16 gdn_key_head_dim=32 gdn_value_head_dim=32 gdn_chunk_size=64 "
    "vocab_size=512 max_target_length=256 attention=dot_product megablox=false "
    "ici_pipeline_parallelism=1"
).split()
# MaxText's dataset_type=synthetic batch as its trainer draws it (train.initialize also sets the PRNG
# implementation) -> argv[1]; argv[2:] are MaxText key=value overrides.
MAXTEXT_BATCH = """
import os, sys
import jax, numpy as np, maxtext
from maxtext.input_pipeline import input_pipeline_interface
from maxtext.trainers.pre_train import train
from maxtext.utils import maxtext_utils
cfg, _ = train.initialize([sys.argv[0], os.path.join(os.path.dirname(maxtext.__file__), 'configs', 'base.yml'),
                           *sys.argv[2:]])
mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
batch = next(input_pipeline_interface.create_data_iterator(cfg, mesh)[0])
np.savez(sys.argv[1], inputs=np.asarray(batch['inputs']), targets=np.asarray(batch['targets']))
"""
# maxtext.experimental.mpmd.jax.deepseek.config.from_maxtext of a MaxText config (argv[1:]: key=value overrides) ->
# one line of what it
# carried; its reads of keys this MaxText lacks raise.
DEEPSEEK_FROM_MAXTEXT = """
import os, sys
import maxtext
from maxtext.configs import pyconfig
from maxtext.experimental.mpmd.jax.deepseek import config as ds_config
cfg = pyconfig.initialize([sys.argv[0], os.path.join(os.path.dirname(maxtext.__file__), 'configs', 'base.yml'),
                           *sys.argv[1:]])
c = ds_config.from_maxtext(cfg)
print('carried', c.first_num_dense_layers, c.n_routing_groups, c.topk_routing_group, c.mtp_num_layers,
      c.moe_use_megatron_seq_aux_loss)
"""


@functools.cache
def tiny_pipeline(dense=False):
  return dsp.build(dsp.make_parser().parse_args(TINY_DENSE if dense else TINY))


@functools.cache
def qwen_tiny_pipeline():
  return dsp.build(dsp.make_parser().parse_args(QWEN_TINY))


@functools.cache
def exact_tiny_pipeline():
  return dsp.build(dsp.make_parser().parse_args(EXACT_TINY))


def as_maxtext_scanned(trees, scan_axis=1):
  """ds_ep chunk trees -> a scanned MaxText's names and layout {path tuple: array} (MaxText without PP), what
  maxtext_to_chunks maps back: decoder/dense_layers and decoder/moe_layers stack their layers on
  scan_axis (a routed bias on axis 0), the MTP layer under mtp_block/mtp_layer_1/mtp_1_*, one token embedding."""
  stacks, flat = {}, {}
  layer = lambda rest: (
      (("DeepSeekMoeBlock_0",) + (() if rest[1] == "shared_experts" else ("MoeBlock_0",)) + rest[1:])
      if rest[0] == "moe"
      else rest
  )
  for tree in trees:
    for path, v in jax.tree_util.tree_flatten_with_path(tree)[0]:
      keys, v = tuple(k.key for k in path), np.asarray(v)
      if keys[0] in ("dense_layers", "layers"):
        name = ("decoder", "dense_layers" if keys[0] == "dense_layers" else "moe_layers") + layer(keys[1:])
        stacks.setdefault(name, []).extend(v)  # the chunk's layers, in order
      elif keys[0] == "mtp":
        if keys[1] != "token_embedder":  # the copy of the decoder's table
          rest = (
              (("mtp_1_transformer_layer",) + layer(keys[2:])) if keys[1] == "layer" else (f"mtp_1_{keys[1]}",) + keys[2:]
          )
          flat[("mtp_block", "mtp_layer_1") + rest] = v
      else:
        flat[keys if keys[0] == "token_embedder" else ("decoder",) + keys] = v
  for name, per_layer in stacks.items():
    flat[name] = np.stack(per_layer, axis=0 if name[-2:] == ("gate", "bias") else scan_axis)
  return flat


def as_maxtext(trees, n_stages):
  """ds_ep chunk trees -> MaxText's names and layout {path tuple: array}, what maxtext_to_chunks maps
  back: a layer leaf stacked [repeat, stage, ...], the routed experts under DeepSeekMoeBlock_0/MoeBlock_0."""
  layers, flat = {}, {}
  for tree in trees:
    for path, v in jax.tree_util.tree_flatten_with_path(tree)[0]:
      keys = tuple(k.key for k in path)
      if keys[0] == "layers":
        rest = keys[1:]
        if rest[0] == "moe":
          rest = ("DeepSeekMoeBlock_0",) + (() if rest[1] == "shared_experts" else ("MoeBlock_0",)) + rest[1:]
        layers.setdefault(rest, []).append(np.asarray(v)[0])
      else:
        flat[keys if keys[0] == "token_embedder" else ("decoder",) + keys] = np.asarray(v)
  for rest, per_chunk in layers.items():
    stacked = np.stack(per_chunk).reshape(len(trees) // n_stages, n_stages, *per_chunk[0].shape)
    flat[("decoder", "pipeline_module", "layers") + rest] = stacked
  return flat


def as_maxtext_qwen(port, scan_axis):
  """A maxtext.experimental.mpmd.jax.qwen35 parameter tree (the block axis first, a block's layers layer_0..) -> a
  scanned MaxText qwen3_5's names and layout {path tuple: array}, what qwen_to_chunks maps back: the block axis at
  scan_axis, and a block's layers as MaxText's nested scan holds them, local_layers -- the linear layers stacked on
  the axis after the block axis -- and global_layer."""
  flat, layers = {}, port["decoder"]["layers"]
  cycle = len(layers)
  for path, v in jax.tree_util.tree_flatten_with_path(port)[0]:
    keys = tuple(k.key for k in path)
    if keys[:2] != ("decoder", "layers"):
      flat[keys] = v
    elif keys[2] == f"layer_{cycle - 1}":
      flat[("decoder", "layers", "global_layer") + keys[3:]] = np.moveaxis(v, 0, scan_axis)
    elif keys[2] == "layer_0":
      leaf = lambda i, rest=keys[3:]: functools.reduce(lambda t, k: t[k], rest, layers[f"layer_{i}"])
      local = np.stack([leaf(i) for i in range(cycle - 1)], axis=1)  # [NB, cycle - 1, ...]
      flat[("decoder", "layers", "local_layers") + keys[3:]] = np.moveaxis(local, (0, 1), (scan_axis, scan_axis + 1))
  return flat


def qwen_chunk_key(j, rest):
  """A port layer's leaf (its path under layer_<i>) -> its key as the j-th layer of a ds_ep chunk: the mixer,
  MaxText's 'attention', is 'gdn' -- or 'attn' for gated attention, whose leaves sit under 'attention'/'attention'."""
  if rest[0] == "attention":
    rest = ("attn",) + rest[2:] if rest[1] == "attention" else ("gdn",) + rest[1:]
  return "/".join(("layers", f"{j:02d}") + rest)


def leaves(tree):
  """{path: (shape, dtype)} of a chunk tree; the path as keystr, so a flat 'a/b' key is not ['a']['b']."""
  return {
      jax.tree_util.keystr(path): (tuple(v.shape), str(v.dtype))
      for path, v in jax.tree_util.tree_flatten_with_path(tree)[0]
  }


def run(args, devs=2, timeout=600):
  """python3 args on `devs` CPU devices, from the directory the package imports from -> (returncode, output)."""
  env = dict(
      os.environ,
      JAX_PLATFORMS="cpu",
      XLA_FLAGS=f"--xla_force_host_platform_device_count={devs}",
      PYTHONPATH=testing.child_pythonpath(__name__),
  )
  r = subprocess.run([sys.executable, *args], cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout)
  return r.returncode, r.stdout + r.stderr


def check(t, rc, out, refusal=()):
  """The run `run` gave, its returncode and output: a success, or with `refusal` a non-zero exit with each of those
  strings in its output."""
  if refusal:
    t.assertNotEqual(rc, 0, out[-5000:])
    for s in refusal:
      t.assertIn(s, out, out[-5000:])
  else:
    t.assertEqual(rc, 0, out[-5000:])


class MaxTextToChunks(unittest.TestCase):

  def test_circular_mapping(self):
    """Chunk c is MaxText's [repeat c // S, stage c % S]; the MoE block is renamed; the embedding goes to
    chunk 0 and the head to the last chunk."""
    S, R = 4, 2
    wi = np.arange(R * S * 3).reshape(R, S, 3)
    flat = {
        ("decoder", "pipeline_module", "layers", "DeepSeekMoeBlock_0", "MoeBlock_0", "wi_0"): wi,
        ("decoder", "pipeline_module", "layers", "DeepSeekMoeBlock_0", "shared_experts", "wo", "kernel"): wi,
        ("decoder", "pipeline_module", "layers", "self_attention", "wq_a", "kernel"): wi,
        ("token_embedder", "embedding"): np.zeros(2),
        ("decoder", "decoder_norm", "scale"): np.ones(2),
        ("decoder", "logits_dense", "kernel"): np.ones(3),
    }
    chunks = mp.maxtext_to_chunks(flat, S, R * S)
    self.assertEqual(len(chunks), R * S)
    for c, tree in enumerate(chunks):
      for k in ("layers/moe/wi_0", "layers/moe/shared_experts/wo/kernel", "layers/self_attention/wq_a/kernel"):
        np.testing.assert_array_equal(tree[k], wi[c // S, c % S][None])
    self.assertIn("token_embedder/embedding", chunks[0])
    self.assertEqual(
        {k for k in chunks[-1] if not k.startswith("layers/")}, {"decoder_norm/scale", "logits_dense/kernel"}
    )
    self.assertFalse([k for t in chunks[1:-1] for k in t if not k.startswith("layers/")])

  def test_scanned_layout(self):
    """Without PP: a scanned leaf's layer axis (MaxText's param_scan_axis) comes first; the MoE layers go to the
    chunks in the layer split's counts, the dense layers to chunk 0; a split that does not hold the layers, or a
    scanned leaf without a split, is an error."""
    E, L, D = 3, 5, 2
    wi = np.arange(E * L * D).reshape(E, L, D)  # [E, layer, D]: MaxText's scan axis 1
    norm = np.arange(D * L).reshape(D, L)
    dense = np.arange(4 * 2).reshape(4, 2)  # [D, 2 dense layers]
    flat = {
        ("decoder", "moe_layers", "DeepSeekMoeBlock_0", "MoeBlock_0", "wi_0"): wi,
        ("decoder", "moe_layers", "pre_self_attention_layer_norm", "scale"): norm,
        ("decoder", "dense_layers", "mlp", "wi_0", "kernel"): dense,
        ("token_embedder", "embedding"): np.zeros(2),
        ("decoder", "decoder_norm", "scale"): np.ones(2),
        ("decoder", "logits_dense", "kernel"): np.ones(3),
    }
    split = [2, 1, 2]
    chunks = mp.maxtext_to_chunks(flat, 1, 3, split, 1)
    lo = 0
    for c, n in enumerate(split):
      np.testing.assert_array_equal(chunks[c]["layers/moe/wi_0"], np.moveaxis(wi, 1, 0)[lo : lo + n])
      np.testing.assert_array_equal(chunks[c]["layers/pre_self_attention_layer_norm/scale"], norm.T[lo : lo + n])
      lo += n
    np.testing.assert_array_equal(chunks[0]["dense_layers/mlp/wi_0/kernel"], dense.T)
    # slices of MaxText's arrays, not copies: a big init holds every chunk until the files are written
    self.assertTrue(all(np.shares_memory(chunks[c]["layers/moe/wi_0"], wi) for c in range(3)))
    self.assertFalse([k for t in chunks[1:] for k in t if k.startswith("dense_layers/")])
    self.assertIn("token_embedder/embedding", chunks[0])
    self.assertEqual(
        {k for k in chunks[-1] if not k.startswith("layers/")}, {"decoder_norm/scale", "logits_dense/kernel"}
    )
    with self.assertRaisesRegex(ValueError, "5 MoE layers, the layer split"):
      mp.maxtext_to_chunks(flat, 1, 3, [1, 1, 1], 1)
    with self.assertRaisesRegex(ValueError, "2 chunks, not 3"):
      mp.maxtext_to_chunks(flat, 1, 3, [2, 3], 1)
    with self.assertRaisesRegex(ValueError, "give the layer split"):
      mp.maxtext_to_chunks(flat, 1, 3)

  def test_one_repeat_and_unknown_leaves(self):
    """At one repeat MaxText has no repeat axis; a leaf with no chunk to go to is an error."""
    v = np.arange(4 * 2).reshape(4, 2)
    flat = {("decoder", "pipeline_module", "layers", "pre_self_attention_layer_norm", "scale"): v}
    chunks = mp.maxtext_to_chunks(flat, 4, 4)
    for c in range(4):
      np.testing.assert_array_equal(chunks[c]["layers/pre_self_attention_layer_norm/scale"], v[c][None])
    with self.assertRaisesRegex(ValueError, "no place"):
      mp.maxtext_to_chunks({("decoder", "mtp_block", "w"): v}, 4, 4)

  def test_scanned_mapping(self):
    """A scanned MaxText: dense and MoE layer leaves cut out of their stacks (param_scan_axis; the routed
    bias, an MoEBiasVar, on axis 0) into the chunks of the layer split, the MTP layer and a copy of the
    token embedding to the last chunk."""
    split, n_dense = (1, 2, 1, 1), 2  # chunk 1: dense layer 1 and MoE layer 2
    dense = np.arange(4 * 2 * 3).reshape(4, 2, 3)  # [emb, layer, mlp]
    moe = np.arange(8 * 3 * 2).reshape(8, 3, 2)  # [expert, layer, .]
    bias = np.arange(3 * 8, dtype=np.float32).reshape(3, 8)
    emb = np.ones((5, 4))
    D, M, T = ("decoder", "dense_layers"), ("decoder", "moe_layers"), ("mtp_block", "mtp_layer_1")
    flat = {
        D + ("mlp", "wi_0", "kernel"): dense,
        M + ("DeepSeekMoeBlock_0", "MoeBlock_0", "wi_0"): moe,
        M + ("DeepSeekMoeBlock_0", "MoeBlock_0", "gate", "bias"): bias,
        M + ("DeepSeekMoeBlock_0", "shared_experts", "wo", "kernel"): moe,
        T + ("mtp_1_projection", "kernel"): np.zeros(3),
        T + ("mtp_1_transformer_layer", "DeepSeekMoeBlock_0", "MoeBlock_0", "gate", "bias"): np.zeros(8),
        T + ("mtp_1_transformer_layer", "self_attention", "wq_a", "kernel"): np.ones(2),
        ("token_embedder", "embedding"): emb,
        ("decoder", "decoder_norm", "scale"): np.ones(4),
        ("decoder", "logits_dense", "kernel"): np.ones(3),
    }
    c = mp.maxtext_to_chunks(flat, 1, len(split), split, 1)
    np.testing.assert_array_equal(c[0]["dense_layers/mlp/wi_0/kernel"], dense[:, :1].swapaxes(0, 1))
    np.testing.assert_array_equal(c[1]["dense_layers/mlp/wi_0/kernel"], dense[:, 1:2].swapaxes(0, 1))
    for chunk, moe_layer in ((1, 0), (2, 1), (3, 2)):
      sel = slice(moe_layer, moe_layer + 1)
      np.testing.assert_array_equal(c[chunk]["layers/moe/wi_0"], moe[:, sel].swapaxes(0, 1))
      np.testing.assert_array_equal(c[chunk]["layers/moe/shared_experts/wo/kernel"], moe[:, sel].swapaxes(0, 1))
      np.testing.assert_array_equal(c[chunk]["layers/moe/gate/bias"], bias[sel])
    self.assertEqual(sorted(c[0]), ["dense_layers/mlp/wi_0/kernel", "token_embedder/embedding"])
    self.assertEqual(
        sorted(k for k in c[3] if not k.startswith("layers/")),
        [
            "decoder_norm/scale",
            "logits_dense/kernel",
            "mtp/layer/moe/gate/bias",
            "mtp/layer/self_attention/wq_a/kernel",
            "mtp/projection/kernel",
            "mtp/token_embedder/embedding",
        ],
    )
    np.testing.assert_array_equal(c[3]["mtp/token_embedder/embedding"], emb)

  def test_scanned_refusals(self):
    """A stack of another depth than the model's, and a leaf with no chunk to go to, are errors."""
    with self.assertRaisesRegex(ValueError, r"MaxText scans 2 MoE layers, the layer split \[2, 3\] holds 5$"):
      mp.maxtext_to_chunks({("decoder", "moe_layers", "x"): np.zeros((4, 2))}, 1, 2, (2, 3), 1)
    with self.assertRaisesRegex(ValueError, "stacks 3 layers on axis 1, its others 2"):
      mp.maxtext_to_chunks(
          {("decoder", "moe_layers", "x"): np.zeros((4, 2)), ("decoder", "moe_layers", "y"): np.zeros((4, 3))},
          1,
          2,
          (1, 1),
          1,
      )
    with self.assertRaisesRegex(ValueError, "no place"):
      mp.maxtext_to_chunks({("decoder", "mtp_block", "w"): np.zeros(2)}, 1, 1)

  def test_qwen_layouts_and_unknown_leaves(self):
    """A Qwen3.5-family MaxText without PP, a block's layers as its nested scan holds them (local_layers +
    global_layer), at param_scan_axis 0 and 1 -> qwen_to_chunks: layer l, position l % 4 of block l // 4, is a
    subtree of its chunk (the mixer as gdn or attn), each leaf the port tree's slice bit for bit and the memory of
    MaxText's array (no copy); the embedding goes to chunk 0, the final norm and the head to the last.  A MaxText
    leaf with no place in a chunk -- a vision encoder's, a decoder-level one, one beside the blocks' layers, a block
    layer by any other name (layer_0: the port's own layout, not this MaxText's), one beside a full-attention
    layer's inner attention -- is an error, as is a nested scan of another length."""
    cfg = qc.tiny()  # 8 layers: two blocks of three GDN layers + attention
    cycle, split = cfg.inhomogeneous_layer_cycle_interval, (1, 3, 3, 1)
    rng = np.random.default_rng(0)
    port = jax.tree.map(lambda a: rng.standard_normal(a.shape).astype(np.float32), qm.param_avals(cfg))
    layers = port["decoder"]["layers"]
    for axis in (0, 1):
      with self.subTest(param_scan_axis=axis):
        flat = as_maxtext_qwen(port, axis)
        self.assertTrue(any(p[2] == "local_layers" for p in flat if p[1] == "layers"))
        chunks = mp.qwen_to_chunks(flat, split, cycle, axis)
        self.assertEqual(len(chunks), len(split))
        l = 0
        for c, n in enumerate(split):
          keys = {"token_embedder/embedding"} if c == 0 else set()
          keys |= {"decoder_norm/scale", "logits_dense/kernel"} if c == len(split) - 1 else set()
          for j in range(n):
            for path, v in jax.tree_util.tree_flatten_with_path(layers[f"layer_{l % cycle}"])[0]:
              key = qwen_chunk_key(j, tuple(k.key for k in path))
              keys.add(key)
              np.testing.assert_array_equal(chunks[c][key], v[l // cycle : l // cycle + 1])
            l += 1
          self.assertEqual(set(chunks[c]), keys)
          self.assertTrue(all(any(np.shares_memory(a, v) for v in flat.values()) for a in chunks[c].values()))
        np.testing.assert_array_equal(chunks[0]["token_embedder/embedding"], port["token_embedder"]["embedding"])
        np.testing.assert_array_equal(chunks[-1]["logits_dense/kernel"], port["decoder"]["logits_dense"]["kernel"])
    flat = as_maxtext_qwen(port, 1)
    for extra in (
        ("vision_encoder", "patch", "kernel"),
        ("decoder", "extra_norm", "scale"),
        ("decoder", "layers", "block_norm", "scale"),
        ("decoder", "layers", "layer_0", "x"),
    ):
      with self.subTest(extra="/".join(extra)):
        with self.assertRaisesRegex(ValueError, f'MaxText parameter {"/".join(extra)} has no place in a ds_ep chunk'):
          mp.qwen_to_chunks({**flat, extra: np.zeros(2)}, split, cycle, 1)
    longer = {
        p: np.concatenate([v, v[:, :, :1]], axis=2) if p[2:3] == ("local_layers",) else v for p, v in flat.items()
    }  # four local layers in a block of four
    with self.assertRaisesRegex(ValueError, r"MaxText's blocks hold \['layer_0', .*'layer_4'\], not a block of 4"):
      mp.qwen_to_chunks(longer, split, cycle, 1)
    extra = ("decoder", "layers", "global_layer", "attention", "x")  # [X, NB]: the block axis at 1, as its neighbours'
    beside = {**flat, extra: np.zeros((2, cfg.num_blocks))}  # beside the full-attention layer's inner attention
    with self.assertRaisesRegex(
        ValueError,
        rf"MaxText parameters \['attention/x'\] beside layer {cycle - 1}'s inner attention have no place in a ds_ep chunk",
    ):
      mp.qwen_to_chunks(beside, split, cycle, 1)


class ChunkFiles(unittest.TestCase):

  def test_maxtext_names_through_chunk_files(self):
    """MaxText-named weights -> maxtext_to_chunks -> nest -> ds_ep_pipeline.write_chunk_files (what `init`
    writes) -> Pipeline.chunk_trees_from (--init-from) and params_from_chunks: the pipeline's own weights
    back, bit for bit, in bfloat16."""
    p = tiny_pipeline()
    want = p.chunk_trees(p.init(jax.random.key(1))[0])
    self.assertEqual({v.dtype for v in jax.tree.leaves(want)}, {jnp.dtype(jnp.bfloat16)})
    with tempfile.TemporaryDirectory() as d:
      dsp.write_chunk_files(d, [mp.nest(t) for t in mp.maxtext_to_chunks(as_maxtext(want, p.S), p.S, p.C)])
      self.assertEqual(sorted(os.listdir(d)), [f"chunk{c}.npz" for c in range(p.C)])
      got = p.chunk_trees(p.params_from_chunks(p.chunk_trees_from(d)))
    self.assertEqual(jax.tree.structure(got), jax.tree.structure(want))
    for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(want)):
      self.assertEqual(a.dtype, b.dtype)
      np.testing.assert_array_equal(np.asarray(a).view(np.uint16), np.asarray(b).view(np.uint16))

  def test_exact_names_through_chunk_files(self):
    """DeepSeek-V3's features (dense, mixed and MoE chunks, routed biases, MTP and its embedding copy): the
    pipeline's own weights -> a scanned MaxText's names -> maxtext_to_chunks -> chunk files ->
    params_from_chunks, bit for bit."""
    p = exact_tiny_pipeline()
    want = p.chunk_trees(p.init(jax.random.key(1))[0])
    with tempfile.TemporaryDirectory() as d:
      dsp.write_chunk_files(
          d, [mp.nest(t) for t in mp.maxtext_to_chunks(as_maxtext_scanned(want), p.S, p.C, p.layer_split, 1)]
      )
      got = p.chunk_trees(p.params_from_chunks(p.chunk_trees_from(d)))
    self.assertEqual(jax.tree.structure(got), jax.tree.structure(want))
    for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(want)):
      self.assertEqual(a.dtype, b.dtype)
      np.testing.assert_array_equal(np.asarray(a).view(np.uint8), np.asarray(b).view(np.uint8))

  def test_scanned_maxtext_names_through_chunk_files(self):
    """The same for MaxText without PP (DP over DCN): a pipeline with dense layers in chunk 0, its weights in
    MaxText's scanned names and layout, back through the chunk files bit for bit."""
    p = tiny_pipeline(dense=True)
    want = p.chunk_trees(p.init(jax.random.key(2))[0])
    self.assertIn("dense_layers", want[0])
    split = [len(ch.layers) for ch in p.chunks]
    with tempfile.TemporaryDirectory() as d:
      dsp.write_chunk_files(d, [mp.nest(t) for t in mp.maxtext_to_chunks(as_maxtext_scanned(want), p.S, p.C, split, 1)])
      got = p.chunk_trees(p.params_from_chunks(p.chunk_trees_from(d)))
    self.assertEqual(jax.tree.structure(got), jax.tree.structure(want))
    for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(want)):
      self.assertEqual(a.dtype, b.dtype)
      np.testing.assert_array_equal(np.asarray(a).view(np.uint8), np.asarray(b).view(np.uint8))

  def test_qwen_names_through_chunk_files(self):
    """A Qwen3.5-family MaxText's names and layout (Qwen3.8-2.4T's layers at tiny widths, bf16, a block's layers as
    its nested scan holds them) -> qwen_to_chunks -> nest -> write_chunk_files -> Pipeline.chunk_trees_from and
    params_from_chunks of the Qwen pipeline, whose chunks cut inside the blocks: every leaf where the pipeline has
    it, back bit for bit.  A layer's leaf the pipeline does not have goes to its chunk, and params_from_chunks
    refuses it (one beside a full-attention layer's inner attention qwen_to_chunks refuses itself: MaxTextToChunks)."""
    p = qwen_tiny_pipeline()
    cfg = p.scfg.model
    cycle = cfg.inhomogeneous_layer_cycle_interval
    rng = np.random.default_rng(3)
    port = jax.tree.map(lambda a: rng.standard_normal(a.shape).astype(jnp.bfloat16), qm.param_avals(cfg))
    with tempfile.TemporaryDirectory() as d:
      chunks = mp.qwen_to_chunks(as_maxtext_qwen(port, 1), p.layer_split, cycle, 1)
      dsp.write_chunk_files(d, [mp.nest(t) for t in chunks])
      got = p.chunk_trees(p.params_from_chunks(p.chunk_trees_from(d)))
      want = [mp.nest(t) for t in chunks]
      self.assertEqual(jax.tree.structure(got), jax.tree.structure(want))
      for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(want)):
        self.assertEqual(a.dtype, jnp.bfloat16)
        np.testing.assert_array_equal(np.asarray(a).view(np.uint16), np.asarray(b).view(np.uint16))
    flat = as_maxtext_qwen(port, 1)
    # an extra norm in every linear layer: [emb, NB, cycle - 1], the local layers' axis after the block axis
    flat[("decoder", "layers", "local_layers", "extra_norm", "scale")] = np.zeros(
        (cfg.emb_dim, cfg.num_blocks, cycle - 1), jnp.bfloat16
    )
    chunks = mp.qwen_to_chunks(flat, p.layer_split, cfg.inhomogeneous_layer_cycle_interval, 1)
    self.assertIn("layers/00/extra_norm/scale", chunks[0])
    with self.assertRaisesRegex(
        ValueError,
        r"chunk 0: the weights are not this model's: missing \[\], unexpected "
        r"\[\"\['layers'\]\['00'\]\['extra_norm'\]\['scale'\]\"\]",
    ):
      p.params_from_chunks([mp.nest(t) for t in chunks])

  def test_nest(self):
    self.assertEqual(mp.nest({"a/b/c": 1, "a/d": 2, "e": 3}), {"a": {"b": {"c": 1}, "d": 2}, "e": 3})


class Compare(unittest.TestCase):

  def test_tolerances(self):
    mt = {
        0: {"learning/loss": 10.0, "learning/raw_grad_norm": 1.0},
        1: {"learning/loss": 9.0, "learning/raw_grad_norm": 1.0078125},
    }
    lines, fails = mp.compare(mt, [10.00001, 9.0005], [1.0004, 1.007], [1.0, 1.0078125], 1e-4, 1e-5, 8e-3)
    self.assertEqual(fails, [])
    self.assertEqual(len(lines), 4)
    self.assertEqual(lines[-1], "largest loss rel diff 5.56e-05 (step 1)")
    _, fails = mp.compare(mt, [10.001, 9.0], [1.0, 1.0], [1.0, 1.0], 1e-4, 1e-5, 8e-3)
    self.assertEqual(len(fails), 1)
    self.assertIn("step 0: loss", fails[0])
    _, fails = mp.compare(mt, [10.0, 9.0, 8.0], [1.0, 1.1, 1.0], None, 1e-4, 1e-5, 8e-3)
    self.assertEqual([f.split(":")[0] for f in fails], ["step 1", "step 2"])

  def test_gnorm_precision_follows_maxtexts(self):
    """A MaxText norm off the bf16 grid (fp32 gradients) is held against ds_ep's fp32 norm, not its bf16 rounding:
    1.0003 vs 1.00031 passes a 1e-4 tolerance its bf16 rounding (1.0) would fail; a bf16 one against the rounding."""
    mt = {0: {"learning/loss": 10.0, "learning/raw_grad_norm": 1.0003}}
    _, fails = mp.compare(mt, [10.0], [1.00031], [1.0], 1e-4, 1e-5, 1e-4)
    self.assertEqual(fails, [])
    mt = {0: {"learning/loss": 10.0, "learning/raw_grad_norm": 1.0078125}}
    _, fails = mp.compare(mt, [10.0], [1.0071], [1.0078125], 1e-4, 1e-5, 1e-4)
    self.assertEqual(fails, [])
    _, fails = mp.compare(mt, [10.0], [1.0071], [1.0], 1e-4, 1e-5, 1e-4)
    self.assertIn("gnorm rel diff", fails[0])

  def test_gnorm_precision_is_the_runs(self):
    """One run, one precision: an fp32 norm that falls on the bf16 grid (1.25) next to one off it is still held
    against ds_ep's fp32 norm -- 3.2e-4 off here, which its bf16 rounding (1.25) would hide."""
    mt = {
        0: {"learning/loss": 10.0, "learning/raw_grad_norm": 1.0003},
        1: {"learning/loss": 9.0, "learning/raw_grad_norm": 1.25},
    }
    _, fails = mp.compare(mt, [10.0, 9.0], [1.00031, 1.2504], [1.0, 1.25], 1e-4, 1e-5, 1e-4)
    self.assertEqual([f.split(":")[0] for f in fails], ["step 1"])
    self.assertIn("gnorm rel diff 3.20e-04", fails[0])

  def test_nan_and_uncovered_steps_fail(self):
    mt = {
        0: {"learning/loss": 10.0, "learning/raw_grad_norm": 1.0},
        1: {"learning/loss": 9.0, "learning/raw_grad_norm": 1.0},
    }
    _, fails = mp.compare(mt, [float("nan"), 9.0], [1.0, 1.0], None, 1e-4, 1e-5, 8e-3)
    self.assertEqual(len(fails), 1)
    self.assertIn("not finite", fails[0])
    _, fails = mp.compare(mt, [10.0], [1.0], None, 1e-4, 1e-5, 8e-3)  # MaxText ran a step more
    self.assertEqual(len(fails), 1)
    self.assertIn("not in the ds_ep run", fails[0])

  def test_missing_values_and_zeros_fail(self):
    """A MaxText step without raw_grad_norm (optimizer_memory_host_offload) fails, a zero MaxText value is
    no division by zero, and a ds_ep JSON without warmup_gnorms stops with a message."""
    mt = {
        0: {"learning/loss": 10.0},
        1: {"learning/loss": 9.0, "learning/raw_grad_norm": 0.0},
        2: {"learning/loss": 0.0, "learning/raw_grad_norm": 1.0},
    }
    _, fails = mp.compare(mt, [10.0, 9.0, 0.0], [1.0, 0.5, 1.0], None, 1e-4, 1e-5, 8e-3)
    self.assertEqual([f.split(":")[0] for f in fails], ["step 0", "step 1"])
    self.assertIn("no learning/raw_grad_norm", fails[0])
    self.assertIn("gnorm rel diff", fails[1])
    with tempfile.TemporaryDirectory() as d:
      with open(f"{d}/d.json", "w") as f:
        json.dump({"warmup_losses": [10.5]}, f)
      with self.assertRaisesRegex(SystemExit, "no warmup_gnorms"):
        mp.read_ds_ep(f"{d}/d.json")

  def test_minus_aux(self):
    """A ds_ep run with a load-balance loss (its run JSON's lb_loss_weight): MaxText's learning/loss less its
    learning/moe_lb_loss, which must be there."""
    mt = {0: {"learning/loss": 10.5, "learning/moe_lb_loss": 0.5, "learning/raw_grad_norm": 1.0}}
    self.assertEqual(mp.compare(mt, [10.0], [1.0], None, 1e-4, 1e-5, 8e-3, minus_aux=True)[1], [])
    self.assertIn("step 0: loss", mp.compare(mt, [10.0], [1.0], None, 1e-4, 1e-5, 8e-3)[1][0])
    _, fails = mp.compare(
        {0: {"learning/loss": 10.0, "learning/raw_grad_norm": 1.0}}, [10.0], [1.0], None, 1e-4, 1e-5, 8e-3, minus_aux=True
    )
    self.assertIn("no learning/moe_lb_loss", fails[0])
    with tempfile.TemporaryDirectory() as d:
      for w in (0.0, 0.01):
        with open(f"{d}/d.json", "w") as f:
          json.dump({"config": {"lb_loss_weight": w}, "warmup_losses": [1.0], "warmup_gnorms": [1.0]}, f)
        self.assertEqual(mp.read_ds_ep(f"{d}/d.json")[3], w > 0)

  def test_norm_precision_follows_maxtexts(self):
    """A MaxText norm that is a bf16 value (bf16 weights or gradients) is held to ds_ep's norm rounded to bf16 at
    gnorm_rtol (bf16 ulps), an fp32 one to ds_ep's fp32 norm at gnorm32_rtol: so the 0.053% offset of MaxText's
    base-2 tokamax softmax (the DeepSeek-V3 runs: 0.1118230 vs 0.1117635), far inside a bf16 ulp, fails."""
    b = 0.11181640625  # bf16(0.111763)
    bf16 = {0: {"learning/loss": 1.0, "learning/raw_grad_norm": b}}
    self.assertEqual(mp.compare(bf16, [1.0], [0.111763], [b], 1e-4, 1e-5, 1e-6)[1], [])
    fp32 = {0: {"learning/loss": 1.0, "learning/raw_grad_norm": 0.1117631}}
    self.assertEqual(mp.compare(fp32, [1.0], [0.111763], [b], 1e-4, 1e-5, 8e-3, gnorm32_rtol=1e-6)[1], [])
    self.assertIn("fp32 gnorm", mp.compare(fp32, [1.0], [0.11177], [b], 1e-4, 1e-5, 8e-3, gnorm32_rtol=1e-6)[1][0])
    biased = {0: {"learning/loss": 13.4930911, "learning/raw_grad_norm": 0.111823}}
    fails = mp.compare(biased, [13.4930906], [0.1117635], [b], 1e-4, 1e-5, 8e-3)[1]  # the defaults
    self.assertEqual(len(fails), 1)
    self.assertIn("fp32 gnorm rel diff 5.3", fails[0])

  def test_fp32_norm_after_step_0_and_the_precision_note(self):
    """After step 0 an fp32 norm is held to gnorm32_later_rtol (the trajectories may part), and a ds_ep run
    reduce-scattered in another precision than MaxText's gradients gets a note, not a failure."""
    rec = lambda gn: {"learning/loss": 1.0, "learning/raw_grad_norm": gn}
    mt = {0: rec(0.1117631), 1: rec(0.1117631)}
    lines, fails = mp.compare(mt, [1.0, 1.0], [0.1117631, 0.1117631 * (1 - 5e-4)], None, 1e-4, 1e-5, 8e-3)
    self.assertEqual(fails, [])
    _, fails = mp.compare(mt, [1.0, 1.0], [0.1117631 * (1 - 5e-4), 0.1117631], None, 1e-4, 1e-5, 8e-3)
    self.assertEqual([f.split(":")[0] for f in fails], ["step 0"])
    self.assertFalse([l for l in lines if "note" in l])
    lines, fails = mp.compare(mt, [1.0, 1.0], [0.1117631] * 2, None, 1e-4, 1e-5, 8e-3, rs_dtype="bfloat16")
    self.assertEqual((fails, len([l for l in lines if "note" in l])), ([], 1))
    self.assertEqual(
        len(
            [
                l
                for l in mp.compare(mt, [1.0, 1.0], [0.1117631] * 2, None, 1e-4, 1e-5, 8e-3, rs_dtype="float32")[0]
                if "note" in l
            ]
        ),
        0,
    )
    # --grad-dtype float32 after a bf16 reduce-scatter takes AdamW in MaxText's grad_dtype=float32 precision
    notes = lambda mt, **kw: [
        l
        for l in mp.compare(
            mt, [1.0, 1.0], [0.1117631] * 2, [0.11181640625] * 2, 1e-4, 1e-5, 8e-3, rs_dtype="bfloat16", **kw
        )[0]
        if "note" in l
    ]
    self.assertEqual(notes(mt, grad_dtype="float32"), [])
    self.assertEqual(len(notes(mt, grad_dtype="auto")), 1)
    bf16 = {t: rec(0.11181640625) for t in (0, 1)}
    self.assertIn("took AdamW in float32 (reduce-scattered in bfloat16)", notes(bf16, grad_dtype="float32")[0])
    self.assertEqual(notes(bf16, grad_dtype=None), [])

  def test_norm_precision_is_the_runs(self):
    """Whether MaxText's norms are bf16 or fp32 values is decided over the run: an fp32 run one of whose norms
    lands on a bf16 value (one in 2^16) keeps the fp32 bound at that step -- at step 0 the tight one, not a bf16
    ulp -- and a bf16 run keeps the bf16 comparison throughout; after step 0 an fp32 norm takes gnorm_rtol unless
    gnorm32_later_rtol is given."""
    rec = lambda gn: {"learning/loss": 1.0, "learning/raw_grad_norm": gn}
    on_grid = 0.11181640625  # a bf16 value
    ds = [0.1123, 0.1117631]  # step 0 4.3e-3 off: inside one bf16 ulp
    bf = [float(jnp.bfloat16(g)) for g in ds]
    _, fails = mp.compare({0: rec(on_grid), 1: rec(0.1117631)}, [1.0, 1.0], ds, bf, 1e-4, 1e-5, 8e-3)
    self.assertEqual([f.split(":")[0] for f in fails], ["step 0"])
    self.assertIn("fp32 gnorm rel diff 4.3", fails[0])
    self.assertEqual(
        mp.compare({t: rec(on_grid) for t in (0, 1)}, [1.0, 1.0], ds[:1] * 2, bf[:1] * 2, 1e-4, 1e-5, 8e-3)[1], []
    )
    later = {0: rec(0.1117631), 1: rec(0.1117631)}  # step 1 3e-3 off: inside gnorm_rtol, not 1e-3
    gs = [0.1117631, 0.1117631 * (1 + 3e-3)]
    self.assertEqual(mp.compare(later, [1.0, 1.0], gs, None, 1e-4, 1e-5, 8e-3)[1], [])
    self.assertEqual(
        [f.split(":")[0] for f in mp.compare(later, [1.0, 1.0], gs, None, 1e-4, 1e-5, 8e-3, gnorm32_later_rtol=1e-3)[1]],
        ["step 1"],
    )

  def test_adam_math_note(self):
    """A ds_ep run with fp32 AdamW arithmetic (--adam-math fp32) gets a note: MaxText runs optax's, whose bf16 mu
    takes a bf16 b1; --adam-math optax none."""
    mt = {0: {"learning/loss": 1.0, "learning/raw_grad_norm": 0.1117631}}
    notes = lambda am: [
        l for l in mp.compare(mt, [1.0], [0.1117631], None, 1e-4, 1e-5, 8e-3, adam_math=am)[0] if "--adam-math" in l
    ]
    self.assertEqual((len(notes("fp32")), len(notes("optax")), len(notes(None))), (1, 0, 0))

  def test_control_spread(self):
    """A control run (MaxText computing the same thing another way) bounds every step's loss difference at
    control_factor x its largest MaxText difference -- the DeepSeek-V3 runs: ds_ep against MaxText's base-2
    softmax parts by 5.3e-5 at step 2, the control (another mesh) by at most 5.7e-6."""
    rec = lambda losses: {t: {"learning/loss": l, "learning/raw_grad_norm": 0.11181640625} for t, l in enumerate(losses)}
    mt, ctl = rec((13.4930911, 13.4782204, 13.4382453)), rec((13.4930921, 13.4782261, 13.4382482))
    g, b = [0.111763] * 3, [0.11181640625] * 3
    _, fails = mp.compare(mt, [13.4930906, 13.4782286, 13.4382982], g, b, 1e-4, 1e-5, 8e-3, control=ctl)
    self.assertEqual([f.split(":")[0] for f in fails], ["step 2"])
    self.assertIn("control's spread 5.7", fails[0])
    self.assertEqual(mp.compare(mt, [13.4930907, 13.4782210, 13.4382480], g, b, 1e-4, 1e-5, 8e-3, control=ctl)[1], [])
    elsewhere = {9: {"learning/loss": 1.0, "learning/raw_grad_norm": 0.11181640625}}
    self.assertIn("no step", mp.compare(mt, [13.4930911], g, b, 1e-4, 1e-5, 8e-3, control=elsewhere)[1][0])

  def test_reads_both_files(self):
    with tempfile.TemporaryDirectory() as d:
      with open(f"{d}/m.txt", "w") as f:
        for t, l in enumerate((10.5, 10.25)):
          f.write(json.dumps({"learning/loss": l, "learning/raw_grad_norm": 1.25, "step": float(t)}) + "\n")
      with open(f"{d}/d.json", "w") as f:
        json.dump(
            {"warmup_losses": [10.5, 10.25], "warmup_gnorms": [1.2501, 1.2499], "warmup_gnorms_bf16": [1.25, 1.25]}, f
        )
      self.assertEqual(mp.main(["compare", f"{d}/m.txt", f"{d}/d.json"]), 0)
      mt = mp.read_maxtext_metrics(f"{d}/m.txt")
      self.assertEqual(sorted(mt), [0, 1])
      # --control: a spread of 1e-4 lets the 0 differences pass; 0.0 (a control identical to MaxText) refuses
      for shift, rc in ((1e-4, 0), (0.0, 1)):
        with open(f"{d}/c.txt", "w") as f:
          for t, l in enumerate((10.5, 10.25)):
            f.write(json.dumps({"learning/loss": l + shift, "learning/raw_grad_norm": 1.25, "step": float(t)}) + "\n")
        self.assertEqual(
            mp.main(["compare", f"{d}/m.txt", f"{d}/d.json", "--control", f"{d}/c.txt", "--control-factor", "1"]), rc
        )


class InitChecks(unittest.TestCase):

  def test_override_values(self):
    """`init`'s key=value overrides: an int, a float, a bool, a list / tuple / dict literal (a Qwen3.5 model's
    mrope_section=[3,3,2], which pyconfig refuses as a string), else the string."""
    for v, want in (
        ("8", 8),
        ("1e-5", 1e-5),
        ("true", True),
        ("False", False),
        ("[3,3,2]", [3, 3, 2]),
        ("(1, 2)", (1, 2)),
        ("{'a': 1}", {"a": 1}),
        ("bfloat16", "bfloat16"),
        ("[silu]", "[silu]"),
        ("", ""),
    ):
      with self.subTest(v=v):
        got = mp._value(v)
        self.assertEqual((type(got), got), (type(want), want))


@pytest.mark.scheduled_only
class AgainstMaxText(unittest.TestCase):
  """What a MaxText bump could break unnoticed until the slice: a renamed parameter, another synthetic batch."""

  def test_init_chunk_files_fit_the_pipeline(self):
    """`maxtext_parity init` (MaxText's own init and names) -> OUT/ds_chunks: every chunk has the leaves,
    shapes and dtypes of the matching ds_ep pipeline's own init, and params_from_chunks takes them."""
    d = self.enterContext(tempfile.TemporaryDirectory())
    check(self, *run(["-m", "maxtext.experimental.mpmd.jax.maxtext_parity", "init", d, *MAXTEXT_TINY]))
    self.assertTrue(os.listdir(f"{d}/maxtext_params"))
    p = tiny_pipeline()
    got = p.chunk_trees_from(f"{d}/ds_chunks")
    self.assertEqual([leaves(t) for t in got], [leaves(t) for t in p.chunk_trees(p.init(jax.random.key(0))[0])])
    p.params_from_chunks(got)

  def test_exact_init_chunk_files_fit_the_pipeline(self):
    """`maxtext_parity init --exact --layer-split` (a scanned MaxText on one device: dense layers, routed
    biases, MTP) -> OUT/ds_chunks: every chunk has the leaves, shapes and dtypes of the matching exact
    pipeline's own init, and params_from_chunks takes them."""
    d = self.enterContext(tempfile.TemporaryDirectory())
    p = exact_tiny_pipeline()
    check(
        self,
        *run(
            [
                "-m",
                "maxtext.experimental.mpmd.jax.maxtext_parity",
                "init",
                "--exact",
                "--layer-split",
                ",".join(map(str, p.layer_split)),
                d,
                *MAXTEXT_EXACT_TINY,
            ],
            devs=1,
        ),
    )
    got = p.chunk_trees_from(f"{d}/ds_chunks")
    self.assertEqual([leaves(t) for t in got], [leaves(t) for t in p.chunk_trees(p.init(jax.random.key(0))[0])])
    p.params_from_chunks(got)

  def test_exact_trains_as_maxtext_fp32(self):
    """End to end in fp32 (weights and compute; the pure-JAX kernels): `init --exact`, then MaxText's trainer (a
    scanned decoder on one device) and ds_ep_pipeline (2 stages x 2 chunks: dense, mixed and MoE chunks, the
    routed-bias update, group-limited routing, the aux loss, MTP) from its weights and MaxText's synthetic batch,
    three steps at lr 1e-3 (the loss falls by over 40%): the losses (MaxText's less its load-balance term) and
    the gradient norms agree to fp32 rounding.  The step-0 losses, of one set of weights, within 3e-7: a few fp32
    ulps of the loss (one is 6.4e-8 of it), for the two programs round differently, by amounts that move with jax
    and MaxText -- 6e-9 to 1.2e-7 apart on jax 0.11.1 and 0.12.0.1 with MaxText ce608b918 and 4b5b24c92
    (2026-11-26).  The later losses within 1e-6, and the norms within 1e-5 (measured at most 1.2e-7 and 8.2e-7)."""
    import maxtext

    d = self.enterContext(tempfile.TemporaryDirectory())
    model = MAXTEXT_EXACT_TINY + ["weight_dtype=float32"]
    check(
        self,
        *run(
            [
                "-m",
                "maxtext.experimental.mpmd.jax.maxtext_parity",
                "init",
                "--exact",
                "--layer-split",
                "1,2,1,1",
                f"{d}/w",
                *model,
            ],
            devs=1,
        ),
    )
    check(
        self,
        *run(
            [
                "-c",
                MAXTEXT_BATCH,
                f"{d}/batch.npz",
                "run_name=maxtext_parity_test",
                f"base_output_directory={d}/b",
                "dataset_type=synthetic",
                "per_device_batch_size=4",
                "max_target_length=256",
                "vocab_size=512",
                "enable_checkpointing=false",
                "skip_jax_distributed_system=true",
            ],
            devs=1,
        ),
    )
    batch = np.load(f"{d}/batch.npz")
    np.save(f"{d}/tokens.npy", np.concatenate([batch["inputs"], batch["targets"][:, -1:]], axis=1))
    base = os.path.join(os.path.dirname(maxtext.__file__), "configs", "base.yml")
    check(
        self,
        *run(
            [
                "-m",
                "maxtext.trainers.pre_train.train",
                base,
                "run_name=maxtext_parity_test",
                f"base_output_directory={d}/mt",
                "model_name=deepseek3-671b",
                "override_model_config=true",
                *model,
                "routed_bias=true",
                "mtp_num_layers=1",
                "mtp_loss_scaling_factor=0.1",
                "mtp_reuse_input_embedding=true",
                "dtype=float32",
                "mu_dtype=bfloat16",
                "per_device_batch_size=4",
                "dataset_type=synthetic",
                "sparse_matmul=true",
                "use_ring_of_experts=false",
                "use_ragged_sort=false",
                "use_random_routing=false",
                "float32_weight_sum=true",
                "learning_rate=1e-3",
                "warmup_steps_fraction=0",
                "learning_rate_final_fraction=1.0",
                "adam_b1=0.9",
                "adam_b2=0.95",
                "adam_eps=1e-8",
                "adam_weight_decay=0.1",
                "gradient_clipping_threshold=1.0",
                f"load_parameters_path={d}/w/maxtext_params",
                "enable_checkpointing=true",
                "save_checkpoint_on_start=false",
                "save_checkpoint_on_completion=false",
                "async_checkpointing=false",
                f"metrics_file={d}/m.txt",
                "log_period=1",
                "steps=3",
                "scan_layers=true",
                "remat_policy=full",
                "skip_jax_distributed_system=true",
                "enable_goodput_recording=false",
            ],
            devs=1,
            timeout=1200,
        ),
    )
    check(
        self,
        *run(
            [
                "-m",
                "maxtext.experimental.mpmd.jax.ds_ep_pipeline",
                *EXACT_TINY,
                "--attn",
                "dot_product",
                "--dtype",
                "float32",
                "--weight-dtype",
                "float32",
                "--rs-dtype",
                "float32",
                "--adam-math",
                "optax",
                "--lr",
                "1e-3",
                "--init-from",
                f"{d}/w/ds_chunks",
                "--data",
                f"npy:{d}/tokens.npy",
                "--warmup",
                "3",
                "--steps",
                "1",
                "--json",
                f"{d}/d.json",
            ],
            devs=2,
            timeout=1200,
        ),
    )
    losses, gnorms, bf16, aux = mp.read_ds_ep(f"{d}/d.json")
    self.assertTrue(aux)
    lines, fails = mp.compare(
        mp.read_maxtext_metrics(f"{d}/m.txt"),
        losses,
        gnorms,
        bf16,
        1e-6,
        3e-7,  # step 0: a few fp32 ulps of the loss (the docstring)
        8e-3,
        minus_aux=True,
        gnorm32_rtol=1e-5,
        gnorm32_later_rtol=1e-5,
    )
    self.assertEqual(fails, [], "\n".join(lines))
    self.assertLess(losses[-1], 0.7 * losses[0])

  def test_qwen_init_chunk_files_fit_the_pipeline(self):
    """`maxtext_parity init --model qwen3.8-2.4t-a95b --layer-split` (MaxText's qwen3_5, scanned blocks, on one
    device) -> OUT/ds_chunks: every chunk has the leaves, shapes and dtypes of the matching Qwen pipeline's own
    init (a subtree a layer, the mixer as gdn or attn), and params_from_chunks takes them."""
    d = self.enterContext(tempfile.TemporaryDirectory())
    p = qwen_tiny_pipeline()
    check(
        self,
        *run(
            [
                "-m",
                "maxtext.experimental.mpmd.jax.maxtext_parity",
                "init",
                "--model",
                "qwen3.8-2.4t-a95b",
                "--layer-split",
                "1,3,3,1",
                d,
                *MAXTEXT_QWEN_TINY,
            ],
            devs=1,
        ),
    )
    got = p.chunk_trees_from(f"{d}/ds_chunks")
    self.assertEqual([leaves(t) for t in got], [leaves(t) for t in p.chunk_trees(p.init(jax.random.key(0))[0])])
    p.params_from_chunks(got)

  def test_qwen35_init_chunk_files_fit_the_pipeline(self):
    """`init --model qwen3.5-397b-a17b` cut down, its mRoPE sections too (mrope_section=[3,3,2], a list): the chunk
    files fit the Qwen pipeline (the Qwen3.8 one: a rotary embedding has no parameter)."""
    d = self.enterContext(tempfile.TemporaryDirectory())
    init = [
        "-m",
        "maxtext.experimental.mpmd.jax.maxtext_parity",
        "init",
        "--model",
        "qwen3.5-397b-a17b",
        "--layer-split",
        "1,3,3,1",
    ]
    model = MAXTEXT_QWEN_TINY + ["mrope_section=[3,3,2]"]
    check(self, *run(init + [f"{d}/b16", *model], devs=1))
    p = qwen_tiny_pipeline()
    got = p.chunk_trees_from(f"{d}/b16/ds_chunks")
    self.assertEqual([leaves(t) for t in got], [leaves(t) for t in p.chunk_trees(p.init(jax.random.key(0))[0])])
    p.params_from_chunks(got)

  def test_qwen_init_refuses_a_vision_encoder(self):
    """`init --model qwen3.5-397b-a17b use_multimodal=true` (its vision encoder cut down): ds_ep's Qwen pipeline has
    no place for the encoder's leaves, so init names one and writes nothing: it cuts the chunks before it saves the
    checkpoint."""
    d = self.enterContext(tempfile.TemporaryDirectory())
    vision = [
        "use_multimodal=true",
        "hidden_size_for_vit=64",
        "intermediate_size_for_vit=128",
        "num_attention_heads_for_vit=2",
        "num_hidden_layers_for_vit=1",
        "out_hidden_size_for_vit=256",
    ]
    check(
        self,
        *run(
            [
                "-m",
                "maxtext.experimental.mpmd.jax.maxtext_parity",
                "init",
                "--model",
                "qwen3.5-397b-a17b",
                "--layer-split",
                "1,3,3,1",
                f"{d}/w",
                *MAXTEXT_QWEN_TINY,
                "mrope_section=[3,3,2]",
                *vision,
            ],
            devs=1,
        ),
        refusal=("init --model qwen3.5-397b-a17b: MaxText parameter vision_encoder/", "has no place in a ds_ep chunk"),
    )
    self.assertEqual(os.listdir(d), [])

  def test_qwen_trains_as_maxtext_fp32(self):
    """End to end in fp32 (weights and compute; the pure-JAX kernels) on Qwen3.8-2.4T's layers at tiny widths:
    `init --model qwen3.8-2.4t-a95b`, then MaxText's trainer (its qwen3_5 decoder, scanned blocks, the partial RoPE
    of the yml) on one device, MaxText again on two (FSDP 2: the control), and ds_ep_pipeline from its weights and
    MaxText's synthetic batch -- the default plan (breadth), --batch-split, and interleaved 1F1B's F&B pairs with
    batch split -- five steps at lr 1e-5, top-8 of 8, no clip.  MaxText runs with packing=false: the pipeline's
    Gated DeltaNet takes no segment ids (`ds_ep.qwen_mixer_part`), which is MaxText's unpacked GDN; under packing
    (MaxText's default, read by its GDN from fb42453e1 on) its trainer runs the packed GDN on the synthetic batch's
    all-one segment ids -- the same math in other fp32 roundings -- and its step-0 norm then comes 1e-4 from the
    pipeline's (4b5b24c92, on 2026-11-25: 8x MaxText's own control there, all three arms alike).

    Every bound is the larger of a floor and 3x the control's spread, MaxText's own 1-device vs 2-device difference
    in the same quantity, measured in the same run: every step's loss agrees within max(1e-6 rel, 3x the control's
    largest loss difference), the gradient norm within max(5e-5, 3x the control's) at step 0 (one set of weights)
    and within max(3e-3, 3x the control's largest) after it.  A control may loosen a bound to 10x its floor (the
    loss, step 0) or to 5e-2 (after step 0), no further: one farther from MaxText fails the test as a broken control
    rather than loosen every bound with it.  The control's spread depends on the MaxText revision
    and the environment, and so do the arms', alike: 2.2e-7 at step 0 and at most 4.1e-4 after it in the 2026-11-04
    run (the arms up to 6.5e-4), 1.0e-6 and 1.9e-4 on another host on 2026-11-09
    (the arms up to 2.9e-4), and against 4b5b24c92 on 2026-11-25 4.4e-5 at step 0 (the arms 4.0e-5 to 4.4e-5) and
    9.8e-3 at step 3 (the arms up to 1.02e-2), past the old fixed 3e-3; the test prints the control's and the arms'.
    The floors are of the pipeline's own program structure, which the control does not have: the Gated DeltaNet's
    fp32 gradients are not bit-reproducible across two programs of one model -- on these weights the unsplit
    pipeline is up to 7e-5 a leaf from its own monolith (ds_ep_test.reference), the batch-split one likewise, and
    the port up to 2.7e-5 from itself run in two halves (qwen35_maxtext_parity_test's floor is of the same kind) --
    so the step-0 norms came 7e-7 (breadth) to 1.7e-5 (batch split) from MaxText's where its control was 2.2e-7;
    and AdamW's first step, g / (|g| + eps), turns the last bits of a leaf's tiniest gradients into weights 2 lr
    apart (3-5 sign flips a million elements, port against MaxText): at lr 1e-3 that grows to a 1e-2 norm gap by
    step 2, against MaxText's own control too."""
    import maxtext

    d = self.enterContext(tempfile.TemporaryDirectory())
    model = [x.replace("num_experts_per_tok=2", "num_experts_per_tok=8") for x in MAXTEXT_QWEN_TINY]
    model += ["weight_dtype=float32"]
    tiny = [x if x != "2" or QWEN_TINY[k - 1] != "--topk" else "8" for k, x in enumerate(QWEN_TINY)]
    tiny[tiny.index("--gbs") + 1] = "8"
    check(
        self,
        *run(
            [
                "-m",
                "maxtext.experimental.mpmd.jax.maxtext_parity",
                "init",
                "--model",
                "qwen3.8-2.4t-a95b",
                "--layer-split",
                "1,3,3,1",
                f"{d}/w",
                *model,
            ],
            devs=1,
        ),
    )
    check(
        self,
        *run(
            [
                "-c",
                MAXTEXT_BATCH,
                f"{d}/batch.npz",
                "run_name=maxtext_parity_test",
                f"base_output_directory={d}/b",
                "dataset_type=synthetic",
                "per_device_batch_size=8",
                "max_target_length=256",
                "vocab_size=512",
                "enable_checkpointing=false",
                "skip_jax_distributed_system=true",
            ],
            devs=1,
        ),
    )
    batch = np.load(f"{d}/batch.npz")
    np.save(f"{d}/tokens.npy", np.concatenate([batch["inputs"], batch["targets"][:, -1:]], axis=1))
    base = os.path.join(os.path.dirname(maxtext.__file__), "configs", "base.yml")
    train = [
        "-m",
        "maxtext.trainers.pre_train.train",
        base,
        "run_name=maxtext_parity_test",
        "model_name=qwen3.8-2.4t-a95b",
        "override_model_config=true",
        *model,
        "dtype=float32",
        "matmul_precision=highest",
        "mu_dtype=bfloat16",
        "dataset_type=synthetic",
        "sparse_matmul=true",
        "use_ring_of_experts=false",
        "use_ragged_sort=false",
        "use_random_routing=false",
        "float32_weight_sum=true",
        "packing=false",  # the pipeline's GDN path (the docstring)
        "learning_rate=1e-5",
        "warmup_steps_fraction=0",
        "learning_rate_final_fraction=1.0",
        "adam_b1=0.9",
        "adam_b2=0.95",
        "adam_eps=1e-8",
        "adam_weight_decay=0.1",
        "gradient_clipping_threshold=0",
        f"load_parameters_path={d}/w/maxtext_params",
        "enable_checkpointing=true",
        "save_checkpoint_on_start=false",
        "save_checkpoint_on_completion=false",
        "async_checkpointing=false",
        "log_period=1",
        "steps=5",
        "scan_layers=true",
        "remat_policy=full",
        "skip_jax_distributed_system=true",
        "enable_goodput_recording=false",
    ]
    check(
        self,
        *run(
            train + [f"base_output_directory={d}/mt", f"metrics_file={d}/m.txt", "per_device_batch_size=8"],
            devs=1,
            timeout=1800,
        ),
    )
    check(
        self,
        *run(
            train
            + [
                f"base_output_directory={d}/mt2",
                f"metrics_file={d}/m2.txt",
                "per_device_batch_size=4",
                "ici_fsdp_parallelism=2",
            ],
            devs=2,
            timeout=1800,
        ),
    )
    mt, control = mp.read_maxtext_metrics(f"{d}/m.txt"), mp.read_maxtext_metrics(f"{d}/m2.txt")
    rel = lambda k, t: abs(float(control[t][k]) - float(mt[t][k])) / abs(float(mt[t][k]))
    loss_spread = max(rel("learning/loss", t) for t in mt)
    gnorm_spread = {t: rel("learning/raw_grad_norm", t) for t in mt}

    def bound(what, floor, spread, ceiling):
      """max(floor, 3x the control's spread), the control held to a spread of ceiling / 3 (the docstring)."""
      self.assertLessEqual(
          3 * spread,
          ceiling,
          f"{what}: the control (MaxText on 2 devices) is {spread:.2e} from MaxText, past the {ceiling / 3:.2e} a "
          "control may be (a broken control)",
      )
      return max(floor, 3 * spread)

    loss_rtol = bound("loss", 1e-6, loss_spread, 1e-5)
    gnorm0_rtol = bound("gnorm at step 0", 5e-5, gnorm_spread[0], 5e-4)
    gnorm_later_rtol = bound("gnorm after step 0", 3e-3, max(v for t, v in gnorm_spread.items() if t > 0), 5e-2)
    print(
        "control (MaxText on 2 devices) against MaxText: loss rel diff "
        + " ".join(f'{rel("learning/loss", t):.2e}' for t in sorted(mt))
        + "; gnorm rel diff "
        + " ".join(f"{gnorm_spread[t]:.2e}" for t in sorted(mt))
        + f" -> bounds: loss {loss_rtol:.2e}, gnorm {gnorm0_rtol:.2e} at step 0 and {gnorm_later_rtol:.2e} after"
    )
    for what, extra in (
        ("breadth", []),
        ("batch split", ["--batch-split"]),
        (
            "interleaved pairs + batch split",
            ["--batch-split", "--schedule", "interleaved", "--pair-stages", "all", "--mb", "4"],
        ),
    ):
      with self.subTest(what):
        check(
            self,
            *run(
                [
                    "-m",
                    "maxtext.experimental.mpmd.jax.ds_ep_pipeline",
                    *tiny,
                    *extra,
                    "--dtype",
                    "float32",
                    "--weight-dtype",
                    "float32",
                    "--rs-dtype",
                    "float32",
                    "--adam-math",
                    "optax",
                    "--lr",
                    "1e-5",
                    "--grad-clip",
                    "0",
                    "--init-from",
                    f"{d}/w/ds_chunks",
                    "--data",
                    f"npy:{d}/tokens.npy",
                    "--warmup",
                    "5",
                    "--steps",
                    "1",
                    "--json",
                    f"{d}/d.json",
                ],
                devs=2,
                timeout=1800,
            ),
        )
        losses, gnorms, bf16, aux = mp.read_ds_ep(f"{d}/d.json")
        self.assertFalse(aux)
        lines, fails = mp.compare(
            mt,
            losses,
            gnorms,
            bf16,
            loss_rtol,
            loss_rtol,
            gnorm_later_rtol,
            gnorm32_rtol=gnorm0_rtol,
            gnorm32_later_rtol=gnorm_later_rtol,
            control=control,
        )
        print(what + "\n" + "\n".join(lines))
        self.assertEqual(fails, [], "\n".join(lines))
        self.assertLess(losses[-1], losses[0] - 0.1)

  def test_deepseek_from_maxtext(self):
    """maxtext.experimental.mpmd.jax.deepseek.config.from_maxtext reads only keys this MaxText has, and carries
    deepseek3-671b's dense layers and the MLPerf recipe's routing groups and MTP layer."""
    d = self.enterContext(tempfile.TemporaryDirectory())
    rc, out = run(
        [
            "-c",
            DEEPSEEK_FROM_MAXTEXT,
            "run_name=maxtext_parity_test",
            f"base_output_directory={d}",
            "model_name=deepseek3-671b",
            "n_routing_groups=8",
            "topk_routing_group=4",
            "mtp_num_layers=1",
            "enable_checkpointing=false",
            "skip_jax_distributed_system=true",
        ],
        devs=1,
    )
    check(self, rc, out)
    self.assertIn("carried 3 8 4 1 False", out)

  def test_dp_init_chunk_files_fit_the_pipeline(self):
    """`maxtext_parity init --layer-split` of MaxText without PP (scanned dense and MoE layers, one device), with
    a bfloat16 copy: both chunk file sets fit the dense pipeline's own init, leaves and shapes, the copy in
    bfloat16, and hold the same weights."""
    d = self.enterContext(tempfile.TemporaryDirectory())
    p = tiny_pipeline(dense=True)
    split = ",".join(str(len(ch.layers)) for ch in p.chunks)
    check(
        self,
        *run(
            [
                "-m",
                "maxtext.experimental.mpmd.jax.maxtext_parity",
                "init",
                d,
                "--layer-split",
                split,
                "--cast",
                "bfloat16",
                "weight_dtype=float32",
                *MAXTEXT_TINY_DP,
            ],
            devs=1,
        ),
    )
    for sfx in ("", "_bfloat16"):
      self.assertTrue(os.listdir(f"{d}/maxtext_params{sfx}"))
    own = p.chunk_trees(p.init(jax.random.key(0))[0])
    f32, b16 = p.chunk_trees_from(f"{d}/ds_chunks"), p.chunk_trees_from(f"{d}/ds_chunks_bfloat16")
    shapes = lambda trees: [{k: v[0] for k, v in leaves(t).items()} for t in trees]
    self.assertEqual(shapes(f32), shapes(own))
    self.assertEqual(shapes(b16), shapes(own))
    self.assertEqual({v.dtype for t in f32 for v in jax.tree.leaves(t)}, {np.dtype(np.float32)})
    for a, b in zip(jax.tree.leaves(f32), jax.tree.leaves(b16)):
      np.testing.assert_array_equal(np.asarray(a).astype(jnp.bfloat16).view(np.uint16), np.asarray(b).view(np.uint16))
    p.params_from_chunks(b16)

  def test_maxtext_tokens_are_maxtexts_batch(self):
    """Pipeline.maxtext_tokens() (--data maxtext) is the batch MaxText's trainer draws for
    dataset_type=synthetic, inputs and targets bit for bit (on CPU here: the bits depend on the backend)."""
    d = self.enterContext(tempfile.TemporaryDirectory())
    p = tiny_pipeline()
    check(
        self,
        *run(
            [
                "-c",
                MAXTEXT_BATCH,
                f"{d}/batch.npz",
                "run_name=maxtext_parity_test",
                f"base_output_directory={d}",
                "dataset_type=synthetic",
                "per_device_batch_size=2",
                f"max_target_length={p.seq}",
                f"vocab_size={p.scfg.model.vocab_size}",
                "enable_checkpointing=false",
                "skip_jax_distributed_system=true",
            ]
        ),
    )
    tokens, batch = p.maxtext_tokens(), np.load(f"{d}/batch.npz")
    self.assertEqual(batch["inputs"].shape, (p.gbs, p.seq))
    np.testing.assert_array_equal(tokens[:, :-1], batch["inputs"])
    np.testing.assert_array_equal(tokens[:, 1:], batch["targets"])


if __name__ == "__main__":
  unittest.main()
