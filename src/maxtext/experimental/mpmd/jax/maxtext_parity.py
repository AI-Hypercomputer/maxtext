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

"""Numerical parity of the MPMD pipeline against MaxText: one set of weights, one batch, both trainers.

    # 1. one MaxText init -> a MaxText params checkpoint, and the same weights as ds_ep chunk files
    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \\
        python3 -m maxtext.experimental.mpmd.jax.maxtext_parity init OUT [MaxText key=value ...]
    #    against MaxText without PP (DP over DCN): one device, the ds_ep run's --layer-split, and copies in other
    #    dtypes; --exact for DeepSeek-V3 itself (dense layers, routed biases, MTP), a split summing to every layer
    #    for dense layers spanning the first chunks
    JAX_PLATFORMS=cpu python3 -m maxtext.experimental.mpmd.jax.maxtext_parity init OUT --layer-split 1,1,... \\
        [--cast bfloat16] [--exact] ici_pipeline_parallelism=1 [MaxText key=value ...]
    # 2. MaxText's trainer (maxtext.trainers.pre_train.train): load_parameters_path=OUT/maxtext_params
    #    enable_checkpointing=true save_checkpoint_on_start=false save_checkpoint_on_completion=false
    #    (an older MaxText refuses to load parameters with checkpointing off; these flags work on any.  The
    #    two save_* flags keep it from saving -- by default it saves at step 0, and on several hosts a save into
    #    a host-local base_output_directory times out waiting for directories only process 0 creates; a gs://
    #    one fails MaxText's mkdir check, which finds no directory object there)
    #    metrics_file=M.txt log_period=1 steps=N num_experts_per_tok=K use_random_routing=false
    #    learning_rate=LR warmup_steps_fraction=0 learning_rate_final_fraction=1.0
    #    gradient_clipping_threshold=0, with its own synthetic data;
    #    ds_ep_pipeline: --init-from OUT/ds_chunks --data maxtext --routing topk --topk K --lr LR
    #    --adam-math optax --grad-clip 0 --warmup N --json D.json (the same N), and the same MoE path on both:
    #    use_ring_of_experts / use_ragged_sort / ragged_buffer_factor there, --ragged-sort /
    #    --ragged-buffer-factor here.  With fp32 weights --rs-dtype bfloat16 follows MaxText's gradients, which
    #    its bf16 reduce-scatters sum before the cast to fp32 (--adam-math optax takes each gradient in its
    #    weight's dtype).  And one softmax on both: tokamax's base-2 on both (MaxText's default and ds_ep's: see
    #    softmax below), or --attn splash (or --sa-base2-exp off) here with sa_use_base2_exp=false there
    #    DeepSeek-V3 itself: MaxText with weight_dtype=float32 grad_dtype=float32 sa_use_base2_exp=false (and its
    #    batch saved for ds_ep --data npy:), ds_ep --attn splash --weight-dtype float32 --rs-dtype bfloat16
    #    --adam-math optax (MaxText's casts: see precision)
    #    A Qwen3.5-family model (init --model): MaxText with model_name=<the model> override_model_config=true,
    #    init's key=value overrides and packing=false, ds_ep --model <the model> with the same dims.  ds_ep's Gated
    #    DeltaNet takes no segment ids (ds_ep.qwen_mixer_part), MaxText's GDN under packing=false; under its default
    #    packing MaxText's trainer runs the packed GDN on the synthetic batch's all-one segment ids, the same math
    #    in other roundings: fp32's in fp32 (its step-0 norm came 1e-4 from the pipeline's, 8x MaxText's own
    #    1-vs-2-device spread: maxtext_parity_test.test_qwen_trains_as_maxtext_fp32), bf16's in a bf16 compute
    #    dtype (with fp32 weights its conv keeps the fp32 kernel ds_ep's rounds to bf16)
    # 3. per step: loss and the pre-clip gradient norm.  With bf16 MaxText gradients the norm is a bf16 value,
    #    blind to a 5e-4 difference: prefer an fp32 MaxText norm (held to --gnorm32-rtol at step 0), and add
    #    --control M2.txt, MaxText on another mesh, whose spread bounds every step's loss difference
    python3 -m maxtext.experimental.mpmd.jax.maxtext_parity compare M.txt D.json [--control M2.txt]

What makes the two runs comparable:
  weights  MaxText initializes; `init` writes its params with MaxText's save_params_to_path (what
           load_parameters_path restores; the optimizer starts fresh, as in ds_ep's init_state) and the same
           arrays as ds_ep chunk files.  MaxText stacks a pipelined leaf [repeat, stage, ...]; ds_ep's chunk c
           is MaxText's [c // S, c % S] (circular placement: chunk c runs on stage c % S, repeat c // S), with
           DeepSeekMoeBlock_0/MoeBlock_0 renamed to moe.  Without PP (ici_pipeline_parallelism=1, DP over DCN)
           MaxText scans its layers, a leaf's layer axis at param_scan_axis: DeepSeek's first layers in
           decoder/dense_layers and the MoE layers in decoder/moe_layers, cut into chunks by --layer-split as
           ds_ep_pipeline cuts them (the MoE layers a chunk, the dense ones in chunk 0; or every layer a chunk).
           Its routed biases are MoEBiasVars, not parameters (zero at init; `init` writes them into the params
           checkpoint, where MaxText's loader looks them up), and its MTP layer shares the token embedding,
           which ds_ep's last chunk keeps a copy of.  bfloat16 goes over as raw bits; --cast writes the same
           weights in another dtype too (OUT/maxtext_params_<dtype>, OUT/ds_chunks_<dtype>).
  data     MaxText's dataset_type=synthetic is one batch, the same every step: randint(key 0) under
           threefry2x32, the same bits on every backend.  ds_ep_pipeline --data maxtext regenerates it bit for
           bit (ds_ep_pipeline.MAXTEXT_BATCH_PRNG), and logs its sha256.
  routing  MaxText draws its random routing from NNX rng streams split per (repeat, stage) and advanced
           every pipeline iteration, under unsafe_rbg: not reproducible from outside.  Compare at top-k
           routing (num_experts_per_tok=K use_random_routing=false there, --routing topk --topk K here).
           Top-k of all experts has no selection boundary; below it a last-bit difference upstream can flip
           a token's expert where two router scores tie (bf16 scores near 0.5 at init).  Measured: top-8 of
           16 moved step 0's loss by 4e-6 (top-16 of 16: 2e-7) and then tracked as closely as top-16
           (8 x TPU7x 2x2x2, 2026-10-25).  DeepSeek-V3's own routing (top-8 of 256 in 4 of 8 groups) under
           float32_gate_logits compares as it is: its fp32 scores leave the step-0 losses a few fp32 ulps apart
           (measured 6e-9 to 1.2e-7 relative, by jax and MaxText revision:
           maxtext_parity_test.test_exact_trains_as_maxtext_fp32).
  schedule A constant learning rate on both sides: learning_rate=LR warmup_steps_fraction=0
           learning_rate_final_fraction=1.0 here, --lr LR there.
  optimizer --adam-math optax: ds_ep's AdamW runs optax's chain in the leaf dtypes (bf16 with bf16 weights),
           op for op as maxtext.experimental.mpmd.jax.deepseek.train, which matches MaxText's optax.adamw bit
           for bit; ds_ep's default fp32 arithmetic differs in the last bits, and with bf16 moments that grows
           over the steps.
  norms    MaxText reduces its gradient norm in its gradients' dtype (max_utils.l2norm_pytree): bf16 with bf16
           weights or grad_dtype=bfloat16.  A run whose MaxText norms are all bf16 values is compared with
           ds_ep's fp32 norm rounded to bf16 (warmup_gnorms_bf16), any other with ds_ep's fp32 norm.
  loss     ds_ep's loss is the cross entropy plus MTP's (scaled); a load-balance loss rides its layers' backward
           only, so `compare` takes MaxText's learning/moe_lb_loss off its learning/loss.
  softmax  The attention kernels must compute one softmax.  MaxText's tokamax splash with sa_use_base2_exp=true
           (its base.yml default, and the MLPerf recipe's) scales the bf16 query by log2(e) rounded to bf16
           (1.4453125), a softmax 1.0018x sharper than its logits, as ds_ep's tokamax does by default; ds_ep's
           --attn splash, and tokamax with --sa-base2-exp off, compute the exact one.  Pair --attn splash with
           MaxText sa_use_base2_exp=False, or --attn tokamax_splash with MaxText's default.
           Mismatched, at DeepSeek-V3 dims (4 layers, lr 1e-5): the step-0 loss still agrees (the sharpening
           moves layer 0's attention output by ~0.3%, below the loss's fp32 resolution), MaxText's fp32 gradient
           norm is 0.049% above ds_ep --rs-dtype float32's (0.053% above its bf16 reduce-scatter's) at every
           step, and the loss parts from the second update (+5e-5..+1.3e-4 by step 4: under 1e-5 relative, so
           --loss-rtol cannot see it; an fp32 MaxText norm or --control does).
  precision MaxText weight_dtype=float32 grad_dtype=float32 reduce-scatters its bf16 compute copies' gradients
           (the fp32 routers' kernels' in fp32) and casts the result to fp32: pair it with ds_ep --rs-dtype
           bfloat16 --grad-dtype float32, the same casts but for the routers' kernels (ds_ep reduce-scatters them in
           --rs-dtype).  At DeepSeek-V3 dims and lr 1e-5 the loss then agrees within 1e-5 over 20 steps but at steps
           14 (2.3e-5) and 17 (4.5e-5), two of the steps MaxText's own grad_dtype switch moves most; on the steps
           that switch leaves alone it drifts by +4.3e-7 +- 2.2e-7 a step (steps 10-19: +1.05e-5 on average, 9 of
           10 positive), and the fp32 norm stays 0.5-1.2e-5 below MaxText's.  ds_ep --rs-dtype float32 (an fp32
           reduction) stays within ds_ep's own run-to-run spread (~1e-5) through step 9, then drifts one way by
           ~2e-6 a step (-1.4e-4 at step 17), its norm 3.1-3.8e-5 above MaxText's: the reduction's precision, which
           alone moves ds_ep's loss by ~2.6e-6 a step (bfloat16 against float32, the same run otherwise).
           --rs-dtype bfloat16 with grad_dtype=bfloat16 does not drift but parts at isolated steps of this batch
           (5, 10, 13, 14, 17, 19: 7.7e-5 at step 5, up to 7.9e-4 by step 17), the steps whose loss also moves
           2e-4..1.3e-3 when MaxText alone switches its grad_dtype.  Why (confirmed: at lr 1.1e-5 both moved
           where predicted): the init is all bf16, so every fp32 master starts on the bf16 grid, and AdamW's
           near-sign steps move most weights 0.992-0.996 lr an update; the bf16 copies the forward reads change
           a binade at a time, at the update counts (2m+1) 2^(b-8) / lr (at lr 1e-5 the loss falls ~0.09 at steps
           4, 7, 10, ~0.04 at 2, 5, 8, 11; at 1.1e-5 ~0.09 at 3, 6, 9), and a step n whose fence
           (2m+1) 2^(b-8) / (n lr) falls in 0.97-0.997, inside the per-weight steps, has a 0.01-0.1% difference
           in AdamW's step decide which copies have moved -- one step's gap, the larger the coarser the binade
           (1e-3 at lr 1.1e-5 step 7).  Compare away from those steps, or from an init off the bf16 grid (not
           tried).  bf16 gradients shorten optax's steps by 0.03-0.1% (it divides by 1 - b1^t rounded to
           bf16).  So loss parity cannot check the routed-bias update here (it moves the loss at most 1e-5 in 12
           steps): compare the bias vectors (MaxText's checkpoints).  `compare` notes a pairing that mismatches
           precision or optimizer arithmetic.  (optax multiplies the bf16 mu by bf16 b1, 0.8984375, whatever the
           gradient's dtype; bf16 gradients also round 1-b1 and 1-b2 to bf16 and compute the new mu in bf16.)

What still differs, by design (in the pipelined protocol: bf16 weights and gradients, --rs-dtype bfloat16,
tokamax on both sides): MaxText accumulates pipeline weight gradients in bf16 and all-reduces the routed
experts' over fsdp, ds_ep accumulates in fp32 and reduce-scatters in bf16; the kernels' fusion and summation
order differ; and the global-norm clip compares ds_ep's fp32 norm where MaxText's optax reduces it in bf16, so
the updates agree bit for bit only while the norm stays below the clip (gradient_clipping_threshold, 1.0) or
both norms round to the same bf16 value -- or with the clip off on both sides (gradient_clipping_threshold=0,
--grad-clip 0).  ds_ep's SparseCore ragged sort (--ragged-sort on) is not bit-reproducible from run to run
(identical executables, two runs with identical flags part by up to 1.1e-5 by step 5, nine by up to 1.5e-5;
step 0 agrees); with --ragged-sort off --ragged-buffer-factor 0 (the TensorCore sort, +24% step time at
DeepSeek-V3 dims) three runs, on different slices, agree bit for bit, as MaxText's do.  Measured at small scale
(v6e-4, PP2 x FSDP2, 4 layers, full DeepSeek-V3 per-layer dims, top-16 of 16, bf16; tokamax with base-2 exp
on both sides): loss 5.3e-7 relative at step 0, within 8.7e-6 over 4 steps, the bf16 gradient norm equal at
every step; MaxText against itself with pinned vs default XLA numerics moves the loss by 1e-5.  Against
MaxText DP over DCN (8 x TPU7x 2x2x2, 8 stages, 19 layers, 100 steps, 2026-10-25): 2e-7 to 4e-6
at step 0, within 1e-3 through step 40.  DeepSeek-V3 itself (2 x tpu7x-4x4x4 a side, 3 dense + 1 MoE layer +
MTP, top-8 of 256 in 4 of 8 groups, 6 steps at lr 1e-5; ds_ep --attn splash, MaxText
sa_use_base2_exp=False): with fp32 gradients on both sides the loss agrees within 6.8e-6 at every step and the
fp32 norm within 3.3-3.7e-5 (MaxText's bf16 reduce-scatters), against a MaxText-vs-MaxText spread (EP 4 x FSDP
32 vs EP 8 x FSDP 16) of 6.7e-6; at lr 1e-4 within 3.6e-5 through step 5, where MaxText's base-2 softmax parts
by 3e-4 at step 1 and 4.8e-3 by step 11.  These DeepSeek-V3 runs, and those under precision above, predate
ds_ep's fp32 router kernel (ds_ep.leaf_dtypes): their routers read the bf16 compute copy of the fp32 master,
where MaxText's GateLogit reads the master's values.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import sys
import time

# The model the MPMD runs train, as the MaxText SPMD pipeline baseline of the TPU7x comparison set it, minus
# everything that does not shape a parameter; `init` takes MaxText key=value overrides on top (a run's other
# settings).
MODEL = dict(
    model_name="deepseek3-671b",
    override_model_config=True,
    vocab_size=32768,
    first_num_dense_layers=0,
    base_mlp_dim=2048,
    num_experts_per_tok=8,
    shared_experts=1,
    routed_bias=False,
    dtype="bfloat16",
    weight_dtype="bfloat16",
    custom_mesh_and_rule="pipeline-large-moe",
    allow_split_physical_axes=True,
    max_target_length=4096,
    megablox=True,
    sparse_matmul=True,
    attention="flash",
    use_tokamax_splash=True,
)
RUN = dict(
    run_name="maxtext_parity_init",
    base_output_directory="/tmp/maxtext_parity",
    dataset_type="synthetic",
    enable_checkpointing=False,
    skip_jax_distributed_system=True,
    enable_goodput_recording=False,
)
# `init --exact`: DeepSeek-V3 itself, the model of the exact-DSv3 runs -- deepseek3-671b's own vocabulary, dense
# layers, dense MLP and routed bias (MODEL has the sparse prototype's) and the MLPerf recipe's MTP layer.
EXACT = dict(vocab_size=129280, first_num_dense_layers=3, base_mlp_dim=18432, routed_bias=True, mtp_num_layers=1)
# `init --model qwen3.8-2.4t-a95b` (or a Qwen3.5 model): MaxText's yml of that name -- decoder_block qwen3_5, its
# own vocabulary, rotary embedding (Qwen3.8: base.yml's use_mrope false) and every dim -- with the benchmark's
# numerics; the key=value overrides cut it down (base_num_decoder_layers, num_experts, ...).  Cut to another
# head_dim, a Qwen3.5 model's text mRoPE needs sections summing to rotary_dim // 2 = int(head_dim * 0.25) // 2
# (MaxText checks it; text positions never read the split): mrope_section=[3,3,2] at head_dim 64.  `init` refuses
# a leaf with no place in a chunk (qwen_to_chunks: use_multimodal=true's vision encoder, say) before it writes
# anything.
QWEN_MODEL = dict(
    override_model_config=True,
    dtype="bfloat16",
    weight_dtype="bfloat16",
    max_target_length=4096,
    megablox=True,
    sparse_matmul=True,
    attention="flash",
    use_tokamax_splash=True,
)
QWEN_MODELS = ("qwen3.8-2.4t-a95b", "qwen3.5-397b-a17b", "qwen3.5-35b-a3b")


def _value(v):
  """A key=value override's value: an int, a float, a list, tuple or dict literal (mrope_section=[3,3,2]: pyconfig
  takes a keyword's value as it is, and the string would fail its validation), true / false, else the string."""
  for cast in (int, float):
    try:
      return cast(v)
    except ValueError:
      pass
  if v.strip()[:1] in ("[", "(", "{"):
    try:
      return ast.literal_eval(v)
    except (ValueError, SyntaxError):
      pass
  return {"true": True, "false": False}.get(v.lower(), v)


def maxtext_models_dir():
  """MaxText's configs/models/, its model ymls: maxtext.utils.globals.MAXTEXT_CONFIGS_DIR's models/."""
  from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR  # pylint: disable=import-outside-toplevel

  return os.path.join(MAXTEXT_CONFIGS_DIR, "models")


def _moe_names(rest):
  """MaxText's names inside a DeepSeek layer -> ds_ep's: the MoE block is moe, the routed experts not nested
  (DeepSeekMoeBlock_0/MoeBlock_0 is moe, DeepSeekMoeBlock_0/shared_experts moe/shared_experts)."""
  rest = list(rest)
  if rest and rest[0] == "DeepSeekMoeBlock_0":
    rest = ["moe"] + (rest[2:] if rest[1] == "MoeBlock_0" else rest[1:])
  return rest


def maxtext_to_chunks(flat, n_stages, n_chunks, layer_split=None, scan_axis=1):
  """MaxText's params {path tuple: array} -> ds_ep chunk trees {'/'-joined path: array}, chunk order.

  The pipeline's layout (decoder/pipeline_module/layers, a leaf [repeat, stage, ...]: one layer a chunk), or
  the scanned one of a run without PP (decoder/dense_layers and decoder/moe_layers, a leaf's layer axis at
  `scan_axis`, a routed bias's -- an MoEBiasVar, not a parameter -- at 0), cut by `layer_split` as
  ds_ep_pipeline cuts --layer-split: the MoE layers a chunk, the dense layers all in chunk 0, or -- its sum
  every layer -- all of them in order, the dense ones first.  The MTP layer (mtp_block/mtp_layer_1/mtp_1_*) goes
  to the last chunk's 'mtp', with a copy of the token embedding that MaxText shares with the decoder's input
  (ds_ep's last chunk keeps one: ds_ep.mtp_shapes)."""
  import numpy as np

  chunks = [{} for _ in range(n_chunks)]
  axis = lambda rest: 0 if list(rest[-2:]) == ["gate", "bias"] else scan_axis
  scanned = {
      p: ("dense_layers" if p[1] == "dense_layers" else "layers", _moe_names(p[2:]))
      for p in flat
      if p[:2] in (("decoder", "dense_layers"), ("decoder", "moe_layers"))
  }
  owner = {}  # 'dense_layers' / 'layers' -> the chunk of each of those layers, in order
  if scanned:
    if layer_split is None:
      raise ValueError(
          f'MaxText parameter {"/".join(next(iter(scanned)))} is a scanned layer leaf: give the layer ' "split"
      )
    if len(layer_split) != n_chunks:
      raise ValueError(f"layer split {list(layer_split)}: {len(layer_split)} chunks, not {n_chunks}")
    depth = {}
    for p, (kind, rest) in scanned.items():
      n = flat[p].shape[axis(rest)]
      if depth.setdefault(kind, n) != n:
        raise ValueError(
            f'MaxText {kind} leaf {"/".join(rest)} stacks {n} layers on axis {axis(rest)}, its others ' f"{depth[kind]}"
        )
    n_dense, n_moe = depth.get("dense_layers", 0), depth.get("layers", 0)
    chunk_of = [c for c, n in enumerate(layer_split) for _ in range(n)]
    if len(chunk_of) == n_moe:  # the MoE layers a chunk, the dense ones in chunk 0
      owner = {"dense_layers": [0] * n_dense, "layers": chunk_of}
    elif len(chunk_of) == n_dense + n_moe:  # every layer a chunk, in order
      owner = {"dense_layers": chunk_of[:n_dense], "layers": chunk_of[n_dense:]}
    else:
      raise ValueError(
          f"MaxText scans {n_moe} MoE layers, the layer split {list(layer_split)} holds "
          f"{len(chunk_of)}" + (f" (nor every layer, {n_dense + n_moe} with the {n_dense} dense ones)" if n_dense else "")
      )
  for path, v in flat.items():
    if path in scanned:
      kind, rest = scanned[path]
      layers = np.moveaxis(v, axis(rest), 0)  # the layer axis first, the rest in order: a view
      for c in range(n_chunks):
        mine = [i for i, o in enumerate(owner[kind]) if o == c]
        if mine:  # a chunk's layers of a kind are consecutive: a slice, a view, no copy
          chunks[c]["/".join([kind] + rest)] = layers[mine[0] : mine[-1] + 1]
    elif path[:3] == ("decoder", "pipeline_module", "layers"):
      rest = _moe_names(path[3:])
      stacked = v if n_chunks > n_stages else v[None]  # [repeat, stage, ...]; no repeat axis at V=1
      for c in range(n_chunks):
        chunks[c]["/".join(["layers"] + rest)] = stacked[c // n_stages, c % n_stages][None]
    elif path[:2] == ("mtp_block", "mtp_layer_1") and path[2].startswith("mtp_1_"):
      name = path[2][len("mtp_1_") :]
      rest = (["layer"] + _moe_names(path[3:])) if name == "transformer_layer" else [name, *path[3:]]
      chunks[-1]["/".join(["mtp"] + rest)] = v
    elif path == ("token_embedder", "embedding"):
      chunks[0]["token_embedder/embedding"] = v
    elif path == ("decoder", "decoder_norm", "scale"):
      chunks[-1]["decoder_norm/scale"] = v
    elif path == ("decoder", "logits_dense", "kernel"):
      chunks[-1]["logits_dense/kernel"] = v
    else:
      raise ValueError(f'MaxText parameter {"/".join(path)} has no place in a ds_ep chunk')
  if any(k.startswith("mtp/") for k in chunks[-1]):
    chunks[-1]["mtp/token_embedder/embedding"] = chunks[0]["token_embedder/embedding"]
  return chunks


def qwen_to_chunks(flat, layer_split, cycle, scan_axis=1):
  """A MaxText qwen3_5's params {path tuple: array} (no PP: its scanned blocks, a block's layers as its nested scan
  holds them, local_layers + global_layer: maxtext.experimental.mpmd.jax.qwen35.convert) -> ds_ep chunk trees
  {'/'-joined path: array}, chunk order, cut by `layer_split` (every layer a chunk; ds_ep_pipeline --layer-split).
  Layer l is position l % cycle of block l // cycle; in its chunk it is one subtree (ds_ep.qwen_layer_key) whose
  leaves keep a layer axis of one: the common leaves under their names, its mixer -- MaxText's 'attention' -- as
  'gdn', or as 'attn' for a full-attention layer (whose leaves sit under 'attention'/'attention').  Views of
  MaxText's arrays wherever a slice is one.  A MaxText parameter with no place in a chunk (a vision encoder's, a
  decoder-level leaf of a later MaxText, one beside a full-attention layer's inner 'attention', which 'attn' leaves
  out) is an error, as in maxtext_to_chunks; a layer's other extra leaf goes to its chunk, where params_from_chunks
  reports it."""
  import jax
  import numpy as np
  from maxtext.experimental.mpmd.jax import ds_ep
  from maxtext.experimental.mpmd.jax.qwen35 import convert

  places = {("token_embedder", "embedding"), ("decoder", "decoder_norm", "scale"), ("decoder", "logits_dense", "kernel")}
  layer_keys = {"local_layers", "global_layer"}  # MaxText's nested scan of a block
  for path in flat:
    if tuple(path) not in places and not (
        tuple(path[:2]) == ("decoder", "layers") and len(path) > 3 and path[2] in layer_keys
    ):
      raise ValueError(f'MaxText parameter {"/".join(path)} has no place in a ds_ep chunk')
  port = convert.from_flat([(list(p), v) for p, v in flat.items()], scan_axis, asarray=np.asarray)
  blocks = port["decoder"]["layers"]
  if sorted(blocks) != sorted(f"layer_{i}" for i in range(cycle)):  # a nested scan of another length
    raise ValueError(f"MaxText's blocks hold {sorted(blocks)}, not a block of {cycle} layers")
  n_layers = cycle * jax.tree.leaves(blocks)[0].shape[0]
  if sum(layer_split) != n_layers:
    raise ValueError(f"layer split {list(layer_split)} holds {sum(layer_split)} layers, MaxText has {n_layers}")
  chunks, first = [{} for _ in layer_split], 0
  for c, n in enumerate(layer_split):
    for j, l in enumerate(range(first, first + n)):
      lw = jax.tree.map(lambda a, b=l // cycle: a[b : b + 1], blocks[f"layer_{l % cycle}"])  # [1, ...]: a view
      mixer = lw.pop("attention")
      if "attention" in mixer and len(mixer) > 1:  # 'attn' is the inner attention alone
        raise ValueError(
            f"MaxText parameters {sorted('attention/' + k for k in mixer if k != 'attention')} beside "
            f"layer {l}'s inner attention have no place in a ds_ep chunk"
        )
      lw["attn" if "attention" in mixer else "gdn"] = mixer["attention"] if "attention" in mixer else mixer
      for path, v in jax.tree_util.tree_flatten_with_path(lw)[0]:
        chunks[c]["/".join(["layers", ds_ep.qwen_layer_key(j)] + [k.key for k in path])] = v
    first += n
  chunks[0]["token_embedder/embedding"] = port["token_embedder"]["embedding"]
  chunks[-1]["decoder_norm/scale"] = port["decoder"]["decoder_norm"]["scale"]
  chunks[-1]["logits_dense/kernel"] = port["decoder"]["logits_dense"]["kernel"]
  return chunks


def nest(flat):
  """{'/'-joined path: array} -> nested dicts, the chunk trees ds_ep_pipeline.write_chunk_files takes."""
  tree = {}
  for k, v in flat.items():
    *path, leaf = k.split("/")
    node = tree
    for q in path:
      node = node.setdefault(q, {})
    node[leaf] = v
  return tree


def cmd_init(a):
  """MaxText init (CPU is fine: PP=S x FSDP=1 has the PP=S x FSDP=64 shapes; FSDP only shards).  Without PP
  (ici_pipeline_parallelism=1, the layout of a DP-over-DCN run) on one device, its shapes at any sharding, cut
  by --layer-split."""
  import jax
  import numpy as np
  from flax import nnx
  from flax.linen import partitioning as nn_partitioning
  from maxtext.common import checkpointing
  from maxtext.common.common_types import MODEL_MODE_TRAIN
  from maxtext.configs import pyconfig
  from maxtext.utils import model_creation_utils
  import maxtext
  from maxtext.experimental.mpmd.jax import ds_ep_pipeline as dsp

  t0 = time.perf_counter()
  log = lambda *x: print(f"[{time.perf_counter() - t0:7.1f}s]", *x, flush=True)
  split = tuple(int(n) for n in a.layer_split.split(",")) if a.layer_split else None
  qwen = a.model in QWEN_MODELS
  if qwen and (a.exact or not split):
    sys.exit(f"init --model {a.model}: MaxText without PP, cut by --layer-split (every layer a chunk); not --exact")
  kw = dict(QWEN_MODEL, model_name=a.model, **RUN) if qwen else dict(MODEL, **RUN, **(EXACT if a.exact else {}))
  kw.update((k, _value(v)) for k, v in (x.split("=", 1) for x in a.overrides))
  S = int(kw.get("ici_pipeline_parallelism", 1))
  if (S == 1) != (split is not None):
    sys.exit(
        "init: --layer-split goes with ici_pipeline_parallelism=1 (MaxText scans its layers; the split cuts "
        "them into ds_ep chunks), and only there (a pipeline is one layer a chunk)"
    )
  # Init on whatever devices there are: one per stage, no data / fsdp / expert sharding.
  kw.update(
      ici_pipeline_parallelism=S,
      ici_fsdp_parallelism=1,
      ici_expert_parallelism=1,
      ici_data_parallelism=1,
      dcn_data_parallelism=1,
      per_device_batch_size=1,
      compile_topology="",
      compile_topology_num_slices=-1,
  )
  if S > 1:
    kw.update(num_pipeline_microbatches=S, pipeline_fsdp_ag_per_repeat=False, pipeline_fsdp_ag_once=False)
  else:
    kw.pop("custom_mesh_and_rule", None)  # the pipeline's mesh rules
  if jax.device_count() != S:
    sys.exit(
        f"init: run with exactly ici_pipeline_parallelism = {S} devices "
        f"(XLA_FLAGS=--xla_force_host_platform_device_count={S}), have {jax.device_count()}"
    )
  base_yml = os.path.join(os.path.dirname(maxtext.__file__), "configs", "base.yml")
  cfg = pyconfig.initialize([sys.argv[0], base_yml], **kw)
  if S > 1:
    n_chunks = cfg.num_decoder_layers // cfg.num_layers_per_pipeline_stage
    log(
        f"MaxText {os.path.dirname(maxtext.__file__)}: {cfg.num_decoder_layers} layers, {cfg.num_experts} "
        f"experts, {S} stages x {cfg.num_pipeline_repeats} repeats -> {n_chunks} ds_ep chunks; weight_dtype "
        f"{cfg.weight_dtype}"
    )
    if cfg.num_layers_per_pipeline_stage != 1:
      sys.exit("init: ds_ep chunks of one layer need num_layers_per_pipeline_stage=1")
  elif qwen:
    n_chunks = len(split)
    if sum(split) != cfg.num_decoder_layers or not cfg.scan_layers:
      sys.exit(
          f"init: the layer split holds {sum(split)} layers: MaxText has {cfg.num_decoder_layers}, scanned "
          f"(scan_layers {cfg.scan_layers})"
      )
    log(
        f"MaxText {os.path.dirname(maxtext.__file__)}: {cfg.model_name}, {cfg.num_decoder_layers} layers in blocks "
        f"of {cfg.inhomogeneous_layer_cycle_interval}, {cfg.num_experts} experts, use_mrope {cfg.use_mrope}, no PP "
        f"(blocks scanned at axis {cfg.param_scan_axis}) -> {n_chunks} ds_ep chunks {list(split)}; weight_dtype "
        f"{cfg.weight_dtype}"
    )
  else:
    n_chunks = len(split)
    dense = cfg.first_num_dense_layers
    if sum(split) not in (cfg.num_decoder_layers - dense, cfg.num_decoder_layers) or not cfg.scan_layers:
      sys.exit(
          f"init: the layer split holds {sum(split)} layers: MaxText has {cfg.num_decoder_layers - dense} MoE "
          f"layers after {dense} dense, scanned (scan_layers {cfg.scan_layers}): give the MoE layers a chunk or "
          "every layer a chunk, as ds_ep_pipeline --layer-split"
      )
    log(
        f"MaxText {os.path.dirname(maxtext.__file__)}: {cfg.num_decoder_layers} layers ({dense} dense), "
        f"{cfg.mtp_num_layers} MTP, {cfg.num_experts} experts, no PP (layers scanned at axis {cfg.param_scan_axis}) "
        f"-> {n_chunks} ds_ep chunks {list(split)}; weight_dtype {cfg.weight_dtype}"
    )
  create, abstract = model_creation_utils.create_nnx_abstract_model(
      cfg,
      mesh=None,
      devices=jax.devices(),
      model_mode=MODEL_MODE_TRAIN,
      rng_key=jax.random.PRNGKey(cfg.init_weights_seed),
  )
  graphdef, abs_state = nnx.split(abstract)
  shardings = jax.tree.map(lambda x: x.sharding, abs_state)
  with nn_partitioning.axis_rules(cfg.logical_axis_rules), jax.set_mesh(abstract.mesh):
    state = jax.jit(lambda: nnx.state(create()), out_shardings=shardings)()
  model = nnx.merge(graphdef, state)
  from maxtext.layers.moe import MoEBiasVar

  # What load_parameters_path restores: the parameters and the routed biases, MoEBiasVars (zero at init),
  # which its loader looks up in the params collection when the checkpoint has no collection of theirs.
  weights = nnx.state(model, nnx.Any(nnx.Param, MoEBiasVar))
  jax.block_until_ready(weights)
  log("initialized")
  n_params = sum(v.size for _, v in nnx.state(model, nnx.Param).flat_state())
  pure = weights.to_pure_dict()
  flat = {
      tuple(str(p) for p in k): np.asarray(v.get_value() if hasattr(v, "get_value") else v)
      for k, v in weights.flat_state()
  }
  del weights, state, model
  if qwen:  # before anything is written
    # The cut (views, no copy): a leaf with no place in a chunk (use_multimodal=true's vision encoder, say).
    try:
      qwen_to_chunks(flat, split, cfg.inhomogeneous_layer_cycle_interval, cfg.param_scan_axis)
    except ValueError as e:
      sys.exit(f"init --model {a.model}: {e}")
  out = a.out.rstrip("/")
  for dtype in [None] + list(a.cast):
    sfx = f"_{dtype}" if dtype else ""
    tree = pure if dtype is None else jax.tree.map(lambda x: x.astype(dtype), pure)
    checkpointing.save_params_to_path(f"{out}/maxtext_params{sfx}", {"params": tree})
    log(f"MaxText params checkpoint -> {out}/maxtext_params{sfx}")
    del tree
    leaves = flat if dtype is None else {k: v.astype(dtype) for k, v in flat.items()}
    # Local or gs://, like the checkpoint; the files ds_ep_pipeline --init-from reads.
    trees = (
        qwen_to_chunks(leaves, split, cfg.inhomogeneous_layer_cycle_interval, cfg.param_scan_axis)
        if qwen
        else maxtext_to_chunks(leaves, S, n_chunks, split, cfg.param_scan_axis)
    )
    dsp.write_chunk_files(f"{out}/ds_chunks{sfx}", [nest(t) for t in trees])
    log(
        f"ds_ep chunk files -> {out}/ds_chunks{sfx}/chunk0..{n_chunks - 1}.npz "
        f"({n_params / 1e9:.3f}B parameters, {next(iter(leaves.values())).dtype})"
    )
    del leaves


# ------------------------------------------------------------------ compare
def read_maxtext_metrics(path):
  """MaxText's metrics_file (JSON lines, one per logged step) -> {step: record}."""
  from maxtext.experimental.mpmd.jax import xprof_align

  rows = [json.loads(line) for line in xprof_align.read_bytes(path).decode().splitlines() if line.strip()]
  return {int(r["step"]): r for r in rows}


def read_ds_ep(path):
  from maxtext.experimental.mpmd.jax import xprof_align

  d = json.loads(xprof_align.read_bytes(path))
  if d.get("warmup_gnorms") is None:
    sys.exit(f"{path} has no warmup_gnorms (the JSON of an older ds_ep_pipeline): rerun it with this one")
  # A load-balance loss rides ds_ep's layer outputs' backward, out of its loss.
  aux = (d.get("config") or {}).get("lb_loss_weight", 0) > 0
  return d.get("warmup_losses"), d["warmup_gnorms"], d.get("warmup_gnorms_bf16"), aux


def _loss(record, minus_aux):
  return float(record["learning/loss"]) - (float(record["learning/moe_lb_loss"]) if minus_aux else 0.0)


def compare(
    mt,
    losses,
    gnorms,
    gnorms_bf16,
    loss_rtol,
    loss0_rtol,
    gnorm_rtol,
    minus_aux=False,
    gnorm32_rtol=1e-4,
    control=None,
    control_factor=3.0,
    gnorm32_later_rtol=None,
    rs_dtype=None,
    grad_dtype=None,
    adam_math=None,
    weight_dtype=None,
):
  """-> (table lines, failures).  `minus_aux`: the ds_ep run has a load-balance loss, which is in MaxText's
  learning/loss (learning/moe_lb_loss) and not in ds_ep's: compare MaxText's loss without it.  A run whose MaxText
  norms are all bf16 values (bf16 gradients: MaxText reduces the norm in bf16) is held against ds_ep's norms
  rounded to bf16 at `gnorm_rtol` (bf16 ulps), any other against its fp32 norms at `gnorm32_rtol` at step 0 (one
  set of weights: the gradient itself) and after at `gnorm32_later_rtol`, by default `gnorm_rtol` (the
  trajectories may part): one run, one precision, so an fp32 norm that happens to fall on the bf16 grid is not
  taken for a bf16 one.  `control`: the
  metrics of a MaxText run that computes the same thing another way (another mesh): every step's loss must also
  be within `control_factor` x the largest MaxText-vs-control difference of any step.  `rs_dtype`, `grad_dtype`,
  `weight_dtype`: ds_ep's gradient reduce-scatter dtype, its --grad-dtype (the dtype the reduced gradient enters
  AdamW in; None or 'auto': under --adam-math optax the weight's, else rs_dtype) and its weight dtype, to note a
  pairing whose precision does not match MaxText's gradients; `adam_math` (--adam-math), to note a ds_ep AdamW
  that is not optax's arithmetic, MaxText's."""
  import ml_dtypes
  import numpy as np

  noise = None
  if control is not None:
    keys = ("learning/loss",) + (("learning/moe_lb_loss",) if minus_aux else ())
    usable = lambda r: all(k in r for k in keys)
    common = [t for t in control if t in mt and usable(control[t]) and usable(mt[t])]
    if not common:
      return [], [f'the control run shares no step with the MaxText run (with {" and ".join(keys)})']
    diffs = [abs(_loss(control[t], minus_aux) - _loss(mt[t], minus_aux)) for t in common]
    noise = max(diffs)
    if not all(math.isfinite(x) for x in diffs) or noise == 0:
      return [], [
          f"the control's spread is {noise:.3g}: a control must be finite and differ from MaxText (MaxText "
          "is bit-reproducible on one mesh: run the control on another)"
      ]
  lines = [
      f'step  MaxText loss{"-lb" if minus_aux else "   "} ds_ep loss      rel diff   MaxText raw gnorm  '
      "ds_ep gnorm (as compared)  rel diff" + ("   |ds-MT| / control spread" if control else "")
  ]
  fails, noted, worst = [], False, (0.0, None)
  if grad_dtype not in (None, "auto"):
    adam_dtype = grad_dtype
  elif adam_math == "optax" and weight_dtype:
    adam_dtype = weight_dtype  # optax takes each gradient in its weight's dtype
  else:
    adam_dtype = rs_dtype
  norms = [
      float(mt[t]["learning/raw_grad_norm"]) for t in range(len(losses)) if t in mt and "learning/raw_grad_norm" in mt[t]
  ]
  bf16 = all(float(ml_dtypes.bfloat16(x)) == x for x in norms if math.isfinite(x))
  if adam_math not in (None, "optax"):
    lines.append(
        f"      note: ds_ep ran --adam-math {adam_math}, AdamW in fp32 with fp32 constants, where MaxText's "
        "optax multiplies its bf16 mu by bf16 b1 (0.8984375): pair --adam-math optax"
    )
  for t, (l, g) in enumerate(zip(losses, gnorms)):
    if t not in mt:
      fails.append(f"step {t}: not in the MaxText metrics")
      continue
    # MaxText logs no raw_grad_norm under optimizer_memory_host_offload.
    want = ("learning/loss", "learning/raw_grad_norm") + (("learning/moe_lb_loss",) if minus_aux else ())
    missing = [k for k in want if k not in mt[t]]
    if missing:
      fails.append(f'step {t}: MaxText logged no {" or ".join(missing)}')
      continue
    ml, mg = _loss(mt[t], minus_aux), float(mt[t]["learning/raw_grad_norm"])
    gb = gnorms_bf16[t] if gnorms_bf16 and bf16 else float(np.float32(g))
    if not all(math.isfinite(x) for x in (l, ml, gb, mg)):
      fails.append(f"step {t}: not finite (MaxText loss {ml}, gnorm {mg}; ds_ep loss {l}, gnorm {gb})")
      continue
    dl, dg = abs(l - ml) / max(abs(ml), 1e-30), abs(gb - mg) / max(abs(mg), 1e-30)
    ctl = f"   {abs(l - ml) / noise:9.1f}" if noise else ("   {:>9}".format("-") if control else "")
    as_compared = f"bf16 {gb:.6g}" if bf16 and gnorms_bf16 else "fp32"
    lines.append(f"{t:4d}  {ml:13.7f}  {l:13.7f}  {dl:9.2e}   {mg:17.6g}  {g:11.6f} ({as_compared:>11})  {dg:9.2e}{ctl}")
    if worst[1] is None or dl > worst[0]:
      worst = (dl, t)
    if adam_dtype and not noted and (adam_dtype == "bfloat16") != bf16:
      noted = True
      how = (
          f"reduce-scattered in {rs_dtype}"
          if adam_dtype == rs_dtype
          else f"took AdamW in {adam_dtype}" + (f" (reduce-scattered in {rs_dtype})" if rs_dtype else "")
      )
      lines.append(
          f'      note: MaxText\'s gradients are {"bf16" if bf16 else "fp32"} (its norm is) and ds_ep '
          f"{how}: such a pairing parts from about the fifth update (see precision)"
      )
    if dl > (loss0_rtol if t == 0 else loss_rtol):
      fails.append(f"step {t}: loss rel diff {dl:.2e}")
    if bf16:
      tol = gnorm_rtol
    elif t == 0:
      tol = gnorm32_rtol
    else:
      tol = gnorm_rtol if gnorm32_later_rtol is None else gnorm32_later_rtol
    if dg > tol:
      fails.append(f'step {t}: {"bf16" if bf16 else "fp32"} gnorm rel diff {dg:.2e}')
    if control is not None and t not in common:
      fails.append(f"step {t}: not in the control run")
    elif control is not None and abs(l - ml) > max(control_factor * noise, 1e-7 * abs(ml)):
      fails.append(
          f"step {t}: loss diff {abs(l - ml):.2e} > {max(control_factor * noise, 1e-7 * abs(ml)):.2e} "
          f"({control_factor:g} x the control's spread {noise:.2e}, at least 1e-7 x the loss)"
      )
  if worst[1] is not None:
    lines.append(f"largest loss rel diff {worst[0]:.2e} (step {worst[1]})")
  extra = sorted(t for t in mt if t >= len(losses))
  if extra:
    fails.append(f"MaxText steps {extra[0]}..{extra[-1]} are not in the ds_ep run ({len(losses)} warmup steps)")
  return lines, fails


def ds_ep_config(path):
  """The run JSON's config (the ds_ep_pipeline flags) -> dict."""
  from maxtext.experimental.mpmd.jax import xprof_align

  return json.loads(xprof_align.read_bytes(path)).get("config") or {}


def cmd_compare(a):
  mt = read_maxtext_metrics(a.maxtext_metrics)
  losses, gnorms, gnorms_bf16, aux = read_ds_ep(a.ds_ep_json)
  if not losses:
    sys.exit(f"{a.ds_ep_json} has no warmup losses: take the JSON of the process running the last stage")
  control = read_maxtext_metrics(a.control) if a.control else None
  cfg = ds_ep_config(a.ds_ep_json)
  lines, fails = compare(
      mt,
      losses,
      gnorms,
      gnorms_bf16,
      a.loss_rtol,
      a.loss0_rtol,
      a.gnorm_rtol,
      minus_aux=aux,
      gnorm32_rtol=a.gnorm32_rtol,
      control=control,
      control_factor=a.control_factor,
      gnorm32_later_rtol=a.gnorm32_later_rtol,
      rs_dtype=cfg.get("rs_dtype"),
      grad_dtype=cfg.get("grad_dtype"),
      adam_math=cfg.get("adam_math"),
      weight_dtype=cfg.get("weight_dtype"),
  )
  print("\n".join(lines))
  print("PASS" if not fails else "FAIL: " + "; ".join(fails))
  return 0 if not fails else 1


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  sub = ap.add_subparsers(dest="cmd", required=True)
  p = sub.add_parser("init", help="MaxText init -> OUT/maxtext_params and OUT/ds_chunks")
  p.add_argument("out")
  p.add_argument("overrides", nargs="*", help="MaxText key=value overrides (model shape, stages, repeats)")
  p.add_argument("--exact", action="store_true", help="DeepSeek-V3 itself (EXACT), not the sparse prototype")
  p.add_argument(
      "--model",
      default="deepseek3-671b",
      choices=("deepseek3-671b",) + QWEN_MODELS,
      help="MaxText's model: DeepSeek-V3's sparse prototype (MODEL), or a Qwen3.5-family yml as it is "
      "(QWEN_MODEL: ds_ep_pipeline --model of the same name; no PP, --layer-split every layer a "
      "chunk), its trainer run with packing=false (ds_ep's Gated DeltaNet takes no segment ids: the module "
      "docstring, step 2)",
  )
  p.add_argument(
      "--layer-split",
      default="",
      help="without PP (ici_pipeline_parallelism=1): the ds_ep run's --layer-split, its MoE layers a chunk "
      "(the dense ones in chunk 0, e.g. 1,1,1,1) or every layer a chunk (e.g. 2,2,1,1)",
  )
  p.add_argument(
      "--cast",
      action="append",
      default=[],
      metavar="DTYPE",
      help="also write the same weights in DTYPE (OUT/maxtext_params_DTYPE, OUT/ds_chunks_DTYPE)",
  )
  p = sub.add_parser("compare", help="MaxText's metrics_file vs a ds_ep_pipeline run JSON, per step")
  p.add_argument("maxtext_metrics")
  p.add_argument("ds_ep_json")
  # Defaults from the small-scale measurement above, with headroom for depth 8 and 16 microbatches.
  p.add_argument("--loss0-rtol", type=float, default=1e-5)
  p.add_argument("--loss-rtol", type=float, default=1e-4)
  p.add_argument(
      "--gnorm-rtol",
      type=float,
      default=8e-3,
      help="a bf16 MaxText norm: one bf16 ulp is 3.9e-3 to 7.8e-3 " "relative (2^-8 to 2^-7)",
  )
  p.add_argument(
      "--gnorm32-rtol",
      type=float,
      default=1e-4,
      help="an fp32 MaxText norm (weight_dtype and grad_dtype float32) at step 0: MaxText's bf16 "
      "reduce-scatters alone leave 3.3-3.7e-5 against ds_ep --rs-dtype float32, the softmax bias of "
      "sa_use_base2_exp 4.9e-4 (5.3e-4 against --rs-dtype bfloat16)",
  )
  p.add_argument(
      "--gnorm32-later-rtol",
      type=float,
      default=None,
      help="an fp32 MaxText norm after step 0 (default --gnorm-rtol): 3-4e-5 over 20 steps at lr 1e-5; at "
      "lr 1e-4 the matched trajectories part, 2e-3 by step 11",
  )
  p.add_argument(
      "--control",
      help="the metrics_file of a MaxText run computing the same thing another way (another "
      "mesh): fail a step whose loss is further from MaxText's than --control-factor x the control's spread",
  )
  p.add_argument("--control-factor", type=float, default=3.0)
  a = ap.parse_args(argv)
  return cmd_init(a) if a.cmd == "init" else cmd_compare(a)


if __name__ == "__main__":
  sys.exit(main())
