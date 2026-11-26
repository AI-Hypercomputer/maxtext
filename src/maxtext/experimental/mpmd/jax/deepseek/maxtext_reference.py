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

"""MaxText's own DeepSeek-V3 (NNX), built and driven as the reference this port is held to.

`build` resolves a MaxText config and builds MaxText's NNX DeepSeek-V3 on one device; `convert.from_nnx_model`
copies its weights into this port's parameter tree.  The pairs below run the same computation on both sides:

  * `maxtext_logits` / `port_logits`: the forward pass over one batch.
  * `moe_block` + `maxtext_moe_vjp` / `port_moe_vjp`: one MoE layer's routed-and-shared block, unscanned, forward
    and backward on the same activations and output cotangent, with the block's load-balance loss and routed-bias
    update.
  * `maxtext_train` / `port_train`: optimizer steps.  The MaxText side is assembled out of MaxText's and optax's
    own pieces (`max_utils.cross_entropy_with_logits`, `optax.clip_by_global_norm`, `optax.adamw`,
    `maxtext_utils.create_learning_rate_schedule`), driven the way `trainers/pre_train/train.py` drives them on the
    NNX path; the port side is `deepseek/train.py`, which has no optax and no Flax.

Every computation is jitted with `numerics()` as its XLA compile options.  Then whatever the two programs compute
with the same ops comes out bit-identical.  What they compute differently:

  * The norms.  MaxText stacks the scanned layers' weights at `param_scan_axis` (1), the port at axis 0, so a
    global norm sums the same squares in another order: the gradient norm, and through the clip every update after
    the first (AdamW's first update divides the gradient by its own magnitude), agree to fp32 rounding.
  * The routed experts under megablox, MaxText's default.  Off TPU MaxText interprets megablox's Pallas kernels;
    the port runs `jax.lax.ragged_dot` with fp32 accumulation.  The weight gradients differ by design: megablox's
    `tgmm` sums each expert's rows in a tile order no XLA-level op reproduces.  The expert outputs and input
    gradients agree but for a rare bf16 rounding, which in the whole model spreads: attention carries it to every
    later token, the backward to every layer below.  With `megablox=False` MaxText runs `jax.lax.ragged_dot` too,
    and `config.from_maxtext` drops the port's fp32 accumulation to match.  But MaxText pads a grouped matmul's
    rows (tokens x top-k) with zeros to a multiple of `wi_tile_fwd_batch_seq`, 512 by default, and the port does
    not; on CPU a matmul over the padded rows can round differently, most often the expert weight gradients, which
    sum over the rows.  A tile that divides the rows pads none.

On one CPU device at `TINY`, over init seeds 0-23: with `megablox=False` and the rows unpadded, the logits, every
gradient leaf and the loss of three steps are bit-exact at every seed, the norms within 1.2e-6 and the parameters
within 1.1e-7 (relative to each leaf's largest magnitude).  Padded to 512 rows, on jax 0.11.1, the first gradients
differed at 16 of the seeds (up to 2.2e-2) and the later losses at 9.  With megablox, an isolated MoE block's
output, input gradient and non-expert parameter gradients are bit-exact at every seed and the expert weight
gradients within 7.4e-4; the whole model's logits differ at seeds 19 and 20 (from one token on, up to 7.5e-3), and
its first gradients at 2 of seeds 0-8 (up to 1.8e-2).

One device is deliberate: on more than one, MaxText shards its activations through `with_logical_constraint` while
the port lets XLA choose, so the two sum weight gradients over a different partition of the batch.

Measured before the port, against an older MaxText, on one v6e chip with MaxText's defaults (megablox, the splash
kernel) and excess precision off, at DeepSeek-V3's 61-layer shapes with vocabulary 2048 and 64 tokens (3 steps) and
at deepseek3-tiny with 2048 tokens (2 steps): the loss, the gradient norm and the parameter norm bit-exact at every
step, and every gradient leaf but the three stacked expert weights bit-exact (at 2048 tokens those within 2.4e-7
absolute; with `gmm_exact_drhs`, which reproduces `tgmm`'s order at num_experts times the FLOPs, all 31 leaves at
the 61-layer shapes).
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax import nnx

from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils
from maxtext.utils import model_creation_utils
from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR

from maxtext.experimental.mpmd.jax.deepseek import convert
from maxtext.experimental.mpmd.jax.deepseek import model as ds_model
from maxtext.experimental.mpmd.jax.deepseek import moe as ds_moe
from maxtext.experimental.mpmd.jax.deepseek import train as ds_train

BASE_OVERRIDES = {
    "run_name": "parity",
    "enable_checkpointing": False,
    "dataset_type": "synthetic",
    "skip_jax_distributed_system": True,
    "override_model_config": True,
}

# deepseek3-tiny cut down to build and run on CPU in seconds: 1 dense and 3 MoE layers of 8 experts (top 2).
# max_target_length < 128 makes MaxText autoselect the bf16 dot-product attention, which the port mirrors.
TINY = {
    "base_num_decoder_layers": 4,
    "first_num_dense_layers": 1,
    "base_emb_dim": 64,
    "base_mlp_dim": 128,
    "base_moe_mlp_dim": 64,
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "vocab_size": 256,
    "max_target_length": 64,
    "max_prefill_predict_length": 16,
    "per_device_batch_size": 1.0,
}

# The stacked expert weights of an MoE layer (`MoeBlock_0/<name>`), whose gradients megablox computes with `tgmm`.
EXPERT_WEIGHTS = ("wi_0", "wi_1", "wo")

METRICS = ("loss", "total_weights", "raw_grad_norm", "grad_norm", "param_norm")


def numerics(backend=None):
  """The XLA compile options every computation here is jitted with.

  * `xla_allow_excess_precision=False`.  The option (on by default) lets XLA drop an intermediate bf16 rounding
    when it fuses the producer into its consumer.  Which roundings survive then depends on fusion decisions, and
    MaxText's graph and this one fuse differently -- MaxText carries extra `with_logical_constraint` and
    `out_sharding` nodes -- so two op-for-op identical programs come out apart.  On TPU that is up to a bf16 ulp
    per layer (every MLA and MoE sub-block compared alone is bit-exact either way; only the fused whole diverges);
    on CPU, 3 of the 31 gradient leaves of `TINY` with `megablox=False`.  A measurement knob, not a training one.
  * On CPU, `xla_cpu_experimental_ynn_fusion_type=""`: no YNNPACK fusions, so every bf16 matmul accumulates in fp32
    through XLA's own emitters, the same on any thread count.  With them on (the default since jax 0.11.2) a bf16
    weight gradient's contraction depends on the thread count.

  Passed per compile rather than through XLA_FLAGS, they apply whatever flags the process was started with, and
  change nothing else in it.
  """
  options = {"xla_allow_excess_precision": False}
  if (backend or jax.default_backend()) == "cpu":
    options["xla_cpu_experimental_ynn_fusion_type"] = ""
  return options


def _jit(fn, compiler_options):
  return jax.jit(fn, compiler_options=numerics() if compiler_options is None else compiler_options)


def build(preset=None, *, devices=None, config_path=None, **overrides):
  """Resolves a MaxText config and builds its NNX DeepSeek-V3, on `devices` (the first device by default).

  Args:
    preset: config overrides on top of `BASE_OVERRIDES`; `TINY` when None.
    devices: the devices of the model's mesh.  One keeps the comparison exact (see the module docstring).
    config_path: the base config; MaxText's `configs/base.yml` when None.
    **overrides: more config overrides, applied last.  `model_name` (default `deepseek3-tiny`) picks the model.

  Returns:
    `(config, model)`.  The config's global batch sizes count every device of the process; `batch_size` counts the
    model's.
  """
  merged = {**BASE_OVERRIDES, **(TINY if preset is None else preset), **overrides}
  # `model_name` selects a yml under configs/models and so has to travel in argv, not as a keyword override.
  model_name = merged.pop("model_name", "deepseek3-tiny")
  config_path = config_path or os.path.join(MAXTEXT_CONFIGS_DIR, "base.yml")
  cfg = pyconfig.initialize([None, config_path, f"model_name={model_name}"], **merged)
  devices = list(devices) if devices is not None else jax.devices()[:1]
  with nn.logical_axis_rules(cfg.logical_axis_rules):
    create = model_creation_utils.get_nnx_create_model_fn(
        cfg,
        mesh=None,
        devices=devices,
        model_mode=MODEL_MODE_TRAIN,
        rng_key=jax.random.PRNGKey(cfg.init_weights_seed),
    )
    model = create()
  return cfg, model


def batch_size(cfg, model):
  """The global batch over the model's own mesh: `per_device_batch_size` rows a device."""
  return int(cfg.per_device_batch_size * model.mesh.size)


def forward_batch(cfg, rows):
  """A plain non-packed forward batch of `rows` sequences: `(tokens, positions, segment_ids)`."""
  length = int(cfg.max_target_length)
  rng = np.random.default_rng(1234)
  tokens = jnp.asarray(rng.integers(0, cfg.vocab_size, size=(rows, length)), jnp.int32)
  positions = jnp.broadcast_to(jnp.arange(length, dtype=jnp.int32), (rows, length))
  segment_ids = jnp.ones((rows, length), jnp.int32)
  return tokens, positions, segment_ids


def train_batch(cfg, rows):
  """MaxText's synthetic training batch (`SyntheticDataIterator`) of `rows` sequences."""
  return ds_train.synthetic_batch(
      vocab_size=cfg.vocab_size,
      batch_size=rows,
      seq_len=int(cfg.max_target_length),
      packing=bool(cfg.packing),
      max_segments_per_seq=int(cfg.max_segments_per_seq),
  )


# --------------------------------------------------------------------------- #
# Forward
# --------------------------------------------------------------------------- #


def maxtext_logits(cfg, model, batch, *, compiler_options=None):
  """MaxText's logits for a `forward_batch`."""
  graphdef, state = nnx.split(model)

  def forward(state, tokens, positions, segment_ids):
    return nnx.merge(graphdef, state)(
        decoder_input_tokens=tokens,
        decoder_positions=positions,
        decoder_segment_ids=segment_ids,
        enable_dropout=False,
        model_mode=MODEL_MODE_TRAIN,
    )

  with nn.logical_axis_rules(cfg.logical_axis_rules):
    return jax.block_until_ready(_jit(forward, compiler_options)(state, *batch))


def port_logits(port_cfg, params, aux, batch, *, compiler_options=None):
  """The port's logits for a `forward_batch`, from `convert.from_nnx_model`'s `(params, aux)`."""

  def forward(params, aux, tokens, positions, segment_ids):
    return ds_model.forward(params, aux, tokens, positions, segment_ids, port_cfg)

  return jax.block_until_ready(_jit(forward, compiler_options)(params, aux, *batch))


# --------------------------------------------------------------------------- #
# One MoE block
# --------------------------------------------------------------------------- #


def moe_block(model, cfg, layer=0):
  """Returns `decoder.moe_layers.DeepSeekMoeBlock_0` with MoE layer `layer` sliced out.

  The NNX decoder keeps the MoE layers as one module whose parameters carry an extra scan axis at
  `cfg.param_scan_axis`; the module body itself is scan-agnostic, so dropping that axis makes it directly callable.
  """
  block = model.decoder.moe_layers.DeepSeekMoeBlock_0
  graphdef, params, rest = nnx.split(block, nnx.Param, ...)
  n_layers = cfg.num_decoder_layers - cfg.first_num_dense_layers

  def slicer(axis):
    # `nnx.Param` is stacked at `param_scan_axis`; everything else (the `MoEBiasVar` routing bias) at axis 0.
    def slice_leaf(x):
      x = jnp.asarray(x)
      if x.ndim > axis and x.shape[axis] == n_layers:
        return jnp.take(x, layer, axis=axis)
      return x

    return slice_leaf

  return nnx.merge(graphdef, jax.tree.map(slicer(cfg.param_scan_axis), params), jax.tree.map(slicer(0), rest))


def moe_block_params(block):
  """The port's `moe.routed_and_shared_moe` parameters of one unscanned `moe_block`, the routing bias included."""
  routed = block.MoeBlock_0
  shared = block.shared_experts
  return {
      "MoeBlock_0": {
          "gate": {
              "kernel": jnp.asarray(routed.gate.kernel.get_value()),
              "bias": jnp.asarray(routed.gate.bias.get_value()),
          },
          **{name: jnp.asarray(getattr(routed, name).get_value()) for name in EXPERT_WEIGHTS},
      },
      "shared_experts": {
          name: {"kernel": jnp.asarray(getattr(shared, name).kernel.get_value())} for name in EXPERT_WEIGHTS
      },
  }


def _vjp(fn, params, x, ct, compiler_options):
  """`(out, d_params, d_x, extras)` of `fn(params, x) -> (out, extras)`, `out` pulled back from the cotangent `ct`."""
  out, vjp, extras = _jit(lambda p, x: jax.vjp(fn, p, x, has_aux=True), compiler_options)(params, x)
  d_params, d_x = _jit(lambda f, c: f(c), compiler_options)(vjp, ct)
  return out, d_params, d_x, extras


def maxtext_moe_vjp(cfg, block, x, ct, *, compiler_options=None):
  """`(out, d_params, d_x, (lb_loss, bias_updates))` of MaxText's block, `d_params` in the port's layout (none for
  the routing bias).

  `lb_loss` and `bias_updates` are the block's other two outputs, its load-balance loss and its routed bias's update
  (each None when its weight is zero), as computed and not differentiated.
  """
  graphdef, params, rest = nnx.split(block, nnx.Param, ...)

  def fn(p, x):
    out, lb_loss, bias_updates = nnx.merge(graphdef, p, rest, copy=True)(x)
    return out, (lb_loss, bias_updates)

  with nn.logical_axis_rules(cfg.logical_axis_rules):
    out, d_params, d_x, extras = _vjp(fn, params, x, ct, compiler_options)
  return out, convert.tree_from_nnx_state(d_params, param_scan_axis=0), d_x, extras


def port_moe_vjp(port_cfg, params, x, ct, *, compiler_options=None):
  """`(out, d_params, d_x, (lb_loss, bias_updates))` of `moe.routed_and_shared_moe`, `d_params` without the routing
  bias's entry; the last two as `maxtext_moe_vjp`'s.

  The bias has to stay inside the traced parameter tree on this side: hoisting it into a jit constant lets XLA fold
  the `+ bias` away, which perturbs the bf16 gate logits enough to flip a few top-k selections.  MaxText produces no
  cotangent for it (a `MoEBiasVar`), so its entry is dropped from the gradient.

  `lb_loss` is MaxText's Switch-style loss even under `moe_use_megatron_seq_aux_loss`: the sequence-wise one
  DeepSeek-V3 trains with is ds_ep's `seq_aux_loss`.
  """

  def fn(p, x):
    out, lb_loss, bias_updates = ds_moe.routed_and_shared_moe(p, x, port_cfg)
    return out, (lb_loss, bias_updates)

  out, d_params, d_x, extras = _vjp(fn, params, x, ct, compiler_options)
  d_params["MoeBlock_0"]["gate"].pop("bias")
  return out, d_params, d_x, extras


# --------------------------------------------------------------------------- #
# Optimizer steps
# --------------------------------------------------------------------------- #


def train_config(cfg, steps):
  """The port's `TrainConfig` from MaxText's resolved config."""
  return ds_train.TrainConfig(
      steps=steps,
      learning_rate=cfg.learning_rate,
      learning_rate_final_fraction=cfg.learning_rate_final_fraction,
      learning_rate_schedule_steps=cfg.learning_rate_schedule_steps,
      warmup_steps_fraction=cfg.warmup_steps_fraction,
      lr_schedule_type=cfg.lr_schedule_type,
      adam_b1=cfg.adam_b1,
      adam_b2=cfg.adam_b2,
      adam_eps=cfg.adam_eps,
      adam_eps_root=cfg.adam_eps_root,
      adam_weight_decay=cfg.adam_weight_decay,
      gradient_clipping_threshold=cfg.gradient_clipping_threshold,
      z_loss_multiplier=cfg.z_loss_multiplier,
  )


def maxtext_loss(model, cfg, batch):
  """`train.py::loss_fn`, NNX branch, stripped to what this config reaches."""
  logits = model(
      decoder_input_tokens=batch["inputs"],
      decoder_positions=batch["inputs_position"],
      decoder_segment_ids=batch["inputs_segmentation"],
      enable_dropout=False,
      model_mode=MODEL_MODE_TRAIN,
  )
  one_hot_targets = jax.nn.one_hot(batch["targets"], cfg.vocab_size)
  xent, z_loss = max_utils.cross_entropy_with_logits(logits, one_hot_targets, z_loss=cfg.z_loss_multiplier)
  xent = xent * (batch["targets_segmentation"] != 0)
  z_loss = z_loss * (batch["targets_segmentation"] != 0)
  xent_sum = jnp.sum(xent)
  total_weights = jnp.sum(batch["targets_segmentation"] != 0)
  loss = xent_sum / (total_weights + 1e-8)
  aux = {"total_weights": total_weights, "z_loss": jnp.sum(z_loss) / (total_weights + 1e-8)}
  return loss, aux


def maxtext_train(cfg, model, batch, steps, *, compiler_options=None):
  """`steps` MaxText optimizer steps from the model's weights (the model itself is left as it is).

  Returns:
    `(history, params, raw_grads)`: each step's `METRICS` as float64 numpy scalars, the final parameters and the
    first step's unclipped gradients, both in the port's layout.
  """
  graphdef, params_state, rest = nnx.split(model, nnx.Param, ...)
  schedule = maxtext_utils.create_learning_rate_schedule(cfg)
  tx = optax.adamw(
      learning_rate=schedule,
      b1=cfg.adam_b1,
      b2=cfg.adam_b2,
      eps=cfg.adam_eps,
      eps_root=cfg.adam_eps_root,
      weight_decay=cfg.adam_weight_decay,
      mu_dtype="float32",
  )
  clip = optax.clip_by_global_norm(cfg.gradient_clipping_threshold)
  opt_state = tx.init(params_state)

  def diff_wrapper(curr_params, rest_state):
    local_model = nnx.merge(graphdef, curr_params, rest_state, copy=True)
    return maxtext_loss(local_model, cfg, batch)

  def step(params_state, rest_state, opt_state):
    (loss, aux), raw_grads = jax.value_and_grad(diff_wrapper, argnums=0, has_aux=True)(params_state, rest_state)
    grads, _ = clip.update(raw_grads, None, None)
    updates, opt_state = tx.update(grads, opt_state, params_state)
    params_state = optax.apply_updates(params_state, updates)
    metrics = {
        "loss": loss,
        "total_weights": aux["total_weights"],
        "raw_grad_norm": max_utils.l2norm_pytree(raw_grads),
        "grad_norm": max_utils.l2norm_pytree(grads),
        "param_norm": max_utils.l2norm_pytree(params_state),
    }
    return params_state, opt_state, raw_grads, metrics

  history = []
  raw_grads0 = None
  with nn.logical_axis_rules(cfg.logical_axis_rules):
    step = _jit(step, compiler_options)
    for _ in range(steps):
      params_state, opt_state, raw_grads, metrics = step(params_state, rest, opt_state)
      if raw_grads0 is None:
        raw_grads0 = raw_grads
      history.append(jax.tree.map(lambda x: np.asarray(x, np.float64), metrics))
  return (
      history,
      convert.tree_from_nnx_state(params_state, param_scan_axis=cfg.param_scan_axis),
      convert.tree_from_nnx_state(raw_grads0, param_scan_axis=cfg.param_scan_axis),
  )


def port_train(port_cfg, params, aux, batch, train_cfg, *, compiler_options=None):
  """`train_cfg.steps` optimizer steps of the port, returning what `maxtext_train` does."""
  schedule = ds_train.learning_rate_schedule(train_cfg)
  opt_state = ds_train.adamw_init(params, train_cfg)

  def step(params, opt_state):
    grad_fn = jax.value_and_grad(ds_train.compute_loss, has_aux=True)
    (loss, metrics), raw_grads = grad_fn(params, aux, batch, port_cfg, train_cfg)
    grads = ds_train.clip_by_global_norm(raw_grads, train_cfg.gradient_clipping_threshold)
    updates, opt_state = ds_train.adamw_update(grads, opt_state, params, train_cfg, schedule)
    params = ds_train.apply_updates(params, updates)
    out = {
        "loss": loss,
        "total_weights": metrics["total_weights"],
        "raw_grad_norm": ds_train.global_norm(raw_grads),
        "grad_norm": ds_train.global_norm(grads),
        "param_norm": ds_train.global_norm(params),
    }
    return params, opt_state, raw_grads, out

  step = _jit(step, compiler_options)
  history = []
  raw_grads0 = None
  for _ in range(train_cfg.steps):
    params, opt_state, raw_grads, metrics = step(params, opt_state)
    if raw_grads0 is None:
      raw_grads0 = raw_grads
    history.append(jax.tree.map(lambda x: np.asarray(x, np.float64), metrics))
  return history, params, raw_grads0
