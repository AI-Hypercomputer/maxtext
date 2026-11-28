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

"""The DeepSeek-V3 port in `deepseek/` against MaxText's own DeepSeek-V3 (NNX), on CPU.

    JAX_PLATFORMS=cpu python3 -m pytest -q tests/unit/experimental/mpmd/jax/deepseek_maxtext_parity_test.py

Five tests, about a minute and a half on a CPU with MaxText's import: marked scheduled_only (MaxText's scheduled CI
runs them).  Every check passes against the MaxText around it, on jax 0.12.0.1.

Each test builds MaxText's DeepSeek-V3 at `maxtext_reference.TINY` (deepseek3-tiny cut to 1 dense and 3 MoE layers
of 8 experts, 16 under `GROUP_ROUTING`, 64 tokens) on one CPU device, copies its weights into the port's tree with
`deepseek/convert.py` and runs both on the same inputs, every computation jitted with `maxtext_reference.numerics()`:

- `test_forward_logits`: the logits, bit-exact; and the port's tree written back into MaxText's model reads back
  as written.
- `test_moe_block`: one MoE layer's routed-and-shared block on random activations and output cotangent, with
  MaxText's default experts (megablox, its Pallas kernels interpreted on CPU) against the port's fp32-accumulating
  `ragged_dot`: the output and d_inputs bit-exact, and every parameter gradient but the stacked expert weights'.
  Those come from megablox's `tgmm`, which sums each expert's rows in its own tile order: within `EXPERT_GRAD_RTOL`.
- `test_train_steps`: three AdamW steps of MaxText's train step (loss, global-norm clip, optax AdamW, MaxText's
  schedule) against the port's: the learning rates, each step's loss and token count and every first-step gradient
  bit-exact; the norms within `NORM_RTOL`, the final parameters within `PARAM_RTOL`.
- `test_group_routing`: DeepSeek-V3's group-limited routing (`GROUP_ROUTING`) under a random routing bias (MaxText
  starts the bias at zero, where choosing experts on the biased scores and weighing them with the unbiased ones are
  one thing): the logits, the loss and every first-step gradient bit-exact.  The port without the group mask, or
  without the bias, computes other logits, so the case holds both.
- `test_moe_block_group_routing`: one MoE block with that routing, the routed bias's update and the sequence-wise
  aux loss (`moe_use_megatron_seq_aux_loss`) on, over two sequences: the output, d_inputs, every parameter gradient
  and the port's bias update (`moe.bias_update`) bit-exact, and the pipeline's aux loss (`ds_ep.seq_aux_loss`, which
  ds_ep_pipeline trains with) within `AUX_LOSS_RTOL` of MaxText's.  The group mask and the bias each change the
  experts of some token, so neither is idle.

The forward, train and routing checks run MaxText's `jax.lax.ragged_dot` experts (`RAGGED_DOT`).  With megablox the
expert outputs and input gradients agree with the port's but for a rare bf16 rounding, and in the whole model one
spreads (through attention to every later token, through the backward to every layer below): over init seeds 0-23
the logits differed at 2 (up to 7.5e-3 relative) and the first gradients at 2 of seeds 0-8 (up to 1.8e-2), so a
strict check of the whole model would break whenever MaxText's init changed.  With `megablox=False` and the rows
unpadded (below), all 24 seeds were bit-exact, in the routing checks too.  The MoE block test holds the megablox
path instead: at every seed its output, d_inputs and non-expert gradients were bit-exact.

`RAGGED_DOT` also sets `wi_tile_fwd_batch_seq` to 128, which divides a grouped matmul's rows (tokens x top-k: 64 x 2,
64 x 4 under `GROUP_ROUTING`).  MaxText pads the rows with zeros to a multiple of it, 512 by default, and the port
does not; on CPU a matmul over 512 rows can round differently from one over 128, the expert weight gradients (summed
over the rows) most often.  At 512, on jax 0.11.1, `test_train_steps` failed at 19 of the 24 seeds (seed 0 passed):
the first-step gradients differed at 16, by up to 2.2e-2 relative.

The bounds, each a relative error `max|a - b| / max|a|` per leaf, against the worst over init seeds 0-23:
- `EXPERT_GRAD_RTOL` = 2**-8, one bf16 ulp at the leaf's largest magnitude: the weights are cast to bf16 inside the
  layer, so their gradients are bf16 values, and a different fp32 summation order flips some roundings.  Measured
  7.4e-4 (7.1e-4 at seed 0, in 15 of `wi_1`'s 32768 values); 5x margin.
- `NORM_RTOL` = 1e-5: MaxText stacks the scanned layers' weights at `param_scan_axis` 1, the port at 0, so the global
  norms (and through the clip, the updates after the first) sum the same squares in another order.  Measured 1.2e-6;
  8x margin.
- `PARAM_RTOL` = 1e-6: the final parameters, whose updates differ through the clip's norm by about an fp32 ulp.
  Measured 1.1e-7 (7.4e-8 at seed 0); 9x margin.
- `AUX_LOSS_RTOL` = 1e-6: `ds_ep.seq_aux_loss` and MaxText's `RoutedMoE.megatron_seq_aux_loss` add up the same fp32
  terms in another order (a token fraction as count / (seq x k) against mean(count) / k; per sequence a sum over the
  experts, then a sum over the sequences, against one mean over both).  Measured 1.9e-7, two fp32 ulps (one at seed
  0, none at 11 of the seeds); 5x margin.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maxtext.experimental.mpmd.jax import ds_ep
from maxtext.experimental.mpmd.jax import testing
from maxtext.experimental.mpmd.jax.deepseek import config as ds_config
from maxtext.experimental.mpmd.jax.deepseek import convert
from maxtext.experimental.mpmd.jax.deepseek import maxtext_reference as mr
from maxtext.experimental.mpmd.jax.deepseek import moe as ds_moe
from maxtext.experimental.mpmd.jax.deepseek import train as ds_train
from maxtext.utils import maxtext_utils

EXPERT_GRAD_RTOL = 2.0**-8
NORM_RTOL = 1e-5
PARAM_RTOL = 1e-6
AUX_LOSS_RTOL = 1e-6
STEPS = 3
# MaxText's `jax.lax.ragged_dot` experts, op for op the port's (`config.from_maxtext` follows `megablox`), on rows it
# does not pad (`wi_tile_fwd_batch_seq` dividing them; see the module docstring).
RAGGED_DOT = {"megablox": False, "wi_tile_fwd_batch_seq": 128}
# DeepSeek-V3's group-limited routing (deepseek3-671b with the 8 groups and top 4 its yml recommends: a token's top 8
# of the 128 experts of its best 4 of 8 groups) at TINY's scale: a token's top 4 of the 8 experts of its best 2 of 4
# groups.  16 experts, so that a group holds 4 and its score, the sum of its two best, is not simply its sum.
GROUP_ROUTING = {"num_experts": 16, "num_experts_per_tok": 4, "n_routing_groups": 4, "topk_routing_group": 2}
# The random routing bias's scale: half the spread of the router's sigmoid scores at init (std 0.2), enough to change
# some of the experts of about half the tokens.
ROUTING_BIAS_STD = 0.1
# Five MaxText builds and their compiles, about a minute and a half on a CPU.
pytestmark = pytest.mark.scheduled_only


def setUpModule():
  testing.require_cpu("CPU test (its bounds are CPU measurements)")


def _leaves(tree):
  return {
      "/".join(str(getattr(k, "key", k)) for k in path): np.asarray(leaf)
      for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]
  }


def _rel(expected, actual):
  """`max|expected - actual| / max|expected|`: the error relative to the leaf's largest magnitude."""
  expected = np.asarray(expected, np.float64)
  actual = np.asarray(actual, np.float64)
  return float(np.abs(expected - actual).max() / max(np.abs(expected).max(), np.finfo(np.float64).tiny))


def _is_expert_weight(path):
  """`.../MoeBlock_0/{wi_0,wi_1,wo}`: a stacked expert weight (the shared experts' are `.../<name>/kernel`)."""
  parts = path.split("/")
  return len(parts) >= 2 and parts[-2] == "MoeBlock_0" and parts[-1] in mr.EXPERT_WEIGHTS


def _set_routing_bias(cfg, model):
  """Writes a random routing bias (`ROUTING_BIAS_STD`) into MaxText's model; returns the port's `(params, aux)` of it."""
  params, aux = convert.from_nnx_model(model, param_scan_axis=cfg.param_scan_axis)
  rng = np.random.default_rng(7)
  aux = jax.tree.map(lambda b: jnp.asarray(ROUTING_BIAS_STD * rng.standard_normal(b.shape), b.dtype), aux)
  convert.to_nnx_state(model, params, aux, param_scan_axis=cfg.param_scan_axis)
  return params, aux


def _pipeline_aux_loss(port_cfg, gate, x):
  """ds_ep_pipeline's sequence-wise aux loss of the router `gate` on `x [b, s, emb]`, as `ds_ep._ring_moe` adds it
  up: `ds_ep.seq_aux_loss` of the router's scores (`ds_ep.moe_scores`), summed over the `b` sequences and divided by
  `b`."""
  scfg = ds_ep.StageConfig(model=port_cfg, fsdp=1, ep=1, gmm_impl="ragged_dot")  # moe_scores reads only the model

  def loss(gate, x):
    return jnp.sum(ds_ep.seq_aux_loss(ds_ep.moe_scores({"gate": gate}, x, scfg), port_cfg)) / x.shape[0]

  return jax.jit(loss, compiler_options=mr.numerics())(gate, x)


class MaxTextModelParityTest(unittest.TestCase):

  def assert_bit_exact(self, label, expected, actual):
    expected, actual = np.asarray(expected), np.asarray(actual)
    self.assertEqual((actual.shape, actual.dtype), (expected.shape, expected.dtype), label)
    self.assertTrue(np.isfinite(expected.astype(np.float32)).all(), f"{label}: MaxText's is not finite")
    if not np.array_equal(expected, actual):
      self.fail(f"{label} differs from MaxText's: relative {_rel(expected, actual):.3e}")

  def assert_trees(self, label, expected, actual, rtol=lambda path: 0.0):
    """Leaf by leaf: the same paths, shapes and dtypes; bit-exact where `rtol(path)` is 0, else `_rel` <= it."""
    expected, actual = _leaves(expected), _leaves(actual)
    self.assertEqual(sorted(actual), sorted(expected), f"{label}: the trees' leaves differ")
    failures = []
    for path, want in expected.items():
      got, bound = actual[path], rtol(path)
      if (got.shape, got.dtype) != (want.shape, want.dtype):
        failures.append(f"{path}: {got.shape} {got.dtype}, MaxText's {want.shape} {want.dtype}")
      elif bound == 0.0 and not np.array_equal(want, got):
        failures.append(f"{path}: not bit-exact (relative {_rel(want, got):.3e})")
      elif bound and (np.isnan(_rel(want, got)) or _rel(want, got) > bound):
        failures.append(f"{path}: relative {_rel(want, got):.3e} > {bound:g}")
    self.assertFalse(failures, f"{label} against MaxText's:\n  " + "\n  ".join(failures))

  def test_forward_logits(self):
    cfg, model = mr.build(**RAGGED_DOT)
    port_cfg = ds_config.from_maxtext(cfg)
    batch = mr.forward_batch(cfg, mr.batch_size(cfg, model))
    params, aux = convert.from_nnx_model(model, param_scan_axis=cfg.param_scan_axis)
    logits = mr.maxtext_logits(cfg, model, batch)
    port_logits = mr.port_logits(port_cfg, params, aux, batch)
    self.assert_bit_exact("logits", logits, port_logits)
    # `to_nnx_state` inverts `from_nnx_model`: every leaf lands at its MaxText path, its scan axis put back.
    doubled = jax.tree.map(lambda x: x * 2, (params, aux))
    convert.to_nnx_state(model, *doubled, param_scan_axis=cfg.param_scan_axis)
    self.assert_trees("read back", doubled, convert.from_nnx_model(model, param_scan_axis=cfg.param_scan_axis))

  def test_moe_block(self):
    cfg, model = mr.build()
    port_cfg = ds_config.from_maxtext(cfg)
    block = mr.moe_block(model, cfg)
    rng = np.random.default_rng(0)
    shape = (mr.batch_size(cfg, model), int(cfg.max_target_length), cfg.emb_dim)
    x = jnp.asarray(rng.standard_normal(shape), jnp.bfloat16)
    ct = jnp.asarray(rng.standard_normal(shape), jnp.bfloat16)
    out, d_params, d_x, _ = mr.maxtext_moe_vjp(cfg, block, x, ct)
    port_out, port_d_params, port_d_x, _ = mr.port_moe_vjp(port_cfg, mr.moe_block_params(block), x, ct)
    self.assert_bit_exact("output", out, port_out)
    self.assert_bit_exact("d_inputs", d_x, port_d_x)
    self.assertEqual(sum(map(_is_expert_weight, _leaves(d_params))), len(mr.EXPERT_WEIGHTS))
    self.assert_trees(
        "d_params", d_params, port_d_params, rtol=lambda path: EXPERT_GRAD_RTOL if _is_expert_weight(path) else 0.0
    )

  def test_train_steps(self):
    cfg, model = mr.build(steps=STEPS, **RAGGED_DOT)
    port_cfg = ds_config.from_maxtext(cfg)
    train_cfg = mr.train_config(cfg, STEPS)
    schedule = maxtext_utils.create_learning_rate_schedule(cfg)
    port_schedule = ds_train.learning_rate_schedule(train_cfg)
    self.assertEqual([float(port_schedule(i)) for i in range(STEPS)], [float(schedule(i)) for i in range(STEPS)])

    batch = mr.train_batch(cfg, mr.batch_size(cfg, model))
    params, aux = convert.from_nnx_model(model, param_scan_axis=cfg.param_scan_axis)
    history, final, grads = mr.maxtext_train(cfg, model, batch, STEPS)
    port_history, port_final, port_grads = mr.port_train(port_cfg, params, aux, batch, train_cfg)

    self.assert_trees("first step's gradients", grads, port_grads)
    for step, (want, got) in enumerate(zip(history, port_history)):
      for key in ("loss", "total_weights"):
        self.assertEqual(got[key], want[key], f"step {step}: {key}")
      for key in ("raw_grad_norm", "grad_norm", "param_norm"):
        self.assertLessEqual(_rel(want[key], got[key]), NORM_RTOL, f"step {step}: {key} {got[key]} vs {want[key]}")
    self.assertNotEqual(history[-1]["loss"], history[0]["loss"], "the steps left the weights where they were")
    self.assert_trees("final parameters", final, port_final, rtol=lambda path: PARAM_RTOL)

  def test_group_routing(self):
    cfg, model = mr.build(steps=1, **RAGGED_DOT, **GROUP_ROUTING)
    port_cfg = ds_config.from_maxtext(cfg)
    params, aux = _set_routing_bias(cfg, model)
    rows = mr.batch_size(cfg, model)

    batch = mr.forward_batch(cfg, rows)
    logits = mr.maxtext_logits(cfg, model, batch)
    self.assert_bit_exact("logits", logits, mr.port_logits(port_cfg, params, aux, batch))
    # A port that ignored the groups or the bias would compute other logits: the check above holds both.
    for label, other_cfg, other_aux in (
        ("without the group mask", port_cfg.replace(n_routing_groups=-1, topk_routing_group=-1), aux),
        ("with the routing bias at zero", port_cfg, jax.tree.map(jnp.zeros_like, aux)),
    ):
      other = mr.port_logits(other_cfg, params, other_aux, batch)
      self.assertFalse(np.array_equal(other, logits), f"the port's logits {label} are MaxText's too")

    batch = mr.train_batch(cfg, rows)
    history, _, grads = mr.maxtext_train(cfg, model, batch, 1)
    port_history, _, port_grads = mr.port_train(port_cfg, params, aux, batch, mr.train_config(cfg, 1))
    self.assertEqual(port_history[0]["loss"], history[0]["loss"])
    self.assert_trees("first step's gradients", grads, port_grads)

  def test_moe_block_group_routing(self):
    cfg, model = mr.build(
        **RAGGED_DOT,
        **GROUP_ROUTING,
        routed_bias_update_rate=1e-3,
        load_balance_loss_weight=1e-2,
        moe_use_megatron_seq_aux_loss=True,
    )
    port_cfg = ds_config.from_maxtext(cfg)
    _set_routing_bias(cfg, model)
    block = mr.moe_block(model, cfg)
    port_params = mr.moe_block_params(block)
    # Two sequences, so that the aux loss's per-sequence statistics are not the batch's.
    rng = np.random.default_rng(1)
    shape = (2, int(cfg.max_target_length), cfg.emb_dim)
    x = jnp.asarray(rng.standard_normal(shape), jnp.bfloat16)
    ct = jnp.asarray(rng.standard_normal(shape), jnp.bfloat16)
    out, d_params, d_x, (aux_loss, bias_updates) = mr.maxtext_moe_vjp(cfg, block, x, ct)
    # The port block's load-balance loss is the Switch-style one, not DeepSeek-V3's (see `port_moe_vjp`).
    port_out, port_d_params, port_d_x, (_, port_bias_updates) = mr.port_moe_vjp(port_cfg, port_params, x, ct)
    self.assert_bit_exact("output", out, port_out)
    self.assert_bit_exact("d_inputs", d_x, port_d_x)
    self.assert_trees("d_params", d_params, port_d_params)
    self.assert_bit_exact("routed bias update", bias_updates, port_bias_updates)
    port_aux_loss = _pipeline_aux_loss(port_cfg, port_params["MoeBlock_0"]["gate"], x)
    self.assertLessEqual(_rel(aux_loss, port_aux_loss), AUX_LOSS_RTOL, f"aux loss {port_aux_loss} vs {aux_loss}")

    # The group mask and the bias each decide some token's experts here: the checks above hold both.
    gate = port_params["MoeBlock_0"]["gate"]
    chosen = np.sort(ds_moe.route(gate, x, port_cfg)[1], axis=-1)
    for label, other_gate, other_cfg in (
        ("the group mask", gate, port_cfg.replace(n_routing_groups=-1, topk_routing_group=-1)),
        ("the routing bias", {**gate, "bias": jnp.zeros_like(gate["bias"])}, port_cfg),
    ):
      other = np.sort(ds_moe.route(other_gate, x, other_cfg)[1], axis=-1)
      self.assertFalse(np.array_equal(other, chosen), f"{label} changes no token's experts")


if __name__ == "__main__":
  unittest.main()
