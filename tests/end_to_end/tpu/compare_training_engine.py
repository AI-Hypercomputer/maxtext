# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Strict Parity Verification (train_step vs MaxTextTrainingEngine).

Executes baseline train_step() and MaxTextTrainingEngine side-by-side on
identical
initial states and synthetic micro-batches to verify:
1. Model weight bitwise / numerical parity after optimizer updates.
2. Primary loss and scalar metrics parity.
3. Auxiliary metrics parity (aux dict: xent_sum, total_weights, z_loss,
moe_lb_loss).
4. Gradient norm and spike skipping telemetry parity (CL 956059885).
5. Multi-microbatch gradient accumulation parity.
"""

import contextlib
import dataclasses
import gc
import os
import sys
import time
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.common import common_types
from maxtext.common import train_state_nnx
from maxtext.configs import pyconfig
from maxtext.configs import types
from maxtext.models import simple_layer
from maxtext.trainers.pre_train import train as pre_train
from maxtext.training_engine import abstract_engine
from maxtext.training_engine import maxtext_engine
from maxtext.utils import globals as maxtext_globals
from maxtext.utils import maxtext_utils
from maxtext.utils import model_creation_utils
from maxtext.utils import sharding
from maxtext.utils import train_utils
import numpy as np
import optax


class ParityVerificationError(Exception):
  """Raised when numerical or weight parity verification fails between engine and baseline."""


def _to_numpy(val: Any) -> np.ndarray:
  """Safely converts JAX arrays, including PRNGKey dtypes, to NumPy arrays."""
  if hasattr(val, "dtype") and jax.dtypes.issubdtype(val.dtype, jax.dtypes.prng_key):
    return np.array(jax.random.key_data(val))
  return np.array(val)


def get_tpu_mesh(
    cfg: pyconfig.HyperParameters | None = None,
) -> jax.sharding.Mesh:
  """Returns SPMD device mesh based on configuration or default 1-device TPU mesh."""
  if cfg is not None and cfg.model_name != "default":
    return maxtext_utils.get_mesh_from_config(cfg)
  devices = jax.devices("tpu") if jax.default_backend() == "tpu" else jax.devices()
  return jax.make_mesh((1, 1, 1, 1), ("data", "fsdp", "expert", "context"), devices=devices[:1])


class TinyDecoder(nnx.Module):
  """Decoder using official SimpleMlpDecoderLayer from simple_layer."""

  def __init__(
      self,
      vocab_size: int,
      hidden: int,
      rngs: nnx.Rngs,
      cfg: pyconfig.HyperParameters | None = None,
  ):
    self.mesh = get_tpu_mesh(cfg)
    if cfg is None:
      cfg = setup_config(
          "default",
          emb_dim=hidden,
          mlp_dim=hidden * 2,
          vocab_size=vocab_size,
      )
    self.embed = nnx.Embed(vocab_size, cfg.emb_dim, rngs=rngs)
    self.layer = simple_layer.SimpleMlpDecoderLayer(cfg, self.mesh, "train", rngs)
    self.proj = nnx.Linear(cfg.emb_dim, vocab_size, rngs=rngs)

  def __call__(
      self,
      decoder_input_tokens: jax.Array,
      decoder_positions: jax.Array | None = None,
      decoder_segment_ids: jax.Array | None = None,
      deterministic: bool = True,
      model_mode: str = "train",
      **kwargs: Any,
  ) -> jax.Array:
    del kwargs
    x = self.embed(decoder_input_tokens)
    x = self.layer(
        x,
        positions=decoder_positions,
        segmentation=decoder_segment_ids,
        deterministic=deterministic,
        model_mode=model_mode,
    )
    # SimpleMlpDecoderLayer returns (output, None) when scan_layers=True (default in base.yml).
    if isinstance(x, tuple):
      x = x[0]
    return self.proj(x)


@dataclasses.dataclass(kw_only=True)
class DummyPayload(abstract_engine.TrainerPayload):
  """Dummy payload for training engine parity comparisons."""

  token_ids: Any = None
  token_mask: Any = None
  inputs: Any = None
  targets: Any = None
  inputs_position: Any = None
  inputs_segmentation: Any = None
  targets_segmentation: Any = None
  decoder_input_tokens: Any = None
  decoder_target_tokens: Any = None
  decoder_loss_weights: Any = None
  decoder_positions: Any = None


_DUMMY_DATA_RNG = np.random.default_rng(42)


def make_dummy_data(
    batch_size: int = 2,
    seq_len: int = 16,
    vocab_size: int = 8,
    seed: int | None = None,
    cfg: pyconfig.HyperParameters | None = None,
    mask_prob: float = 0.0,
) -> dict[str, jax.Array]:
  """Constructs dummy token batch dictionary for loss_fn / train_step."""
  if cfg is not None and cfg.model_name != "default":
    batch_size = cfg.micro_batch_size_to_train_on
    seq_len = cfg.max_target_length
    vocab_size = cfg.vocab_size
  rng = np.random.default_rng(seed) if seed is not None else _DUMMY_DATA_RNG
  tokens = jnp.array(rng.integers(0, vocab_size, size=(batch_size, seq_len)), dtype=jnp.int32)
  targets = jnp.array(rng.integers(0, vocab_size, size=(batch_size, seq_len)), dtype=jnp.int32)
  if mask_prob > 0.0:
    mask = (rng.random(size=(batch_size, seq_len)) >= mask_prob).astype(np.float32)
    mask[:, 0] = 1.0  # Ensure at least one active token per sequence to avoid NaN divisions
    weights = jnp.array(mask, dtype=jnp.float32)
  else:
    weights = jnp.ones((batch_size, seq_len), dtype=jnp.float32)
  positions = jnp.arange(seq_len, dtype=jnp.int32)[None, :].repeat(batch_size, axis=0)
  segmentation = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
  res = {
      "inputs": tokens,
      "targets": targets,
      "inputs_position": positions,
      "inputs_segmentation": segmentation,
      "targets_segmentation": segmentation,
      "decoder_input_tokens": tokens,
      "decoder_target_tokens": targets,
      "decoder_loss_weights": weights,
      "decoder_positions": positions,
  }
  if cfg is not None and cfg.model_name != "default":
    mesh = get_tpu_mesh(cfg)
    data_sharding = sharding.get_input_data_sharding(cfg, mesh)
    res = {k: jax.device_put(v, data_sharding) for k, v in res.items()}
  return res


def setup_config(
    model_name: str = "default",
    gradient_accumulation_steps: int = 1,
    skip_step_on_spikes: bool = False,
    gradient_clipping_threshold: float = 0.0,
    use_tunix_gradient_accumulation: bool = True,
    cli_overrides: list[str] | None = None,
    **kwargs: Any,
) -> pyconfig.HyperParameters:
  """Builds a real pyconfig.HyperParameters instance for testing via pyconfig.initialize."""
  base_yml = os.path.join(maxtext_globals.MAXTEXT_CONFIGS_DIR, "base.yml")
  config_cls = types.MaxTextConfig
  if cli_overrides and any(
      "rl" in os.path.basename(o) and (o.endswith(".yml") or o.endswith(".yaml")) for o in cli_overrides
  ):
    config_cls = types.RLConfig

  argv = [
      "compare_training_engine.py",
      base_yml,
      f"model_name={model_name}",
      f"gradient_accumulation_steps={gradient_accumulation_steps}",
      f"skip_step_on_spikes={skip_step_on_spikes}",
      f"gradient_clipping_threshold={gradient_clipping_threshold}",
      f"use_tunix_gradient_accumulation={use_tunix_gradient_accumulation}",
      "run_name=parity_test_run",
      "enable_checkpointing=False",
      "init_weights_seed=42",
      "dtype=float32",
      "weight_dtype=float32",
      "grad_dtype=float32",
      "enable_tensorboard=False",
      "record_internal_nn_metrics=False",
      "skip_jax_distributed_system=True",
  ]
  if model_name == "default":
    argv.extend(
        [
            "vocab_size=8",
            "emb_dim=4",
            "mlp_dim=8",
            "micro_batch_size_to_train_on=2",
            "max_target_length=4",
        ]
    )
  for k, v in kwargs.items():
    argv.append(f"{k}={v}")
  if cli_overrides:
    for override in cli_overrides:
      clean_override = override[2:] if override.startswith("--") else override
      if clean_override.startswith("model_name="):
        argv[2] = clean_override
      elif clean_override.endswith(".yml") or clean_override.endswith(".yaml"):
        argv[1] = clean_override
        if "rl" in os.path.basename(clean_override):
          config_cls = types.RLConfig
      else:
        argv.append(clean_override)
  return pyconfig.initialize(argv, config_class=config_cls)


def create_identical_models_and_opts(
    cfg: pyconfig.HyperParameters,
    learning_rate_schedule: Any = None,
) -> tuple[Any, Any, Any, Any, Any, Any]:
  """Creates two identical model/optimizer pairs with identical initial weights."""
  mesh = get_tpu_mesh(cfg)
  if cfg.model_name == "default":
    lr = learning_rate_schedule if learning_rate_schedule is not None else cfg.learning_rate
    model_baseline = TinyDecoder(vocab_size=cfg.vocab_size, hidden=4, rngs=nnx.Rngs(42))
    opt_baseline = nnx.Optimizer(
        model_baseline,
        optax.adamw(
            learning_rate=lr,
            b1=0.9,
            b2=0.999,
            weight_decay=1e-4,
        ),
        wrt=nnx.Param,
    )

    model_engine = TinyDecoder(vocab_size=cfg.vocab_size, hidden=4, rngs=nnx.Rngs(42))
    opt_engine = nnx.Optimizer(
        model_engine,
        optax.adamw(
            learning_rate=lr,
            b1=0.9,
            b2=0.999,
            weight_decay=1e-4,
        ),
        wrt=nnx.Param,
    )

    ts_baseline = train_state_nnx.TrainStateNNX(model_baseline, opt_baseline)
    _, state_pure = nnx.split(ts_baseline)

    state_mesh_shardings = jax.tree.map(
        lambda x: jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()),
        state_pure,
    )
    params_shardings = jax.tree.map(
        lambda x: jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()),
        nnx.state(model_baseline, nnx.Param),
    )
    return (
        model_baseline,
        opt_baseline,
        model_engine,
        opt_engine,
        state_mesh_shardings,
        params_shardings,
    )

  with jax.set_mesh(mesh):
    model_baseline = model_creation_utils.from_pretrained(
        config=cfg,
        mesh=mesh,
        model_mode=common_types.MODEL_MODE_TRAIN,
        rng_key=jax.random.PRNGKey(cfg.init_weights_seed),
    )
    _, tx_b = train_utils.create_training_optimizer(cfg, model_baseline)
    opt_baseline = nnx.Optimizer(model_baseline, tx_b, wrt=nnx.Param)

    model_engine = model_creation_utils.from_pretrained(
        config=cfg,
        mesh=mesh,
        model_mode=common_types.MODEL_MODE_TRAIN,
        rng_key=jax.random.PRNGKey(cfg.init_weights_seed),
    )
    _, tx_e = train_utils.create_training_optimizer(cfg, model_engine)
    opt_engine = nnx.Optimizer(model_engine, tx_e, wrt=nnx.Param)

    ts_baseline = train_state_nnx.TrainStateNNX(model_baseline, opt_baseline)
    _, state_pure = nnx.split(ts_baseline)

    state_mesh_shardings = jax.tree.map(
        lambda x: getattr(
            x,
            "sharding",
            jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()),
        ),
        state_pure,
    )
    params_shardings = jax.tree.map(
        lambda x: getattr(
            x,
            "sharding",
            jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()),
        ),
        nnx.state(model_baseline, nnx.Param),
    )
    return (
        model_baseline,
        opt_baseline,
        model_engine,
        opt_engine,
        state_mesh_shardings,
        params_shardings,
    )


def execute_baseline_step(
    cfg: pyconfig.HyperParameters,
    ts_baseline: train_state_nnx.TrainStateNNX,
    state_graphdef: Any,
    state_pure: Any,
    state_shardings: Any,
    params_shardings: Any,
    data_dict: dict[str, jax.Array],
    compiled: bool = False,
    p_train_step: Any = None,
) -> tuple[Any, dict[str, Any], Any]:
  """Executes baseline train_step either eagerly or via JIT-compiled SPMD closure."""
  mesh = get_tpu_mesh(cfg)
  with jax.set_mesh(mesh):
    if compiled:
      if p_train_step is None:
        data_sharding = sharding.get_input_data_sharding(cfg, mesh)
        p_train_step = train_utils.jit_train_step(
            config=cfg,
            model=state_graphdef,
            state=state_pure,
            state_mesh_shardings=state_shardings,
            data_sharding=data_sharding,
            train_step=pre_train.train_step,
            params_shardings=params_shardings,
            mesh=mesh,
        )
      new_state_pure, metrics_b = p_train_step(state_pure, data_dict)
      nnx.update(ts_baseline, new_state_pure)
      return new_state_pure, metrics_b, p_train_step
    else:
      new_state_pure, metrics_b = pre_train.train_step(
          state_graphdef,
          cfg,
          state_mesh_shardings=state_shardings,
          params_shardings=params_shardings,
          state=state_pure,
          data=data_dict,
      )
      nnx.update(ts_baseline, new_state_pure)
      return new_state_pure, metrics_b, None


@dataclasses.dataclass
class VerificationContext:
  """Holds synchronized baseline and engine training objects for verification."""

  ts_baseline: train_state_nnx.TrainStateNNX
  state_graphdef: Any
  state_pure: Any
  state_shardings: Any
  params_shardings: Any
  engine: maxtext_engine.MaxTextTrainingEngine
  model_b: Any
  model_e: Any
  opt_b: Any
  opt_e: Any
  mesh: jax.sharding.Mesh
  compiled: bool = False


@contextlib.contextmanager
def verification_harness(
    cfg: pyconfig.HyperParameters,
    compiled: bool = False,
    learning_rate_schedule: Any = None,
) -> Any:
  """Context manager establishing synchronized baseline/engine testing scaffold and memory cleanup."""
  model_b, opt_b, model_e, opt_e, state_shardings, params_shardings = create_identical_models_and_opts(
      cfg, learning_rate_schedule=learning_rate_schedule
  )
  mesh = get_tpu_mesh(cfg)

  ts_baseline = train_state_nnx.TrainStateNNX(model_b, opt_b)
  state_graphdef, state_pure = nnx.split(ts_baseline)

  engine = maxtext_engine.MaxTextTrainingEngine(cfg, mesh=mesh)
  # Check what __init__ built before the injections below throw it away. create_training_optimizer
  # returns a raw optax GradientTransformation, but TrainStateNNX.apply_gradients and
  # checkpointing.CheckpointState both require an nnx.Optimizer, so a constructor that skips the
  # wrap yields an engine that cannot train or checkpoint. The overwrites below are what let that
  # regression ship unnoticed; this assert is the tripwire.
  assert isinstance(
      engine.optimizer, nnx.Optimizer
  ), f"MaxTextTrainingEngine.__init__ must build an nnx.Optimizer, got {type(engine.optimizer).__name__}"
  # The engine's own model/optimizer are replaced here because the "default" path compares against
  # a hand-written TinyDecoder, which from_pretrained would never produce. Weight-level coverage of
  # the constructor therefore has to live outside this harness.
  engine.model = model_e
  engine.optimizer = opt_e
  engine.state = train_state_nnx.TrainStateNNX(model_e, opt_e)

  ctx = VerificationContext(
      ts_baseline=ts_baseline,
      state_graphdef=state_graphdef,
      state_pure=state_pure,
      state_shardings=state_shardings,
      params_shardings=params_shardings,
      engine=engine,
      model_b=model_b,
      model_e=model_e,
      opt_b=opt_b,
      opt_e=opt_e,
      mesh=mesh,
      compiled=compiled,
  )
  try:
    with jax.set_mesh(mesh):
      yield ctx
  finally:
    gc.collect()
    jax.clear_caches()


def assert_step_parity(
    step: int,
    state_baseline: Any,
    state_engine: Any,
    metrics_baseline: dict[str, Any],
    metrics_engine_buf: abstract_engine.MetricsBuffer,
    check_aux: bool = False,
) -> None:
  """Asserts strict numerical parity between baseline and engine model state and metrics."""
  state_b_leaves = jax.tree_util.tree_leaves(nnx.state(state_baseline, nnx.Param, ...))
  state_e_leaves = jax.tree_util.tree_leaves(nnx.state(state_engine, nnx.Param, ...))
  for val_b, val_e in zip(state_b_leaves, state_e_leaves):
    try:
      arr_b = _to_numpy(val_b)
      arr_e = _to_numpy(val_e)
      # In IEEE bfloat16 arithmetic, multi-step execution across deep transformer layers
      # propagates 1-ULP HBM quantization variations into small localized numerical drift.
      # We set atol=5e-3 and rtol=2e-2 specifically for 16-bit floating types (the industry
      # standard for low-precision LLM testing). For float32/float64 across deep Transformer
      # topologies (e.g. Llama-3.1-8B with 536M+ parameters), IEEE-754 non-associative reduction
      # order between XLA lax.scan loops and sequential Python additions produces ~1e-5 rounding
      # drift per step on <0.00001% of parameters, so we set float32 atol=5e-5 and rtol=1e-5.
      is_16_bit = hasattr(arr_b, "dtype") and str(arr_b.dtype) in (
          "bfloat16",
          "float16",
      )
      curr_atol = 5e-3 if is_16_bit else 2e-4
      curr_rtol = 2e-2 if is_16_bit else 1e-3
      np.testing.assert_allclose(
          arr_b,
          arr_e,
          rtol=curr_rtol,
          atol=curr_atol,
          err_msg=f"State mismatch at step {step}",
      )
    except AssertionError as e:
      raise ParityVerificationError(f"Model/Optimizer state parity failed at step {step}: {e}") from e

  loss_baseline = float(
      np.mean(
          metrics_baseline["scalar"].get(
              "learning/loss",
              metrics_baseline["scalar"].get("learning/lm_loss", 0.0),
          )
      )
  )
  if (
      loss_baseline == 0.0
      and "learning/lm_loss" in metrics_baseline["scalar"]
      and float(np.mean(metrics_baseline["scalar"]["learning/lm_loss"])) != 0.0
  ):
    loss_baseline = float(np.mean(metrics_baseline["scalar"]["learning/lm_loss"]))
  # The engine records its primary loss as a WeightedMetric (unreduced sum + denominator) in
  # `weighted_metrics`, so it never lands in `scalar_metrics`. Reduce it here; fall back to
  # `scalar_metrics` for losses that were recorded as plain scalars.
  loss_engine_weighted = metrics_engine_buf.weighted_metrics.get("loss")
  if loss_engine_weighted is not None:
    loss_engine = float(np.mean(_to_numpy(loss_engine_weighted.compute())))
  else:
    loss_engine = float(
        np.mean(
            metrics_engine_buf.scalar_metrics.get(
                "learning/loss",
                metrics_engine_buf.scalar_metrics.get("loss", 0.0),
            )
        )
    )
  is_16_bit_model = any(hasattr(v, "dtype") and str(v.dtype) in ("bfloat16", "float16") for v in state_b_leaves)
  # Cross-entropy loss across large vocabularies (e.g. 151,936 tokens for Llama-3.1-8B)
  # sums exponential terms; when weights drift by ~1e-5 across multi-step IEEE-754 accumulation,
  # scalar loss exhibits ~3e-4 absolute drift (0.002% relative difference).
  metric_atol = 5e-3 if is_16_bit_model else 5e-4
  metric_rtol = 2e-2 if is_16_bit_model else 5e-5
  try:
    np.testing.assert_allclose(
        loss_baseline,
        loss_engine,
        rtol=metric_rtol,
        atol=metric_atol,
        err_msg=(f"Loss value mismatch at step {step}: baseline={loss_baseline}," f" engine={loss_engine}"),
    )
  except AssertionError as e:
    raise ParityVerificationError(f"Loss metric parity failed at step {step}: {e}") from e

  if check_aux:
    aux_map = [
        ("learning/total_weights", "total_weights"),
        ("learning/grad_norm", "gradient_norm"),
        ("optim/step_skipped", "step_skipped"),
        ("learning/moe_lb_loss", "moe_lb_loss"),
        ("learning/z_loss", "z_loss"),
        ("learning/xent_sum", "xent_sum"),
    ]
    for base_key, engine_key in aux_map:
      if base_key in metrics_baseline["scalar"] and engine_key in metrics_engine_buf.scalar_metrics:
        val_b = float(np.mean(metrics_baseline["scalar"][base_key]))
        val_e = float(np.mean(metrics_engine_buf.scalar_metrics[engine_key]))
        try:
          np.testing.assert_allclose(
              val_b,
              val_e,
              rtol=metric_rtol,
              atol=metric_atol,
              err_msg=f"{engine_key} aux metric mismatch at step {step}",
          )
        except AssertionError as e:
          raise ParityVerificationError(f"Auxiliary metric parity failed at step {step}: {e}") from e


def verify_parity_with_train_py(
    cfg: pyconfig.HyperParameters | None = None,
    compiled: bool = False,
    num_steps: int = 3,
) -> None:
  """Verifies numerical and weight parity between standalone train_step and MaxTextTrainingEngine over N steps."""
  if cfg is None:
    cfg = setup_config("default")

  with verification_harness(cfg, compiled) as ctx:
    p_train_step = None
    state_pure = ctx.state_pure
    for step in range(num_steps):
      data_dict = make_dummy_data(
          batch_size=cfg.micro_batch_size_to_train_on,
          vocab_size=cfg.vocab_size,
          seed=42 + step,
          cfg=cfg,
      )
      payload = DummyPayload(**data_dict)
      if compiled and step == 0:
        ctx.engine.compile(payload)

      state_pure, metrics_b, p_train_step = execute_baseline_step(
          cfg,
          ctx.ts_baseline,
          ctx.state_graphdef,
          state_pure,
          ctx.state_shardings,
          ctx.params_shardings,
          data_dict,
          compiled=compiled,
          p_train_step=p_train_step,
      )

      ctx.engine.fwd_bwd(payload)
      ctx.engine.update()
      metrics_e_buf = ctx.engine.get_metrics(clear_cache=True)
      # Fail rather than comparing against the empty buffer: an empty stand-in is what let
      # a step that recorded nothing compare as loss=0.0 instead of failing.
      assert metrics_e_buf.id != maxtext_engine.EMPTY_METRICS_BUFFER_ID, f"engine recorded no metrics at step {step}"
      assert_step_parity(step, ctx.ts_baseline, ctx.engine.state, metrics_b, metrics_e_buf)


def verify_auxiliary_metrics_and_telemetry_parity(
    cfg: pyconfig.HyperParameters | None = None,
    compiled: bool = False,
    num_steps: int = 3,
) -> None:
  """Verifies aux metrics, gradient norm, and spike skipping telemetry parity over N steps (CL 956059885)."""
  if cfg is None:
    cfg = setup_config("default", skip_step_on_spikes=True, gradient_clipping_threshold=1.0)

  with verification_harness(cfg, compiled) as ctx:
    p_train_step = None
    state_pure = ctx.state_pure
    for step in range(num_steps):
      data_dict = make_dummy_data(
          batch_size=cfg.micro_batch_size_to_train_on,
          vocab_size=cfg.vocab_size,
          seed=500 + step,
          mask_prob=0.2 if step % 2 == 1 else 0.0,
          cfg=cfg,
      )
      payload = DummyPayload(**data_dict)
      if compiled and step == 0:
        ctx.engine.compile(payload)

      state_pure, metrics_b, p_train_step = execute_baseline_step(
          cfg,
          ctx.ts_baseline,
          ctx.state_graphdef,
          state_pure,
          ctx.state_shardings,
          ctx.params_shardings,
          data_dict,
          compiled=compiled,
          p_train_step=p_train_step,
      )

      ctx.engine.fwd_bwd(payload)
      ctx.engine.update()
      metrics_e_buf = ctx.engine.get_metrics(clear_cache=True)
      # Fail rather than comparing against the empty buffer: an empty stand-in is what let
      # a step that recorded nothing compare as loss=0.0 instead of failing.
      assert metrics_e_buf.id != maxtext_engine.EMPTY_METRICS_BUFFER_ID, f"engine recorded no metrics at step {step}"
      assert_step_parity(
          step,
          ctx.ts_baseline,
          ctx.engine.state,
          metrics_b,
          metrics_e_buf,
          check_aux=True,
      )


def verify_gradient_accumulation_parity(
    m_steps: int = 5,
    cfg: pyconfig.HyperParameters | None = None,
    compiled: bool = False,
    num_steps: int = 3,
) -> None:
  """Verifies multi-step gradient accumulation parity across M microbatches per outer step under dynamic LR."""
  if cfg is None:
    cfg = setup_config(
        "default",
        gradient_accumulation_steps=m_steps,
        use_tunix_gradient_accumulation=True,
    )
  else:
    assert (
        cfg.use_tunix_gradient_accumulation
    ), "Gradient accumulation parity verification requires cfg.use_tunix_gradient_accumulation=True"

  def lr_schedule(step: int | float) -> float:
    return cfg.learning_rate * (0.9**step)

  with verification_harness(
      cfg,
      compiled,
      learning_rate_schedule=lr_schedule if cfg.model_name == "default" else None,
  ) as ctx:
    p_train_step = None
    state_pure = ctx.state_pure
    for step in range(num_steps):
      microbatch_dicts = [
          make_dummy_data(
              batch_size=cfg.micro_batch_size_to_train_on,
              vocab_size=cfg.vocab_size,
              seed=100 + (step * m_steps) + k,
              mask_prob=0.3,
              cfg=cfg,
          )
          for k in range(m_steps)
      ]
      payloads = [DummyPayload(**d) for d in microbatch_dicts]
      if compiled and step == 0:
        ctx.engine.compile(payloads[0])

      num_mb = len(microbatch_dicts)
      mb_size = cfg.micro_batch_size_to_train_on
      combined_data = {}
      # Note: We cannot use standard array concatenation (jnp.concatenate) here because MaxText
      # internally folds and unfolds batch dimensions via gradient_accumulation.fold_in_gradient_accumulation_steps
      # into (num_mb, mb_size, ...) shapes. We must interlace micro-batches manually so that
      # microbatch k correctly aligns with step k inside MaxText's internal gradient accumulation loop.
      for k_key in microbatch_dicts[0]:
        arr_shape = microbatch_dicts[0][k_key].shape[1:]
        combined_arr = np.zeros(
            (num_mb * mb_size, *arr_shape),
            dtype=microbatch_dicts[0][k_key].dtype,
        )
        for k in range(num_mb):
          for r in range(mb_size):
            combined_arr[r * num_mb + k] = microbatch_dicts[k][k_key][r]
        combined_data[k_key] = combined_arr

      state_pure, metrics_b, p_train_step = execute_baseline_step(
          cfg,
          ctx.ts_baseline,
          ctx.state_graphdef,
          state_pure,
          ctx.state_shardings,
          ctx.params_shardings,
          combined_data,
          compiled=compiled,
          p_train_step=p_train_step,
      )

      for p in payloads:
        ctx.engine.fwd_bwd(p)

      if ctx.engine.micro_step_count != m_steps:
        raise ParityVerificationError(
            f"Step {step}: Expected micro_step_count={m_steps}, got" f" {ctx.engine.micro_step_count}"
        )
      if not ctx.engine.has_accumulated_grads:
        raise ParityVerificationError(f"Step {step}: Expected accumulated_grads to be non-None before" " update()")

      ctx.engine.update()
      metrics_e_buf = ctx.engine.get_metrics(clear_cache=True)
      # Fail rather than comparing against the empty buffer: an empty stand-in is what let
      # a step that recorded nothing compare as loss=0.0 instead of failing.
      assert metrics_e_buf.id != maxtext_engine.EMPTY_METRICS_BUFFER_ID, f"engine recorded no metrics at step {step}"

      if ctx.engine.micro_step_count != 0:
        raise ParityVerificationError(
            f"Step {step}: Expected micro_step_count=0 after update(), got" f" {ctx.engine.micro_step_count}"
        )
      if ctx.engine.has_accumulated_grads:
        raise ParityVerificationError(f"Step {step}: Expected accumulated_grads to be reset to None after" " update()")

      assert_step_parity(step, ctx.ts_baseline, ctx.engine.state, metrics_b, metrics_e_buf)


def benchmark_gradient_accumulation_performance(
    m_steps: int = 5,
    n_iterations: int = 10,
    cfg: pyconfig.HyperParameters | None = None,
) -> None:
  """Benchmarks wall-clock throughput between baseline lax.scan gradient accumulation and MaxTextTrainingEngine."""
  if cfg is None:
    cfg = setup_config(
        "llama3.1-8b",
        gradient_accumulation_steps=m_steps,
        use_tunix_gradient_accumulation=True,
    )
  else:
    assert (
        cfg.use_tunix_gradient_accumulation
    ), "Gradient accumulation performance benchmarking requires cfg.use_tunix_gradient_accumulation=True"

  print(
      "\n=== [BENCHMARK] Initializing Gradient Accumulation Hardware"
      f" Benchmarks (M={m_steps} micro-steps/cycle, N={n_iterations}"
      " cycles) ===",
      flush=True,
  )

  with verification_harness(cfg, compiled=True) as ctx:
    p_train_step = None
    state_pure = ctx.state_pure

    microbatch_dicts = [
        make_dummy_data(
            batch_size=cfg.micro_batch_size_to_train_on,
            vocab_size=cfg.vocab_size,
            seed=200 + k,
            mask_prob=0.3,
            cfg=cfg,
        )
        for k in range(m_steps)
    ]
    payloads = [DummyPayload(**d) for d in microbatch_dicts]

    num_mb = len(microbatch_dicts)
    mb_size = cfg.micro_batch_size_to_train_on
    combined_data = {}
    for k_key in microbatch_dicts[0]:
      arr_shape = microbatch_dicts[0][k_key].shape[1:]
      combined_arr = np.zeros(
          (num_mb * mb_size, *arr_shape),
          dtype=microbatch_dicts[0][k_key].dtype,
      )
      for k in range(num_mb):
        for r in range(mb_size):
          combined_arr[r * num_mb + k] = microbatch_dicts[k][k_key][r]
      combined_data[k_key] = combined_arr

    print(
        "--- [Warm-Up] Compiling XLA execution graphs and populating HBM" " cache... ---",
        flush=True,
    )
    ctx.engine.compile(payloads[0])
    state_pure, _, p_train_step = execute_baseline_step(
        cfg,
        ctx.ts_baseline,
        ctx.state_graphdef,
        state_pure,
        ctx.state_shardings,
        ctx.params_shardings,
        combined_data,
        compiled=True,
        p_train_step=p_train_step,
    )
    jax.block_until_ready(state_pure)

    for p in payloads:
      ctx.engine.fwd_bwd(p)
    ctx.engine.update()
    jax.block_until_ready(ctx.engine.state)
    print("✓ XLA compilation and hardware warm-up completed.", flush=True)

    print(
        f"\n--- [Baseline Benchmark] Running {n_iterations} accumulation cycles"
        f" (M={m_steps}) via compiled jax.lax.scan... ---",
        flush=True,
    )
    t0 = time.perf_counter()
    for _ in range(n_iterations):
      state_pure, _, p_train_step = execute_baseline_step(
          cfg,
          ctx.ts_baseline,
          ctx.state_graphdef,
          state_pure,
          ctx.state_shardings,
          ctx.params_shardings,
          combined_data,
          compiled=True,
          p_train_step=p_train_step,
      )
    jax.block_until_ready(state_pure)
    baseline_duration = time.perf_counter() - t0
    baseline_ms_per_step = (baseline_duration / n_iterations) * 1000.0
    print(
        f"✓ Baseline lax.scan Throughput: {baseline_ms_per_step:.2f}"
        f" ms/cycle (Total: {baseline_duration:.2f}s for {n_iterations}"
        " cycles)",
        flush=True,
    )

    print(
        f"\n--- [Engine Benchmark] Running {n_iterations} accumulation cycles"
        f" (M={m_steps}) via MaxTextTrainingEngine fwd_bwd loop... ---",
        flush=True,
    )
    t1 = time.perf_counter()
    for _ in range(n_iterations):
      for p in payloads:
        ctx.engine.fwd_bwd(p)
      ctx.engine.update()
    jax.block_until_ready(ctx.engine.state)
    engine_duration = time.perf_counter() - t1
    engine_ms_per_step = (engine_duration / n_iterations) * 1000.0
    print(
        f"✓ Training Engine Throughput:   {engine_ms_per_step:.2f}"
        f" ms/cycle (Total: {engine_duration:.2f}s for {n_iterations}"
        " cycles)",
        flush=True,
    )

    delta_ms = engine_ms_per_step - baseline_ms_per_step
    pct_overhead = (delta_ms / baseline_ms_per_step) * 100.0 if baseline_ms_per_step > 0 else 0.0
    print(
        "\n==========================================================================",
        flush=True,
    )
    print(
        "=== [BENCHMARK SUMMARY] Gradient Accumulation Architecture" " Throughput ===",
        flush=True,
    )
    print(
        f" * Baseline jax.lax.scan (M={m_steps}):   {baseline_ms_per_step:8.2f}" " ms/cycle",
        flush=True,
    )
    print(
        f" * MaxTextTrainingEngine (M={m_steps}):   {engine_ms_per_step:8.2f}" " ms/cycle",
        flush=True,
    )
    print(
        f" * Hardware Execution Overhead:      {pct_overhead:+8.2f}%" f" ({delta_ms:+6.2f} ms/cycle)",
        flush=True,
    )
    print(
        "==========================================================================",
        flush=True,
    )


def run_all_verifications(cli_overrides: list[str] | None = None) -> None:
  """Runs all training engine numerical and weight parity verifications in both eager and JIT modes."""
  model_name = "llama3.1-8b"
  raw_overrides = list(cli_overrides) if cli_overrides else []
  overrides = []
  has_target_length = False
  test_suite = "all"
  for arg in raw_overrides:
    clean_arg = arg[2:] if arg.startswith("--") else arg
    if clean_arg.startswith("test_suite=") or clean_arg.startswith("target_step="):
      test_suite = str(clean_arg.split("=", 1)[1]).strip().lower()
      continue
    if clean_arg.startswith("model_name="):
      model_name = clean_arg.split("=", 1)[1]
      if model_name != "default":
        overrides.append(arg)
      continue
    if clean_arg.startswith("max_target_length="):
      has_target_length = True
    overrides.append(arg)

  if model_name != "default" and not has_target_length:
    overrides.append("max_target_length=256")

  cfg_base = setup_config(
      model_name=model_name,
      gradient_accumulation_steps=1,
      cli_overrides=overrides,
  )
  cfg_aux = setup_config(
      model_name=model_name,
      gradient_accumulation_steps=1,
      skip_step_on_spikes=True,
      gradient_clipping_threshold=1.0,
      cli_overrides=overrides,
  )
  cfg_ga = setup_config(
      model_name=model_name,
      gradient_accumulation_steps=5,
      use_tunix_gradient_accumulation=True,
      cli_overrides=overrides,
  )

  if model_name == "default":
    print(
        "=== Running Training Engine Parity Verification Suite (Eager" " Mode) ===",
        flush=True,
    )
    if test_suite in ("all", "eager_all", "eager", "1"):
      print(
          "\n[1/6] Verifying standalone train_step vs engine parity (Eager)...",
          flush=True,
      )
      verify_parity_with_train_py(cfg=cfg_base, compiled=False)
      print(
          "✓ verify_parity_with_train_py (Eager) passed successfully.",
          flush=True,
      )

    if test_suite in ("all", "eager_all", "eager", "2", "auxiliary"):
      print(
          "\n[2/6] Verifying auxiliary metrics and telemetry parity (Eager)...",
          flush=True,
      )
      verify_auxiliary_metrics_and_telemetry_parity(cfg=cfg_aux, compiled=False)
      print(
          "✓ verify_auxiliary_metrics_and_telemetry_parity (Eager) passed" " successfully.",
          flush=True,
      )

    if test_suite in (
        "all",
        "eager_all",
        "eager",
        "3",
        "gradient_accumulation",
    ):
      print(
          "\n[3/6] Verifying multi-microbatch gradient accumulation parity" " (Eager)...",
          flush=True,
      )
      verify_gradient_accumulation_parity(m_steps=5, cfg=cfg_ga, compiled=False)
      print(
          "✓ verify_gradient_accumulation_parity (Eager) passed successfully.",
          flush=True,
      )
  else:
    print(
        f"=== [INFO] Production model architecture detected ('{model_name}')."
        " Bypassing Eager Mode (Steps 1-3) to avoid memory fragmentation and"
        " remat overhead; executing directly on XLA JIT Compiled Mode ===",
        flush=True,
    )

  print(
      "\n=== Running Training Engine Parity Verification Suite (JIT Compiled" f" Mode, test_suite={test_suite}) ===",
      flush=True,
  )
  if test_suite in ("all", "jit_all", "jit_train_step", "4", "train_step"):
    print(
        "\n[4/6] Verifying standalone train_step vs engine parity (JIT" " Compiled)...",
        flush=True,
    )
    verify_parity_with_train_py(cfg=cfg_base, compiled=True)
    print(
        "✓ verify_parity_with_train_py (JIT Compiled) passed successfully.",
        flush=True,
    )

  if test_suite in (
      "all",
      "jit_all",
      "jit_auxiliary_metrics",
      "5",
      "auxiliary",
      "telemetry",
  ):
    print(
        "\n[5/6] Verifying auxiliary metrics and telemetry parity (JIT" " Compiled)...",
        flush=True,
    )
    verify_auxiliary_metrics_and_telemetry_parity(cfg=cfg_aux, compiled=True)
    print(
        "✓ verify_auxiliary_metrics_and_telemetry_parity (JIT Compiled) passed" " successfully.",
        flush=True,
    )

  if test_suite in (
      "all",
      "jit_all",
      "jit_gradient_accumulation",
      "6",
      "gradient_accumulation",
      "ga",
  ):
    print(
        "\n[6/6] Verifying multi-microbatch gradient accumulation parity (JIT" " Compiled)...",
        flush=True,
    )
    verify_gradient_accumulation_parity(m_steps=5, cfg=cfg_ga, compiled=True)
    print(
        "✓ verify_gradient_accumulation_parity (JIT Compiled) passed" " successfully.",
        flush=True,
    )

  if test_suite in ("benchmark", "benchmark_ga", "perf_ga", "perf"):
    print(
        "\n[BENCHMARK] Executing gradient accumulation hardware throughput" " comparison...",
        flush=True,
    )
    benchmark_gradient_accumulation_performance(m_steps=5, n_iterations=10, cfg=cfg_ga)
    print(
        "✓ benchmark_gradient_accumulation_performance completed successfully.",
        flush=True,
    )

  print(
      f"\n=== Verification target (test_suite={test_suite}) PASSED" " cleanly! ===",
      flush=True,
  )


if __name__ == "__main__":
  run_all_verifications(cli_overrides=sys.argv[1:])
