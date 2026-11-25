# Copyright 2023–2026 Google LLC
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

"""Minimal m3 training loop for diffusion transformers, with the flow-matching loss.

The diffusion counterpart of `loop.py`: the model takes noisy latents, timesteps and text embeddings and predicts
the flow-matching velocity (`FlaxFlowMatchScheduler`, as in MaxDiffusion's `WanTrainer`). The VAE and text encoder
stay frozen and are not run: training reads precomputed `{latents, encoder_hidden_states}` (MaxDiffusion-format
TFRecords, `dataset_type=tfrecord`) or synthetic batches of the real shapes (`synthetic`; `synthetic_repeat`
repeats one batch, an overfitting check).

The loop is model-agnostic except for the transformer kwargs; `config.model_name` selects the model through the
existing diffusion registries:

- `DIFFUSERS_PARAM_MAPPING[model_name].COMPONENTS` names the transformer and VAE classes. The batch shapes come from
  their static `vae_cls.latent_shape(vae_config, ...)` and `transformer_cls.text_embedding_shape(...)`.
- the transformer kwargs come from the Wan pipeline's `_wan_model_kwargs` (TODO: a model-agnostic pipeline hook), so
  training and sampling build the same model; the loop adds `remat_policy` (`REMAT_POLICIES`).

The transformer is restored from a converted checkpoint (`checkpoint_directory`, written by
`checkpoint_conversion/to_maxtext_diffusers.py`) as `weights_dtype` master weights, sharded directly onto the mesh,
or randomly initialized from the yml's `transformer_config`. At the end it is exported, with the source checkpoint's
other components (VAE, text encoder, scheduler), under `<output_dir>/export`, so
`sample.py checkpoint_directory=<output_dir>/export` runs on it.

Fields read: `max_train_steps`, `learning_rate`, the `trainer.optimizer` fields (`adam_*`, `warmup_steps_fraction`,
`learning_rate_schedule_steps`, gradient clipping), `per_device_batch_size`, `height` / `width` / `num_frames`,
`max_sequence_length`, `dataset_type` / `train_data_dir`, `log_period`, `seed`, `remat_policy`, the mesh fields
(`ici_*` / `dcn_*_parallelism`, `core.sharding.mesh_from_config`), the model runtime fields (`_wan_model_kwargs`)
and the profiler fields (`enable_profiler`, `skip_first_n_steps_for_profiler`,
`profiler_steps`; the trace is written under `tensorboard_dir`).
`gradient_accumulation_steps > 1` and batch sizes that do not split over the mesh are rejected rather than emulated.

Example (Wan 2.1 14B on one TPU v6e-8 host; float32 master weights fit with unscanned layers)::

    python3 -m maxtext.m3.train.loop_diffusers src/maxtext/m3/models/wan/configs/wan2.1-t2v-14b.yml \\
      checkpoint_directory=gs://<bucket>/<converted checkpoint> output_dir=gs://<bucket>/<run> \\
      weights_dtype=float32 scan_layers=false remat_policy=FULL ici_fsdp_parallelism=8 ici_context_parallelism=1 \\
      height=256 width=448 num_frames=17 max_train_steps=10
"""

import functools
import json
import math
import os
import time
from collections.abc import Callable, Sequence
from typing import Literal

from absl import app
from etils import epath
from flax import nnx
from flax.traverse_util import unflatten_dict
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
import optax
import orbax.checkpoint as ocp
from pydantic import ConfigDict, Field

from maxtext.checkpoint_conversion.to_maxtext_diffusers import (
    _get_item_from_composite,
    _import_class,
    create_diffusers_checkpoint_manager,
    save_diffusers_checkpoint,
)
from maxtext.checkpoint_conversion.utils.param_mapping import DIFFUSERS_PARAM_MAPPING
from maxtext.m3.configs import config as config_lib
from maxtext.m3.core.sharding import mesh_from_config
from maxtext.m3.diffuser_pipelines.pipeline_utils import Profiler, _to_int_path, profiler_enabled

# TODO: Wan-specific model kwargs; move them behind a pipeline / model hook (e.g. a `DiffusionPipeline.transformer_kwargs`
# classmethod, with `state_shardings` in `pipeline_utils`) when a second model family is trained.
from maxtext.m3.diffuser_pipelines.wan_pipeline import _wan_model_kwargs, model_init_config_for_wan, state_shardings
from maxtext.m3.models.diffuser_schedulers.scheduling_flow_match import FlaxFlowMatchScheduler
from maxtext.utils import max_logging

# Mesh axes the batch dimension is split over (the `activation_batch` rule of the diffusion ymls).
_BATCH_AXES = ("data", "fsdp")

# `remat_policy` -> the `jax.checkpoint` policy of each transformer layer (None: no remat).
# TODO: replace with `maxtext.m3.infra.remat.remat_policy(...)` once #6249 (generic m3 remat, same policy names plus
# custom per-tensor save / offload) lands; #6111 (Wan remat PR6) is superseded by #6249 + this trainer PR.
REMAT_POLICIES = {
    "NONE": None,
    "FULL": jax.checkpoint_policies.nothing_saveable,
    "MATMUL_WITHOUT_BATCH": jax.checkpoint_policies.dots_with_no_batch_dims_saveable,
}


def remat_policy(name: str):
  """The `jax.checkpoint` policy for `config.remat_policy` (case-insensitive), or None for no remat."""
  if name.upper() not in REMAT_POLICIES:
    raise ValueError(f"loop_diffusers supports remat_policy in {sorted(REMAT_POLICIES)}; got {name!r}.")
  return REMAT_POLICIES[name.upper()]


class TrainKeys(config_lib.OptimizerConfig):
  """The training keys the diffusion config does not expose as typed fields, validated (fail-fast) from its flat keys.

  The optimizer fields (`adam_*`, clipping, warmup) are `OptimizerConfig`'s; flat `key=value` overrides do not reach
  `config.trainer.optimizer` (nested), so they are read here.
  TODO: declare these in `m3/configs/diffusers_config.py` and read them from `config` directly.
  """

  model_config = ConfigDict(extra="ignore")

  dataset_type: Literal["synthetic", "synthetic_repeat", "tfrecord"] = "synthetic"
  gradient_accumulation_steps: int = 1
  skip_first_n_steps_for_profiler: int = 0
  profiler_steps: int = 1
  # The architecture for training from scratch (a dict, or JSON on the command line).
  transformer_config: dict | str = Field(default_factory=dict)
  vae_config: dict | str = Field(default_factory=dict)

  @classmethod
  def from_config(cls, config) -> "TrainKeys":
    return cls.model_validate(config.get_keys())


def make_optimizer(
    trainer: config_lib.TrainerConfig, opt: config_lib.OptimizerConfig
) -> tuple[optax.GradientTransformation, optax.Schedule]:
  """AdamW with linear warmup to `learning_rate` then constant (MaxDiffusion's WarmupHold), optional clipping."""
  schedule_steps = opt.learning_rate_schedule_steps if opt.learning_rate_schedule_steps > 0 else trainer.max_train_steps
  warmup_steps = int(schedule_steps * opt.warmup_steps_fraction)
  schedule = optax.constant_schedule(trainer.learning_rate)
  if warmup_steps > 0:
    warmup = optax.linear_schedule(0.0, trainer.learning_rate, warmup_steps)
    schedule = optax.join_schedules([warmup, schedule], [warmup_steps])
  tx = optax.adamw(schedule, b1=opt.adam_b1, b2=opt.adam_b2, eps=opt.adam_eps, weight_decay=opt.adam_weight_decay)
  if opt.opt_enable_grad_global_norm_clipping:
    tx = optax.chain(optax.clip_by_global_norm(opt.max_grad_norm), tx)
  if opt.opt_enable_grad_clipping:
    tx = optax.chain(optax.clip(opt.max_grad_value), tx)
  return tx, schedule


# ==============================================================================
# Data: precomputed latents + text embeddings
# ==============================================================================


@functools.lru_cache(maxsize=8)
def _synthetic_batch_fn(latent_shape: tuple, text_shape: tuple, mesh: Mesh):
  """Jitted random batch of the real shapes, generated on device and sharded along the batch dimension."""

  def body(key: jax.Array) -> dict[str, jax.Array]:
    latent_key, text_key = jax.random.split(key)
    return {
        "latents": jax.random.normal(latent_key, latent_shape, jnp.float32),
        "encoder_hidden_states": jax.random.normal(text_key, text_shape, jnp.float32),
    }

  return jax.jit(body, out_shardings=NamedSharding(mesh, P(_BATCH_AXES)))


def tfrecord_batches(data_dir: str, batch_size: int, seed: int):
  """MaxDiffusion-format TFRecords (serialized float32 `latents` / `encoder_hidden_states`), in a seeded order."""
  import tensorflow as tf  # pylint: disable=import-outside-toplevel  # only needed for TFRecord datasets

  tf.config.set_visible_devices([], "GPU")
  files = sorted(tf.io.gfile.glob(os.path.join(data_dir, "*.tfrec*")))
  if not files:
    raise FileNotFoundError(f"No TFRecord files found in {data_dir}")
  features = {
      "latents": tf.io.FixedLenFeature([], tf.string),
      "encoder_hidden_states": tf.io.FixedLenFeature([], tf.string),
  }

  def parse(example):
    parsed = tf.io.parse_single_example(example, features)
    return {k: tf.io.parse_tensor(v, out_type=tf.float32) for k, v in parsed.items()}

  options = tf.data.Options()
  options.deterministic = True
  ds = tf.data.TFRecordDataset(files).with_options(options).map(parse, num_parallel_calls=tf.data.AUTOTUNE)
  ds = ds.shuffle(1024, seed=seed).repeat().batch(batch_size, drop_remainder=True)
  yield from ds.prefetch(tf.data.AUTOTUNE).as_numpy_iterator()


def make_batch_fn(config, dataset_type: str, mesh: Mesh, latent_shape: tuple, text_shape: tuple) -> Callable[[int], dict]:
  """`step -> batch` of global arrays sharded along the batch dimension, for `dataset_type`."""
  key = jax.random.key(config.seed)
  if dataset_type in ("synthetic", "synthetic_repeat"):
    fn = _synthetic_batch_fn(latent_shape, text_shape, mesh)
    return lambda step: fn(key if dataset_type == "synthetic_repeat" else jax.random.fold_in(key, step))
  batches = tfrecord_batches(config.train_data_dir, latent_shape[0], config.seed)
  sharding = NamedSharding(mesh, P(_BATCH_AXES))

  def to_global(_):
    # Every process reads the same batch; each device takes its shard (multi-host safe).
    host = next(batches)
    return {k: jax.make_array_from_callback(v.shape, sharding, lambda idx, v=v: v[idx]) for k, v in host.items()}

  return to_global


# ==============================================================================
# Loss and train step
# ==============================================================================


def loss_fn(model: nnx.Module, batch: dict, rng: jax.Array, scheduler: FlaxFlowMatchScheduler) -> jax.Array:
  """Weighted flow-matching MSE between the predicted and the target velocity, in float32."""
  t_rng, n_rng, d_rng = jax.random.split(rng, 3)
  latents = batch["latents"]
  timesteps = scheduler.sample_timesteps(t_rng, latents.shape[0])
  noise = jax.random.normal(n_rng, latents.shape, latents.dtype)
  noisy_latents, target, weight = scheduler.apply_flow_match(noise, latents, timesteps)
  pred = model(
      noisy_latents, timesteps, batch["encoder_hidden_states"], deterministic=False, rngs=nnx.Rngs(dropout=d_rng)
  )
  per_sample = jnp.mean(jnp.square(target - pred.astype(jnp.float32)), axis=tuple(range(1, pred.ndim)))
  return jnp.mean(per_sample * weight)


def make_train_step(scheduler: FlaxFlowMatchScheduler, schedule: optax.Schedule, param_shardings, opt_shardings):
  """`(model, optimizer, batch, rng) -> metrics`, one forward/backward/update; mutates `model` and `optimizer`.

  The updated params / optimizer state are constrained to their input shardings, so the second step sees the same
  input shardings as the first and does not recompile.
  """

  @nnx.jit(donate_argnums=(0, 1))
  def train_step(model, optimizer, batch, rng):
    loss, grads = nnx.value_and_grad(loss_fn)(model, batch, rng, scheduler)
    optimizer.update(model, grads)
    nnx.update(model, jax.lax.with_sharding_constraint(nnx.state(model, nnx.Param), param_shardings))
    nnx.update(optimizer, jax.lax.with_sharding_constraint(nnx.state(optimizer), opt_shardings))
    return {"loss": loss, "grad_norm": optax.tree.norm(grads), "learning_rate": schedule(optimizer.step[...] - 1)}

  return train_step


# ==============================================================================
# Model and checkpoints (the converted `<component>_state` / `<component>_config` layout)
# ==============================================================================


def _latest(checkpoint_dir: str, name: str) -> tuple[ocp.CheckpointManager, int]:
  """A manager for item `name` of a converted checkpoint, and its latest step."""
  manager = create_diffusers_checkpoint_manager(checkpoint_dir, use_async=False, item_names=(name,))
  step = manager.latest_step()
  if step is None:
    raise FileNotFoundError(f"No checkpoint step found in {checkpoint_dir}")
  return manager, step


def restore_config(checkpoint_dir: str, component: str) -> dict | None:
  """The `<component>_config` JSON of a converted checkpoint (no weights read); None if the item is absent."""
  name = f"{component}_config"
  manager, step = _latest(checkpoint_dir, name)
  if not (epath.Path(checkpoint_dir) / str(step) / name).exists():
    return None
  restored = manager.restore(step, args=ocp.args.Composite(**{name: ocp.args.JsonRestore()}))[name]
  return {k: v for k, v in dict(restored).items() if not k.startswith("_")}


def _param_path(path) -> tuple:
  """Checkpoint tree path -> nnx state path (list indices are strings on disk; legacy trees end in `value`)."""
  keys = _to_int_path(tuple(getattr(p, "key", p) for p in path))
  return keys[:-1] if keys and keys[-1] == "value" else keys


def restore_transformer_params(checkpoint_dir: str, param_shardings: nnx.State, dtype) -> nnx.State:
  """`transformer_state`, read sharded directly onto the mesh (each host reads only its shards) as `dtype`.

  `param_shardings` is the model's `nnx.Param` state with a `NamedSharding` per leaf; the result has the same
  structure, ready for `nnx.update`.
  """
  by_path = {path: s.get_value() if isinstance(s, nnx.Variable) else s for path, s in nnx.to_flat_state(param_shardings)}
  manager, step = _latest(checkpoint_dir, "transformer_state")
  abstract = jax.tree_util.tree_map_with_path(
      lambda path, meta: jax.ShapeDtypeStruct(meta.shape, dtype, sharding=by_path[_param_path(path)]),
      _get_item_from_composite(manager.item_metadata(step), "transformer_state"),
  )
  restored = manager.restore(step, args=ocp.args.Composite(transformer_state=ocp.args.StandardRestore(abstract)))
  values = {_param_path(path): v for path, v in jax.tree_util.tree_flatten_with_path(restored["transformer_state"])[0]}
  missing = [path for path in by_path if path not in values]
  if missing:
    raise KeyError(f"transformer_state in {checkpoint_dir} lacks {len(missing)} model params, e.g. {missing[:3]}")
  return nnx.from_flat_state(
      [(path, var.replace(value=values[path])) for path, var in nnx.to_flat_state(param_shardings)]
  )


def create_model(config, transformer_cls: type, kwargs: dict, mesh: Mesh, checkpoint_dir: str) -> nnx.Module:
  """`transformer_cls(**kwargs)` built directly on its mesh shardings: restored from `checkpoint_dir`, or randomly
  initialized."""
  graphdef, abstract_state = nnx.split(nnx.eval_shape(lambda: transformer_cls(rngs=nnx.Rngs(0), **kwargs)))
  shardings = state_shardings(abstract_state, mesh)
  # Flat leaves: Variable metadata (e.g. init fns) differs between traces, so whole-state out_shardings mismatch.
  sharding_leaves = jax.tree.leaves(shardings, is_leaf=lambda x: isinstance(x, NamedSharding))
  init_leaves = jax.jit(
      lambda: jax.tree.leaves(nnx.state(transformer_cls(rngs=nnx.Rngs(config.seed), **kwargs))),
      out_shardings=sharding_leaves,
  )
  model = nnx.merge(graphdef, jax.tree.unflatten(jax.tree.structure(abstract_state), init_leaves()))
  if checkpoint_dir:
    params = restore_transformer_params(checkpoint_dir, nnx.filter_state(shardings, nnx.Param), config.weights_dtype)
    nnx.update(model, params)
    max_logging.log(f"Restored transformer params from {checkpoint_dir}")
  return model


def place_state(obj, state_filter, mesh: Mesh):
  """Puts `obj`'s state on its mesh shardings (single-device leaves are replicated) and returns them."""
  state = nnx.state(obj, *state_filter)
  replicated = NamedSharding(mesh, P())
  shardings = jax.tree.map(
      lambda x: x.sharding if isinstance(getattr(x, "sharding", None), NamedSharding) else replicated, state
  )
  nnx.update(obj, jax.device_put(state, shardings))
  return shardings


def export_for_inference(export_dir: str, model, transformer_config: dict, source_checkpoint_dir: str, config) -> None:
  """Writes the trained transformer in the converted layout `sample.py` reads, plus the source checkpoint's other
  items (VAE, text encoder, scheduler) when there is one.

  Each param is stored in the dtype the source checkpoint stored it in (e.g. norms in float32, the rest in the
  converter's `weights_dtype`), since the inference pipeline expects the checkpoint's dtypes; without a source
  checkpoint, in the converter's dtype for `weights_dtype` (`param_mapping.<model>.param_dtype`).
  """
  if source_checkpoint_dir:
    manager, step = _latest(source_checkpoint_dir, "transformer_state")
    metadata = _get_item_from_composite(manager.item_metadata(step), "transformer_state")
    source_dtypes = {_param_path(path): meta.dtype for path, meta in jax.tree_util.tree_flatten_with_path(metadata)[0]}
  else:
    source_dtypes = {}
  mapping = DIFFUSERS_PARAM_MAPPING[config.model_name]

  def stored(path: tuple, value: jax.Array) -> jax.Array:
    dtype = source_dtypes.get(path) or mapping.param_dtype("transformer", "-".join(map(str, path)), config.weights_dtype)
    return value if dtype is None else value.astype(dtype)

  params = nnx.to_flat_state(nnx.state(model, nnx.Param))
  state = unflatten_dict({tuple(map(str, path)): stored(path, var[...]) for path, var in params})
  save_diffusers_checkpoint(export_dir, {"transformer": state}, {"transformer": transformer_config})
  if not source_checkpoint_dir or jax.process_index() != 0:
    return
  source_step = create_diffusers_checkpoint_manager(source_checkpoint_dir, use_async=False).latest_step()

  def copy_tree(src: epath.Path, dst: epath.Path) -> None:  # explicit walk: `rglob("**")` is rejected on GCS
    dst.mkdir(parents=True, exist_ok=True)
    for entry in src.iterdir():
      if entry.is_dir():
        copy_tree(entry, dst / entry.name)
      else:
        entry.copy(dst / entry.name, overwrite=True)

  for item in (epath.Path(source_checkpoint_dir) / str(source_step)).iterdir():
    if item.is_dir() and not item.name.startswith("transformer_"):
      copy_tree(item, epath.Path(export_dir) / "0" / item.name)


# ==============================================================================
# Training loop
# ==============================================================================


def _log_step(
    step: int, metrics: dict, last: tuple[int, float], samples_per_step: int
) -> tuple[float, tuple[int, float]]:
  """Blocks until `step` finishes, logs it, and returns its (window-mean) loss and `(step, completion time)`.

  `last` is the previously logged `(step, completion time)`; step time and `metrics["loss"]` are means over the steps
  in between.
  """
  metrics = {k: float(v) for k, v in jax.device_get(metrics).items()}
  done = time.perf_counter()
  step_time = (done - last[1]) / (step - last[0])
  heap = (jax.local_devices()[0].memory_stats() or {}).get("peak_bytes_in_use", 0) / 2**30
  max_logging.log(
      f"step {step}: mean loss {metrics['loss']:.5f}, grad_norm {metrics['grad_norm']:.4f}, "
      f"lr {metrics['learning_rate']:.3e}, {step_time:.3f}s, {samples_per_step / step_time:.2f} samples/s, "
      f"peak heap {heap:.2f} GiB"
  )
  return metrics["loss"], (step, done)


def train_loop(config) -> float:
  """Builds (restores or initializes) the transformer, runs `max_train_steps` steps, exports, returns the last loss."""
  output_dir = config.output_dir or config.base_output_directory
  if not output_dir:
    raise ValueError("output_dir is required.")
  if not config.tensorboard_dir:
    config.tensorboard_dir = os.path.join(output_dir, "tensorboard", "")
  keys = TrainKeys.from_config(config)
  if keys.gradient_accumulation_steps > 1:
    raise ValueError(
        "loop_diffusers does not accumulate gradients; got "
        f"gradient_accumulation_steps={keys.gradient_accumulation_steps}."
    )

  mesh = mesh_from_config(config)
  checkpoint_dir = config.checkpoint_directory or ""
  # The yml's `transformer_config` / `vae_config` blocks (or JSON command-line overrides); a checkpoint's take
  # precedence, as they fix the architecture.
  transformer_config, vae_config = (
      dict(json.loads(v) if isinstance(v, str) else v) for v in (keys.transformer_config, keys.vae_config)
  )
  if checkpoint_dir:
    transformer_config = restore_config(checkpoint_dir, "transformer")
    vae_config = restore_config(checkpoint_dir, "vae") or vae_config  # e.g. an export of a from-scratch run
  elif not transformer_config:
    raise ValueError("Training from scratch needs `transformer_config` (the architecture) in the config.")
  components = DIFFUSERS_PARAM_MAPPING[config.model_name].COMPONENTS
  transformer_cls, vae_cls = _import_class(components["transformer"]), _import_class(components["vae"])
  batch = int(config.global_batch_size_to_train_on)
  latent_shape = vae_cls.latent_shape(vae_config, batch, config.num_frames, config.height, config.width)
  text_shape = transformer_cls.text_embedding_shape(transformer_config, batch, config.max_sequence_length)
  if latent_shape[1] != transformer_config.get("in_channels", latent_shape[1]):
    raise ValueError(
        f"The VAE makes {latent_shape[1]}-channel latents but the transformer takes in_channels="
        f"{transformer_config['in_channels']}; check `vae_config` / `transformer_config`."
    )
  batch_shards = math.prod(mesh.shape[axis] for axis in _BATCH_AXES)
  if batch == 0 or batch % batch_shards:
    raise ValueError(
        f"per_device_batch_size={config.per_device_batch_size} gives a global batch of {batch}, which does "
        f"not split over the {batch_shards} shards of mesh axes {_BATCH_AXES}."
    )
  kwargs = _wan_model_kwargs(
      config, model_init_config_for_wan(transformer_config), ckpt_scan_layers=transformer_config.get("scan_layers")
  )
  kwargs["remat_policy"] = remat_policy(config.remat_policy)
  max_train_steps = config.trainer.max_train_steps
  log_period = max(config.trainer.log_period, 1)

  with jax.set_mesh(mesh):
    model = create_model(config, transformer_cls, kwargs, mesh, checkpoint_dir)
    tx, schedule = make_optimizer(config.trainer, keys)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    num_params = sum(x.size for x in jax.tree.leaves(nnx.state(model, nnx.Param)))
    max_logging.log(
        f"loop_diffusers: {num_params / 1e9:.2f}B params, latents {latent_shape}, text {text_shape}, "
        f"mesh {dict(mesh.shape)}, weights {config.weights_dtype}, activations {config.activations_dtype}"
    )

    scheduler, _ = FlaxFlowMatchScheduler.from_config({}, dtype=jnp.float32)
    train_step = make_train_step(
        scheduler, schedule, place_state(model, (nnx.Param,), mesh), place_state(optimizer, (), mesh)
    )
    # Walk the model and optimizer graphs once, so `nnx.jit` does not split and merge them on every step.
    step_fn = nnx.cached_partial(train_step, model, optimizer)
    batch_fn = make_batch_fn(config, keys.dataset_type, mesh, latent_shape, text_shape)
    rng = jax.random.key(config.seed)
    # xprof window: skip the first steps (compilation), then trace `profiler_steps` steps.
    first_profiled = min(keys.skip_first_n_steps_for_profiler, max_train_steps - 1)
    last_profiled = min(first_profiled + max(keys.profiler_steps, 1), max_train_steps) - 1
    profiler = Profiler(config) if profiler_enabled(config) else None
    # Steps are logged one iteration late, as in `loop.py`: reading step N-1's metrics after dispatching step N keeps
    # one step queued. Step time and loss are averaged over the steps since the previous log (the per-step loss is
    # noisy: every step draws new timesteps and noise); step 0's time includes compilation.
    loss, pending, last, loss_sum = float("nan"), None, (-1, time.perf_counter()), 0.0
    for step in range(max_train_steps):
      if profiler is not None and step == first_profiled:
        profiler.start()
      with jax.profiler.StepTraceAnnotation("train", step_num=step):
        metrics = step_fn(batch_fn(step), jax.random.fold_in(rng, step))
      loss_sum = loss_sum + metrics["loss"]  # on device: does not wait for the step
      if pending is not None:
        loss, last = _log_step(*pending, last, latent_shape[0])
      pending = None
      if step == 0 or (step + 1) % log_period == 0 or step + 1 == max_train_steps:
        pending, loss_sum = (step, {**metrics, "loss": loss_sum / (step - last[0])}), 0.0
      if profiler is not None and step == last_profiled:
        jax.block_until_ready(metrics)
        profiler.stop()
        max_logging.log(f"loop_diffusers: profiled steps {first_profiled}..{step} under {config.tensorboard_dir}")
    if pending is not None:
      loss, last = _log_step(*pending, last, latent_shape[0])

    export_dir = os.path.join(output_dir, "export")
    export_for_inference(export_dir, model, transformer_config, checkpoint_dir, config)
    max_logging.log(f"loop_diffusers: exported the trained checkpoint for sample.py to {export_dir}")
    return loss


def main(argv: Sequence[str]) -> None:
  train_loop(config_lib.initialize(argv))


if __name__ == "__main__":
  app.run(main)
