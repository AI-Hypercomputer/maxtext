# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared, model-agnostic trainer for diffusion models in m3.

Why a separate trainer: the m3 create_model and the training engine expect a model that takes tokens and
positions and returns logits. A diffusion backbone (DiT) instead takes latents, timesteps and text
embeddings, and predicts a flow-matching (velocity) target rather than next-token logits.

Design:
  * The backbone class is `transformer_class` (a `package.module.ClassName` import path; Wan's `WanModel` by
    default). Its architecture comes from the converted checkpoint's `transformer_config` when
    `checkpoint_directory` is set, else from the config's `transformer_config` block (training from scratch).
  * Weights: with `checkpoint_directory` (a converted Orbax checkpoint, e.g. from `to_maxtext_diffusers.py`), the
    params are restored sharded directly onto the mesh as `weights_dtype` (use float32 master weights for
    training; native Orbax I/O, no dependency on the converter). Without it, the backbone is randomly initialized
    on the mesh.
  * The VAE and text encoder are frozen and not run here: training consumes pre-computed
    `{latents, encoder_hidden_states}` (MaxDiffusion-compatible TFRecords) or synthetic batches.
  * Loss: weighted flow-matching MSE from `FlaxFlowMatchScheduler` (`sample_timesteps`, `apply_flow_match`),
    matching MaxDiffusion's `WanTrainer`.
  * Checkpoints: the train state (params + optimizer) is saved under `<output_dir>/train_state`; the final
    params are exported in the inference layout so `sample.py checkpoint_directory=<output_dir>/export` works
    (the source checkpoint's other items, e.g. VAE / scheduler, are copied alongside when there is one).

Usage:
  python -m maxtext.m3.train.train_diffusers src/maxtext/m3/models/wan/configs/wan2.1-t2v-14b.yml \
      checkpoint_directory=<converted ckpt> output_dir=<dir> weights_dtype=float32 dataset_type=synthetic \
      max_train_steps=100 learning_rate=1e-5
"""

import functools
import importlib
import inspect
import json
import os
import time
from typing import Any, Iterator, Sequence

from absl import logging as absl_logging
from etils import epath
from flax import nnx
import flax.linen as nn
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
import optax
import orbax.checkpoint as ocp

from maxtext.m3.configs import config as config_lib
from maxtext.m3.models.diffuser_schedulers.scheduling_flow_match import FlaxFlowMatchScheduler
from maxtext.m3.models.wan.utils_wan import (
    Profiler,
    create_device_mesh,
    get_flash_block_sizes,
    get_precision,
    profiler_enabled,
)
from maxtext.utils import max_logging

# ==============================================================================
# 1. Optimizer
# ==============================================================================

_TRAIN_DEFAULTS = {
    "learning_rate": 1e-5,
    "max_train_steps": 1500,
    "learning_rate_schedule_steps": -1,
    "warmup_steps_fraction": 0.1,
    "adam_b1": 0.9,
    "adam_b2": 0.999,
    "adam_eps": 1e-8,
    "adam_weight_decay": 0.0,
    "opt_enable_grad_global_norm_clipping": False,
    "max_grad_norm": 1.0,
    "opt_enable_grad_clipping": False,
    "max_grad_value": 1.0,
    "checkpoint_every": -1,
    "log_period": 10,
    "dataset_type": "synthetic",
    "train_data_dir": "",
    "gradient_accumulation_steps": 1,
    "enable_checkpointing": True,
    # Dense bf16 peak per chip, used for MFU (TPU v6e: 918 TFLOP/s).
    "peak_tflops_per_chip": 918.0,
    # > 0: keep only the first N transformer blocks (from scratch only), for single-host per-layer perf proxies.
    "num_layers_override": 0,
}


def _cfg(config, name: str):
  """Training knob from the config (top level, `trainer.optimizer`), else the MaxDiffusion default.

  Values are coerced to the default's type: CLI overrides such as `learning_rate=1e-5` are parsed by YAML
  as strings.
  """
  default = _TRAIN_DEFAULTS[name]
  value = default
  optimizer_cfg = getattr(getattr(config, "trainer", None), "optimizer", None)
  if name in config:
    value = config[name]
  elif optimizer_cfg is not None and hasattr(optimizer_cfg, name):
    value = getattr(optimizer_cfg, name)
  if isinstance(default, bool):
    return value if isinstance(value, bool) else str(value).lower() in ("true", "1")
  if isinstance(default, (int, float)):
    return type(default)(value)
  return value


def create_learning_rate_schedule(config) -> optax.Schedule:
  """Linear warmup to `learning_rate`, then constant (MaxDiffusion's WarmupHold policy)."""
  schedule_steps = _cfg(config, "learning_rate_schedule_steps")
  if schedule_steps <= 0:
    schedule_steps = _cfg(config, "max_train_steps")
  warmup_steps = int(schedule_steps * _cfg(config, "warmup_steps_fraction"))
  constant = optax.constant_schedule(_cfg(config, "learning_rate"))
  if warmup_steps == 0:
    return constant
  warmup = optax.linear_schedule(0.0, _cfg(config, "learning_rate"), warmup_steps)
  return optax.join_schedules([warmup, constant], [warmup_steps])


def create_optimizer(config, schedule: optax.Schedule) -> optax.GradientTransformation:
  """AdamW with optional global-norm / value clipping and gradient accumulation.

  With `gradient_accumulation_steps=k > 1`, `optax.MultiSteps` averages the gradients of k micro-steps and applies
  one AdamW update (the LR schedule counts optimizer steps), so the effective batch is k x the micro-batch.
  """
  tx = optax.adamw(
      schedule,
      b1=_cfg(config, "adam_b1"),
      b2=_cfg(config, "adam_b2"),
      eps=_cfg(config, "adam_eps"),
      weight_decay=_cfg(config, "adam_weight_decay"),
  )
  if _cfg(config, "opt_enable_grad_global_norm_clipping"):
    tx = optax.chain(optax.clip_by_global_norm(_cfg(config, "max_grad_norm")), tx)
  if _cfg(config, "opt_enable_grad_clipping"):
    tx = optax.chain(optax.clip(_cfg(config, "max_grad_value")), tx)
  accumulation_steps = _cfg(config, "gradient_accumulation_steps")
  if accumulation_steps > 1:
    tx = optax.MultiSteps(tx, every_k_schedule=accumulation_steps)
  return tx


# ==============================================================================
# 2. Data: pre-computed latents + text embeddings
# ==============================================================================


def _dict_cfg(config, name: str) -> dict:
  """A dict-valued config entry (yml block, or a JSON string from a CLI override); `{}` if absent."""
  value = config.get(name) or {}
  return dict(json.loads(value) if isinstance(value, str) else value)


def import_class(path: str):
  """`"pkg.module.ClassName"` -> the class (the model yml names its modules by import path)."""
  module_name, _, class_name = str(path).rpartition(".")
  if not module_name:
    raise ValueError(f"Expected a `package.module.ClassName` import path, got {path!r}.")
  return getattr(importlib.import_module(module_name), class_name)


def latent_and_text_shapes(config, transformer_config: dict, vae_config: dict) -> tuple[tuple, tuple]:
  """Per-batch shapes of `latents` (B, C, F, H, W) and `encoder_hidden_states` (B, L, D).

  VAE compression from the diffusers-style `vae_config` (`temperal_downsample`: 2**sum in time, causal first frame;
  2**len in space); latent channels and text width from the transformer config.
  """
  downsample = vae_config.get("temperal_downsample") or [False, True, True]
  temporal_scale, spatial_scale = 2 ** sum(bool(d) for d in downsample), 2 ** len(downsample)
  batch = int(config.global_batch_size_to_train_on)
  latents = (
      batch,
      transformer_config["in_channels"],
      (config.num_frames - 1) // temporal_scale + 1,
      config.height // spatial_scale,
      config.width // spatial_scale,
  )
  text = (batch, config.max_sequence_length, transformer_config["text_dim"])
  return latents, text


# ==============================================================================
# 2b. Model construction (sharded, random init or restored params)
# ==============================================================================

# Backbone class when the yml does not name one (`transformer_class: package.module.ClassName`).
_DEFAULT_TRANSFORMER_CLASS = "maxtext.m3.models.wan.transformer_wan.WanModel"


def model_init_kwargs(model_cls, transformer_config: dict, config, mesh: Mesh) -> dict:
  """`model_cls.__init__` kwargs: architecture keys from `transformer_config` + runtime settings from `config`."""
  accepted = set(inspect.signature(model_cls.__init__).parameters) - {"self", "rngs"}
  runtime = {
      "mesh": mesh,
      "dtype": config.activations_dtype,
      "weights_dtype": config.weights_dtype,
      "attention": config.attention,
      "precision": get_precision(config),
      "flash_block_sizes": get_flash_block_sizes(config),
      "remat_policy": config.remat_policy,
      "names_which_can_be_saved": config.names_which_can_be_saved,
      "names_which_can_be_offloaded": config.names_which_can_be_offloaded,
      "flash_min_seq_length": config.flash_min_seq_length,
      "dropout": config.dropout,
      "mask_padding_tokens": config.mask_padding_tokens,
      # The param layout (stacked or per-layer blocks) is fixed by a restored checkpoint's config.
      "scan_layers": transformer_config.get("scan_layers", config.scan_layers),
      "enable_jax_named_scopes": config.enable_jax_named_scopes,
      "attention_config": {
          "use_base2_exp": config.use_base2_exp,
          "use_experimental_scheduler": config.use_experimental_scheduler,
          "ulysses_shards": getattr(config, "ulysses_shards", -1),
          "ulysses_attention_chunks": getattr(config, "ulysses_attention_chunks", 1),
      },
  }
  kwargs = {k: v for k, v in dict(transformer_config).items() if k in accepted}
  kwargs.update({k: v for k, v in runtime.items() if k in accepted})
  return kwargs


def abstract_sharded_model(model_cls, kwargs: dict, mesh: Mesh, logical_axis_rules):
  """(graphdef, abstract state, per-leaf NamedShardings) of `model_cls(**kwargs)`, without allocating weights."""
  abstract = nnx.eval_shape(lambda: model_cls(rngs=nnx.Rngs(0), **kwargs))
  graphdef, abstract_state = nnx.split(abstract)
  shardings = nn.logical_to_mesh_sharding(nnx.get_partition_spec(abstract_state), mesh, logical_axis_rules)
  replicated = NamedSharding(mesh, P())
  shardings = jax.tree.map(
      lambda s: s if isinstance(s, NamedSharding) else replicated,
      shardings,
      is_leaf=lambda x: x is None or isinstance(x, NamedSharding),
  )
  return graphdef, abstract_state, shardings


def create_sharded_model(model_cls, kwargs: dict, mesh: Mesh, logical_axis_rules, params=None, seed: int = 0):
  """Builds `model_cls(**kwargs)` directly on its mesh shardings.

  `params=None` keeps the random init (training from scratch); otherwise `params` (an `nnx.Param` state, e.g.
  restored sharded from a converted checkpoint) replaces the initial values.
  """
  graphdef, abstract_state, shardings = abstract_sharded_model(model_cls, kwargs, mesh, logical_axis_rules)
  # Flat leaves: Variable metadata (e.g. init fns) differs between traces, so whole-state out_shardings mismatch.
  state_treedef = jax.tree.structure(abstract_state)
  sharding_leaves = jax.tree.leaves(shardings, is_leaf=lambda x: isinstance(x, NamedSharding))

  @functools.partial(jax.jit, out_shardings=sharding_leaves)
  def init_leaves():
    return jax.tree.leaves(nnx.state(model_cls(rngs=nnx.Rngs(seed), **kwargs)))

  model = nnx.merge(graphdef, jax.tree.unflatten(state_treedef, init_leaves()))
  if params is not None:
    nnx.update(model, params)
  return model


def train_tflops_per_sample(transformer_config: dict, latent_shape: tuple, text_len: int) -> float:
  """Model TFLOPs of one training sample (forward + backward = 3 x forward) of a DiT, full non-causal attention.

  Per token per layer (forward): self-attn QKVO 8D^2, cross-attn Q/O 4D^2, FFN 4DF, self-attn scores+AV 4SD,
  cross-attn scores+AV 4TD (cross-attn K/V on the text tokens is negligible). Same count as the GPU baseline in
  b/572019856 (Wan 2.1 14B, S=75,600: 19.57 PFLOP / sample).
  """
  d = transformer_config["num_attention_heads"] * transformer_config["attention_head_dim"]
  f = transformer_config["ffn_dim"]
  p_t, p_h, p_w = transformer_config.get("patch_size", (1, 2, 2))
  _, _, frames, height, width = latent_shape
  s = (frames // p_t) * (height // p_h) * (width // p_w)
  per_token_layer = 12 * d * d + 4 * d * f + 4 * s * d + 4 * text_len * d
  return 3 * transformer_config["num_layers"] * per_token_layer * s / 1e12


def synthetic_batches(
    latent_shape: tuple, text_shape: tuple, seed: int, repeat: bool = False, start_step: int = 0
) -> Iterator[dict]:
  """Endless random batches with the real shapes; `repeat=True` yields one fixed batch (overfit check).

  Batch k depends only on `(seed, k)`, so a run resumed at `start_step` sees the same data as an uninterrupted one.
  """

  def make(step: int) -> dict:
    rng = np.random.default_rng((seed, step))
    return {
        "latents": rng.standard_normal(latent_shape, dtype=np.float32),
        "encoder_hidden_states": rng.standard_normal(text_shape, dtype=np.float32),
    }

  fixed = make(0) if repeat else None
  step = start_step
  while True:
    yield fixed if repeat else make(step)
    step += 1


def tfrecord_batches(data_dir: str, batch_size: int, seed: int, start_step: int = 0) -> Iterator[dict]:
  """MaxDiffusion-format TFRecords: serialized float32 tensors `latents` and `encoder_hidden_states`.

  Deterministic order (sorted files, seeded shuffle), and `start_step` batches are skipped, so the data cursor is
  just the micro-step count stored with the train state.
  """
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
  ds = ds.shuffle(1024, seed=seed).repeat().batch(batch_size, drop_remainder=True).skip(start_step)
  yield from ds.prefetch(tf.data.AUTOTUNE).as_numpy_iterator()


# ==============================================================================
# 3. Loss and train step
# ==============================================================================


def flow_match_loss(model, batch: dict, rng: jax.Array, scheduler: FlaxFlowMatchScheduler) -> jax.Array:
  """Weighted flow-matching MSE between the predicted and target velocity."""
  t_rng, n_rng, d_rng = jax.random.split(rng, 3)
  latents = batch["latents"]
  timesteps = scheduler.sample_timesteps(t_rng, latents.shape[0])
  noise = jax.random.normal(n_rng, latents.shape, latents.dtype)
  noisy_latents, target, weight = scheduler.apply_flow_match(noise, latents, timesteps)
  pred = model(
      noisy_latents,
      timesteps,
      batch["encoder_hidden_states"],
      deterministic=False,
      rngs=nnx.Rngs(dropout=d_rng),
  )
  per_sample = jnp.mean(jnp.square(target - pred.astype(jnp.float32)), axis=tuple(range(1, pred.ndim)))
  return jnp.mean(per_sample * weight)


def make_train_step(
    scheduler: FlaxFlowMatchScheduler, schedule: optax.Schedule, accumulation_steps: int, param_shardings, opt_shardings
):
  """Returns a jitted `(model, optimizer, batch, rng) -> metrics` that updates model + optimizer in place.

  The updated params / optimizer state are constrained to their input shardings, so the second call sees
  the same input shardings as the first and does not recompile.
  """

  @nnx.jit(donate_argnums=(0, 1))
  def train_step(model, optimizer, batch, rng):
    loss, grads = nnx.value_and_grad(lambda m: flow_match_loss(m, batch, rng, scheduler))(model)
    optimizer.update(model, grads)
    nnx.update(model, jax.lax.with_sharding_constraint(nnx.state(model, nnx.Param), param_shardings))
    nnx.update(optimizer, jax.lax.with_sharding_constraint(nnx.state(optimizer), opt_shardings))
    optimizer_step = (optimizer.step.value - 1) // accumulation_steps
    return {"loss": loss, "grad_norm": optax.global_norm(grads), "learning_rate": schedule(optimizer_step)}

  return train_step


def _mesh_shardings(state: nnx.State, mesh: Mesh):
  """Per-leaf mesh shardings of a concrete nnx state; leaves not yet on the mesh (single-device) are replicated."""
  replicated = NamedSharding(mesh, P())
  return jax.tree.map(lambda x: x.sharding if isinstance(x.sharding, NamedSharding) else replicated, state)


def place_state(obj, state_filter, mesh: Mesh):
  """Puts `obj`'s state on its mesh shardings and returns them (train-step inputs then match its outputs)."""
  state = nnx.state(obj, *state_filter)
  shardings = _mesh_shardings(state, mesh)
  nnx.update(obj, jax.device_put(state, shardings))
  return shardings


def to_global_batch(host_batch: dict, sharding: NamedSharding) -> dict:
  """Builds global arrays from a batch that is identical on every process (multi-host safe)."""
  return {
      k: jax.make_array_from_callback(v.shape, sharding, lambda idx, v=v: v[idx].astype(np.float32))
      for k, v in host_batch.items()
  }


# ==============================================================================
# 4. Checkpoints
# ==============================================================================


def _str_keys(tree: Any) -> Any:
  """Orbax-friendly nested dict: list indices (ints) become strings, as in converted checkpoints."""
  if isinstance(tree, dict):
    return {str(k): _str_keys(v) for k, v in tree.items()}
  return tree


def _flat(state: nnx.State) -> dict[str, jax.Array]:
  """`{"a/b/0/kernel": array}` view of an nnx state (Orbax needs string keys)."""
  return {"/".join(map(str, path)): var.value for path, var in nnx.to_flat_state(state)}


# Converted (inference) checkpoint layout: one Orbax step with `<component>_state` (param tree, string keys) and
# `<component>_config` (JSON) items per component, as written by `checkpoint_conversion/to_maxtext_diffusers.py`.


def _component_manager(checkpoint_dir: str, item_names: Sequence[str], create: bool = False) -> ocp.CheckpointManager:
  handlers = {
      n: ocp.JsonCheckpointHandler() if n.endswith("_config") else ocp.StandardCheckpointHandler() for n in item_names
  }
  return ocp.CheckpointManager(
      epath.Path(checkpoint_dir),
      item_names=tuple(item_names),
      item_handlers=handlers,
      options=ocp.CheckpointManagerOptions(create=create, enable_async_checkpointing=False),
  )


def _latest_step(manager: ocp.CheckpointManager, checkpoint_dir: str) -> int:
  step = manager.latest_step()
  if step is None:
    raise FileNotFoundError(f"No checkpoint step found in {checkpoint_dir}")
  return step


def restore_component_config(checkpoint_dir: str, component: str) -> dict | None:
  """The `<component>_config` JSON of a converted checkpoint (no weights read); None if absent."""
  name = f"{component}_config"
  manager = _component_manager(checkpoint_dir, (name,))
  step = _latest_step(manager, checkpoint_dir)
  if not (epath.Path(checkpoint_dir) / str(step) / name).exists():
    return None
  restored = manager.restore(step, args=ocp.args.Composite(**{name: ocp.args.JsonRestore()}))
  return {k: v for k, v in dict(restored[name]).items() if not k.startswith("_")}


def _normalize_param_path(path: tuple) -> tuple:
  """Checkpoint key path -> nnx state path: list indices are strings ("0") on disk; drop a trailing `value`."""
  keys = tuple(int(k) if isinstance(k, str) and k.isdigit() else k for k in (getattr(p, "key", p) for p in path))
  return keys[:-1] if keys and keys[-1] == "value" else keys


def restore_component_params(checkpoint_dir: str, component: str, param_shardings: nnx.State, dtype) -> nnx.State:
  """Restores `<component>_state` sharded directly onto the mesh (each host reads only its shards), as `dtype`.

  `param_shardings` is the model's `nnx.Param` state with a `NamedSharding` per leaf; the result has the same
  structure, ready for `nnx.update`.
  """
  name = f"{component}_state"
  manager = _component_manager(checkpoint_dir, (name,))
  step = _latest_step(manager, checkpoint_dir)
  metadata = manager.item_metadata(step)
  metadata = metadata[name] if name in metadata.keys() else getattr(metadata, name)  # Composite metadata
  by_path = dict(nnx.to_flat_state(param_shardings))

  def target(path, meta):
    sharding = by_path[_normalize_param_path(path)]
    sharding = getattr(sharding, "value", sharding)
    return jax.ShapeDtypeStruct(meta.shape, dtype, sharding=sharding)

  abstract = jax.tree_util.tree_map_with_path(target, metadata)
  restored = manager.restore(step, args=ocp.args.Composite(**{name: ocp.args.StandardRestore(abstract)}))[name]
  values = {_normalize_param_path(path): v for path, v in jax.tree_util.tree_flatten_with_path(restored)[0]}
  missing = [path for path, _ in nnx.to_flat_state(param_shardings) if path not in values]
  if missing:
    raise KeyError(f"{name} in {checkpoint_dir} lacks {len(missing)} model params, e.g. {missing[:3]}")
  return nnx.from_flat_state(
      [(path, var.replace(value=values[path])) for path, var in nnx.to_flat_state(param_shardings)]
  )


def train_state_tree(model, optimizer) -> dict:
  """Everything needed to resume: params (the optimizer does not hold the model) + optimizer state."""
  return {"params": _flat(nnx.state(model, nnx.Param)), "optimizer": _flat(nnx.state(optimizer))}


def load_train_state_tree(model, optimizer, tree: dict) -> None:
  """Inverse of `train_state_tree`: writes restored arrays back into `model` and `optimizer`."""
  for obj, state, flat in (
      (model, nnx.state(model, nnx.Param), tree["params"]),
      (optimizer, nnx.state(optimizer), tree["optimizer"]),
  ):
    for path, var in nnx.to_flat_state(state):
      var.value = flat["/".join(map(str, path))]
    nnx.update(obj, state)


def create_train_state_manager(output_dir: str, config) -> ocp.CheckpointManager:
  every = _cfg(config, "checkpoint_every")
  if every <= 0:
    every = _cfg(config, "max_train_steps")
  return ocp.CheckpointManager(
      epath.Path(output_dir) / "train_state",
      options=ocp.CheckpointManagerOptions(create=True, save_interval_steps=every, max_to_keep=2),
  )


def export_for_inference(export_dir: str, model, transformer_config: dict, source_checkpoint_dir: str = "") -> None:
  """Writes the trained transformer in the converted-checkpoint layout `sample.py` reads.

  Items other than `transformer_*` (e.g. `vae_state`, `vae_config`, `scheduler_config`) are copied from the
  source checkpoint when training started from one.
  """
  names = ("transformer_state", "transformer_config")
  manager = _component_manager(export_dir, names, create=True)
  manager.save(
      0,
      args=ocp.args.Composite(
          transformer_state=ocp.args.StandardSave(_str_keys(nnx.state(model, nnx.Param).to_pure_dict())),
          transformer_config=ocp.args.JsonSave(transformer_config),
      ),
  )
  manager.wait_until_finished()
  if not source_checkpoint_dir or jax.process_index() != 0:
    return
  src_step = _latest_step(_component_manager(source_checkpoint_dir, names), source_checkpoint_dir)
  src, dst = epath.Path(source_checkpoint_dir) / str(src_step), epath.Path(export_dir) / "0"
  for item in src.iterdir():
    if item.name in names or not item.is_dir():
      continue
    for f in item.rglob("*"):
      if f.is_file():
        target = dst / item.name / f.relative_to(item)
        target.parent.mkdir(parents=True, exist_ok=True)
        f.copy(target)


# ==============================================================================
# 5. Training loop
# ==============================================================================


def train(config) -> None:
  """Builds (restores or randomly initializes) the backbone, then runs `max_train_steps` flow-matching steps."""
  output_dir = config.output_dir or config.base_output_directory
  if not output_dir:
    raise ValueError("output_dir is required.")
  if not config.tensorboard_dir:
    config.tensorboard_dir = os.path.join(output_dir, "tensorboard", "")

  devices_array = create_device_mesh(config)
  mesh = Mesh(devices_array, config.mesh_axes)
  # Sequence-parallel attention kernels (ulysses / ring) need sequence-sharded q / kv rules: pass them explicitly as
  # a `logical_axis_rules` override (see run_wan_train_ctk.sh); the trainer uses the config's rules as given.
  logical_axis_rules = config.logical_axis_rules
  transformer_cls = import_class(config.get("transformer_class", _DEFAULT_TRANSFORMER_CLASS))
  checkpoint_dir = config.checkpoint_directory or ""
  if checkpoint_dir:  # the architecture is fixed by the checkpoint
    transformer_config = restore_component_config(checkpoint_dir, "transformer")
    vae_config = restore_component_config(checkpoint_dir, "vae") or _dict_cfg(config, "vae_config")
  else:  # from scratch: the architecture comes from the config (`transformer_config` block / override)
    transformer_config, vae_config = _dict_cfg(config, "transformer_config"), _dict_cfg(config, "vae_config")
    if not transformer_config:
      raise ValueError("Training from scratch needs `transformer_config` (architecture) in the config.")
  num_layers_override = _cfg(config, "num_layers_override")
  if num_layers_override > 0:
    if checkpoint_dir:
      raise ValueError("num_layers_override is only supported when training from scratch (no checkpoint_directory).")
    transformer_config["num_layers"] = num_layers_override

  scheduler, _ = FlaxFlowMatchScheduler.from_config({}, dtype=jnp.float32)
  schedule = create_learning_rate_schedule(config)
  max_train_steps = _cfg(config, "max_train_steps")
  log_period = max(_cfg(config, "log_period"), 1)
  latent_shape, text_shape = latent_and_text_shapes(config, transformer_config, vae_config)
  dataset_type = _cfg(config, "dataset_type")
  data_sharding = NamedSharding(mesh, P(("data", "fsdp")))

  with mesh, nn_partitioning.axis_rules(logical_axis_rules):
    kwargs = model_init_kwargs(transformer_cls, transformer_config, config, mesh)
    params = None
    if checkpoint_dir:
      _, _, shardings = abstract_sharded_model(transformer_cls, kwargs, mesh, logical_axis_rules)
      param_shardings = nnx.filter_state(shardings, nnx.Param)
      params = restore_component_params(checkpoint_dir, "transformer", param_shardings, config.weights_dtype)
      max_logging.log(f"Restored transformer params from {checkpoint_dir}")
    model = create_sharded_model(transformer_cls, kwargs, mesh, logical_axis_rules, params=params, seed=config.seed)
    optimizer = nnx.Optimizer(model, create_optimizer(config, schedule), wrt=nnx.Param)
    num_params = sum(x.size for x in jax.tree.leaves(nnx.state(model, nnx.Param)))
    max_logging.log(
        f"Training {num_params / 1e9:.2f}B params; latents {latent_shape}, text {text_shape}, "
        f"mesh {dict(mesh.shape)}, weights {config.weights_dtype}, activations {config.activations_dtype}"
    )

    enable_checkpointing = _cfg(config, "enable_checkpointing")
    manager = create_train_state_manager(output_dir, config) if enable_checkpointing else None
    start_step = 0
    if manager is not None and manager.latest_step() is not None:
      start_step = manager.latest_step()
      restored_tree = manager.restore(start_step, args=ocp.args.StandardRestore(train_state_tree(model, optimizer)))
      load_train_state_tree(model, optimizer, restored_tree)
      max_logging.log(f"Resumed train state from step {start_step}")
    # The data cursor is the (micro-)step: both iterators are deterministic and seek to `start_step`, and the
    # per-step RNG is `fold_in(seed, step)`, so a resumed run continues exactly like an uninterrupted one.
    if dataset_type in ("synthetic", "synthetic_repeat"):
      batches = synthetic_batches(
          latent_shape, text_shape, config.seed, repeat=dataset_type == "synthetic_repeat", start_step=start_step
      )
    else:
      batches = tfrecord_batches(_cfg(config, "train_data_dir"), latent_shape[0], config.seed, start_step=start_step)

    accumulation_steps = _cfg(config, "gradient_accumulation_steps")
    param_shardings = place_state(model, (nnx.Param,), mesh)
    opt_shardings = place_state(optimizer, (), mesh)
    train_step = make_train_step(scheduler, schedule, accumulation_steps, param_shardings, opt_shardings)
    tflops_per_sample = train_tflops_per_sample(transformer_config, latent_shape, text_shape[1])
    peak_tflops = _cfg(config, "peak_tflops_per_chip")
    num_chips = mesh.devices.size
    max_logging.log(
        f"{tflops_per_sample:.1f} TFLOP / sample, micro-batch {latent_shape[0]}, gradient accumulation "
        f"{accumulation_steps} (global batch {latent_shape[0] * accumulation_steps}), {num_chips} chips, "
        f"processes {jax.process_count()}, checkpointing {enable_checkpointing}"
    )
    rng = jax.random.key(config.seed)
    # xprof window (same keys as sample.py / the yml): skip the first steps (compile), then trace
    # `profiler_steps` steady-state steps; the trace lands under `tensorboard_dir` (uploaded if gs://).
    profiling = profiler_enabled(config)
    first_profiling_step = start_step + int(config.get("skip_first_n_steps_for_profiler", 0))
    last_profiling_step = first_profiling_step + max(int(config.get("profiler_steps", 1)), 1) - 1
    profiler = None
    step_start, last_logged_step = time.perf_counter(), start_step
    for step in range(start_step, max_train_steps):
      if profiling and step == min(first_profiling_step, max_train_steps - 1):
        profiler = Profiler(config)
        profiler.start()
      batch = to_global_batch(next(batches), data_sharding)
      with jax.profiler.StepTraceAnnotation("train", step_num=step):
        metrics = train_step(model, optimizer, batch, jax.random.fold_in(rng, step))
        if profiler is not None:
          jax.block_until_ready(metrics)  # keep each traced step's device work inside its annotation
      if profiler is not None and step == min(last_profiling_step, max_train_steps - 1):
        profiler.stop()
        profiler = None
        max_logging.log(
            f"Profiled train steps {first_profiling_step + 1}..{step + 1}; trace under {config.tensorboard_dir}"
        )
      if step == start_step or (step + 1) % log_period == 0 or step + 1 == max_train_steps:
        metrics = jax.device_get(metrics)
        elapsed = time.perf_counter() - step_start
        steps_done = (step + 1) - last_logged_step
        step_time = elapsed / steps_done
        tflops_per_chip = tflops_per_sample * latent_shape[0] / step_time / num_chips
        stats = jax.local_devices()[0].memory_stats() or {}
        # Heap only (params, optimizer state, batch); program temporaries (activations) are reserved separately.
        peak_hbm = f" peak_heap {stats['peak_bytes_in_use'] / 2**30:.2f}GiB" if "peak_bytes_in_use" in stats else ""
        max_logging.log(
            f"step {step + 1}/{max_train_steps} loss {float(metrics['loss']):.5f} "
            f"grad_norm {float(metrics['grad_norm']):.4f} lr {float(metrics['learning_rate']):.3e} "
            f"step_time {step_time:.3f}s TFLOP/s/chip {tflops_per_chip:.1f} MFU {100 * tflops_per_chip / peak_tflops:.2f}%"
            f"{peak_hbm}{' (includes compile)' if step == start_step else ''}"
        )
        step_start, last_logged_step = time.perf_counter(), step + 1
      if manager is not None:
        manager.save(step + 1, args=ocp.args.StandardSave(train_state_tree(model, optimizer)))
    if manager is None:
      return
    manager.wait_until_finished()

    export_dir = os.path.join(output_dir, "export")
    export_for_inference(export_dir, model, transformer_config, source_checkpoint_dir=checkpoint_dir)
    max_logging.log(f"Exported trained checkpoint for sample.py to {export_dir}")


def main(argv: Sequence[str] | None = None) -> None:
  # max_logging uses absl INFO, which is dropped unless absl.app drives the program.
  absl_logging.set_verbosity(absl_logging.INFO)
  absl_logging.use_absl_handler()
  train(config_lib.initialize(argv))


if __name__ == "__main__":
  main()
