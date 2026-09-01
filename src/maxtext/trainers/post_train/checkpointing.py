# Copyright 2023-2026 Google LLC
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

"""Checkpointing for the Tunix post-training trainers, in MaxText's on-disk layout.

Lives here rather than in `maxtext.common.checkpointing` because the manager subclasses
Tunix's, and `maxtext.common.checkpointing` is imported by pre-training and inference, which
run without Tunix installed.
"""

import os
from typing import Any, Sequence

from flax import nnx
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from tunix.sft import checkpoint_manager as tunix_checkpoint_manager

from maxtext.common import checkpointing
from maxtext.common import train_state_nnx
from maxtext.utils import max_logging

# The item MaxText stores a checkpoint under, matching create_orbax_checkpoint_manager.
_ITEM_NAME = "items"

# What Tunix stored a checkpoint under, kept registered so old checkpoints still restore.
_TUNIX_ITEM_NAMES = ("model_params", "optimizer_state")

# The Tunix adapter's only child module. DPO and RL train through the adapter, so its state
# carries this extra level and a MaxText checkpoint must not.
_ADAPTER_CHILD = "base"


def unwrap_model(model: nnx.Module) -> nnx.Module:
  """Returns the MaxText model, unwrapping the Tunix adapter if there is one.

  Matches on the child module rather than on `TunixMaxTextAdapter` itself, so any equivalent
  wrapper unwraps the same way.

  Args:
    model: The model a Tunix trainer holds.

  Returns:
    The wrapped model, or `model` itself if it is not wrapped.
  """
  base = getattr(model, _ADAPTER_CHILD, None)
  if isinstance(base, nnx.Module):
    return unwrap_model(base)
  return model


def _drop_adapter_level(tree):
  """Removes the adapter level wherever it wraps a weight-shaped subtree.

  The optimizer is built over the adapter, so its accumulators (mu, nu, acc_grads) are keyed by
  the adapter's graph and carry the level even though the weights they shadow do not.

  Args:
    tree: A pure dict, typically the optimizer state.

  Returns:
    The same tree with every `{"base": subtree}` replaced by `subtree`.
  """
  if isinstance(tree, dict):
    if set(tree) == {_ADAPTER_CHILD}:
      return _drop_adapter_level(tree[_ADAPTER_CHILD])
    return {k: _drop_adapter_level(v) for k, v in tree.items()}
  if isinstance(tree, list):
    return [_drop_adapter_level(v) for v in tree]
  return tree


def _add_adapter_level(tree, guide):
  """Inverse of `_drop_adapter_level`.

  Args:
    tree: A pure dict with the adapter level removed.
    guide: The same tree before removal, giving the positions to restore.

  Returns:
    `tree` with the adapter level put back wherever `guide` carries it.
  """
  if isinstance(guide, dict) and set(guide) == {_ADAPTER_CHILD}:
    return {_ADAPTER_CHILD: _add_adapter_level(tree, guide[_ADAPTER_CHILD])}
  if isinstance(guide, dict) and isinstance(tree, dict):
    return {k: (_add_adapter_level(v, guide[k]) if k in guide else v) for k, v in tree.items()}
  if isinstance(guide, list) and isinstance(tree, list) and len(guide) == len(tree):
    return [_add_adapter_level(t, g) for t, g in zip(tree, guide)]
  return tree


def _drop_inject_hyperparams(opt_state):
  """Strips the `optax.inject_hyperparams` state wrapper if present.

  RL and distillation trainers wrap their optimizer in `inject_hyperparams`. To produce
  a checkpoint fully compatible with pre-training, we strip the outer shell and only save
  the inner state.

  Args:
    opt_state: The optimizer state dict to inspect.

  Returns:
    The inner state if `inject_hyperparams` was found, otherwise `opt_state`.
  """
  if isinstance(opt_state, dict) and {"count", "hyperparams", "hyperparams_states", "inner_state"}.issubset(
      opt_state.keys()
  ):
    return opt_state["inner_state"]
  return opt_state


def _add_inject_hyperparams(restored_opt_state, guide, step):
  """Restores the `optax.inject_hyperparams` wrapper state.

  Args:
    restored_opt_state: The bare inner state loaded from disk.
    guide: The currently initialized optimizer state dict, used as a structural guide.
    step: The global step to restore into the wrapper's count.

  Returns:
    The reconstructed full state dict.
  """
  if isinstance(guide, dict) and {"count", "hyperparams", "hyperparams_states", "inner_state"}.issubset(guide.keys()):

    new_state = dict(guide)
    new_state["inner_state"] = restored_opt_state
    new_state["count"] = jnp.array(step, dtype=guide["count"].dtype)
    return new_state
  return restored_opt_state


class MaxTextLayoutCheckpointManager(tunix_checkpoint_manager.CheckpointManager):
  """Tunix checkpoint manager that reads and writes MaxText's on-disk layout.

  Tunix stores `nnx.state(model)` verbatim under a `model_params` item. MaxText stores the Linen
  layout under `items`: weights in `params/params`, the optimizer in `opt_state` and `step`, and
  NNX-only state such as rngs in `nnx_aux`. Converting on the way out keeps post-training
  checkpoints loadable by pre-training and everything else that reads MaxText checkpoints.

  Checkpoints written before this existed are still in the Tunix layout, so `maybe_restore`
  falls back to the base class for those.
  """

  def __init__(self, root_directory=None, options=None, extra_item_handlers=None, config=None):
    """Initializes the manager.

    Args:
      root_directory: Directory to write checkpoints to. None disables checkpointing.
      options: Orbax `CheckpointManagerOptions`.
      extra_item_handlers: Handlers for items a subclass saves besides the state.
      config: The run's config, read for the metadata the checkpoint stores.
    """
    self._config = config
    super().__init__(root_directory=root_directory, options=options)
    # The base class built a manager over Tunix's item names. Close it before replacing it with
    # one that knows MaxText's layout, or its open handles and threads outlive it.
    # pylint: disable=access-member-before-definition
    if getattr(self, "_checkpoint_manager", None) is not None:
      self._checkpoint_manager.close()
    # pylint: enable=access-member-before-definition

    if root_directory is not None:
      # Pathways only supports the persistence APIs, so drop ocdbt/zarr3 there as Tunix does.
      pathways = "proxy" in os.getenv("JAX_PLATFORMS", "")

      # Orbax otherwise materialises the whole tree on the host at once (its default concurrency
      # is ~89GiB), which OOMKills the container the trainer runs in. MaxText already has a knob
      # for this, checkpoint_storage_concurrent_gb, but it was never plumbed into this path.
      concurrent_gb = getattr(config, "checkpoint_storage_concurrent_gb", None) if config is not None else None

      def pytree_handler():
        kwargs = {"use_ocdbt": not pathways, "use_zarr3": not pathways}
        if concurrent_gb:
          # Only the device-to-host budget: that is the one that decides how much of the tree is
          # resident in host memory at once. Capping save_concurrent_gb/restore_concurrent_gb as
          # well breaks reads of any single array larger than the cap, e.g. llama3.1-8b's
          # mlp.wi_0.kernel at 3.75GiB ("Requested more bytes than we reserved space for").
          kwargs["save_device_host_concurrent_gb"] = concurrent_gb
        return ocp.PyTreeCheckpointHandler(**kwargs)

      handlers = {
          _ITEM_NAME: pytree_handler(),
          # Tunix's item names stay registered so `maybe_restore` can fall back to checkpoints
          # written before the layout change.
          **{name: pytree_handler() for name in _TUNIX_ITEM_NAMES},
          "custom_metadata": ocp.JsonCheckpointHandler(),
          **(extra_item_handlers or {}),
      }
      self._checkpoint_manager = ocp.CheckpointManager(
          root_directory,
          item_handlers=handlers,
          options=options,
      )
    else:
      self._checkpoint_manager = None

  def wait_until_finished(self):
    """Blocks until outstanding async checkpoint writes are complete."""
    if getattr(self, "_checkpoint_manager", None) is not None:
      self._checkpoint_manager.wait_until_finished()

  def close(self):
    """Closes the checkpoint manager."""
    if getattr(self, "_checkpoint_manager", None) is not None:
      self._checkpoint_manager.close()
    # Tunix's __init__ opened a checkpointer of its own that this class replaces but cannot
    # avoid creating. Nothing here writes through it; close it so its handles and threads do
    # not outlive the run. Its close() reads the attribute without checking, so only call it
    # once there is something to close.
    if getattr(self, "_checkpointer", None) is not None:
      super().close()

  def latest_step(self) -> int | None:
    """Returns the latest step saved, reloading from storage if not cached."""
    if getattr(self, "_checkpoint_manager", None) is None:
      return None
    step = self._checkpoint_manager.latest_step()
    if step is None:
      steps = self.all_steps(read=True)
      return steps[-1] if steps else None
    return step

  def all_steps(self, read: bool = False) -> Sequence[int]:
    """Returns all steps tracked by the manager."""
    if getattr(self, "_checkpoint_manager", None) is None:
      return []
    return self._checkpoint_manager.all_steps(read=read)

  def model_to_checkpoint(self, model: nnx.Module) -> nnx.Module:
    """Returns the module whose weights belong in the checkpoint.

    Args:
      model: The model the trainer holds.

    Returns:
      The module to checkpoint. Subclasses override this when it is not the trainer's model.
    """
    return unwrap_model(model)

  def _train_state(self, model, optimizer):
    """Returns the `{model, optimizer}` state to checkpoint.

    Args:
      model: The model the trainer holds.
      optimizer: The trainer's optimizer, or None to checkpoint weights only.

    Returns:
      An `nnx.State` shaped like the one pre-training checkpoints.
    """
    return nnx.state(train_state_nnx.TrainStateNNX(self.model_to_checkpoint(model), optimizer))

  def _extra_save_args(self, step):
    """Returns save args for items a subclass stores besides the state.

    Args:
      step: The step being saved.

    Returns:
      A dict of item name to Orbax save args. Empty by default.
    """
    del step
    return {}

  def save(  # pylint: disable=too-many-positional-arguments
      self,
      step: int,
      model: nnx.Module,
      optimizer: nnx.Optimizer | None = None,
      save_only_lora_params: bool = False,
      force: bool = False,
      custom_metadata: dict[str, Any] | None = None,
  ) -> bool:
    """Saves the model and optimizer in MaxText's on-disk layout.

    Args:
      step: The step to save at.
      model: The model the trainer holds.
      optimizer: The trainer's optimizer, or None to save weights only.
      save_only_lora_params: Whether to save only the LoRA params.
      force: Whether to save regardless of the save decision policy.
      custom_metadata: Metadata to store with the checkpoint.

    Returns:
      Whether a checkpoint was written.
    """
    if self._checkpoint_manager is None:
      return False
    if not force and not self._checkpoint_manager.should_save(step):
      return False

    state = self._train_state(model, optimizer)
    if save_only_lora_params:
      state = nnx.split_state(state, nnx.LoRAParam, ...)[0]
    items = train_state_nnx.to_checkpoint_dict(state)
    if "opt_state" in items:
      inner = _drop_inject_hyperparams(items["opt_state"])
      if inner is not items["opt_state"]:
        # to_checkpoint_dict ran against the inject_hyperparams shell. It puts mu and nu into
        # the Linen `params` collection by finding those keys at the top of the optimizer
        # state, and behind the shell they are not there, so it left them bare. Convert what
        # was behind it, or pre-training finds the accumulators one level short.
        inner = train_state_nnx.opt_state_to_linen(inner)
      items["opt_state"] = inner
      if self.model_to_checkpoint(model) is not model:
        items["opt_state"] = _drop_adapter_level(items["opt_state"])
    jax.block_until_ready(items)

    save_args = {
        _ITEM_NAME: ocp.args.PyTreeSave(item=items, save_args=jax.tree.map(lambda _: ocp.SaveArgs(), items)),
        **self._extra_save_args(step),
    }
    # The config-derived keys are the ones pre-training writes; a caller's own keys win.
    metadata = checkpointing.checkpoint_custom_metadata(self._config)
    metadata.update(custom_metadata or {})

    if not force and step in self.all_steps():
      max_logging.log(f"Step {step} already exists in MaxText layout. Skipping save.")
      return False

    try:
      saved = self._checkpoint_manager.save(
          step,
          args=ocp.args.Composite(**save_args),
          custom_metadata=metadata,
          force=force,
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      if "StepAlreadyExistsError" in type(e).__name__:
        max_logging.log(f"Step {step} already exists. Skipping save.")
        saved = False
      else:
        raise e
    if saved:
      max_logging.log(f"Saved post-training checkpoint at step {step} in MaxText's on-disk layout")
    return saved

  def maybe_restore(
      self,
      model: nnx.Module,
      optimizer: nnx.Optimizer | None = None,
      step: int | None = None,
      restore_only_lora_params: bool = False,
  ) -> tuple[int, dict[str, Any]]:
    """Restores the model and optimizer in place from the latest checkpoint.

    Args:
      model: The model to restore into.
      optimizer: The optimizer to restore into, or None to skip it.
      step: The step to restore from. Defaults to the latest.
      restore_only_lora_params: Whether to restore only the LoRA params.

    Returns:
      A tuple of the restored step (0 if there is no checkpoint) and its custom metadata.
    """
    if self._checkpoint_manager is None:
      return 0, {}
    if step is None:
      step = self._checkpoint_manager.latest_step()
      if step is None:
        return 0, {}

    metadata = self._checkpoint_manager.metadata(step)
    if _ITEM_NAME not in metadata.item_metadata:
      max_logging.log(f"Step {step} predates MaxText-layout post-training checkpoints; restoring the Tunix layout")
      return super().maybe_restore(model, optimizer, step=step, restore_only_lora_params=restore_only_lora_params)

    state = self._train_state(model, optimizer)
    if restore_only_lora_params:
      # save() narrows the state the same way, so a LoRA run writes only its adapter. Restoring
      # against the full state asks for weights the checkpoint never held.
      state = nnx.split_state(state, nnx.LoRAParam, ...)[0]
    target = train_state_nnx.to_checkpoint_dict(state)
    opt_state_guide = target.get("opt_state")
    is_wrapped = self.model_to_checkpoint(model) is not model
    if is_wrapped and opt_state_guide is not None:
      target["opt_state"] = _drop_adapter_level(opt_state_guide)

    restored = self._checkpoint_manager.restore(
        step,
        args=ocp.args.Composite(
            **{
                _ITEM_NAME: ocp.args.PyTreeRestore(
                    item=target,
                    restore_args=ocp.checkpoint_utils.construct_restore_args(target),
                )
            }
        ),
    )

    restored_items = dict(restored[_ITEM_NAME])
    if "opt_state" in restored_items and opt_state_guide is not None:
      restored_items["opt_state"] = _add_inject_hyperparams(restored_items["opt_state"], opt_state_guide, step)
      if is_wrapped:
        restored_items["opt_state"] = _add_adapter_level(restored_items["opt_state"], opt_state_guide)

    new_state = checkpointing.linen_items_to_nnx(restored_items, state)
    nnx.update(self.model_to_checkpoint(model), new_state["model"])
    if optimizer is not None and "optimizer" in new_state:
      nnx.update(optimizer, new_state["optimizer"])

    max_logging.log(f"Restored post-training checkpoint from step {step}")
    return step, (metadata.custom_metadata if metadata else {}) or {}


def install(trainer, checkpoint_dir: str, config=None) -> None:
  """Replaces a Tunix trainer's checkpoint manager with the MaxText-layout one and restores.

  `PeftTrainer.__init__` builds its own manager and restores from it, so callers pass a
  `checkpoint_root_directory` of None and call this straight afterwards instead.

  Args:
    trainer: A Tunix `PeftTrainer` or subclass.
    checkpoint_dir: Directory to read and write checkpoints in.
    config: The run's config, read for the metadata the checkpoint stores.
  """
  # enable_checkpointing is a documented MaxText flag, and until now this path ignored it: the
  # manager was installed regardless, so Tunix saved at the end of training no matter what the
  # config said. That save is not free -- an 8B SFT writes 44.9 GiB and the transfer to host
  # OOMKills the container -- so being able to turn it off is the difference between a smoke test
  # that reports whether the trainer runs and one that cannot get past its first save.
  # post_train_skip_checkpointing exists because neither existing flag can say this:
  # enable_checkpointing is validated as required whenever load_parameters_path is set, and
  # checkpoint_period=0 makes the shared checkpointing path raise ZeroDivisionError on
  # `step % config.checkpoint_period`.
  if config is not None and (
      not getattr(config, "enable_checkpointing", True) or getattr(config, "post_train_skip_checkpointing", False)
  ):
    max_logging.log("Checkpoint saving disabled: skipping post-train checkpoint manager install.")
    return

  if trainer.checkpoint_manager is not None:
    trainer.checkpoint_manager.close()

  trainer.checkpoint_manager = MaxTextLayoutCheckpointManager(
      root_directory=checkpoint_dir,
      options=trainer.config.checkpointing_options,
      config=config,
  )
  # pylint: disable=protected-access
  trainer._train_steps, trainer._restored_custom_metadata = trainer.checkpoint_manager.maybe_restore(
      trainer.model,
      trainer.optimizer,
      restore_only_lora_params=getattr(trainer, "_lora_enabled", False),
  )
  trainer._iter_steps = trainer._train_steps * trainer.config.get_with_default("gradient_accumulation_steps", 1)
  # pylint: enable=protected-access
