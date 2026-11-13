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

"""Runtime hooks integrating Multi-Pass Precision Emulation across MaxText workflows.

Non-intrusively patches:
1. `pyconfig`: Allows `emulate_*` / `multipass_*` CLI and YAML parameters without
   Pydantic `extra="forbid"` errors, parses `MultipassEmulationConfig`, and
   auto-discovers `multipass_emulation_metadata.json` when loading checkpoints.
2. `model_creation_utils`: Annotates NNX module paths on model creation /
   `from_pretrained`, installs layer operator interceptors, and materializes
   weight multi-pass reconstruction on loaded checkpoints when
   `apply_weight_multipass_on_load=True`.
3. `checkpointing` & `orbax`: Materializes weight multi-pass reconstruction into
   saved parameter trees when `bake_weight_multipass_on_save=True` and writes
   `multipass_emulation_metadata.json`.
4. `maxtext_vllm_rollout`: Ensures RL rollouts (`MaxTextVllmRollout` /
   `MaxTextVllmSampler`) sync emulated actor weights on load and update.
5. `eval.runner.server_manager`: Propagates `emulate_*` config settings into
   `vllm_kwargs["additional_config"]["maxtext_config"]` so in-process vLLM
   evaluates multi-pass checkpoints with full fidelity.
"""

from __future__ import annotations

import functools
import logging
import math
import os
from typing import Any

from maxtext.codesign.interception import (
    annotate_model_paths,
    build_qwix_multipass_provider,
    emulation_jax_step_context,
    get_active_multipass_config,
    get_current_emulation_step,
    install_layer_patches,
    set_active_multipass_config,
    set_current_emulation_step,
)
from maxtext.codesign.multi_pass.checkpoint_utils import (
    apply_multipass_to_model_weights,
    apply_multipass_to_pytree,
    load_multipass_metadata,
    save_multipass_metadata,
)
from maxtext.codesign.multi_pass.multipass_config import (
    DEFAULT_MULTIPASS_CONFIG_KEYS,
    MultipassEmulationConfig,
)

_HOOKS_INSTALLED = False
_STASHED_EMULATION_RAW: dict[str, Any] = {}


def _is_emulation_key(key: str) -> bool:
  return key.startswith(("emulate_", "emulation_", "multipass_")) or key in (
      "multipass_emulation",
      "sparsity_emulation",
  )


def extract_and_attach_multipass_config(
    config: Any, extra_overrides: dict[str, Any] | None = None
) -> MultipassEmulationConfig:
  """Parses `MultipassEmulationConfig` from `config` + `extra_overrides` and attaches it."""
  user_keys: dict[str, Any] = {}
  if hasattr(config, "get_keys"):
    for k, v in config.get_keys().items():
      if _is_emulation_key(k):
        user_keys[k] = v
  elif isinstance(config, dict):
    for k, v in config.items():
      if _is_emulation_key(k):
        user_keys[k] = v
  if extra_overrides:
    for k, v in extra_overrides.items():
      if v is not None:
        user_keys[k] = v

  multipass_cfg = MultipassEmulationConfig.from_maxtext_config(user_keys)
  merged: dict[str, Any] = dict(DEFAULT_MULTIPASS_CONFIG_KEYS)
  merged.update(user_keys)

  if not multipass_cfg.enabled:
    for ckpt_attr in (
        "load_parameters_path",
        "load_full_state_path",
        "checkpoint_dir",
    ):
      ckpt_path = getattr(config, ckpt_attr, "") if not isinstance(config, dict) else config.get(ckpt_attr, "")
      if ckpt_path:
        loaded_meta = load_multipass_metadata(str(ckpt_path))
        if loaded_meta is not None and loaded_meta.enabled:
          multipass_cfg = loaded_meta
          logging.info(
              "[MultipassEmulation] Auto-loaded multi-pass metadata from %s: %s",
              ckpt_path,
              multipass_cfg.summary(),
          )
          break

  if hasattr(config, "get_keys"):
    raw_keys = config.get_keys()
    for k, v in merged.items():
      if _is_emulation_key(k):
        raw_keys[k] = v
    raw_keys["multipass_emulation"] = multipass_cfg
    raw_keys["sparsity_emulation"] = multipass_cfg
    raw_keys["emulate_multipass_enabled"] = multipass_cfg.enabled

  if not isinstance(config, dict):
    for attr_name in ("multipass_emulation", "sparsity_emulation"):
      try:
        object.__setattr__(config, attr_name, multipass_cfg)
      except Exception:  # pylint: disable=broad-exception-caught
        pass

  if multipass_cfg.enabled:
    set_active_multipass_config(multipass_cfg)
    install_layer_patches()
    logging.info("[MultipassEmulation] Active config: %s", multipass_cfg.summary())

    ckpt_dir = getattr(config, "checkpoint_dir", "") if not isinstance(config, dict) else config.get("checkpoint_dir", "")
    if ckpt_dir:
      save_multipass_metadata(str(ckpt_dir), multipass_cfg)

  return multipass_cfg


extract_and_attach_sparsity_config = extract_and_attach_multipass_config


def _patch_pyconfig() -> None:
  """Patches `maxtext.configs.pyconfig` to accept `emulate_*` / `multipass_*` keys seamlessly."""
  try:
    from maxtext.configs import pyconfig  # pylint: disable=import-outside-toplevel
  except ImportError:
    return

  if getattr(pyconfig, "_emulation_patched", False):
    return

  orig_prepare = pyconfig._prepare_for_pydantic  # pylint: disable=protected-access
  orig_initialize = pyconfig.initialize
  orig_initialize_pydantic = getattr(pyconfig, "initialize_pydantic", None)

  @functools.wraps(orig_prepare)
  def _patched_prepare_for_pydantic(raw_keys: dict[str, Any], *args: Any, **kwargs: Any) -> dict[str, Any]:
    _STASHED_EMULATION_RAW.clear()
    filtered: dict[str, Any] = {}
    for k, v in raw_keys.items():
      if _is_emulation_key(k):
        _STASHED_EMULATION_RAW[k] = v
      else:
        filtered[k] = v
    return orig_prepare(filtered, *args, **kwargs)

  @functools.wraps(orig_initialize)
  def _patched_initialize(argv, **kwargs):
    stashed_kw: dict[str, Any] = {}
    clean_kw: dict[str, Any] = {}
    for k, v in kwargs.items():
      if _is_emulation_key(k):
        stashed_kw[k] = v
      else:
        clean_kw[k] = v
    config = orig_initialize(argv, **clean_kw)
    combined_overrides = dict(_STASHED_EMULATION_RAW)
    combined_overrides.update(stashed_kw)
    extract_and_attach_multipass_config(config, combined_overrides)
    return config

  pyconfig._prepare_for_pydantic = _patched_prepare_for_pydantic  # pylint: disable=protected-access
  pyconfig.initialize = _patched_initialize

  try:
    from maxtext.configs import types as mt_types  # pylint: disable=import-outside-toplevel

    if getattr(mt_types.MaxTextConfig, "__hash__", None) is None:
      mt_types.MaxTextConfig.__hash__ = object.__hash__  # type: ignore[method-assign]
  except Exception:  # pylint: disable=broad-exception-caught
    pass

  if orig_initialize_pydantic is not None:

    @functools.wraps(orig_initialize_pydantic)
    def _patched_initialize_pydantic(argv, **kwargs):
      stashed_kw = {k: v for k, v in kwargs.items() if _is_emulation_key(k)}
      clean_kw = {k: v for k, v in kwargs.items() if not _is_emulation_key(k)}
      config = orig_initialize_pydantic(argv, **clean_kw)
      if hasattr(pyconfig, "HyperParameters") and not isinstance(config, pyconfig.HyperParameters):
        config = pyconfig.HyperParameters(config)
      combined_overrides = dict(_STASHED_EMULATION_RAW)
      combined_overrides.update(stashed_kw)
      extract_and_attach_multipass_config(config, combined_overrides)
      return config

    pyconfig.initialize_pydantic = _patched_initialize_pydantic

  pyconfig._emulation_patched = True  # pylint: disable=protected-access


def _patch_model_creation() -> None:
  """Patches `model_creation_utils` and `quantizations` to wire multi-pass emulation into models."""
  try:
    from maxtext.layers import quantizations  # pylint: disable=import-outside-toplevel
    from maxtext.utils import model_creation_utils  # pylint: disable=import-outside-toplevel
  except ImportError:
    return

  if getattr(model_creation_utils, "_emulation_patched", False):
    return

  orig_create_nnx_abstract = getattr(model_creation_utils, "create_nnx_abstract_model", None)
  orig_create_model = getattr(model_creation_utils, "create_model", None)
  orig_from_pretrained = getattr(model_creation_utils, "from_pretrained", None)

  if orig_create_nnx_abstract is not None:

    @functools.wraps(orig_create_nnx_abstract)
    def _patched_create_nnx_abstract(config, *args, **kwargs):
      try:
        import wandb  # pylint: disable=import-outside-toplevel
        from jax._src import monitoring as jax_mon  # pylint: disable=import-outside-toplevel

        if getattr(wandb, "run", None) is None and hasattr(jax_mon, "_scalar_listeners"):
          jax_mon._scalar_listeners.clear()  # pylint: disable=protected-access
      except Exception:  # pylint: disable=broad-exception-caught
        pass
      mp_cfg = getattr(config, "multipass_emulation", None) or get_active_multipass_config()
      res = orig_create_nnx_abstract(config, *args, **kwargs)
      if mp_cfg is not None and mp_cfg.enabled:
        if isinstance(res, tuple) and len(res) >= 2:
          annotate_model_paths(res[1], mp_cfg)
        else:
          annotate_model_paths(res, mp_cfg)
      return res

    model_creation_utils.create_nnx_abstract_model = _patched_create_nnx_abstract

  if orig_create_model is not None:

    @functools.wraps(orig_create_model)
    def _patched_create_model(config, *args, **kwargs):
      mp_cfg = getattr(config, "multipass_emulation", None) or get_active_multipass_config()
      model = orig_create_model(config, *args, **kwargs)
      if mp_cfg is not None and mp_cfg.enabled:
        annotate_model_paths(model, mp_cfg)
      return model

    model_creation_utils.create_model = _patched_create_model

  if orig_from_pretrained is not None:

    @functools.wraps(orig_from_pretrained)
    def _patched_from_pretrained(config, *args, **kwargs):
      mp_cfg = getattr(config, "multipass_emulation", None)
      if mp_cfg is None:
        mp_cfg = extract_and_attach_multipass_config(config)
      model, mesh = orig_from_pretrained(config, *args, **kwargs)
      if mp_cfg is not None and mp_cfg.enabled:
        annotate_model_paths(model, mp_cfg)
        if mp_cfg.apply_weight_multipass_on_load and mp_cfg.start_step <= 0:
          num_mod = apply_multipass_to_model_weights(model, mp_cfg)
          logging.info(
              "[MultipassEmulation] Materialized weight multi-pass reconstruction on load to %d tensors.",
              num_mod,
          )
      return model, mesh

    model_creation_utils.from_pretrained = _patched_from_pretrained

  orig_maybe_quantize = getattr(quantizations, "maybe_quantize_model", None)
  if orig_maybe_quantize is not None:

    @functools.wraps(orig_maybe_quantize)
    def _patched_maybe_quantize_model(model, config, *args, **kwargs):
      model = orig_maybe_quantize(model, config, *args, **kwargs)
      mp_cfg = getattr(config, "multipass_emulation", None) or get_active_multipass_config()
      if mp_cfg is not None and mp_cfg.enabled:
        if not getattr(config, "enable_nnx", True):
          try:
            import qwix  # pylint: disable=import-outside-toplevel

            provider = build_qwix_multipass_provider(mp_cfg)
            model = qwix.quantize_model(model, provider)
          except Exception as e:  # pylint: disable=broad-exception-caught
            logging.warning("[MultipassEmulation] Qwix Linen wrapping skipped: %s", e)
      return model

    quantizations.maybe_quantize_model = _patched_maybe_quantize_model

  model_creation_utils._emulation_patched = True  # pylint: disable=protected-access


def _patch_checkpointing() -> None:
  """Patches `maxtext.common.checkpointing` so saved checkpoints can bake multi-pass weights."""
  try:
    from maxtext.common import checkpointing  # pylint: disable=import-outside-toplevel
  except ImportError:
    return

  if getattr(checkpointing, "_emulation_patched", False):
    return

  orig_save = getattr(checkpointing, "save_checkpoint", None)
  if orig_save is not None:

    @functools.wraps(orig_save)
    def _patched_save_checkpoint(  # pylint: disable=keyword-arg-before-vararg
        checkpoint_manager,
        step,
        state,
        config=None,
        data_iterator=None,
        force=False,
        *args,
        **kwargs,
    ):
      mp_cfg = (
          getattr(config, "multipass_emulation", None) if config is not None else None
      ) or get_active_multipass_config()
      if mp_cfg is not None and mp_cfg.enabled and mp_cfg.bake_weight_multipass_on_save:
        try:
          if hasattr(state, "params") and isinstance(state.params, dict):
            new_params, n_mod = apply_multipass_to_pytree(state.params, mp_cfg)
            if n_mod > 0 and hasattr(state, "replace"):
              state = state.replace(params=new_params)
        except Exception as e:  # pylint: disable=broad-exception-caught
          logging.warning(
              "[MultipassEmulation] Could not bake weight multi-pass before save_checkpoint: %s",
              e,
          )
      return orig_save(
          checkpoint_manager,
          step,
          state,
          config=config,
          data_iterator=data_iterator,
          force=force,
          *args,
          **kwargs,
      )

    checkpointing.save_checkpoint = _patched_save_checkpoint

  checkpointing._emulation_patched = True  # pylint: disable=protected-access


def _patch_vllm_and_eval() -> None:
  """Patches `MaxTextVllmRollout`/`MaxTextVllmSampler` and `VllmServerManager`."""
  try:
    from maxtext.integration.vllm import maxtext_vllm_rollout  # pylint: disable=import-outside-toplevel

    if not getattr(maxtext_vllm_rollout, "_emulation_patched", False):
      orig_sampler_load = maxtext_vllm_rollout.MaxTextVllmSampler.load_checkpoint
      orig_rollout_update = maxtext_vllm_rollout.MaxTextVllmRollout.update_params

      @functools.wraps(orig_sampler_load)
      def _patched_sampler_load_checkpoint(self, state, *args, **kwargs):
        mp_cfg = get_active_multipass_config()
        if (
            mp_cfg is not None
            and mp_cfg.enabled
            and (mp_cfg.apply_weight_multipass_on_load or mp_cfg.bake_weight_multipass_on_save)
            and not isinstance(state, str)
        ):
          state, _ = apply_multipass_to_pytree(state, mp_cfg)
        return orig_sampler_load(self, state, *args, **kwargs)

      @functools.wraps(orig_rollout_update)
      def _patched_rollout_update_params(self, params, filter_types=None):
        mp_cfg = (
            getattr(
                getattr(self, "_maxtext_config", None),
                "multipass_emulation",
                None,
            )
            or get_active_multipass_config()
        )
        if (
            mp_cfg is not None
            and mp_cfg.enabled
            and (mp_cfg.apply_weight_multipass_on_load or mp_cfg.bake_weight_multipass_on_save)
        ):
          params, _ = apply_multipass_to_pytree(params, mp_cfg)
        return orig_rollout_update(self, params, filter_types=filter_types)

      maxtext_vllm_rollout.MaxTextVllmSampler.load_checkpoint = _patched_sampler_load_checkpoint
      maxtext_vllm_rollout.MaxTextVllmRollout.update_params = _patched_rollout_update_params
      maxtext_vllm_rollout._emulation_patched = True  # pylint: disable=protected-access
  except ImportError:
    pass

  try:
    from maxtext.eval.runner import server_manager  # pylint: disable=import-outside-toplevel

    if not getattr(server_manager, "_emulation_patched", False):
      orig_start = server_manager.VllmServerManager.start

      @functools.wraps(orig_start)
      def _patched_vllm_server_start(self, vllm_kwargs, config):
        mp_cfg = getattr(config, "multipass_emulation", None) or get_active_multipass_config()
        if mp_cfg is not None and mp_cfg.enabled:
          add_cfg = vllm_kwargs.setdefault("additional_config", {})
          mt_cfg = add_cfg.setdefault("maxtext_config", {})
          if hasattr(config, "get_keys"):
            for k, v in config.get_keys().items():
              if (
                  _is_emulation_key(k)
                  and k not in ("multipass_emulation", "sparsity_emulation")
                  and v not in (None, "", [])
              ):
                mt_cfg[k] = v
        return orig_start(self, vllm_kwargs, config)  # pylint: disable=too-many-function-args

      server_manager.VllmServerManager.start = _patched_vllm_server_start
      server_manager._emulation_patched = True  # pylint: disable=protected-access
  except ImportError:
    pass


def disable_emulation_on_model(model: Any) -> Any:
  """Marks all modules inside an NNX `model` with `_emulation_disabled=True`."""
  try:
    from flax import nnx  # pylint: disable=import-outside-toplevel

    if isinstance(model, nnx.Module):
      for _, node in nnx.iter_graph(model):
        if isinstance(node, nnx.Module):
          object.__setattr__(node, "_emulation_disabled", True)
  except Exception:  # pylint: disable=broad-exception-caught
    pass
  return model


disable_sparsity_on_model = disable_emulation_on_model


def install_model_role_hook(workflow: str) -> None:
  """Installs multi-model role awareness (`rl`, `dpo`, `distill`) onto `from_pretrained`."""
  workflow_norm = workflow.strip().lower()
  if workflow_norm not in ("rl", "grpo", "dpo", "distill", "distillation"):
    return

  try:
    from maxtext.utils import model_creation_utils  # pylint: disable=import-outside-toplevel
  except ImportError:
    return

  orig_from_pretrained = getattr(
      model_creation_utils, "_orig_emulation_from_pretrained", model_creation_utils.from_pretrained
  )
  model_creation_utils._orig_emulation_from_pretrained = orig_from_pretrained  # pylint: disable=protected-access
  call_counter = {"count": 0}

  @functools.wraps(orig_from_pretrained)
  def _role_aware_from_pretrained(config, *args, **kwargs):
    call_counter["count"] += 1
    idx = call_counter["count"]
    explicit_role = str(getattr(config, "_emulation_model_role", "")).strip().lower()
    cfg = getattr(config, "multipass_emulation", None) or get_active_multipass_config()
    if cfg is None:
      cfg = extract_and_attach_multipass_config(config)

    if cfg is not None and cfg.enabled:
      # In MaxText RL/GRPO, DPO, and Distillation trainers, the primary trainable
      # policy/student model is instantiated first (idx == 1) and the frozen
      # reference/teacher model is instantiated second (idx == 2), unless an
      # explicit `_emulation_model_role` attribute is set on `config`.
      is_distill_teacher = explicit_role == "teacher" or (
          not explicit_role and workflow_norm in ("distill", "distillation") and idx == 2
      )
      is_ref_model = explicit_role in ("reference", "ref") or (
          not explicit_role and workflow_norm in ("rl", "grpo", "dpo") and idx == 2
      )

      if is_distill_teacher and not cfg.emulate_teacher_model:
        prev_load = cfg.apply_weight_multipass_on_load
        object.__setattr__(cfg, "apply_weight_multipass_on_load", False)
        try:
          teacher_model, mesh = orig_from_pretrained(config, *args, **kwargs)
        finally:
          object.__setattr__(cfg, "apply_weight_multipass_on_load", prev_load)
        disable_emulation_on_model(teacher_model)
        logging.info("[HardwareEmulation][Distill] Teacher model kept un-emulated (emulate_teacher_model=False).")
        return teacher_model, mesh

      if is_ref_model and not cfg.emulate_reference_model:
        prev_load = cfg.apply_weight_multipass_on_load
        object.__setattr__(cfg, "apply_weight_multipass_on_load", False)
        try:
          ref_model, mesh = orig_from_pretrained(config, *args, **kwargs)
        finally:
          object.__setattr__(cfg, "apply_weight_multipass_on_load", prev_load)
        disable_emulation_on_model(ref_model)
        logging.info(
            "[HardwareEmulation][%s] Reference model kept un-emulated (emulate_reference_model=False).",
            workflow_norm.upper(),
        )
        return ref_model, mesh

    return orig_from_pretrained(config, *args, **kwargs)

  model_creation_utils.from_pretrained = _role_aware_from_pretrained
  model_creation_utils._emulation_role_hook = workflow_norm  # pylint: disable=protected-access


def _patch_orbax_checkpointer() -> None:
  """Hooks Orbax `CheckpointManager.save` and Tunix `PeftTrainer` for step-gated execution."""
  try:
    import orbax.checkpoint as ocp  # pylint: disable=import-outside-toplevel

    if not getattr(ocp.CheckpointManager, "_emulation_patched", False):
      orig_ocp_save = ocp.CheckpointManager.save

      @functools.wraps(orig_ocp_save)
      def _patched_ocp_save(  # pylint: disable=keyword-arg-before-vararg
          self, step, items=None, save_kwargs=None, *args, **kwargs
      ):
        mp_cfg = get_active_multipass_config()
        if mp_cfg is not None and mp_cfg.enabled and mp_cfg.bake_weight_multipass_on_save:
          try:
            if items is not None and isinstance(items, dict):
              items, _ = apply_multipass_to_pytree(items, mp_cfg)
            if hasattr(self, "directory"):
              save_multipass_metadata(str(self.directory), mp_cfg, {"last_saved_step": step})
          except Exception:  # pylint: disable=broad-exception-caught
            pass
        return orig_ocp_save(self, step, items=items, save_kwargs=save_kwargs, *args, **kwargs)

      ocp.CheckpointManager.save = _patched_ocp_save
      ocp.CheckpointManager._emulation_patched = True  # pylint: disable=protected-access
  except ImportError:
    pass

  try:
    from tunix.sft import peft_trainer  # pylint: disable=import-outside-toplevel

    if not getattr(peft_trainer.PeftTrainer, "_emulation_patched", False):
      orig_init = peft_trainer.PeftTrainer.__init__
      orig_close = peft_trainer.PeftTrainer.close

      @functools.wraps(orig_init)
      def _patched_peft_init(self, *args, **kwargs):
        orig_create_fn = getattr(self, "create_train_step_fn", None)
        wrapped_create_fn = None
        if orig_create_fn is not None:

          @functools.wraps(orig_create_fn)
          def _wrapped_create_train_step_fn(*c_args, **c_kwargs):
            step_fn = orig_create_fn(*c_args, **c_kwargs)

            @functools.wraps(step_fn)
            def _step_gated_train_step(model, optimizer, *s_args, **s_kwargs):
              step_var = getattr(optimizer, "step", None)
              jax_step = None
              if step_var is not None:
                try:
                  jax_step = step_var.get_value() if hasattr(step_var, "get_value") else step_var[...]
                except Exception:  # pylint: disable=broad-exception-caught
                  jax_step = getattr(step_var, "value", None)
              if jax_step is None:
                try:
                  import jax  # pylint: disable=import-outside-toplevel

                  for raw_leaf in jax.tree.leaves(getattr(optimizer, "opt_state", None)):
                    leaf = raw_leaf.get_value() if hasattr(raw_leaf, "get_value") else raw_leaf
                    if (
                        hasattr(leaf, "dtype")
                        and hasattr(leaf, "ndim")
                        and leaf.ndim == 0
                        and str(leaf.dtype).startswith(("int", "uint"))
                    ):
                      jax_step = leaf
                      break
                except Exception:  # pylint: disable=broad-exception-caught
                  pass
              if not getattr(_step_gated_train_step, "_logged_trace", False):
                _step_gated_train_step._logged_trace = True  # pylint: disable=protected-access
                msg = "[HardwareEmulation] Step-gated train_step traced" f" (jax_step={type(jax_step).__name__})"
                logging.info("%s", msg)
                print(msg, flush=True)
              with emulation_jax_step_context(jax_step):
                return step_fn(model, optimizer, *s_args, **s_kwargs)

            return _step_gated_train_step

          wrapped_create_fn = _wrapped_create_train_step_fn
          self.create_train_step_fn = wrapped_create_fn

        res = orig_init(self, *args, **kwargs)
        if wrapped_create_fn is not None:
          self.create_train_step_fn = wrapped_create_fn
        orig_gen_fn = getattr(self, "gen_model_input_fn", None)
        if callable(orig_gen_fn):

          @functools.wraps(orig_gen_fn)
          def _wrapped_gen_model_input_fn(batch):
            if isinstance(batch, dict) and "_emulation_active" in batch:
              batch = {k: v for k, v in batch.items() if k != "_emulation_active"}
            return orig_gen_fn(batch)

          self.gen_model_input_fn = _wrapped_gen_model_input_fn
        return res

      @functools.wraps(orig_close)
      def _patched_peft_close(self, *args, **kwargs):
        mp_cfg = get_active_multipass_config()
        if mp_cfg is not None and mp_cfg.enabled and mp_cfg.bake_weight_multipass_on_save and hasattr(self, "model"):
          apply_multipass_to_model_weights(self.model, mp_cfg)
        try:
          return orig_close(self, *args, **kwargs)
        finally:
          try:
            import jax  # pylint: disable=import-outside-toplevel
            from jax._src import monitoring as jax_mon  # pylint: disable=import-outside-toplevel

            if hasattr(jax, "monitoring") and hasattr(jax.monitoring, "clear_event_listeners"):
              jax.monitoring.clear_event_listeners()
            if hasattr(jax_mon, "_scalar_listeners"):
              jax_mon._scalar_listeners.clear()  # pylint: disable=protected-access
          except Exception:  # pylint: disable=broad-exception-caught
            pass

      peft_trainer.PeftTrainer.__init__ = _patched_peft_init
      peft_trainer.PeftTrainer.close = _patched_peft_close
      orig_with_loss_fn = getattr(peft_trainer.PeftTrainer, "with_loss_fn", None)
      if callable(orig_with_loss_fn):

        @functools.wraps(orig_with_loss_fn)
        def _patched_with_loss_fn(self, loss_fn, *args, **kwargs):
          @functools.wraps(loss_fn)
          def _wrapped_loss_fn(model, *l_args, **l_kwargs):
            l_kwargs.pop("_emulation_active", None)
            return loss_fn(model, *l_args, **l_kwargs)

          return orig_with_loss_fn(self, _wrapped_loss_fn, *args, **kwargs)

        peft_trainer.PeftTrainer.with_loss_fn = _patched_with_loss_fn
      peft_trainer.PeftTrainer._emulation_patched = True  # pylint: disable=protected-access
  except ImportError:
    pass


def _patch_training_step_hooks() -> None:
  """Patches `BaseTrainingHooks`, `train_utils.jit_train_step`, and `MetricLogger` for step-gated execution."""
  tunix_step_override: dict[str, int | None] = {"step": None}

  try:
    from maxtext.utils import train_utils  # pylint: disable=import-outside-toplevel

    if not getattr(train_utils, "_emulation_patched", False):
      orig_jit_train_step = getattr(train_utils, "jit_train_step", None)
      if callable(orig_jit_train_step):

        @functools.wraps(orig_jit_train_step)
        def _patched_jit_train_step(
            config,
            model,
            state,
            state_mesh_shardings,
            data_sharding,
            train_step,
            params_shardings,
            mesh=None,
        ):
          @functools.wraps(train_step)
          def _step_gated_pretrain_step(
              m_arg,
              c_arg,
              sms_arg,
              ps_arg,
              state_arg,
              data_arg,
              *s_args,
              **s_kwargs,
          ):
            jax_step = None
            try:
              import jax  # pylint: disable=import-outside-toplevel

              opt_sub = (
                  state_arg.get("optimizer", state_arg)
                  if isinstance(state_arg, dict)
                  else getattr(state_arg, "optimizer", state_arg)
              )
              for raw_leaf in jax.tree.leaves(opt_sub):
                leaf = raw_leaf.get_value() if hasattr(raw_leaf, "get_value") else getattr(raw_leaf, "value", raw_leaf)
                if (
                    hasattr(leaf, "dtype")
                    and hasattr(leaf, "ndim")
                    and leaf.ndim == 0
                    and str(leaf.dtype).startswith(("int", "uint"))
                ):
                  jax_step = leaf
                  break
            except Exception:  # pylint: disable=broad-exception-caught
              pass
            if not getattr(_step_gated_pretrain_step, "_logged_trace", False):
              _step_gated_pretrain_step._logged_trace = True  # pylint: disable=protected-access
              msg = "[HardwareEmulation] Pre-train step-gated train_step traced" f" (jax_step={type(jax_step).__name__})"
              logging.info("%s", msg)
              print(msg, flush=True)
            with emulation_jax_step_context(jax_step):
              return train_step(
                  m_arg,
                  c_arg,
                  sms_arg,
                  ps_arg,
                  state_arg,
                  data_arg,
                  *s_args,
                  **s_kwargs,
              )

          return orig_jit_train_step(
              config,
              model,
              state,
              state_mesh_shardings,
              data_sharding,
              _step_gated_pretrain_step,
              params_shardings,
              mesh=mesh,
          )

        train_utils.jit_train_step = _patched_jit_train_step
        train_utils._emulation_patched = True  # pylint: disable=protected-access
  except ImportError:
    pass

  try:
    from maxtext.trainers.post_train import hooks as pt_hooks  # pylint: disable=import-outside-toplevel

    if not getattr(pt_hooks.BaseTrainingHooks, "_emulation_patched", False):
      orig_step_start = pt_hooks.BaseTrainingHooks.on_train_step_start
      orig_step_end = pt_hooks.BaseTrainingHooks.on_train_step_end
      orig_eval_start = pt_hooks.BaseTrainingHooks.on_eval_step_start

      @functools.wraps(orig_step_start)
      def _patched_on_train_step_start(self, train_ctx):
        step = int(getattr(train_ctx, "train_steps", 0))
        set_current_emulation_step(step)
        cfg = get_active_multipass_config()
        if cfg is not None and cfg.enabled:
          if cfg.start_step > 0 and step == cfg.start_step:
            msg = (
                f"[HardwareEmulation] Step {step} reached"
                f" (start_step={cfg.start_step}): activating hardware"
                f" emulation: {cfg.summary()}"
            )
            logging.info("%s", msg)
            print(msg, flush=True)
          elif cfg.end_step is not None and cfg.end_step >= 0 and step == cfg.end_step + 1:
            msg = (
                f"[HardwareEmulation] Step {step} reached (after"
                f" end_step={cfg.end_step}): deactivating hardware emulation."
            )
            logging.info("%s", msg)
            print(msg, flush=True)
        return orig_step_start(self, train_ctx)

      @functools.wraps(orig_step_end)
      def _patched_on_train_step_end(self, train_ctx, train_step, *args, **kwargs):
        prev_step = get_current_emulation_step()
        zero_idx_step = max(0, int(train_step) - 1)
        set_current_emulation_step(zero_idx_step)
        tunix_step_override["step"] = zero_idx_step
        try:
          return orig_step_end(self, train_ctx, train_step, *args, **kwargs)
        finally:
          tunix_step_override["step"] = None
          set_current_emulation_step(prev_step)

      @functools.wraps(orig_eval_start)
      def _patched_on_eval_step_start(self, train_ctx):
        step = int(getattr(train_ctx, "train_steps", 0))
        set_current_emulation_step(step)
        return orig_eval_start(self, train_ctx)

      pt_hooks.BaseTrainingHooks.on_train_step_start = _patched_on_train_step_start
      pt_hooks.BaseTrainingHooks.on_train_step_end = _patched_on_train_step_end
      pt_hooks.BaseTrainingHooks.on_eval_step_start = _patched_on_eval_step_start
      pt_hooks.BaseTrainingHooks._emulation_patched = True  # pylint: disable=protected-access

    if not getattr(pt_hooks.BaseDataHooks, "_emulation_patched", False):
      orig_load_next = pt_hooks.BaseDataHooks.load_next_train_batch
      orig_load_eval = pt_hooks.BaseDataHooks.load_next_eval_batch

      @functools.wraps(orig_load_next)
      def _patched_load_next_train_batch(self, train_ctx):
        if os.environ.get("EMULATION_REPEAT_FIRST_BATCH") == "1":
          cached = getattr(self, "_emulation_cached_first_batch", None)
          if cached is not None:
            self.train_batch = cached
            return cached
          batch = orig_load_next(self, train_ctx)
          if batch is not None:
            self._emulation_cached_first_batch = batch  # pylint: disable=protected-access
          return batch
        return orig_load_next(self, train_ctx)

      @functools.wraps(orig_load_eval)
      def _patched_load_next_eval_batch(self, train_ctx):
        step = int(getattr(train_ctx, "train_steps", 0))
        set_current_emulation_step(step)
        eval_idx = int(train_ctx.training_hooks.eval_metadata.get("eval_step_count", 0))
        eval_cache = getattr(self, "_emulation_eval_batches", None)
        if eval_cache is None:
          eval_cache = []
          self._emulation_eval_batches = eval_cache  # pylint: disable=protected-access
        cache_complete = getattr(self, "_emulation_eval_batches_complete", False)

        if eval_idx < len(eval_cache):
          raw_batch = eval_cache[eval_idx]
        elif not cache_complete:
          raw_batch = orig_load_eval(self, train_ctx)
          if raw_batch is not None and isinstance(raw_batch, dict):
            clean_batch = {k: v for k, v in raw_batch.items() if k != "_emulation_active"}
            eval_cache.append(clean_batch)
            raw_batch = clean_batch
          else:
            if eval_cache:
              self._emulation_eval_batches_complete = True  # pylint: disable=protected-access
            raw_batch = None
        else:
          raw_batch = None

        if raw_batch is not None and isinstance(raw_batch, dict):
          batch = dict(raw_batch)
          cfg = get_active_multipass_config()
          if cfg is not None and cfg.enabled and cfg.is_active_at_step(step):
            if "inputs" in batch:
              batch["_emulation_active"] = batch["inputs"][:, :1]
          self.eval_batch = batch
          return batch

        self.eval_batch = None
        return None

      pt_hooks.BaseDataHooks.load_next_train_batch = _patched_load_next_train_batch
      pt_hooks.BaseDataHooks.load_next_eval_batch = _patched_load_next_eval_batch
      pt_hooks.BaseDataHooks._emulation_patched = True  # pylint: disable=protected-access
  except ImportError:
    pass

  try:
    from maxtext.common import metric_logger  # pylint: disable=import-outside-toplevel

    if not getattr(metric_logger.MetricLogger, "_emulation_patched", False):
      orig_init = metric_logger.MetricLogger.__init__
      orig_write_metrics = metric_logger.MetricLogger.write_metrics

      @functools.wraps(orig_init)
      def _patched_metric_logger_init(self, config, *args, **kwargs):
        if getattr(config, "enable_wandb", False):
          if not os.environ.get("WANDB_API_KEY"):
            os.environ.setdefault("WANDB_MODE", "offline")
        try:
          orig_init(self, config, *args, **kwargs)
        except Exception as exc:  # pylint: disable=broad-exception-caught
          if getattr(config, "enable_wandb", False) and ("No API key configured" in str(exc)):
            os.environ["WANDB_MODE"] = "offline"
            orig_init(self, config, *args, **kwargs)
          else:
            raise
        if getattr(config, "enable_wandb", False):
          try:
            import wandb  # pylint: disable=import-outside-toplevel

            if getattr(wandb, "run", None) is not None:
              get_url_fn = getattr(wandb.run, "get_url", None)
              url = (
                  getattr(wandb.run, "url", None)
                  or (get_url_fn() if callable(get_url_fn) else None)
                  or getattr(wandb.run, "dir", "offline")
              )
              msg = f"[HardwareEmulation] W&B Run URL: {url}"
              logging.info("%s", msg)
              print(msg, flush=True)
          except Exception:  # pylint: disable=broad-exception-caught
            pass

      @functools.wraps(orig_write_metrics)
      def _patched_write_metrics(self, metrics, step, metric_type="train"):
        cfg = get_active_multipass_config()
        if isinstance(metrics, dict):
          scalars = metrics.setdefault("scalar", {})
          if metric_type == "train" and cfg is not None and cfg.enabled:
            if tunix_step_override["step"] is not None:
              cur_step = tunix_step_override["step"]
            else:
              cur_step = int(step)
              set_current_emulation_step(cur_step)
              if cfg.start_step > 0 and cur_step == cfg.start_step:
                msg = (
                    f"[HardwareEmulation] Step {cur_step} reached"
                    f" (start_step={cfg.start_step}): activating hardware"
                    f" emulation: {cfg.summary()}"
                )
                logging.info("%s", msg)
                print(msg, flush=True)
            is_act = cfg.is_active_at_step(cur_step)
            scalars["learning/emulation_active"] = 1.0 if is_act else 0.0
            if cfg.rules:
              r0 = cfg.rules[0]
              scalars["learning/emulation_lhs_passes"] = float(
                  r0.lhs.num_passes if (is_act and r0.lhs and r0.lhs.enabled) else 1
              )
              scalars["learning/emulation_rhs_passes"] = float(
                  r0.rhs.num_passes if (is_act and r0.rhs and r0.rhs.enabled) else 1
              )
              scalars["learning/emulation_active_gemm_pairs"] = float(len(r0.effective_fwd_pairs) if is_act else 1)
          elif metric_type == "eval":
            if "eval/avg_loss" in scalars:
              avg_l = float(scalars["eval/avg_loss"])
              scalars["eval/avg_sft_loss"] = avg_l
              scalars["eval/avg_perplexity"] = float(math.exp(max(-20.0, min(20.0, avg_l))))
              scalars["eval/token_accuracy_prob"] = float(math.exp(-max(0.0, min(20.0, avg_l))))
            if cfg is not None and cfg.enabled:
              scalars["eval/emulation_active"] = 1.0 if cfg.is_active_at_step(int(step)) else 0.0
        return orig_write_metrics(self, metrics, step, metric_type=metric_type)

      orig_flush = getattr(metric_logger.MetricLogger, "flush_metrics_and_cleanup", None)
      if callable(orig_flush):

        @functools.wraps(orig_flush)
        def _patched_flush_metrics_and_cleanup(self, *args, **kwargs):
          orig_flush(self, *args, **kwargs)
          if getattr(self.config, "enable_wandb", False):
            try:
              import wandb  # pylint: disable=import-outside-toplevel

              if getattr(wandb, "run", None) is not None:
                get_url_fn = getattr(wandb.run, "get_url", None)
                url = (
                    getattr(wandb.run, "url", None)
                    or (get_url_fn() if callable(get_url_fn) else None)
                    or getattr(wandb.run, "dir", "offline")
                )
                msg = f"[HardwareEmulation] W&B Run URL: {url}"
                logging.info("%s", msg)
                print(msg, flush=True)
                wandb.finish()
            except Exception:  # pylint: disable=broad-exception-caught
              pass

        metric_logger.MetricLogger.flush_metrics_and_cleanup = _patched_flush_metrics_and_cleanup

      metric_logger.MetricLogger.__init__ = _patched_metric_logger_init
      metric_logger.MetricLogger.write_metrics = _patched_write_metrics
      metric_logger.MetricLogger._emulation_patched = True  # pylint: disable=protected-access
  except ImportError:
    pass


def install_emulation_hooks(workflow: str | None = None) -> None:
  """Installs all hardware emulation runtime hooks (idempotent).

  Args:
    workflow: Optional workflow name (`"pretrain"`, `"sft"`, `"rl"`, `"dpo"`,
      `"distill"`, `"eval"`). When set to `"rl"`, `"dpo"`, or `"distill"`,
      installs multi-model role hooks so Teacher/Reference models respect
      `emulate_teacher_model` / `emulate_reference_model`.
  """
  global _HOOKS_INSTALLED
  if not _HOOKS_INSTALLED:
    _patch_pyconfig()
    _patch_model_creation()
    _patch_checkpointing()
    _patch_orbax_checkpointer()
    _patch_training_step_hooks()
    _patch_vllm_and_eval()
    install_layer_patches()
    _HOOKS_INSTALLED = True
  if workflow:
    install_model_role_hook(workflow)
