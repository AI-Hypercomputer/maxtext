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

"""MaxText TPU inference runtime for Laya System 1 decision models (Agent & Router)."""

import json
import os
import threading
from typing import Any, Dict, Optional

from flax import nnx
from huggingface_hub import snapshot_download
import jax
import jax.numpy as jnp
import numpy as np
from safetensors.torch import load_file

from laya.agent import Agent as HFLayaAgent, _fix_tokenizer_config, _load_tokenizer
from laya.common import clamp_temperature
from laya.hooks import normalise_hooks
from laya.router import Router as HFLayaRouter, _split, normalise_name

from maxtext.checkpoint_conversion.utils.param_mapping import (
    LAYA_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    LAYA_MAXTEXT_TO_HF_PARAM_MAPPING,
)
from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.utils import max_logging
from maxtext.utils import maxtext_utils
from maxtext.utils import model_creation_utils
from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR


@nnx.jit
def _jit_laya_forward(
    model: Any,
    input_ids: jax.Array,
    positions: jax.Array,
    segment_ids: jax.Array,
    qtype: jax.Array,
    marker_pos: jax.Array,
    marker_mask: jax.Array,
) -> tuple[jax.Array, jax.Array]:
  """JIT-compiled MaxText NNXDecoder + LayaDecisionHead forward pass on TPU."""
  decoder = model.decoder
  y = decoder._apply_embedding(  # pylint: disable=protected-access
      shared_embedding=model.token_embedder,
      decoder_input_tokens=input_ids,
      decoder_positions=positions,
      deterministic=True,
      model_mode=MODEL_MODE_TRAIN,
  )
  for i in range(decoder.config.base_num_decoder_layers):
    layer = getattr(decoder, f"layers_{i}")
    y, _ = layer(
        y,
        decoder_segment_ids=segment_ids,
        decoder_positions=positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )
  hidden_states = decoder.decoder_norm(y)
  opt_logits, act_logits = decoder.decision_head(
      enc_hidden_states=hidden_states,
      attention_mask=segment_ids,
      qtype=qtype,
      marker_pos=marker_pos,
      marker_mask=marker_mask,
  )
  act_probs = jax.nn.softmax(act_logits.astype(jnp.float32), axis=-1)
  return opt_logits.astype(jnp.float32), act_probs


def _resolve_maxtext_model_name(model_id_or_path: str, subfolder: Optional[str]) -> str:
  """Maps a Laya repo/path and subfolder to its MaxText model_name."""
  sub = (subfolder or "").strip("/").lower()
  path_low = str(model_id_or_path).lower()
  if sub == "multilingual" or "multilingual" in path_low:
    return "laya-multilingual"
  if sub in ("typed-decisions", "typed_decisions") or "typed-decisions" in path_low:
    return "laya-typed-decisions"
  return "laya"


def _load_safetensors_into_nnx_model(nnx_model: Any, weights_path: str, mt_config: Any) -> None:
  """Populates an initialized MaxText NNX Laya model from a safetensors file using PARAM_MAPPING."""
  raw_sd = load_file(weights_path)
  hf_cfg_dict = {"num_hidden_layers": mt_config.base_num_decoder_layers}
  param_map = LAYA_MAXTEXT_TO_HF_PARAM_MAPPING(hf_cfg_dict, mt_config, scan_layers=False)
  hook_map = LAYA_MAXTEXT_TO_HF_PARAM_HOOK_FN(hf_cfg_dict, mt_config, scan_layers=False, saving_to_hf=False)

  _, state = nnx.split(nnx_model)
  for path_tuple, param_var in nnx.to_flat_state(state):
    if not isinstance(param_var, nnx.Param):
      continue
    mt_key = "params-" + "-".join(str(p) for p in path_tuple)
    if mt_key not in param_map:
      raise KeyError(f"MaxText parameter key {mt_key} missing from LAYA_MAXTEXT_TO_HF_PARAM_MAPPING")
    hf_key = param_map[mt_key]
    if hf_key not in raw_sd:
      raise KeyError(f"HF parameter key {hf_key} (mapped from {mt_key}) missing from {weights_path}")
    arr = raw_sd[hf_key].detach().cpu().float().numpy()
    hooks = hook_map.get(mt_key)
    if hooks is not None:
      if not isinstance(hooks, list):
        hooks = [hooks]
      for fn in hooks:
        arr = fn(arr, param_var.value.shape)
    sharding = getattr(param_var.value, "sharding", None)
    jax_arr = jnp.asarray(arr, dtype=param_var.value.dtype)
    if sharding is not None:
      jax_arr = jax.device_put(jax_arr, sharding)
    param_var.value = jax_arr
  nnx.update(nnx_model, state)


class Agent(HFLayaAgent):
  """MaxText TPU runtime for Laya System 1 decision models."""

  def __init__(  # pylint: disable=super-init-not-called
      self,
      model_id_or_path: str = "convaiinnovations/laya",
      device: Optional[str] = None,
      token: Optional[str] = None,
      subfolder: Optional[str] = None,
      fast: bool = False,
      compile: bool = False,  # pylint: disable=redefined-builtin
      lang_temperatures: Optional[Dict[str, Dict[str, Any]]] = None,
      hooks=None,
      on_predict_start=None,
      on_predict_end=None,
      hooks_raise: bool = True,
      hooks_concurrent: bool = True,
      orbax_checkpoint_path: Optional[str] = None,
  ):
    self.hooks = normalise_hooks(hooks, on_predict_start, on_predict_end)
    self.hooks_raise = bool(hooks_raise)
    self.hooks_concurrent = bool(hooks_concurrent)
    self._hooks_lock = threading.RLock() if not hooks_concurrent else None
    self._hooks_mutex = threading.Lock()
    self.model_id = model_id_or_path
    self.device = "tpu"
    self._fast = None

    model_dir = model_id_or_path
    if not os.path.exists(model_dir):
      if model_id_or_path.startswith(("/", "./", "../")) or os.path.isabs(model_id_or_path):
        raise FileNotFoundError(f"Local model path not found: {model_id_or_path!r}.")
      prefix = f"{subfolder}/" if subfolder else ""
      kw = {
          "token": token or os.environ.get("HF_TOKEN") or None,
          "allow_patterns": [
              prefix + name
              for name in (
                  "rl_agent_config.json",
                  "model.safetensors",
                  "tokenizer/*",
                  "encoder/*",
              )
          ],
      }
      model_dir = snapshot_download(model_id_or_path, **kw)

    if subfolder:
      cand = os.path.join(model_dir, subfolder)
      if os.path.isdir(cand):
        model_dir = cand

    _fix_tokenizer_config(model_dir)

    cfg_path = os.path.join(model_dir, "rl_agent_config.json")
    if not os.path.exists(cfg_path):
      raise FileNotFoundError(f"Incompatible model: {model_dir!r} does not contain 'rl_agent_config.json'.")
    with open(cfg_path, "rt", encoding="utf8") as f:
      self.cfg = json.load(f)

    weights_path = os.path.join(model_dir, "model.safetensors")
    if not os.path.exists(weights_path):
      raise FileNotFoundError(f"Incompatible model: 'model.safetensors' not found in {model_dir!r}.")

    tok_dir = os.path.join(model_dir, "tokenizer")
    self.tok = _load_tokenizer(tok_dir, self.cfg)

    self.temperature_raw = self.cfg.get("temperature", [1.0, 1.0, 1.0])
    self.temperature_by_options_raw = self.cfg.get("temperature_by_options", {})
    self.temperature = [clamp_temperature(t) for t in self.temperature_raw]
    self.temperature_by_options = {k: clamp_temperature(v) for k, v in self.temperature_by_options_raw.items()}

    self.lang_temperatures = {}
    for l, lcfg in (lang_temperatures or {}).items():
      norm_l = l.split("-")[0].lower()
      t_raw = lcfg.get("temperature", self.temperature_raw)
      tbo_raw = lcfg.get("temperature_by_options", {})
      self.lang_temperatures[norm_l] = {
          "temperature": [clamp_temperature(t) for t in t_raw],
          "temperature_by_options": {k: clamp_temperature(v) for k, v in tbo_raw.items()},
      }

    self.maxtext_model_name = _resolve_maxtext_model_name(model_id_or_path, subfolder)
    if orbax_checkpoint_path is None:
      default_orbax = f"/dev/shm/hengtaoguo/checkpoints/{self.maxtext_model_name}_orbax/0/items"
      if os.path.exists(default_orbax):
        orbax_checkpoint_path = default_orbax

    base_yml = os.path.join(MAXTEXT_CONFIGS_DIR, "base.yml")
    argv = [
        "",
        base_yml,
        f"model_name={self.maxtext_model_name}",
        "scan_layers=false",
        "dtype=float32",
        "weight_dtype=float32",
        "matmul_precision=highest",
        "float32_qk_product=true",
        "float32_logits=true",
        "per_device_batch_size=1",
        "max_prefill_predict_length=512",
        "max_target_length=512",
        "async_checkpointing=false",
    ]
    if orbax_checkpoint_path:
      argv.append(f"load_parameters_path={orbax_checkpoint_path}")

    self.mt_config = pyconfig.initialize(argv)
    devices_array = maxtext_utils.create_device_mesh(self.mt_config)
    self.mesh = jax.sharding.Mesh(devices_array, self.mt_config.mesh_axes)

    if orbax_checkpoint_path:
      max_logging.log(
          f"Loading MaxText Laya model ({self.maxtext_model_name}) from Orbax checkpoint: {orbax_checkpoint_path}"
      )
      self.model = model_creation_utils.from_pretrained(self.mt_config, mesh=self.mesh, model_mode=MODEL_MODE_TRAIN)
    else:
      max_logging.log(f"Loading MaxText Laya model ({self.maxtext_model_name}) from safetensors: {weights_path}")
      self.model = model_creation_utils.create_nnx_model(self.mt_config, mesh=self.mesh, model_mode=MODEL_MODE_TRAIN)
      _load_safetensors_into_nnx_model(self.model, weights_path, self.mt_config)

  def _forward(self, b: Dict):
    """Runs the MaxText NNXDecoder + LayaDecisionHead forward pass on TPU."""
    input_ids_np = b["input_ids"].detach().cpu().numpy().astype(np.int32)
    attention_mask_np = b["attention_mask"].detach().cpu().numpy().astype(np.int32)
    marker_pos_np = b["marker_pos"].detach().cpu().numpy().astype(np.int32)
    marker_mask_np = b["marker_mask"].detach().cpu().numpy().astype(np.bool_)
    qtype_np = b["qtype"].detach().cpu().numpy().astype(np.int32)

    orig_batch, seq_len = input_ids_np.shape
    num_devices = jax.device_count()
    if orig_batch % num_devices != 0:
      pad_rows = num_devices - (orig_batch % num_devices)
      input_ids_np = np.pad(input_ids_np, ((0, pad_rows), (0, 0)), mode="edge")
      attention_mask_np = np.pad(attention_mask_np, ((0, pad_rows), (0, 0)), mode="edge")
      marker_pos_np = np.pad(marker_pos_np, ((0, pad_rows), (0, 0)), mode="edge")
      marker_mask_np = np.pad(marker_mask_np, ((0, pad_rows), (0, 0)), mode="edge")
      qtype_np = np.pad(qtype_np, ((0, pad_rows),), mode="edge")

    padded_batch = input_ids_np.shape[0]
    positions_np = np.broadcast_to(np.arange(seq_len, dtype=np.int32)[None, :], (padded_batch, seq_len))

    opt_logits_j, act_probs_j = _jit_laya_forward(
        self.model,
        jnp.asarray(input_ids_np),
        jnp.asarray(positions_np),
        jnp.asarray(attention_mask_np),
        jnp.asarray(qtype_np),
        jnp.asarray(marker_pos_np),
        jnp.asarray(marker_mask_np),
    )
    opt_logits_out = np.asarray(opt_logits_j)[:orig_batch]
    act_probs_out = np.asarray(act_probs_j)[:orig_batch]
    return opt_logits_out, act_probs_out


RLAgent = Agent


class Router(HFLayaRouter):
  """MaxText TPU Router that lazily loads MaxText Laya Agents and routes requests."""

  def load(self, name: str) -> Agent:
    """Returns the MaxText TPU Agent for `name`, loading it on first use."""
    key = normalise_name(name)
    with self._lock:
      if key in self._agents:
        self._touch(key)
        return self._agents[key]
      repo, sub = _split(self.models[key])
      agent = Agent(repo, device=self.device, token=self.token, subfolder=sub)
      self._agents[key] = agent
      self._order.append(key)
      evicted = self._evict_locked()
    self._dispatch_lifecycle("on_evict", evicted)
    return agent


def load(
    model_id_or_path: str = "convaiinnovations/laya",
    device: Optional[str] = None,
    token: Optional[str] = None,
    subfolder: Optional[str] = None,
    fast: bool = False,
    lang_temperatures: Optional[Dict[str, Dict[str, Any]]] = None,
    hooks=None,
    on_predict_start=None,
    on_predict_end=None,
    hooks_raise: bool = True,
    hooks_concurrent: bool = True,
    orbax_checkpoint_path: Optional[str] = None,
) -> Agent:
  """Loads a MaxText TPU Laya Agent."""
  return Agent(
      model_id_or_path,
      device=device,
      token=token,
      subfolder=subfolder,
      fast=fast,
      lang_temperatures=lang_temperatures,
      hooks=hooks,
      on_predict_start=on_predict_start,
      on_predict_end=on_predict_end,
      hooks_raise=hooks_raise,
      hooks_concurrent=hooks_concurrent,
      orbax_checkpoint_path=orbax_checkpoint_path,
  )


def main() -> None:
  """Runs the golden Laya example (/home/hengtaoguo_google_com/projects/laya.py) on TPU using MaxText."""
  router = Router()

  state = "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."
  questions = {
      "department": {
          "type": "choice",
          "instructions": "Which department should handle this?",
          "criteria": {
              "billing": "invoices, payments, refunds",
              "technical": "bugs, outages, system errors",
              "other": "everything else",
          },
      },
      "urgency": {
          "type": "score",
          "instructions": "How urgent is this?",
          "criteria": ["not urgent", "soon", "blocking"],
      },
      "churn_risk": {
          "type": "noul",
          "instructions": "Does the user threaten to cancel or leave?",
      },
  }

  result = router.predict(state, questions)
  dept = result["answers"]["department"]["choice"]
  churn = result["answers"]["churn_risk"]["noul"]
  routed_model = result["routing"]["model"]
  print(dept)  # billing
  print(churn)  # 0.879
  print(routed_model)  # english
  print("Full MaxText TPU Router result:", json.dumps(result, indent=2))

  assert dept == "billing", f"Expected 'billing', got {dept!r}"
  assert abs(churn - 0.879) < 1e-3, f"Expected churn_risk.noul == 0.879, got {churn}"
  assert routed_model == "english", f"Expected 'english', got {routed_model!r}"


if __name__ == "__main__":
  main()
