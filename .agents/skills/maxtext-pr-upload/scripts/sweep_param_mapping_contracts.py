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

"""Deterministic dry-run sweep for MaxText PARAM_MAPPING & HF streaming contracts.

Exercises cross-model and sharding contracts that single-model 4-CPU unit tests
miss when reviewing or authoring changes to `param_mapping.py`, `hf_shape.py`,
`hf_model_configs.py`, `load_dynamic.py`, or `hf_streaming_load.py`:

1. BF16 / Null-Subconfig `hf_config_dict` Sweep (`BF16_CONFIG_MAPPING_CRASH`,
   `NULL_SUBCONFIG_JSON_CRASH`):
   Sweeps every registered model in `PARAM_MAPPING` and `HOOK_FNS` (both
   `scan_layers=False` and `scan_layers=True`) under:
   - Stock `HF_MODEL_CONFIGS[model].to_dict()`,
   - An unquantized checkpoint `config.json` (no `quantization_config` key),
     catching `hf_config_dict['quantization_config'] = None` mutations that
     crash downstream `.get('quantization_config', {}).get(...)` hooks,
   - A checkpoint `config.json` containing `{"text_config": null}`.
2. Unscaled FP8-to-BF16 Silent Cast Gate (`UNSCALED_FP8_TO_BF16_SILENT_CAST`):
   When `hf_streaming_load` is present, verifies that a lone FP8 safetensors
   source mapped into a `bfloat16` target weight without `.scale` keys is
   rejected loudly via `_check_float8_sources` rather than silently cast via
   `astype`.
3. Padded Multi-Key `(weight, scale)` Split & Composite Tuple Gate
   (`PADDED_TUPLE_SPLIT_DEGRADATION`, `COMPOSITE_TUPLE_SPLIT_CRASH`):
   When `hf_streaming_load` is present, verifies that `_load_request` computes
   `shared_split_factor` AFTER `_padded_rows` (splitting padded `(324096, 1280)`
   + `(324096, 10)` across all 64 and 12 devices instead of collapsing to 8-way
   or failing on unpadded `324000`), and that `shared_split_factor` handles
   mismatched-row composite QKV tuples without crashing.

Exit code: 0 = PASS, 1 = contract violation found, 2 = usage/import error.
"""

import argparse
from collections.abc import Sequence
import importlib.util
import inspect
import json
import os
import re
import sys
import tempfile
import types
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")


def _default_repo_root() -> str:
  """Resolves MaxText repository root from __file__ ancestors or CWD."""
  starts = (
      os.path.dirname(os.path.abspath(__file__)),
      os.path.abspath(os.getcwd()),
  )
  for start in starts:
    cur = start
    while True:
      if os.path.isfile(os.path.join(cur, "src", "maxtext", "configs", "types.py")):
        return cur
      parent = os.path.dirname(cur)
      if parent == cur:
        break
      cur = parent
  return os.path.abspath(os.getcwd())


class _PermissiveMaxTextConfig:
  """Lightweight mock MaxText config providing safe defaults for mapping dry-runs."""

  _DEFAULTS: dict[str, Any] = {
      "scan_layers": False,
      "param_scan_axis": 1,
      "weight_dtype": "bfloat16",
      "dtype": "bfloat16",
      "quantization": "",
      "megablox": False,
      "sparse_matmul": False,
      "inhomogeneous_layer_cycle_interval": 4,
      "num_experts": 8,
      "num_experts_per_tok": 2,
      "base_num_decoder_layers": 4,
      "num_Unhandled": 0,
      "mtp_num_layers": 1,
      "engram_num_layers": 0,
      "engram_layers": [],
      "logits_via_embedding": False,
      "tie_word_embeddings": False,
      "decoder_block": "llama2",
      "attention_type": "global",
      "mla_naive_kvcache": False,
      "use_multimodal": True,
      "use_audio": True,
      "use_qwix_quantization": False,
  }

  def __init__(
      self,
      model_name: str,
      scan_layers: bool,
      yml_defaults: dict[str, Any] | None = None,
  ) -> None:
    if yml_defaults:
      for k, v in yml_defaults.items():
        setattr(self, k, v)
    self.model_name = model_name
    self.scan_layers = scan_layers
    self._sync_derived_fields()

  def _sync_derived_fields(self) -> None:
    """Populates derived architectural dimensions from `base_*` attributes."""
    for derived_k, base_k in (
        ("num_decoder_layers", "base_num_decoder_layers"),
        ("emb_dim", "base_emb_dim"),
        ("mlp_dim", "base_mlp_dim"),
        ("moe_mlp_dim", "base_moe_mlp_dim"),
        ("num_query_heads", "base_num_query_heads"),
        ("num_kv_heads", "base_num_kv_heads"),
    ):
      cur = self.__dict__.get(derived_k)
      base_val = self.__dict__.get(base_k, self._DEFAULTS.get(base_k))
      if (cur is None or (isinstance(cur, int) and cur < 0)) and isinstance(base_val, int):
        setattr(self, derived_k, base_val)

  def __getattr__(self, name: str) -> Any:
    if name in self._DEFAULTS:
      return self._DEFAULTS[name]
    if name.startswith(("use_", "enable_", "is_", "has_", "tie_")):
      return False
    if name.endswith(("_ratios", "_types", "_schedule")):
      n_layers = int(self.__dict__.get("base_num_decoder_layers", 4))
      return [0] * max(1, n_layers)
    if name.endswith(
        (
            "_source_layers",
            "_engram_layers",
            "_moe_layers",
            "_dense_layers",
            "_indices",
        )
    ):
      return []
    if name.endswith(("_layers", "_experts", "_heads", "_dim", "_size", "_rank")):
      return 4
    return 1


class _FakeDevice:
  """Minimal mock JAX device supporting `.id` and `.process_index`."""

  def __init__(self, dev_id: int, process_index: int = 0) -> None:
    self.id = dev_id
    self.process_index = process_index

  def __repr__(self) -> str:
    return f"FakeDevice(id={self.id}, process={self.process_index})"

  def __hash__(self) -> int:
    return hash((self.id, self.process_index))

  def __eq__(self, other: object) -> bool:
    return isinstance(other, _FakeDevice) and self.id == other.id and self.process_index == other.process_index


class _FakeSharding:
  """Minimal mock sharding carrying a device set."""

  def __init__(self, devices: tuple[Any, ...]) -> None:
    self._devices = devices
    self.device_set = set(devices)


def _import_maxtext_modules(maxtext_root: str) -> dict[str, Any]:
  """Imports checkpoint conversion modules from `maxtext_root/src`."""
  src_dir = os.path.join(os.path.abspath(maxtext_root), "src")
  if os.path.isdir(src_dir) and src_dir not in sys.path:
    sys.path.insert(0, src_dir)
  if os.path.abspath(maxtext_root) not in sys.path:
    sys.path.insert(0, os.path.abspath(maxtext_root))

  from maxtext.checkpoint_conversion.utils import hf_model_configs  # pylint: disable=import-outside-toplevel
  from maxtext.checkpoint_conversion.utils import hf_shape  # pylint: disable=import-outside-toplevel
  from maxtext.checkpoint_conversion.utils import load_dynamic  # pylint: disable=import-outside-toplevel
  from maxtext.checkpoint_conversion.utils import param_mapping  # pylint: disable=import-outside-toplevel

  mods: dict[str, Any] = {
      "hf_model_configs": hf_model_configs,
      "hf_shape": hf_shape,
      "load_dynamic": load_dynamic,
      "param_mapping": param_mapping,
      "hf_streaming_load": None,
  }
  try:
    from maxtext.checkpoint_conversion.utils import hf_streaming_load  # pylint: disable=import-outside-toplevel

    mods["hf_streaming_load"] = hf_streaming_load
  except ImportError:
    pass
  return mods


def _load_yaml_file(path: str) -> dict[str, Any]:
  """Loads a YAML mapping file if PyYAML is installed and the path exists."""
  if not os.path.isfile(path):
    return {}
  try:
    import yaml  # pylint: disable=import-outside-toplevel
  except ImportError:
    return {}
  try:
    with open(path, encoding="utf-8") as f:
      data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}
  except (OSError, ValueError, yaml.YAMLError):
    return {}


def _load_model_yaml_defaults(
    maxtext_root: str,
    model_key: str,
    base_cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
  """Returns merged `base.yml` + `models/<model_key>.yml` when the model YAML exists."""
  cfg_root = os.path.join(os.path.abspath(maxtext_root), "src/maxtext/configs")
  model_yml_path = os.path.join(cfg_root, "models", f"{model_key}.yml")
  if not os.path.isfile(model_yml_path):
    return {}
  if "base" not in base_cache:
    base_cache["base"] = _load_yaml_file(os.path.join(cfg_root, "base.yml"))

  def _read_chain(yml_path: str, seen: set[str]) -> dict[str, Any]:
    real = os.path.realpath(yml_path)
    if real in seen or not os.path.isfile(real):
      return {}
    seen.add(real)
    cur = _load_yaml_file(real)
    parent_ref = cur.get("base_config")
    merged: dict[str, Any] = {}
    if isinstance(parent_ref, str) and parent_ref.strip():
      cand = os.path.join(os.path.dirname(real), parent_ref.strip())
      if not os.path.isfile(cand):
        cand = os.path.join(cfg_root, parent_ref.strip())
      merged.update(_read_chain(cand, seen))
    merged.update(cur)
    return merged

  out = dict(base_cache["base"])
  out.update(_read_chain(model_yml_path, set()))
  return out


def _populate_mt_cfg_from_hf(
    mt_cfg: _PermissiveMaxTextConfig,
    hf_dict: dict[str, Any],
    has_model_yml: bool = False,
) -> None:
  """Copies key architectural counts and schedules from `hf_dict` onto `mt_cfg`."""
  tc = hf_dict.get("text_config")
  src = tc if isinstance(tc, dict) else hf_dict
  if isinstance(src, dict):
    if not has_model_yml and "num_hidden_layers" in src and isinstance(src["num_hidden_layers"], int):
      mt_cfg.base_num_decoder_layers = src["num_hidden_layers"]
      mt_cfg.num_decoder_layers = src["num_hidden_layers"]
    for hf_k, mt_k in (
        ("num_experts", "num_experts"),
        ("n_routed_experts", "num_experts"),
        ("mtp_num_hidden_layers", "mtp_num_layers"),
        ("num_nextn_predict_layers", "mtp_num_layers"),
        ("full_attention_interval", "inhomogeneous_layer_cycle_interval"),
    ):
      if not has_model_yml and isinstance(src.get(hf_k), int):
        setattr(mt_cfg, mt_k, src[hf_k])
    for hf_k, mt_k in (
        ("compress_ratios", "compress_ratios"),
        ("kv_source_layers", "ds41_kv_source_layers"),
        ("index_source_layers", "ds41_index_source_layers"),
        ("engram_layers", "ds41_engram_layers"),
        ("layer_types", "layer_types"),
    ):
      if mt_k not in mt_cfg.__dict__ and isinstance(src.get(hf_k), list):
        setattr(mt_cfg, mt_k, list(src[hf_k]))
    mt_cfg._sync_derived_fields()  # pylint: disable=protected-access


def _sweep_param_mappings(
    mods: dict[str, Any],
    selected_models: set[str] | None,
    maxtext_root: str = ".",
) -> tuple[list[dict[str, Any]], int]:
  """Sweeps PARAM_MAPPING, HOOK_FNS, and load_dynamic config overrides."""
  issues: list[dict[str, Any]] = []
  pm_mod = mods["param_mapping"]
  hm_mod = mods["hf_model_configs"]
  ld_mod = mods["load_dynamic"]

  param_map_reg: dict[str, Any] = pm_mod.PARAM_MAPPING
  hook_reg: dict[str, Any] = pm_mod.HOOK_FNS
  hf_cfgs: dict[str, Any] = hm_mod.HF_MODEL_CONFIGS

  get_sig = inspect.signature(ld_mod.get_hf_config_and_mappings)
  accepts_path = "path" in get_sig.parameters
  ckpt_qcfg_fn = getattr(ld_mod, "_checkpoint_quantization_config", None)

  # 1a. Check `_checkpoint_quantization_config` with `{"text_config": null}`
  if ckpt_qcfg_fn is not None:
    with tempfile.TemporaryDirectory() as tmp_null:
      with open(os.path.join(tmp_null, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"model_type": "synthetic", "text_config": None}, f)
      try:
        ckpt_qcfg_fn(tmp_null)
      except Exception as exc:  # pylint: disable=broad-exception-caught
        issues.append(
            {
                "code": "NULL_SUBCONFIG_JSON_CRASH",
                "severity": "HIGH",
                "model": "<global>",
                "message": (
                    "_checkpoint_quantization_config crashed when config.json has"
                    f' `"text_config": null`: {type(exc).__name__}: {exc}'
                ),
            }
        )

  swept = 0
  base_yml_cache: dict[str, dict[str, Any]] = {}
  with tempfile.TemporaryDirectory() as tmp_bf16:
    with open(os.path.join(tmp_bf16, "config.json"), "w", encoding="utf-8") as f:
      json.dump({"model_type": "unquantized_bf16"}, f)

    for model_key in sorted(param_map_reg):
      if selected_models and model_key not in selected_models:
        continue
      if model_key not in hf_cfgs:
        continue
      swept += 1
      hf_obj = hf_cfgs[model_key]
      try:
        hf_dict_base = hf_obj.to_dict()
      except Exception as exc:  # pylint: disable=broad-exception-caught
        issues.append(
            {
                "code": "HF_MODEL_CONFIG_TO_DICT_CRASH",
                "severity": "HIGH",
                "model": model_key,
                "message": (
                    f"HF_MODEL_CONFIGS[{model_key!r}].to_dict() crashed"
                    f" (e.g. `quantization_config=None`): {type(exc).__name__}:"
                    f" {exc}"
                ),
            }
        )
        continue

      yml_defaults = _load_model_yaml_defaults(maxtext_root, model_key, base_yml_cache)
      for scan_layers in (False, True):
        mt_cfg = _PermissiveMaxTextConfig(model_key, scan_layers=scan_layers, yml_defaults=yml_defaults)
        _populate_mt_cfg_from_hf(mt_cfg, hf_dict_base, has_model_yml=bool(yml_defaults))

        # Test stock mapping + hooks
        try:
          pm_mod.PARAM_MAPPING[model_key](dict(hf_dict_base), mt_cfg, scan_layers=scan_layers)
          if model_key in hook_reg:
            hook_reg[model_key](
                dict(hf_dict_base),
                mt_cfg,
                scan_layers=scan_layers,
                saving_to_hf=False,
            )
        except NotImplementedError as exc:
          if scan_layers and "scan" in str(exc).lower():
            continue
          issues.append(
              {
                  "code": "STOCK_CONFIG_MAPPING_CRASH",
                  "severity": "HIGH",
                  "model": model_key,
                  "message": (
                      f"Stock PARAM_MAPPING/HOOK_FNS crashed for {model_key}"
                      f" (scan_layers={scan_layers}): {type(exc).__name__}: {exc}"
                  ),
              }
          )
          continue
        except Exception as exc:  # pylint: disable=broad-exception-caught
          issues.append(
              {
                  "code": "STOCK_CONFIG_MAPPING_CRASH",
                  "severity": "HIGH",
                  "model": model_key,
                  "message": (
                      f"Stock PARAM_MAPPING/HOOK_FNS crashed for {model_key}"
                      f" (scan_layers={scan_layers}): {type(exc).__name__}: {exc}"
                  ),
              }
          )
          continue

        # Test unquantized BF16 checkpoint config.json override via load_dynamic
        if accepts_path:
          try:
            ld_mod.get_hf_config_and_mappings(mt_cfg, path=tmp_bf16)
          except NotImplementedError as exc:
            if scan_layers and "scan" in str(exc).lower():
              continue
            issues.append(
                {
                    "code": "BF16_CONFIG_MAPPING_CRASH",
                    "severity": "HIGH",
                    "model": model_key,
                    "message": (
                        f"get_hf_config_and_mappings({model_key},"
                        f" scan_layers={scan_layers}) crashed on unquantized BF16"
                        f" config.json: {type(exc).__name__}: {exc}"
                    ),
                }
            )
          except Exception as exc:  # pylint: disable=broad-exception-caught
            issues.append(
                {
                    "code": "BF16_CONFIG_MAPPING_CRASH",
                    "severity": "HIGH",
                    "model": model_key,
                    "message": (
                        f"get_hf_config_and_mappings({model_key},"
                        f" scan_layers={scan_layers}) crashed on unquantized BF16"
                        f" config.json: {type(exc).__name__}: {exc}"
                    ),
                }
            )

  return issues, swept


def _check_streaming_fp8_and_sharding_contracts(
    mods: dict[str, Any],
) -> list[dict[str, Any]]:
  """Checks hf_streaming_load FP8 scale safety and padded multi-key split factors."""
  issues: list[dict[str, Any]] = []
  hsl = mods.get("hf_streaming_load")
  if hsl is None:
    return issues

  has_jax = importlib.util.find_spec("jax") is not None
  if has_jax:
    import jax  # pylint: disable=import-outside-toplevel
    import jax.numpy as jnp  # pylint: disable=import-outside-toplevel
    import numpy as np  # pylint: disable=import-outside-toplevel
  else:
    jax = jnp = np = None  # type: ignore[assignment]

  # 1. Unscaled FP8-to-BF16 silent cast gate (active when FP8 streaming or
  # `_check_float8_sources` is present on `hf_streaming_load`).
  check_fp8_fn = getattr(hsl, "_check_float8_sources", None)
  hsl_module_src = inspect.getsource(hsl)
  hsl_src = inspect.getsource(hsl.load_hf_params_streaming)
  has_fp8_streaming = check_fp8_fn is not None or re.search(r"float8|fp8", hsl_module_src, re.IGNORECASE) is not None
  if has_fp8_streaming:
    if check_fp8_fn is None or "_check_float8_sources(" not in hsl_src:
      issues.append(
          {
              "code": "UNSCALED_FP8_TO_BF16_SILENT_CAST",
              "severity": "HIGH",
              "model": "<hf_streaming_load>",
              "message": (
                  "hf_streaming_load.load_hf_params_streaming does not call"
                  " `_check_float8_sources`; a BF16-only PARAM_MAPPING pointed"
                  " at an FP8 safetensors checkpoint will silently cast raw"
                  " 1-byte FP8 weights to bfloat16 without applying block scales."
              ),
          }
      )
    elif has_jax:
      # Verify `_check_float8_sources` actually raises ValueError on a lone
      # float8_e4m3fn source targeting bfloat16.
      dev0 = jax.devices()[0]
      shd0 = jax.sharding.SingleDeviceSharding(dev0)
      target_tree = {"base": {"proj": jax.ShapeDtypeStruct(shape=(128, 128), dtype=jnp.bfloat16, sharding=shd0)}}
      cfg = types.SimpleNamespace(scan_layers=False, param_scan_axis=1)
      plan = hsl.build_plan(
          {"params-base-proj": "model.proj.weight"},
          {},
          target_tree,
          cfg,
      )
      hf_meta = {"model.proj.weight": jax.ShapeDtypeStruct(shape=(128, 128), dtype=jnp.float8_e4m3fn)}
      try:
        check_fp8_fn(plan, hf_meta)
        issues.append(
            {
                "code": "UNSCALED_FP8_TO_BF16_SILENT_CAST",
                "severity": "HIGH",
                "model": "<hf_streaming_load>",
                "message": (
                    "_check_float8_sources did not raise ValueError when a lone"
                    " float8_e4m3fn tensor mapped into a bfloat16 weight without"
                    " scale tensors."
                ),
            }
        )
      except ValueError:
        pass

  # 2. Composite QKV mismatched axis-0 tuple split check
  shared_split_fn = getattr(hsl, "shared_split_factor", None)
  if shared_split_fn is not None and has_jax:
    qkv_metas = [
        jax.ShapeDtypeStruct(shape=(1152, 1152), dtype=jnp.bfloat16),
        jax.ShapeDtypeStruct(shape=(576, 1152), dtype=jnp.bfloat16),
        jax.ShapeDtypeStruct(shape=(576, 1152), dtype=jnp.bfloat16),
    ]
    try:
      factor = shared_split_fn(qkv_metas, num_devices=8, min_bytes_to_split=1)
      if factor < 1:
        raise ValueError(f"Invalid split factor {factor}")
    except Exception as exc:  # pylint: disable=broad-exception-caught
      issues.append(
          {
              "code": "COMPOSITE_TUPLE_SPLIT_CRASH",
              "severity": "HIGH",
              "model": "<hf_streaming_load>",
              "message": (
                  "shared_split_factor crashed on composite QKV tensors with"
                  f" mismatched axis-0 shapes: {type(exc).__name__}: {exc}"
              ),
          }
      )

  # 3. Padded multi-key `(weight, scale)` split check on 64 and 12 devices
  load_req_fn = getattr(hsl, "_load_request", None)
  write_cls = getattr(hsl, "_Write", None)
  target_cls = getattr(hsl, "_Target", None)
  if load_req_fn is not None and write_cls is not None and target_cls is not None and has_jax:

    def _dummy_pad_hook(x: Any, *args: Any) -> Any:
      del args
      return x

    _dummy_pad_hook.pads_rows_to_target = True  # type: ignore[attr-defined]

    scale_dtype = getattr(jnp, "float8_e8m0fnu", jnp.uint8)
    hf_meta_padded = {
        "model.embed.weight": jax.ShapeDtypeStruct(shape=(324000, 1280), dtype=jnp.float8_e4m3fn),
        "model.embed.scale": jax.ShapeDtypeStruct(shape=(324000, 10), dtype=scale_dtype),
    }

    for n_devs in (64, 12):
      # Patch `_load_mesh` / `_replicated_sharding` / `_devices_of` if real
      # `jax.sharding.Mesh` rejects mock devices; test `shared_split_factor` +
      # `_padded_rows` directly using exact `_load_request` source inspection or
      # mock Mesh.
      padded_fn = getattr(hsl, "_padded_rows", None)
      if padded_fn is not None and shared_split_fn is not None:
        fake_devs = tuple(_FakeDevice(i) for i in range(n_devs))
        tgt = target_cls(
            name="base.embed",
            mt_key="params-base-embed",
            shape=(324096, 1280),
            dtype=np.dtype(jnp.bfloat16),
            sharding=_FakeSharding(fake_devs),
            axes=(),
            hook_shape=(324096, 1280),
            hooks=_dummy_pad_hook,
        )
        wr = write_cls(
            targets=(tgt,),
            index=(),
            hf_source=("model.embed.weight", "model.embed.scale"),
        )
        # Inspect `_load_request` AST/source order: in the buggy `1c400eedc`
        # commit, `shared_split_factor` was called on raw `hf_metadata[k]`
        # BEFORE `_padded_rows`, yielding factor=2 on 64 chips (instead of 64)
        # and factor=2 on 12 chips (where 324096 % 12 == 0 gives factor=12!).
        lr_src = inspect.getsource(load_req_fn)
        pos_pad = lr_src.find("_padded_rows(")
        pos_split = lr_src.find("shared_split_factor(")
        if pos_pad != -1 and pos_split != -1 and pos_split < pos_pad:
          issues.append(
              {
                  "code": "PADDED_TUPLE_SPLIT_DEGRADATION",
                  "severity": "HIGH",
                  "model": "<hf_streaming_load>",
                  "message": (
                      "_load_request calls `shared_split_factor` on unpadded"
                      " `hf_metadata` shapes BEFORE `_padded_rows`, collapsing"
                      " 64-chip padded `(weight, scale)` loads to <=8-way splits"
                      " or causing IndivisibleError on 12 devices."
                  ),
              }
          )
          break
        # Also verify padded shapes produce full `n_devs` split!
        pad_sig = inspect.signature(padded_fn)
        metas_after_pad = []
        for k in wr.hf_keys:
          m = hf_meta_padded[k]
          rows = padded_fn(wr, m, hf_meta_padded) if len(pad_sig.parameters) >= 3 else padded_fn(wr, m)
          metas_after_pad.append(jax.ShapeDtypeStruct(shape=(rows,) + tuple(m.shape)[1:], dtype=m.dtype))
        factor = shared_split_fn(metas_after_pad, num_devices=n_devs, min_bytes_to_split=1)
        if factor != n_devs:
          issues.append(
              {
                  "code": "PADDED_TUPLE_SPLIT_DEGRADATION",
                  "severity": "HIGH",
                  "model": "<hf_streaming_load>",
                  "message": (
                      "Padded `(324096, 1280)` + `(324096, 10)` tuple split across"
                      f" {factor} shards instead of {n_devs} devices."
                  ),
              }
          )

  return issues


def main(argv: Sequence[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description="Sweep MaxText PARAM_MAPPING and HF streaming contracts.")
  parser.add_argument(
      "--maxtext-root",
      default=_default_repo_root(),
      help="Path to the MaxText repository root (containing src/maxtext/).",
  )
  parser.add_argument(
      "--models",
      default="",
      help="Optional comma-separated subset of model keys to sweep.",
  )
  args = parser.parse_args(argv)

  try:
    mods = _import_maxtext_modules(args.maxtext_root)
  except Exception as exc:  # pylint: disable=broad-exception-caught
    sys.stderr.write(f"Import error loading MaxText from {args.maxtext_root}: {exc}\n")
    return 2

  selected = {m.strip() for m in args.models.split(",") if m.strip()} if args.models else None
  issues, models_swept = _sweep_param_mappings(mods, selected, maxtext_root=args.maxtext_root)
  issues.extend(_check_streaming_fp8_and_sharding_contracts(mods))

  report = {
      "status": "FAIL" if issues else "PASS",
      "models_swept": models_swept,
      "issue_count": len(issues),
      "issues": issues,
  }
  sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
  return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
  sys.exit(main())
