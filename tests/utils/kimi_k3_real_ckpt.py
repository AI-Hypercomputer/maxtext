"""Real-checkpoint forward-logit check for Kimi-K3: HF reference (PyTorch) vs MaxText.

Runs the released `moonshotai/Kimi-K3` weights, truncated to the first `--num_layers`
decoder layers (4 .. 93; 93 is the full model), through both the HF reference modeling code
and MaxText, and compares the next-token logits on a fixed 16-token prompt. Runs on a
large-RAM CPU host (not a unit test; the full model needs ~3 TB). Each stage is a separate
process so peak memory stays bounded; the stages share a work directory.

  python tests/utils/kimi_k3_real_ckpt.py download       [--num_layers N] [--work DIR]
  python tests/utils/kimi_k3_real_ckpt.py torch_oracle   [--num_layers N] [--mxfp4_experts]
  python tests/utils/kimi_k3_real_ckpt.py convert        [--num_layers N] [--mxfp4_experts] [--mt_dir DIR|gs://...]
  python tests/utils/kimi_k3_real_ckpt.py maxtext_logits [--num_layers N] [--mxfp4_experts] [--mt_dir ...]
  python tests/utils/kimi_k3_real_ckpt.py compare        [--num_layers N]

Layout of --work (default ~/kimi_k3_{N}layer):
  hf/                 config.json, model.safetensors.index.json, tokenizer files and only the
                      shards holding layers 0..N-1 + embed/norm/lm_head/output_attn_res
                      (the original 96-shard index is kept; loaders only open needed shards)
  tokens_prompt.npy   fixed input ids [1, SEQ]
  logits_pt.npy / logits_mt.npy
The MaxText (Orbax v1) checkpoint goes to --mt_dir (local path or gs://).

`--mxfp4_experts` keeps the routed experts in their released MXFP4 form on both sides (the
oracle and MaxText decode only the gathered top-k experts); it is required for the full
model and is bit-identical to the dense path. The HF reference's KDA kernels (fla, Triton)
do not run on CPU, so the oracle patches them with `tests/utils/kimi_k3_kda_reference.py`.

Why 4 layers is the smallest useful size: layer 0 is the dense-MLP KDA layer, 1-2 are
latent-MoE KDA layers and 3 is the first (NoPE, gated) MLA layer, so every block type is
exercised. The AttnRes projections are `Linear(hidden, 1)` at both layer and model level, so
truncation needs no weight slicing.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import resource
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # pylint: disable=wrong-import-position

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (_REPO, os.path.join(_REPO, "src")):
  if _p not in sys.path:
    sys.path.insert(0, _p)

REPO_ID = "moonshotai/Kimi-K3"
NUM_LAYERS = 4
FULL_ATTN_LAYERS_0IDX = [3]  # MaxText / mapping convention
SEQ = 16
BATCH = 1
LM_PREFIX = "language_model."
_LAYER_RE = re.compile(r"^language_model\.model\.layers\.(\d+)\.")


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
def _log(msg: str) -> None:
  print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _peak_rss_gb() -> float:
  return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # ru_maxrss is KB on Linux


def get_full_attn_layers_0idx(num_layers: int) -> list[int]:
  """Computes 0-indexed full-attention (MLA) layers for any truncation."""
  real_full_attn = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 92]
  return [i for i in real_full_attn if i < num_layers]


def _wanted_lm_key(key: str, num_layers: int = 4) -> bool:
  """True for LM keys the model needs (HF key names, with `language_model.` prefix)."""
  if not key.startswith(LM_PREFIX):
    return False  # vision_tower / mm_projector
  m = _LAYER_RE.match(key)
  if m is None:
    return True  # embed_tokens, norm, lm_head, output_attn_res_*
  return int(m.group(1)) < num_layers


def _load_index(hf_dir: str) -> dict[str, str]:
  with open(os.path.join(hf_dir, "model.safetensors.index.json"), encoding="utf-8") as f:
    return json.load(f)["weight_map"]


PROMPT = "The capital of France is Paris. The capital of Germany is Berlin. " "The capital of Italy is"


def _tokens(work: str) -> np.ndarray:
  """Real tokenized prompt, exactly SEQ tokens.

  Deliberately a strongly-patterned prompt: the repeated schema drives peaked logits,
  so the top-1/top-2 margin sits well above bfloat16 quantization noise.

  An earlier revision used uniformly random token IDs. On gibberish input the model has
  no confident continuation, and the observed top-1 margins collapsed to ~0.0625 at logit
  magnitude ~10 -- about 2 bf16 ULPs. Since the reference implementation is itself bf16,
  its own argmax ordering at those positions was below its own quantization noise, making
  the resulting top-1 disagreements unresolvable in principle rather than indicative of a
  real numerical discrepancy.
  """
  path = os.path.join(work, "tokens_prompt.npy")
  if os.path.exists(path):
    return np.load(path)

  hf_dir = os.path.join(work, "hf")
  from transformers import AutoTokenizer  # pylint: disable=import-outside-toplevel

  try:
    tok = AutoTokenizer.from_pretrained(hf_dir, trust_remote_code=True)
  except Exception as e:  # pylint: disable=broad-except
    # Kimi ships a tiktoken-based tokenizer via remote code. If the local dir is missing
    # one of those pieces, let the hub resolve the whole set.
    _log(f"local tokenizer load failed ({e}); falling back to hub {REPO_ID}")
    tok = AutoTokenizer.from_pretrained(REPO_ID, trust_remote_code=True)
  ids = tok(PROMPT, add_special_tokens=False)["input_ids"]

  if len(ids) < SEQ:
    raise ValueError(
        f"prompt tokenizes to {len(ids)} tokens but SEQ={SEQ}; padding is refused here "
        "because pad tokens would reintroduce the low-confidence regime this prompt exists "
        "to avoid. Use a longer prompt."
    )
  # Keep the tail so the sequence ends on a natural continuation point rather than being
  # truncated mid-pattern.
  toks = np.asarray(ids[-SEQ:], dtype=np.int32).reshape(BATCH, SEQ)
  np.save(path, toks)
  _log(f"prompt={PROMPT!r} -> {len(ids)} tokens, kept last {SEQ}: {toks.tolist()}")
  return toks


def _hf_linear_attn_config(num_layers: int, full_attn_layers: list[int]) -> dict:
  """1-indexed HF `linear_attn_config` for the truncation."""
  from maxtext.checkpoint_conversion.utils.hf_model_configs import kimi_k3_dict  # pylint: disable=import-outside-toplevel

  lac = dict(kimi_k3_dict["linear_attn_config"])
  lac["kda_layers"] = [i + 1 for i in range(num_layers) if i not in full_attn_layers]
  lac["full_attn_layers"] = [i + 1 for i in full_attn_layers]
  return lac


def _maxtext_overrides(
    num_layers: int, full_attn_layers: list[int], dtype: str = "float32", mxfp4_experts: bool = False
) -> dict:
  """Overrides on top of the production `kimi-k3.yml` (dims come from the yml)."""
  return {
      "override_model_config": True,
      "base_num_decoder_layers": num_layers,
      "full_attn_layers": list(full_attn_layers),
      # Routed experts: dense (dequantized at conversion) or packed MXFP4 (decoded in forward).
      "routed_experts_weight_format": "mxfp4" if mxfp4_experts else "bf16",
      # Sequence. Keep original == max so MLA's YaRN mscale rescaling stays off (MLA is NoPE anyway).
      "max_target_length": SEQ,
      "max_position_embeddings": SEQ,
      "original_max_position_embeddings": SEQ,
      "max_prefill_predict_length": SEQ,
      "per_device_batch_size": BATCH,
      "global_batch_size_to_train_on": BATCH,
      # Numerics: float32 or bfloat16.
      "dtype": dtype,
      "weight_dtype": dtype,
      "matmul_precision": "highest",
      "attention": "dot_product",
      "mla_naive_kvcache": True,
      # Unscanned only.
      "scan_layers": False,
      "enable_checkpointing": False,
      "enable_dropout": False,
      "hardware": "cpu",
      "skip_jax_distributed_system": True,
  }


def _maxtext_config(num_layers: int, full_attn_layers: list[int], dtype: str = "float32", mxfp4_experts: bool = False):
  from maxtext.configs import pyconfig  # pylint: disable=import-outside-toplevel
  from tests.utils.test_helpers import get_test_config_path  # pylint: disable=import-outside-toplevel

  return pyconfig.initialize(
      [sys.argv[0], get_test_config_path()],
      model_name="kimi-k3",
      **_maxtext_overrides(num_layers, full_attn_layers, dtype=dtype, mxfp4_experts=mxfp4_experts),
  )


# -----------------------------------------------------------------------------
# stage: download
# -----------------------------------------------------------------------------
def stage_download(work: str, num_layers: int = 4, link_from: Optional[str] = None) -> None:
  """Downloads or links required shards from Hugging Face for the test run."""
  from huggingface_hub import hf_hub_download  # pylint: disable=import-outside-toplevel

  hf_dir = os.path.join(work, "hf")
  os.makedirs(hf_dir, exist_ok=True)

  # Optionally hardlink already-downloaded files (e.g. from a smaller --num_layers run).
  ref_hf_dir = os.path.expanduser(link_from) if link_from else None
  if ref_hf_dir and os.path.isdir(ref_hf_dir) and os.path.abspath(ref_hf_dir) != os.path.abspath(hf_dir):
    for f in os.listdir(ref_hf_dir):
      src_path = os.path.join(ref_hf_dir, f)
      dst_path = os.path.join(hf_dir, f)
      if os.path.isfile(src_path) and not os.path.exists(dst_path):
        try:
          os.link(src_path, dst_path)
          _log(f"hardlinked {f} from {ref_hf_dir}")
        except OSError:
          pass

  for fname in ("config.json", "model.safetensors.index.json"):
    if not os.path.exists(os.path.join(hf_dir, fname)):
      hf_hub_download(REPO_ID, fname, local_dir=hf_dir)

  # Tokenizer files, needed by `_tokens` to build a real prompt. Only a few MB. Kimi-K3
  # ships a tiktoken-based custom tokenizer (no HF `tokenizer.json`), so we need the
  # vocab model plus the remote-code modules that implement it. Treated as optional so a
  # repo layout change cannot break the (very expensive) weight download.
  for fname in ("tokenizer_config.json", "tiktoken.model", "tokenization_kimi.py", "encoding_k3.py"):
    if not os.path.exists(os.path.join(hf_dir, fname)):
      try:
        hf_hub_download(REPO_ID, fname, local_dir=hf_dir)
        _log(f"downloaded {fname}")
      except Exception as e:  # pylint: disable=broad-except
        _log(f"WARNING: could not download {fname}: {e}")
  weight_map = _load_index(hf_dir)
  wanted = [k for k in weight_map if _wanted_lm_key(k, num_layers=num_layers)]
  shards = sorted({weight_map[k] for k in wanted})
  missing_shards = [s for s in shards if not os.path.exists(os.path.join(hf_dir, s))]
  _log(f"{len(wanted)} keys across {len(shards)} shards ({len(missing_shards)} missing): {shards}")

  def _dl(fname):
    t0 = time.time()
    path = hf_hub_download(REPO_ID, fname, local_dir=hf_dir)
    _log(f"downloaded {fname} ({os.path.getsize(path) / 1e9:.1f} GB) in {time.time() - t0:.0f}s")

  if missing_shards:
    max_workers = min(len(missing_shards), 8)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
      list(ex.map(_dl, missing_shards))
  _log("download done")


# -----------------------------------------------------------------------------
# stage: torch_oracle
# -----------------------------------------------------------------------------
def stage_torch_oracle(
    work: str,
    num_layers: int = 4,
    dtype: str = "float32",
    mxfp4_experts: bool = False,
    delete_shards_after_load: bool = False,
) -> None:
  """Runs the PyTorch oracle model on real checkpoint weights and saves logits."""
  import torch  # pylint: disable=import-outside-toplevel

  from tests.utils.kimi_k3_parity_utils import load_hf_reference  # pylint: disable=import-outside-toplevel

  hf_dir = os.path.join(work, "hf")
  torch.set_grad_enabled(False)
  target_pt_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32

  # --- config: real text_config, truncated to num_layers, eager attention ----------------------
  hf_config_mod, hf_model_mod = load_hf_reference(patch_kda=True)  # HF's fallback KDA is not a valid oracle

  # Patch FusedRMSNormGated to avoid dtype promotion when running in bfloat16 on CPU
  if hasattr(hf_model_mod, "FusedRMSNormGated"):

    def _patched_fused_forward(self, x, gate):
      x_f = x.float()
      variance = x_f.pow(2).mean(-1, keepdim=True)
      out_dtype = self.weight.dtype if getattr(self, "weight", None) is not None else x.dtype
      normed = (x_f * torch.rsqrt(variance + self.variance_epsilon) * self.weight.float()).to(out_dtype)
      if self.activation == "sigmoid":
        normed = (normed.float() * torch.sigmoid(gate.float())).to(out_dtype)
      return normed

    hf_model_mod.FusedRMSNormGated.forward = _patched_fused_forward

  with open(os.path.join(hf_dir, "config.json"), encoding="utf-8") as f:
    text_cfg = json.load(f)["text_config"]
  for k in ("quantization_config", "dtype", "torch_dtype", "auto_map", "_name_or_path", "transformers_version"):
    text_cfg.pop(k, None)
  full_attn_layers = get_full_attn_layers_0idx(num_layers)
  text_cfg.update(
      num_hidden_layers=num_layers,
      linear_attn_config=_hf_linear_attn_config(num_layers, full_attn_layers),
      use_cache=False,
      _attn_implementation="eager",
  )
  cfg = hf_config_mod.KimiLinearConfig(**text_cfg)
  _log(f"HF config: layers={cfg.num_hidden_layers} linear_attn_config={cfg.linear_attn_config} dtype={target_pt_dtype}")

  model = load_torch_oracle(
      hf_dir,
      cfg,
      hf_model_mod,
      num_layers=num_layers,
      target_pt_dtype=target_pt_dtype,
      keep_mxfp4=mxfp4_experts,
      delete_shards_after_load=delete_shards_after_load,
  )
  n_params = sum(p.numel() for p in model.parameters())
  n_packed = sum(b.numel() for n, b in model.named_buffers() if n.endswith(".weight_packed"))
  _log(
      f"torch model ready: {n_params / 1e9:.2f}B dense params + {2 * n_packed / 1e9:.2f}B MXFP4 expert params, "
      f"peak RSS {_peak_rss_gb():.0f} GB"
  )

  # --- forward ------------------------------------------------------------------------------
  toks = _tokens(work)
  t0 = time.time()
  logits = model(input_ids=torch.from_numpy(toks).long()).logits.float().numpy()
  _log(f"torch forward in {time.time() - t0:.0f}s; logits {logits.shape} absmax {np.abs(logits).max():.4f}")
  if not np.isfinite(logits).all():
    raise RuntimeError("non-finite torch logits")
  np.save(os.path.join(work, "logits_pt.npy"), logits)
  _log(f"torch top-1: {logits[0].argmax(-1).tolist()}")
  _log(f"torch_oracle done (peak RSS {_peak_rss_gb():.0f} GB)")


_TMPFS_ROOT = "/dev/shm/"


def _delete_tmpfs_shard(path: str) -> None:
  """Deletes a loaded shard to free RAM, but only if it really lives on /dev/shm.

  Refuses anything else (e.g. a symlink into a persistent-disk copy), so the flag can never
  destroy the only on-disk copy of the checkpoint.
  """
  real = os.path.realpath(path)
  if not real.startswith(_TMPFS_ROOT):
    raise RuntimeError(f"refusing to delete {path} -> {real}: not under {_TMPFS_ROOT}")
  os.remove(real)
  _log(f"deleted {real} (tmpfs)")


def load_torch_oracle(
    hf_dir: str,
    cfg,
    hf_model_mod,
    num_layers: int,
    target_pt_dtype,
    keep_mxfp4: bool = False,
    delete_shards_after_load: bool = False,
):
  """Builds the HF reference on meta and assigns weights straight from the safetensors shards.

  keep_mxfp4=False: routed experts are dequantized at load time into dense `nn.Linear`s.
  keep_mxfp4=True:  routed experts are swapped for `Mxfp4Linear` and hold the released
    `weight_packed` / `weight_scale` bytes verbatim, decoded on every forward call. The
    forward is bit-identical to the dense mode, at ~1/4 of the bf16 expert memory.
  delete_shards_after_load: delete each shard right after it is loaded (tmpfs shards only),
    so a /dev/shm copy of the checkpoint and the growing model never coexist in full.
  """
  if delete_shards_after_load and not keep_mxfp4:
    # The dense path may read a weight_scale from a *later* shard, which must still exist.
    raise ValueError("delete_shards_after_load requires keep_mxfp4=True")
  import torch  # pylint: disable=import-outside-toplevel
  from safetensors import safe_open  # pylint: disable=import-outside-toplevel

  from maxtext.checkpoint_conversion.utils import mxfp4  # pylint: disable=import-outside-toplevel
  from tests.utils import kimi_k3_torch_mxfp4  # pylint: disable=import-outside-toplevel

  with torch.device("meta"):
    model = hf_model_mod.KimiLinearForCausalLM(cfg)
    if keep_mxfp4:
      kimi_k3_torch_mxfp4.patch_experts_mxfp4(model)
  model.eval()
  expected = set(model.state_dict().keys())

  weight_map = _load_index(hf_dir)
  wanted = [k for k in weight_map if _wanted_lm_key(k, num_layers=num_layers)]
  by_shard: dict[str, list[str]] = {}
  for k in wanted:
    by_shard.setdefault(weight_map[k], []).append(k)

  loaded: set[str] = set()
  for shard, keys in sorted(by_shard.items()):
    t0 = time.time()
    partial: dict[str, torch.Tensor] = {}
    with safe_open(os.path.join(hf_dir, shard), framework="pt") as f:
      for key in keys:
        if keep_mxfp4 and mxfp4.is_mxfp4_sidecar_key(key):
          # Raw bytes, same key as the Mxfp4Linear buffer.
          partial[key[len(LM_PREFIX) :]] = f.get_tensor(key).contiguous().view(torch.uint8).clone()
          continue
        if mxfp4.is_mxfp4_sidecar_key(key) and key.endswith(".weight_scale"):
          continue
        if key.endswith(".weight_packed"):
          scale_key = key[: -len("weight_packed")] + "weight_scale"
          packed = f.get_tensor(key).contiguous().view(torch.uint8).numpy()
          scale_shard = weight_map[scale_key]
          if scale_shard == shard:
            scale_t = f.get_tensor(scale_key)
          else:
            with safe_open(os.path.join(hf_dir, scale_shard), framework="pt") as g:
              scale_t = g.get_tensor(scale_key)
          scale = scale_t.contiguous().view(torch.uint8).numpy()
          dense = mxfp4.dequantize_mxfp4_packed(packed, scale, dtype=np.float32)
          partial[key[len(LM_PREFIX) : -len("_packed")]] = torch.from_numpy(dense).to(target_pt_dtype)
        else:
          t = f.get_tensor(key).to(target_pt_dtype)
          if key.endswith(".self_attn.A_log"):
            n = cfg.linear_attn_config["num_heads"]
            if t.shape[0] != n:
              if t.shape[0] < n or bool((t[n:] != 0).any()):
                raise RuntimeError(f"{key}: shape {tuple(t.shape)} is not zero-padded num_heads={n}")
              t = t[:n].clone()
          partial[key[len(LM_PREFIX) :]] = t
    unexpected = set(partial) - expected
    if unexpected:
      raise RuntimeError(f"{shard}: keys not in model: {sorted(unexpected)[:10]}")
    model.load_state_dict(partial, strict=False, assign=True)
    loaded |= set(partial)
    _log(f"loaded {len(partial)} tensors from {shard} in {time.time() - t0:.0f}s (peak RSS {_peak_rss_gb():.0f} GB)")
    del partial
    if delete_shards_after_load:
      _delete_tmpfs_shard(os.path.join(hf_dir, shard))

  missing = expected - loaded
  if missing:
    raise RuntimeError(f"model params not covered by checkpoint: {sorted(missing)[:20]}")
  still_meta = [n for n, p in list(model.named_parameters()) + list(model.named_buffers()) if p.is_meta]
  if still_meta:
    raise RuntimeError(f"params/buffers still on meta: {still_meta[:20]}")
  return model


# -----------------------------------------------------------------------------
# stage: convert
# -----------------------------------------------------------------------------
def stage_convert(
    work: str,
    mt_dir: str,
    num_layers: int = 4,
    save_dtype: str = "float32",
    mxfp4_experts: bool = False,
    save_budget_gb: int | None = None,
) -> None:
  """Runs `to_maxtext` (lazy, single device) into `mt_dir` (a local path or gs:// URI).

  Peak host memory during the save tracks `save_budget_gb` (MaxText's
  `checkpoint_storage_concurrent_gb`), not the checkpoint size.
  """
  from unittest import mock  # pylint: disable=import-outside-toplevel

  from maxtext.checkpoint_conversion import to_maxtext  # pylint: disable=import-outside-toplevel
  from maxtext.checkpoint_conversion.utils import hf_model_configs  # pylint: disable=import-outside-toplevel
  from maxtext.checkpoint_conversion.utils.hf_model_configs import HF_MODEL_CONFIGS  # pylint: disable=import-outside-toplevel
  from tests.utils.test_helpers import get_test_config_path  # pylint: disable=import-outside-toplevel

  hf_dir = os.path.join(work, "hf")
  out_dir = mt_dir
  full_attn_layers = get_full_attn_layers_0idx(num_layers)

  hf_kwargs = dict(hf_model_configs.kimi_k3_dict)
  hf_kwargs["num_hidden_layers"] = num_layers
  hf_kwargs["linear_attn_config"] = _hf_linear_attn_config(num_layers, full_attn_layers)
  hf_cfg = hf_model_configs.KimiK3Config(**hf_kwargs)

  args = [sys.argv[0], get_test_config_path(), "model_name=kimi-k3", f"base_output_directory={out_dir}"]
  args += [f"run_name=kimi_k3_{num_layers}layer_real"]
  if save_budget_gb is not None:
    args += [f"checkpoint_storage_concurrent_gb={save_budget_gb}"]
  args += [
      f"{k}={v}"
      for k, v in _maxtext_overrides(num_layers, full_attn_layers, dtype=save_dtype, mxfp4_experts=mxfp4_experts).items()
  ]
  t0 = time.time()
  sampler = _MemSampler(interval_s=float(os.environ.get("KIMI_MEM_SAMPLE_S", "30")))
  orig_save = to_maxtext.save_weights_to_checkpoint

  def _tagged_save(*a, **kw):
    sampler.set_phase("save")  # lazy leaves are loaded and written in this phase
    try:
      return orig_save(*a, **kw)
    finally:
      sampler.set_phase("done")

  sampler.start("transform")
  try:
    with (
        mock.patch.dict(HF_MODEL_CONFIGS, {"kimi-k3": hf_cfg}),
        mock.patch.object(to_maxtext, "save_weights_to_checkpoint", _tagged_save),
    ):
      to_maxtext.main(
          args,
          lazy_load_tensors=True,
          hf_model_path=hf_dir,
          save_dtype=save_dtype,
          simulated_cpu_devices_count=1,
      )
  finally:
    sampler.stop()
  items = os.path.join(out_dir, "0", "items")
  if out_dir.startswith("gs://"):
    from etils import epath  # pylint: disable=import-outside-toplevel

    if not epath.Path(items).exists():
      raise RuntimeError(f"no checkpoint at {items}")
    ckpt_gb = float("nan")  # not walked on GCS; use `gcloud storage du` if needed
  else:
    if not os.path.isdir(items):
      raise RuntimeError(f"no checkpoint at {items}")
    ckpt_gb = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(items) for f in fs) / 1e9
  _log(f"convert done -> {items} in {(time.time() - t0) / 60:.1f} min (peak RSS {_peak_rss_gb():.0f} GB)")
  _log(f"checkpoint size on disk: {ckpt_gb:.1f} GB")
  sampler.report(ckpt_gb)


class _MemSampler:
  """Background sampler of this process's memory, split by kind, tagged with a phase.

  `RssAnon` is what actually has to fit in RAM; `RssFile` is mostly mmap'd safetensors
  pages, which the kernel can reclaim; `RssShmem` is tmpfs-backed. Used to extrapolate
  the convert's memory needs from 13 layers to the full 93.
  """

  _KEYS = ("RssAnon", "RssFile", "RssShmem")

  def __init__(self, interval_s: float = 30.0):
    import threading  # pylint: disable=import-outside-toplevel

    self._interval = interval_s
    self._stop = threading.Event()
    self._thread = threading.Thread(target=self._run, daemon=True)
    self._phase = "init"
    self._peaks: dict[str, dict[str, float]] = {}
    self._min_avail: dict[str, float] = {}

  @staticmethod
  def _read() -> dict[str, float]:
    """Reads memory metrics from /proc/self/status and /proc/meminfo."""
    out = {}
    with open("/proc/self/status", encoding="utf-8") as f:
      for line in f:
        k, _, v = line.partition(":")
        if k in _MemSampler._KEYS:
          out[k] = int(v.split()[0]) / 1e6  # kB -> GB
    with open("/proc/meminfo", encoding="utf-8") as f:
      for line in f:
        if line.startswith("MemAvailable:"):
          out["MemAvailable"] = int(line.split()[1]) / 1e6
    return out

  def _sample(self) -> None:
    s = self._read()
    p = self._peaks.setdefault(self._phase, {k: 0.0 for k in self._KEYS})
    for k in self._KEYS:
      p[k] = max(p[k], s.get(k, 0.0))
    self._min_avail[self._phase] = min(self._min_avail.get(self._phase, float("inf")), s.get("MemAvailable", 0.0))
    _log(
        f"[mem] phase={self._phase} anon={s.get('RssAnon', 0):.1f} file={s.get('RssFile', 0):.1f} "
        f"shmem={s.get('RssShmem', 0):.1f} sys_avail={s.get('MemAvailable', 0):.0f} GB"
    )

  def _run(self) -> None:
    while not self._stop.wait(self._interval):
      self._sample()

  def start(self, phase: str) -> None:
    self._phase = phase
    self._sample()
    self._thread.start()

  def set_phase(self, phase: str) -> None:
    self._sample()  # close out the previous phase with a final reading
    self._phase = phase
    self._sample()

  def stop(self) -> None:
    self._stop.set()
    self._thread.join(timeout=5)
    self._sample()

  def report(self, ckpt_gb: float) -> None:
    _log(f"[mem] per-phase peaks (GB); checkpoint {ckpt_gb:.1f} GB")
    for phase, p in self._peaks.items():
      ratio = p["RssAnon"] / ckpt_gb if ckpt_gb else float("nan")
      _log(
          f"[mem]   {phase:12s} anon={p['RssAnon']:8.1f} ({ratio:4.2f}x ckpt) file={p['RssFile']:8.1f} "
          f"shmem={p['RssShmem']:8.1f} min_sys_avail={self._min_avail.get(phase, float('nan')):8.0f}"
      )


# -----------------------------------------------------------------------------
# stage: maxtext_logits
# -----------------------------------------------------------------------------
def stage_maxtext_logits(
    work: str, mt_dir: str, num_layers: int = 4, dtype: str = "float32", mxfp4_experts: bool = False
) -> None:
  """Runs the MaxText model on checkpoint weights and saves logits."""
  import jax  # pylint: disable=import-outside-toplevel
  import jax.numpy as jnp  # pylint: disable=import-outside-toplevel
  from flax import nnx  # pylint: disable=import-outside-toplevel

  from maxtext.common import checkpointing  # pylint: disable=import-outside-toplevel
  from maxtext.common.common_types import MODEL_MODE_TRAIN  # pylint: disable=import-outside-toplevel
  from maxtext.utils import model_creation_utils  # pylint: disable=import-outside-toplevel
  from tests.utils.kimi_k3_parity_utils import positions_and_segments  # pylint: disable=import-outside-toplevel

  items = os.path.join(mt_dir, "0", "items")
  full_attn_layers = get_full_attn_layers_0idx(num_layers)
  cfg = _maxtext_config(num_layers, full_attn_layers, dtype=dtype, mxfp4_experts=mxfp4_experts)

  t0 = time.time()
  _, abstract_model = model_creation_utils.create_nnx_abstract_model(cfg, model_mode=MODEL_MODE_TRAIN)
  graphdef, params_abs, rest_abs = nnx.split(abstract_model, nnx.Param, ...)
  n_params = sum(int(np.prod(a.shape)) for a in jax.tree.leaves(params_abs))
  _log(f"abstract model built in {time.time() - t0:.0f}s: {n_params / 1e9:.2f}B params")

  t0 = time.time()
  params = checkpointing.load_params_from_path(
      items,
      params_abs,
      cfg.checkpoint_storage_concurrent_gb,
      cfg.checkpoint_storage_use_ocdbt,
      cfg.checkpoint_storage_use_zarr3,
  )
  _log(f"restored params in {(time.time() - t0) / 60:.1f} min (peak RSS {_peak_rss_gb():.0f} GB)")

  def _materialize(a):
    if isinstance(a, jax.ShapeDtypeStruct):
      if jnp.issubdtype(a.dtype, jax.dtypes.prng_key):
        return jnp.broadcast_to(jax.random.key(0), a.shape)
      return jnp.zeros(a.shape, a.dtype)
    return a

  rest = jax.tree.map(_materialize, rest_abs)
  model = nnx.merge(graphdef, params, rest)

  toks = _tokens(work)
  positions, _ = positions_and_segments(BATCH, SEQ)
  t0 = time.time()
  logits = np.asarray(
      model(jnp.asarray(toks), positions, model_mode=MODEL_MODE_TRAIN, enable_dropout=False),
      dtype=np.float32,
  )
  _log(f"maxtext forward in {time.time() - t0:.0f}s; logits {logits.shape} absmax {np.abs(logits).max():.4f}")
  np.save(os.path.join(work, "logits_mt.npy"), logits)
  if not np.isfinite(logits).all():
    raise RuntimeError("non-finite maxtext logits")

  pt_path = os.path.join(work, "logits_pt.npy")
  if not os.path.exists(pt_path):
    _log("no logits_pt.npy yet - run torch_oracle to compare")
    return
  logits_pt = np.load(pt_path)
  diff = np.abs(logits - logits_pt)
  top1_mt, top1_pt = logits[0].argmax(-1), logits_pt[0].argmax(-1)
  agree = int((top1_mt == top1_pt).sum())
  _log(f"max|diff| {diff.max():.3e}  mean|diff| {diff.mean():.3e}  torch absmax {np.abs(logits_pt).max():.4f}")
  _log(f"per-position max|diff|: {[f'{d:.2e}' for d in diff[0].max(-1)]}")
  _log(f"top-1 agreement {agree}/{SEQ}   mt={top1_mt.tolist()}   pt={top1_pt.tolist()}")
  _log(f"maxtext_logits done (peak RSS {_peak_rss_gb():.0f} GB)")
  stage_compare(work, dtype=dtype)


# -----------------------------------------------------------------------------
# stage: compare
# -----------------------------------------------------------------------------
def stage_compare(work: str, dtype: str = "bfloat16") -> None:
  """Applies the parity criteria to already-saved logits. Cheap and re-runnable."""
  from tests.utils.kimi_k3_parity_utils import assert_logit_parity  # pylint: disable=import-outside-toplevel

  logits_pt = np.load(os.path.join(work, "logits_pt.npy"))
  logits_mt = np.load(os.path.join(work, "logits_mt.npy"))
  assert_logit_parity(logits_mt, logits_pt, dtype=dtype)
  _log("PARITY PASSED")


# -----------------------------------------------------------------------------
def main() -> None:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("stage", choices=["download", "tokens", "torch_oracle", "convert", "maxtext_logits", "compare"])
  p.add_argument("--num_layers", type=int, default=4, help="Number of decoder layers (default: 4)")
  p.add_argument(
      "--dtype",
      default=None,
      choices=["float32", "bfloat16"],
      help="Dtype for execution and checkpoint (default: float32 if num_layers<=4 else bfloat16)",
  )
  p.add_argument("--work", default=None, help="Working directory (default: ~/kimi_k3_{num_layers}layer)")
  p.add_argument(
      "--mt_dir",
      default=None,
      help="Where `to_maxtext` writes the Orbax checkpoint. Default: /dev/shm/kimi_k3_{num_layers}layer_mt",
  )
  p.add_argument(
      "--mxfp4_experts",
      action="store_true",
      help="Keep routed experts packed MXFP4 (decoded in forward) in both the torch oracle and MaxText. "
      "Required for the full 93-layer model; bit-identical to the dense path.",
  )
  p.add_argument(
      "--save_budget_gb",
      type=int,
      default=None,
      help="convert: host-memory budget for the lazy checkpoint save, in GB "
      "(MaxText checkpoint_storage_concurrent_gb; default 96).",
  )
  p.add_argument(
      "--link_from",
      default=None,
      help="download: hardlink files that already exist in this HF directory instead of re-downloading.",
  )
  p.add_argument(
      "--delete_shards_after_load",
      action="store_true",
      help="torch_oracle: delete each HF shard right after it is loaded. Only allowed when the shards "
      "resolve to /dev/shm and --mxfp4_experts is set; frees tmpfs memory while the oracle grows.",
  )
  a = p.parse_args()

  dtype = a.dtype if a.dtype is not None else ("float32" if a.num_layers <= 4 else "bfloat16")
  work = a.work if a.work is not None else os.path.expanduser(f"~/kimi_k3_{a.num_layers}layer")
  mt_dir = a.mt_dir if a.mt_dir is not None else f"/dev/shm/kimi_k3_{a.num_layers}layer_mt"
  mx = a.mxfp4_experts

  os.makedirs(work, exist_ok=True)
  _log(f"stage={a.stage} num_layers={a.num_layers} dtype={dtype} mxfp4_experts={mx} work={work} mt_dir={mt_dir}")
  if a.stage == "download":
    stage_download(work, num_layers=a.num_layers, link_from=a.link_from)
  elif a.stage == "tokens":
    _log(f"tokens: {_tokens(work).tolist()}")
  elif a.stage == "torch_oracle":
    stage_torch_oracle(
        work,
        num_layers=a.num_layers,
        dtype=dtype,
        mxfp4_experts=mx,
        delete_shards_after_load=a.delete_shards_after_load,
    )
  elif a.stage == "convert":
    stage_convert(
        work, mt_dir, num_layers=a.num_layers, save_dtype=dtype, mxfp4_experts=mx, save_budget_gb=a.save_budget_gb
    )
  elif a.stage == "compare":
    stage_compare(work, dtype=dtype)
  else:
    stage_maxtext_logits(work, mt_dir, num_layers=a.num_layers, dtype=dtype, mxfp4_experts=mx)


if __name__ == "__main__":
  main()
