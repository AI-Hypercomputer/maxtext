# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""End-to-end model driver and golden generator CLI for Zero-Gen verification."""

from collections.abc import Callable, Sequence
import sys
from typing import Any

from etils import epath
from flax import nnx
import jax.numpy as jnp
from maxtext.configs import pyconfig
from maxtext.m3 import models as m3_models
from maxtext.m3.models.qwen38_flash_next import loss as qwen38_loss
from tests.utils import zero_gen_logits_checker as checker
import numpy as np

torch: Any
transformers: Any
modeling_qwen4_exp: Any

try:
  import torch  # pylint: disable=g-import-not-at-top
  import transformers  # pylint: disable=g-import-not-at-top
  from transformers.models.qwen4_exp import modeling_qwen4_exp  # pylint: disable=g-import-not-at-top
except ImportError:
  torch = None
  transformers = None
  modeling_qwen4_exp = None


def build_byte_mod_test_tokens(seq_len: int, vocab_size: int = 1024, batch_size: int = 1) -> np.ndarray:
  """Deterministically encodes GOLDEN_BATCH_TEXT into an integer token array for unit tests."""
  token_vals = np.frombuffer(checker.GOLDEN_BATCH_TEXT.encode("utf-8"), dtype=np.uint8).astype(np.int64) % vocab_size
  return np.tile(np.tile(token_vals, int(np.ceil(seq_len / len(token_vals))))[:seq_len][None, :], (batch_size, 1))


def encode_golden_batch_hf(tokenizer_or_path: Any, seq_len: int, batch_size: int = 1) -> np.ndarray:
  """Encodes GOLDEN_BATCH_TEXT using HuggingFace AutoTokenizer."""
  is_path = isinstance(tokenizer_or_path, (str, epath.Path))
  tok = transformers.AutoTokenizer.from_pretrained(str(tokenizer_or_path)) if is_path else tokenizer_or_path
  ids = np.asarray(tok.encode(checker.GOLDEN_BATCH_TEXT, add_special_tokens=False), dtype=np.int64)
  return np.tile(np.tile(ids, int(np.ceil(seq_len / len(ids))))[:seq_len][None, :], (batch_size, 1))


def encode_golden_batch_maxtext(tokenizer_or_path: Any, seq_len: int, batch_size: int = 1) -> np.ndarray:
  """Encodes GOLDEN_BATCH_TEXT using MaxText HuggingFace tokenizer path."""
  return encode_golden_batch_hf(tokenizer_or_path, seq_len=seq_len, batch_size=batch_size)


def assert_token_id_parity(jax_tokens: np.ndarray, pytorch_tokens: np.ndarray) -> None:
  """Asserts exact shape and element-wise token ID parity between JAX and PyTorch."""
  if (j_arr := np.asarray(jax_tokens)).shape != (p_arr := np.asarray(pytorch_tokens)).shape:
    raise ValueError(f"Token shape mismatch: jax={j_arr.shape} vs pytorch={p_arr.shape}")
  if len(diff := np.argwhere(j_arr != p_arr)) > 0:
    raise ValueError(f"pytorch_token_ids != jax_token_ids at {len(diff)} positions, first mismatch: {tuple(diff[0])}")


def extract_model_prefill_outputs(model: Any, tokens: np.ndarray, *, return_aux: bool = True) -> dict[str, Any]:
  """Executes prefill forward pass; aux is list[dict] or {"layer_aux": ...} dict."""
  toks, out = jnp.asarray(tokens, dtype=jnp.int32), {"tokens": np.asarray(tokens, dtype=np.int64)}
  res = model(toks, return_aux=True) if return_aux else model(toks)
  logits, aux = res if return_aux else (res[0] if isinstance(res, (tuple, list)) else res, None)
  out["logits"] = np.asarray(logits, dtype=np.float32)
  if aux is not None:
    layer_auxes = aux["layer_aux"] if isinstance(aux, dict) else aux
    for k, out_k in (("router_indices", "routing_indices"), ("selected_block_indices", "indexer_selected_blocks")):
      if lst := [np.asarray(a[k]) for a in layer_auxes if k in a]:
        out[out_k] = np.stack(lst, axis=0)
  return out


def derive_selected_blocks_from_mask(
    mask: np.ndarray, index_block_size: int = 4, index_budget_blocks: int = 512
) -> np.ndarray:
  """Derives selected block indices from QSA indexer mask under CL 4 N1 rule."""
  b_sz, s_len, _ = (arr := np.asarray(mask)[:, 0] if np.asarray(mask).ndim == 4 else np.asarray(mask)).shape
  is_sel = (arr == 0.0) if np.issubdtype(arr.dtype, np.floating) else (arr != 0)
  res, bs = np.full((b_sz, s_len, index_budget_blocks), -1, dtype=np.int32), index_block_size
  for b in range(b_sz):
    for q in range(s_len):
      c = [m for m in range((q + 1) // bs) if np.all(is_sel[b, q, m * bs : (m + 1) * bs])][:index_budget_blocks]
      res[b, q, : len(c)] = c
  return res


def extract_pytorch_prefill_outputs(
    pt_model: Any,
    tokens: np.ndarray,
    *,
    index_block_size: int = 4,
    index_budget_blocks: int = 512,
) -> dict[str, Any]:
  """Executes prefill forward pass on PyTorch reference model with hooks for router and indexer."""
  cfg = getattr(c := getattr(pt_model, "config", None), "text_config", c)
  b_size = getattr(cfg, "indexer_compress_ratio", index_block_size)
  n_budget = getattr(cfg, "indexer_budget", index_budget_blocks * b_size) // b_size
  router_outs, indexer_outs, handles = [], [], []
  for mod in pt_model.modules():
    if isinstance(mod, modeling_qwen4_exp.Qwen4ExpTextTopKRouter):
      handles.append(mod.register_forward_hook(lambda _, __, o, ro=router_outs: ro.append(o[2].detach().cpu().numpy())))
    elif isinstance(mod, modeling_qwen4_exp.Qwen4ExpTextQSAIndexer):
      handles.append(mod.register_forward_hook(lambda _, __, o, io=indexer_outs: io.append(o.detach().cpu().numpy())))
  try:
    with torch.no_grad():
      logits = pt_model(torch.tensor(tokens, dtype=torch.long)).logits.detach().cpu().numpy().astype(np.float32)
  finally:
    for h in handles:
      h.remove()
  b, s = tokens.shape[:2]
  result = {"tokens": np.asarray(tokens, dtype=np.int64), "logits": logits}
  if router_outs:
    result["routing_indices"] = np.stack([r.reshape(b, s, -1).astype(np.int32) for r in router_outs], 0)
  if indexer_outs:
    res_blocks = [derive_selected_blocks_from_mask(m, b_size, n_budget) for m in indexer_outs]
    result["indexer_selected_blocks"] = np.stack(res_blocks, 0)
  return result


def run_generate_goldens(
    golden_path: str | epath.Path,
    *,
    seq_len: int,
    hf_ckpt_path: str | epath.Path | None = None,
    pt_model: Any | None = None,
    tokenizer: Any | None = None,
    tokens: np.ndarray | None = None,
    target_mask: np.ndarray | None = None,
    dtype_str: str = "float32",
    index_block_size: int = 4,
    index_budget_blocks: int = 512,
) -> dict[str, Any]:
  """Generates and serializes golden prefill tensors via PyTorch CPU reference model."""
  if tokens is not None:
    eval_tokens = np.asarray(tokens, dtype=np.int64)
  elif (tok_src := tokenizer or hf_ckpt_path) is not None:
    eval_tokens = encode_golden_batch_hf(tok_src, seq_len=seq_len)
  else:
    raise ValueError("Either tokens, tokenizer, or hf_ckpt_path must be provided to determine tokens.")

  if (model := pt_model) is None and hf_ckpt_path is None:
    raise ValueError("Either pt_model or hf_ckpt_path must be provided.")
  if model is None:
    model = modeling_qwen4_exp.Qwen4ExpForCausalLM.from_pretrained(str(hf_ckpt_path), torch_dtype=torch.float32)

  outputs = extract_pytorch_prefill_outputs(
      model, eval_tokens, index_block_size=index_block_size, index_budget_blocks=index_budget_blocks
  )
  checker.save_golden_npz(golden_path, target_mask=target_mask, seq_len=seq_len, dtype_str=dtype_str, **outputs)
  return {"passed": True, "golden_path": str(golden_path), **outputs}


def run_prefill_logits_check(
    model: Any,
    golden_path: str | epath.Path,
    jax_tokens: np.ndarray,
    *,
    seq_len: int | None = None,
    mask: np.ndarray | None = None,
    require_aux: bool = True,
) -> dict[str, Any]:
  """Verifies Tier 1A routing, QSA indexer blocks, and Tier 1B logits against golden .npz."""
  golden = checker.load_golden_npz(golden_path)
  eval_tokens = np.asarray(golden["tokens"], dtype=np.int64)
  assert_token_id_parity(jax_tokens, eval_tokens)
  outputs = extract_model_prefill_outputs(model, eval_tokens, return_aux=require_aux)

  def _chk_aux(k, fn):
    if not (require_aux and k in golden):
      return {"passed": True, "skipped": True}
    return fn(outputs[k], golden[k]) if k in outputs else {"passed": False, "error": f"Golden contains {k}."}

  routing = _chk_aux("routing_indices", checker.check_routing_parity)
  indexer = _chk_aux("indexer_selected_blocks", checker.check_indexer_block_parity)
  logits = checker.check_logits_parity(
      outputs["logits"], golden["logits"], mask=mask, seq_len=int(seq_len or eval_tokens.shape[1])
  )
  passed = bool(routing["passed"] and indexer["passed"] and logits["passed"])
  return {"passed": passed, "routing": routing, "indexer_blocks": indexer, "logits": logits}


def run_tinymmlu_check(
    model: Any,
    golden_path: str | epath.Path,
    jax_tokens: np.ndarray,
    *,
    tol_nats: float = 1e-4,
    shift: bool = True,
) -> dict[str, Any]:
  """Checks tinyMMLU cross-entropy and perplexity parity against golden outputs."""
  golden = checker.load_golden_npz(golden_path)
  assert_token_id_parity(jax_tokens, eval_tokens := np.asarray(golden["tokens"], dtype=np.int64))
  jax_logits = extract_model_prefill_outputs(model, eval_tokens, return_aux=False)["logits"]
  return checker.check_tinymmlu_loss_delta(
      jax_logits, golden["logits"], eval_tokens, target_mask=golden.get("target_mask"), tol_nats=tol_nats, shift=shift
  )


def run_holdout_ce_check(
    post_sft_model: Any,
    pre_sft_source: Any | str | epath.Path,
    tokens: np.ndarray | None = None,
    target_mask: np.ndarray | None = None,
    shift: bool = True,
) -> dict[str, Any]:
  """Verifies post-SFT holdout CE loss improvement and finite total loss."""
  if isinstance(pre_sft_source, (str, epath.Path)):
    golden = checker.load_golden_npz(pre_sft_source)
    eval_tokens = np.asarray(tokens if tokens is not None else golden["tokens"], dtype=np.int64)
    pre_logits = np.asarray(golden["logits"], dtype=np.float32)
    eval_mask = target_mask if target_mask is not None else golden.get("target_mask")
  else:
    eval_tokens = np.asarray(
        tokens if tokens is not None else build_byte_mod_test_tokens(512),
        dtype=np.int64,
    )
    pre_logits = extract_model_prefill_outputs(pre_sft_source, eval_tokens, return_aux=False)["logits"]
    eval_mask = target_mask

  post_logits = extract_model_prefill_outputs(post_sft_model, eval_tokens, return_aux=False)["logits"]
  total_loss = 0.0
  if eval_tokens.shape[1] >= 2:
    tj = jnp.asarray(eval_tokens, dtype=jnp.int32)
    mj = jnp.asarray(eval_mask, dtype=jnp.float32) if eval_mask is not None else None
    inp, tgt, t_mask = (tj[:, :-1], tj[:, 1:], mj[:, 1:] if mj is not None else None) if shift else (tj, tj, mj)
    t_loss, _ = qwen38_loss.compute_total_loss(
        post_sft_model,
        inp,
        tgt,
        target_mask=t_mask,
        use_mtp=getattr(post_sft_model, "mtp", None) is not None,
    )
    total_loss = float(t_loss)
  res = checker.check_holdout_ce(post_logits, pre_logits, eval_tokens, target_mask=eval_mask, shift=shift)
  return {**res, "total_loss": total_loss}


def run_cli(
    argv: Sequence[str] | None = None,
    model_factory: Callable[[dict[str, str]], Any] | None = None,
    token_factory: Callable[[str, int], np.ndarray] | None = None,
) -> dict[str, Any]:
  """CLI entry point parsing arguments, constructing model, and executing Zero-Gen mode."""
  args = list(sys.argv[1:] if argv is None else argv)
  parsed = {arg.split("=", 1)[0].removeprefix("--"): arg.split("=", 1)[1] for arg in args if "=" in arg}
  if not (mode := parsed.get("mode")):
    raise ValueError("CLI argument --mode is required.")
  if mode not in {
      "prefill_logits",
      "tier1b_logits",
      "tinymmlu",
      "holdout_ce",
      "generate_goldens",
  }:
    raise ValueError(f"Unsupported mode: {mode}")
  if not parsed.get("golden_tensor_path"):
    raise ValueError("golden_tensor_path is required for CLI execution.")
  if mode == "generate_goldens" and "seq_len" not in parsed:
    raise ValueError("CLI argument --seq_len is required when mode=generate_goldens.")

  raw_seq_len = int(parsed["seq_len"]) if "seq_len" in parsed else None
  seq_len, p_str = raw_seq_len or 512, parsed["golden_tensor_path"]
  default_fname = "tinymmlu_100q.npz" if mode == "tinymmlu" else f"golden_batch_{seq_len}.npz"
  resolved = epath.Path(p_str if p_str.endswith(".npz") else f"{p_str}/{default_fname}")
  if mode == "generate_goldens":
    return run_generate_goldens(
        resolved,
        seq_len=seq_len,
        hf_ckpt_path=parsed.get("hf_ckpt_path"),
        tokenizer=parsed.get("tokenizer_path"),
    )

  if token_factory is not None:
    jax_tokens = token_factory(str(resolved), seq_len)
  elif "tokenizer_path" in parsed:
    jax_tokens = encode_golden_batch_maxtext(parsed["tokenizer_path"], seq_len=seq_len)
  else:
    raise ValueError("tokenizer_path or token_factory is required to verify JAX token ID" " parity.")

  if model_factory is not None:
    model = model_factory(parsed)
  else:
    model = m3_models.create_model(pyconfig.initialize(args), None, rngs=nnx.Rngs(params=0))
  if mode in {"prefill_logits", "tier1b_logits"}:
    req_aux = mode != "tier1b_logits"
    return run_prefill_logits_check(model, resolved, jax_tokens, seq_len=raw_seq_len, require_aux=req_aux)
  if mode == "tinymmlu":
    return run_tinymmlu_check(model, resolved, jax_tokens)
  return run_holdout_ce_check(model, resolved, tokens=jax_tokens)
