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

"""Next-token scoring for any MaxText model served through vLLM on TPU.

This is a thin shim over ``maxtext.inference.vllm_decode``: the engine is built with the exact
same arguments (``create_vllm_llm``) and chat prompts with the same ``build_chat_messages``; on
top of that it adds *scoring* instead of sampling - the next-token distribution (top-k
log-probs) and the probabilities of user supplied candidate continuations. This is the
primitive needed by "decision" / bounded classification models, multiple-choice evals,
reward scoring, etc. It works for every model supported by the MaxText<->vLLM adapter.

Example usage (single prompt, score option letters A/B)::

  python3 -m maxtext.inference.vllm_score src/maxtext/configs/base.yml \
      model_name=qwen3.5-9b \
      tokenizer_path=Qwen/Qwen3.5-9B \
      load_parameters_path=<your_checkpoint_path> \
      vllm_hf_overrides='{architectures: ["MaxTextForCausalLM"]}' \
      ici_tensor_parallelism=4 \
      hbm_utilization_vllm=0.6 \
      max_target_length=2048 \
      use_chat_template=true \
      system_prompt="Answer with only the option letter A or B." \
      "prompt='What is the capital of Germany? A: Berlin, B: Paris'" \
      --candidates=A,B --top_k=10 --method=auto \
      --enable_thinking=false --enable_prefix_caching=false

Add ``use_chat_template=true`` (and optionally ``system_prompt=...``) to wrap ``prompt`` with the
tokenizer chat template exactly as ``vllm_decode`` does.

Batch mode: ``--prompts_file=<jsonl>`` where every line is
``{"prompt": "...", "candidates": ["A", "B"]}`` (``candidates`` optional; strings or token
ids). Batch prompts are used as supplied; chat-template and thinking flags do not apply.
Results are printed and optionally written to ``--output_file`` as JSONL.

Notes on log-prob support in vLLM on TPU (``tpu_inference`` JAX backend):

* ``SamplingParams(logprobs=K)`` returns the top-``K`` entries of ``log_softmax(raw_logits)``
  for every generated token (``logprobs_mode`` only switches raw vs. *processed* logprobs;
  raw logits are not materialized on TPU).
* ``SamplingParams(prompt_logprobs=k)`` always includes the log-prob of the *actual* prompt
  token regardless of rank. Appending a candidate to the prompt and reading its prompt
  log-prob is therefore an exact way to score an arbitrary candidate token.
* ``SamplingParams.logprob_token_ids`` / ``allowed_token_ids`` exist upstream but are not
  wired into the TPU sampler yet, so they are not relied upon here.

Because ``softmax`` restricted to a candidate set is invariant to the constant shift between
raw logits and log-probabilities, candidate probabilities computed from vLLM log-probs are
identical to the ones computed from raw logits.
"""

import json
import os
from typing import Any, Sequence

from absl import app
from absl import flags
import jax
import numpy as np

from maxtext.common.common_types import Config
from maxtext.configs import pyconfig
from maxtext.inference import vllm_decode
import maxtext.integration.vllm.maxtext_vllm_adapter as adapter
from maxtext.utils import max_logging
import transformers
from vllm import LLM
from vllm.inputs import TokensPrompt
from vllm.sampling_params import SamplingParams

DEFAULT_TOP_K = 20

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "candidates", "", "Comma separated candidate text continuations. Use JSONL integer values for raw token IDs."
)
flags.DEFINE_string(
    "prompts_file",
    "",
    "JSONL with {prompt, candidates} per line. Prompts must already be formatted; chat-template flags are ignored.",
)
flags.DEFINE_string("output_file", "", "Optional JSONL output path.")
flags.DEFINE_integer("top_k", DEFAULT_TOP_K, "Number of next-token log-probs to request / print per prompt.")
flags.DEFINE_enum("method", "auto", ["auto", "top_k", "exact"], "Candidate scoring method (see VllmScorer).")
flags.DEFINE_bool("add_special_tokens", False, "Let the tokenizer add BOS/special tokens to text prompts.")
flags.DEFINE_bool(
    "enable_prefix_caching",
    False,
    "Enable vLLM prefix caching. Disabled by default for scoring: on hybrid (GDN/mamba) models the TPU "
    "prefix-cache path was observed to change next-token log-probs for identical prompts.",
)
flags.DEFINE_integer(
    "repeat",
    1,
    "Run the scoring pass this many times (determinism check); results of the last run are reported.",
)
flags.DEFINE_bool(
    "enable_thinking",
    False,
    "Passed to tokenizer.apply_chat_template when use_chat_template=true. Off by default for scoring: Qwen3-style "
    "templates otherwise leave the prompt inside an open <think> block, so the next token is reasoning text rather "
    "than the answer.",
)

flags.register_validator("top_k", lambda value: value >= 1, message="top_k must be at least 1.")
flags.register_validator("repeat", lambda value: value >= 1, message="repeat must be at least 1.")

TokenIds = Sequence[int]
PromptLike = str | TokenIds


def setup_scoring_environment() -> None:
  """Set the TPU environment for scoring with an in-process vLLM engine.

  ``VLLM_ENABLE_V1_MULTIPROCESSING=0`` runs the vLLM ``EngineCore`` in-process. This is
  required because MaxText's ``pyconfig.initialize`` already initializes the JAX TPU backend
  in the calling process; a forked ``EngineCore`` would otherwise deadlock on its first XLA
  compilation (same approach as ``maxtext.eval.runner.server_manager``).
  """
  os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
  os.environ["SKIP_JAX_PRECOMPILE"] = "1"
  os.environ["NEW_MODEL_DESIGN"] = "1"

  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  if "xla_tpu_spmd_rng_bit_generator_unsafe" not in os.environ.get("LIBTPU_INIT_ARGS", ""):
    os.environ["LIBTPU_INIT_ARGS"] = (
        os.environ.get("LIBTPU_INIT_ARGS", "") + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"
    )


class VllmScorer:
  """Model-agnostic next-token scoring on top of a ``vllm.LLM`` engine.

  Typical usage::

    adapter.register(config)
    llm = vllm_decode.create_vllm_llm(config, max_logprobs=DEFAULT_TOP_K, enable_prefix_caching=False)
    scorer = VllmScorer(llm)
    top = scorer.next_token_logprobs(["Paris is the"])                    # {token_id: logprob}
    lps = scorer.candidate_logprobs([prompt_ids], [[tok_a, tok_b]])       # exact candidate scores

  Prompts may be text or pre-tokenized ids. Text prompts are tokenized with
  ``add_special_tokens=False`` so chat-templated prompts are not altered.
  """

  def __init__(self, llm: LLM, default_top_k: int = DEFAULT_TOP_K):
    self.llm = llm
    self.tokenizer = llm.get_tokenizer()
    if default_top_k < 1:
      raise ValueError("default_top_k must be at least 1")
    self.default_top_k = default_top_k
    max_logprobs = llm.llm_engine.model_config.max_logprobs
    if max_logprobs != -1 and default_top_k > max_logprobs:
      max_logging.log(
          f"VllmScorer: default_top_k={default_top_k} exceeds engine max_logprobs={max_logprobs}; "
          f"clamping. Pass max_logprobs=<k> to create_vllm_llm to raise the limit."
      )
      self.default_top_k = max_logprobs

  def to_token_ids(self, prompt: PromptLike, add_special_tokens: bool = False) -> list[int]:
    """Tokenize text or copy an existing token sequence."""
    if isinstance(prompt, str):
      return list(self.tokenizer.encode(prompt, add_special_tokens=add_special_tokens))
    return [int(t) for t in prompt]

  def _tokens_prompts(self, prompts: Sequence[PromptLike], add_special_tokens: bool) -> list[TokensPrompt]:
    return [TokensPrompt(prompt_token_ids=self.to_token_ids(p, add_special_tokens)) for p in prompts]

  def next_token_logprobs(
      self,
      prompts: Sequence[PromptLike],
      top_k: int | None = None,
      add_special_tokens: bool = False,
  ) -> list[dict[int, float]]:
    """Returns the top-``k`` next-token log-probs (``log_softmax`` of raw logits) per prompt.

    One single-token generation request (``max_tokens=1``, greedy) is issued per prompt; all prompts are
    batched by the vLLM scheduler.

    Returns:
      A list (one per prompt) of ``{token_id: logprob}`` dicts containing the top-``k`` tokens
      (the greedy token is always among them).
    """
    k = self.default_top_k if top_k is None else top_k
    max_logprobs = self.llm.llm_engine.model_config.max_logprobs
    if k < 1 or (max_logprobs != -1 and k > max_logprobs):
      raise ValueError(f"top_k must be positive and within the engine max_logprobs limit ({max_logprobs})")
    sampling_params = SamplingParams(max_tokens=1, temperature=0.0, logprobs=k, detokenize=False)
    outputs = self.llm.generate(self._tokens_prompts(prompts, add_special_tokens), sampling_params, use_tqdm=False)
    results: list[dict[int, float]] = []
    for output in outputs:
      step_logprobs = output.outputs[0].logprobs
      if not step_logprobs:
        raise RuntimeError("vLLM did not return logprobs; make sure the engine was created with max_logprobs>=k.")
      results.append({int(tid): float(lp.logprob) for tid, lp in step_logprobs[0].items()})
    return results

  def greedy_next_token(self, prompts: Sequence[PromptLike], add_special_tokens: bool = False) -> list[int]:
    """Returns the arg-max next token id for each prompt."""
    sampling_params = SamplingParams(max_tokens=1, temperature=0.0, detokenize=False)
    outputs = self.llm.generate(self._tokens_prompts(prompts, add_special_tokens), sampling_params, use_tqdm=False)
    return [int(o.outputs[0].token_ids[0]) for o in outputs]

  def exact_candidate_logprobs(
      self,
      prompts: Sequence[PromptLike],
      candidate_token_ids: Sequence[TokenIds],
      add_special_tokens: bool = False,
  ) -> list[np.ndarray]:
    """Exact next-token log-probs of arbitrary candidate tokens via ``prompt_logprobs``.

    For every (prompt, candidate) pair one request ``prompt + [candidate]`` is issued and the
    log-prob of the appended candidate (conditioned on the prompt) is read back from vLLM's
    prompt log-probs. Never misses a candidate; costs ``sum(len(cands))`` prefill requests
    (batched by the scheduler).
    """
    if len(prompts) != len(candidate_token_ids):
      raise ValueError("prompts and candidate_token_ids must have the same length")
    for candidates in candidate_token_ids:
      for candidate in candidates:
        _validate_token_id(self.tokenizer, candidate)
    requests: list[TokensPrompt] = []
    owners: list[tuple[int, int]] = []
    for i, (prompt, cands) in enumerate(zip(prompts, candidate_token_ids)):
      ids = self.to_token_ids(prompt, add_special_tokens)
      for j, cand in enumerate(cands):
        requests.append(TokensPrompt(prompt_token_ids=ids + [int(cand)]))
        owners.append((i, j))
    if not requests:
      return [np.zeros((0,), dtype=np.float64) for _ in prompts]

    # Request a positive count to avoid backends treating zero as disabled.
    # The actual candidate token is included regardless of its rank.
    sampling_params = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1, detokenize=False)
    outputs = self.llm.generate(requests, sampling_params, use_tqdm=False)

    results = [np.full((len(c),), np.nan, dtype=np.float64) for c in candidate_token_ids]
    for (i, j), output, request in zip(owners, outputs, requests):
      cand = request["prompt_token_ids"][-1]
      prompt_logprobs = output.prompt_logprobs
      if not prompt_logprobs or prompt_logprobs[-1] is None or cand not in prompt_logprobs[-1]:
        raise RuntimeError(f"vLLM did not return a prompt logprob for candidate token {cand} (prompt {i}).")
      results[i][j] = float(prompt_logprobs[-1][cand].logprob)
    return results

  def candidate_logprobs(
      self,
      prompts: Sequence[PromptLike],
      candidate_token_ids: Sequence[TokenIds],
      method: str = "auto",
      top_k: int | None = None,
      add_special_tokens: bool = False,
      *,
      top_k_logprobs: Sequence[dict[int, float]] | None = None,
  ) -> list[np.ndarray]:
    """Returns ``log_softmax`` scores of the given candidate next tokens for each prompt.

    Args:
      prompts: Batch of prompts (text or token ids).
      candidate_token_ids: Per-prompt list of candidate token ids to score.
      method: ``"top_k"`` (one request per prompt; raises if a candidate is outside the top-k),
        ``"exact"`` (``prompt_logprobs`` based, one request per candidate) or ``"auto"``
        (top-k first, exact fallback only for prompts with missing candidates).
      top_k: Number of logprobs to request in the top-k pass (defaults to ``default_top_k``).
      add_special_tokens: Whether to let the tokenizer add special tokens to text prompts.
      top_k_logprobs: Optional precomputed distributions aligned with prompts, using the
        same tokenization and top_k. Ignored for the exact method.

    Returns:
      A list of ``np.ndarray`` (float64) aligned with ``candidate_token_ids``.
    """
    if method not in ("auto", "top_k", "exact"):
      raise ValueError(f"Unknown method {method!r}; expected 'auto', 'top_k' or 'exact'.")
    if len(prompts) != len(candidate_token_ids):
      raise ValueError("prompts and candidate_token_ids must have the same length")
    for candidates in candidate_token_ids:
      for candidate in candidates:
        _validate_token_id(self.tokenizer, candidate)

    if method == "exact":
      return self.exact_candidate_logprobs(prompts, candidate_token_ids, add_special_tokens)

    if top_k_logprobs is not None and len(top_k_logprobs) != len(prompts):
      raise ValueError("prompts and top_k_logprobs must have the same length")
    top = (
        self.next_token_logprobs(prompts, top_k=top_k, add_special_tokens=add_special_tokens)
        if top_k_logprobs is None
        else top_k_logprobs
    )
    results: list[np.ndarray] = []
    missing: list[int] = []
    for i, (dist, cands) in enumerate(zip(top, candidate_token_ids)):
      scores = np.array([dist.get(int(c), np.nan) for c in cands], dtype=np.float64)
      if np.isnan(scores).any():
        missing.append(i)
      results.append(scores)

    if missing:
      if method == "top_k":
        raise RuntimeError(
            f"Candidates outside top-{top_k or self.default_top_k} for prompts {missing}; "
            "increase top_k / max_logprobs or use method='exact' or 'auto'."
        )
      max_logging.log(f"VllmScorer: falling back to exact prompt_logprobs scoring for prompts {missing}.")
      exact = self.exact_candidate_logprobs(
          [prompts[i] for i in missing], [candidate_token_ids[i] for i in missing], add_special_tokens
      )
      for i, scores in zip(missing, exact):
        results[i] = scores
    return results

  def candidate_probabilities(
      self,
      prompts: Sequence[PromptLike],
      candidate_token_ids: Sequence[TokenIds],
      **kwargs: Any,
  ) -> list[np.ndarray]:
    """Softmax over the candidate set only (identical to ``softmax(raw_logits[cands])``)."""
    return [
        _candidate_probabilities(scores) for scores in self.candidate_logprobs(prompts, candidate_token_ids, **kwargs)
    ]


def _parse_candidates(raw: Any) -> list[Any]:
  """Parse comma-separated options or copy a JSON candidate list."""
  if raw is None or raw == "":
    return []
  if isinstance(raw, str):
    raw = [c for c in raw.split(",") if c != ""]
  return list(raw)


def _validate_token_id(tokenizer: Any, token_id: int) -> None:
  """Reject non-integer and out-of-vocabulary candidate token IDs."""
  if isinstance(token_id, (bool, np.bool_)) or not isinstance(token_id, (int, np.integer)):
    raise ValueError("Candidate token IDs must be integers")
  if not 0 <= token_id < len(tokenizer):
    raise ValueError(f"Candidate token ID {token_id} is outside the tokenizer vocabulary")


def candidate_token_id(
    tokenizer: Any,
    prompt_text: str,
    prompt_ids: Sequence[int],
    candidate: Any,
    add_special_tokens: bool = False,
) -> int:
  """Resolves a candidate (token id or string) to the single token id that follows ``prompt_ids``.

  Only integer candidates are raw token IDs; numeric strings are candidate text.
  Strings are tokenized *in context* using the original prompt and its special-token
  setting (prompt + candidate) so that BPE merges across the
  boundary are detected. Multi-token candidates are rejected: next-token scoring is only well
  defined for single-token continuations.
  """
  if add_special_tokens and prompt_ids and prompt_ids[-1] == tokenizer.eos_token_id:
    raise ValueError("The tokenizer appended EOS to the prompt; use add_special_tokens=false for next-token scoring.")
  if isinstance(candidate, bool) or not isinstance(candidate, (int, str)):
    raise ValueError("Candidates must be text strings or integer token IDs")
  if isinstance(candidate, int):
    _validate_token_id(tokenizer, candidate)
    return candidate
  combined = tokenizer.encode(prompt_text + candidate, add_special_tokens=add_special_tokens)
  suffix = combined[len(prompt_ids) :]
  if list(combined[: len(prompt_ids)]) != list(prompt_ids) or len(suffix) != 1:
    raise ValueError(
        f"Candidate {candidate!r} is not a single token at the prompt boundary (got {len(suffix)} tokens: {suffix})."
    )
  return int(suffix[0])


def load_requests(config: Config, tokenizer: Any) -> list[dict[str, Any]]:
  """Builds the list of {prompt, prompt_ids, candidates, candidate_ids} requests from flags/config."""
  requests: list[dict[str, Any]] = []
  if FLAGS.prompts_file:
    with open(FLAGS.prompts_file, "r", encoding="utf-8") as f:
      for line in f:
        line = line.strip()
        if not line:
          continue
        item = json.loads(line)
        requests.append({"prompt": item["prompt"], "candidates": _parse_candidates(item.get("candidates"))})
  else:
    prompt = config.prompt
    if config.use_chat_template:
      # Same chat formatting as vllm_decode (system_prompt / multimodal placeholders included),
      # plus enable_thinking so scoring happens at the answer position.
      prompt = tokenizer.apply_chat_template(
          vllm_decode.build_chat_messages(config),
          tokenize=False,
          add_generation_prompt=True,
          add_special_tokens=False,
          enable_thinking=FLAGS.enable_thinking,
      )
    requests.append({"prompt": prompt, "candidates": _parse_candidates(FLAGS.candidates)})

  for req in requests:
    req["prompt_ids"] = tokenizer.encode(req["prompt"], add_special_tokens=FLAGS.add_special_tokens)
    req["candidate_ids"] = [
        candidate_token_id(tokenizer, req["prompt"], req["prompt_ids"], candidate, FLAGS.add_special_tokens)
        for candidate in req["candidates"]
    ]
  return requests


def score_requests(scorer: VllmScorer, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
  """Runs top-k next-token scoring and candidate scoring for all requests."""
  tokenizer = scorer.tokenizer
  prompt_ids = [req["prompt_ids"] for req in requests]
  top = scorer.next_token_logprobs(prompt_ids, top_k=FLAGS.top_k)

  with_cands = [i for i, req in enumerate(requests) if req["candidate_ids"]]
  cand_scores: dict[int, np.ndarray] = {}
  if with_cands:
    scores = scorer.candidate_logprobs(
        [prompt_ids[i] for i in with_cands],
        [requests[i]["candidate_ids"] for i in with_cands],
        method=FLAGS.method,
        top_k=FLAGS.top_k,
        top_k_logprobs=[top[i] for i in with_cands],
    )
    cand_scores = dict(zip(with_cands, scores))

  results = []
  for i, req in enumerate(requests):
    dist = top[i]
    greedy_id = max(dist, key=dist.get)
    result: dict[str, Any] = {
        "prompt": req["prompt"],
        "num_prompt_tokens": len(req["prompt_ids"]),
        "greedy_token": {"id": greedy_id, "text": tokenizer.decode([greedy_id]), "logprob": dist[greedy_id]},
        "top_k": [
            {"id": tid, "text": tokenizer.decode([tid]), "logprob": lp}
            for tid, lp in sorted(dist.items(), key=lambda kv: kv[1], reverse=True)[: FLAGS.top_k]
        ],
    }
    if i in cand_scores:
      lps = cand_scores[i]
      probs = _candidate_probabilities(lps)
      result["candidates"] = [
          {"candidate": str(c), "id": tid, "logprob": float(lp), "prob_among_candidates": float(p)}
          for c, tid, lp, p in zip(req["candidates"], req["candidate_ids"], lps, probs)
      ]
      result["prediction"] = str(req["candidates"][int(np.argmax(lps))])
    results.append(result)
  return results


def _candidate_probabilities(logprobs: np.ndarray) -> np.ndarray:
  """Normalize log-probabilities over the supplied candidate set."""
  if len(logprobs) == 0:
    return np.zeros((0,), dtype=np.float64)
  weights = np.exp(logprobs - np.max(logprobs))
  return weights / np.sum(weights)


def log_results(results: Sequence[dict[str, Any]]) -> None:
  """Log each prediction with options ranked by candidate probability."""
  for index, result in enumerate(results, start=1):
    heading = f"Result {index}"
    if "prediction" in result:
      heading += f" — prediction: {result['prediction']}"
    lines = [heading, f"Prompt ({result['num_prompt_tokens']} tokens): {result['prompt']!r}"]
    candidates = result.get("candidates", [])
    if candidates:
      ranked = sorted(candidates, key=lambda option: option["prob_among_candidates"], reverse=True)
      option_width = max(len("Option"), *(len(repr(option["candidate"])) for option in ranked))
      lines.append(f"{'Option':<{option_width}}  {'Option-prob':>12}  {'Log-prob':>10}")
      for option in ranked:
        lines.append(
            f"{option['candidate']!r:<{option_width}}  {option['prob_among_candidates']:>12.2%}  "
            f"{option['logprob']:>10.4f}"
        )
    greedy = result["greedy_token"]
    lines.append(f"Greedy next token: {greedy['text']!r} (id={greedy['id']}, log-prob={greedy['logprob']:.4f})")
    if result["top_k"]:
      lines.append("Top-k next tokens:")
      for token in result["top_k"]:
        lines.append(f"  {token['text']!r} (id={token['id']}, log-prob={token['logprob']:.4f})")
    max_logging.log("\n".join(lines))


def main(argv: Sequence[str]) -> None:
  """Load the model, score requests, and report the option outcomes."""
  setup_scoring_environment()

  config = pyconfig.initialize(argv)
  adapter.register(config)

  # Resolve prompts/candidates first (fail fast on multi-token candidates) - same tokenizer
  # source as vllm_decode; the engine's tokenizer is identical since vLLM loads tokenizer_path.
  tokenizer = transformers.AutoTokenizer.from_pretrained(config.tokenizer_path, token=config.hf_access_token)
  requests = load_requests(config, tokenizer)

  llm = vllm_decode.create_vllm_llm(config, max_logprobs=FLAGS.top_k, enable_prefix_caching=FLAGS.enable_prefix_caching)
  scorer = VllmScorer(llm, default_top_k=FLAGS.top_k)

  for i in range(FLAGS.repeat):
    results = score_requests(scorer, requests)
    if FLAGS.repeat > 1:
      summary = [
          [round(c["logprob"], 6) for c in r["candidates"]]
          if "candidates" in r
          else round(r["greedy_token"]["logprob"], 6)
          for r in results
      ]
      max_logging.log(f"[repeat {i + 1}/{FLAGS.repeat}] candidate/greedy logprobs per prompt: {summary}")

  log_results(results)

  if FLAGS.output_file:
    with open(FLAGS.output_file, "w", encoding="utf-8") as f:
      for result in results:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
    max_logging.log(f"Wrote {len(results)} results to {FLAGS.output_file}")


if __name__ == "__main__":
  app.run(main)
