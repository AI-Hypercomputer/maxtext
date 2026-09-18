#!/usr/bin/env python3
"""Build a GSM8K prompts JSONL for the rl_logprob_parity harness.

WHY THIS EXISTS
---------------
The offline parity harness (`compare_trainer_sampler.py`) measures a sampler-vs-trainer
KL of ~0.003 nats/token. Production measures ~0.075 on the same model, same engines and
same 512+1024 shape -- 25x worse. One of the few surviving explanations is the CORPUS:
the harness tokenizes MaxText markdown/python sources, while production runs GSM8K in
the VTC prompt template. Corpus+length alone is known to be first-order (it swung
per-token OOB by 27 points between the two committed reference runs on the PR branch
with zero numerics change), so it has to be excluded before anything more expensive is
built.

PRODUCTION APPLIES NO CHAT TEMPLATE
-----------------------------------
This is the single most important fact here, and it is easy to get wrong. The
distributed GSM8K job runs with `CHAT_PARSER=raw`:

    run_raiden_tis.sh:162  ->  k8s_launcher.sh:502  ->  run_rollout_node.py:85-86
    ->  chat_template_parser.parser.RawTextParser  (parser.py:192-206)

`RawTextParser.parse` joins the non-empty message contents with "\\n". The system
message is empty and therefore dropped, so the string that reaches vLLM is
`VTC_PROMPT_TEMPLATE.format(question)` **verbatim** -- no <|im_start|>, no system
preamble, no <think> block, no generation prompt.

That is deliberate. RawTextParser's own docstring (parser.py:170-184) explains it:
a completion-style prompt that ends mid-structure -- here, with an *opened*
`<reasoning>` tag the model is expected to close -- would be broken by a chat
template, which would terminate the user turn and re-open an assistant turn,
orphaning the open tag.

Two consequences this script is built around:

1. `--chat-template` is OPT-IN, not the default. The default reproduces production.
   The opt-in arm exists because feeding an instruct-tuned model a raw completion-style
   prompt is plausibly off-distribution, which would raise output entropy and widen the
   sampler/trainer gap -- a testable contributor to the 25x. Running both arms measures
   that directly.

2. Each row must END with the open `<reasoning>` tag, exactly where production starts
   generating. `stage_tokenize` truncates with `ids[:SEQ_LEN]`, which keeps the HEAD
   (compare_trainer_sampler.py:154), so a row longer than SEQ_LEN would be cut in the
   middle of some earlier problem and generation would start from arbitrary text. This
   script therefore emits rows that are EXACTLY `--min-tokens` long, making that slice
   an identity operation. (Production's own `_left_pad` also keeps the tail,
   batch_assembly.py:143-148, so tail-preserving is the faithful choice.)

WHAT IT DELIBERATELY DOES *NOT* DO
----------------------------------
It does not emit one prompt per row. A real VTC-wrapped GSM8K prompt is only ~150-250
tokens, and the harness requires every row to be at least SEQ_LEN
(compare_trainer_sampler.py:152-156). Padding to 512 is not representable: the harness's
trainer stage hardcodes `pos = arange(seq_len)` and `seg = ones`
(compare_trainer_sampler.py:510-511), so left-pad positions would be scored as real
tokens. Instead each row is a run of earlier GSM8K question/solution pairs followed by
a final, un-answered VTC prompt. The token distribution is faithful and the generation
start point is faithful; the prompt *layout* (one short unpadded prompt) is not.

Read the result accordingly: this isolates CORPUS, not prompt layout. A faithful
single-prompt run needs left-padding support in the harness and is a separate change.
"""

import argparse
import json
import os
import re
import sys

# Verbatim from tunix/utils/gsm8k_vtc.py:44-50. Non-raw triple-quoted string, so
# "\\boxed" renders as "\boxed" and "{{}}" renders as "{}" after .format().
VTC_PROMPT_TEMPLATE = """Solve the following math problem.
First, put your detailed step-by-step reasoning process inside <reasoning>...</reasoning> tags.
Then, put your final numerical answer inside <answer>\\boxed{{}}</answer> tags. Do not put anything else in the answer tags.

Problem: {}
<reasoning>
"""


def load_gsm8k(path, split, limit):
  """Return a list of (question, answer) pairs.

  Order of preference:
    1. `--gsm8k-file`, a local JSONL of {"question","answer"} -- this is what the
       launcher ships, and it is preferred precisely because it CANNOT fail. The first
       version of this script downloaded the dataset in the pod and died there; the pod
       was then garbage-collected, so the failure left no log at all. Data that travels
       with the job removes an entire class of remote failure from a measurement whose
       only purpose is to produce one number.
    2. HuggingFace `datasets`.
    3. TFDS.
  """
  if path:
    rows = []
    with open(path, encoding="utf-8") as fh:
      for ln in fh:
        ln = ln.strip()
        if ln:
          rec = json.loads(ln)
          rows.append((rec["question"], rec["answer"]))
    print(f"[corpus] loaded {len(rows)} rows from {path}", flush=True)
    return rows[:limit] if limit else rows

  try:
    from datasets import load_dataset  # pylint: disable=g-import-not-at-top

    ds = load_dataset("openai/gsm8k", "main", split=split)
    rows = [(r["question"], r["answer"]) for r in ds]
    print(f"[corpus] loaded {len(rows)} rows from HF openai/gsm8k split={split}", flush=True)
    return rows[:limit] if limit else rows
  except Exception as exc:  # pylint: disable=broad-except
    print(f"[corpus] HF datasets path failed ({exc!r}); trying TFDS", flush=True)

  import tensorflow_datasets as tfds  # pylint: disable=g-import-not-at-top

  ds = tfds.load("gsm8k", split=split,
                 data_dir=os.environ.get("TFDS_DATA_DIR", "/tmp/gsm8k_data"))
  rows = []
  for ex in tfds.as_numpy(ds):
    q = ex["question"].decode("utf-8") if isinstance(ex["question"], bytes) else str(ex["question"])
    a = ex["answer"].decode("utf-8") if isinstance(ex["answer"], bytes) else str(ex["answer"])
    rows.append((q, a))
  print(f"[corpus] loaded {len(rows)} rows from TFDS gsm8k split={split}", flush=True)
  return rows[:limit] if limit else rows


def as_episode(answer):
  """Render a GSM8K reference solution as a COMPLETED VTC episode.

  The raw dataset answer is terse and carries `<<48/2=24>>` calculator annotations and a
  `#### 72` final line -- a format the policy never emits. Filler that looks like model
  output keeps the row on-distribution, which matters because the whole point of this
  corpus is to measure how the token distribution affects sampler/trainer divergence.
  """
  body, _, final = answer.partition("####")
  body = re.sub(r"<<[^>]*>>", "", body).strip()
  final = final.strip()
  return f"{body}\n</reasoning>\n<answer>\\boxed{{{final}}}</answer>"


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", required=True)
  ap.add_argument("--rows", type=int, default=32, help="number of JSONL rows to emit")
  ap.add_argument("--min-tokens", type=int, default=512,
                  help="exact token length of every row; must equal the harness --prompt-len")
  ap.add_argument("--split", default="train")
  ap.add_argument("--gsm8k-file", default=None,
                  help="local JSONL of {\"question\",\"answer\"} to use instead of downloading. "
                       "The launcher ships one; prefer it, because a download that fails in "
                       "the pod takes the whole measurement with it.")
  ap.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B")
  ap.add_argument("--chat-template", action="store_true",
                  help="OPT-IN contrast arm: wrap each prompt in the Qwen chat template. "
                       "Production does NOT do this (CHAT_PARSER=raw), so leave this off "
                       "for the production-faithful run.")
  ap.add_argument("--natural", action="store_true",
                  help="emit ONE bare GSM8K prompt per row at its real length (~150 tokens) with no "
                       "filler and no length floor. Pair with the harness's --left-pad. The filler "
                       "machinery exists only to satisfy the old requirement that every row be exactly "
                       "--min-tokens dense tokens -- which is the artificial layout --left-pad exists "
                       "to stop using.")
  args = ap.parse_args()
  if args.rows <= 0:
    args.rows = 32

  from transformers import AutoTokenizer  # pylint: disable=g-import-not-at-top

  tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
  target = args.min_tokens

  def enc(text):
    # No kwargs, matching VllmSampler.tokenize (vllm_sampler.py:427-431), which leaves
    # add_special_tokens at the HF default.
    return tok(text)["input_ids"]

  def wrap(prompt):
    if not args.chat_template:
      return prompt  # production: RawTextParser passes the VTC prompt through verbatim
    return tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )

  pairs = load_gsm8k(args.gsm8k_file, args.split, limit=args.rows * 40)
  if not pairs:
    sys.exit("[corpus] no GSM8K rows loaded")

  if args.natural:
    # One problem per row, exactly the string production hands vLLM. No filler, no
    # concatenation, no length surgery -- so nothing here can silently alter the
    # prompt the model is asked to continue.
    rows = [wrap(VTC_PROMPT_TEMPLATE.format(q)) for q, _ in pairs[: args.rows]]
    if len(rows) < args.rows:
      sys.exit(f"[corpus] only {len(rows)}/{args.rows} GSM8K pairs available")
    lens = sorted(len(enc(t)) for t in rows)
    if lens[-1] > target:
      # stage_tokenize keeps the TAIL under --left-pad, so an over-long row would lose
      # its head. Say so rather than letting it happen quietly.
      print(f"[corpus] WARNING: {sum(l > target for l in lens)} row(s) exceed --min-tokens={target} "
            f"(max {lens[-1]}); the harness will keep their tail", flush=True)
    with open(args.out, "w", encoding="utf-8") as fh:
      for text in rows:
        fh.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    print(f"[corpus] wrote {len(rows)} NATURAL-length rows to {args.out}; "
          f"token length min {lens[0]} med {lens[len(lens) // 2]} max {lens[-1]} "
          f"(chat_template={args.chat_template})", flush=True)
    print(f"[corpus] row 0 tail: {rows[0][-120:]!r}", flush=True)
    return

  rows, i = [], 0
  while len(rows) < args.rows and i < len(pairs):
    # The LAST unit is an un-answered prompt, so the row ends exactly where production
    # begins generating: on the open <reasoning> tag. Everything before it is filler
    # whose only job is to reach the length floor.
    final_q, _ = pairs[i]
    i += 1
    tail = wrap(VTC_PROMPT_TEMPLATE.format(final_q))

    filler = []
    while i < len(pairs) and len(enc("\n\n".join(filler + [tail]))) < target:
      q, a = pairs[i]
      i += 1
      filler.append(wrap(VTC_PROMPT_TEMPLATE.format(q)) + as_episode(a))

    ids = enc("\n\n".join(filler + [tail]))
    if len(ids) < target:
      break  # ran out of source data mid-row

    # Keep the TAIL, then re-encode: decode(ids[-target:]) is not guaranteed to
    # re-tokenize to exactly `target` tokens (BPE merges across the new boundary), and
    # a row that re-encodes shorter than target would be rejected by stage_tokenize
    # while a longer one would be head-truncated. Nudge the cut point until the
    # round-trip is exact.
    text = None
    for cut in range(target, min(len(ids), target + 24) + 1):
      cand = tok.decode(ids[-cut:])
      if len(enc(cand)) == target:
        text = cand
        break
    if text is None:
      print(f"[corpus] WARNING: row {len(rows)} would not round-trip to exactly "
            f"{target} tokens; skipping", flush=True)
      continue

    # Load-bearing: the row must end with the final prompt unit, so that generation
    # starts exactly where it would in the corresponding production config. If this
    # trips, the left-truncation ate into the final prompt and the run would be
    # measuring generation from arbitrary mid-problem text while looking perfectly
    # healthy.
    #
    # The expected suffix is derived from `tail` rather than hardcoded, because the two
    # arms legitimately end differently:
    #   raw  -> "...<reasoning>\n"                     (open tag, model continues it)
    #   chat -> "...<|im_start|>assistant\n<think>\n"  (chat template CLOSES the user
    #            turn and orphans the <reasoning> tag -- exactly the breakage that
    #            RawTextParser exists to avoid, parser.py:170-184)
    probe = tail[-100:]
    if not text.endswith(probe):
      sys.exit(f"[corpus] row {len(rows)} does not end with the final prompt unit; "
               f"expected suffix={probe!r} got tail={text[-100:]!r}")
    rows.append(text)

  if len(rows) < args.rows:
    sys.exit(f"[corpus] only built {len(rows)}/{args.rows} rows from {len(pairs)} GSM8K pairs")

  with open(args.out, "w", encoding="utf-8") as fh:
    for text in rows:
      fh.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

  lens = sorted(len(enc(t)) for t in rows)
  print(f"[corpus] wrote {len(rows)} rows to {args.out}", flush=True)
  print(f"[corpus] chat_template={args.chat_template} "
        f"token lengths: min={lens[0]} med={lens[len(lens) // 2]} max={lens[-1]} "
        f"(all must equal {target})", flush=True)
  print(f"[corpus] row 0 tail: {rows[0][-220:]!r}", flush=True)


if __name__ == "__main__":
  main()
