# Vendored subset of OpenAI's simple-evals (https://github.com/openai/simple-evals).
# Vendored from upstream commit: 652c89d
#
# These files are copied verbatim from the upstream project and are licensed
# under the MIT License (see LICENSE in this directory), with one mechanical
# deviation: in common.py, the MULTILINGUAL_ANSWER_REGEXES entries containing
# "\s" were given raw-string (r"...") prefixes to silence Python 3.12's
# SyntaxWarning for invalid escape sequences. The string values are unchanged.
# Only the grader-free evals required by the MaxText simple_evals runner are
# vendored:
#
#   types.py       Eval / Sampler / Result base types.
#   common.py      Shared helpers (HTML report, answer extraction, aggregation).
#   mmlu_eval.py   MMLU eval (no grader model required).
#   gpqa_eval.py   GPQA eval (no grader model required).
#   drop_eval.py   DROP eval (no grader model required).
#   mgsm_eval.py   MGSM eval (no grader model required).
#
# Grader-dependent evals (math, simpleqa, browsecomp, healthbench) are
# intentionally not vendored yet; they require an LLM grader endpoint.
#
# GSM8K and AIME are not part of upstream simple-evals. Their Eval
# implementations live under maxtext.eval.native_evals instead of here, since
# this package is for verbatim third-party vendoring only; see that package
# for details.
