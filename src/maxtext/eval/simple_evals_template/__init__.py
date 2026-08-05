# Copyright 2026 Google LLC
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

# Vendored subset of OpenAI's simple-evals (https://github.com/openai/simple-evals).
# Vendored from upstream commit: 652c89d
#
# These files are copied verbatim from the upstream project and are licensed
# under the upstream license (see LICENSE in this directory), with one mechanical
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
