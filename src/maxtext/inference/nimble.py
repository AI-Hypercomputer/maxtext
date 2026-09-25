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

"""Bespoke-Nimble bounded schema candidate classification on TPU via MaxText."""

import os
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
import jax
import jax.numpy as jnp
from transformers import AutoTokenizer

from maxtext.configs import pyconfig
from maxtext.inference.maxengine import maxengine

try:
  from maxtext.inference.parallel_schema import prepare_prompts, choice_key
except ImportError:
  try:
    from parallel_schema import prepare_prompts, choice_key
  except ImportError:
    from .parallel_schema import prepare_prompts, choice_key


def _find_base_config() -> str:
  cwd_config = "src/maxtext/configs/base.yml"
  if os.path.exists(cwd_config):
    return cwd_config
  pkg_config = Path(__file__).resolve().parent.parent / "configs" / "base.yml"
  if pkg_config.exists():
    return str(pkg_config)
  return "src/maxtext/configs/base.yml"


def decision_result(row: Mapping[str, Any], logits: Sequence[float]) -> dict[str, Any]:
  """Computes candidate decision and softmax probabilities for a field row."""
  logits_arr = np.asarray(logits[: len(row["choices"])], dtype=np.float64)
  exp_logits = np.exp(logits_arr - np.max(logits_arr))
  probs = exp_logits / np.sum(exp_logits)
  index = int(np.argmax(logits_arr))
  choices = row["choices"]
  result = {
      "prediction": choices[index],
      "probabilities": {choice_key(k): float(v) for k, v in zip(choices, probs)},
      "logits": {choice_key(k): float(v) for k, v in zip(choices, logits_arr)},
  }
  gold = row.get("labels")
  if gold is not None:
    log_probs = logits_arr - np.log(np.sum(exp_logits))
    result.update(
        correct=index == gold,
        nll=float(-log_probs[gold]),
        brier=float(sum((p - int(i == gold)) ** 2 for i, p in enumerate(probs))),
    )
  if row.get("kind") == "score":
    result["prediction"] = int(choices[index])
    result["expected_score"] = float(sum(int(v) * p for v, p in zip(choices, probs)))
    if gold is not None:
      result["score_absolute_error"] = float(abs(result["expected_score"] - int(choices[gold])))
  if row.get("kind") == "noul":
    result["probability_true"] = result["probabilities"]["true"]
  return result


class NimbleModel:
  """Drop-in TPU/MaxText implementation of NimbleModel for bounded schema classification."""

  def __init__(
      self,
      checkpoint_path: str = "gs://hengtaoguo-maxtext-logs/checkpoints/bespoke-nimble-9b/unscanned/2026-09-23/0/items",
      model_id: str = "Qwen/Qwen3.5-9B",
      max_length: int = 2048,
      hf_access_token: str = "<your_hf_token>",
  ):
    self.tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=hf_access_token or None,
    )
    self.max_length = max_length
    base_config = _find_base_config()
    args = [
        base_config,
        "run_name=nimble_inference",
        "model_name=qwen3.5-9b",
        f"tokenizer_path={model_id}",
        f"load_parameters_path={checkpoint_path}",
        "tokenizer_type=huggingface",
        "per_device_batch_size=1",
        "max_prefill_predict_length=256",
        "max_target_length=260",
        "ici_tensor_parallelism=4",
        "scan_layers=false",
        "weight_dtype=bfloat16",
        "prompt=placeholder",
    ]
    if hf_access_token:
      args.append(f"hf_access_token={hf_access_token}")

    self.config = pyconfig.initialize(args)
    self.engine = maxengine.MaxEngine(self.config)
    rng = jax.random.PRNGKey(1234)
    rng, rng_load = jax.random.split(rng)
    self.params = self.engine.load_params(rng_load)
    self.rng = rng

  def score(
      self,
      context: str,
      schema: Mapping[str, Any],
      score_fields: Sequence[str] = (),
  ) -> dict[str, Any]:
    """Scores candidate choices across all fields defined in the schema."""
    prepared = prepare_prompts(self.tokenizer, context, schema, self.max_length)
    score_fields_set = set(score_fields)
    if not score_fields_set.issubset(schema):
      raise ValueError("Unknown score field")
    fields = {}
    pad_id = self.tokenizer.pad_token_id or 0

    for i, name in enumerate(prepared.names):
      row = {
          "input_ids": prepared.full_ids[i],
          "candidate_ids": prepared.candidate_ids[i],
          "choices": prepared.choices[i],
      }
      if schema[name]["type"] == "boolean":
        row["kind"] = "noul"
      if name in score_fields_set:
        if schema[name]["type"] != "enum":
          raise ValueError("Score fields must be integer-valued enums")
        values = [int(v) for v in row["choices"]]
        if len(set(values)) != len(values):
          raise ValueError("Score values must be distinct integers")
        row["kind"] = "score"

      input_ids = row["input_ids"]
      true_length = len(input_ids)
      padded_tokens = np.full((self.config.max_prefill_predict_length,), pad_id, dtype=np.int32)
      padded_tokens[:true_length] = input_ids

      self.rng, rng_prefill = jax.random.split(self.rng)
      prefill_result, _ = self.engine.prefill(
          params=self.params,
          padded_tokens=jnp.array(padded_tokens),
          true_length=true_length,
          rng=rng_prefill,
          slot=0,
      )
      logits = np.array(prefill_result["logits"][0, 0, :])
      cand_logits = logits[row["candidate_ids"]]
      fields[name] = decision_result(row, cand_logits)

    return {
        "output": {name: r["prediction"] for name, r in fields.items()},
        "fields": fields,
    }


def main():
  model = NimbleModel()
  result = model.score(
      context="The store accepts returns within 30 days. This item was bought 12 days ago.",
      schema={
          "eligible": {
              "type": "boolean",
              "description": "Is this item within the store return window?",
          }
      },
  )
  print(result["output"])
  print(result["fields"]["eligible"]["probabilities"])


if __name__ == "__main__":
  main()
