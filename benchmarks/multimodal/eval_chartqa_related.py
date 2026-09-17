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

"""Evaluates ChartQA benchmark CSV results using the official relaxed accuracy metric.

Following Methani et al. (2020) and Masry et al. (2022) (ChartQA):
- Numeric answers are considered correct if within 5% of the gold answer.
- Non-numeric answers require an exact case-insensitive match.
"""

import argparse
import csv
import os
import re
import subprocess
import tempfile
from typing import Optional


def extract_pred(output_str: str) -> str:
  """Extracts the answer from <answer>...</answer> tags if present, otherwise strips string."""
  if not output_str:
    return ""
  text = str(output_str).strip()
  match = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
  if match:
    return match.group(1).strip()
  return text


def to_float(text: str) -> Optional[float]:
  """Converts a string to float, handling commas and percentage signs."""
  try:
    clean = str(text).strip().replace(",", "")
    if clean.endswith("%"):
      return float(clean.rstrip("%")) / 100.0
    return float(clean)
  except ValueError:
    return None


def chartqa_relaxed_correctness(target: str, prediction: str, max_relative_change: float = 0.05) -> bool:
  """Calculates relaxed correctness following Methani et al. (2020) / ChartQA standard."""
  pred_clean = str(prediction).strip().rstrip(".")
  target_clean = str(target).strip().rstrip(".")

  # Handle list answers like "[Germany, United States]"
  if target_clean.startswith("[") and target_clean.endswith("]") and pred_clean.startswith("[") and pred_clean.endswith("]"):
    target_items = [item.strip() for item in target_clean[1:-1].split(",")]
    pred_items = [item.strip() for item in pred_clean[1:-1].split(",")]
    if len(target_items) != len(pred_items):
      return False
    return all(chartqa_relaxed_correctness(t, p, max_relative_change) for t, p in zip(target_items, pred_items))

  prediction_float = to_float(pred_clean)
  target_float = to_float(target_clean)

  if prediction_float is not None and target_float is not None:
    if target_float == 0.0:
      return prediction_float == 0.0
    relative_change = abs(prediction_float - target_float) / abs(target_float)
    return relative_change <= max_relative_change
  else:
    return pred_clean.lower() == target_clean.lower()


def chartqa_relaxed_normalized(target: str, prediction: str, max_relative_change: float = 0.05) -> bool:
  """Relaxed correctness with post-processing for base models (equations, units, % scaling)."""
  if chartqa_relaxed_correctness(target, prediction, max_relative_change):
    return True
  p = str(prediction).strip()
  # Extract final result after '=' if model output an equation (e.g. "39/3 = 13")
  if "=" in p:
    p = p.split("=")[-1].strip()
  if chartqa_relaxed_correctness(target, p, max_relative_change):
    return True

  # Check numeric extraction with unit stripping and percentage/decimal scaling
  t_str = str(target).strip()
  m_t = re.search(r"[-+]?\d[\d,]*\.?\d*", t_str)
  m_p = re.search(r"[-+]?\d[\d,]*\.?\d*", p)
  is_numeric_target = t_str.replace(",", "").replace("%", "").replace(".", "").replace("-", "").isdigit()
  if m_t and m_p and is_numeric_target:
    try:
      tf = float(m_t.group(0).replace(",", ""))
      pf = float(m_p.group(0).replace(",", ""))
      for mult in (1.0, 100.0, 0.01):
        if tf == 0.0:
          if pf * mult == 0.0:
            return True
        elif abs(pf * mult - tf) / abs(tf) <= max_relative_change:
          return True
    except ValueError:
      pass
  return False


def resolve_csv_path(csv_path: str) -> tuple[str, bool]:
  """Resolves local or GCS path, downloading to a temp file if on GCS."""
  if not os.path.exists(csv_path) and not csv_path.startswith("gs://") and "/" in csv_path:
    candidate_gcs = f"gs://{csv_path}"
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    temp_file.close()
    res = subprocess.run(["gsutil", "cp", candidate_gcs, temp_file.name], capture_output=True, text=True, check=False)
    if res.returncode == 0:
      return temp_file.name, True
  if csv_path.startswith("gs://"):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    temp_file.close()
    subprocess.run(["gsutil", "cp", csv_path, temp_file.name], check=True)
    return temp_file.name, True
  return csv_path, False


def main():
  parser = argparse.ArgumentParser(description="Evaluate ChartQA results CSV with relaxed accuracy.")
  parser.add_argument("csv_path", type=str, help="Path to the input CSV file (local or gs://).")
  parser.add_argument("--output_csv", type=str, default=None, help="Optional path to save updated CSV.")
  args = parser.parse_args()

  local_path, is_temp = resolve_csv_path(args.csv_path)
  try:
    rows = []
    with open(local_path, mode="r", encoding="utf-8") as f:
      reader = csv.DictReader(f)
      fieldnames = list(reader.fieldnames) if reader.fieldnames else []
      for row in reader:
        rows.append(row)
  finally:
    if is_temp and os.path.exists(local_path):
      os.remove(local_path)

  total = len(rows)
  orig_correct = 0
  judge_correct = 0
  relaxed_correct_count = 0
  normalized_correct_count = 0

  gained = []
  lost = []

  for row in rows:
    pred = extract_pred(row.get("output", ""))
    label = row.get("label", "")
    is_rel_correct = chartqa_relaxed_correctness(label, pred)
    is_norm_correct = chartqa_relaxed_normalized(label, pred)
    row["extracted_pred"] = pred
    row["relaxed_correct"] = str(is_rel_correct)
    row["relaxed_normalized_correct"] = str(is_norm_correct)

    if is_rel_correct:
      relaxed_correct_count += 1
    if is_norm_correct:
      normalized_correct_count += 1

    orig_val = str(row.get("is_correct", "")).lower() == "true"
    if orig_val:
      orig_correct += 1

    judge_val = str(row.get("jetski_judge", "")).lower() == "true"
    if judge_val:
      judge_correct += 1

    if not orig_val and is_rel_correct:
      gained.append(row)
    elif orig_val and not is_rel_correct:
      lost.append(row)

  print(f"File: {args.csv_path}")
  print(f"Total examples: {total}")
  if "is_correct" in fieldnames and total > 0:
    print(f"Original 'is_correct' accuracy:             {orig_correct / total:.4f} ({orig_correct}/{total})")
  if "jetski_judge" in fieldnames and total > 0:
    print(f"Original 'jetski_judge' accuracy:           {judge_correct / total:.4f} ({judge_correct}/{total})")
  if total > 0:
    print(f"ChartQA Relaxed Accuracy (strict 5% tol):   {relaxed_correct_count / total:.4f} ({relaxed_correct_count}/{total})")
    print(f"ChartQA Relaxed Accuracy (+base norm/units): {normalized_correct_count / total:.4f} ({normalized_correct_count}/{total})")

  print(f"\nExamples flipped False -> True under Relaxed Accuracy: {len(gained)}")
  for r in gained[:10]:
    print(f"  Q{r.get('question ID')}: label='{r.get('label')}', pred='{r.get('extracted_pred')}'")

  print(f"\nExamples flipped True -> False under Relaxed Accuracy: {len(lost)}")
  for r in lost[:10]:
    print(f"  Q{r.get('question ID')}: label='{r.get('label')}', pred='{r.get('extracted_pred')}'")

  if args.output_csv:
    out_fields = fieldnames + [
        c for c in ["extracted_pred", "relaxed_correct", "relaxed_normalized_correct"] if c not in fieldnames
    ]
    with open(args.output_csv, mode="w", encoding="utf-8", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=out_fields)
      writer.writeheader()
      writer.writerows(rows)
    print(f"\nSaved updated results to: {args.output_csv}")


if __name__ == "__main__":
  main()
