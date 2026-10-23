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

"""Automated Model Lifecycle Audit Tool for MaxText.

Implements the structural lifecycle audit across model configs, documentation,
and checkpoint conversion scripts:
  1. Rule 1 (N-2 Generation Cap): At most 2 active generations per core model
     family (`gemma`, `qwen`, `deepseek`, `llama`). Flags any N-2 or older
     generation still in `src/maxtext/configs/models/`.
  2. Rule 2 (Standalone Converter Check): Flags any active model family that
     still depends on legacy `standalone_scripts/` instead of unified
     `to_maxtext.py` / `to_huggingface.py`, and inventories legacy standalone
     scripts targeting active or deprecated families.
  3. Rule 3 (90-Day `deprecated/` Grace Period Expiry): Audits all YAML files
     in `src/maxtext/configs/models/deprecated/*.yml` and reports deprecation
     status and days remaining in the 90-day grace window.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import datetime
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any
import yaml

CORE_FAMILIES: tuple[str, ...] = ("gemma", "qwen", "deepseek", "llama")
DEFAULT_GRACE_PERIOD_DAYS: int = 90

# Maps standalone script filenames/prefixes to model families.
STANDALONE_SCRIPT_FAMILY_MAP: dict[str, str] = {
    "convert_deepseek_family_ckpt.py": "deepseek",
    "convert_deepseek_family_unscanned_ckpt.py": "deepseek",
    "deepseek_dequantize.py": "deepseek",
    "convert_gemma_chkpt.py": "gemma",
    "convert_gemma2_chkpt.py": "gemma",
    "convert_gemma3_chkpt.py": "gemma",
    "convert_gpt_oss_ckpt.py": "gpt-oss",
    "convert_gpt_oss_unscanned_ckpt.py": "gpt-oss",
    "dequantize_mxfp4.py": "gpt-oss",
    "convert_gpt3_ckpt_from_paxml.py": "gpt3",
    "convert_qwen3_moe.py": "qwen",
    "convert_qwen3_next_scanned.py": "qwen",
    "convert_qwen3_next_unscanned.py": "qwen",
    "llama_ckpt_conversion_inference_only.py": "llama",
    "llama_mistral_mixtral_orbax_to_hf.py": "llama",
    "llama_or_mistral_ckpt.py": "llama",
    "llama4_ckpt_unscanned.py": "llama",
}


@dataclass
class GenerationRuleResult:
  """Result of evaluating Rule 1 (N-2 Generation Cap) for a model family."""

  family: str
  active_generations: list[str]
  deprecated_generations: list[str]
  active_models: list[str]
  violating_generations: list[str]
  violating_models: list[str]
  passed: bool


@dataclass
class ConverterRuleResult:
  """Result of evaluating Rule 2 (Standalone Converter Check) for an active family."""

  family: str
  active_models: list[str]
  unified_supported_models: list[str]
  unmapped_active_models: list[str]
  standalone_scripts: list[str]
  depends_only_on_standalone: bool
  passed: bool


@dataclass
class DeprecatedModelStatus:
  """Result of evaluating Rule 3 (90-Day Grace Period) for a deprecated YAML."""

  model_name: str
  family: str
  generation: str
  deprecated_date: str
  days_in_deprecated: int
  days_remaining: int
  expired: bool
  status: str
  in_tiering_docs: bool = False


@dataclass
class TieringEntry:
  """Entry parsed from docs/reference/models/tiering.md."""

  platform: str
  tier: str
  display_name: str
  normalized_model_name: str


@dataclass
class LifecycleAuditReport:
  """Complete Model Lifecycle Audit report."""

  reference_date: str
  grace_period_days: int
  active_model_count: int
  deprecated_model_count: int
  rule1_generation_cap: list[GenerationRuleResult] = field(default_factory=list)
  rule2_standalone_converters: list[ConverterRuleResult] = field(default_factory=list)
  rule3_deprecated_grace_period: list[DeprecatedModelStatus] = field(default_factory=list)
  tiering_entries: list[TieringEntry] = field(default_factory=list)
  deprecated_models_in_tiering: list[str] = field(default_factory=list)

  @property
  def rule1_passed(self) -> bool:
    return all(r.passed for r in self.rule1_generation_cap)

  @property
  def rule2_passed(self) -> bool:
    return all(r.passed for r in self.rule2_standalone_converters)

  @property
  def rule3_expired_count(self) -> int:
    return sum(1 for r in self.rule3_deprecated_grace_period if r.expired)

  def to_dict(self) -> dict[str, Any]:
    return {
        "reference_date": self.reference_date,
        "grace_period_days": self.grace_period_days,
        "summary": {
            "active_model_count": self.active_model_count,
            "deprecated_model_count": self.deprecated_model_count,
            "rule1_passed": self.rule1_passed,
            "rule2_passed": self.rule2_passed,
            "rule3_expired_count": self.rule3_expired_count,
            "deprecated_models_in_tiering_count": len(self.deprecated_models_in_tiering),
        },
        "rule1_generation_cap": [asdict(r) for r in self.rule1_generation_cap],
        "rule2_standalone_converters": [asdict(r) for r in self.rule2_standalone_converters],
        "rule3_deprecated_grace_period": [asdict(r) for r in self.rule3_deprecated_grace_period],
        "tiering_entries": [asdict(t) for t in self.tiering_entries],
        "deprecated_models_in_tiering": self.deprecated_models_in_tiering,
    }


def _find_repo_root() -> Path:
  """Resolves the MaxText repository root directory."""
  current = Path(__file__).resolve()
  for parent in current.parents:
    if (parent / "src" / "maxtext" / "configs" / "models").is_dir():
      return parent
  return current.parents[3]


def parse_model_family_and_generation(model_name: str, yaml_data: dict[str, Any] | None = None) -> tuple[str, str | None]:
  """Extracts (family, generation_id) from a MaxText model config name and optional YAML content.

  Returns `(family, generation)` where `generation` is None for unversioned/custom templates
  (such as `deepseek-custom` or `envy-test`).
  """
  yaml_data = yaml_data or {}
  decoder_block = str(yaml_data.get("decoder_block", "")).lower()

  # Gemma family: gemma-2b -> 1, gemma2-9b -> 2, gemma3-4b -> 3, gemma4-26b -> 4
  m = re.match(r"^gemma(\d+)?-", model_name)
  if m:
    gen_num = m.group(1) or "1"
    return "gemma", f"gemma{gen_num}"

  # Qwen family: qwen2.5-7b -> qwen2.5, qwen3-* -> qwen3, qwen3.5-* / qwen3.8-* (decoder_block=qwen3_5) -> qwen3.5
  m = re.match(r"^qwen(\d+(?:\.\d+)?)-", model_name)
  if m:
    raw_ver = m.group(1)
    if decoder_block == "qwen3_5" or raw_ver in ("3.5", "3.8"):
      return "qwen", "qwen3.5"
    if raw_ver.startswith("3"):
      return "qwen", "qwen3"
    if raw_ver.startswith("2"):
      return "qwen", f"qwen{raw_ver}"
    return "qwen", f"qwen{raw_ver}"

  # DeepSeek family: deepseek2-* -> deepseek2, deepseek3* -> deepseek3, deepseek4* -> deepseek4
  if model_name.startswith("deepseek"):
    if model_name == "deepseek-custom":
      return "deepseek", None
    m = re.match(r"^deepseek(\d+(?:\.\d+)?)-", model_name)
    if m:
      major_ver = m.group(1).split(".")[0]
      return "deepseek", f"deepseek{major_ver}"
    return "deepseek", None

  # Llama family: llama2-* -> llama2, llama3-* -> llama3, llama3.1-* -> llama3.1, llama3.3-* -> llama3.3, llama4-* -> llama4
  m = re.match(r"^llama(\d+(?:\.\d+)?)-", model_name)
  if m:
    return "llama", f"llama{m.group(1)}"

  # Other families
  if model_name.startswith("gpt-oss-"):
    return "gpt-oss", "gpt-oss"
  if model_name.startswith("gpt3-"):
    return "gpt3", "gpt3"
  if model_name.startswith("mistral-"):
    return "mistral", "mistral"
  if model_name.startswith("mixtral-"):
    return "mixtral", "mixtral"
  if model_name.startswith("olmo3-"):
    return "olmo", "olmo3"
  if model_name.startswith("kimi-k2"):
    return "kimi", "kimi-k2"
  if model_name.startswith("glm"):
    m_glm = re.match(r"^(glm\d+(?:\.\d+)?)", model_name)
    return "glm", (m_glm.group(1) if m_glm else "glm")
  if model_name.startswith("weaver"):
    return "weaver", "weaver"
  if model_name.startswith("envy"):
    return "envy", None

  prefix = model_name.split("-")[0]
  return prefix, prefix


def _generation_sort_key(gen: str) -> tuple[float, str]:
  """Extracts numeric version tuple for chronological generation sorting."""
  m = re.search(r"(\d+(?:\.\d+)?)", gen)
  if m:
    return (float(m.group(1)), gen)
  return (0.0, gen)


def _load_yaml_safe(path: Path) -> dict[str, Any]:
  """Loads a YAML file into a dictionary, returning an empty dict on failure."""
  try:
    with path.open("r", encoding="utf-8") as f:
      data = yaml.safe_load(f)
      return data if isinstance(data, dict) else {}
  except Exception:  # pylint: disable=broad-exception-caught
    return {}


def parse_tiering_markdown(tiering_path: Path) -> list[TieringEntry]:
  """Parses docs/reference/models/tiering.md (or docs/guides/optimization/tiering.md)."""
  if not tiering_path.is_file():
    return []

  entries: list[TieringEntry] = []
  current_platform = "unknown"
  current_tier = "unknown"

  for raw_line in tiering_path.read_text(encoding="utf-8").splitlines():
    line = raw_line.strip()
    if line.startswith("## "):
      current_platform = line[3:].strip()
    elif line.startswith("### "):
      current_tier = line[4:].strip()
    elif line.startswith("|") and not line.startswith("| :") and not line.startswith("| Model"):
      cols = [c.strip() for c in line.strip("|").split("|")]
      if cols and cols[0]:
        display_name = cols[0]
        normalized = _normalize_display_model_name(display_name)
        entries.append(
            TieringEntry(
                platform=current_platform,
                tier=current_tier,
                display_name=display_name,
                normalized_model_name=normalized,
            )
        )
  return entries


def _normalize_display_model_name(display_name: str) -> str:
  """Normalizes a human-readable model name from tiering.md (e.g. 'Llama 3.1 8B' -> 'llama3.1-8b')."""
  s = display_name.strip().lower()
  s = re.sub(r"^(llama|gemma|qwen|deepseek)\s+(\d+(?:\.\d+)?)\s+", r"\1\2-", s)
  s = re.sub(r"\s+", "-", s)
  return s


def parse_unified_converter_models(param_mapping_path: Path) -> set[str]:
  """Extracts model names registered in PARAM_MAPPING / HOOK_FNS in param_mapping.py."""
  if not param_mapping_path.is_file():
    return set()

  content = param_mapping_path.read_text(encoding="utf-8")
  # Extract PARAM_MAPPING and HOOK_FNS blocks
  models: set[str] = set()
  for block_name in ("PARAM_MAPPING", "HOOK_FNS"):
    idx = content.find(f"{block_name} = {{")
    if idx == -1:
      continue
    sub = content[idx:]
    end_idx = sub.find("\n}")
    if end_idx != -1:
      sub = sub[: end_idx + 2]
    for match in re.finditer(r'"([a-zA-Z0-9_.-]+)"\s*:', sub):
      models.add(match.group(1))
  return models


def scan_standalone_scripts(standalone_dir: Path) -> dict[str, list[str]]:
  """Maps model families to standalone converter scripts present in standalone_scripts/."""
  family_scripts: dict[str, list[str]] = {}
  if not standalone_dir.is_dir():
    return family_scripts

  for script_path in sorted(standalone_dir.glob("*.py")):
    name = script_path.name
    if name == "__init__.py":
      continue
    family = STANDALONE_SCRIPT_FAMILY_MAP.get(name)
    if family is None:
      lower_name = name.lower()
      for candidate in ("deepseek", "gemma", "qwen", "llama", "gpt_oss", "gpt3", "mistral", "mixtral"):
        if candidate in lower_name:
          family = candidate.replace("_", "-")
          break
      else:
        family = "other"
    family_scripts.setdefault(family, []).append(name)
  return family_scripts


def _resolve_deprecated_date(yaml_path: Path, repo_root: Path, reference_date: datetime.date) -> datetime.date:
  """Determines the date a YAML config was moved to deprecated/."""
  # 1. Check explicit metadata comment `# deprecated_date: YYYY-MM-DD` in YAML header
  try:
    head_lines = yaml_path.read_text(encoding="utf-8").splitlines()[:25]
    for line in head_lines:
      m = re.search(r"deprecated_date\s*:\s*(\d{4}-\d{2}-\d{2})", line)
      if m:
        return datetime.date.fromisoformat(m.group(1))
  except Exception:  # pylint: disable=broad-exception-caught
    pass

  # 2. Check git log for the most recent commit touching or renaming the file in deprecated/
  try:
    rel_path = yaml_path.relative_to(repo_root)
    proc = subprocess.run(
        ["git", "log", "-1", "--format=%cs", "--", str(rel_path)],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
        timeout=2,
    )
    date_str = proc.stdout.strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
      return datetime.date.fromisoformat(date_str)
  except Exception:  # pylint: disable=broad-exception-caught
    pass

  # 3. Fallback to reference_date (staged/uncommitted move in current working tree)
  return reference_date


def _is_test_or_variant_exempt_from_converter(model_name: str) -> bool:
  """Returns True for synthetic/test/tuning variant configs that share base model converters."""
  exempt_suffixes = (
      "-test",
      "-tiny",
      "-custom",
      "-2dfsdp",
      "-batchsplit",
      "-lineage",
  )
  if model_name.startswith(("envy-", "weaver")):
    return True
  return any(model_name.endswith(suffix) for suffix in exempt_suffixes)


def run_audit(
    repo_root: Path | None = None,
    grace_period_days: int = DEFAULT_GRACE_PERIOD_DAYS,
    reference_date: datetime.date | None = None,
) -> LifecycleAuditReport:
  """Executes the 3-rule Model Lifecycle Audit across the repository."""
  repo_root = (repo_root or _find_repo_root()).resolve()
  reference_date = reference_date or datetime.date.today()

  models_dir = repo_root / "src" / "maxtext" / "configs" / "models"
  deprecated_dir = models_dir / "deprecated"
  standalone_dir = repo_root / "src" / "maxtext" / "checkpoint_conversion" / "standalone_scripts"
  param_mapping_path = repo_root / "src" / "maxtext" / "checkpoint_conversion" / "utils" / "param_mapping.py"

  tiering_path = repo_root / "docs" / "reference" / "models" / "tiering.md"
  if not tiering_path.is_file():
    tiering_path = repo_root / "docs" / "guides" / "optimization" / "tiering.md"

  active_yamls = sorted(models_dir.glob("*.yml")) if models_dir.is_dir() else []
  deprecated_yamls = sorted(deprecated_dir.glob("*.yml")) if deprecated_dir.is_dir() else []

  # Parse active and deprecated models
  active_by_family: dict[str, list[tuple[str, str | None]]] = {}
  for yml in active_yamls:
    model_name = yml.stem
    data = _load_yaml_safe(yml)
    family, gen = parse_model_family_and_generation(model_name, data)
    active_by_family.setdefault(family, []).append((model_name, gen))

  deprecated_by_family: dict[str, list[tuple[str, str | None, Path]]] = {}
  deprecated_model_set: set[str] = set()
  for yml in deprecated_yamls:
    model_name = yml.stem
    deprecated_model_set.add(model_name)
    data = _load_yaml_safe(yml)
    family, gen = parse_model_family_and_generation(model_name, data)
    deprecated_by_family.setdefault(family, []).append((model_name, gen, yml))

  # Parse tiering.md
  tiering_entries = parse_tiering_markdown(tiering_path)
  deprecated_in_tiering = sorted(
      {t.normalized_model_name for t in tiering_entries if t.normalized_model_name in deprecated_model_set}
  )

  # --- Rule 1: N-2 Generation Cap (max 2 active generations per core family) ---
  rule1_results: list[GenerationRuleResult] = []
  for family in CORE_FAMILIES:
    active_entries = active_by_family.get(family, [])
    deprecated_entries = deprecated_by_family.get(family, [])

    active_gens = sorted(
        {gen for _, gen in active_entries if gen is not None},
        key=_generation_sort_key,
    )
    deprecated_gens = sorted(
        {gen for _, gen, _ in deprecated_entries if gen is not None},
        key=_generation_sort_key,
    )
    active_models = sorted(m for m, _ in active_entries)

    if len(active_gens) > 2:
      violating_gens = active_gens[:-2]
      violating_set = set(violating_gens)
      violating_models = sorted(m for m, gen in active_entries if gen in violating_set)
      passed = False
    else:
      violating_gens = []
      violating_models = []
      passed = True

    rule1_results.append(
        GenerationRuleResult(
            family=family,
            active_generations=active_gens,
            deprecated_generations=deprecated_gens,
            active_models=active_models,
            violating_generations=violating_gens,
            violating_models=violating_models,
            passed=passed,
        )
    )

  # --- Rule 2: Standalone Converter Check ---
  unified_models = parse_unified_converter_models(param_mapping_path)
  standalone_by_family = scan_standalone_scripts(standalone_dir)

  rule2_results: list[ConverterRuleResult] = []
  for family in sorted(active_by_family.keys()):
    if family in ("envy", "weaver"):
      continue
    family_models = sorted(m for m, _ in active_by_family[family])
    production_models = [m for m in family_models if not _is_test_or_variant_exempt_from_converter(m)]
    if not production_models:
      continue

    supported: list[str] = []
    unmapped: list[str] = []
    for m in production_models:
      # Check exact match or normalized match (e.g. qwen3-480b-a35b -> qwen3-coder-480b-a35b)
      if m in unified_models or m.replace("qwen3-480b", "qwen3-coder-480b") in unified_models:
        supported.append(m)
      else:
        unmapped.append(m)

    scripts = standalone_by_family.get(family, [])
    depends_only_on_standalone = len(supported) == 0 and len(scripts) > 0
    passed = not depends_only_on_standalone

    rule2_results.append(
        ConverterRuleResult(
            family=family,
            active_models=production_models,
            unified_supported_models=supported,
            unmapped_active_models=unmapped,
            standalone_scripts=scripts,
            depends_only_on_standalone=depends_only_on_standalone,
            passed=passed,
        )
    )

  # --- Rule 3: 90-Day deprecated/ Grace Period Expiry ---
  rule3_results: list[DeprecatedModelStatus] = []
  for yml in deprecated_yamls:
    model_name = yml.stem
    data = _load_yaml_safe(yml)
    family, gen = parse_model_family_and_generation(model_name, data)
    dep_date = _resolve_deprecated_date(yml, repo_root, reference_date)
    days_elapsed = max(0, (reference_date - dep_date).days)
    days_remaining = grace_period_days - days_elapsed
    expired = days_remaining <= 0
    status = "EXPIRED_READY_FOR_REMOVAL" if expired else "IN_GRACE_PERIOD"
    rule3_results.append(
        DeprecatedModelStatus(
            model_name=model_name,
            family=family,
            generation=gen or family,
            deprecated_date=dep_date.isoformat(),
            days_in_deprecated=days_elapsed,
            days_remaining=days_remaining,
            expired=expired,
            status=status,
            in_tiering_docs=model_name in deprecated_in_tiering,
        )
    )

  return LifecycleAuditReport(
      reference_date=reference_date.isoformat(),
      grace_period_days=grace_period_days,
      active_model_count=len(active_yamls),
      deprecated_model_count=len(deprecated_yamls),
      rule1_generation_cap=rule1_results,
      rule2_standalone_converters=rule2_results,
      rule3_deprecated_grace_period=rule3_results,
      tiering_entries=tiering_entries,
      deprecated_models_in_tiering=deprecated_in_tiering,
  )


def format_markdown_report(report: LifecycleAuditReport) -> str:
  """Formats the LifecycleAuditReport as a clean Markdown summary report."""
  lines: list[str] = []
  lines.append("# MaxText Model Lifecycle Audit Report")
  lines.append("")
  lines.append(f"- **Reference Date:** `{report.reference_date}`")
  lines.append(f"- **Active Models (`configs/models/*.yml`):** {report.active_model_count}")
  lines.append(f"- **Sunset Models (`configs/models/deprecated/*.yml`):** {report.deprecated_model_count}")
  lines.append(f"- **Grace Period Window:** {report.grace_period_days} days")
  lines.append("")

  # Rule 1 Table
  lines.append("## Rule 1: `N-2` Generation Cap (Max 2 Active Generations per Core Family)")
  lines.append("")
  lines.append("| Family | Active Generations | Sunset Generations (`deprecated/`) | Violating Generations | Status |")
  lines.append("| :--- | :--- | :--- | :--- | :--- |")
  for r1 in report.rule1_generation_cap:
    active_str = ", ".join(f"`{g}`" for g in r1.active_generations) or "None"
    dep_str = ", ".join(f"`{g}`" for g in r1.deprecated_generations) or "None"
    viol_str = ", ".join(f"`{g}`" for g in r1.violating_generations) or "None"
    status_str = "PASS" if r1.passed else f"FAIL ({len(r1.violating_models)} models)"
    lines.append(f"| `{r1.family}` | {active_str} | {dep_str} | {viol_str} | **{status_str}** |")
  lines.append("")

  # Rule 2 Table
  lines.append("## Rule 2: Unified Checkpoint Converter Coverage (`to_maxtext.py` / `to_huggingface.py`)")
  lines.append("")
  lines.append(
      "| Active Family | Unified Converter Models | Unmapped Active Models | Legacy `standalone_scripts/` | Status |"
  )
  lines.append("| :--- | :--- | :--- | :--- | :--- |")
  for r2 in report.rule2_standalone_converters:
    unified_str = f"{len(r2.unified_supported_models)}/{len(r2.active_models)}"
    unmapped_str = ", ".join(f"`{m}`" for m in r2.unmapped_active_models) or "None"
    scripts_str = ", ".join(f"`{s}`" for s in r2.standalone_scripts) or "None"
    status_str = "PASS" if r2.passed else "FAIL (Standalone Only)"
    lines.append(f"| `{r2.family}` | {unified_str} | {unmapped_str} | {scripts_str} | **{status_str}** |")
  lines.append("")

  # Rule 3 Table
  lines.append(f"## Rule 3: `deprecated/` 90-Day Grace Period Status ({report.deprecated_model_count} Sunset Configs)")
  lines.append("")
  lines.append(
      "| Deprecated Model | Family | Generation | Deprecated Date | Days Elapsed | Days Remaining | Listed in `tiering.md` | Status |"
  )
  lines.append("| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :--- |")
  for r3 in report.rule3_deprecated_grace_period:
    tier_flag = "Yes (Stale)" if r3.in_tiering_docs else "No"
    lines.append(
        f"| `{r3.model_name}` | `{r3.family}` | `{r3.generation}` | `{r3.deprecated_date}` | "
        f"{r3.days_in_deprecated} | {r3.days_remaining} | {tier_flag} | `{r3.status}` |"
    )
  lines.append("")
  return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description="Run the MaxText Model Lifecycle Audit.")
  parser.add_argument(
      "--format",
      choices=("markdown", "json"),
      default="markdown",
      help="Output format: 'markdown' (default) or 'json'.",
  )
  parser.add_argument(
      "--repo-root",
      type=Path,
      default=None,
      help="Path to MaxText repository root (auto-detected if omitted).",
  )
  parser.add_argument(
      "--grace-days",
      type=int,
      default=DEFAULT_GRACE_PERIOD_DAYS,
      help="Deprecation grace period in days (default: 90).",
  )
  parser.add_argument(
      "--reference-date",
      type=str,
      default=None,
      help="Reference date in YYYY-MM-DD format (defaults to today).",
  )
  args = parser.parse_args(argv)

  ref_date = datetime.date.fromisoformat(args.reference_date) if args.reference_date else None
  report = run_audit(
      repo_root=args.repo_root,
      grace_period_days=args.grace_days,
      reference_date=ref_date,
  )

  if args.format == "json":
    print(json.dumps(report.to_dict(), indent=2))
  else:
    print(format_markdown_report(report))

  return 0


if __name__ == "__main__":
  sys.exit(main())
